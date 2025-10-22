
#%%
# ========================== Load dataset and tokenizer ==========================
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from datasets import load_from_disk
from torch.utils.data import DataLoader
import torch
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from dataclasses import dataclass
from typing import List, Dict


#%%
# ========================== Set the device  ============================

if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Using MPS backend")
else:
    device = torch.device('cpu')
    print("Using CPU")

#%%
# ========================== Load dataset and tokenizer==========================

train_dataset = load_from_disk("./ultrachat_train_formatted").select(range(10000))
eval_dataset = load_from_disk("./ultrachat_eval_formatted").select(range(200))


tokenizer = AutoTokenizer.from_pretrained("tokenizer_aae1", use_fast=True)
print(train_dataset.column_names)
print(f"Padding token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
print(f"Special tokens: {tokenizer.additional_special_tokens}")
print(f"Tokenizer vocabulary size: {len(tokenizer)}")
print(f"Special tokens added: {tokenizer.all_special_tokens}")

#check if model tokenizer has eos and/or pad tokens
print(tokenizer.bos_token_id, tokenizer.bos_token)
print(tokenizer.eos_token_id)
print(tokenizer.pad_token_id)

print(train_dataset[0])

#%%
print((str(device)))

# %%
# ========================== Load the saved model ==========================

# set the attention implementation based on whether gpu is available

if torch.cuda.is_available():
    attention_type = "flash_attention_2"
else:
    attention_type = "sdpa"

model_name = "model_aae1"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map='auto',
    trust_remote_code=True,
    attn_implementation=attention_type,
    dtype=torch.bfloat16,          # Full bf16 precision
)

# Enable gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()

print(f"Model dtype: {model.dtype}")
    
# %%
# ========================== Set SFTCofig parameters ==========================
output_dir = 'model_logs'

if torch.cuda.is_available():
    bf16_check=True
    bf16_full_eval_check=True
    tf32_check=True
    fp16_check=False
    report_to_check="tensorboard"
else:
    bf16_check=False
    bf16_full_eval_check=False
    tf32_check=False
    fp16_check=False
    report_to_check=None

training_args = SFTConfig(
   
   # -------------------------
    # Output and Logging
    # -------------------------
    output_dir=output_dir,
    overwrite_output_dir=True,
    report_to=report_to_check,          # enable TensorBoard only
    logging_dir=f"{output_dir}/logs",
    logging_strategy="steps",
    logging_steps=1,                   # print every 10 steps (adjust as desired)
    log_level="info",
    disable_tqdm=False, 

    # -------------------------
    # Batch Sizes - Optimized for H200
    # -------------------------
    per_device_train_batch_size=16, # originally set to 8
    per_device_eval_batch_size=16, # originally set to 8
    gradient_accumulation_steps=4,

    # -------------------------
    # Training Regime
    # -------------------------
    max_steps=-1,
    num_train_epochs=1,

    # -------------------------
    # Optimization
    # -------------------------
    learning_rate=2.0e-04,
    # lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    # warmup_steps=100,

    lr_scheduler_type="linear",   # cosine is too aggressive for tiny datasets
    warmup_steps=20,              # a few steps of warmup
    # max_steps=-1,

    weight_decay=0.01,
    max_grad_norm=1.0,
    optim="adamw_torch_fused",

    # -------------------------
    # Memory Optimizations
    # -------------------------
    gradient_checkpointing=False,
    # gradient_checkpointing_kwargs={"use_reentrant": False},  # more stable
    # Optional: activation offloading if needed
    # activation_offloading=True,  
    # activation_offloading_params={"device": "cpu"},

    # -------------------------
    # Precision
    # -------------------------
    bf16=bf16_check,
    bf16_full_eval=bf16_full_eval_check,
    tf32=tf32_check,
    fp16=fp16_check, # specify bf16=True instead when training on GPUs that support bf16
    
    # -------------------------
    # Evaluation
    # -------------------------
    do_eval=True,
    eval_strategy="steps",       # evaluate every N steps
    eval_steps=500,               # <-- run evaluation every 50 steps (adjust to your dataset)
    save_strategy="steps",       # save model when evaluation runs
    save_steps=500,               # save every 50 steps too (aligned with eval)
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    
    
    # push_to_hub=True,
    # hub_model_id="zephyr-7b-sft-lora",
    # hub_strategy="every_save",
    
    dataset_text_field="text",
    

    # Sequence handling
    max_length=None, 
    group_by_length=True,
    packing=False,

    # -------------------------
    # Reproducibility
    # -------------------------
    seed=42,
    data_seed=42,

    # -------------------------
    # Performance
    # -------------------------
    dataloader_num_workers=8,
    dataloader_pin_memory=True,
    dataloader_prefetch_factor=2,
)


# %%
# ============================ peft configuration ==========================
peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[                     # All linear layers in Mistral
        "q_proj",    # Query projection
        "k_proj",    # Key projection  
        "v_proj",    # Value projection
        "o_proj",    # Output projection
        # "gate_proj", # MLP gate
        # "up_proj",   # MLP up
        # "down_proj", # MLP down
    ],
    # Note: Not targeting embedding layers as they're already trainable after resize
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # verify LoRA params are trainable
#%%
# =========================== Create logging utility =============================

from transformers import TrainerCallback


# === Custom Callback ===
from collections import deque
from transformers import TrainerCallback

class MovingAverageLossCallback(TrainerCallback):
    
    """
    Prints running loss, moving average, and learning rate to terminal
    for every `logging_interval` steps. Works with TRL SFTTrainer.
    """
    def __init__(self, logging_interval=1, moving_average_window=50):
        self.logging_interval = logging_interval
        self.moving_window = deque(maxlen=moving_average_window)
        self.running_loss = 0.0
        self.steps = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if "loss" in logs:
            loss = logs["loss"]
            self.running_loss += loss
            self.steps += 1
            self.moving_window.append(loss)

            if state.global_step % self.logging_interval == 0:
                running_avg = self.running_loss / self.steps
                moving_avg = sum(self.moving_window) / len(self.moving_window)

                # Extract LR from optimizer
                optimizer = kwargs.get("optimizer") or getattr(kwargs.get("model"), "optimizer", None)
                if optimizer and hasattr(optimizer, "param_groups"):
                    lr = optimizer.param_groups[0]["lr"]
                    lr_str = f", LR: {lr:.2e}"
                else:
                    lr_str = ""

                print(f"[Step {state.global_step}] Running Loss Avg: {running_avg:.4f}, "
                      f"Moving Loss Avg: {moving_avg:.4f}{lr_str}")

                self.running_loss = 0.0
                self.steps = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            print(f"[Step {state.global_step}] Validation Loss: {metrics['eval_loss']:.4f}")

# %%
# ======================= configure the model for sft =======================
model = model.to(device)
trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[MovingAverageLossCallback()],
    )

# model.print_trainable_parameters()

#%%
# ========================= Compile the model ==================================
if torch.__version__ >= "2.0.0":
    print("\n" + "="*70)
    print("🚀 Compiling model with torch.compile...")
    print("This will take 2-3 minutes on first training step, then speeds up!")
    print("="*70 + "\n")
    
    # Compile the trainer's model (not the original model variable)
    trainer.model = torch.compile(trainer.model, mode="reduce-overhead")
    
    print("✅ Model wrapped for compilation!\n")
else:
    print("⚠️  PyTorch version < 2.0. Skipping torch.compile (consider upgrading)")

#%%
# ============ Sanity check: tokenizer and model alignment ====================
print("Tokenizer vocab size:", len(tokenizer))
print("Model embedding size:", model.get_input_embeddings().weight.shape[0])

# Verify special tokens are recognized
for tok in ["<|pad|>", "<|user|>", "<|assistant|>", "<|system|>"]:
    tok_id = tokenizer.convert_tokens_to_ids(tok)
    print(f"{tok}: ID={tok_id}")
    assert tok_id < model.get_input_embeddings().weight.shape[0], "❌ Token ID out of range!"
print("✅ Model and tokenizer are fully aligned!")


# %%
# =========================== 8. TRAINING ====================================
# Train the model

trainer.train()
# %%
encoded = tokenizer(train_dataset[0]["text"])
print(tokenizer.convert_ids_to_tokens(encoded["input_ids"][-10:]))
# %%
