
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

train_dataset = load_from_disk("./ultrachat_train_formatted").select(range(1000))
eval_dataset = load_from_disk("./ultrachat_eval_formatted").select(range(500))


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

#%%
print((str(device)))

# %%
# ========================== Load the saved model ==========================

# set the attention implementation based on whether gpu is available

if torch.cuda.is_available():
    attention_type = "flash_attention_2",
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

print(fp16_check)

training_args = SFTConfig(
   
   # -------------------------
    # Output and Logging
    # -------------------------
    output_dir=output_dir,
    overwrite_output_dir=True,
    report_to=[],          # enable TensorBoard only
    logging_dir=f"{output_dir}/logs",
    logging_strategy="steps",
    logging_steps=10,                   # print every 10 steps (adjust as desired)
    log_level="info",
    disable_tqdm=False, 

    # -------------------------
    # Batch Sizes - Optimized for H200
    # -------------------------
    per_device_eval_batch_size=4, # originally set to 8
    per_device_train_batch_size=4, # originally set to 8
    gradient_accumulation_steps=2,

    # -------------------------
    # Training Regime
    # -------------------------
    max_steps=-1,
    num_train_epochs=1,

    # -------------------------
    # Optimization
    # -------------------------
    learning_rate=2.0e-05,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    # warmup_steps=100,
    weight_decay=0.01,
    max_grad_norm=1.0,
    optim="adamw_torch_fused",

    # -------------------------
    # Memory Optimizations
    # -------------------------
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},  # more stable
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

#%%
training_args.fp16

# %%
# ============================ peft configuration ==========================
peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

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
        
    )
# %%
# params check
for n, p in model.named_parameters():
    if p.requires_grad:
        print("First trainable param:", n, p.shape)
        break

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


#%%
# =========================== Create logging utility =============================

from transformers import TrainerCallback

class LossLoggingCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.train_loss = 0.0
        self.train_steps = 0

    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step"""
        logs = kwargs.get("logs", {})
        loss = logs.get("loss")
        if loss is not None:
            self.train_loss += loss
            self.train_steps += 1

        if self.train_steps > 0 and state.global_step % args.logging_steps == 0:
            avg_loss = self.train_loss / self.train_steps
            print(f"[Step {state.global_step}] Training Loss: {avg_loss:.4f}")
            self.train_loss = 0.0
            self.train_steps = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called at each evaluation step"""
        if metrics and "eval_loss" in metrics:
            print(f"[Step {state.global_step}] Validation Loss: {metrics['eval_loss']:.4f}")

trainer.add_callback(LossLoggingCallback())

# %%
# =========================== 8. TRAINING ====================================
# Train the model

trainer.train()
# %%
