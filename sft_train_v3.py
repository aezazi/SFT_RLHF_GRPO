
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
eval_dataset = load_from_disk("./ultrachat_eval_formatted").select(range(1000))


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
print(torch.cuda.is_available())

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
# model.gradient_checkpointing_enable()

print(f"Model dtype: {model.dtype}")
    
# %%
# ========================== Set SFTCofig parameters ==========================
# specify output directory for logging
output_dir = 'model_logs'

# set some SFTConfig params based on if cuda is available
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
    per_device_train_batch_size=8, # originally set to 8
    per_device_eval_batch_size=8, # originally set to 8
    gradient_accumulation_steps=1,

   
    # -------------------------
    # Optimization and training
    # -------------------------
    learning_rate=2.0e-04,
    
    # for bug fixing
    # lr_scheduler_type="linear",   
    # max_steps=50,
    # warmup_steps=5,

    # for more extended training
    lr_scheduler_type="cosine",
    max_steps = -1,
    warmup_ratio=0.1,
    
    num_train_epochs=1,

    weight_decay=0.01,
    max_grad_norm=1.0,
    optim="adamw_torch_fused",

    # -------------------------
    # Memory Optimizations
    # -------------------------
    gradient_checkpointing=True,
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
    eval_steps=500,               
    save_strategy="steps",       # save model when evaluation runs
    save_steps=500,              
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
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
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



#%%
# Match dtype of model (if loaded with bf16)
dtype = next(model.parameters()).dtype
for n, p in model.named_parameters():
    if p.requires_grad:
        p.data = p.data.to(dtype)

print("Model dtype:", next(model.parameters()).dtype)
trainable_dtypes = {p.dtype for n, p in model.named_parameters() if p.requires_grad}
print("Trainable dtypes:", trainable_dtypes)


#%%
# =========================== Create logging utility =============================

from transformers import TrainerCallback

# === Custom Callback ===
from collections import deque
from transformers import TrainerCallback

class MovingAverageLossCallback(TrainerCallback):
    """
    Logs running/moving loss, learning rate, and gradient norm for LoRA parameters.
    """

    def __init__(self, logging_interval=10, moving_average_window=50):
        self.logging_interval = logging_interval
        self.moving_window = deque(maxlen=moving_average_window)
        self.running_loss = 0.0
        self.steps = 0
        self.last_grad_norm = 0.0
        self.last_grad_count = 0

    def _get_lora_grad_norm(self, model):
        """Compute total grad norm across LoRA parameters only."""
        total_norm = 0.0
        count = 0
        for n, p in model.named_parameters():
            if p.requires_grad and "lora" in n and p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                count += 1
        return (total_norm ** 0.5, count)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Capture gradients BEFORE optimizer clears them."""
        if model is not None:
            self.last_grad_norm, self.last_grad_count = self._get_lora_grad_norm(model)

    def on_log(self, args, state, control, logs=None, model=None, optimizer=None, **kwargs):
        if logs is None or "loss" not in logs:
            return

        loss = logs["loss"]
        self.running_loss += loss
        self.steps += 1
        self.moving_window.append(loss)

        if state.global_step % self.logging_interval == 0:
            running_avg = self.running_loss / self.steps
            moving_avg = sum(self.moving_window) / len(self.moving_window)
            self.running_loss = 0.0
            self.steps = 0

            # Get LR
            lr = optimizer.param_groups[0]["lr"] if optimizer else None

            print(f"[Step {state.global_step}] "
                  f"Loss: {loss:.4f}, "
                  f"Running Avg: {running_avg:.4f}, "
                  f"Moving Avg: {moving_avg:.4f}, "
                  f"LoRA Grad Norm: {self.last_grad_norm:.3f} ({self.last_grad_count} params), "
                  f"LR: {lr:.2e}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            print(f"\n[Eval @ Step {state.global_step}] Eval Loss: {metrics['eval_loss']:.4f}\n")

# %%
# ======================= configure the model for sft =======================


trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[MovingAverageLossCallback()],
    )


# #%%
# # Put model in train mode just to be explicit
# model.train()

# # Check whether parameters require gradients
# requires_grad_list = [(n, p.requires_grad) for n, p in model.named_parameters()]

# # Print summary
# frozen = [n for n, g in requires_grad_list if not g]
# trainable = [n for n, g in requires_grad_list if g]

# print(f"Total parameters: {len(requires_grad_list)}")
# print(f"Trainable: {len(trainable)} | Frozen: {len(frozen)}")

# # Optionally show examples of frozen params if any
# if frozen:
#     print("Frozen parameters (first 10):")
#     for name in frozen[:10]:
#         print(f"  {name}")
# else:
#     print("✅ All parameters are trainable.")

# #%%
# # Inside or right after trainer = SFTTrainer(...)
# opt_params = list(trainer.model.parameters())
# trainable_params = [p for p in opt_params if p.requires_grad]

# print(f"Trainable parameter count: {sum(p.numel() for p in trainable_params):,}")
# print(f"Total parameter count: {sum(p.numel() for p in opt_params):,}")

#%%
import pprint
pprint.pprint(trainer.train_dataset[2]['text'])

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
# Right before trainer.train()
print("\n=== LoRA Parameter Check ===")
lora_params = [(n, p.requires_grad, p.shape) for n, p in trainer.model.named_parameters() if 'lora' in n]
print(f"Total LoRA parameters found: {len(lora_params)}")
if lora_params:
    print("First 5 LoRA parameters:")
    for n, req_grad, shape in lora_params[:5]:
        print(f"  {n}: requires_grad={req_grad}, shape={shape}")
else:
    print("⚠️ WARNING: No LoRA parameters found!")

#%%
# BEFORE trainer.train()
print("\n=== Capturing LoRA params BEFORE training ===")
lora_params_before = {}
for n, p in trainer.model.named_parameters():
    if 'lora' in n and p.requires_grad:
        lora_params_before[n] = p.detach().clone()

example_param_name = list(lora_params_before.keys())[0]
print(f"Example param: {example_param_name}")
print(f"First 5 values: {lora_params_before[example_param_name].flatten()[:5]}")

#%%
trainer.train()


#%%
# AFTER trainer.train()
print("\n=== Checking LoRA params AFTER training ===")
params_changed = 0
params_unchanged = 0

for n, p in trainer.model.named_parameters():
    if 'lora' in n and p.requires_grad and n in lora_params_before:
        before = lora_params_before[n]
        after = p.detach()
        
        if not torch.allclose(before, after, rtol=1e-5):
            params_changed += 1
        else:
            params_unchanged += 1

print(f"Parameters that CHANGED: {params_changed}")
print(f"Parameters UNCHANGED: {params_unchanged}")

# Show example
print(f"\nExample param: {example_param_name}")
print(f"Before: {lora_params_before[example_param_name].flatten()[:5]}")
print(f"After:  {trainer.model.state_dict()[example_param_name].flatten()[:5]}")
# %%
