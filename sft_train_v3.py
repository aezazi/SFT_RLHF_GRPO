
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

train_dataset = load_from_disk("./ultrachat_train_formatted")
eval_dataset = load_from_disk("./ultrachat_eval_formatted").select(range(10000))

print(len(train_dataset))
print(len(eval_dataset))

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
    logging_steps=10,                   # print every 10 steps (adjust as desired)
    log_level="info",
    disable_tqdm=False, 

    # -------------------------
    # Batch Sizes - Optimized for H200
    # -------------------------
    per_device_train_batch_size=16, # originally set to 8
    per_device_eval_batch_size=16, # originally set to 8
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
    save_steps=1000,              
    save_total_limit=3,
    load_best_model_at_end=False,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    
    # push_to_hub=True,
    # hub_model_id="zephyr-7b-sft-lora",
    # hub_strategy="every_save",
    
    dataset_text_field="text",
    

    # Sequence handling
    max_length=4096, 
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
        "gate_proj", # MLP gate
        "up_proj",   # MLP up
        "down_proj", # MLP down
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
import csv
from datetime import datetime

# Simplified callback for just loss tracking
class SimpleStepCallback(TrainerCallback):
    """Log to CSV file at user-defined intervals."""
    
    def __init__(self, logging_steps=10, log_file="training_progress.csv"):
        self.logging_steps = logging_steps
        self.log_file = log_file
        self.start_time = datetime.now()
        
        # Create CSV file with header
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Step', 'Loss', 'Learning_Rate', 'Elapsed_Time_Minutes'])
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.logging_steps == 0:
            if state.log_history:
                last_log = state.log_history[-1]
                loss = last_log.get('loss', None)
                lr = last_log.get('learning_rate', None)
                
                if loss is not None:
                    elapsed = (datetime.now() - self.start_time).total_seconds() / 60
                    
                    with open(self.log_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([state.global_step, loss, lr if lr else 0, round(elapsed, 2)])

# %%
# ======================= configure the model for sft =======================
trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[SimpleStepCallback()],
    )

#%%
# ======================= check trainer formatting and special tokens =======================
import pprint
pprint.pprint(trainer.train_dataset[2]['text'])

#%%
# ========================= Compile the model ==================================
# if torch.__version__ >= "2.0.0":
#     print("\n" + "="*70)
#     print("🚀 Compiling model with torch.compile...")
#     print("="*70 + "\n")
    
#     # Suppress the repetitive CUDA Graph warnings
#     import os
#     os.environ['TORCHDYNAMO_VERBOSE'] = '0'
    
#     trainer.model = torch.compile(trainer.model, mode="reduce-overhead")
    
#     print("✅ Model wrapped for compilation!\n")
# else:
#     print("⚠️  PyTorch version < 2.0. Skipping torch.compile (consider upgrading)")

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
# ==================== Check Lora parameters are present ===================
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
# =========================== Start training ====================================
trainer.train(resume_from_checkpoint="model_logs/checkpoint-2000")



# %%
# the code below is not part of training. It's a check to see if the trainer is properly inserting padding tokens
# Get a batch from the training dataloader
# After creating the trainer, before calling trainer.train()

# from torch.utils.data import DataLoader

# # Get a batch from the training dataloader
# dataloader = trainer.get_train_dataloader()
# batch = next(iter(dataloader))

# # Check the batch
# print("=== Batch Inspection ===")
# print(f"Batch keys: {batch.keys()}")
# print(f"Input IDs shape: {batch['input_ids'].shape}")
# print(f"Attention mask shape: {batch['attention_mask'].shape}")

# # Check if padding token is present
# pad_token_id = tokenizer.pad_token_id
# print(f"\nPadding token ID: {pad_token_id}")

# # Count padding tokens in first sample
# first_sample = batch['input_ids'][0]
# num_pad_tokens = (first_sample == pad_token_id).sum().item()
# total_tokens = len(first_sample)
# print(f"Sample 0: {num_pad_tokens}/{total_tokens} tokens are padding ({num_pad_tokens/total_tokens*100:.1f}%)")

# # Show where padding starts in first sample
# print(f"\nFirst sample input_ids (last 20 tokens):")
# print(first_sample[-20:])
# print(f"Attention mask (last 20):")
# print(batch['attention_mask'][0][-20:])

# # Check all samples in batch
# print(f"\n=== Padding Statistics for Batch ===")
# for i in range(len(batch['input_ids'])):
#     sample = batch['input_ids'][i]
#     num_pads = (sample == pad_token_id).sum().item()
#     seq_len = len(sample)
#     print(f"Sample {i}: Length={seq_len}, Padding={num_pads} ({num_pads/seq_len*100:.1f}%)")
# %%
