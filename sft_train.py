#%%
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset

model_name = "mistralai/Mistral-7B-v0.1"

#%%
#============================================================================
# 1. LOAD TOKENIZER WE CREATED
#============================================================================

tokenizer = AutoTokenizer.from_pretrained("./tokenizer_with_specials")
# inspect special tokens
print(f"Padding token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
print(f"Special tokens: {tokenizer.additional_special_tokens}")
print(f" <|im_start|> token: {tokenizer.additional_special_tokens[0]}")
print(f"Tokenizer vocabulary size: {len(tokenizer)}")
print(f"Special tokens added: {tokenizer.all_special_tokens}")


#%%
# load formatted datasets 
# ============================================================================
# 2. LOAD FORMATTED DATASETS
# ============================================================================
from datasets import load_from_disk

train_dataset_formatted = load_from_disk("./ultrachat_train_formatted")
eval_dataset_formatted = load_from_disk("./ultrachat_eval_formatted")

#%%
# define quantization options and configuratio
# ============================================================================
# 2. QUANTIZATION CONFIGURATION
# ============================================================================

# OPTION 1: 4-bit Quantization (~3.5GB VRAM for model)
# - Fastest training, lowest memory
# - Good quality with LoRA
# - Best for experimentation
bnb_config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,                    # Use 4-bit quantization
    bnb_4bit_quant_type="nf4",           # NormalFloat4 (recommended)
    bnb_4bit_compute_dtype=torch.bfloat16, # Compute in bf16
    bnb_4bit_use_double_quant=True,      # Nested quantization for more memory savings
)

# OPTION 2: 8-bit Quantization (~7GB VRAM for model)
# - Better quality than 4-bit
# - Still very memory efficient
# - Recommended for H200 - best quality/speed tradeoff
bnb_config_8bit = BitsAndBytesConfig(
    load_in_8bit=True,                   # Use 8-bit quantization
    llm_int8_threshold=6.0,              # Threshold for outlier detection
    llm_int8_has_fp16_weight=False,      # Use int8 weights
)

# OPTION 3: Full Precision (No Quantization) (~14GB VRAM for model in bf16)
# - Best quality
# - Slower than quantized options
# - Your H200 can easily handle this
# - Recommended if you want maximum performance
bnb_config_full = None  # Don't pass quantization_config to from_pretrained


# bnb_config = bnb_config_4bit    # Fast, efficient (recommended for starting)
bnb_config = bnb_config_8bit  # Better quality (recommended for H200)
# bnb_config = None             # Full precision (best quality, H200 can handle it)

# For quantized models (4-bit or 8-bit)
if bnb_config is not None:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",                    # Automatic device placement
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # Enable Flash Attention 2
        dtype=torch.bfloat16,          # Use bf16 for non-quantized layers
    )
    
    # Resize embeddings to accommodate new tokens
    model.resize_token_embeddings(len(tokenizer))
    
    # Prepare model for k-bit training (enables gradient checkpointing, etc.)
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True
    )
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    print(f"Model loaded with quantization and Flash Attention 2")

# For full precision (no quantization)
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        dtype=torch.bfloat16,          # Full bf16 precision
    )
    
    # Resize embeddings to accommodate new tokens
    model.resize_token_embeddings(len(tokenizer))
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    print(f"Model loaded in full precision (bf16) with Flash Attention 2")

print(f"Model dtype: {model.dtype}")


# %%
#=============================================================================
# 4. LORA CONFIGURATION
# ============================================================================

peft_config = LoraConfig(
    r=64,                                # LoRA rank (higher = more parameters, better quality)
    lora_alpha=128,                      # LoRA scaling factor (typically 2*r)
    lora_dropout=0.05,                   # Dropout for regularization
    bias="none",                         # Don't train bias parameters
    task_type="CAUSAL_LM",              # Causal language modeling
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

# Apply LoRA to the model
model = get_peft_model(model, peft_config)

# Print trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)")


#%%
# ============================================================================
# 6. TRAINING CONFIGURATION (SFTConfig replaces TrainingArguments)
# ============================================================================

output_dir = "./mistral-7b-ultrachat-sft"

training_args = SFTConfig(
    output_dir=output_dir,
    overwrite_output_dir=True,
    report_to="tensorboard",
    logging_dir=f"{output_dir}/logs",
    logging_steps=10,
    logging_strategy="steps",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.01,
    max_grad_norm=1.0,
    optim="paged_adamw_8bit",
    bf16=True,
    bf16_full_eval=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    do_eval=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    seed=42,
    data_seed=42,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    log_level="info",
    disable_tqdm=False,
    dataset_text_field=None,
    max_length=2048,
    packing=True,                        # Set to False if packing issues arise
    completion_only_loss=False,
    # response_template="<|im_start|>assistant\n",  # <-- ADD THIS
    dataset_kwargs={
        "add_special_tokens": False,
        "append_concat_token": False,
    },
)


#%%
# Configure SFT Trainer
# ============================================================================
# 7. SFT TRAINER CONFIGURATION
# ============================================================================

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_formatted,   # pre-formatted
    eval_dataset=eval_dataset_formatted,     # pre-formatted
)

#%%
# ============================================================================
# 8. TRAINING
# ============================================================================

print("\n" + "="*80)
print("STARTING TRAINING")
print("="*80)
print(f"Train dataset size: {len(train_dataset)}")
print(f"Eval dataset size: {len(eval_dataset)}")
print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"Total training steps: {len(train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps) * training_args.num_train_epochs}")
print(f"Quantization: {'4-bit' if bnb_config == bnb_config_4bit else '8-bit' if bnb_config == bnb_config_8bit else 'Full precision'}")
print(f"Flash Attention 2: Enabled")
print("="*80 + "\n")

# Train the model
trainer.train()
