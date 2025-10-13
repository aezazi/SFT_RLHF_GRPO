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
# # 1.  load and inspect the saved custom tokenizer
tokenizer = AutoTokenizer.from_pretrained("./tokenizer_with_specials")
tokenizer.pad_token
#%%
# ============================================================================
# 1. TOKENIZER SETUP WITH CUSTOM CHAT TEMPLATE
# ============================================================================

model_name = "mistralai/Mistral-7B-v0.1"

# Load tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # Add special tokens
# special_tokens_dict = {
#     "pad_token": "<|pad|>",
#     "additional_special_tokens": [
#         "<|im_start|>",
#         "<|im_end|>",
#         "<|endoftext|>"
#     ]
# }

# tokenizer.add_special_tokens(special_tokens_dict)
# tokenizer.padding_side = 'right'
# tokenizer.add_bos_token = False
# tokenizer.add_eos_token = False

# Define ChatML template
chat_template = """{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}{% elif message['role'] == 'user' %}{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}{% elif message['role'] == 'assistant' %}{{ '<|im_start|>assistant\n' }}{% generation %}{{ message['content'] }}{% endgeneration %}{{ '<|im_end|>\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"""

tokenizer.chat_template = chat_template

print(f"Tokenizer vocabulary size: {len(tokenizer)}")
print(f"Special tokens added: {tokenizer.all_special_tokens}")

#%%
# Verify it works
test_messages = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"}
]

result = tokenizer.apply_chat_template(
    test_messages,
    tokenize=True,
    return_assistant_tokens_mask=True,
    return_dict=True,
    add_generation_prompt=False
)

print(f"Result type: {type(result)}")
if 'assistant_masks' in result:
    print("✓ Chat template supports assistant_only_loss!")
    print(f"Input IDs: {result['input_ids']}")
    print(f"Assistant mask: {result['assistant_masks']}")
    print(f"\nMask breakdown:")
    tokens = tokenizer.convert_ids_to_tokens(result['input_ids'])
    for token, mask in zip(tokens, result['assistant_masks']):
        indicator = "→ TRAIN" if mask == 1 else ""
        print(f"  {token:20s} mask={mask} {indicator}")
else:
    print("✗ Chat template does NOT support assistant_only_loss")



#%%
# define quantization options and configuration

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
# ============================================================================
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
# 5. LOAD DATASET AND CREATE FUNCTION TO APPLY CHAT TEMPLATE
# ============================================================================

# Load UltraChat dataset
dataset = load_dataset("HuggingFaceH4/ultrachat_200k")
train_dataset = dataset["train_sft"]
eval_dataset = dataset["test_sft"]

print(f"Train dataset size: {len(train_dataset)}")
print(f"Eval dataset size: {len(eval_dataset)}")

# Formatting function to apply chat template
def formatting_func(example):
    """Apply chat template and return dict with input_ids"""
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}

#%%
# ============================================================================
# 6. TRAINING CONFIGURATION (SFTConfig replaces TrainingArguments)
# ============================================================================

output_dir = "./mistral-7b-ultrachat-sft"

training_args = SFTConfig(
    # Output and logging
    output_dir=output_dir,
    overwrite_output_dir=True,
    report_to="tensorboard",            # Change to "wandb" if using Weights & Biases
    logging_dir=f"{output_dir}/logs",
    logging_steps=10,
    logging_strategy="steps",
    
    # Training regime
    num_train_epochs=3,                 # Number of epochs
    max_steps=-1,                       # -1 means use num_train_epochs
    
    # Batch sizes - Optimized for H200
    per_device_train_batch_size=8,      # Large batch possible with H200
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,      # Effective batch size = 8*2 = 16
    
    # Optimization
    learning_rate=2e-4,                 # Standard for LoRA
    lr_scheduler_type="cosine",         # Cosine decay with warmup
    warmup_ratio=0.03,                  # 3% warmup steps
    weight_decay=0.01,                  # L2 regularization
    max_grad_norm=1.0,                  # Gradient clipping
    optim="paged_adamw_8bit",          # Memory-efficient optimizer
    
    # Precision
    bf16=True,                          # Use bfloat16 (H200 supports it)
    bf16_full_eval=True,               # Use bf16 for evaluation too
    
    # Memory optimizations
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},  # More stable
    
    # Evaluation
    do_eval=True,
    eval_strategy="epoch",        # Evaluate at end of each epoch
    eval_steps=None,                    # Not used when strategy="epoch"
    
    # Saving
    save_strategy="epoch",              # Save at end of each epoch
    save_total_limit=2,                 # Keep only 2 best checkpoints
    load_best_model_at_end=True,       # Load best model after training
    metric_for_best_model="eval_loss", # Use eval loss to determine best
    greater_is_better=False,           # Lower loss is better
    
    # Reproducibility
    seed=42,
    data_seed=42,
    
    # Performance
    dataloader_num_workers=4,          # Parallel data loading
    dataloader_pin_memory=True,        # Faster data transfer to GPU
    
    # Misc
    log_level="info",
    disable_tqdm=False,
    
    # ========================================================================
    # SFT-SPECIFIC PARAMETERS (New in SFTConfig)
    # ========================================================================
    
    # Dataset formatting
    dataset_text_field=None,           # We use formatting_func instead
    
    # Sequence handling
    max_length=2048,               # Maximum sequence length
    packing=True,                     # Set True if sequences are short/variable
    
    # RESPONSE MASKING (replaces DataCollatorForCompletionOnlyLM)
    # completion_only_loss=True,         # Only compute loss on completions
    # assistant_only_loss=True, 
    
    # Additional SFT configs
    dataset_kwargs={
        "add_special_tokens": False,   # We handle special tokens in template
        "append_concat_token": False,  # Don't add extra tokens
    },
    
)


#%%
# 7. SFT TRAINER CONFIGURATION
# ============================================================================

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,  # Use this parameter name
    formatting_func=formatting_func,
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

#%%

# %%
