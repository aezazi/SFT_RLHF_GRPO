
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
import gc


#Clear GPU memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print("✅ GPU memory cleared")


#%%
#====================  1. LOAD SAVED MODEL AND TOKENIZER =====================
#=============================================================================
model_name = "model_with_specials"
tokenizer = AutoTokenizer.from_pretrained("tokenizer_with_specials")
# model = AutoModelForCausalLM.from_pretrained("model_with_specials")
# inspect special tokens
print(f"Padding token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
print(f"Special tokens: {tokenizer.additional_special_tokens}")
print(f" <|im_start|> token: {tokenizer.additional_special_tokens[0]}")
print(f"Tokenizer vocabulary size: {len(tokenizer)}")
print(f"Special tokens added: {tokenizer.all_special_tokens}")


#%% 
# ===================== 2. LOAD FORMATTED DATASETS ===========================
# ============================================================================
from datasets import load_from_disk

train_dataset_formatted = load_from_disk("./ultrachat_train_formatted")
eval_dataset_formatted = load_from_disk("./ultrachat_eval_formatted")

sample = train_dataset_formatted[0]
print(tokenizer.decode(sample["input_ids"]))
print(sample.keys())
print(f"Labels unique IDs: {set(sample['labels'])}")

ids = sample["input_ids"]
labels = sample["labels"]

for i, (tid, lbl) in enumerate(zip(ids, labels)):
    if lbl != -100:
        print(f"First unmasked label at position {i}: token_id={tid}, token={tokenizer.decode([tid])}")
        break

#%%
# ====================== 3. QUANTIZATION CONFIGURATION =======================
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
# bnb_config = bnb_config_8bit  # Better quality (recommended for H200)
bnb_config = None             # Full precision (best quality, H200 can handle it)

# For quantized models (4-bit or 8-bit)
if bnb_config is not None:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=None,                    # Automatic device placement
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # Enable Flash Attention 2
        dtype=torch.bfloat16,          # Use bf16 for non-quantized layers
        
    )
    
    # Resize embeddings to accommodate new tokens
    # model.resize_token_embeddings(len(tokenizer))
    
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
        device_map=None,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        # attn_implementation="sdpa",  # Use PyTorch's scaled_dot_product_attention
        dtype=torch.bfloat16,          # Full bf16 precision
    )
    
    # Resize embeddings to accommodate new tokens
    # model.resize_token_embeddings(len(tokenizer))
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    print(f"Model loaded in full precision (bf16) with Flash Attention 2")

print(f"Model dtype: {model.dtype}")


# %%
#===================== 4. LORA CONFIGURATION ================================
#=============================================================================

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
# ========================= 5. SET TRAINING PARAMETERS  ======================
# ============================================================================

# instantiate custom data collator to handle dynamic padding
import dynamic_padding_util
data_collator = dynamic_padding_util.DataCollatorForCompletionOnlyLM(tokenizer=tokenizer)

output_dir = "./model_logs"
# model.gradient_checkpointing_disable()
print(f"Gradient checkpointing: {model.is_gradient_checkpointing}")
training_args = SFTConfig(
    # -------------------------
    # Output and Logging
    # -------------------------
    output_dir=output_dir,
    overwrite_output_dir=True,
    report_to="tensorboard",           # optional: "wandb" if you prefer
    logging_dir=f"{output_dir}/logs",
    logging_steps=5,
    logging_strategy="steps",

    # -------------------------
    # Training Regime
    # -------------------------
    num_train_epochs=1,
    max_steps=-1,                       # use num_train_epochs

    # -------------------------
    # Batch Sizes - Optimized for H200
    # -------------------------
    per_device_train_batch_size=20,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,   

    # -------------------------
    # Optimization
    # -------------------------
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    # warmup_steps=100,
    weight_decay=0.01,
    max_grad_norm=1.0,
    optim="adamw_torch_fused",

    # -------------------------
    # Precision
    # -------------------------
    bf16=True,
    bf16_full_eval=True,
    tf32=True,

    # -------------------------
    # Memory Optimizations
    # -------------------------
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},  # more stable
    # Optional: activation offloading if needed
    # activation_offloading=True,  
    # activation_offloading_params={"device": "cpu"},

    # -------------------------
    # Evaluation
    # -------------------------
    do_eval=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

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
    log_level="info",
    disable_tqdm=False,

    # -------------------------
    # SFT-Specific Parameters
    # -------------------------
    # Sequence handling
    max_length=2048, 
    packing=False,                       # keeps sequences contiguous in memory

    # Masking - since dataset already has assistant-only masks
    completion_only_loss=False,          # <-- important
    # assistant_only_loss=False,         # optional, can remain False
    dataset_text_field=None,             # not used, pre-formatted dataset

    # Additional dataset kwargs
    dataset_kwargs={
        "add_special_tokens": False,    # handled in your tokenizer
        "append_concat_token": False,
    },
)

#%%
# ======================= 6. SFT TRAINER CONFIGURATION =======================
# ============================================================================
model = model.to("cuda")
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=train_dataset_formatted,   # pre-formatted
    eval_dataset=eval_dataset_formatted,     # pre-formatted
    data_collator=data_collator,
    
)

# params check
for n, p in model.named_parameters():
    if p.requires_grad:
        print("First trainable param:", n, p.shape)
        break

model.print_trainable_parameters()

#%%
# ======================= COMPILE THE MODEL =======================
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
#=========================== 7. LOGGING UTILITY  =============================
#===========================================================================
from transformers import TrainerCallback
import time
import torch
import torch.distributed as dist

class VerboseTrainingCallback(TrainerCallback):
    """
    Enhanced training logger that tracks:
    - Loss and learning rate
    - Training progress (steps, epochs, warmup status)
    - GPU memory usage
    - Throughput (tokens/sec)
    - Moving average loss for trend detection
    """

    def __init__(self, trainer=None):
        self.last_time = None
        self.last_step = 0
        self.trainer = trainer
        self.loss_history = []
        self.start_time = time.time()
        
    def on_train_begin(self, args, state, control, **kwargs):
        """Calculate total training steps at the start."""
        if self.trainer is not None:
            steps_per_epoch = len(self.trainer.get_train_dataloader()) // args.gradient_accumulation_steps
            self.total_steps = steps_per_epoch * args.num_train_epochs
            self.warmup_steps = int(args.warmup_ratio * self.total_steps) if hasattr(args, 'warmup_ratio') and args.warmup_ratio else args.warmup_steps
        else:
            # Fallback calculation
            self.total_steps = state.max_steps if state.max_steps > 0 else 10000
            self.warmup_steps = getattr(args, 'warmup_steps', 100)
        
        print("\n" + "="*70)
        print(f"Training Configuration:")
        print(f"  Total steps: {self.total_steps:,}")
        print(f"  Warmup steps: {self.warmup_steps:,}")
        print(f"  Epochs: {args.num_train_epochs}")
        print(f"  Batch size per device: {args.per_device_train_batch_size}")
        print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
        print(f"  Effective batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
        print(f"  Learning rate: {args.learning_rate:.2e}")
        print("="*70 + "\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        step = state.global_step
        loss = logs.get("loss", None)
        lr = logs.get("learning_rate", None)

        # Track loss history for moving average
        if loss is not None:
            self.loss_history.append(loss)
            # Keep last 50 losses for moving average
            if len(self.loss_history) > 50:
                self.loss_history.pop(0)

        # --- Get distributed info ---
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0

        # --- Estimate tokens/sec ---
        batch_size = args.per_device_train_batch_size
        seq_len = getattr(args, "max_seq_length", 2048)
        grad_accum = args.gradient_accumulation_steps

        # Global effective batch (across GPUs)
        global_batch = batch_size * world_size * grad_accum

        # Time tracking
        now = time.time()
        tokens_per_sec = None
        if self.last_time is not None:
            elapsed = now - self.last_time
            steps_since = step - self.last_step
            if elapsed > 0:
                approx_tokens_per_step = global_batch * seq_len
                tokens_per_sec = (approx_tokens_per_step * steps_since) / elapsed

        self.last_time = now
        self.last_step = step

        # --- GPU memory stats (only print from rank 0) ---
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated() / 1e9
            mem_reserved = torch.cuda.memory_reserved() / 1e9
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            mem_free = total_mem - mem_allocated
        else:
            mem_allocated = mem_reserved = mem_free = 0

        # --- Build log string ---
        if rank == 0:  # only main process prints
            # Progress indicator
            progress_pct = (step / self.total_steps * 100) if hasattr(self, 'total_steps') else 0
            log_str = f"[Step {step:>5}"
            
            if hasattr(self, 'total_steps'):
                log_str += f"/{self.total_steps}"
            
            log_str += f" ({progress_pct:.1f}%)]"
            
            # Warmup indicator
            if hasattr(self, 'warmup_steps') and step <= self.warmup_steps:
                warmup_pct = (step / self.warmup_steps * 100)
                log_str += f" [WARMUP {warmup_pct:.0f}%]"
            
            # Loss with moving average
            if loss is not None:
                log_str += f" loss={loss:.4f}"
                if len(self.loss_history) >= 10:
                    avg_loss = sum(self.loss_history[-10:]) / 10
                    log_str += f" (avg={avg_loss:.4f})"
            
            # Learning rate
            if lr is not None:
                log_str += f" lr={lr:.2e}"
            
            # Throughput
            if tokens_per_sec is not None:
                log_str += f" | {tokens_per_sec:>6,.0f} tok/s"
            
            # GPU memory (compact format)
            log_str += f" | GPU: {mem_allocated:.1f}G/{total_mem:.0f}G ({mem_allocated/total_mem*100:.0f}%)"
            
            print(log_str, flush=True)
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Print summary at end of each epoch."""
        rank = dist.get_rank() if dist.is_initialized() else 0
        
        if rank == 0:
            elapsed_time = time.time() - self.start_time
            hours = int(elapsed_time // 3600)
            minutes = int((elapsed_time % 3600) // 60)
            
            print("\n" + "="*70)
            print(f"Epoch {state.epoch:.0f} Complete!")
            print(f"  Time elapsed: {hours}h {minutes}m")
            print(f"  Steps completed: {state.global_step:,}")
            
            if len(self.loss_history) >= 10:
                recent_avg = sum(self.loss_history[-10:]) / 10
                print(f"  Recent avg loss: {recent_avg:.4f}")
            
            if hasattr(state, 'best_metric') and state.best_metric is not None:
                print(f"  Best eval loss so far: {state.best_metric:.4f}")
            
            print("="*70 + "\n")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Print final summary."""
        rank = dist.get_rank() if dist.is_initialized() else 0
        
        if rank == 0:
            total_time = time.time() - self.start_time
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            seconds = int(total_time % 60)
            
            print("\n" + "="*70)
            print("🎉 Training Complete!")
            print(f"  Total time: {hours}h {minutes}m {seconds}s")
            print(f"  Total steps: {state.global_step:,}")
            print(f"  Final loss: {self.loss_history[-1]:.4f}" if self.loss_history else "")
            
            if hasattr(state, 'best_metric') and state.best_metric is not None:
                print(f"  Best eval loss: {state.best_metric:.4f}")
            
            print("="*70 + "\n")

# Pass trainer reference for better step calculations
trainer.add_callback(VerboseTrainingCallback(trainer=trainer))

#%%
# ============ DIAGNOSTIC 1: Check What's Actually Running ============
print("="*70)
print("CONFIGURATION VERIFICATION")
print("="*70)

# Check training args
print(f"\n1. Training Args:")
print(f"   per_device_train_batch_size: {training_args.per_device_train_batch_size}")
print(f"   gradient_accumulation_steps: {training_args.gradient_accumulation_steps}")
print(f"   gradient_checkpointing: {training_args.gradient_checkpointing}")
print(f"   bf16: {training_args.bf16}")
print(f"   tf32: {training_args.tf32}")
print(f"   optim: {training_args.optim}")

# Check actual batch size being used
print(f"\n2. Actual DataLoader Batch:")
test_loader = trainer.get_train_dataloader()
test_batch = next(iter(test_loader))
print(f"   Batch shape: {test_batch['input_ids'].shape}")
print(f"   First dimension (batch_size): {test_batch['input_ids'].shape[0]}")

if test_batch['input_ids'].shape[0] != 20:
    print(f"   ❌ PROBLEM: Expected 20, got {test_batch['input_ids'].shape[0]}")
else:
    print(f"   ✅ Batch size is correct (20)")

# Check model type
print(f"\n3. Model Info:")
print(f"   Model type: {type(model)}")
print(f"   Model dtype: {model.dtype}")
print(f"   Is compiled: {'_orig_mod' in dir(model)}")  # torch.compile wrapper check

# Check if gradient checkpointing is actually off
print(f"\n4. Gradient Checkpointing Status:")
if hasattr(model, 'gradient_checkpointing'):
    print(f"   model.gradient_checkpointing: {model.gradient_checkpointing}")
if hasattr(model, 'is_gradient_checkpointing'):
    print(f"   model.is_gradient_checkpointing: {model.is_gradient_checkpointing}")

# Check actual memory usage with one step
print(f"\n5. Memory Test:")
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

test_batch_mem = next(iter(trainer.get_train_dataloader()))
test_batch_mem = {k: v.to(model.device) for k, v in test_batch_mem.items()}

mem_before = torch.cuda.memory_allocated() / 1e9
print(f"   Memory before forward: {mem_before:.1f}G")

model.train()
outputs = model(**test_batch_mem)
loss = outputs.loss

mem_forward = torch.cuda.memory_allocated() / 1e9
print(f"   Memory after forward: {mem_forward:.1f}G")

loss.backward()

mem_backward = torch.cuda.memory_allocated() / 1e9
peak_mem = torch.cuda.max_memory_allocated() / 1e9

print(f"   Memory after backward: {mem_backward:.1f}G")
print(f"   PEAK memory: {peak_mem:.1f}G")

if peak_mem < 30:
    print(f"   ❌ CRITICAL PROBLEM: Peak memory is way too low!")
    print(f"   Something is preventing proper batch size / disabling optimizations")
elif peak_mem < 50:
    print(f"   ⚠️  Memory lower than expected")
else:
    print(f"   ✅ Memory usage looks reasonable")

model.zero_grad()

print("="*70 + "\n")

#%%
#=========================== 8. TRAINING ====================================
#============================================================================
# Train the model

trainer.train()

# %%
# ============ DIAGNOSTIC: Verify Gradients Flow ==============
import numpy as np

print("\n" + "="*60)
print("DIAGNOSTIC: Checking if model trains properly")
print("="*60)

# Get a batch
test_batch = next(iter(trainer.get_train_dataloader()))
test_batch = {k: v.to(model.device) for k, v in test_batch.items()}

# Forward pass
model.train()
outputs = model(**test_batch)
loss = outputs.loss

print(f"Test forward pass - Loss: {loss.item():.4f}")

# Check if loss is computed correctly
assert not torch.isnan(loss), "❌ Loss is NaN!"
assert loss.item() > 0, "❌ Loss is not positive!"

# Backward pass
loss.backward()

# Check gradients
grad_norms = []
for name, param in model.named_parameters():
    if param.requires_grad and param.grad is not None:
        grad_norm = param.grad.norm().item()
        grad_norms.append(grad_norm)
        if len(grad_norms) <= 3:  # Print first 3
            print(f"  {name}: grad_norm={grad_norm:.6f}")

if len(grad_norms) == 0:
    print("❌ NO GRADIENTS! Model is not training!")
else:
    print(f"\n✅ Gradients flowing! {len(grad_norms)} parameters have gradients")
    print(f"   Mean grad norm: {np.mean(grad_norms):.6f}")
    print(f"   Max grad norm: {np.max(grad_norms):.6f}")

# Clean up
model.zero_grad()
print("="*60 + "\n")

# %%
# ============ CHECK 2: Label Distribution ==============
print("\n" + "="*60)
print("DIAGNOSTIC: Checking label distribution in batches")
print("="*60)

for i, batch in enumerate(trainer.get_train_dataloader()):
    if i >= 3:  # Check first 3 batches
        break
    
    labels = batch['labels']
    total_tokens = labels.numel()
    masked_tokens = (labels == -100).sum().item()
    trainable_tokens = total_tokens - masked_tokens
    
    print(f"\nBatch {i}:")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Masked (-100): {masked_tokens} ({100*masked_tokens/total_tokens:.1f}%)")
    print(f"  Trainable: {trainable_tokens} ({100*trainable_tokens/total_tokens:.1f}%)")
    
    if trainable_tokens == 0:
        print("  ❌ NO TRAINABLE TOKENS IN THIS BATCH!")
    elif trainable_tokens / total_tokens < 0.05:
        print("  ⚠️  Very few trainable tokens (< 5%)")
    else:
        print("  ✅ Reasonable number of trainable tokens")

print("="*60 + "\n")

# ============ CHECK 3: LoRA Status ==============
print("\n" + "="*60)
print("DIAGNOSTIC: Checking LoRA adapter status")
print("="*60)

model.print_trainable_parameters()

lora_layers = [name for name, _ in model.named_parameters() if 'lora' in name.lower()]
print(f"\nFound {len(lora_layers)} LoRA parameters")

if len(lora_layers) == 0:
    print("❌ NO LORA LAYERS FOUND!")
else:
    print(f"✅ LoRA is active. Sample layers:")
    for name in lora_layers[:5]:
        print(f"  - {name}")

print("="*60 + "\n")

# ============ CHECK 4: Loss Analysis ==============
print("\n" + "="*60)
print("DIAGNOSTIC: Analyzing loss computation")
print("="*60)

test_batch = next(iter(trainer.get_train_dataloader()))
test_batch = {k: v.to(model.device) for k, v in test_batch.items()}

with torch.no_grad():
    outputs = model(**test_batch)
    
    logits = outputs.logits
    labels = test_batch['labels']
    
    valid_mask = (labels != -100)
    num_valid = valid_mask.sum().item()
    
    print(f"Batch shape: {labels.shape}")
    print(f"Valid (non-masked) tokens: {num_valid} / {labels.numel()}")
    print(f"Percentage trainable: {100 * num_valid / labels.numel():.2f}%")
    
    if num_valid > 0:
        valid_logits = logits[valid_mask]
        valid_labels = labels[valid_mask]
        
        predicted_ids = valid_logits.argmax(dim=-1)
        
        correct = (predicted_ids == valid_labels).sum().item()
        accuracy = correct / num_valid
        
        print(f"Prediction accuracy: {accuracy:.2%}")
        print(f"Loss: {outputs.loss.item():.4f}")
        
        unique_predictions = torch.unique(predicted_ids)
        print(f"Unique predicted tokens: {len(unique_predictions)} / {num_valid}")
        
        if len(unique_predictions) < 10:
            print(f"⚠️  Model predicting very few unique tokens: {unique_predictions.tolist()[:20]}")
    else:
        print("❌ No valid tokens to train on!")

print("="*60 + "\n")
# %%
#%%
print(f"Model config attention: {model.config._attn_implementation}")
print(f"Expected: flash_attention_2")

# Also check if flash-attn is actually installed
try:
    import flash_attn
    print(f"✅ flash-attn installed: {flash_attn.__version__}")
except ImportError:
    print("❌ flash-attn NOT installed! This is the problem!")
# %%
#%%
# ============ CRITICAL CHECKS ============

print("="*70)
print("CRITICAL CONFIGURATION CHECKS")
print("="*70)

# 1. Check vocab size mismatch
print(f"\n1. Vocabulary Size Check:")
print(f"   Tokenizer vocab size: {len(tokenizer)}")
print(f"   Model config vocab size: {model.config.vocab_size}")
print(f"   Embedding weight shape: {model.get_input_embeddings().weight.shape}")
print(f"   LM head weight shape: {model.get_output_embeddings().weight.shape}")

if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
    print(f"   ❌ MISMATCH! Embedding has {model.get_input_embeddings().weight.shape[0]} but tokenizer has {len(tokenizer)}")
else:
    print(f"   ✅ Vocabulary sizes match")

# 2. Check LoRA application
print(f"\n2. LoRA Configuration:")
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"   Total parameters: {total:,}")
print(f"   Trainable parameters: {trainable:,}")
print(f"   Trainable %: {100*trainable/total:.2f}%")

if trainable / total > 0.1:  # More than 10%
    print(f"   ❌ WARNING: Training {100*trainable/total:.1f}% of model!")
    print(f"      LoRA may not be applied correctly")
elif trainable / total < 0.01:  # Less than 1%
    print(f"   ⚠️  Only training {100*trainable/total:.2f}% - very few parameters")
else:
    print(f"   ✅ LoRA looks correctly applied (~2-3% expected)")

# 3. Check actual GPU memory used by model
print(f"\n3. GPU Memory Usage:")
torch.cuda.empty_cache()
mem_allocated = torch.cuda.memory_allocated() / 1e9
mem_reserved = torch.cuda.memory_reserved() / 1e9
total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9

print(f"   Allocated: {mem_allocated:.1f}G")
print(f"   Reserved: {mem_reserved:.1f}G")
print(f"   Total GPU: {total_mem:.1f}G")
print(f"   Free: {total_mem - mem_allocated:.1f}G")

if mem_allocated > 25:
    print(f"   ❌ Model using {mem_allocated:.1f}G - too much for 7B model in BF16!")
elif mem_allocated < 12:
    print(f"   ⚠️  Model using only {mem_allocated:.1f}G - seems low")
else:
    print(f"   ✅ Model memory usage looks normal")

# 4. Check gradient checkpointing state
print(f"\n4. Gradient Checkpointing:")
print(f"   is_gradient_checkpointing: {model.is_gradient_checkpointing}")
if hasattr(model, 'base_model'):
    if hasattr(model.base_model, 'is_gradient_checkpointing'):
        print(f"   base_model.is_gradient_checkpointing: {model.base_model.is_gradient_checkpointing}")

# 5. Check attention implementation
print(f"\n5. Attention Implementation:")
print(f"   Config says: {model.config._attn_implementation}")

# 6. Test with MINIMAL batch
print(f"\n6. Testing MINIMAL Batch (1 example, 512 tokens):")
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

test_input = {
    'input_ids': torch.randint(0, len(tokenizer), (1, 512), device='cuda'),
    'attention_mask': torch.ones((1, 512), device='cuda'),
    'labels': torch.randint(0, len(tokenizer), (1, 512), device='cuda'),
}

try:
    model.train()
    
    mem_before = torch.cuda.memory_allocated() / 1e9
    
    outputs = model(**test_input)
    loss = outputs.loss
    
    mem_forward = torch.cuda.memory_allocated() / 1e9
    
    loss.backward()
    
    mem_backward = torch.cuda.memory_allocated() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    
    print(f"   Before: {mem_before:.1f}G")
    print(f"   After forward: {mem_forward:.1f}G (delta: +{mem_forward-mem_before:.1f}G)")
    print(f"   After backward: {mem_backward:.1f}G (delta: +{mem_backward-mem_forward:.1f}G)")
    print(f"   Peak: {peak:.1f}G")
    
    # For batch=1, seq=512, this should use ~20-25G total
    if peak > 35:
        print(f"   ❌ CRITICAL: Single small example used {peak:.1f}G!")
        print(f"      Expected ~20-25G max")
    else:
        print(f"   ✅ Memory usage looks reasonable")
    
    model.zero_grad()
    
except Exception as e:
    print(f"   ❌ ERROR: {e}")

torch.cuda.empty_cache()

# 7. Check sequence lengths in actual dataset
print(f"\n7. Checking Actual Sequence Lengths:")
if 'train_dataset_formatted' in globals():
    lengths = [len(ex['input_ids']) for ex in train_dataset_formatted.select(range(min(100, len(train_dataset_formatted))))]
    print(f"   Min: {min(lengths)}")
    print(f"   Max: {max(lengths)}")
    print(f"   Mean: {sum(lengths)/len(lengths):.0f}")
    print(f"   Median: {sorted(lengths)[len(lengths)//2]}")
    
    if max(lengths) > 6000:
        print(f"   ❌ CRITICAL: Some sequences are {max(lengths)} tokens!")
        print(f"      This explains the OOM")
        print(f"      With batch=8 and seq=6000+, you'd need massive memory")

print("="*70)
# %%
