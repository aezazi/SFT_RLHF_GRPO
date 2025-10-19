#%%
# ============ VERIFY CLEAN START ============
import torch
import gc

mem_alloc = torch.cuda.memory_allocated() / 1e9
mem_reserved = torch.cuda.memory_reserved() / 1e9

print(f"GPU memory allocated: {mem_alloc:.1f}G")
print(f"GPU memory reserved: {mem_reserved:.1f}G")

if mem_alloc > 0.5:
    print("❌ GPU not clean! Kill Python process and restart")
else:
    print("✅ Clean start confirmed")

#%%
# ============ IMPORTS ============
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
from datasets import load_from_disk
import dynamic_padding_util

#%%
# ============ LOAD TOKENIZER ============
tokenizer = AutoTokenizer.from_pretrained("./tokenizer_with_specials")
print(f"✅ Tokenizer: {len(tokenizer)} tokens")
print(f"   GPU memory: {torch.cuda.memory_allocated() / 1e9:.1f}G")

#%%
# ============ LOAD DATASETS ============
train_dataset_formatted = load_from_disk("./ultrachat_train_formatted")
eval_dataset_formatted = load_from_disk("./ultrachat_eval_formatted")
print(f"✅ Datasets loaded")
print(f"   GPU memory: {torch.cuda.memory_allocated() / 1e9:.1f}G")

#%%
# ============ LOAD MODEL ============
print("\n" + "="*70)
print("Loading model...")
print("="*70)

mem_before = torch.cuda.memory_allocated() / 1e9
print(f"GPU memory before: {mem_before:.1f}G")

model = AutoModelForCausalLM.from_pretrained(
    "./model_with_specials",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    # attn_implementation="sdpa",
    attn_implementation="flash_attention_2",
)

model.gradient_checkpointing_enable()
print(f"✅ Gradient checkpointing enabled: {model.is_gradient_checkpointing}")

mem_after = torch.cuda.memory_allocated() / 1e9
print(f"GPU memory after: {mem_after:.1f}G")
print(f"Model added: +{mem_after - mem_before:.1f}G")

if mem_after > 20:
    print(f"\n❌ CRITICAL: Model using {mem_after:.1f}G (should be ~14-15G)")
    print(f"Your saved model file is corrupted!")
    print(f"You need to recreate it from base Mistral")
elif mem_after < 12:
    print(f"\n⚠️  Model using {mem_after:.1f}G (seems low, expected ~14-15G)")
else:
    print(f"\n✅ Model memory looks correct!")

print("="*70)

#%%
# ============ APPLY LORA ============
peft_config = LoraConfig(
    r=64,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

mem_with_lora = torch.cuda.memory_allocated() / 1e9
print(f"\nGPU memory with LoRA: {mem_with_lora:.1f}G")

#%%
# ============ CONFIGURE TRAINING ============
data_collator = dynamic_padding_util.DataCollatorForDynamicPadding(tokenizer=tokenizer)

#%%
# ============ CONFIGURE TRAINING ============

training_args = SFTConfig(
    output_dir="./model_logs_full_conv_v2",
    overwrite_output_dir=True,
    report_to="tensorboard",
    logging_dir="./model_logs_full_conv_v2/logs",
    logging_steps=1,
    logging_strategy="steps",
    
    num_train_epochs=1,
    max_steps=-1,
    
    # OPTIMIZED: batch_size=24 with checkpointing
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=32,
    
    # CRITICAL CHANGE: Lower LR for full conversation training
    learning_rate=2e-5,  # ← Changed from 1e-4 to 3e-5
    lr_scheduler_type="constant_with_warmup",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    max_grad_norm=1.0,
    optim="adamw_torch_fused",
    
    bf16=True,
    bf16_full_eval=True,
    tf32=True,
    
    # ENABLE gradient checkpointing (needed for long sequences)
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    
    do_eval=True,
    eval_strategy="steps",
    eval_steps=400,
    save_strategy="steps",
    save_steps=400,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    
    seed=42,
    data_seed=42,
    
    dataloader_num_workers=8,
    dataloader_pin_memory=True,
    dataloader_prefetch_factor=2,
    log_level="info",
    disable_tqdm=False,
    
    # max_length=3072,  # ← Correct parameter name
    packing=False,
    dataset_text_field=None,
    dataset_kwargs={
        "add_special_tokens": False,
        "append_concat_token": False,
    },
)

#%%
# ============ CREATE TRAINER WITH CHECKPOINTING ============

model.gradient_checkpointing_enable()
print(f"✅ Gradient checkpointing enabled: {model.is_gradient_checkpointing}")

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=train_dataset_formatted,
    eval_dataset=eval_dataset_formatted,
    data_collator=data_collator,
)

#%%
# ============ TEST MEMORY WITH CHECKPOINTING ON ============
print("\n" + "="*70)
print(f"TESTING WITH GRADIENT CHECKPOINTING ON, BATCH_SIZE: {training_args.per_device_train_batch_size}   accumulation steps: {training_args.gradient_accumulation_steps}")
print("="*70)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

test_batch = next(iter(trainer.get_train_dataloader()))
test_batch = {k: v.to(model.device) for k, v in test_batch.items()}

print(f"Batch shape: {test_batch['input_ids'].shape}")

try:
    model.train()
    outputs = model(**test_batch)
    loss = outputs.loss
    
    mem_forward = torch.cuda.memory_allocated() / 1e9
    print(f"After forward: {mem_forward:.1f}G")
    
    loss.backward()
    
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"Peak memory: {peak:.1f}G / 98G")
    
    if peak < 85:
        print(f"\n✅ EXCELLENT! batch_size=24 works with checkpointing!")
        print(f"   This is your optimal configuration")
    else:
        print(f"\n⚠️  batch_size={training_args.per_device_train_batch_size} at limit, may need to reduce")
    
    model.zero_grad()
    print("\n✅ Memory test PASSED!")
    
except RuntimeError as e:
    if "out of memory" in str(e):
        print(f"\n❌ OOM with batch_size={training_args.per_device_train_batch_size}")
        print(f"   Try lower batch size")
    else:
        raise e

print("="*70)

#%%
# ============ COMPILE ======================================

if torch.__version__ >= "2.0.0":
    print("\n🚀 Compiling model...")
    trainer.model = torch.compile(trainer.model, mode="reduce-overhead")

#%%
# ============ CHECK IF TORCH.COMPILE IS ACTIVE ============

print("="*70)
print("CHECKING TORCH.COMPILE STATUS")
print("="*70)

# Method 1: Check for compilation wrapper
print("\n1. Check Model Type:")
print(f"   Model type: {type(trainer.model)}")
print(f"   Model class: {trainer.model.__class__.__name__}")

# Method 2: Check for _orig_mod attribute (definitive test)
print("\n2. Check for Compilation Wrapper:")
if hasattr(trainer.model, '_orig_mod'):
    print(f"   ✅ Model IS compiled (has _orig_mod attribute)")
    print(f"   Original model type: {type(trainer.model._orig_mod)}")
else:
    print(f"   ❌ Model is NOT compiled (no _orig_mod attribute)")

# Method 3: Check for __wrapped__ attribute
print("\n3. Check for __wrapped__:")
if hasattr(trainer.model, '__wrapped__'):
    print(f"   ✅ Model has __wrapped__ (compilation wrapper present)")
else:
    print(f"   ℹ️  No __wrapped__ attribute")

# Method 4: Check torch version
print("\n4. PyTorch Version:")
import torch
print(f"   PyTorch: {torch.__version__}")
if torch.__version__ >= "2.0.0":
    print(f"   ✅ torch.compile is available (PyTorch >= 2.0)")
else:
    print(f"   ❌ torch.compile NOT available (need PyTorch >= 2.0)")

print("="*70)

# Summary
print("\n📊 SUMMARY:")
if hasattr(trainer.model, '_orig_mod'):
    print("   ✅ torch.compile IS ENABLED and ACTIVE")
    print("   Your model is benefiting from compilation optimizations")
else:
    print("   ❌ torch.compile is NOT active")
    print("   Your model is running without compilation")

print("="*70)

#%%
# ============ CHECK IF FLASH ATTENTION 2 IS ACTIVE ============

print("="*70)
print("CHECKING FLASH ATTENTION 2 STATUS")
print("="*70)

# Method 1: Check model config
print("\n1. Model Configuration:")
print(f"   Attention implementation: {model.config._attn_implementation}")

if model.config._attn_implementation == "flash_attention_2":
    print(f"   ✅ Config says: flash_attention_2")
elif model.config._attn_implementation == "sdpa":
    print(f"   ⚠️  Config says: sdpa (PyTorch native)")
elif model.config._attn_implementation == "eager":
    print(f"   ⚠️  Config says: eager (standard attention)")
else:
    print(f"   ❓ Config says: {model.config._attn_implementation}")

# Method 2: Check if flash_attn package is installed
print("\n2. Flash Attention Package:")
try:
    import flash_attn
    print(f"   ✅ flash-attn installed: version {flash_attn.__version__}")
except ImportError:
    print(f"   ❌ flash-attn NOT installed")
    print(f"   → Model cannot use Flash Attention 2!")

# Method 3: Check actual attention module implementation
print("\n3. Checking Actual Attention Modules:")
try:
    # Navigate to actual attention layers
    if hasattr(model, 'base_model'):
        base = model.base_model.model  # LoRA wrapped model
    else:
        base = model
    
    # Check first attention layer
    first_layer = base.model.layers[0]
    attn_class = first_layer.self_attn.__class__.__name__
    
    print(f"   Attention class: {attn_class}")
    
    # Check for Flash Attention specific attributes
    if hasattr(first_layer.self_attn, 'config'):
        if hasattr(first_layer.self_attn.config, '_attn_implementation'):
            impl = first_layer.self_attn.config._attn_implementation
            print(f"   Layer config: {impl}")
    
    # Flash Attention 2 uses specific class names
    if "Flash" in attn_class:
        print(f"   ✅ Using Flash Attention class!")
    elif "SDPA" in attn_class or "sdpa" in attn_class.lower():
        print(f"   ⚠️  Using SDPA class")
    else:
        print(f"   ℹ️  Standard attention class")
        
except Exception as e:
    print(f"   ⚠️  Could not inspect: {e}")

# Method 4: Test with actual forward pass (advanced)
print("\n4. Testing Attention Behavior:")
try:
    import torch
    
    # Create tiny test input
    test_input = torch.randint(0, 32000, (1, 128), device='cuda')
    test_mask = torch.ones((1, 128), device='cuda')
    
    # Check if flash_attn is called during forward pass
    # We can't directly test this without profiling, but we can check memory pattern
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    model.eval()
    with torch.no_grad():
        _ = model(input_ids=test_input, attention_mask=test_mask)
    
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"   Forward pass peak memory: {peak_mem:.2f}G")
    
    # Flash Attention 2 should use less memory for same sequence length
    # This is a rough heuristic
    if peak_mem < 1.0:  # Very small for 128 token test
        print(f"   ✅ Memory pattern consistent with Flash Attention 2")
    else:
        print(f"   ℹ️  Memory usage: {peak_mem:.2f}G")
    
except Exception as e:
    print(f"   ⚠️  Could not test: {e}")

print("\n" + "="*70)

# Summary
print("📊 SUMMARY:")
print("="*70)

has_flash = False
try:
    import flash_attn
    has_flash = True
except:
    pass

if model.config._attn_implementation == "flash_attention_2" and has_flash:
    print("✅ Flash Attention 2 IS ENABLED AND ACTIVE")
    print("   Your model is using Flash Attention 2 for fast computation")
    print("   This explains the ~7,000 tok/s throughput!")
elif model.config._attn_implementation == "flash_attention_2" and not has_flash:
    print("⚠️  WARNING: Config says flash_attention_2 but package not installed!")
    print("   Model likely fell back to SDPA")
elif model.config._attn_implementation == "sdpa":
    print("ℹ️  Using SDPA (PyTorch Scaled Dot Product Attention)")
    print("   This is fast but not as fast as Flash Attention 2")
else:
    print(f"ℹ️  Using: {model.config._attn_implementation}")

print("="*70)


#%%
# =================== LOGGING UTILITY ================================

# Add callback
from transformers import TrainerCallback
import time
import torch.distributed as dist

class VerboseTrainingCallback(TrainerCallback):
    def __init__(self, trainer=None):
        self.last_time = None
        self.last_step = 0
        self.loss_history = []
        self.start_time = time.time()
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        
        step = state.global_step
        loss = logs.get("loss")
        lr = logs.get("learning_rate")
        
        if loss:
            self.loss_history.append(loss)
            if len(self.loss_history) > 50:
                self.loss_history.pop(0)
        
        now = time.time()
        tokens_per_sec = None
        
        if self.last_time is not None:
            elapsed = now - self.last_time
            if elapsed > 0:
                batch_size = args.per_device_train_batch_size
                seq_len = getattr(args, "max_length", 2048)
                tokens_per_sec = (batch_size * seq_len * (step - self.last_step)) / elapsed
        
        self.last_time = now
        self.last_step = step
        
        rank = dist.get_rank() if dist.is_initialized() else 0
        
        if rank == 0:
            mem = torch.cuda.memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            log_str = f"[Step {step:>5}] "
            if loss:
                log_str += f"loss={loss:.4f} "
                if len(self.loss_history) >= 10:
                    avg = sum(self.loss_history[-10:]) / 10
                    log_str += f"(avg={avg:.4f}) "
            if lr:
                log_str += f"lr={lr:.2e} "
            if tokens_per_sec:
                log_str += f"| {tokens_per_sec:>6,.0f} tok/s "
            log_str += f"| GPU: {mem:.1f}G/{total:.0f}G ({mem/total*100:.0f}%)"
            
            print(log_str)

trainer.add_callback(VerboseTrainingCallback(trainer=trainer))


#%%
# ============ TRAIN ================================
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

print("\n" + "="*70)
print("🚀 STARTING TRAINING")
print("="*70)
print(f"Batch size: {training_args.per_device_train_batch_size}")
print(f"Gradient checkpointing: {training_args.gradient_checkpointing}")
print(f"Max sequence length: {training_args.max_length}")
print("="*70 + "\n")

trainer.train()

# %%

