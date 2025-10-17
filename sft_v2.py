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
    attn_implementation="sdpa",
)

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
    lora_alpha=128,
    lora_dropout=0.05,
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
data_collator = dynamic_padding_util.DataCollatorForCompletionOnlyLM(tokenizer=tokenizer)

training_args = SFTConfig(
    output_dir="./model_logs_final",
    overwrite_output_dir=True,
    report_to="tensorboard",
    logging_dir="./model_logs_final/logs",
    logging_steps=10,
    
    num_train_epochs=3,
    
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,
    
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.01,
    max_grad_norm=1.0,
    optim="adamw_torch_fused",
    
    bf16=True,
    bf16_full_eval=True,
    tf32=True,
    
    gradient_checkpointing=False,
    
    do_eval=True,
    eval_strategy="steps",
    eval_steps=5000,
    save_strategy="steps",
    save_total_limit=2,
    load_best_model_at_end=True,
    
    seed=42,
    data_seed=42,
    
    dataloader_num_workers=8,
    dataloader_pin_memory=True,
    dataloader_prefetch_factor=2,
    
    max_length=3072,
    packing=False,
    completion_only_loss=False,
    dataset_text_field=None,
    dataset_kwargs={
        "add_special_tokens": False,
        "append_concat_token": False,
    },
)

#%%
# ============ CREATE TRAINER ============
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=train_dataset_formatted,
    eval_dataset=eval_dataset_formatted,
    data_collator=data_collator,
)

print("✅ Trainer created")

#%%
# ============ MEMORY TEST ============
print("\n" + "="*70)
print("TESTING MEMORY WITH BATCH_SIZE=16")
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
    
    if peak < 70:
        print(f"\n✅ EXCELLENT! Can increase batch_size to 20+")
    elif peak < 85:
        print(f"\n✅ GOOD! batch_size=16 works well")
    else:
        print(f"\n⚠️  batch_size=16 at limit, stay here")
    
    model.zero_grad()
    print("\n✅ Memory test PASSED!")
    
except RuntimeError as e:
    if "out of memory" in str(e):
        print(f"\n❌ OOM with batch_size=16, reduce to 12")
    else:
        raise e

print("="*70)

#%%
# ============ START TRAINING ============
# Only proceed if memory test passed!

if torch.__version__ >= "2.0.0":
    print("\n🚀 Compiling model...")
    trainer.model = torch.compile(trainer.model, mode="reduce-overhead")

from transformers import TrainerCallback
import time

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
                seq_len = args.max_seq_length
                tokens_per_sec = (batch_size * seq_len * (step - self.last_step)) / elapsed
        
        self.last_time = now
        self.last_step = step
        
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

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

print("\n" + "="*70)
print("🚀 STARTING TRAINING")
print("="*70)
print(f"Batch size: {training_args.per_device_train_batch_size}")
print(f"Gradient checkpointing: {training_args.gradient_checkpointing}")
print("="*70 + "\n")

trainer.train()