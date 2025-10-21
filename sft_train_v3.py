
#%%
# ---------------------- 0 Load dataset and tokenizer----------------------
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from datasets import load_from_disk
from torch.utils.data import DataLoader
import torch
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from dataclasses import dataclass
from typing import List, Dict
# import collator_bucket 
# from collator_bucket import BucketSampler, DataCollatorForPadding

#%%
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
# ----------------------1️⃣ Load dataset and tokenizer----------------------

train_dataset = load_from_disk("./ultrachat_train_formatted").select(range(1000))
eval_dataset = load_from_disk("./ultrachat_eval_formatted").select(range(500))


tokenizer = AutoTokenizer.from_pretrained("tokenizer_aae1", use_fast=True)
print(train_dataset.column_names)
print(f"Padding token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
print(f"Special tokens: {tokenizer.additional_special_tokens}")
print(f"Tokenizer vocabulary size: {len(tokenizer)}")
print(f"Special tokens added: {tokenizer.all_special_tokens}")


# %%
model_name = "model_aae1"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map='auto',
    trust_remote_code=True,
    # attn_implementation="flash_attention_2",
    attn_implementation="sdpa",  # Use PyTorch's scaled_dot_product_attention
    dtype=torch.bfloat16,          # Full bf16 precision
)

# Resize embeddings to accommodate new tokens
# model.resize_token_embeddings(len(tokenizer))

# Enable gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()

print(f"Model dtype: {model.dtype}")
    
# %%
output_dir = 'model_logs'

training_args = SFTConfig(
    fp16=True, # specify bf16=True instead when training on GPUs that support bf16
    do_eval=True,
    eval_strategy="epoch",
    gradient_accumulation_steps=128,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    learning_rate=2.0e-05,
    log_level="info",
    logging_steps=5,
    logging_strategy="steps",
    lr_scheduler_type="cosine",
    max_steps=-1,
    num_train_epochs=1,
    output_dir=output_dir,
    overwrite_output_dir=True,
    per_device_eval_batch_size=1, # originally set to 8
    per_device_train_batch_size=1, # originally set to 8
    # push_to_hub=True,
    # hub_model_id="zephyr-7b-sft-lora",
    # hub_strategy="every_save",
    # report_to="tensorboard",
    dataset_text_field="text",
    save_strategy="no",
    save_total_limit=None,
    seed=42,
    max_length=None,
    packing=False,
)
# %%
# based on config
peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
# %%
trainer = SFTTrainer(
        model=model_name,
        # model_init_kwargs=model_kwargs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        
    )
# %%
