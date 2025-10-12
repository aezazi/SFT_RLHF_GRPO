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
# 1.  load and inspect the saved custom tokenizer
tokenizer = AutoTokenizer.from_pretrained("./tokenizer_with_specials")
tokenizer.pad_token

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
# Create Huggingface DataCollatorForCompletionOnlyLM 
"""
We use 
"""
from trl import DataCollatorForCompletionOnlyLM

response_template = "<|im_start|>assistant\n"

collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer,
)


#%%

#%%
# load the dataset
dataset_ultrachat = load_dataset("HuggingFaceH4/ultrachat_200k")

# Access the splits
train_dataset = dataset_ultrachat["train_sft"] 
eval_dataset = dataset_ultrachat["test_sft"]





#%%
# Create a chat template

# Code created by Claude. The chat template is based on the Jinja2 template syntax, which is what HuggingFace tokenizers expect for chat templates. A conversation is formatted into a single tokenizable sequence for a given model. https://huggingface.co/docs/transformers/en/chat_templating_writing

chat_template = """
{% if messages[0]['role'] == 'system' %}
    {{ '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}
{% endif %}

{% for message in messages %}
    {% if message['role'] != 'system' %}
        {{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}
    {% endif %}
{% endfor %}

{% if add_generation_prompt %}
    {{ '<|im_start|>assistant\n' }}
{% endif %}
"""

# set the tokenizers chat template to the template we created
tokenizer.chat_template = chat_template
# %%


# %%
# Save the resized model
output_dir = "./mistral-7b-resized"

# Save the model
model.save_pretrained(output_dir)