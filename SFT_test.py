#%%
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

#%%

dataset = load_dataset("HuggingFaceTB/smoltalk", 'all')
# %%
device = "cuda" if torch.cuda.is_available() else "mps"
print(device)
# %%

# Configure model and tokenizer

# %%
from datasets import Dataset
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")

chat1 = [
    {"role": "user", "content": "Which is bigger, the moon or the sun?"},
    {"role": "assistant", "content": "The sun."}
]
chat2 = [
    {"role": "user", "content": "Which is bigger, a virus or a bacterium?"},
    {"role": "assistant", "content": "A bacterium."}
]

dataset = Dataset.from_dict({"chat": [chat1, chat2]})
dataset = dataset.map(lambda x: {"formatted_chat": tokenizer.apply_chat_template(x["chat"], tokenize=False, add_generation_prompt=False)})
print(dataset['formatted_chat'][0])

# %%
# These will use different templates automatically
# mistral_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
# qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat")
smol_tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "How can I help you"}
]

# Each will format according to its model's template
# mistral_chat = mistral_tokenizer.apply_chat_template(messages, tokenize=False)
# qwen_chat = qwen_tokenizer.apply_chat_template(messages, tokenize=False)
smol_chat = smol_tokenizer.apply_chat_template(messages, tokenize=False)

print(smol_chat)



# %%
#%%
# load the model and inspect whether bos, eos and padding tokens exist in the tokenizer for this model
model = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model)
#check if model tokenizer has eos and/or pad tokens
print(tokenizer.bos_token_id)
print(tokenizer.eos_token_id)
print(tokenizer.pad_token_id)

# add special tokens to the model tokenizer to facilitate chat template
def add_tokens(tokenizer=None, special_tokens=None):
    
    tokenizer.add_special_tokens(special_tokens)
    tokenizer.padding_side = 'right' # Set padding side to right (standard for dedicated pad token)
    # disable this model's automatically adding built-in begining of sequence and end of sequence tokens since we are creating our own custom tokens
    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False

    
# Define all special tokens including dedicated padding token
special_tokens_dict = {
    "pad_token": "<|pad|>",
    "additional_special_tokens": [
        "<|im_start|>",
        "<|im_end|>",
        "<|endoftext|>"
    ]
}

# Add special tokens to tokenizer
add_tokens(tokenizer= tokenizer, special_tokens=special_tokens_dict)

# inspect special tokens
print(f"Padding token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
print(f"Special tokens: {tokenizer.additional_special_tokens}")
print(f" <|im_start|> token: {tokenizer.additional_special_tokens[0]}")

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