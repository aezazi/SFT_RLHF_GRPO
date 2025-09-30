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
