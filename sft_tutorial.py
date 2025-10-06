#%%
import torch
from datasets import load_dataset

# %%
# based on config
raw_datasets = load_dataset("HuggingFaceH4/ultrachat_200k")

# %%
import pprint
print(type(raw_datasets))
print(raw_datasets.keys())
print(type(raw_datasets['train_sft']), len(raw_datasets['train_sft']))
pprint.pprint(raw_datasets['train_sft']['messages'][0])
# print(raw_datasets['train_sft']['messages'])

# %%
"""
The DatasetDict in Hugging Face's datasets library is a container for multiple Dataset objects, typically representing different splits of a dataset like "train," "validation," and "test." It acts like a Python dictionary where keys are the names of the splits (e.g., "train", "validation", "test") and values are Dataset objects corresponding to those splits. 

The way the tutorial is setting this up is that the entire training data is being placed as a value in DatasetDict container with key 'train'. Likewise for the test data. To this, we have to first extract the train and test data from  raw_datasets into a regular Python dictionary and then convert back to a DatasetDict object

Note that select is a method of the Dataset class
"""
from datasets import DatasetDict
indices = range(0,3)


# take the raw_dateset download and place the train and test data into a Python dictionary and pass that to the the DatasetDict object constructor as the argument. 
dataset_dict = DatasetDict(
                {'train': raw_datasets['train_sft'].select(indices),
                'test': raw_datasets['test_sft'].select(indices)}
)

dataset_dict = DatasetDict(dataset_dict)

print(f'type of dataset_dict : {type(dataset_dict)}')
print(f'type of dataset_dict["train"] : {type(dataset_dict['train'])} length: {len(dataset_dict['train'])}')
print(f'dataset_dict keys : {dataset_dict.keys()}')


# %%
import pprint

"""
each item in 'train' contains a multi-turn conversation. each item is a datasets dictionary-like object with keys 'prompt', 'prompt_id', 'messages'. We are interested in the 'messages'

each message is a list of dictionaries. Each dictionary is a turn in a conversation. The keys of each dictionary are 'content' and 'role'.
"""
ex_num = 1

example = dataset_dict['train'][ex_num]
print(example.keys(), '\n')

example_message = example['messages']

print(f'example_message {ex_num} type: {type(example_message)} length: {len(example_message)} \n')
pprint.pprint(example)


#%%
# this is the first and second turn
print(f'example_msg {ex_num} turn 0 type: {type(example_message[0])}   keys: {example_message[0].keys()}\n {example_message[0]} \n')

print(f'example_msg {ex_num} turn 1 type: {type(example_message[1])}   keys: {example_message[1].keys()}\n {example_message[0]} \n')

# %%
for example in dataset_dict['train']:
    messages = example['messages']
    print(f'turns in this conversation: {len(messages)}')


#%%
# Claude generated code. The HuggingFaceH4/ultrachat_200k dataset  does not include a "system" role. the code below inserts a system role at the begining of each example. As best as I could determine, models like GPT, Claude etc. use a very simple, fixed system message (if at all) during base SFT training.  System message flexibility is added later through RLHF/post-training


def add_system_message(example):
    messages = example['messages']
    if messages[0]['role'] != 'system':
        system_msg = {"role": "system", "content": "You are a helpful assistant."}
        messages = [system_msg] + messages
    return {'messages': messages}

dataset_dict_with_sys_role = dataset_dict['train'].map(add_system_message)
print("Original first example:")
print(dataset_dict['train']['messages'][0])  

print(f'\ntype of dataset_dict_with_sys_role: {type(dataset_dict_with_sys_role)}')
print(f'dataset_dict_with_sys_role features: {dataset_dict_with_sys_role.features}')


print("\nWith system message:")

print(dataset_dict_with_sys_role['messages'][0])

#%%

# Access to the the mistral model requires that you log into huggingface and generate an access token. Then the access token must be provided as per the code below

from transformers import AutoTokenizer
from huggingface_hub import login
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

hf_token = os.getenv("hf_access_token")
print(f"Your API Key: {hf_token}")
login(token=hf_token)

#%%
model = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model)
#check if model tokenizer has eos and/or pad tokens
print(tokenizer.bos_token_id)
print(tokenizer.eos_token_id)
print(tokenizer.pad_token_id)


#%%
# add special tokens to tokenizer as suggested by Claude. After some research, I decided to use the following as suggested by Claude. I also decided to use a dedicated pad token instead of using the eot token for padding as is sometimes done. I find these tokens to be a lot more readable and easy to follow when testing ad debugging
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
num_added = tokenizer.add_special_tokens(special_tokens_dict)

print(f"Added {num_added} special tokens to tokenizer")
print(f"Padding token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
print(f"Special tokens: {tokenizer.additional_special_tokens}")

print(f" <|im_start|> token: {tokenizer.additional_special_tokens}")

# Set padding side to right (standard for dedicated pad token)
tokenizer.padding_side = 'right'


#%%
# test the tokenizer
test_text = ['these are very dark and awful days.  I fear things will get much worse.', 'Who knows what will happen']

tokenizer.add_bos_token = False
tokenizer.add_eos_token = False

tokenized = tokenizer(test_text)
print(type(tokenized))
print(tokenized)
tokenizer.decode(tokenized['input_ids'][0])
# print(tokenized['input_ids'][0].decode())

#%%
# Create a chat template

# disable this model's built-in begining of sequence and end of sequence tokens since we created our own custom tokens above
tokenizer.add_bos_token = False
tokenizer.add_eos_token = False

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
# test chat template formatter

# messages = [
#     {"role": "system", "content": "You are helpful."},
#     {"role": "user", "content": "What is 2+2?"},
#     {"role": "assistant", "content": "4"},
#     {"role": "user", "content": "What about 3+3?"}
# ]

messages = dataset_dict_with_sys_role['messages'][0]

# Training scenario (add_generation_prompt=False)
formatted_train = tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=False
)
print("TRAINING FORMAT:")
print(formatted_train)
print()

# Inference scenario (add_generation_prompt=True)
formatted_inference = tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
)
print("INFERENCE FORMAT:")
print(formatted_inference)

#%%
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal language modeling, not masked
)

# %%
column_names = list(dataset_dict["train"].features)
column_names
# %%
f = dataset_dict['train'].map(formatter)
f
# %%
