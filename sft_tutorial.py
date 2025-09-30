#%%
import torch
from datasets import load_dataset
# %%
# based on config
raw_datasets = load_dataset("HuggingFaceH4/ultrachat_200k")
# %%
print(type(raw_datasets))
print(raw_datasets.keys())
print(type(raw_datasets['train_sft']), len(raw_datasets['train_sft']))
print(raw_datasets['train_sft'])
print(raw_datasets['train_sft']['messages'])

# %%
"""
The DatasetDict in Hugging Face's datasets library is a container for multiple Dataset objects, typically representing different splits of a dataset like "train," "validation," and "test." It acts like a Python dictionary where keys are the names of the splits (e.g., "train", "validation", "test") and values are Dataset objects corresponding to those splits. 

The way the tutorial is setting this up is that the entire training data is being placed as a value in DatasetDict container with key 'train'. Likewise for the test data.

Note thatselect is a method of the Dataset class
"""
from datasets import DatasetDict
indices = range(0,100)

dataset_dict = {'train': raw_datasets['train_sft'].select(indices),
                'test': raw_datasets['test_sft'].select(indices)
                }

raw_datasets_dict = DatasetDict(dataset_dict)
print(type(raw_datasets_dict))
print(f'type of "train" key: {type(raw_datasets_dict['train'])} length: {len(raw_datasets_dict['train'])}\n')


# %%
import pprint

"""
each item in 'train' contains a multi-turn conversation. each item is a datasets dictionary-like object with keys 'prompt', 'prompt_id', 'messages'. We are interestes in the 'messages'
"""
example = raw_datasets_dict['train'][0]
print(example.keys(), '\n')
pprint.pprint(example)

# %%
"""
each message is a list of dictionaries. Each dictionary is a turn in a conversation. The keys of each dictionary are 'content' and 'role'.
"""
example_messages = example['messages']

# there were 8 turns (back and forths) in this example
print(f'example_messages type: {type(example_messages)} length: {len(example_messages)}')

# this is the first turn
pprint.pprint(f'example_msg item 0 type: {type(example_messages[0])} \n {example_messages[1]}')

# %%
from transformers import AutoTokenizer
model = "mistralai/Mistral-7B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model)

print(tokenizer.eos_token_id)

# %%
