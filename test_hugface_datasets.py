#%%
import torch
from datasets import load_dataset

# %%
# This is a huggingface dictionary like object see comments below for details: 
dataset_ultrachat = load_dataset("HuggingFaceH4/ultrachat_200k")
# %%
# %%
column_names = dataset_ultrachat['train_sft'].column_names
column_names
# %%
f = dataset_ultrachat['train_sft'].map(formatter)
f
# %%
test = load_dataset("HuggingFaceH4/ultrachat_200k", split='train_sft')


# %%
print(type(test))
print(test.column_names)