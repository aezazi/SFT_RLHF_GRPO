#%%
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
import torch
#%%

dataset = load_dataset("HuggingFaceTB/smoltalk", 'all')
# %%
