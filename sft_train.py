#%%
from datasets import load_dataset
from trl import SFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from transformers import BitsAndBytesConfig
import torch

#%%
# load the model
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

# load the saved custom tokenizer
tokenizer = AutoTokenizer.from_pretrained("./tokenizer_with_specials")

# resize the vocab size of the model to match the modified tokenizer
model.resize_token_embeddings(len(tokenizer))

