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


# %%
# load the templated dataset
from datasets import load_from_disk
dataset_templated = load_from_disk("./dataset_ultrachat_train_sft_templated")
# %%
print(dataset_templated.column_names)
print(dataset_templated['templated_msgs'][0])
# %%


# %%
# Save the resized model
output_dir = "./mistral-7b-resized"

# Save the model
model.save_pretrained(output_dir)