#%%
# imports
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from datasets import load_dataset
from huggingface_hub import login
import os
from dotenv import load_dotenv

# %%
# load the huggingface ultrachat dataset
dataset_ultrachat = load_dataset("HuggingFaceH4/ultrachat_200k")

#%%
# Load the model
# Access to the the mistral model requires that you log into huggingface and generate an access token. Then the access token must be provided as per the code below

# Load environment variables from .env file
load_dotenv()

hf_token = os.getenv("hf_access_token")
print(f"Your API Key: {hf_token}")
login(token=hf_token)

#%%
# load the model and tokenizer
model_name = "mistralai/Mistral-7B-v0.1"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# %%
# this function checks, for each example, if the messages' first turn in the conversation includes a "system" role. If not, it creates a turn with a "system" role and content and prepends it the converstation
def add_system_message(example):
    """ 
    args: example. one example (row) from the train_sft split

    The HuggingFaceH4/ultrachat_200k dataset  does not include a "system" role. the code below inserts a system role at the begining of each example. As best as I could determine, models like GPT, Claude etc. use a very simple, fixed system message (if at all) during base SFT training.  System message flexibility is added later through RLHF/post-training. I used Claude ot generate some of this code.
    """
    messages = example['messages'] # extract the messages column from train_sft split
    
    if messages[0]['role'] != 'system':
        system_msg = {"role": "system", "content": "You are a helpful assistant."}
        messages = [system_msg] + messages
    
    return {'messages': messages}


# create train and eval datasets with system role
dataset_train = dataset_ultrachat["train_sft"].map(add_system_message)
dataset_eval = dataset_ultrachat["test_sft"].map(add_system_message)

print(type(dataset_train))

print(f"Train dataset size: {len(dataset_train)}")
print(f"Eval dataset size: {len(dataset_eval)}")

#%%
# Examine dataset_train 
print(f'\ntype of dataset_train: {type(dataset_train)}')
print(f'dataset_train features: {dataset_train.features}\n')

print("Original first example:")
print(dataset_ultrachat['train_sft']['messages'][0])  

print("\nWith system message:")
print(dataset_train['messages'][0])

#check if model tokenizer has eos and/or pad tokens
print(tokenizer.bos_token_id)
print(tokenizer.eos_token_id)
print(tokenizer.pad_token_id)

#%%
# add special tokens to the model tokenizer to facilitate chat template

"""
In order to perform SFT, we need to create a "chat template". A chat template basically adds special tokens to the conversations in our dataset to help the model learn a chat conversational structure. 

After some research, I decided to use a template structure suggested by Claude. I also decided to use a dedicated pad token instead of using the eot token for padding as is sometimes done. I find these tokens to be a lot more readable and easy to follow when testing and debugging. However, note that there are many approaches to creating chat templates. The main takeaways from my research were to be consistent and avoid designs that might cause the model to get confused as to the purpose of the token. This is why I decided to use a dedicated pad token instead of using the eot token as padding. I was never able to understand how models that do this avoid confusing a legitimate eot token with padding.

Here is an explanation of why the special tokens are created with the pad token getting it's own individual key while the other custom tokens are placed in a list with key "additional_special_token"

The Two Categories of Special Tokens
1. Standard Special Tokens (dedicated keys)
These have predefined roles across all Huggingface tokenizers (although tokenizers may well use just a subset):

bos_token - Beginning of sequence (e.g., <s>)
eos_token - End of sequence (e.g., </s>)
pad_token - Padding token
unk_token - Unknown token
sep_token - Separator token (used in some models like BERT)
cls_token - Classification token (used in some models like BERT)
mask_token - Mask token (for masked language modeling)

These have specific behaviors built into the tokenizer. For example:

pad_token is automatically used when you pad sequences
eos_token might be used to signal when generation should stop

2. Additional Special Tokens (list)
These are custom tokens you want to add that don't fit the predefined roles:

additional_special_tokens - A list of any custom special tokens you want

These tokens are treated as special (won't be split during tokenization) but don't have automatic behavior.

Why pad_token gets its own key:

The tokenizer needs to know: "When I pad, use THIS token"
When you call tokenizer.pad(), it automatically uses tokenizer.pad_token
It has functional significance beyond just being "special"

Why the others go in additional_special_tokens:

They mark structure in your chat format
But the tokenizer doesn't need to automatically use them for anything
You manually insert them via your chat template

The key insight: dedicated keys give tokens automatic behavior, additional_special_tokens just marks them as "don't split these during tokenization". For chat formatting, you usually want full manual control, so additional_special_tokens is the right choice for <|im_start|> and <|im_end|>.

FYI:
These attributes exist on ALL HuggingFace tokenizers
tokenizer.bos_token
tokenizer.eos_token
tokenizer.pad_token
tokenizer.unk_token
tokenizer.sep_token
tokenizer.cls_token
tokenizer.mask_token

# And their IDs
tokenizer.bos_token_id
tokenizer.eos_token_id
# etc.
"""
    
# Define all special tokens including dedicated padding token
special_tokens_dict = {
    "pad_token": "<|pad|>",
    "additional_special_tokens": [
        "<|im_start|>",
        "<|im_end|>",
        "<|endoftext|>"
    ]
}

tokenizer.add_special_tokens(special_tokens_dict)
tokenizer.padding_side = 'right' # Set padding side to right (standard for dedicated pad token)

# Assign my custom tokens as the tokenizer's eos and bos tokens. Disable automatic insertion of eos and bos tokens since our custom chat template does this. Resize the model's embedding space to accommodate the tokens we added:
# 1. Define which tokens serve as BOS/EOS (needed for model.generate)
tokenizer.bos_token = "<|im_start|>"
tokenizer.eos_token = "<|endoftext|>"

# 2. Assign those IDs in the model config
model.config.bos_token_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
model.config.eos_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
model.config.pad_token_id = tokenizer.convert_tokens_to_ids("<|pad|>")

# 3. Prevent automatic BOS/EOS insertion
tokenizer.add_bos_token = False
tokenizer.add_eos_token = False

model.resize_token_embeddings(len(tokenizer))

# inspect special tokens
print(f"Padding token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
print(f"Special tokens: {tokenizer.additional_special_tokens}")
print(f" <|im_start|> token: {tokenizer.additional_special_tokens[0]}")
print(f"Tokenizer vocabulary size: {len(tokenizer)}")
print(f"Special tokens added: {tokenizer.all_special_tokens}")


#%%
# Define ChatML template with non-assistant masking

chat_template = """{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}{% elif message['role'] == 'user' %}{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}{% elif message['role'] == 'assistant' %}{{ '<|im_start|>assistant\n' }}{% generation %}{{ message['content'] }}{% endgeneration %}{{ '<|im_end|>\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"""

tokenizer.chat_template = chat_template

#%%
# Verify it works
test_messages = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"}
]

result = tokenizer.apply_chat_template(
    test_messages,
    tokenize=True,
    return_assistant_tokens_mask=True,
    return_dict=True,
    add_generation_prompt=False
)

print(f"Result type: {type(result)}")
if 'assistant_masks' in result:
    print("✓ Chat template supports assistant_only_loss!")
    print(f"Input IDs: {result['input_ids']}")
    print(f"Assistant mask: {result['assistant_masks']}")
    print(f"\nMask breakdown:")
    tokens = tokenizer.convert_ids_to_tokens(result['input_ids'])
    for token, mask in zip(tokens, result['assistant_masks']):
        indicator = "→ TRAIN" if mask == 1 else ""
        print(f"  {token:20s} mask={mask} {indicator}")
else:
    print("✗ Chat template does NOT support assistant_only_loss")



#%%
# Create function apply chat template to datasets

def formatting_func(example):
    result = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        add_generation_prompt=False
    )

    input_ids = result["input_ids"]
    assistant_mask = result["assistant_masks"]
    labels = [tok if mask else -100 for tok, mask in zip(input_ids, assistant_mask)]

    return {
        "input_ids": input_ids,
        "labels": labels,
    }

#%%
# test the formatting function

# Mock conversation
example = {
    "messages": [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Hi, can you tell me a joke?"},
        {"role": "assistant", "content": "Sure! Why did the math book look sad? Because it had too many problems."},
        {"role": "user", "content": "Haha, another one please!"},
        {"role": "assistant", "content": "What do you call a fake noodle? An impasta!"},
    ]
}


# appply formatting function to example
tokenized = formatting_func(example)
input_ids = tokenized["input_ids"]
labels = tokenized["labels"]

# Decode and visualize which tokens are masked/unmasked
decoded_text = tokenizer.decode(input_ids)
decoded_labels = "".join(
    [tokenizer.decode([t]) if l != -100 else "█" for t, l in zip(input_ids, labels)]
)

print("\n=== Full Tokenized Conversation ===")
print(decoded_text)

print("\n=== Mask Visualization (█ = masked, assistant text = visible) ===")
print(decoded_labels)

# Verify correctness (basic assertion)
assert any(l != -100 for l in labels), "No assistant tokens unmasked!"
assert sum(l != -100 for l in labels) < len(labels), "All tokens unmasked — masking failed!"

print("\n✅ Masking logic works correctly!")

#%%
# Apply formatting function to train and eval datasets
print("Processing train dataset...")
dataset_train_formatted = dataset_train.map(
    formatting_func,
    batched=False,        # True if your formatting_func can handle batches
    remove_columns=dataset_train.column_names  # Keep only tokenized data
)

print("Processing eval dataset...")
dataset_eval_formatted = dataset_eval.map(
    formatting_func,
    batched=False,
    remove_columns=dataset_eval.column_names
)

dataset_train_formatted.save_to_disk("./ultrachat_train_formatted")
dataset_eval_formatted.save_to_disk("./ultrachat_eval_formatted")

#%%
# save the customized model and tokenizer so we don't have to redo all the above if we want to re-use our custom token and chat template
save_path_tok = "./tokenizer_with_specials"
save_path_mod = "./model_with_specials"
tokenizer.save_pretrained(save_path_tok)
model.save_pretrained(save_path_mod)

# %%
# test load the saved model
test_load = AutoModelForCausalLM.from_pretrained("./model_with_specials")
# %%
from datasets import load_from_disk
test_load_train_dataset_formatted = load_from_disk("./ultrachat_train_formatted")
import numpy as np

test_load_train_dataset_formatted.column_names


#%%
lengths = [len(example['input_ids']) for example in test_load_train_dataset_formatted]
print(f"Mean length: {np.mean(lengths):.0f}")
print(f"Median length: {np.median(lengths):.0f}")
print(f"95th percentile: {np.percentile(lengths, 95):.0f}")
print(f"99th percentile: {np.percentile(lengths, 99):.0f}")
print(f"Max length: {np.max(lengths)}")
print(f"% over 2048: {100 * np.mean(np.array(lengths) > 2048):.1f}%")
# %%
