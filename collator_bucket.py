#%%
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset, Sampler
from datasets import load_from_disk
import torch
from dataclasses import dataclass
from typing import List, Dict
import math

#%%
# Load dataset and tokenizer
dataset = load_from_disk("./ultrachat_train_formatted_tutorial")
test_dataset = dataset.select(range(6))  # pick first 6 examples for testing
tokenizer = AutoTokenizer.from_pretrained("tokenizer_aae1")

#%%
# Data collator (same as before)
@dataclass
class DataCollatorForCompletionOnlyLM:
    tokenizer: AutoTokenizer
    pad_to_multiple_of: int = 8
    label_pad_token_id: int = -100

    def __call__(self, batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        input_ids = [b["input_ids"] for b in batch]
        attention_masks = [b["attention_mask"] for b in batch]
        labels = [b["labels"] for b in batch]

        max_length = max(len(ids) for ids in input_ids)
        if self.pad_to_multiple_of is not None and max_length % self.pad_to_multiple_of != 0:
            max_length = ((max_length // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of

        batch_padded = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_masks},
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )

        padded_labels = [
            l + [self.label_pad_token_id] * (max_length - len(l)) for l in labels
        ]
        batch_padded["labels"] = torch.tensor(padded_labels, dtype=torch.long)

        return batch_padded

collator = DataCollatorForCompletionOnlyLM(tokenizer)

#%%
# --- Bucketing Sampler ---
class BucketSampler(Sampler):
    """
    Groups examples of similar lengths into the same batch to reduce padding.
    """
    def __init__(self, data_source: Dataset, batch_size: int):
        self.data_source = data_source
        self.batch_size = batch_size
        
        # Sort indices by sequence length (descending)
        self.sorted_indices = sorted(
            range(len(data_source)), 
            key=lambda i: len(data_source[i]["input_ids"]),
            reverse=True
        )

        # Break sorted indices into batches
        self.batches = [
            self.sorted_indices[i:i + batch_size] 
            for i in range(0, len(self.sorted_indices), batch_size)
        ]

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)

#%%
# Create DataLoader using BucketSampler
batch_size = 2
loader = DataLoader(
    test_dataset,
    batch_sampler=BucketSampler(test_dataset, batch_size=batch_size),
    collate_fn=collator
)

#%%
# Test bucketing: print batch shapes and first few tokens
for i, batch in enumerate(loader):
    print(f"\n=== Batch {i} ===")
    print("input_ids shape:", batch["input_ids"].shape)
    print("attention_mask shape:", batch["attention_mask"].shape)
    print("labels shape:", batch["labels"].shape)
    print("First example input_ids (first 50 tokens):")
    print(batch["input_ids"][0, :50])
    print("First example labels (first 50 tokens):")
    print(batch["labels"][0, :50])

# %%
print(tokenizer.decode(batch["input_ids"][0, :]))
# %%
