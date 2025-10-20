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
import random
# -------------------------
# create_length_buckets()
# -------------------------
def create_length_buckets(dataset, bucket_size: int = 100):
    """
    Group dataset indices into buckets of similar sequence lengths.

    Args:
      dataset: HuggingFace dataset (or list-like) where dataset[i]["input_ids"] exists.
      bucket_size: length-range for each bucket (e.g. 100 tokens).

    Returns:
      List[List[int]] : list of buckets, each bucket is a list of dataset indices.
    """
    if len(dataset) == 0:
        return []

    # compute lengths and sort indices by length
    lengths = [len(dataset[i]["input_ids"]) for i in range(len(dataset))]
    sorted_indices = sorted(range(len(dataset)), key=lambda i: lengths[i])

    buckets = []
    current_bucket = []
    current_min = lengths[sorted_indices[0]]

    for idx in sorted_indices:
        seq_len = lengths[idx]
        if seq_len > current_min + bucket_size and current_bucket:
            buckets.append(current_bucket)
            current_bucket = []
            current_min = seq_len
        current_bucket.append(idx)

    if current_bucket:
        buckets.append(current_bucket)

    return buckets


# -------------------------
# ShuffledBucketBatchSampler
# -------------------------
class ShuffledBucketBatchSampler(Sampler):
    """
    Samples batches of indices grouped by bucket. Each epoch:
      - bucket order is shuffled
      - indices within each bucket are shuffled
      - final list of batches is shuffled
    """
    def __init__(self, buckets: List[List[int]], batch_size:int, shuffle:bool=True, seed: int = None):
        self.buckets = buckets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.seed is not None:
            random.seed(self.seed)

        bucket_order = list(range(len(self.buckets)))
        if self.shuffle:
            random.shuffle(bucket_order)

        batch_indices = []
        for b_idx in bucket_order:
            bucket = list(self.buckets[b_idx])  # copy
            if self.shuffle:
                random.shuffle(bucket)
            # split bucket into batches of size batch_size
            for i in range(0, len(bucket), self.batch_size):
                batch_indices.append(bucket[i:i+self.batch_size])

        if self.shuffle:
            random.shuffle(batch_indices)

        for batch in batch_indices:
            yield batch

    def __len__(self):
        return sum(math.ceil(len(b) / self.batch_size) for b in self.buckets)


# -------------------------
# Example usage (assumes test_dataset and collator exist)
# -------------------------
# Parameters
batch_size = 2
bucket_size = 1000  # tune to your length distribution

# 1) Create buckets from your dataset (HuggingFace Dataset or list-like)
buckets = create_length_buckets(test_dataset, bucket_size=bucket_size)
print(f"Created {len(buckets)} buckets (example bucket sizes): {[len(b) for b in buckets[:10]]}")

# 2) Create sampler and DataLoader
sampler = ShuffledBucketBatchSampler(buckets=buckets, batch_size=batch_size, shuffle=True, seed=None)
loader = DataLoader(
    test_dataset,
    batch_sampler=sampler,
    collate_fn=collator
)

# 3) Quick sanity test: show first token-lengths and shapes for the first few batches
print("\n-- Epoch simulation: showing first 4 batches --")
for i, batch in enumerate(loader):
    if i >= 4:
        break
    seq_lens = batch["attention_mask"].sum(dim=1).tolist()  # approximate effective lengths
    print(f"Batch {i}: shapes {batch['input_ids'].shape}, seq_lens={seq_lens}")
    # print first tokens of first sequence (non-verbose)
    print("  first seq first token ids:", batch["input_ids"][0, :8].tolist())

# 4) Demonstrate shuffle across two epochs (recreate sampler to reshuffle)
print("\n-- Compare two epochs (first batch indices only) --")
sampler1 = ShuffledBucketBatchSampler(buckets=buckets, batch_size=batch_size, shuffle=True, seed=123)
sampler2 = ShuffledBucketBatchSampler(buckets=buckets, batch_size=batch_size, shuffle=True, seed=456)

first_epoch_batches = list(iter(sampler1))
second_epoch_batches = list(iter(sampler2))

print("First epoch, first 3 batches (index lists):", first_epoch_batches[:3])
print("Second epoch, first 3 batches (index lists):", second_epoch_batches[:3])

# %%
