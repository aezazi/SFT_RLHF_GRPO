#%%
"""
Example of how padding and loss masking works during training
"""

import torch
import torch.nn.functional as F

# Simulated example
print("="*60)
print("EXAMPLE: Why we need -100 for padded positions")
print("="*60)

# Two sequences of different lengths
seq1_tokens = [100, 200, 300, 400, 500]  # length 5
seq2_tokens = [100, 200, 300]            # length 3

# Pad to same length (6) using pad_token_id = 0
PAD_TOKEN_ID = 0
max_length = 6

seq1_padded = seq1_tokens + [PAD_TOKEN_ID] * (max_length - len(seq1_tokens))
seq2_padded = seq2_tokens + [PAD_TOKEN_ID] * (max_length - len(seq2_tokens))

print("\nSequence 1 (original length 5):", seq1_padded)
print("Sequence 2 (original length 3):", seq2_padded)

#%%

# Create labels (shifted by 1 for next-token prediction)
# For language modeling: input[:-1] predicts labels[1:]
seq1_input = seq1_padded[:-1]   # [100, 200, 300, 400, 500]
seq1_labels = seq1_padded[1:]   # [200, 300, 400, 500, PAD]

seq2_input = seq2_padded[:-1]   # [100, 200, 300, PAD, PAD]
seq2_labels = seq2_padded[1:]   # [200, 300, PAD, PAD, PAD]

print("\n" + "="*60)
print("WITHOUT proper masking (BAD):")
print("="*60)
print("Seq1 - Input: ", seq1_input)
print("Seq1 - Labels:", seq1_labels)
print("       Model learns: 100→200, 200→300, 300→400, 400→500, 500→PAD")
print("       Problem: Learning to predict PAD token! ❌")

print("\nSeq2 - Input: ", seq2_input)
print("Seq2 - Labels:", seq2_labels)
print("       Model learns: 100→200, 200→300, 300→PAD, PAD→PAD, PAD→PAD")
print("       Problem: Learning PAD→PAD patterns! ❌")


#%%
# Now with proper masking
seq1_labels_masked = seq1_labels.copy()
seq1_labels_masked[-1] = -100  # Mask the padded position

seq2_labels_masked = seq2_labels.copy()
seq2_labels_masked[-3:] = [-100, -100, -100]  # Mask all padded positions

print("\n" + "="*60)
print("WITH proper masking (GOOD):")
print("="*60)
print("Seq1 - Input: ", seq1_input)
print("Seq1 - Labels:", seq1_labels_masked)
print("       Model learns: 100→200, 200→300, 300→400, 400→500, [IGNORED]")
print("       ✓ No loss computed on padded position")

print("\nSeq2 - Input: ", seq2_input)
print("Seq2 - Labels:", seq2_labels_masked)
print("       Model learns: 100→200, 200→300, [IGNORED], [IGNORED], [IGNORED]")
print("       ✓ No loss computed on any padded positions")


#%%
# Demonstrate with actual PyTorch loss calculation
print("\n" + "="*60)
print("PYTORCH LOSS CALCULATION DEMO:")
print("="*60)

# Simulated model predictions (logits)
vocab_size = 1000
batch_size = 2
seq_length = 5

# Random logits (what model predicts)
logits = torch.randn(batch_size, seq_length, vocab_size)

# Labels without masking
labels_bad = torch.tensor([
    [200, 300, 400, 500, 0],      # seq1: last token is PAD (0)
    [200, 300, 0, 0, 0]            # seq2: last 3 tokens are PAD (0)
])

# Labels with masking
labels_good = torch.tensor([
    [200, 300, 400, 500, -100],    # seq1: masked pad position
    [200, 300, -100, -100, -100]   # seq2: masked pad positions
])

# Compute loss (CrossEntropyLoss ignores -100 by default)
loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

# Reshape for loss calculation
logits_flat = logits.view(-1, vocab_size)
labels_bad_flat = labels_bad.view(-1)
labels_good_flat = labels_good.view(-1)

loss_bad = loss_fn(logits_flat, labels_bad_flat)
loss_good = loss_fn(logits_flat, labels_good_flat)

print(f"\nLoss WITHOUT masking: {loss_bad.item():.4f}")
print(f"  (computed over all {batch_size * seq_length} positions)")

# Count non-masked positions
non_masked = (labels_good_flat != -100).sum().item()
print(f"\nLoss WITH masking: {loss_good.item():.4f}")
print(f"  (computed over only {non_masked} non-padded positions)")

print("\n" + "="*60)
print("KEY TAKEAWAY:")
print("="*60)
print("✓ Set padded positions to -100 in labels")
print("✓ PyTorch CrossEntropyLoss automatically ignores -100")
print("✓ Model only learns from real tokens, not padding")
print("✓ Prevents model from learning to generate padding tokens")
print("="*60)

# Show what data collator does
print("\n" + "="*60)
print("WHAT DATA COLLATOR DOES AUTOMATICALLY:")
print("="*60)
print("""
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # We're doing causal LM, not masked LM
)

# When you pass sequences to the collator, it:
# 1. Pads sequences to the same length using tokenizer.pad_token_id
# 2. Creates labels by shifting input_ids by 1
# 3. Sets all padded positions in labels to -100
# 4. Returns a batch ready for training

# You don't have to manually set -100!
# The collator handles it automatically.
""")
