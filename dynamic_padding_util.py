from dataclasses import dataclass
from typing import Any, Dict, List
import torch

@dataclass
class DataCollatorForDynamicPadding:
    """
    Data collator that:
    1. Pads dynamically to the longest sequence in each batch
    2. Preserves your labels (which already equal input_ids - no masking)
    3. Pads labels with -100 for padding tokens only
    """
    tokenizer: Any
    
    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        # Find max length in this batch
        max_length = max(len(f["input_ids"]) for f in features)
        
        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []
        
        for feature in features:
            input_ids = feature["input_ids"]
            labels = feature["labels"]
            
            # Calculate padding needed for this example
            padding_length = max_length - len(input_ids)
            
            # Pad input_ids with pad_token_id
            padded_input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            
            # Pad labels with -100 (ignore_index for padding tokens only)
            padded_labels = labels + [-100] * padding_length
            
            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = [1] * len(input_ids) + [0] * padding_length
            
            batch_input_ids.append(padded_input_ids)
            batch_labels.append(padded_labels)
            batch_attention_mask.append(attention_mask)
        
        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
        }