"""
ChatML Template Formatter for UltraChat Dataset
Uses dedicated <|pad|> token for padding (recommended approach)
"""

def format_conversation_chatml(training_example, system_message="You are a helpful assistant."):
    """
    Format a conversation using ChatML template.
    
    Args:
        messages: List of dicts with 'role' and 'content' keys
        system_message: Optional system message (default: generic helpful assistant)
    
    Returns:
        Formatted string ready for tokenization
    """
    # Start with system message
    formatted = f"<|im_start|>system\n{system_message}<|im_end|>\n"

    # extract convesation messages from training_example
    messages = training_example['messages']
    
    # Add each message in the conversation
    for message in messages:
        role = message['role']
        content = message['content']
        formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    
    # Add end of text token (for training)
    formatted += "<|endoftext|>"
    
    return formatted


def format_conversation_for_training(messages, system_message="You are a helpful assistant."):
    """
    Format conversation and create labels for training (masking non-assistant turns).
    
    Returns:
        tuple: (formatted_text, labels_mask)
        labels_mask: List of tuples (start_idx, end_idx) for assistant responses
    """
    formatted = f"<|im_start|>system\n{system_message}<|im_end|>\n"
    assistant_segments = []
    
    for message in messages:
        role = message['role']
        content = message['content']
        
        # Track where assistant responses start and end
        if role == 'assistant':
            start_idx = len(formatted) + len(f"<|im_start|>{role}\n")
            formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
            end_idx = len(formatted) - len("<|im_end|>\n")
            assistant_segments.append((start_idx, end_idx))
        else:
            formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    
    formatted += "<|endoftext|>"
    
    return formatted, assistant_segments


def setup_tokenizer_with_chat_tokens(tokenizer):
    """
    Add chat template special tokens and dedicated padding token to tokenizer.
    
    Args:
        tokenizer: Your model's tokenizer
    
    Returns:
        tokenizer: Modified tokenizer with new tokens
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
    
    # Add special tokens to tokenizer
    num_added = tokenizer.add_special_tokens(special_tokens_dict)
    
    print(f"Added {num_added} special tokens to tokenizer")
    print(f"Padding token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"Special tokens: {tokenizer.additional_special_tokens}")
    
    # Set padding side to right (standard for dedicated pad token)
    tokenizer.padding_side = 'right'
    
    return tokenizer


def initialize_new_token_embeddings(model, tokenizer, old_vocab_size):
    """
    Resize model embeddings and initialize new token embeddings properly.
    
    Args:
        model: Your language model
        tokenizer: Tokenizer with new tokens added
        old_vocab_size: Vocabulary size before adding new tokens
    """
    import torch
    
    # Resize token embeddings
    model.resize_token_embeddings(len(tokenizer))
    
    # Initialize new token embeddings from average of existing embeddings
    # This is better than random initialization
    with torch.no_grad():
        # Get input embeddings
        input_embeddings = model.get_input_embeddings().weight
        
        # Calculate average of existing token embeddings
        avg_embedding = input_embeddings[:old_vocab_size].mean(dim=0)
        
        # Initialize new tokens with the average
        num_new_tokens = len(tokenizer) - old_vocab_size
        input_embeddings[old_vocab_size:] = avg_embedding.unsqueeze(0).repeat(num_new_tokens, 1)
        
        # Do the same for output embeddings if they exist and are not tied
        if hasattr(model, 'get_output_embeddings') and model.get_output_embeddings() is not None:
            output_embeddings = model.get_output_embeddings().weight
            if output_embeddings.data_ptr() != input_embeddings.data_ptr():
                # Output embeddings are not tied, initialize them too
                output_embeddings[old_vocab_size:] = avg_embedding.unsqueeze(0).repeat(num_new_tokens, 1)
    
    print(f"Initialized {num_new_tokens} new token embeddings")


# Example usage
if __name__ == "__main__":
    # Example UltraChat conversation
    ultrachat_example = [
        {'content': 'These instructions apply to section-based themes (Responsive 6.0+, Retina 4.0+, Parallax 3.0+ Turbo 2.0+, Mobilia 5.0+). What theme version am I using?\nOn your Collections pages & Featured Collections sections, you can easily show the secondary image of a product on hover by enabling one of the theme\'s built-in settings!\nYour Collection pages & Featured Collections sections will now display the secondary product image just by hovering over that product image thumbnail.\nDoes this feature apply to all sections of the theme or just specific ones as listed in the text material?',
         'role': 'user'},
        {'content': 'This feature only applies to Collection pages and Featured Collections sections of the section-based themes listed in the text material.',
         'role': 'assistant'},
        {'content': 'Can you guide me through the process of enabling the secondary image hover feature on my Collection pages and Featured Collections sections?',
         'role': 'user'},
        {'content': 'Sure, here are the steps to enable the secondary image hover feature on your Collection pages and Featured Collections sections:\n\n1. Log in to your Shopify account and go to your Online Store.\n2. Click on Customize theme for the section-based theme you are using.\n3. Navigate to the Collection pages or Featured Collections section where you want to enable the secondary image hover feature.\n4. With the section open, click on the section\'s setting (gear) icon in the top-left corner.\n5. In the settings panel that appears, look for an option labeled \'Image display\' or \'Image hover\'.\n6. If available, select \'Show secondary image on hover\'.\n7. Save the changes and preview the Collection/Featured Collection page to see the effect.\n\nIf you\'re having trouble finding the setting, the best thing to do is refer to your theme\'s documentation, since the location and labeling of settings can vary between themes.',
         'role': 'assistant'},
    ]
    
    # Format the conversation
    formatted_text = format_conversation_chatml(ultrachat_example)
    print("=== FORMATTED CONVERSATION ===")
    print(formatted_text)
    print("\n" + "="*50 + "\n")
    
    # Format for training with label masking
    formatted_with_labels, assistant_segments = format_conversation_for_training(ultrachat_example)
    print("=== ASSISTANT RESPONSE SEGMENTS (for loss calculation) ===")
    for i, (start, end) in enumerate(assistant_segments):
        print(f"Assistant response {i+1}: characters {start} to {end}")
        print(f"Content preview: {formatted_with_labels[start:min(start+100, end)]}...")
        print()
    
    print("\n" + "="*50)
    print("COMPLETE TRAINING SETUP EXAMPLE:")
    print("="*50)
    print("""
# Step 1: Setup tokenizer with chat tokens
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("your-pretrained-model")
model = AutoModelForCausalLM.from_pretrained("your-pretrained-model")

# Save old vocab size
old_vocab_size = len(tokenizer)

# Add special tokens (including dedicated <|pad|> token)
tokenizer = setup_tokenizer_with_chat_tokens(tokenizer)

# Initialize new token embeddings
initialize_new_token_embeddings(model, tokenizer, old_vocab_size)

# Step 2: Format your dataset
from datasets import load_dataset

dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

def format_example(example):
    formatted = format_conversation_chatml(example['messages'])
    return {"text": formatted}

formatted_dataset = dataset.map(format_example)

# Step 3: Setup data collator (handles padding automatically)
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal language modeling, not masked
)

# The data collator will:
# - Pad sequences using <|pad|> token (right padding)
# - Create labels by shifting input_ids
# - Set padded positions to -100 in labels
# - Handle everything automatically!

# Step 4: Train
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./ultrachat-sft",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    warmup_steps=100,
    logging_steps=10,
    save_steps=500,
    fp16=True,  # Use mixed precision if you have a compatible GPU
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=formatted_dataset,
    data_collator=data_collator,
)

trainer.train()
    """)
    
    print("\n" + "="*50)
    print("KEY BENEFITS OF DEDICATED <|pad|> TOKEN:")
    print("="*50)
    print("✓ No ambiguity between padding and end-of-sequence")
    print("✓ Can use standard right-side padding")
    print("✓ Cleaner separation of concerns")
    print("✓ <|endoftext|> retains its semantic meaning")
    print("✓ Easier to debug and understand")
    print("="*50)
