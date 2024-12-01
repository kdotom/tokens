import transformers
import torch
import os
import numpy as np

# Set the model ID and token
model_id = "meta-llama/Llama-3.2-1B"
token = os.getenv('HF_TOKEN')

# Initialize the tokenizer with padding
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, token=token)
tokenizer.pad_token = tokenizer.eos_token

# Load the model directly
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token=token,
    pad_token_id=tokenizer.pad_token_id
)

def generate_with_embeddings(messages, model, tokenizer, max_new_tokens=50, temperature=0.7):
    """Generate text autoregressively using embeddings"""
    
    # Initial tokenization
    inputs = tokenizer(
        messages,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True
    )
    current_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    
    # Store generated token ids
    generated_ids = [current_ids[0].tolist()]
    
    # Get initial embeddings
    with torch.no_grad():
        current_embeddings = model.get_input_embeddings()(current_ids)
    
    # Store all embeddings
    all_embeddings = [current_embeddings.detach().cpu().to(torch.float32).numpy()]
    
    # Store individual newly generated tokens
    new_tokens = []
    
    # Generate tokens one at a time
    for i in range(max_new_tokens):
        # Get model output for current sequence
        with torch.no_grad():
            outputs = model(
                inputs_embeds=current_embeddings,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True
            )
        
        # Get logits for next token
        next_token_logits = outputs.logits[:, -1, :]
        
        # Apply temperature
        next_token_logits = next_token_logits / temperature
        
        # Get probabilities
        probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
        
        # Sample next token
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Store the new token
        new_tokens.append(next_token[0].item())
        
        # Stop if we hit the EOS token
        if next_token[0].item() == tokenizer.eos_token_id:
            break
            
        # Add new token to sequence
        current_ids = torch.cat([current_ids, next_token], dim=1)
        generated_ids.append(next_token[0].item())
        
        # Update attention mask
        attention_mask = torch.ones_like(current_ids)
        
        # Get embeddings for the whole sequence
        with torch.no_grad():
            current_embeddings = model.get_input_embeddings()(current_ids)
        
        # Store embeddings
        all_embeddings.append(current_embeddings.detach().cpu().to(torch.float32).numpy())
    
    # Combine all generated tokens
    generated_sequence = current_ids[0].tolist()
    
    return {
        'input_text': messages,
        'output_text': tokenizer.decode(generated_sequence, skip_special_tokens=True),
        'input_tokens': tokenizer.convert_ids_to_tokens(generated_ids[0]),
        'output_tokens': tokenizer.convert_ids_to_tokens(generated_sequence),
        'all_embeddings': all_embeddings,  # List of embeddings at each generation step
        'generated_ids': generated_ids,
        'new_tokens': new_tokens  # Just the newly generated tokens
    }

# Test message
messages = "What's the weather like in spring?"

# Generate and get results
results = generate_with_embeddings(messages, model, tokenizer)

# Print results
print("Input Text:")
print(results['input_text'])

print("\nOutput Text:")
print(results['output_text'])

print("\nInput Token IDs:", generated_ids[0])

print("\nOutput Token IDs:", generated_sequence)

print("\nNumber of generation steps:", len(results['all_embeddings']))
print("Shape of final embeddings:", results['all_embeddings'][-1].shape)

# Print the generation progression
print("\nToken generation progression:")
for i, token_id in enumerate(results['new_tokens']):
    if i < 10:  # Print first 10 steps
        token_text = tokenizer.convert_ids_to_tokens(token_id)
        print(f"Step {i}: {token_id} -> {token_text}")
