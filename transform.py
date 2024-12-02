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

def get_axis_vector(v, axis_index):
   k = torch.zeros_like(v)
   k[axis_index] = 1.0
   return k

def rotate_around_axis(v, k, theta):
   theta = torch.tensor(theta, dtype=v.dtype, device=v.device)
   v_parallel = k * torch.dot(k, v)
   v_perp = v - v_parallel
   return v_parallel + v_perp * torch.cos(theta) + v_perp * torch.sin(theta)

import numpy as np
import matplotlib.pyplot as plt

def plot_vector_decomposition(k, v):
   k = k.to(torch.float32).detach().cpu().numpy()
   v = v.to(torch.float32).detach().cpu().numpy()

   k = k / np.linalg.norm(k)
   v_parallel = k * np.dot(k, v)
   v_perp = v - v_parallel

   plt.figure(figsize=(8, 8))
   plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
   plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

   plt.quiver(0, 0, v_parallel[0], v_parallel[1], angles='xy', scale_units='xy', scale=1, color='r', label='v_parallel')
   plt.quiver(0, 0, v_perp[0], v_perp[1], angles='xy', scale_units='xy', scale=1, color='b', label='v_perp')
   plt.quiver(0, 0, k[0], k[1], angles='xy', scale_units='xy', scale=1, color='g', label='k')
   plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='purple', label='v')

   plt.xlim(-2, 2)
   plt.ylim(-2, 2)
   plt.grid(True)
   plt.legend()
   plt.axis('equal')
   plt.show()

def plot_array_elements(v):
   v = v.to(torch.float32).detach().cpu().numpy()
   
   plt.figure(figsize=(12, 6))
   plt.plot(range(len(v)), v, 'b-')
   plt.scatter(range(len(v)), v, c='blue')
   
   for i, val in enumerate(v):
       plt.annotate(str(i), (i, val), xytext=(5, 5), textcoords='offset points')
   
   plt.xlabel('Array Index')
   plt.ylabel('Value')
   plt.grid(True)
   plt.show()

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
        target_embedding_idx = 1
        N = 2 # number of embedding elements to rotate
        angle_of_rotation = torch.pi/50
        embedding_axes_of_rotation = torch.topk(abs(current_embeddings[0][target_embedding_idx]), k=N).indices.tolist()
        print(embedding_axes_of_rotation)
        custom_embedding = True
        for i in range(N):
            embedding_axis_of_rotation = embedding_axes_of_rotation[i]
            k = get_axis_vector(current_embeddings[0][target_embedding_idx], axis_index=embedding_axis_of_rotation) # axis of rotation
            if custom_embedding:
                # Rotate embedding around specified axis
                print(f'Axis of Rotation: {embedding_axis_of_rotation}')
                current_embeddings[0][target_embedding_idx] = rotate_around_axis(current_embeddings[0][target_embedding_idx], k, theta=angle_of_rotation) # rotation operation
        plot_vector_decomposition(k, current_embeddings[0][target_embedding_idx]) # plot new embedding elements along k_parallel and k_perp
        plot_array_elements(current_embeddings[0][target_embedding_idx]) # plot new embedding element values
        target_embedding = current_embeddings[0][target_embedding_idx]
        print(current_embeddings[0][target_embedding_idx])
    
    # Store all embeddings
    all_embeddings = [current_embeddings.detach().cpu().to(torch.float32).numpy()]
    
    # Store individual newly generated tokens
    new_tokens = []
    
    print("\nGeneration Progress:")
    print(f"Input: {tokenizer.decode(current_ids[0])}")
    print(f"Initial embedding shape: {current_embeddings.shape}")
    initial_embedding_values = current_embeddings[0, 0, :5].detach().cpu().to(torch.float32).numpy()
    print(f"First 5 values of initial embedding: {initial_embedding_values}")
    print("-" * 70)
    
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
        
        # Get output embeddings from the last hidden state
        output_embeddings = outputs.hidden_states[-1]
        
        # Get logits for next token
        next_token_logits = outputs.logits[:, -1, :]
        
        # Apply temperature
        next_token_logits = next_token_logits / temperature
        
        # Get probabilities
        probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
        
        # Sample next token
        next_token = torch.multinomial(probs, num_samples=1)

        next_token_id = next_token[0].item()
        
        # Store the new token
        new_tokens.append(next_token_id)
        
        # Print progress
        print(f"Step {i+1}:")
        print(f"New token: {next_token_id} -> '{tokenizer.decode([next_token_id])}'")
        print(f"Current text: {tokenizer.decode(current_ids[0])}▌")
        print(f"Output embedding shape: {output_embeddings.shape}")
        last_embedding_values = output_embeddings[0, -1, :5].detach().cpu().to(torch.float32).numpy()
        print(f"Last 5 values of output embedding: {last_embedding_values}")
        print("-" * 70)
        
        # Stop if we hit the EOS token
        if next_token_id == tokenizer.eos_token_id:
            break
            
        # Add new token to sequence
        current_ids = torch.cat([current_ids, next_token], dim=1)
        generated_ids[0].append(next_token_id)

        print(next_token_id)
        print(generated_ids)
        
        if i == 0:
            # modify current ids
            target_token_logits = next_token_logits = outputs.logits[:, target_embedding_idx, :]
            probs_target = torch.nn.functional.softmax(target_token_logits, dim=-1)
            target_token = torch.multinomial(probs_target, num_samples = 1)
            target_token_id = target_token[0].item()
            print(target_token_id)
            current_ids[0][target_embedding_idx] = target_token_id
            generated_ids[0][target_embedding_idx] = target_token_id

        
        # Update attention mask

        attention_mask = torch.ones_like(current_ids)
        
        # Get embeddings for the whole sequence
        with torch.no_grad():
            current_embeddings = model.get_input_embeddings()(current_ids)
            current_embeddings[0][target_embedding_idx] = target_embedding
        # Store embeddings
        all_embeddings.append(current_embeddings.detach().cpu().to(torch.float32).numpy())
    
    # Combine all generated tokens
    generated_sequence = current_ids[0].tolist()
    
    return {
        'input_text': messages,
        'output_text': tokenizer.decode(generated_sequence, skip_special_tokens=True),
        'input_token_ids': generated_ids[0],
        'output_token_ids': generated_sequence,
        'all_embeddings': all_embeddings,
        'new_tokens': new_tokens
    }

# Test message
messages = r"apple has the following meaning: "

# Generate and get results
results = generate_with_embeddings(messages, model, tokenizer)

# Print final results
print("\nFinal Results:")
print("Input Text:", results['input_text'])
print("Input Token IDs:", results['input_token_ids'])
print("Output Token IDs:", results['output_token_ids'])
print("Final Output:", results['output_text'])
print("Number of generation steps:", len(results['all_embeddings']))
print("Shape of final embeddings:", results['all_embeddings'][-1].shape)
