import os
import json
from safetensors import safe_open
import torch

def find_tensor_dir():
    """Find the tensor directory containing the model weights"""
    base_path = "model_cache/llama-3.1-8b-instruct/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/"
    
    try:
        entries = os.listdir(base_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Base path not found: {base_path}")
    
    directories = [entry for entry in entries if os.path.isdir(os.path.join(base_path, entry))]
    
    if not directories:
        raise ValueError(f"No directories found in {base_path}")
    
    return os.path.join(base_path, directories[0])

def load_tensor_mapping(tensor_dir):
    """Load the weight map from model.safetensors.index.json"""
    index_path = os.path.join(tensor_dir, "model.safetensors.index.json")
    try:
        with open(index_path, 'r') as f:
            data = json.load(f)
        return data["weight_map"]
    except Exception as e:
        raise Exception(f"Error loading tensor mapping: {str(e)}")

def load_tensor(tensor_dir, tensor_name="model.norm.weight"):
    """Load a tensor using the weight mapping"""
    # Load the weight mapping
    weight_map = load_tensor_mapping(tensor_dir)
    
    if tensor_name not in weight_map:
        raise KeyError(f"Tensor {tensor_name} not found in weight map")
    
    # Get the file containing this tensor
    file_name = weight_map[tensor_name]
    file_path = os.path.join(tensor_dir, file_name)
    
    try:
        with safe_open(file_path, framework="torch") as f:  # Changed to use torch framework
            try:
                # Get tensor data
                tensor = f.get_tensor(tensor_name)
                
                # Convert to float32 if it's bfloat16
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.to(torch.float32)
                
                return {
                    'file': file_name,
                    'tensor_name': tensor_name,
                    'shape': tensor.shape,
                    'dtype': tensor.dtype,
                    'sample': tensor.flatten()[:5].tolist(),
                    'stats': {
                        'mean': tensor.mean().item(),
                        'std': tensor.std().item(),
                        'min': tensor.min().item(),
                        'max': tensor.max().item()
                    }
                }
            except Exception as e:
                print(f"Error processing tensor: {str(e)}")
                raise
    except Exception as e:
        print(f"Error loading file {file_name}: {str(e)}")
        raise

def main():
    try:
        # Find the tensor directory
        tensor_dir = find_tensor_dir()
        print(f"Found tensor directory: {tensor_dir}")
        
        # List all available tensor names from the mapping
        weight_map = load_tensor_mapping(tensor_dir)
        print("\nAvailable tensors:")
        for i, tensor_name in enumerate(sorted(weight_map.keys())[:5]):  # Show first 5 tensors
            print(f"{i+1}. {tensor_name}")
        print("...")
        
        # Load and print the weight
        result = load_tensor(tensor_dir)
        
        print("\nWeight Information:")
        print(f"File: {result['file']}")
        print(f"Tensor: {result['tensor_name']}")
        print(f"Shape: {result['shape']}")
        print(f"Dtype: {result['dtype']}")
        print(f"First 5 values: {result['sample']}")
        print("\nStatistics:")
        print(f"Mean: {result['stats']['mean']:.6f}")
        print(f"Std Dev: {result['stats']['std']:.6f}")
        print(f"Min: {result['stats']['min']:.6f}")
        print(f"Max: {result['stats']['max']:.6f}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
