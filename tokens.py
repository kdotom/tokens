from dotenv import load_dotenv
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import nn
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
CACHE_DIR = "model_cache/llama-3.1-8b-instruct"
EMBEDDING_DIM = 2048

def setup_auth():
    """Use existing HF_TOKEN from environment"""
    token = os.getenv('HF_TOKEN')
    if not token:
        raise EnvironmentError("HF_TOKEN not found in environment variables")
    return token

class LlamaEmbeddingProcessor:
    def __init__(self):
        self.token = setup_auth()
        
        try:
            logger.info(f"Loading model {MODEL_NAME}...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME,
                use_fast=True,
                cache_dir=CACHE_DIR,
                token=self.token
            )
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16,  # Explicitly set dtype
                device_map="auto",
                cache_dir=CACHE_DIR,
                token=self.token
            )
            
            # Set pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.tokenizer.pad_token_id
                
            logger.info("Model and tokenizer loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading model or tokenizer: {e}")
            raise
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16  # Store dtype for consistency
        logger.info(f"Using device: {self.device}")

    def words_to_embeddings(self, text):
        """Convert text to token embeddings"""
        # Tokenize the text
        tokens = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        logger.info(f"\nInput text: {text}")
        logger.info(f"Tokens: {tokens.tolist()}")
        logger.info(f"Token words: {self.tokenizer.convert_ids_to_tokens(tokens[0])}")
        
        # Get embeddings from the model's embedding layer
        with torch.no_grad():
            embeddings = self.model.get_input_embeddings()(tokens).to(dtype=self.dtype)
        
        logger.info(f"Embedding shape: {embeddings.shape}")
        logger.info(f"Embedding dtype: {embeddings.dtype}")
        logger.info(f"Embedding sample (first token, first 5 dims): {embeddings[0, 0, :5].tolist()}")
        
        return embeddings, tokens

    def process_embeddings(self, embeddings, attention_mask=None):
        """Process embeddings through the model"""
        # Ensure correct dtype
        embeddings = embeddings.to(dtype=self.dtype)
        
        if attention_mask is None:
            attention_mask = torch.ones(
                embeddings.shape[0],
                embeddings.shape[1],
                dtype=torch.long,
                device=self.device
            )
        
        # Forward pass through transformer layers
        with torch.no_grad():
            outputs = self.model(
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            # Get logits and convert them back to embedding space
            logits = outputs.logits
            
            # Get the predicted token embeddings
            predicted_token_ids = torch.argmax(logits, dim=-1)
            output_embeddings = self.model.get_input_embeddings()(predicted_token_ids)
        
        logger.info(f"\nOutput embedding shape: {output_embeddings.shape}")
        logger.info(f"Output embedding dtype: {output_embeddings.dtype}")
        logger.info(f"Output embedding sample (first token, first 5 dims): {output_embeddings[0, 0, :5].tolist()}")
        
        return output_embeddings, logits

    def embeddings_to_words(self, logits):
        """Convert logits to words"""
        # Get the most likely token IDs
        predicted_token_ids = torch.argmax(logits, dim=-1)
        
        # Convert token IDs to words
        predicted_words = self.tokenizer.batch_decode(predicted_token_ids, skip_special_tokens=True)
        
        logger.info(f"\nPredicted token IDs shape: {predicted_token_ids.shape}")
        logger.info(f"Predicted words: {predicted_words}")
        
        return predicted_words, predicted_token_ids

def main():
    try:
        # Initialize processor
        processor = LlamaEmbeddingProcessor()
        
        # Example text
        input_text = "The first 10 digits of pi are "
        
        # Step 1: Convert words to embeddings
        logger.info("\n=== Step 1: Words to Embeddings ===")
        input_embeddings, input_tokens = processor.words_to_embeddings(input_text)
        
        # Step 2: Process embeddings through the model
        logger.info("\n=== Step 2: Process Embeddings ===")
        output_embeddings, logits = processor.process_embeddings(input_embeddings)
        
        # Step 3: Convert embeddings back to words
        logger.info("\n=== Step 3: Embeddings to Words ===")
        output_words, output_tokens = processor.embeddings_to_words(logits)
        
        # Print summary
        logger.info("\n=== Process Summary ===")
        logger.info(f"Input text: {input_text}")
        logger.info(f"Input tokens: {input_tokens.tolist()}")
        logger.info(f"Input embedding shape: {input_embeddings.shape}")
        logger.info(f"Output embedding shape: {output_embeddings.shape}")
        logger.info(f"Output tokens: {output_tokens.tolist()}")
        logger.info(f"Output text: {' '.join(output_words)}")
        
        # Example with custom embeddings
        logger.info("\n=== Custom Embedding Example ===")
        # Create custom embeddings with correct dtype
        custom_embeddings = torch.randn(
            1, 5, processor.model.config.hidden_size,
            device=processor.device,
            dtype=processor.dtype  # Use float16
        )
        logger.info(f"Custom embedding shape: {custom_embeddings.shape}")
        logger.info(f"Custom embedding dtype: {custom_embeddings.dtype}")
        
        output_embeddings, logits = processor.process_embeddings(custom_embeddings)
        output_words, _ = processor.embeddings_to_words(logits)
        
        logger.info(f"Generated text from custom embeddings: {' '.join(output_words)}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--clear-cache', action='store_true', help='Clear the model cache before running')
    parser.add_argument('--text', type=str, help='Input text to process', 
                       default="The quick brown fox jumps over the lazy dog.")
    args = parser.parse_args()
    
    if args.clear_cache and os.path.exists(CACHE_DIR):
        import shutil
        logger.info("Clearing cache...")
        shutil.rmtree(CACHE_DIR)
    
    main()
