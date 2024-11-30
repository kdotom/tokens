from dotenv import load_dotenv
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import nn
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TokenAnalyzer:
    def __init__(self):
        self.token = os.getenv('HF_TOKEN')
        self.model_name = "meta-llama/Llama-3.2-1B"
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True,
            token=self.token
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            token=self.token
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def analyze_sequence(self, text):
        """Analyze the full sequence of token processing"""
        logger.info("\n=== Input Analysis ===")
        logger.info(f"Original text: '{text}'")
        
        # Input tokenization
        input_tokens = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        input_token_strs = self.tokenizer.convert_ids_to_tokens(input_tokens[0])
        
        # Get embeddings
        with torch.no_grad():
            input_embeddings = self.model.get_input_embeddings()(input_tokens)
            
            # Process through model
            outputs = self.model(
                inputs_embeds=input_embeddings,
                return_dict=True
            )
            
            # Get output tokens
            output_logits = outputs.logits
            output_tokens = torch.argmax(output_logits, dim=-1)
            output_token_strs = self.tokenizer.convert_ids_to_tokens(output_tokens[0])
            output_text = self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
        
        # Display detailed analysis
        logger.info("\n=== Token Processing Steps ===")
        logger.info("\n1. Input Token Analysis:")
        logger.info("-" * 80)
        logger.info(f"{'Position':<10} {'Token ID':<10} {'Token':<20} {'Token String':<30}")
        logger.info("-" * 80)
        
        for pos, (token_id, token_str) in enumerate(zip(input_tokens[0].tolist(), input_token_strs)):
            display_str = token_str.replace('Ġ', '[SPACE]')
            logger.info(f"{pos:<10} {token_id:<10} {display_str:<20} {self.explain_token(token_str):<30}")
        
        logger.info("\n2. Embedding Shapes:")
        logger.info(f"Input embedding shape:  {input_embeddings.shape}")
        logger.info(f"Input embedding dtype:  {input_embeddings.dtype}")
        
        logger.info("\n3. Output Token Analysis:")
        logger.info("-" * 80)
        logger.info(f"{'Position':<10} {'Token ID':<10} {'Token':<20} {'Token String':<30}")
        logger.info("-" * 80)
        
        for pos, (token_id, token_str) in enumerate(zip(output_tokens[0].tolist(), output_token_strs)):
            display_str = token_str.replace('Ġ', '[SPACE]')
            logger.info(f"{pos:<10} {token_id:<10} {display_str:<20} {self.explain_token(token_str):<30}")
        
        logger.info("\n4. Final Output:")
        logger.info(f"Generated text: '{output_text}'")
        
        # Token matching analysis
        logger.info("\n5. Token Matching Analysis:")
        logger.info("-" * 80)
        logger.info("Position  Input Token -> Output Token")
        logger.info("-" * 80)
        
        for pos in range(min(len(input_token_strs), len(output_token_strs))):
            in_tok = input_token_strs[pos].replace('Ġ', '[SPACE]')
            out_tok = output_token_strs[pos].replace('Ġ', '[SPACE]')
            match = "✓" if input_tokens[0][pos] == output_tokens[0][pos] else "×"
            logger.info(f"{pos:<8}  {in_tok:<20} -> {out_tok:<20} {match}")
            
        return {
            'input_tokens': input_tokens,
            'input_token_strs': input_token_strs,
            'output_tokens': output_tokens,
            'output_token_strs': output_token_strs,
            'output_text': output_text
        }
    
    def explain_token(self, token):
        """Provide a human-readable explanation of a token"""
        if token == '<|begin_of_text|>':
            return "Start of text marker"
        elif token == self.tokenizer.pad_token:
            return "Padding token"
        elif token == self.tokenizer.eos_token:
            return "End of text marker"
        elif token.startswith('Ġ'):
            return f"Space + '{token[1:]}'"
        else:
            return f"'{token}'"

def main():
    try:
        analyzer = TokenAnalyzer()
        
        # Test cases
        test_texts = [
"""
Montgomery, Alabama
Juneau, Alaska
Phoenix, Arizona
Little Rock, Arkansas
Sacramento, California
Denver, Colorado
Hartford, Connecticut
Dover, Delaware
Tallahassee, Florida
Atlanta, Georgia
Honolulu, Hawaii
Boise, Idaho
Springfield, Illinois
Indianapolis, Indiana
Des Moines, Iowa
Topeka, Kansas
Frankfort, Kentucky
Baton Rouge, Louisiana
Augusta, Maine
Annapolis, Maryland
Boston, Massachusetts
Lansing, Michigan
Saint Paul, Minnesota
Jackson, Mississippi
Jefferson City, Missouri
Helena, Montana
Lincoln, Nebraska
Carson City, Nevada
Concord, New Hampshire
Trenton, New Jersey
Santa Fe, New Mexico
Albany, New York
Raleigh, North Carolina
Bismarck, North Dakota"""
        ,]
        
        for text in test_texts:
            logger.info("\n" + "="*80)
            results = analyzer.analyze_sequence(text)
            logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()
