import re
import json
import os
from collections import defaultdict

pattern = r'\w+|[\u4e00-\u9fff]|[^\w\s]'

class SimpleTokenizer:
    def __init__(self, vocab_size=30000, special_tokens=None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or {
            "<pad>": 0,
            "<unk>": 1,
            "<s>": 2,
            "</s>": 3,
            "<mask>": 4
        }
        self.token_to_id = {}
        self.id_to_token = {}
        self.vocab_initialized = False
        
    def train(self, texts, min_frequency=2):
        """Train tokenizer on a list of texts."""
        # Count word frequencies
        word_counts = defaultdict(int)
        for text in texts:
            # Simple word-level tokenization with punctuation handling
            words = re.findall(pattern, text.lower())
            for word in words:
                word_counts[word] += 1
        
        # Filter by frequency and limit vocabulary size
        valid_words = [
            word for word, count in 
            sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
            if count >= min_frequency
        ]
        
        # Ensure we don't exceed vocab_size (accounting for special tokens)
        available_slots = self.vocab_size - len(self.special_tokens)
        valid_words = valid_words[:available_slots]
        
        # Create vocabulary
        self.token_to_id = {**self.special_tokens}
        next_id = len(self.special_tokens)
        
        for word in valid_words:
            self.token_to_id[word] = next_id
            next_id += 1
        
        # Create reverse mapping
        self.id_to_token = {id_: token for token, id_ in self.token_to_id.items()}
        self.vocab_initialized = True
        
        print(f"Vocabulary initialized with {len(self.token_to_id)} tokens")
        
    def tokenize(self, text):
        """Convert text to a list of tokens."""
        if not self.vocab_initialized:
            raise ValueError("Tokenizer must be trained before use")
            
        words = re.findall(pattern, text.lower())
        return words
    
    def encode(self, text, add_special_tokens=True):
        """Convert text to token IDs."""
        tokens = self.tokenize(text)
        
        # Convert tokens to IDs, using <unk> for OOV tokens
        token_ids = []
        
        if add_special_tokens:
            token_ids.append(self.token_to_id["<s>"])
            
        for token in tokens:
            token_id = self.token_to_id.get(token, self.token_to_id["<unk>"])
            token_ids.append(token_id)
            
        if add_special_tokens:
            token_ids.append(self.token_to_id["</s>"])
            
        return token_ids
    
    def decode(self, token_ids, skip_special_tokens=True):
        """Convert token IDs back to text."""
        tokens = []
        for token_id in token_ids:
            token = self.id_to_token.get(token_id, "<unk>")
            if skip_special_tokens and token in self.special_tokens:
                continue
            tokens.append(token)
        
        # Simple joining - this is a simplified approach
        return " ".join(tokens)
    
    def save_pretrained(self, save_directory):
        """Save tokenizer vocabulary and configuration to files."""
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
            
        # Save vocab
        vocab_file = os.path.join(save_directory, "vocab.json")
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=2)
            
        # Save config
        config = {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens
        }
        config_file = os.path.join(save_directory, "tokenizer_config.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
            
        print(f"Tokenizer saved to {save_directory}")
    
    @classmethod
    def from_pretrained(cls, directory):
        """Load a tokenizer from a directory."""
        # Load config
        config_file = os.path.join(directory, "tokenizer_config.json")
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        # Initialize with config
        tokenizer = cls(
            vocab_size=config["vocab_size"],
            special_tokens=config["special_tokens"]
        )
        
        # Load vocab
        vocab_file = os.path.join(directory, "vocab.json")
        with open(vocab_file, 'r', encoding='utf-8') as f:
            tokenizer.token_to_id = json.load(f)
            
        # Create reverse mapping
        tokenizer.id_to_token = {id_: token for token, id_ in tokenizer.token_to_id.items()}
        tokenizer.vocab_initialized = True
        
        print(f"Loaded tokenizer with {len(tokenizer.token_to_id)} tokens")
        return tokenizer
    
    def batch_encode_plus(self, texts, max_length=None, padding=False, truncation=False):
        """Encode a batch of texts, with support for padding and truncation."""
        batch_encoding = {
            "input_ids": [],
            "attention_mask": []
        }
        
        for text in texts:
            input_ids = self.encode(text)
            
            # Handle truncation
            if truncation and max_length and len(input_ids) > max_length:
                # Keep special tokens at start and end if possible
                if self.token_to_id["<s>"] == input_ids[0] and self.token_to_id["</s>"] == input_ids[-1]:
                    input_ids = [input_ids[0]] + input_ids[1:-1][:max_length-2] + [input_ids[-1]]
                else:
                    input_ids = input_ids[:max_length]
            
            attention_mask = [1] * len(input_ids)
            
            batch_encoding["input_ids"].append(input_ids)
            batch_encoding["attention_mask"].append(attention_mask)
        
        # Handle padding
        if padding and max_length:
            for i, ids in enumerate(batch_encoding["input_ids"]):
                padding_length = max_length - len(ids)
                if padding_length > 0:
                    batch_encoding["input_ids"][i] = ids + [self.token_to_id["<pad>"]] * padding_length
                    batch_encoding["attention_mask"][i] = batch_encoding["attention_mask"][i] + [0] * padding_length
        
        return batch_encoding