import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import GPT2LMHeadModel as HF_GPT2LMHeadModel
from transformers import GPT2Config as HF_GPT2Config
from transformers import AutoConfig as HF_AutoConfig, AutoTokenizer as HF_AutoTokenizer

class GPT2Config:
    def __init__(
        self,
        vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        activation_function="gelu",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon


class GPT2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        
        # Register buffer for causal mask
        mask = torch.tril(torch.ones(config.n_positions, config.n_positions))
        self.register_buffer("mask", mask.view(1, 1, config.n_positions, config.n_positions))
    
    def forward(self, x, layer_past=None, use_cache=False):
        batch_size, seq_len, n_embd = x.shape
        
        # Linear projection for query, key, value
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape to (batch, head, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        
        # Compute attention
        attn_weights = q @ k.transpose(2, 3) / math.sqrt(self.head_dim)
        
        # Apply causal mask
        mask = self.mask[:, :, :seq_len, :seq_len]
        attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention
        attn_output = attn_weights @ v
        
        # Reshape back to (batch, seq_len, n_embd)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_embd)
        
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        
        return attn_output


class GPT2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.resid_pdrop)
        
        if config.activation_function == 'gelu':
            self.act = F.gelu
        elif config.activation_function == 'relu':
            self.act = F.relu
        else:
            raise ValueError(f"Unsupported activation: {config.activation_function}")
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class GPT2Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(config)
    
    def forward(self, x):
        # Self-attention with residual connection
        attn_output = self.attn(self.ln_1(x))
        x = x + attn_output
        
        # MLP with residual connection
        mlp_output = self.mlp(self.ln_2(x))
        x = x + mlp_output
        
        return x


class GPT2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)  # Token embeddings
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)  # Position embeddings
        self.drop = nn.Dropout(config.embd_pdrop)
        
        # Transformer blocks
        self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
    
    def forward(self, input_ids, position_ids=None, attention_mask=None):
        # Get input shape
        batch_size, seq_length = input_ids.size()
        
        # Generate position ids if none provided
        if position_ids is None:
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Get token and position embeddings
        token_embeddings = self.wte(input_ids)
        position_embeddings = self.wpe(position_ids)
        
        # Combine embeddings
        x = token_embeddings + position_embeddings
        x = self.drop(x)
        
        # Apply transformer blocks
        for block in self.h:
            x = block(x)
        
        # Apply final layer norm
        x = self.ln_f(x)
        
        return x


class GPT2LMHeadModel(nn.Module):
    def __init__(self, config=None, pretrained_model_name="gpt2"):
        """
        Initialize a GPT2 model with optional loading from Hugging Face
        
        Args:
            config: Optional custom config
            pretrained_model_name: Name of pretrained model from Hugging Face ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")
        """
        super().__init__()
        
        # If no config provided, create default
        if config is None:
            config = GPT2Config()
        
        # Create our model architecture
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Tie weights between embedding and output layer
        self.lm_head.weight = self.transformer.wte.weight
        
        # Load pretrained weights if specified
        if pretrained_model_name:
            self.load_from_huggingface(pretrained_model_name)
    
    def load_from_huggingface(self, model_name_or_path):
        """
        Load pretrained weights from Hugging Face model
        
        Args:
            model_name_or_path: Model name ("gpt2", "gpt2-medium", etc) or path to local model
        """
        print(f"Loading pretrained weights from {model_name_or_path}...")
        
        # Load pretrained model from Hugging Face
        hf_model = HF_GPT2LMHeadModel.from_pretrained(model_name_or_path)
        hf_config = hf_model.config
        
        # Update our config to match the loaded model
        self.update_config_from_hf(hf_config)
        
        # Transfer weights from HF model to our model
        self._copy_weights_from_hf_gpt2(hf_model)
        
        print("Successfully loaded weights!")
    
    def update_config_from_hf(self, hf_config):
        """Update our model's config to match the HF config"""
        config = self.transformer.config
        
        # Update config attributes
        config.vocab_size = hf_config.vocab_size
        config.n_positions = hf_config.n_positions
        config.n_embd = hf_config.n_embd
        config.n_layer = hf_config.n_layer
        config.n_head = hf_config.n_head
        config.resid_pdrop = hf_config.resid_pdrop
        config.embd_pdrop = hf_config.embd_pdrop
        config.attn_pdrop = hf_config.attn_pdrop
        config.layer_norm_epsilon = hf_config.layer_norm_epsilon
        
        # Recreate modules that depend on config parameters
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight
    
    def _copy_weights_from_hf_gpt2(self, hf_model):
        """Copy weights from Hugging Face model to our model"""
        # Copy token embeddings
        self.transformer.wte.weight.data.copy_(hf_model.transformer.wte.weight.data)
        
        # Copy position embeddings
        self.transformer.wpe.weight.data.copy_(hf_model.transformer.wpe.weight.data)
        
        # Copy transformer blocks
        for i, block in enumerate(self.transformer.h):
            hf_block = hf_model.transformer.h[i]
            
            # Copy attention weights
            block.attn.c_attn.weight.data.copy_(hf_block.attn.c_attn.weight.data)
            block.attn.c_attn.bias.data.copy_(hf_block.attn.c_attn.bias.data)
            block.attn.c_proj.weight.data.copy_(hf_block.attn.c_proj.weight.data)
            block.attn.c_proj.bias.data.copy_(hf_block.attn.c_proj.bias.data)
            
            # Copy MLP weights
            block.mlp.c_fc.weight.data.copy_(hf_block.mlp.c_fc.weight.data)
            block.mlp.c_fc.bias.data.copy_(hf_block.mlp.c_fc.bias.data)
            block.mlp.c_proj.weight.data.copy_(hf_block.mlp.c_proj.weight.data)
            block.mlp.c_proj.bias.data.copy_(hf_block.mlp.c_proj.bias.data)
            
            # Copy layer norms
            block.ln_1.weight.data.copy_(hf_block.ln_1.weight.data)
            block.ln_1.bias.data.copy_(hf_block.ln_1.bias.data)
            block.ln_2.weight.data.copy_(hf_block.ln_2.weight.data)
            block.ln_2.bias.data.copy_(hf_block.ln_2.bias.data)
        
        # Copy final layer norm
        self.transformer.ln_f.weight.data.copy_(hf_model.transformer.ln_f.weight.data)
        self.transformer.ln_f.bias.data.copy_(hf_model.transformer.ln_f.bias.data)
    
    def forward(self, input_ids, position_ids=None, attention_mask=None, labels=None):
        # Get transformer output
        transformer_output = self.transformer(input_ids, position_ids, attention_mask)
        
        # Apply language model head
        lm_logits = self.lm_head(transformer_output)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate cross entropy loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return {"logits": lm_logits, "loss": loss}
    
    def generate(self, input_ids=None, attention_mask=None, max_length=100, temperature=1.0, 
                 top_k=0, top_p=0.9, repetition_penalty=1.0, do_sample=True, **kwargs):
        """
        Text generation with various sampling strategies
        
        Args:
            input_ids: Input token ids
            attention_mask: Attention mask
            max_length: Maximum length of generated sequence
            temperature: Sampling temperature
            top_k: Keep only top k tokens for sampling
            top_p: Keep tokens with cumulative probability >= top_p
            repetition_penalty: Penalty for repeating tokens
            do_sample: Whether to sample or use greedy decoding
            **kwargs: Additional arguments
        """
        # Extract any additional inputs from kwargs
        if 'position_ids' in kwargs:
            position_ids = kwargs['position_ids']
        else:
            position_ids = None
        
        # Make sure input_ids is on the right device
        input_ids = input_ids.to(next(self.parameters()).device)
        
        # Current sequence length
        cur_len = input_ids.shape[1]
        
        # Set effective max length based on current input length
        effective_max_length = max(max_length, cur_len)
        
        # Generate tokens up to max_length
        for _ in range(cur_len, effective_max_length):
            with torch.no_grad():
                # Forward pass
                outputs = self.forward(input_ids, position_ids, attention_mask)
                next_token_logits = outputs["logits"][:, -1, :]
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for i in range(input_ids.shape[0]):
                        for token_id in set(input_ids[i].tolist()):
                            next_token_logits[i, token_id] /= repetition_penalty
                
                # Filter based on top-k
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Filter based on top-p
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to keep the first token above threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Scatter sorted tensors to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample from filtered distribution
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append next token to sequence
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                # Update attention mask if provided
                if attention_mask is not None:
                    attention_mask = torch.cat([
                        attention_mask,
                        attention_mask.new_ones((attention_mask.shape[0], 1))
                    ], dim=-1)
        
        return input_ids


# Auto classes for compatibility with transformers library
class AutoConfig:
    @staticmethod
    def from_pretrained(pretrained_model_name_or_path, **kwargs):
        """
        Load a configuration from Hugging Face's model hub
        """
        return HF_AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

class AutoTokenizer:
    @staticmethod
    def from_pretrained(pretrained_model_name_or_path, **kwargs):
        """
        Load a tokenizer from Hugging Face's model hub
        """
        return HF_AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)

class AutoModelForCausalLM:
    @staticmethod
    def from_pretrained(pretrained_model_name_or_path, **kwargs):
        """
        Load a pretrained GPT2 model from Hugging Face
        """
        config = None
        if 'config' in kwargs:
            config = kwargs.pop('config')
        
        return GPT2LMHeadModel(config=config, pretrained_model_name=pretrained_model_name_or_path)


def main():
    # from transformers import AutoModelForCausalLM, AutoTokenizer
    # from torchdiy.transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    # 指定 GPT-2 模型名稱
    model_name = "gpt2"  # 任何有 generate 方法的模型都可以使用
    # model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    # 自動載入模型，配置與 tokenizer
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print('config:', config)
    # 設定輸入文本
    print('============================')
    input_text = "Once upon a time"
    inputs = tokenizer(input_text, return_tensors="pt")

    # 讓模型生成文字
    output_tokens = model.generate(
        **inputs,
        max_length=100,  # 生成最多 100 個 token
        temperature=0.7,  # 控制隨機性，較低的值產生較為穩定的結果
        top_p=0.9,  # Top-p (nucleus sampling)
        # repetition_penalty=1.1,  # 防止重複詞彙
        do_sample=True  # 啟用隨機取樣
    )

    # 解碼並輸出生成的文本
    generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    print("生成的文本：")
    print(generated_text)
    
    return model, tokenizer

if __name__ == "__main__":
    main()
