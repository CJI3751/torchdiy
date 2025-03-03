import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from . import lm

GPT2Config = transformers.GPT2Config
GPT2Model = transformers.GPT2Model
GPT2Tokenizer = transformers.GPT2Tokenizer

# 本程式由 ccc 指揮 grok 產生
class GPT2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = self.n_embd // self.n_head
        
        # 注意力層的 QKV 投影，這裡調整順序與 transformers 一致
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # 輸出投影
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # Dropout
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        
    def forward(self, x, attn_mask=None):
        B, T, C = x.size()  # Batch, Sequence length, Embedding dim
        
        # 計算 Q, K, V
        qkv = self.c_attn(x).split(self.n_embd, dim=2)
        q, k, v = [t.view(B, T, self.n_head, self.head_dim).transpose(1, 2) for t in qkv]  # [B, n_head, T, head_dim]
        
        # 注意力計算
        attn = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))  # [B, n_head, T, T]
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, -float('inf'))
        
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        y = attn @ v  # [B, n_head, T, head_dim]
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # [B, T, n_embd]
        
        # 輸出投影
        y = self.resid_dropout(self.c_proj(y))
        return y

class GPT2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 前饋層，調整維度順序
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.resid_pdrop)
        self.act = nn.GELU()
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class GPT2Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = GPT2MLP(config)
        
    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln_1(x), attn_mask)  # 殘差連接
        x = x + self.mlp(self.ln_2(x))  # 殘差連接
        return x

class GPT2LMHeadModel(lm.CausalLMModel):
    config_class = GPT2Config

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        # 詞嵌入層
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            'wpe': nn.Embedding(config.n_positions, config.n_embd),
            'drop': nn.Dropout(config.embd_pdrop),
            'h': nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)]),
            'ln_f': nn.LayerNorm(config.n_embd)
        })
        
        # LM Head，調整維度順序
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # 共享權重
        self.lm_head.weight = self.transformer.wte.weight
        
        # 初始化權重
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """初始化權重"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, model_name):
        """
        Initialize a pretrained GPT-2 model by copying weights from a huggingface/transformers checkpoint.
        """
        assert model_name in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}, \
            f"Model name must be one of 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', got {model_name}"
        
        # Load the Hugging Face model and config
        from transformers import GPT2LMHeadModel
        model_hf = GPT2LMHeadModel.from_pretrained(model_name)
        config = model_hf.config
        
        # Create our custom model with the same config
        model = cls(config)
        sd = model.state_dict()
        sd_hf = model_hf.state_dict()

        # Filter out parameters we don't need (like masked_bias)
        keys = [k for k in sd_hf if not k.endswith('attn.masked_bias')]
        
        # Define parameters that need transposition due to Conv1D vs Linear difference
        transposed = [
            'attn.c_attn.weight', 
            'attn.c_proj.weight',
            'mlp.c_fc.weight',
            'mlp.c_proj.weight'
        ]
        
        # Verify we have the same number of parameters to copy
        assert len(keys) == len(sd), f"Parameter count mismatch: HF model has {len(keys)}, our model has {len(sd)}"
        
        # Copy weights with proper handling of transposed parameters
        for k in keys:
            if any(k.endswith(w) for w in transposed):
                # Special treatment for weights that need transposition
                assert sd_hf[k].shape[::-1] == sd[k].shape, \
                    f"Shape mismatch for {k}: HF {sd_hf[k].shape} vs ours {sd[k].shape}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # Direct copy for other parameters
                assert sd_hf[k].shape == sd[k].shape, \
                    f"Shape mismatch for {k}: HF {sd_hf[k].shape} vs ours {sd[k].shape}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        
        # Load the updated state dict into our model
        model.load_state_dict(sd)
        return model

    def forward(self, 
                input_ids=None,
                attention_mask=None,
                labels=None,
                **kwargs):
        batch_size, seq_length = input_ids.size()
        
        # 位置編碼
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        # 詞嵌入和	sig位置嵌入
        inputs_embeds = self.transformer.wte(input_ids)
        position_embeds = self.transformer.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.transformer.drop(hidden_states)
        
        # 建立因果注意力遮罩
        causal_mask = torch.triu(
            torch.ones((seq_length, seq_length), device=input_ids.device), 
            diagonal=1
        ).bool()
        if attention_mask is not None:
            causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
            attention_mask = attention_mask.unsqueeze(-1).expand(-1, seq_length).bool()
            causal_mask = causal_mask | (~attention_mask)
        
        # Transformer Blocks
        for block in self.transformer.h:
            hidden_states = block(hidden_states, attn_mask=causal_mask)
        
        hidden_states = self.transformer.ln_f(hidden_states)
        
        # LM Head
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
        )
