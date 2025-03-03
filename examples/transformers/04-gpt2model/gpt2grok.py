import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Tokenizer, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

class GPT2LMHeadModel(PreTrainedModel):
    config_class = GPT2Config
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        # 詞嵌入層
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        
        # Transformer Decoder 層
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=config.n_embd,
                nhead=config.n_head,
                dim_feedforward=config.n_inner if config.n_inner is not None else 4*config.n_embd,
                dropout=config.attn_pdrop,
                activation='gelu',
                batch_first=True,
                norm_first=True
            ) for _ in range(config.n_layer)
        ])
        
        # LayerNorm
        self.ln_f = nn.LayerNorm(config.n_embd)
        
        # LM Head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # 共享權重
        self.lm_head.weight = self.wte.weight
        
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

    def forward(self, 
                input_ids=None,
                attention_mask=None,
                labels=None,
                **kwargs):
        batch_size, seq_length = input_ids.size()
        
        # 位置編碼
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        # 詞嵌入和位置嵌入
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)
        
        # 建立因果注意力遮罩 (2D 形狀: [seq_length, seq_length])
        causal_mask = torch.triu(
            torch.ones((seq_length, seq_length), device=input_ids.device), 
            diagonal=1
        ).bool()  # True 表示遮罩位置
        
        # 如果有 attention_mask，與因果遮罩結合
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(-1).expand(-1, seq_length).bool()
            causal_mask = causal_mask | (~attention_mask)  # 將 padding 位置也遮罩
            
        # Transformer Decoder
        memory = torch.zeros_like(hidden_states)  # dummy memory for compatibility
        for layer in self.h:
            hidden_states = layer(
                hidden_states,
                memory,
                tgt_mask=causal_mask  # 傳遞 2D 遮罩
            )
        
        hidden_states = self.ln_f(hidden_states)
        
        # LM Head
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # 計算損失
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
        )
    
    def generate(self,
                 input_ids,
                 max_length=50,
                 temperature=1.0,
                 top_p=1.0,
                 do_sample=False,
                 **kwargs):
        """簡單的生成函數"""
        self.eval()
        generated = input_ids
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                outputs = self(input_ids=generated)
                next_token_logits = outputs.logits[:, -1, :] / temperature
                
                # Top-p sampling
                if do_sample and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # 修正 top-p 遮罩
                    next_token_logits.scatter_(1, sorted_indices, 
                                             sorted_logits.masked_fill(sorted_indices_to_remove, -float('inf')))
                
                # 採樣或取最大值
                if do_sample:
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                generated = torch.cat([generated, next_token], dim=1)
                
                # 如果生成 EOS 則停止
                if next_token.item() == self.config.eos_token_id:
                    break
        
        return generated

def main():
    config = GPT2Config.from_pretrained("gpt2")
    model = GPT2LMHeadModel(config)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # 設定輸入文本
    input_text = "Once upon a time"
    inputs = tokenizer(input_text, return_tensors="pt")

    # 讓模型生成文字
    output_tokens = model.generate(
        inputs["input_ids"],
        max_length=100,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

    # 解碼並輸出生成的文本
    generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    print("生成的文本：")
    print(generated_text)

if __name__ == "__main__":
    main()