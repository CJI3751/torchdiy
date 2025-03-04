import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Tokenizer
from transformers.modeling_utils import PreTrainedModel

class GPT2LMHeadModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        # 定義模型架構
        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.positional_encoding = nn.Parameter(torch.zeros(1, config.n_positions, config.n_embd))
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.n_embd,
            nhead=config.n_head,
            dim_feedforward=config.n_ffn,
            dropout=config.resid_pdrop
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.n_layer)
        
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # 初始化權重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, input_ids, attention_mask=None):
        seq_len, batch_size = input_ids.size()
        
        # Embedding 和位置編碼
        embeddings = self.embedding(input_ids) * torch.sqrt(torch.tensor(self.config.n_embd, dtype=torch.float32))
        embeddings = embeddings + self.positional_encoding[:, :seq_len, :]
        
        # 生成遮罩
        tgt_mask = self._generate_square_subsequent_mask(seq_len).to(input_ids.device)
        
        # Transformer Decoder
        # 注意：這裡傳入的 memory 為 None，因為 GPT-2 不需要編碼器的輸出
        transformer_output = self.transformer_decoder(
            tgt=embeddings,
            memory=None,  # 不需要 memory
            tgt_mask=tgt_mask
        )
        
        # LayerNorm 和線性層
        transformer_output = self.ln_f(transformer_output)
        logits = self.lm_head(transformer_output)
        
        return logits
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def generate(self, input_ids, max_length=100, temperature=1.0, top_p=0.9, do_sample=True):
        self.eval()
        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                # 獲取當前輸入的 logits
                logits = self(input_ids)
                
                # 取最後一個 token 的 logits
                next_token_logits = logits[-1, :]
                
                # 溫度調整
                next_token_logits = next_token_logits / temperature
                
                # Top-p (nucleus sampling)
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # 取樣或貪婪選擇
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # 將新 token 加入輸入
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
        
        return input_ids

def main():

    # 定義模型配置
    config = GPT2Config(
        vocab_size=50257,  # GPT-2 的詞彙表大小
        n_embd=768,        # 嵌入維度
        n_layer=12,        # 層數
        n_head=12,         # 注意力頭數
        n_ffn=3072,        # 前饋網絡的隱藏層大小
        n_positions=1024,  # 最大序列長度
        resid_pdrop=0.1,   # Dropout 概率
        initializer_range=0.02  # 權重初始化範圍
    )

    # 初始化模型和 tokenizer
    model = GPT2LMHeadModel(config)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # 設定輸入文本
    input_text = "Once upon a time"
    inputs = tokenizer(input_text, return_tensors="pt")

    # 讓模型生成文字
    output_tokens = model.generate(
        inputs["input_ids"],
        max_length=100,  # 生成最多 100 個 token
        temperature=0.7,  # 控制隨機性
        top_p=0.9,        # Top-p (nucleus sampling)
        do_sample=True    # 啟用隨機取樣
    )

    # 解碼並輸出生成的文本
    generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    print("生成的文本：")
    print(generated_text)

if __name__ == "__main__":
    main()
