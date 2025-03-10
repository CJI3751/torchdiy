import json
from typing import Dict, List, Optional

class HuggingFaceTokenizer:
    def __init__(self, vocab_file: str, special_tokens_map_file: Optional[str] = None):
        # 載入詞彙表
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self.vocab: Dict[str, int] = json.load(f)  # 假設 vocab 是 JSON 格式
        self.reverse_vocab: Dict[int, str] = {v: k for k, v in self.vocab.items()}

        # 載入特殊 token
        self.special_tokens: Dict[str, str] = {}
        if special_tokens_map_file:
            with open(special_tokens_map_file, 'r', encoding='utf-8') as f:
                self.special_tokens = json.load(f)

    def tokenize(self, text: str) -> List[str]:
        """將文本分割為 token（簡單的空白分割）"""
        return text.split()

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """將 token 轉換為 ID"""
        return [self.vocab.get(token, self.vocab.get(self.special_tokens.get('unk_token', '<unk>'))) for token in tokens]

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """將 ID 轉換為 token"""
        return [self.reverse_vocab.get(id, self.special_tokens.get('unk_token', '<unk>')) for id in ids]

    def encode(self, text: str) -> List[int]:
        """將文本轉換為 token ID 序列"""
        tokens = self.tokenize(text)
        return self.convert_tokens_to_ids(tokens)

    def decode(self, ids: List[int]) -> str:
        """將 token ID 序列轉換回文本"""
        tokens = self.convert_ids_to_tokens(ids)
        return ' '.join(tokens)

# 使用示例
if __name__ == "__main__":
    # 假設 vocab.json 和 special_tokens_map.json 在當前目錄
    vocab_file = "model_files/vocab.json"
    special_tokens_map_file = "model_files/special_tokens_map.json"

    # 初始化 Tokenizer
    tokenizer = HuggingFaceTokenizer(vocab_file, special_tokens_map_file)

    # 測試
    text = "Hello, world! This is a test."
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    print("Original text:", text)
    print("Encoded IDs:", encoded)
    print("Decoded text:", decoded)