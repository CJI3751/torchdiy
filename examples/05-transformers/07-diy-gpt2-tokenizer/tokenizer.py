# tokenizer.py
import os
import json
import regex as re
from typing import Dict, List, Tuple, Union, Optional

class GPT2Tokenizer:
    """
    自己實作的 GPT2Tokenizer，不依賴 Hugging Face transformers 庫
    使用 Byte-Pair Encoding (BPE) 演算法進行分詞
    """
    
    def __init__(
        self,
        vocab_file: str,
        merges_file: str,
        errors: str = "replace",
        unk_token: str = "<|endoftext|>",
        bos_token: str = "<|endoftext|>",
        eos_token: str = "<|endoftext|>",
        pad_token: Optional[str] = None,
    ):
        self.vocab_file = vocab_file
        self.merges_file = merges_file
        self.errors = errors
        
        # 載入詞彙表和合併規則
        self.encoder = self._load_vocab(vocab_file)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = self._load_merges(merges_file)
        
        # 建立 byte-to-unicode 映射表
        self.byte_encoder = self._bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        
        # 用於 BPE 分詞的正則表達式，GPT-2使用此模式來分詞
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        
        # 特殊 token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        
        # 特殊 token ID
        self.unk_token_id = self.encoder.get(self.unk_token, 0)
        self.bos_token_id = self.encoder.get(self.bos_token, 0)
        self.eos_token_id = self.encoder.get(self.eos_token, 0)
        self.pad_token_id = self.encoder.get(self.pad_token) if self.pad_token else None
        
        # 快取已分詞的字詞，提高效率
        self.cache = {}
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> "GPT2Tokenizer":
        """
        從預訓練模型路徑或名稱載入 tokenizer
        
        Args:
            model_name_or_path: 可以是本地目錄路徑或預設的模型名稱
                               例如 "openai-community/gpt2"
        
        Returns:
            GPT2Tokenizer 實例
        """
        # 處理預設模型名稱 (這裡的實現是將模型名稱轉換為本地目錄)
        if model_name_or_path.startswith("openai-community/"):
            # 在實際使用中，你需要下載並保存這些檔案，或者實現一個下載功能
            # 這邊簡化處理，假設已經存在於某個位置
            base_name = model_name_or_path.split("/")[-1]
            model_path = os.path.join(os.path.dirname(__file__), "models", base_name)
        else:
            model_path = model_name_or_path
        
        # 檢查檔案是否存在
        vocab_file = os.path.join(model_path, "vocab.json")
        merges_file = os.path.join(model_path, "merges.txt")
        
        if not os.path.isfile(vocab_file) or not os.path.isfile(merges_file):
            # 下載檔案的邏輯可以在這裡實現
            # 為了簡化，這裡假設檔案已經存在，否則拋出錯誤
            raise ValueError(
                f"找不到必要的詞彙表和合併規則檔案。請確保 {vocab_file} 和 {merges_file} 存在。"
            )
        
        return cls(vocab_file, merges_file)
    
    def _load_vocab(self, vocab_file: str) -> Dict[str, int]:
        """載入詞彙表"""
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        return vocab
    
    def _load_merges(self, merges_file: str) -> Dict[Tuple[str, str], int]:
        """載入 BPE 合併規則"""
        merges = {}
        with open(merges_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) == 2:
                    merges[tuple(parts)] = i
        return merges
    
    def _bytes_to_unicode(self) -> Dict[int, str]:
        """
        建立 byte-to-unicode 映射表，
        這是 GPT-2 tokenizer 的一個特點，用於處理所有 UTF-8 字符
        """
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        cs = bs.copy()
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        cs = [chr(c) for c in cs]
        return dict(zip(bs, cs))
    
    def _get_pairs(self, word: List[str]) -> set:
        """獲取所有相鄰符號對"""
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs
    
    def _bpe(self, token: str) -> str:
        """
        應用 BPE 算法進行分詞
        """
        if token in self.cache:
            return self.cache[token]
        
        word = list(token)
        pairs = self._get_pairs(word)
        
        if not pairs:
            return token
        
        while True:
            # 找到優先級最高的對
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            
            if bigram not in self.bpe_ranks:
                break
                
            first, second = bigram
            new_word = []
            i = 0
            
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
                    
            word = new_word
            if len(word) == 1:
                break
                
            pairs = self._get_pairs(word)
        
        result = ' '.join(word)
        self.cache[token] = result
        return result
    
    def tokenize(self, text: str) -> List[str]:
        """將文本分詞為 token"""
        tokens = []
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8', errors=self.errors))
            tokens.extend(bpe_token for bpe_token in self._bpe(token).split(' '))
        return tokens
    
    def encode(self, text: str) -> List[int]:
        """將文本轉換為 token ID"""
        tokens = self.tokenize(text)
        return [self.encoder.get(token, self.unk_token_id) for token in tokens]
    
    def decode(self, token_ids: List[int]) -> str:
        """將 token ID 轉換回文本"""
        text = '/'.join([self.decoder.get(token_id, self.unk_token) for token_id in token_ids])
        byte_array = bytearray([self.byte_decoder[c] for c in text])
        return byte_array.decode('utf-8', errors=self.errors)
    
    def __call__(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
    ) -> Dict:
        """支持 HuggingFace 風格的調用"""
        if isinstance(text, str):
            # 單個文本處理
            input_ids = self.encode(text)
            
            # 添加特殊 token
            if add_special_tokens:
                if self.bos_token_id is not None:
                    input_ids = [self.bos_token_id] + input_ids
                if self.eos_token_id is not None:
                    input_ids = input_ids + [self.eos_token_id]
            
            # 處理截斷
            if truncation and max_length and len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
            
            # 創建 attention_mask
            attention_mask = [1] * len(input_ids)
            
            # 處理填充
            if padding and max_length and len(input_ids) < max_length:
                pad_token_id = self.pad_token_id if self.pad_token_id is not None else 0
                pad_length = max_length - len(input_ids)
                input_ids = input_ids + [pad_token_id] * pad_length
                attention_mask = attention_mask + [0] * pad_length
            
            result = {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
            
        else:
            # 批量處理多個文本
            batch_input_ids = []
            for t in text:
                ids = self.encode(t)
                
                # 添加特殊 token
                if add_special_tokens:
                    if self.bos_token_id is not None:
                        ids = [self.bos_token_id] + ids
                    if self.eos_token_id is not None:
                        ids = ids + [self.eos_token_id]
                
                # 處理截斷
                if truncation and max_length and len(ids) > max_length:
                    ids = ids[:max_length]
                
                batch_input_ids.append(ids)
            
            # 處理填充
            if padding:
                # 找出最長的序列長度
                if max_length:
                    max_len = max_length
                else:
                    max_len = max(len(ids) for ids in batch_input_ids)
                
                pad_token_id = self.pad_token_id if self.pad_token_id is not None else 0
                padded_input_ids = []
                attention_mask = []
                
                for ids in batch_input_ids:
                    padded_ids = ids + [pad_token_id] * (max_len - len(ids))
                    mask = [1] * len(ids) + [0] * (max_len - len(ids))
                    padded_input_ids.append(padded_ids)
                    attention_mask.append(mask)
                
                result = {
                    'input_ids': padded_input_ids,
                    'attention_mask': attention_mask
                }
            else:
                result = {
                    'input_ids': batch_input_ids,
                    'attention_mask': [[1] * len(ids) for ids in batch_input_ids]
                }
        
        # 如果需要將結果轉換為特定格式的張量
        if return_tensors == "pt":
            # 如果需要 PyTorch 張量，需要導入 torch
            try:
                import torch
                for k, v in result.items():
                    if isinstance(v[0], list):
                        result[k] = torch.tensor(v)
                    else:
                        result[k] = torch.tensor([v])
            except ImportError:
                raise ImportError("要使用 return_tensors='pt'，需要安裝 PyTorch")
        elif return_tensors == "tf":
            # 如果需要 TensorFlow 張量
            try:
                import tensorflow as tf
                for k, v in result.items():
                    if isinstance(v[0], list):
                        result[k] = tf.constant(v)
                    else:
                        result[k] = tf.constant([v])
            except ImportError:
                raise ImportError("要使用 return_tensors='tf'，需要安裝 TensorFlow")
        elif return_tensors == "np":
            # 如果需要 NumPy 數組
            try:
                import numpy as np
                for k, v in result.items():
                    if isinstance(v[0], list):
                        result[k] = np.array(v)
                    else:
                        result[k] = np.array([v])
            except ImportError:
                raise ImportError("要使用 return_tensors='np'，需要安裝 NumPy")
        
        return result


# 下載與儲存 GPT-2 詞彙表和合併規則的輔助方法
def download_gpt2_tokenizer_files(save_dir="./models/gpt2"):
    """
    下載 GPT-2 的詞彙表和合併規則文件，並保存到指定目錄
    
    Args:
        save_dir: 保存目錄
    """
    import requests
    
    # 創建目錄
    os.makedirs(save_dir, exist_ok=True)
    
    # 下載詞彙表
    vocab_url = "https://huggingface.co/gpt2/resolve/main/vocab.json"
    vocab_path = os.path.join(save_dir, "vocab.json")
    
    if not os.path.exists(vocab_path):
        response = requests.get(vocab_url)
        if response.status_code == 200:
            with open(vocab_path, "wb") as f:
                f.write(response.content)
            print(f"成功下載詞彙表到 {vocab_path}")
        else:
            print(f"下載詞彙表失敗，狀態碼: {response.status_code}")
    
    # 下載合併規則
    merges_url = "https://huggingface.co/gpt2/resolve/main/merges.txt"
    merges_path = os.path.join(save_dir, "merges.txt")
    
    if not os.path.exists(merges_path):
        response = requests.get(merges_url)
        if response.status_code == 200:
            with open(merges_path, "wb") as f:
                f.write(response.content)
            print(f"成功下載合併規則到 {merges_path}")
        else:
            print(f"下載合併規則失敗，狀態碼: {response.status_code}")