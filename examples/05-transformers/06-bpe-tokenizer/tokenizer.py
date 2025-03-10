import re
import json
import os
from collections import defaultdict, Counter

# pattern = r'[\u4e00-\u9fff]|\w+|[^\w\s]'
pattern = r'\w+|[^\w\s]'

class BPETokenizer:
    def __init__(self, vocab_size=30000):
        self.vocab_size = vocab_size
        self.special_tokens = {
            "<pad>": 0,
            "<unk>": 1,
            "<s>": 2,
            "</s>": 3,
            "<mask>": 4
        }
        self.merges = {}  # BPE合併操作
        self.vocab = {}   # 詞彙表
        self.inverse_vocab = {}  # 反向詞彙表
        self.initialized = False
        
    def _get_stats(self, words):
        """計算所有符號對的頻率"""
        pairs = defaultdict(int)
        for word, freq in words.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs
        
    def _merge_pair(self, pair, words):
        """將符號對合併為一個新符號"""
        new_words = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        
        for word, freq in words.items():
            parts = word.split()
            if len(parts) == 1:  # 單個符號，無需合併
                new_words[word] = freq
                continue
                
            # 用新符號替換所有出現的符號對
            i = 0
            new_parts = []
            while i < len(parts):
                if i < len(parts) - 1 and parts[i] == pair[0] and parts[i + 1] == pair[1]:
                    new_parts.append(replacement)
                    i += 2
                else:
                    new_parts.append(parts[i])
                    i += 1
            
            new_word = ' '.join(new_parts)
            new_words[new_word] = freq
            
        return new_words
        
    def train(self, texts, num_merges=None):
        """從文本語料庫訓練BPE模型"""
        if num_merges is None:
            # 設置合併次數為詞彙表大小減去特殊標記數量
            num_merges = self.vocab_size - len(self.special_tokens)
        
        # 預處理：將每個文本轉換為字符序列
        all_words = defaultdict(int)
        for text in texts:
            # 使用正則表達式將文本分割成單詞和標點符號
            tokens = re.findall(pattern, text.lower())
            for token in tokens:
                # 將每個詞表示為單個字符之間有空格的序列
                char_sequence = ' '.join(list(token))
                all_words[char_sequence] += 1
                
        # BPE訓練：迭代合併最常見的符號對
        for i in range(num_merges):
            # 找出最常見的符號對
            pairs = self._get_stats(all_words)
            if not pairs:
                break
                
            # 獲取頻率最高的符號對
            best_pair = max(pairs, key=pairs.get)
            
            # 將該符號對添加到合併操作中
            self.merges[best_pair] = i + len(self.special_tokens)
            
            # 在所有單詞中應用合併
            all_words = self._merge_pair(best_pair, all_words)
            
            if (i + 1) % 1000 == 0:
                print(f"Completed {i + 1} merges")
        
        # 建立詞彙表
        self.vocab = {**self.special_tokens}  # 從特殊標記開始
        next_id = len(self.special_tokens)
        
        # 將所有合併後的符號添加到詞彙表
        for pair, _ in self.merges.items():
            merged = ''.join(pair)
            if merged not in self.vocab:
                self.vocab[merged] = next_id
                next_id += 1
        
        # 將所有單詞中的符號添加到詞彙表
        for word in all_words:
            for symbol in word.split():
                if symbol not in self.vocab:
                    self.vocab[symbol] = next_id
                    next_id += 1
        
        # 創建反向詞彙表
        self.inverse_vocab = {id_: token for token, id_ in self.vocab.items()}
        self.initialized = True
        
        print(f"Vocabulary initialized with {len(self.vocab)} tokens")
    
    def _tokenize_word(self, word):
        """使用學習到的BPE合併規則將單詞分解為子詞標記"""
        if not word:
            return []
            
        # 將單詞分解為字符序列
        chars = list(word)
        word_tokens = ' '.join(chars)
        
        # 應用合併規則
        while True:
            pairs = self._get_stats({word_tokens: 1})
            if not pairs:
                break
                
            # 找出可以合併的符號對
            mergeable_pairs = [pair for pair in pairs if pair in self.merges]
            if not mergeable_pairs:
                break
                
            # 應用優先級最高（最早學習到）的合併
            best_pair = min(mergeable_pairs, key=lambda pair: self.merges[pair])
            word_tokens = ' '.join(self._merge_pair(best_pair, {word_tokens: 1}))
        
        return word_tokens.split()
    
    def tokenize(self, text):
        """將文本分詞為BPE標記"""
        if not self.initialized:
            raise ValueError("Tokenizer must be trained before use")
            
        # 使用正則表達式將文本分割成單詞和標點符號
        words = re.findall(pattern, text.lower())
        tokens = []
        
        for word in words:
            word_tokens = self._tokenize_word(word)
            tokens.extend(word_tokens)
            
        return tokens
    
    def encode(self, text, add_special_tokens=True):
        """將文本轉換為標記ID"""
        tokens = self.tokenize(text)
        
        # 將標記轉換為ID
        token_ids = []
        
        if add_special_tokens:
            token_ids.append(self.vocab["<s>"])
            
        for token in tokens:
            token_id = self.vocab.get(token, self.vocab["<unk>"])
            token_ids.append(token_id)
            
        if add_special_tokens:
            token_ids.append(self.vocab["</s>"])
            
        return token_ids
    
    def decode(self, token_ids, skip_special_tokens=True):
        """將標記ID轉換回文本"""
        tokens = []
        for token_id in token_ids:
            token = self.inverse_vocab.get(token_id, "<unk>")
            if skip_special_tokens and token in self.special_tokens:
                continue
            tokens.append(token)
        
        # 簡單連接 - 這是一個簡化的方法，實際上可能需要更複雜的處理
        return '/'.join(tokens)
    
    def save_pretrained(self, save_directory):
        """將分詞器詞彙表和配置保存到文件"""
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
            
        # 保存詞彙表
        vocab_file = os.path.join(save_directory, "vocab.json")
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
            
        # 保存合併操作
        merges_file = os.path.join(save_directory, "merges.json")
        # 將元組鍵轉換為字符串
        merges_dict = {' '.join(pair): idx for pair, idx in self.merges.items()}
        with open(merges_file, 'w', encoding='utf-8') as f:
            json.dump(merges_dict, f, ensure_ascii=False, indent=2)
            
        # 保存配置
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
        """從目錄加載分詞器"""
        # 加載配置
        config_file = os.path.join(directory, "tokenizer_config.json")
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        # 使用配置初始化
        tokenizer = cls(
            vocab_size=config["vocab_size"]
        )
        tokenizer.special_tokens = config["special_tokens"]
        
        # 加載詞彙表
        vocab_file = os.path.join(directory, "vocab.json")
        with open(vocab_file, 'r', encoding='utf-8') as f:
            tokenizer.vocab = json.load(f)
            
        # 加載合併操作
        merges_file = os.path.join(directory, "merges.json")
        with open(merges_file, 'r', encoding='utf-8') as f:
            merges_dict = json.load(f)
            # 將字符串鍵轉換回元組
            tokenizer.merges = {tuple(k.split()): v for k, v in merges_dict.items()}
            
        # 創建反向詞彙表
        tokenizer.inverse_vocab = {int(id_): token for token, id_ in tokenizer.vocab.items()}
        tokenizer.initialized = True
        
        print(f"Loaded tokenizer with {len(tokenizer.vocab)} tokens")
        return tokenizer
    
    def batch_encode_plus(self, texts, max_length=None, padding=False, truncation=False):
        """編碼一批文本，支持填充和截斷"""
        batch_encoding = {
            "input_ids": [],
            "attention_mask": []
        }
        
        for text in texts:
            input_ids = self.encode(text)
            
            # 處理截斷
            if truncation and max_length and len(input_ids) > max_length:
                # 如果可能，保留開頭和結尾的特殊標記
                if self.vocab["<s>"] == input_ids[0] and self.vocab["</s>"] == input_ids[-1]:
                    input_ids = [input_ids[0]] + input_ids[1:-1][:max_length-2] + [input_ids[-1]]
                else:
                    input_ids = input_ids[:max_length]
            
            attention_mask = [1] * len(input_ids)
            
            batch_encoding["input_ids"].append(input_ids)
            batch_encoding["attention_mask"].append(attention_mask)
        
        # 處理填充
        if padding and max_length:
            for i, ids in enumerate(batch_encoding["input_ids"]):
                padding_length = max_length - len(ids)
                if padding_length > 0:
                    batch_encoding["input_ids"][i] = ids + [self.vocab["<pad>"]] * padding_length
                    batch_encoding["attention_mask"][i] = batch_encoding["attention_mask"][i] + [0] * padding_length
        
        return batch_encoding
