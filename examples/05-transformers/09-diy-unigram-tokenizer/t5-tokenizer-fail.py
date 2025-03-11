import numpy as np
import re
import collections
import math
from typing import List, Dict, Tuple, Set, Optional

class UnigramTokenizer:
    def __init__(
        self,
        vocab_size: int = 8000,
        character_coverage: float = 0.9995,
        unk_token: str = "<unk>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        pad_token: str = "<pad>",
    ):
        """
        初始化Unigram Language Model分词器
        
        Args:
            vocab_size: 词汇表大小
            character_coverage: 字符覆盖率
            unk_token: 未知标记
            bos_token: 句子开始标记
            eos_token: 句子结束标记
            pad_token: 填充标记
        """
        self.vocab_size = vocab_size
        self.character_coverage = character_coverage
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        
        # 特殊标记ID
        self.special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        
        # 初始化词汇表和概率
        self.vocab = {}
        self.token_to_id = {}
        self.id_to_token = {}
        self.token_to_score = {}  # log probability scores
        
        # 初始化其他
        self.trained = False
    
    def _get_corpus_stats(self, corpus: List[str]) -> Tuple[Dict[str, int], Set[str]]:
        """
        获取语料库统计信息
        
        Args:
            corpus: 文本语料库
        
        Returns:
            character_freq: 字符频率
            charset: 字符集合
        """
        character_freq = collections.defaultdict(int)
        charset = set()
        
        for text in corpus:
            for char in text:
                character_freq[char] += 1
                charset.add(char)
        
        return character_freq, charset
    
    def _initialize_seed_vocab(self, charset: Set[str]) -> Dict[str, int]:
        """
        初始化种子词汇表
        
        Args:
            charset: 字符集合
        
        Returns:
            seed_vocab: 种子词汇表
        """
        seed_vocab = {}
        
        # 添加特殊标记
        for token in self.special_tokens:
            seed_vocab[token] = 1
        
        # 添加单个字符
        for char in charset:
            seed_vocab[char] = 1
        
        # 推导基本子词组合 (可选：这里可以添加常用二元、三元组等)
        return seed_vocab
    
    def _expand_vocab(self, corpus: List[str], seed_vocab: Dict[str, int]) -> Dict[str, int]:
        """
        扩展词汇表
        
        Args:
            corpus: 文本语料库
            seed_vocab: 种子词汇表
        
        Returns:
            expanded_vocab: 扩展后的词汇表
        """
        # 这里我们从种子词汇表开始，添加常见的字符对
        expanded_vocab = seed_vocab.copy()
        
        # 用于统计字符对频率
        pair_counts = collections.defaultdict(int)
        
        # 收集所有文本中的字符对频率
        for text in corpus:
            chars = list(text)
            for i in range(len(chars) - 1):
                pair = chars[i] + chars[i+1]
                pair_counts[pair] += 1
        
        # 按频率排序并添加到扩展词汇表，直到达到vocab_size的一定比例
        # 这里我们先扩展到vocab_size的两倍，后续会通过剪枝减小
        target_size = min(self.vocab_size * 2, len(pair_counts) + len(seed_vocab))
        
        for pair, count in sorted(pair_counts.items(), key=lambda x: -x[1]):
            if len(expanded_vocab) >= target_size:
                break
            if pair not in expanded_vocab:
                expanded_vocab[pair] = count
        
        return expanded_vocab
    
    def _compute_token_frequency(self, corpus: List[str], vocab: Dict[str, int]) -> Dict[str, int]:
        """
        计算语料库中每个token的频率
        
        Args:
            corpus: 文本语料库
            vocab: 词汇表
        
        Returns:
            token_freq: token频率
        """
        token_freq = collections.defaultdict(int)
        
        # 初始为词汇表中的所有token赋予一个小的计数
        for token in vocab:
            token_freq[token] = 1  # 平滑处理
        
        # 对语料库进行分词并计算频率
        for text in corpus:
            tokens = self._tokenize_with_vocab(text, vocab)
            for token in tokens:
                token_freq[token] += 1
        
        return token_freq
    
    def _tokenize_with_vocab(self, text: str, vocab: Dict[str, int]) -> List[str]:
        """
        使用给定词汇表对文本进行分词
        
        Args:
            text: 输入文本
            vocab: 词汇表
        
        Returns:
            tokens: 分词结果
        """
        # 使用Viterbi算法进行最佳路径分词
        tokens = self._viterbi_segment(text, vocab)
        return tokens
    
    def _viterbi_segment(self, text: str, vocab: Dict[str, int]) -> List[str]:
        """
        使用Viterbi算法进行分词
        
        Args:
            text: 输入文本
            vocab: 词汇表
        
        Returns:
            best_segmentation: 最佳分词结果
        """
        n = len(text)
        
        # 初始化最佳得分和路径
        best_scores = [float('-inf')] * (n + 1)
        best_scores[0] = 0
        best_edges = [None] * (n + 1)
        
        # Viterbi前向计算
        for start in range(n):
            if best_scores[start] == float('-inf'):
                continue
                
            for end in range(start + 1, min(start + 20, n + 1)):  # 限制最大token长度
                token = text[start:end]
                if token in vocab:
                    score = best_scores[start] + math.log(vocab[token])  # 使用频率作为得分
                    if score > best_scores[end]:
                        best_scores[end] = score
                        best_edges[end] = start
        
        # 如果无法完全分词，使用贪心方法处理未覆盖部分
        if best_scores[n] == float('-inf'):
            return self._greedy_segment(text, vocab)
        
        # 回溯构建分词结果
        best_segmentation = []
        end = n
        while end > 0:
            start = best_edges[end]
            token = text[start:end]
            best_segmentation.append(token)
            end = start
        
        return best_segmentation[::-1]  # 反转以获得正确顺序
    
    def _greedy_segment(self, text: str, vocab: Dict[str, int]) -> List[str]:
        """
        贪心分词算法，当Viterbi失败时使用
        
        Args:
            text: 输入文本
            vocab: 词汇表
        
        Returns:
            tokens: 分词结果
        """
        tokens = []
        i = 0
        n = len(text)
        
        while i < n:
            # 尝试最长匹配
            found = False
            for j in range(min(20, n - i), 0, -1):  # 限制最大token长度为20
                token = text[i:i+j]
                if token in vocab:
                    tokens.append(token)
                    i += j
                    found = True
                    break
            
            # 如果没有匹配，添加单个字符并标记为未知
            if not found:
                tokens.append(self.unk_token)
                i += 1
        
        return tokens
    
    def _compute_unigram_model(self, token_freq: Dict[str, int]) -> Dict[str, float]:
        """
        计算Unigram模型概率
        
        Args:
            token_freq: token频率表
        
        Returns:
            token_scores: token的对数概率
        """
        total_count = sum(token_freq.values())
        token_scores = {}
        
        for token, count in token_freq.items():
            # 计算对数概率
            token_scores[token] = math.log(count / total_count)
        
        return token_scores
    
    def _prune_vocab(self, token_freq: Dict[str, int], token_scores: Dict[str, float]) -> Dict[str, float]:
        """
        剪枝词汇表，保留最重要的tokens
        
        Args:
            token_freq: token频率
            token_scores: token得分
        
        Returns:
            pruned_scores: 剪枝后的token得分
        """
        # 保留特殊标记
        pruned_scores = {token: token_scores[token] for token in self.special_tokens if token in token_scores}
        
        # 按概率排序
        sorted_items = sorted(
            [(t, s) for t, s in token_scores.items() if t not in self.special_tokens],
            key=lambda x: (x[1], token_freq.get(x[0], 0)),
            reverse=True
        )
        
        # 保留前vocab_size个
        remaining_slots = self.vocab_size - len(pruned_scores)
        for token, score in sorted_items[:remaining_slots]:
            pruned_scores[token] = score
        
        return pruned_scores
    
    def train(self, corpus: List[str]) -> None:
        """
        训练分词器
        
        Args:
            corpus: 文本语料库
        """
        print("开始训练Unigram分词器...")
        
        # 1. 获取语料库统计信息
        character_freq, charset = self._get_corpus_stats(corpus)
        print(f"语料库中包含 {len(charset)} 个不同字符")
        
        # 2. 初始化种子词汇表
        seed_vocab = self._initialize_seed_vocab(charset)
        print(f"初始种子词汇表大小: {len(seed_vocab)}")
        
        # 3. 扩展词汇表
        expanded_vocab = self._expand_vocab(corpus, seed_vocab)
        print(f"扩展后的词汇表大小: {len(expanded_vocab)}")
        
        # 4. 计算token频率
        token_freq = self._compute_token_frequency(corpus, expanded_vocab)
        
        # 5. 计算Unigram模型
        token_scores = self._compute_unigram_model(token_freq)
        
        # 6. 剪枝词汇表
        final_scores = self._prune_vocab(token_freq, token_scores)
        print(f"最终词汇表大小: {len(final_scores)}")
        
        # 7. 构建映射
        self.token_to_score = final_scores
        self.token_to_id = {token: idx for idx, token in enumerate(final_scores.keys())}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        
        # 完成训练
        self.trained = True
        print("Unigram分词器训练完成!")
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        将文本编码为token ID
        
        Args:
            text: 输入文本
            add_special_tokens: 是否添加特殊标记
        
        Returns:
            token_ids: token ID列表
        """
        if not self.trained:
            raise ValueError("模型尚未训练，请先调用train()方法")
        
        tokens = self._tokenize(text)
        
        if add_special_tokens:
            tokens = [self.bos_token] + tokens + [self.eos_token]
        
        # 将tokens转换为ids
        token_ids = [self.token_to_id.get(token, self.token_to_id[self.unk_token]) for token in tokens]
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        将token ID解码为文本
        
        Args:
            token_ids: token ID列表
            skip_special_tokens: 是否跳过特殊标记
        
        Returns:
            text: 解码后的文本
        """
        if not self.trained:
            raise ValueError("模型尚未训练，请先调用train()方法")
        
        tokens = [self.id_to_token[token_id] for token_id in token_ids if token_id in self.id_to_token]
        
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in self.special_tokens]
        
        # 简单连接所有tokens (这是Unigram模型的特性)
        text = ''.join(tokens)
        
        return text
    
    def _tokenize(self, text: str) -> List[str]:
        """
        将文本分词为tokens
        
        Args:
            text: 输入文本
        
        Returns:
            tokens: 分词结果
        """
        # 使用Viterbi算法基于学习到的unigram模型进行分词
        return self._viterbi_segment_with_model(text)
    
    def _viterbi_segment_with_model(self, text: str) -> List[str]:
        """
        使用训练好的Unigram模型进行Viterbi分词
        
        Args:
            text: 输入文本
        
        Returns:
            best_segmentation: 最佳分词结果
        """
        n = len(text)
        
        # 初始化最佳得分和路径
        best_scores = [float('-inf')] * (n + 1)
        best_scores[0] = 0
        best_edges = [None] * (n + 1)
        
        # Viterbi前向计算
        for start in range(n):
            if best_scores[start] == float('-inf'):
                continue
                
            for end in range(start + 1, min(start + 20, n + 1)):  # 限制最大token长度
                token = text[start:end]
                if token in self.token_to_score:
                    score = best_scores[start] + self.token_to_score[token]
                    if score > best_scores[end]:
                        best_scores[end] = score
                        best_edges[end] = start
        
        # 如果无法完全分词，使用字符级分词
        if best_scores[n] == float('-inf'):
            return self._character_level_fallback(text)
        
        # 回溯构建分词结果
        best_segmentation = []
        end = n
        while end > 0:
            start = best_edges[end]
            token = text[start:end]
            best_segmentation.append(token)
            end = start
        
        return best_segmentation[::-1]  # 反转以获得正确顺序
    
    def _character_level_fallback(self, text: str) -> List[str]:
        """
        当Viterbi算法失败时，退化为字符级分词
        
        Args:
            text: 输入文本
        
        Returns:
            tokens: 分词结果
        """
        tokens = []
        for char in text:
            if char in self.token_to_score:
                tokens.append(char)
            else:
                tokens.append(self.unk_token)
        
        return tokens
    
    def save(self, path: str) -> None:
        """
        保存模型到文件
        
        Args:
            path: 文件路径
        """
        if not self.trained:
            raise ValueError("模型尚未训练，请先调用train()方法")
        
        import json
        
        model_data = {
            "vocab_size": self.vocab_size,
            "character_coverage": self.character_coverage,
            "special_tokens": {
                "unk_token": self.unk_token,
                "bos_token": self.bos_token,
                "eos_token": self.eos_token,
                "pad_token": self.pad_token,
            },
            "token_to_score": self.token_to_score,
            "token_to_id": self.token_to_id,
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)
        
        print(f"模型已保存至 {path}")
    
    @classmethod
    def load(cls, path: str) -> 'UnigramTokenizer':
        """
        从文件加载模型
        
        Args:
            path: 文件路径
        
        Returns:
            tokenizer: 加载的分词器
        """
        import json
        
        with open(path, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
        
        # 创建tokenizer实例
        tokenizer = cls(
            vocab_size=model_data["vocab_size"],
            character_coverage=model_data["character_coverage"],
            unk_token=model_data["special_tokens"]["unk_token"],
            bos_token=model_data["special_tokens"]["bos_token"],
            eos_token=model_data["special_tokens"]["eos_token"],
            pad_token=model_data["special_tokens"]["pad_token"],
        )
        
        # 恢复模型状态
        tokenizer.token_to_score = model_data["token_to_score"]
        tokenizer.token_to_id = {k: int(v) for k, v in model_data["token_to_id"].items()}
        tokenizer.id_to_token = {int(v): k for k, v in tokenizer.token_to_id.items()}
        tokenizer.trained = True
        
        print(f"从 {path} 加载模型成功")
        return tokenizer


# 使用示例
def main():
    # 示例语料库
    corpus = [
        "这是一个简单的例子。",
        "我们正在实现一个Unigram Language Model。",
        "SentencePiece是一种无监督的文本分词器。",
        "它被用于BERT、ALBERT和T5等模型中。",
        "Unigram模型是一种基于统计的分词算法。",
        "它通过最大化语料库的似然来学习子词单元。",
        "HuggingFace提供了许多预训练模型和工具。",
        "Python是一种流行的编程语言。",
    ]
    
    # 创建并训练分词器
    tokenizer = UnigramTokenizer(vocab_size=100)
    tokenizer.train(corpus)
    
    # 测试分词器
    test_text = "这是一个新的Unigram分词器例子"
    token_ids = tokenizer.encode(test_text)
    decoded_text = tokenizer.decode(token_ids)
    
    print("\n测试结果:")
    print(f"原始文本: {test_text}")
    print(f"Token IDs: {token_ids}")
    print(f"解码文本: {decoded_text}")
    
    # 保存和加载测试
    tokenizer.save("unigram_tokenizer.json")
    loaded_tokenizer = UnigramTokenizer.load("unigram_tokenizer.json")
    
    # 使用加载的分词器测试
    token_ids_loaded = loaded_tokenizer.encode(test_text)
    decoded_text_loaded = loaded_tokenizer.decode(token_ids_loaded)
    
    print("\n加载模型测试结果:")
    print(f"Token IDs: {token_ids_loaded}")
    print(f"解码文本: {decoded_text_loaded}")


if __name__ == "__main__":
    main()