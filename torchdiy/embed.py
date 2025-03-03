# 本程式由 ccc 指揮 claude 撰寫
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

def embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
    """
    模仿 torch.nn.functional.embedding 函數的簡單版本。

    參數:
    - input: 輸入的索引張量，形狀為 (batch_size, seq_len) 或 (seq_len,)
    - weight: 嵌入矩陣，形狀為 (num_embeddings, embedding_dim)
    - padding_idx: 可選，指定填充索引，對應的嵌入向量將被設為零
    - max_norm: 可選，對嵌入向量進行歸一化的最大範數
    - norm_type: 範數的類型，默認為 L2 範數
    - scale_grad_by_freq: 是否根據頻率縮放梯度（此版本未實現）
    - sparse: 是否使用稀疏梯度（此版本未實現）

    返回:
    - 提取的嵌入向量，形狀為 (batch_size, seq_len, embedding_dim) 或 (seq_len, embedding_dim)
    """
    # 確保輸入是 LongTensor
    if not isinstance(input, torch.LongTensor):
        input = input.long()

    # 根據索引從嵌入矩陣中提取對應的嵌入向量
    output = weight[input]

    # 處理 padding_idx
    if padding_idx is not None:
        # 將 padding_idx 對應的位置設為零
        mask = (input == padding_idx).unsqueeze(-1)
        output = output.masked_fill(mask, 0.0)

    # 處理 max_norm
    if max_norm is not None:
        # 對嵌入向量進行歸一化
        norms = output.norm(p=norm_type, dim=-1, keepdim=True)
        output = output * torch.clamp(max_norm / norms, max=1.0)

    return output

class Embedding(nn.Module):
    """
    一個與 PyTorch nn.Embedding 相容的自定義 Embedding 實現
    
    參數:
        num_embeddings: 嵌入字典的大小
        embedding_dim: 每個嵌入向量的大小
        padding_idx: 如果給定，在該索引處的嵌入向量填充為零
        max_norm: 如果給定，嵌入向量的範數將被限制為不超過此值
        norm_type: 用於計算嵌入範數的 p 範數類型
        scale_grad_by_freq: 如果為 True，根據詞頻縮放梯度
        sparse: 如果為 True，梯度將是稀疏的
    """
    
    def __init__(self, 
                 num_embeddings: int, # 例如：字典大小（詞彙數量）
                 embedding_dim: int,  # 例如：嵌入向量的維度
                 padding_idx: Optional[int] = None,
                 max_norm: Optional[float] = None,
                 norm_type: float = 2.0, 
                 scale_grad_by_freq: bool = False, 
                 sparse: bool = False):
        super(Embedding, self).__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        
        # 創建嵌入權重
        self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim)) # 例如：詞彙數量*嵌入向量的維度
        
        # 初始化權重
        self.reset_parameters()
        
        # 如果提供了 padding_idx，則將相應的嵌入向量設置為零
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)
    
    def reset_parameters(self):
        """初始化嵌入權重"""
        nn.init.normal_(self.weight)
        # 根據論文 "Efficient Estimation of Word Representations in Vector Space" 進行縮放
        self.weight.data.mul_(1.0 / math.sqrt(self.embedding_dim))
        
        # 如果提供了 padding_idx，則將相應的嵌入向量設置為零
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        前向傳播
        
        參數:
            input: 包含索引的 LongTensor
            
        返回:
            output: 嵌入向量的張量，形狀為 (..., embedding_dim)
        """
        # return F.embedding(
        return embedding(
            input, 
            self.weight, 
            self.padding_idx, 
            self.max_norm,
            self.norm_type, 
            self.scale_grad_by_freq, 
            self.sparse
        )
    
    def extra_repr(self) -> str:
        """返回實例的額外表示信息"""
        s = '{num_embeddings}, {embedding_dim}'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        if self.sparse is not False:
            s += ', sparse={sparse}'
        return s.format(**self.__dict__)