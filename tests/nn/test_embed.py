import pytest
import torch
import torchdiy

def test_embedding():
    # 測試參數
    num_embeddings = 10  # 嵌入字典大小
    embedding_dim = 5    # 嵌入向量維度
    
    # 測試數據
    indices = torch.tensor([1, 2, 3, 0, 4])  # 輸入索引，形狀為 (batch_size,)
    
    # 使用手動實現的 Embedding
    embedding_custom = torchdiy.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
    
    # 使用 PyTorch 內建的 Embedding
    embedding_builtin = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
    
    # 確保兩個模組使用相同的權重
    with torch.no_grad():
        embedding_custom.weight.copy_(embedding_builtin.weight)
    
    # 前向傳播
    output_custom = embedding_custom(indices)
    output_builtin = embedding_builtin(indices)
    
    # 比較結果
    print("手動實現的 Embedding 輸出:", output_custom)
    print("PyTorch 內建的 Embedding 輸出:", output_builtin)
    print("結果是否一致:", torch.allclose(output_custom, output_builtin))
    assert torch.allclose(output_custom, output_builtin), "輸出不一致"
    
    # 測試二維輸入
    indices_2d = torch.tensor([[1, 2], [3, 4]])  # 形狀為 (batch_size, seq_length)
    
    output_custom_2d = embedding_custom(indices_2d)
    output_builtin_2d = embedding_builtin(indices_2d)
    
    print("\n二維輸入測試:")
    print("手動實現的 Embedding 輸出形狀:", output_custom_2d.shape)
    print("PyTorch 內建的 Embedding 輸出形狀:", output_builtin_2d.shape)
    print("結果是否一致:", torch.allclose(output_custom_2d, output_builtin_2d))
    assert torch.allclose(output_custom_2d, output_builtin_2d), "二維輸入的輸出不一致"
    
    # 測試 padding_idx 功能
    embedding_custom_pad = torchdiy.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim, padding_idx=0)
    embedding_builtin_pad = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim, padding_idx=0)
    
    # 確保兩個模組使用相同的權重
    with torch.no_grad():
        embedding_custom_pad.weight.copy_(embedding_builtin_pad.weight)
    
    # 包含 padding_idx 的輸入
    indices_with_pad = torch.tensor([0, 1, 0, 2])
    
    output_custom_pad = embedding_custom_pad(indices_with_pad)
    output_builtin_pad = embedding_builtin_pad(indices_with_pad)
    
    print("\npadding_idx 測試:")
    print("padding 位置的嵌入是否為零 (自定義):", torch.all(output_custom_pad[0] == 0) and torch.all(output_custom_pad[2] == 0))
    print("padding 位置的嵌入是否為零 (內建):", torch.all(output_builtin_pad[0] == 0) and torch.all(output_builtin_pad[2] == 0))
    print("結果是否一致:", torch.allclose(output_custom_pad, output_builtin_pad))
    assert torch.allclose(output_custom_pad, output_builtin_pad), "帶有 padding_idx 的輸出不一致"
    
    print("\n所有測試通過!")
