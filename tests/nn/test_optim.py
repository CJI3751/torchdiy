import pytest
import torch
import torchdiy

# 定義一個 fixture 來共享輸入數據
# @pytest.fixture
def model_custom_builtin():
    model_custom = torch.nn.Linear(5, 1)
    model_builtin = torch.nn.Linear(5, 1)

    # 確保兩個模型使用相同的初始權重
    with torch.no_grad():
        for param_custom, param_builtin in zip(model_custom.parameters(), model_builtin.parameters()):
            param_builtin.copy_(param_custom)
    return model_custom, model_builtin

def optimizer_test(optimizer_custom, optimizer_builtin, model_custom, model_builtin):
    # 創建自定義的 SGD 優化器
    # optimizer_custom = optimizer_custom_class(model_custom.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    # 創建 PyTorch 內建的 SGD 優化器
    # optimizer_builtin = optimizer_builtin_class(model_builtin.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    # 創建一個簡單的損失函數
    criterion = torch.nn.MSELoss()

    # 測試數據
    inputs = torch.randn(10, 5)  # 輸入數據，形狀為 (batch_size, input_dim)
    targets = torch.randn(10, 1)  # 目標數據，形狀為 (batch_size, output_dim)

    # 前向傳播
    outputs_custom = model_custom(inputs)
    outputs_builtin = model_builtin(inputs)

    # 計算損失
    loss_custom = criterion(outputs_custom, targets)
    loss_builtin = criterion(outputs_builtin, targets)

    # 反向傳播
    loss_custom.backward()
    loss_builtin.backward()

    # 更新參數
    optimizer_custom.step()
    optimizer_builtin.step()

    # 比較兩個模型的參數是否一致
    for param_custom, param_builtin in zip(model_custom.parameters(), model_builtin.parameters()):
        print("自定義 SGD 更新後的參數:", param_custom)
        print("PyTorch 內建 SGD 更新後的參數:", param_builtin)
        print("結果是否一致:", torch.allclose(param_custom, param_builtin))
        assert torch.allclose(param_custom, param_builtin), "參數更新不一致"

    print("\n所有測試通過!")

def test_sgd():
        # 測試參數
    learning_rate = 0.01
    momentum = 0.9
    weight_decay = 0.0001
    model_custom, model_builtin = model_custom_builtin()

    optimizer_custom = torchdiy.optim.SGD(model_custom.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    optimizer_builtin = torch.optim.SGD(model_builtin.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    optimizer_test(optimizer_custom, optimizer_builtin, model_custom, model_builtin) 


def test_adam():
    model_custom, model_builtin = model_custom_builtin()

    optimizer_custom = torchdiy.optim.Adam(model_custom.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    optimizer_builtin = torch.optim.Adam(model_builtin.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    optimizer_test(optimizer_custom, optimizer_builtin, model_custom, model_builtin) 
