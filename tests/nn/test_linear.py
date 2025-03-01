import pytest
import torch
import torchdiy

# 定義一個 fixture 來共享輸入數據
@pytest.fixture
def input_tensor():
    return torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)

# 定義一個 fixture 來初始化 torchdiy.nn.Linear
@pytest.fixture
def torchdiy_linear():
    return torchdiy.nn.Linear(in_features=3, out_features=2)

# 定義一個 fixture 來初始化 PyTorch 的 nn.Linear
@pytest.fixture
def builtin_linear(torchdiy_linear):
    linear = torch.nn.Linear(in_features=3, out_features=2)
    # 將權重和偏置設置為與 torchdiy Linear 相同
    with torch.no_grad():
        linear.weight.copy_(torchdiy_linear.weight)
        linear.bias.copy_(torchdiy_linear.bias)
    return linear

# 測試 torchdiy.nn.Linear 的輸出是否與 PyTorch 的 nn.Linear 一致
def test_linear_output(input_tensor, torchdiy_linear, builtin_linear):
    torchdiy_output = torchdiy_linear(input_tensor)
    builtin_output = builtin_linear(input_tensor)
    assert torch.allclose(torchdiy_output, builtin_output), "輸出不一致"

# 測試 torchdiy.nn.Linear 的權重梯度是否與 PyTorch 的 nn.Linear 一致
def test_linear_weight_grad(input_tensor, torchdiy_linear, builtin_linear):
    # 計算 torchdiy Linear 的梯度
    torchdiy_output = torchdiy_linear(input_tensor)
    torchdiy_output.sum().backward()
    torchdiy_weight_grad = torchdiy_linear.weight.grad.clone()

    # 計算 PyTorch Linear 的梯度
    builtin_output = builtin_linear(input_tensor)
    builtin_output.sum().backward()
    builtin_weight_grad = builtin_linear.weight.grad.clone()

    assert torch.allclose(torchdiy_weight_grad, builtin_weight_grad), "權重梯度不一致"

# 測試 torchdiy.nn.Linear 的偏置梯度是否與 PyTorch 的 nn.Linear 一致
def test_linear_bias_grad(input_tensor, torchdiy_linear, builtin_linear):
    # 計算 torchdiy Linear 的梯度
    torchdiy_output = torchdiy_linear(input_tensor)
    torchdiy_output.sum().backward()
    torchdiy_bias_grad = torchdiy_linear.bias.grad.clone()

    # 計算 PyTorch Linear 的梯度
    builtin_output = builtin_linear(input_tensor)
    builtin_output.sum().backward()
    builtin_bias_grad = builtin_linear.bias.grad.clone()

    assert torch.allclose(torchdiy_bias_grad, builtin_bias_grad), "偏置梯度不一致"
