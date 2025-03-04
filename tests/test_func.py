import pytest
import torch
import torchdiy
import numpy as np

def test_tensor():
    custom = torch.zeros(2, 4)
    builtin = torchdiy.zeros(2, 4)
    assert np.allclose(custom.numpy(), builtin.numpy()), "zeros 輸出不一致"

    custom = torch.ones(2, 4)
    builtin = torchdiy.ones(2, 4)
    assert np.allclose(custom.numpy(), builtin.numpy()), "ones 輸出不一致"

    custom = torch.arange(1, 10, 1)
    builtin = torchdiy.arange(1, 10, 1)
    assert np.allclose(custom.numpy(), builtin.numpy()), "ones 輸出不一致"
