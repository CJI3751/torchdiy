from .tensor import Tensor
import numpy as np

def randn(shape, requires_grad=False):
    t = Tensor(shape, requires_grad=requires_grad)
    t.data = np.randn(*shape)
    return t

