"""
from .tensor import Tensor
import numpy as np

def ones(shape, requires_grad=False):
    t = Tensor(shape, requires_grad=requires_grad)
    t.data = np.ones(shape)
    return t

def zeros(shape, requires_grad=False):
    t = Tensor(shape, requires_grad=requires_grad)
    t.data = np.zeros(shape)
    return t

def arange(start, end, step, requires_grad=False):
    data = np.arange(start, end, step)
    t = Tensor(data.shape, requires_grad=requires_grad)
    t.data = data
    return t

def linspace(start, end, steps, requires_grad=False):
    data = np.arange(start, end, steps)
    t = Tensor(data.shape, requires_grad=requires_grad)
    t.data = data
    return t

def rand(shape, requires_grad=False):
    t = Tensor(shape, requires_grad=requires_grad)
    t.data = np.rand(*shape)
    return t

def randn(shape, requires_grad=False):
    t = Tensor(shape, requires_grad=requires_grad)
    t.data = np.randn(*shape)
    return t

def sum(t, dim):
    return Tensor(numpy.sum(t.data, dim))

def mean(t, dim):
    return Tensor(numpy.mean(t.data, dim))

def std(t, dim):
    return Tensor(numpy.std(t.data, dim))
"""