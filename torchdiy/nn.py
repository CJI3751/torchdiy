import torch
import torch.nn as nn

from . import linear
from . import activate
from . import loss
from . import cnn
from . import dropout
# from . import rnn_deepseek
from . import rnn
from . import gru
from . import attention
from . import transformer
from . import embed

functional = nn.functional

Module = nn.Module
Dropout = dropout.Dropout # nn.Dropout

Linear = linear.Linear
ReLU = activate.ReLU
CrossEntropyLoss = loss.CrossEntropyLoss

MaxPool2d = cnn.MaxPool2d
Conv2d = nn.Conv2d # 速度的問題，使用 Conv2d = cnn.Conv2d 會變得很慢，所以還是維持用 nn.Conv2d

# RNN = nn.RNN
RNN = rnn.RNN
GRU = rnn.GRU
# GRU = gru.GRU
LSTM = nn.LSTM
# Embedding = nn.Embedding
Embedding = embed.Embedding

MultiheadAttention = attention.MultiheadAttention
TransformerDecoderLayer = transformer.TransformerDecoderLayer
TransformerDecoder = transformer.TransformerDecoder
