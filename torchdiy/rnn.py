import torch
import torch.nn as nn
import math
from typing import Tuple, Optional, Union
from abc import ABC, abstractmethod

class RecurrentLayerBase(nn.Module, ABC):
    """
    循環層的抽象基類，定義通用接口
    
    參數:
        input_size: 輸入特徵維度
        hidden_size: 隱藏狀態維度
        bias: 是否使用偏置項
        batch_first: 如果為 True，則輸入和輸出張量的形狀為 (batch, seq, feature)
    """
    
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 bias: bool = True, 
                 batch_first: bool = False):
        super(RecurrentLayerBase, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.batch_first = batch_first
    
    @abstractmethod
    def forward(self, 
                input: torch.Tensor, 
                hx: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播
        
        參數:
            input: 輸入序列
            hx: 初始隱藏狀態
                
        返回:
            output: 每個時間步的輸出
            h_n: 最終隱藏狀態
        """
        pass


class RNNLayer(RecurrentLayerBase):
    """
    簡單 RNN 層實現
    
    參數:
        input_size: 輸入特徵維度
        hidden_size: 隱藏狀態維度
        nonlinearity: 非線性激活函數，可以是 'tanh' 或 'relu'
        bias: 是否使用偏置項
        batch_first: 如果為 True，則輸入和輸出張量的形狀為 (batch, seq, feature)
    """
    
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 nonlinearity: str = 'tanh', 
                 bias: bool = True, 
                 batch_first: bool = False):
        super(RNNLayer, self).__init__(input_size, hidden_size, bias, batch_first)
        
        self.nonlinearity = nonlinearity
        
        # 檢查參數有效性
        if nonlinearity not in ['tanh', 'relu']:
            raise ValueError(f"非線性函數必須是 'tanh' 或 'relu'，得到的是 {nonlinearity}")
        
        # 創建激活函數
        self.activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        
        # 定義網絡參數
        self.weight_ih = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        
        if bias:
            self.bias_ih = nn.Parameter(torch.Tensor(hidden_size))
            self.bias_hh = nn.Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        
        # 初始化參數
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化所有權重和偏置"""
        stdv = 1.0 / math.sqrt(self.hidden_size)
        
        nn.init.uniform_(self.weight_ih, -stdv, stdv)
        nn.init.orthogonal_(self.weight_hh)
            
        if self.bias:
            nn.init.uniform_(self.bias_ih, -stdv, stdv)
            nn.init.zeros_(self.bias_hh)
    
    def forward(self, 
                input: torch.Tensor, 
                hx: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播
        
        參數:
            input: 輸入序列。如果 batch_first=True 則形狀為 (batch, seq, feature)
                  否則形狀為 (seq, batch, feature)
            hx: 初始隱藏狀態。形狀為 (batch, hidden_size)
                如果未提供，則初始化為零
                
        返回:
            output: 每個時間步的輸出。如果 batch_first=True 則形狀為 (batch, seq, hidden_size)
                    否則形狀為 (seq, batch, hidden_size)
            h_n: 最終隱藏狀態。形狀為 (batch, hidden_size)
        """
        # 處理 batch_first
        if self.batch_first:
            input = input.transpose(0, 1)  # 變成 (seq, batch, feature)
            
        seq_len, batch_size, _ = input.size()
        
        # 如果沒有提供隱藏狀態，則初始化為零
        if hx is None:
            hx = torch.zeros(batch_size, 
                             self.hidden_size, 
                             dtype=input.dtype, 
                             device=input.device)
        
        # 存儲每個時間步的輸出
        outputs = []
        h_t = hx
        
        # 處理時間序列
        for t in range(seq_len):
            x_t = input[t]
            
            # RNN 單元計算
            gates = torch.mm(x_t, self.weight_ih.t())
            if self.bias_ih is not None:
                gates += self.bias_ih
                
            gates += torch.mm(h_t, self.weight_hh.t())
            if self.bias_hh is not None:
                gates += self.bias_hh
                
            h_t = self.activation(gates)
            outputs.append(h_t)
        
        # 將輸出堆疊成張量
        output = torch.stack(outputs, dim=0)
        
        # 如果 batch_first=True，則需要轉置輸出
        if self.batch_first:
            output = output.transpose(0, 1)
            
        return output, h_t
    
    def extra_repr(self) -> str:
        """返回實例的額外表示信息"""
        s = '{input_size}, {hidden_size}'
        if self.nonlinearity != 'tanh':
            s += ', nonlinearity={nonlinearity}'
        if not self.bias:
            s += ', bias={bias}'
        if self.batch_first:
            s += ', batch_first={batch_first}'
        return s.format(**self.__dict__)


class GRULayer(RecurrentLayerBase):
    """
    GRU 層實現
    
    參數:
        input_size: 輸入特徵維度
        hidden_size: 隱藏狀態維度
        bias: 是否使用偏置項
        batch_first: 如果為 True，則輸入和輸出張量的形狀為 (batch, seq, feature)
    """
    
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 bias: bool = True, 
                 batch_first: bool = False):
        super(GRULayer, self).__init__(input_size, hidden_size, bias, batch_first)
        
        # 定義網絡參數 - GRU 有三組門: reset, update, new
        self.weight_ih_r = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hh_r = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        
        self.weight_ih_z = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hh_z = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        
        self.weight_ih_n = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hh_n = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        
        if bias:
            self.bias_ih_r = nn.Parameter(torch.Tensor(hidden_size))
            self.bias_hh_r = nn.Parameter(torch.Tensor(hidden_size))
            
            self.bias_ih_z = nn.Parameter(torch.Tensor(hidden_size))
            self.bias_hh_z = nn.Parameter(torch.Tensor(hidden_size))
            
            self.bias_ih_n = nn.Parameter(torch.Tensor(hidden_size))
            self.bias_hh_n = nn.Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias_ih_r', None)
            self.register_parameter('bias_hh_r', None)
            self.register_parameter('bias_ih_z', None)
            self.register_parameter('bias_hh_z', None)
            self.register_parameter('bias_ih_n', None)
            self.register_parameter('bias_hh_n', None)
        
        # 初始化參數
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化所有權重和偏置"""
        stdv = 1.0 / math.sqrt(self.hidden_size)
        
        # 初始化所有權重
        for weight in [self.weight_ih_r, self.weight_ih_z, self.weight_ih_n]:
            nn.init.uniform_(weight, -stdv, stdv)
        
        for weight in [self.weight_hh_r, self.weight_hh_z, self.weight_hh_n]:
            nn.init.orthogonal_(weight)
            
        # 初始化所有偏置
        if self.bias:
            for bias in [self.bias_ih_r, self.bias_ih_z, self.bias_ih_n]:
                nn.init.uniform_(bias, -stdv, stdv)
            
            for bias in [self.bias_hh_r, self.bias_hh_z, self.bias_hh_n]:
                nn.init.zeros_(bias)
    
    def forward(self, 
                input: torch.Tensor, 
                hx: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播
        
        參數:
            input: 輸入序列。如果 batch_first=True 則形狀為 (batch, seq, feature)
                  否則形狀為 (seq, batch, feature)
            hx: 初始隱藏狀態。形狀為 (batch, hidden_size)
                如果未提供，則初始化為零
                
        返回:
            output: 每個時間步的輸出。如果 batch_first=True 則形狀為 (batch, seq, hidden_size)
                    否則形狀為 (seq, batch, hidden_size)
            h_n: 最終隱藏狀態。形狀為 (batch, hidden_size)
        """
        # 處理 batch_first
        if self.batch_first:
            input = input.transpose(0, 1)  # 變成 (seq, batch, feature)
            
        seq_len, batch_size, _ = input.size()
        
        # 如果沒有提供隱藏狀態，則初始化為零
        if hx is None:
            hx = torch.zeros(batch_size, 
                             self.hidden_size, 
                             dtype=input.dtype, 
                             device=input.device)
        
        # 存儲每個時間步的輸出
        outputs = []
        h_t = hx
        
        # 處理時間序列
        for t in range(seq_len):
            x_t = input[t]
            
            # 計算重置門 (reset gate)
            r_t = torch.mm(x_t, self.weight_ih_r.t())
            if self.bias_ih_r is not None:
                r_t += self.bias_ih_r
                
            r_t += torch.mm(h_t, self.weight_hh_r.t())
            if self.bias_hh_r is not None:
                r_t += self.bias_hh_r
                
            r_t = torch.sigmoid(r_t)
            
            # 計算更新門 (update gate)
            z_t = torch.mm(x_t, self.weight_ih_z.t())
            if self.bias_ih_z is not None:
                z_t += self.bias_ih_z
                
            z_t += torch.mm(h_t, self.weight_hh_z.t())
            if self.bias_hh_z is not None:
                z_t += self.bias_hh_z
                
            z_t = torch.sigmoid(z_t)
            
            # 計算候選隱藏狀態 (candidate hidden state)
            n_t = torch.mm(x_t, self.weight_ih_n.t())
            if self.bias_ih_n is not None:
                n_t += self.bias_ih_n
                
            n_t += torch.mm(r_t * h_t, self.weight_hh_n.t())
            if self.bias_hh_n is not None:
                n_t += self.bias_hh_n
                
            n_t = torch.tanh(n_t)
            
            # 計算新的隱藏狀態
            h_t = (1 - z_t) * n_t + z_t * h_t
            
            outputs.append(h_t)
        
        # 將輸出堆疊成張量
        output = torch.stack(outputs, dim=0)
        
        # 如果 batch_first=True，則需要轉置輸出
        if self.batch_first:
            output = output.transpose(0, 1)
            
        return output, h_t


class MultiLayerRNN(nn.Module):
    """
    多層可配置的循環神經網絡
    
    參數:
        input_size: 輸入特徵維度
        hidden_size: 隱藏狀態維度
        num_layers: RNN 層數
        rnn_type: 循環層類型，可以是 'rnn' 或 'gru'
        nonlinearity: 非線性激活函數，僅在 rnn_type='rnn' 時有效
        bias: 是否使用偏置項
        batch_first: 如果為 True，則輸入和輸出張量的形狀為 (batch, seq, feature)
        dropout: 如果非零，則在除最後一層外的每層輸出上引入一個 dropout 層
        bidirectional: 如果為 True，則變成雙向 RNN
    """
    
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 num_layers: int = 1, 
                 rnn_type: str = 'rnn',
                 nonlinearity: str = 'tanh', 
                 bias: bool = True, 
                 batch_first: bool = False, 
                 dropout: float = 0., 
                 bidirectional: bool = False):
        super(MultiLayerRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        self.nonlinearity = nonlinearity
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = float(dropout)
        self.bidirectional = bidirectional
        
        self.num_directions = 2 if bidirectional else 1
        
        # 檢查 rnn_type 有效性
        if rnn_type not in ['rnn', 'gru']:
            raise ValueError(f"rnn_type 必須是 'rnn' 或 'gru'，得到的是 {rnn_type}")
            
        if dropout > 0 and num_layers == 1:
            import warnings
            warnings.warn("dropout 參數 > 0，但 num_layers = 1，dropout 將被忽略")
        
        # 定義 dropout 層
        dropout_layer = nn.Dropout(dropout) if dropout > 0 else None
        
        # 創建 RNN 層
        self.rnn_layers = nn.ModuleList()
        
        for layer in range(num_layers):
            for direction in range(self.num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
                
                # 根據類型創建 RNN 層
                if rnn_type == 'rnn':
                    rnn_layer = RNNLayer(
                        input_size=layer_input_size,
                        hidden_size=hidden_size,
                        nonlinearity=nonlinearity,
                        bias=bias,
                        batch_first=False  # 在內部處理
                    )
                elif rnn_type == 'gru':
                    rnn_layer = GRULayer(
                        input_size=layer_input_size,
                        hidden_size=hidden_size,
                        bias=bias,
                        batch_first=False  # 在內部處理
                    )
                
                self.rnn_layers.append(rnn_layer)
        
        # 添加 dropout 層
        if dropout_layer is not None:
            self.dropout_layers = nn.ModuleList([dropout_layer for _ in range(num_layers - 1)])
        else:
            self.dropout_layers = None
    
    def forward(self, 
                input: torch.Tensor, 
                hx: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播
        
        參數:
            input: 輸入序列。如果 batch_first=True 則形狀為 (batch, seq, feature)
                  否則形狀為 (seq, batch, feature)
            hx: 初始隱藏狀態。形狀為 (num_layers * num_directions, batch, hidden_size)
                如果未提供，則初始化為零
                
        返回:
            output: 每個時間步的輸出。如果 batch_first=True 則形狀為 (batch, seq, hidden_size * num_directions)
                    否則形狀為 (seq, batch, hidden_size * num_directions)
            h_n: 最終隱藏狀態。形狀為 (num_layers * num_directions, batch, hidden_size)
        """
        # 處理 batch_first
        if self.batch_first:
            input = input.transpose(0, 1)  # 變成 (seq, batch, feature)
            
        seq_len, batch_size, _ = input.size()
        
        # 如果沒有提供隱藏狀態，則初始化為零
        if hx is None:
            hx = torch.zeros(self.num_layers * self.num_directions, 
                             batch_size, 
                             self.hidden_size, 
                             dtype=input.dtype, 
                             device=input.device)
        
        # 存儲每一層輸出的隱藏狀態
        h_n = torch.zeros_like(hx)
        
        # 將輸入按層處理
        layer_input = input
        
        for layer in range(self.num_layers):
            # 處理每個方向
            layer_outputs = []
            
            for direction in range(self.num_directions):
                # 獲取該層該方向的初始隱藏狀態
                idx = layer * self.num_directions + direction
                h_0 = hx[idx:idx+1].squeeze(0)
                
                # 獲取 RNN 層
                rnn_idx = idx
                rnn = self.rnn_layers[rnn_idx]
                
                # 如果是反向，需要反轉序列
                dir_input = layer_input
                if direction == 1:  # 反向處理
                    dir_input = torch.flip(dir_input, [0])
                
                # 前向傳播
                direction_output, h_t = rnn(dir_input, h_0)
                
                # 如果是反向，需要反轉輸出序列
                if direction == 1:
                    direction_output = torch.flip(direction_output, [0])
                
                # 儲存最終隱藏狀態
                h_n[idx] = h_t
                
                # 添加到層輸出
                layer_outputs.append(direction_output)
            
            # 將正向和反向輸出連接起來
            if self.num_directions == 2:
                layer_output = torch.cat(layer_outputs, dim=2)
            else:
                layer_output = layer_outputs[0]
            
            # 更新下一層的輸入
            layer_input = layer_output
            
            # 應用 dropout (除了最後一層)
            if self.dropout_layers is not None and layer < self.num_layers - 1:
                layer_input = self.dropout_layers[layer](layer_input)
        
        # 處理最終輸出
        output = layer_input
        
        # 如果 batch_first=True，則需要轉置輸出
        if self.batch_first:
            output = output.transpose(0, 1)
            
        return output, h_n
    
    def extra_repr(self) -> str:
        """返回實例的額外表示信息"""
        s = '{input_size}, {hidden_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        s += ', rnn_type={rnn_type}'
        if self.rnn_type == 'rnn' and self.nonlinearity != 'tanh':
            s += ', nonlinearity={nonlinearity}'
        if not self.bias:
            s += ', bias={bias}'
        if self.batch_first:
            s += ', batch_first={batch_first}'
        if self.dropout > 0:
            s += ', dropout={dropout}'
        if self.bidirectional:
            s += ', bidirectional={bidirectional}'
        return s.format(**self.__dict__)

class RNN(MultiLayerRNN):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 num_layers: int = 1, 
                 nonlinearity: str = 'tanh', 
                 bias: bool = True, 
                 batch_first: bool = False, 
                 dropout: float = 0., 
                 bidirectional: bool = False):
        super(MultiLayerRNN, self).__init__(input_size, hidden_size, num_layers, 'rnn', nonlinearity, bias, batch_first, dropout, bidirectional)

class RNN(MultiLayerRNN):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 num_layers: int = 1, 
                 nonlinearity: str = 'tanh', 
                 bias: bool = True, 
                 batch_first: bool = False, 
                 dropout: float = 0., 
                 bidirectional: bool = False):
        super().__init__(input_size, hidden_size, num_layers, 'rnn', nonlinearity, bias, batch_first, dropout, bidirectional)


class GRU(MultiLayerRNN):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 num_layers: int = 1, 
                 nonlinearity: str = 'tanh', 
                 bias: bool = True, 
                 batch_first: bool = False, 
                 dropout: float = 0., 
                 bidirectional: bool = False):
        super().__init__(input_size, hidden_size, num_layers, 'gru', nonlinearity, bias, batch_first, dropout, bidirectional)
