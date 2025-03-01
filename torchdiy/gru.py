import torch
import torch.nn as nn
import torch.nn.functional as F

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False):
        """
        模仿 torch.nn.GRU 的簡單實現。

        參數:
        - input_size: 輸入特徵的維度
        - hidden_size: 隱藏狀態的維度
        - num_layers: GRU 的層數（默認為 1）
        - batch_first: 如果為 True，輸入和輸出的形狀為 (batch_size, seq_len, input_size)
                      如果為 False，形狀為 (seq_len, batch_size, input_size)
        """
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        # 創建 GRU 的權重和偏置
        self.weight_ih = nn.ParameterList()  # 輸入到隱藏的權重
        self.weight_hh = nn.ParameterList()  # 隱藏到隱藏的權重
        self.bias_ih = nn.ParameterList()    # 輸入到隱藏的偏置
        self.bias_hh = nn.ParameterList()    # 隱藏到隱藏的偏置

        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            # 輸入到隱藏的權重 (3 * hidden_size, input_size)
            self.weight_ih.append(nn.Parameter(torch.randn(3 * hidden_size, layer_input_size)))
            # 隱藏到隱藏的權重 (3 * hidden_size, hidden_size)
            self.weight_hh.append(nn.Parameter(torch.randn(3 * hidden_size, hidden_size)))
            # 輸入到隱藏的偏置 (3 * hidden_size)
            self.bias_ih.append(nn.Parameter(torch.randn(3 * hidden_size)))
            # 隱藏到隱藏的偏置 (3 * hidden_size)
            self.bias_hh.append(nn.Parameter(torch.randn(3 * hidden_size)))

    def forward(self, x, h_0=None):
        """
        前向傳播。

        參數:
        - x: 輸入張量，形狀為 (seq_len, batch_size, input_size) 或 (batch_size, seq_len, input_size)
        - h_0: 初始隱藏狀態，形狀為 (num_layers, batch_size, hidden_size)

        返回:
        - output: 輸出張量，形狀為 (seq_len, batch_size, hidden_size) 或 (batch_size, seq_len, hidden_size)
        - h_n: 最後的隱藏狀態，形狀為 (num_layers, batch_size, hidden_size)
        """
        if self.batch_first:
            # 如果 batch_first=True，轉置輸入為 (seq_len, batch_size, input_size)
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.size()

        if h_0 is None:
            # 如果未提供初始隱藏狀態，初始化為零
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)

        # 初始化輸出和隱藏狀態
        output = []
        h_n = []

        for layer in range(self.num_layers):
            h = h_0[layer]  # 當前層的初始隱藏狀態
            layer_output = []

            for t in range(seq_len):
                # 當前時間步的輸入
                x_t = x[t]

                # 計算 GRU 的門控機制
                gates = torch.matmul(x_t, self.weight_ih[layer].t()) + self.bias_ih[layer] + \
                        torch.matmul(h, self.weight_hh[layer].t()) + self.bias_hh[layer]

                # 分割門控
                reset_gate = torch.sigmoid(gates[:self.hidden_size])
                update_gate = torch.sigmoid(gates[self.hidden_size:2 * self.hidden_size])
                new_gate = torch.tanh(gates[2 * self.hidden_size:])

                # 更新隱藏狀態
                h = (1 - update_gate) * h + update_gate * new_gate
                layer_output.append(h)

            # 將當前層的輸出和隱藏狀態保存
            output.append(torch.stack(layer_output, dim=0))
            h_n.append(h)

        # 將輸出和隱藏狀態堆疊
        output = torch.stack(output, dim=0)
        h_n = torch.stack(h_n, dim=0)

        if self.batch_first:
            # 如果 batch_first=True，轉置輸出為 (batch_size, seq_len, hidden_size)
            output = output.transpose(0, 1)

        return output, h_n
