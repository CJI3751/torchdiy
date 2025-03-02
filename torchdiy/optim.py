import torch
import torch.optim as optim
from torch.optim import Optimizer

RMSprop = optim.RMSprop
Adadelta = optim.Adadelta
Adagrad = optim.Adagrad
AdamW = optim.AdamW
SparseAdam = optim.SparseAdam
Adamax = optim.Adamax
ASGD = optim.ASGD
LBFGS = optim.LBFGS
Rprop = optim.Rprop

# SGD = optim.SGD
class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        """
        模仿 torch.optim.SGD 的簡單實現。

        參數:
        - params: 模型參數（可迭代的參數或參數組）
        - lr: 學習率（默認為 0.01）
        - momentum: 動量因子（默認為 0）
        - dampening: 動量的阻尼因子（默認為 0）
        - weight_decay: 權重衰減（L2 正則化，默認為 0）
        - nesterov: 是否使用 Nesterov 動量（默認為 False）
        """
        if lr < 0.0:
            raise ValueError(f"學習率必須為非負數，但得到 {lr}")
        if momentum < 0.0:
            raise ValueError(f"動量必須為非負數，但得到 {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"權重衰減必須為非負數，但得到 {weight_decay}")

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                       weight_decay=weight_decay, nesterov=nesterov)
        super(CustomSGD, self).__init__(params, defaults)

        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov 動量需要動量大於 0 且無阻尼")

    def __setstate__(self, state):
        super(CustomSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """
        執行單步優化。

        參數:
        - closure: 一個可選的閉包，用於重新計算損失（例如在計算梯度時）
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            dampening = group['dampening']
            weight_decay = group['weight_decay']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue

                # 應用權重衰減（L2 正則化）
                if weight_decay != 0:
                    p.grad.data.add_(p.data, alpha=weight_decay)

                # 更新動量
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(p.grad.data).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(p.grad.data, alpha=1 - dampening)

                    if nesterov:
                        # Nesterov 動量
                        p.grad.data = p.grad.data.add(buf, alpha=momentum)
                    else:
                        p.grad.data = buf

                # 更新參數
                p.data.add_(p.grad.data, alpha=-lr)

        return loss

# Adam = optim.Adam
class Adam(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        """
        模仿 torch.optim.Adam 的簡單實現。

        參數:
        - params: 模型參數（可迭代的參數或參數組）
        - lr: 學習率（默認為 0.001）
        - betas: 用於計算梯度及其平方的移動平均的係數（默認為 (0.9, 0.999)）
        - eps: 數值穩定性常數（默認為 1e-8）
        - weight_decay: 權重衰減（L2 正則化，默認為 0）
        """
        if lr < 0.0:
            raise ValueError(f"學習率必須為非負數，但得到 {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"beta1 必須在 [0, 1) 之間，但得到 {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"beta2 必須在 [0, 1) 之間，但得到 {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"eps 必須為非負數，但得到 {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"權重衰減必須為非負數，但得到 {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(Adam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)

    def step(self, closure=None):
        """
        執行單步優化。

        參數:
        - closure: 一個可選的閉包，用於重新計算損失（例如在計算梯度時）
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                # 應用權重衰減（L2 正則化）
                if weight_decay != 0:
                    p.grad.data.add_(p.data, alpha=weight_decay)

                # 獲取狀態
                state = self.state[p]

                # 初始化狀態
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                # 更新狀態
                state['step'] += 1
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                # 更新梯度的一階和二階動量
                exp_avg.mul_(beta1).add_(p.grad.data, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(p.grad.data, p.grad.data, value=1 - beta2)

                # 計算偏差校正
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # 計算更新值
                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)
                step_size = lr / bias_correction1

                # 更新參數
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss