這段程式碼實作了 **Adam 優化器**，讓我們來解析它的數學原理及對應的程式碼部分。

---

## **1. Adam 優化器的公式**
Adam (**Adaptive Moment Estimation**) 是結合 **Momentum** 和 **RMSProp** 的優化方法：
- **Momentum** 提供一階動量來加速梯度下降
- **RMSProp** 控制學習率的自適應性

Adam 的更新規則如下：
\[
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
\]
\[
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
\]
其中：
- \( m_t \) 是一階動量（梯度的移動平均）
- \( v_t \) 是二階動量（梯度平方的移動平均）
- \( g_t \) 是當前梯度
- \( \beta_1, \beta_2 \) 是動量係數，通常設定為 \( \beta_1 = 0.9, \beta_2 = 0.999 \)

由於 \( m_t \) 和 \( v_t \) 初始值為 0，會有偏差，因此進行 **偏差校正**：
\[
\hat{m_t} = \frac{m_t}{1 - \beta_1^t}
\]
\[
\hat{v_t} = \frac{v_t}{1 - \beta_2^t}
\]
最終的參數更新為：
\[
\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v_t}} + \epsilon} \hat{m_t}
\]

---

## **2. 解析程式碼**
### **(1) 初始化**
```python
state['step'] = 0
state['exp_avg'] = torch.zeros_like(p.data)
state['exp_avg_sq'] = torch.zeros_like(p.data)
```
這段程式碼對應於：
- `exp_avg` (\( m_t \))：一階動量
- `exp_avg_sq` (\( v_t \))：二階動量
- `step` (\( t \))：計數器，紀錄優化步驟數

---

### **(2) 計算一階、二階動量**
```python
exp_avg.mul_(beta1).add_(p.grad.data, alpha=1 - beta1)
exp_avg_sq.mul_(beta2).addcmul_(p.grad.data, p.grad.data, value=1 - beta2)
```
這對應於：
\[
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
\]
\[
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
\]

---

### **(3) 偏差校正**
```python
bias_correction1 = 1 - beta1 ** state['step']
bias_correction2 = 1 - beta2 ** state['step']
```
這對應於：
\[
\hat{m_t} = \frac{m_t}{1 - \beta_1^t}
\]
\[
\hat{v_t} = \frac{v_t}{1 - \beta_2^t}
\]

---

### **(4) 計算更新步長**
```python
denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)
step_size = lr / bias_correction1
```
這對應於：
\[
\frac{\eta}{\sqrt{\hat{v_t}} + \epsilon}
\]

---

### **(5) 更新參數**
```python
p.data.addcdiv_(exp_avg, denom, value=-step_size)
```
這對應於：
\[
\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v_t}} + \epsilon} \hat{m_t}
\]

---

## **3. 總結**
這段程式碼完整實作了 Adam 優化器，並包含：
1. **梯度的一階動量**（Momentum）
2. **梯度平方的二階動量**（RMSProp）
3. **偏差修正**
4. **自適應學習率**
5. **L2 正則化（權重衰減）**

這使得 Adam 在處理 **稀疏梯度** 或 **非穩定梯度** 時表現優異，並且比傳統 SGD 更快收斂！🚀