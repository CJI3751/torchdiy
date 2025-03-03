這段程式碼手動實作了 **隨機梯度下降（SGD, Stochastic Gradient Descent）** 優化器，並且支援 **動量（Momentum）、Nesterov 動量** 和 **L2 正則化（權重衰減）**。我們來分析它背後的數學原理。

---

## **1. 基本 SGD 演算法**
最基本的 SGD 更新規則為：
\[
\theta \leftarrow \theta - \eta \nabla L(\theta)
\]
其中：
- \( \theta \) 是模型的參數
- \( \eta \) 是學習率（learning rate）
- \( \nabla L(\theta) \) 是對應的梯度

對應到程式碼：
```python
p.data.add_(p.grad.data, alpha=-lr)
```
這段程式碼等價於：
\[
p = p - \text{lr} \cdot p.\text{grad}
\]
這是最基本的 SGD 更新方式，僅根據當前梯度進行參數更新。

---

## **2. L2 正則化（權重衰減, Weight Decay）**
在許多深度學習模型中，會額外加入 L2 正則化：
\[
\theta \leftarrow \theta - \eta \left( \nabla L(\theta) + \lambda \theta \right)
\]
這相當於在損失函數中加入一個 L2 項：
\[
L'(\theta) = L(\theta) + \frac{\lambda}{2} \|\theta\|^2
\]
其中 \( \lambda \) 是權重衰減係數。

對應到程式碼：
```python
if weight_decay != 0:
    p.grad.data.add_(p.data, alpha=weight_decay)
```
這等價於：
\[
\text{grad} = \text{grad} + \lambda \theta
\]
這樣做可以防止參數變得過大，並有助於提升泛化能力。

---

## **3. 動量法（Momentum）**
SGD 在遇到 **高曲率（curvature 高）、鞍點（saddle point）或噪聲梯度** 時，可能會收斂得較慢。因此，可以加入 **動量（Momentum）**：
\[
v_t = \beta v_{t-1} + \nabla L(\theta)
\]
\[
\theta \leftarrow \theta - \eta v_t
\]
這裡：
- \( v_t \) 是動量
- \( \beta \) 是動量係數

對應到程式碼：
```python
if momentum != 0:
    param_state = self.state[p]
    if 'momentum_buffer' not in param_state:
        buf = param_state['momentum_buffer'] = torch.clone(p.grad.data).detach()
    else:
        buf = param_state['momentum_buffer']
        buf.mul_(momentum).add_(p.grad.data, alpha=1 - dampening)
```
- `buf.mul_(momentum)` 代表 \( v_t = \beta v_{t-1} \)
- `buf.add_(p.grad.data, alpha=1 - dampening)` 代表 \( v_t = \beta v_{t-1} + \nabla L(\theta) \)

然後，動量版本的梯度被用來更新參數：
```python
p.grad.data = buf
```
這樣梯度會帶有歷史資訊，使得更新方向更平滑，並有助於加速收斂。

---

## **4. Nesterov 動量（Nesterov Accelerated Gradient, NAG）**
Nesterov 動量的目標是預測下一步的更新方向，然後再計算梯度：
\[
v_t = \beta v_{t-1} + \nabla L(\theta - \eta \beta v_{t-1})
\]
\[
\theta \leftarrow \theta - \eta v_t
\]
這使得更新方向更準確，並通常能加速收斂。

對應到程式碼：
```python
if nesterov:
    p.grad.data = p.grad.data.add(buf, alpha=momentum)
```
這相當於：
\[
\nabla L(\theta - \eta \beta v)
\]
相較於標準動量，Nesterov 方法能夠 **更快地朝向最小值**，在許多深度學習應用中比普通動量表現更好。

---

## **5. 總結**
這段程式碼完整實作了：
1. **基本 SGD** 更新：
   \[
   \theta \leftarrow \theta - \eta \nabla L(\theta)
   \]
2. **L2 正則化（權重衰減）**：
   \[
   \theta \leftarrow \theta - \eta (\nabla L(\theta) + \lambda \theta)
   \]
3. **動量法（Momentum）**：
   \[
   v_t = \beta v_{t-1} + \nabla L(\theta), \quad \theta \leftarrow \theta - \eta v_t
   \]
4. **Nesterov 動量（NAG）**：
   \[
   v_t = \beta v_{t-1} + \nabla L(\theta - \eta \beta v_{t-1}), \quad \theta \leftarrow \theta - \eta v_t
   \]

這些技術都是為了加快收斂速度並提升模型的優化效果！🚀