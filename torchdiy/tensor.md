https://chatgpt.com/c/67c6599a-a274-8012-80d4-bc3599096505

這段程式碼實作了一個簡單的自動微分 (Autograd) 引擎，允許對張量 (Tensor) 進行數值運算並支援反向傳播 (Backward Propagation) 來計算梯度。其背後的數學主要涉及 **微分、鏈式法則 (Chain Rule) 以及常見的數學運算的梯度計算**。我們逐步分析其數學背景。

---

## **1. 加法 (Addition)**
```python
def __add__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(np.zeros(self.shape)+other)
    out = Tensor(self.data + other.data, (self, other), '+')

    def _backward():
        self.grad += out.grad
        other.grad += out.grad
    out._backward = _backward

    return out
```
### **數學運算**
\[
z = x + y
\]
其對 \( x \) 和 \( y \) 的偏導數為：
\[
\frac{\partial z}{\partial x} = 1, \quad \frac{\partial z}{\partial y} = 1
\]
所以在 `_backward` 中，反向傳播時直接將 `out.grad` 加到 `self.grad` 和 `other.grad`，即：
\[
\text{self.grad} += \text{out.grad}
\]
\[
\text{other.grad} += \text{out.grad}
\]

---

## **2. 乘法 (Multiplication)**
```python
def __mul__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(np.zeros(self.shape)+other)
    out = Tensor(self.data * other.data, (self, other), '*')

    def _backward():
        self.grad += other.data * out.grad
        other.grad += self.data * out.grad
    out._backward = _backward

    return out
```
### **數學運算**
\[
z = x \cdot y
\]
其對 \( x \) 和 \( y \) 的偏導數為：
\[
\frac{\partial z}{\partial x} = y, \quad \frac{\partial z}{\partial y} = x
\]
所以反向傳播時：
\[
\text{self.grad} += \text{other.data} \times \text{out.grad}
\]
\[
\text{other.grad} += \text{self.data} \times \text{out.grad}
\]

---

## **3. 指數函數 (Power)**
```python
def __pow__(self, other):
    assert isinstance(other, (int, float)), "only supporting int/float powers for now"
    out = Tensor(self.data**other, (self,), f'**{other}')

    def _backward():
        self.grad += (other * self.data**(other-1)) * out.grad
    out._backward = _backward

    return out
```
### **數學運算**
\[
z = x^c
\]
其對 \( x \) 的偏導數為：
\[
\frac{\partial z}{\partial x} = c \cdot x^{c-1}
\]
所以：
\[
\text{self.grad} += (c \cdot \text{self.data}^{c-1}) \times \text{out.grad}
\]

---

## **4. ReLU (Rectified Linear Unit)**
```python
def relu(self):
    out = Tensor(np.maximum(0, self.data), (self,), 'relu')

    def _backward():
        self.grad += (out.data > 0) * out.grad
    out._backward = _backward

    return out
```
### **數學運算**
\[
z = \max(0, x)
\]
其導數為：
\[
\frac{\partial z}{\partial x} =
\begin{cases}
1, & x > 0 \\
0, & x \leq 0
\end{cases}
\]
反向傳播時，只有當 \( x > 0 \) 時才讓梯度流回去：
\[
\text{self.grad} += (\text{out.data} > 0) \times \text{out.grad}
\]

---

## **5. 矩陣乘法 (Matrix Multiplication)**
```python
def matmul(self,other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(np.matmul(self.data , other.data), (self, other), 'matmul')

    def _backward():
        self.grad += np.dot(out.grad,other.data.T)
        other.grad += np.dot(self.data.T,out.grad)            
    out._backward = _backward

    return out
```
### **數學運算**
\[
Z = XW
\]
其中，\( X \) 為 \( m \times n \) 矩陣，\( W \) 為 \( n \times p \) 矩陣，則：
\[
\frac{\partial Z}{\partial X} = W^T, \quad \frac{\partial Z}{\partial W} = X^T
\]
所以：
\[
\text{self.grad} += \text{out.grad} \times W^T
\]
\[
\text{other.grad} += X^T \times \text{out.grad}
\]

---

## **6. Softmax**
```python
def softmax(self):
    out = Tensor(np.exp(self.data) / np.sum(np.exp(self.data), axis=1)[:, None], (self,), 'softmax')
    softmax = out.data

    def _backward():
        s = np.sum(out.grad * softmax, 1)
        t = np.reshape(s, [-1, 1])
        self.grad += (out.grad - t) * softmax
    out._backward = _backward

    return out
```
### **數學運算**
\[
s_i = \frac{e^{x_i}}{\sum_j e^{x_j}}
\]
導數：
\[
\frac{\partial s_i}{\partial x_j} =
\begin{cases}
s_i (1 - s_i), & i = j \\
- s_i s_j, & i \neq j
\end{cases}
\]
反向傳播計算：
\[
\text{self.grad} += (\text{out.grad} - t) \times \text{softmax}
\]

---

## **7. Cross-Entropy 損失**
```python
def cross_entropy(self, yb):
    log_probs = self.log()
    zb = yb * log_probs
    outb = zb.sum(axis=1)
    loss = -outb.sum()
    return loss
```
### **數學運算**
\[
L = -\sum p_i \log q_i
\]
這裡 \( p \) 是標籤 (one-hot)，\( q \) 是 Softmax 輸出，微分為：
\[
\frac{\partial L}{\partial q_i} = -\frac{p_i}{q_i}
\]

---

## **8. 反向傳播 (Backward Propagation)**
```python
def backward(self):
    topo = []
    visited = set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
    build_topo(self)

    self.grad = 1
    for v in reversed(topo):
        v._backward()
```
這裡使用 **拓撲排序 (Topological Sorting)** 來計算梯度，確保從最終輸出開始，按照依賴順序進行反向傳播。

---

## **總結**
這段程式碼實作了一個 **簡單的計算圖 (Computational Graph)**，並透過 **鏈式法則 (Chain Rule)** 進行自動微分。主要數學概念包括：
- **基本導數計算** (加法、乘法、指數)
- **非線性函數導數** (ReLU、Softmax)
- **矩陣微分** (MatMul)
- **損失函數導數** (Cross-Entropy)

這樣的框架可用來實作神經網路的梯度計算，是 PyTorch 等深度學習框架的核心機制。