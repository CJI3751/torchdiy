* https://chatgpt.com/c/67c5055b-8884-8012-98b4-9840fb53d7b8


這段程式碼實作了一個 **2D 卷積層 (Conv2d)**，其數學原理來自於 **捲積運算 (Convolution Operation)**，這是 CNN (卷積神經網路) 的核心運算之一。  

---

## **1. 2D 卷積的數學原理**  
在 2D 卷積層中，輸入影像 \( X \) 會與 **可學習的卷積核 (Kernel, Filter) \( W \)** 進行卷積，數學公式如下：

\[
Y_{m,n}^{(k)} = \sum_{c=0}^{C-1} \sum_{i=0}^{H_k-1} \sum_{j=0}^{W_k-1} W^{(k)}_{c,i,j} \cdot X_{c, (m+i), (n+j)} + b^{(k)}
\]

其中：
- \( X \in \mathbb{R}^{C \times H \times W} \) 是輸入張量 (通道數 \( C \)，高度 \( H \)，寬度 \( W \))。
- \( W^{(k)} \in \mathbb{R}^{C \times H_k \times W_k} \) 是第 \( k \) 個輸出通道的卷積核。
- \( H_k, W_k \) 是卷積核的大小。
- \( b^{(k)} \) 是第 \( k \) 個輸出通道的偏置 (如果有的話)。
- \( Y_{m,n}^{(k)} \) 是卷積後輸出張量的第 \( k \) 個通道，在位置 \( (m,n) \) 的值。

這對應到程式碼的這部分：
```python
output[:, k, i, j] = torch.sum(window * self.weight[k], dim=(1, 2, 3))
if self.use_bias:
    output[:, k, i, j] += self.bias[k]
```
這表示：
1. **提取輸入 \( X \) 的一個局部區域 (窗口)**，並與卷積核進行逐元素相乘。
2. **將結果加總，並加上偏置 \( b_k \)**。

---

## **2. 計算輸出尺寸**
程式碼中，輸出特徵圖的大小是這樣計算的：
```python
out_height = (height + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
out_width = (width + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
```
這對應數學公式：
\[
H_{\text{out}} = \frac{H_{\text{in}} + 2P - H_k}{S} + 1
\]
\[
W_{\text{out}} = \frac{W_{\text{in}} + 2P - W_k}{S} + 1
\]
其中：
- \( P \) 是填充 (Padding) 的大小。
- \( S \) 是步幅 (Stride)，決定每次滑動的距離。

---

## **3. Padding (填充)**
卷積過程可能會導致輸出尺寸變小，為了保持輸入輸出大小相近，可以使用 **零填充 (Zero Padding)**。在程式碼中，這部分是：
```python
if self.padding[0] > 0 or self.padding[1] > 0:
    x = F.pad(x, (self.padding[1], self.padding[1], self.padding[0], self.padding[0]))
```
這表示：
- 在 \( H \) 和 \( W \) 的前後各填充 \( P \) 個 0，使輸入大小增加。

若希望輸出大小與輸入相同 (Same Padding)，則需要：
\[
P = \frac{(H_k - 1)}{2}
\]
這樣：
\[
H_{\text{out}} = H_{\text{in}}
\]

---

## **4. Stride (步幅)**
Stride 控制卷積核每次移動的距離，當 \( S > 1 \) 時，輸出尺寸會變小：
- \( S = 1 \) 表示卷積核每次移動 1 個像素，產生密集特徵圖。
- \( S = 2 \) 表示卷積核每次跳 2 個像素，類似於降採樣 (Downsampling)。

在程式碼中，這部分對應：
```python
h_start = i * self.stride[0]
h_end = h_start + self.kernel_size[0]
w_start = j * self.stride[1]
w_end = w_start + self.kernel_size[1]
```
這確保卷積核按照步幅移動。

---

## **5. 總結**
這段程式碼實作了 **手寫 2D 卷積層**，模擬了 PyTorch `nn.Conv2d` 的功能，包含：
- **捲積運算**：局部加權求和 (window * kernel)
- **輸出尺寸計算**
- **Padding 機制**
- **Stride 控制**

這是 CNN 的核心運算，廣泛應用於影像辨識、特徵擷取等深度學習任務！ 🚀