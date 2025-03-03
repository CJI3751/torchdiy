這段程式碼手動實作了 **LogSumExp (LSE) 運算** 和 **交叉熵損失函數 (CrossEntropyLoss)**，這些數學原理在深度學習中非常重要，特別是在分類問題中。

---

## **1. LogSumExp (LSE) 的數學原理**
在許多機器學習應用中，我們經常需要計算：
\[
\log \sum_{i} e^{x_i}
\]
這個運算經常出現在 **Softmax** 計算過程中，但直接計算可能會導致數值溢出 (overflow)，因為指數函數 \( e^{x} \) 可能變得非常大。因此，通常會用以下的數學技巧來避免這個問題：

### **數值穩定的 LogSumExp**
\[
\log \sum_{i} e^{x_i} = m + \log \sum_{i} e^{x_i - m}
\]
其中：
\[
m = \max(x)
\]
這樣做的好處是：
- \( x_i - m \) 會變成較小的數值，避免 \( e^{x_i} \) 變得太大而溢出。
- \( e^{x_i - m} \) 仍然保持了相對關係，因此計算結果不變。

### **對應的 Python 程式碼**
```python
m = torch.max(x, dim=dim, keepdim=True)
result = torch.log(torch.sum(torch.exp(x - m), dim=dim, keepdim=keepdim)) + m
```
這段程式碼使用了 `torch.max(x, dim=dim, keepdim=True)` 來取得最大值 \( m \)，然後執行 `x - m` 來避免數值問題。

---

## **2. 交叉熵損失 (Cross Entropy Loss)**
交叉熵損失 (Cross-Entropy Loss) 是 **分類問題** 中最常用的損失函數，它衡量模型的預測分佈與真實標籤分佈之間的差異。

### **數學公式**
\[
\text{CrossEntropyLoss} = -\sum_{i} y_i \log \hat{y}_i
\]
其中：
- \( y_i \) 是 one-hot 編碼的目標標籤。
- \( \hat{y}_i \) 是模型的 **Softmax 機率**。

由於 `torch.nn.CrossEntropyLoss` 會自動對 logits (未經 softmax 的輸出) 應用 LogSoftmax，因此我們可以直接計算：
\[
\log P(y = c) = \log \frac{e^{z_c}}{\sum_j e^{z_j}}
\]
\[
= z_c - \logsumexp(z)
\]
這就是為什麼程式碼中有：
```python
log_softmax = logits - logsumexp(logits, dim=1, keepdim=True)
```
這相當於計算：
\[
\log P(y = c)
\]

---

### **3. 計算 NLLLoss (Negative Log Likelihood Loss)**
我們需要計算目標類別的負對數機率：
\[
\text{NLLLoss} = -\log P(y = c)
\]
在程式碼中，使用 `gather()` 來選取目標類別對應的 log_softmax：
```python
nll_loss = -log_softmax.gather(1, target.unsqueeze(1)).mean()
```
這相當於：
- `target.unsqueeze(1)`: 讓 `target` 變成 (batch_size, 1) 形狀，使 `gather()` 可以正確索引。
- `log_softmax.gather(1, target.unsqueeze(1))`: 取出 `log_softmax` 中 **對應目標類別的 log 機率**。
- `mean()`: 對 batch 內的所有樣本取平均。

---

## **3. 總結**
這段程式碼實作了：
1. **LogSumExp (LSE) 運算**：
   - 主要用來計算數值穩定的 softmax。
2. **手寫 CrossEntropyLoss**：
   - 先計算 LogSoftmax (避免 softmax 數值問題)。
   - 使用 `gather()` 取得目標類別的 log 機率，計算 **負對數似然損失**。

這與 `torch.nn.CrossEntropyLoss` 的行為完全相同，適用於多分類問題！🚀