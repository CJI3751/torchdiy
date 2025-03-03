* https://chatgpt.com/c/67c5055b-8884-8012-98b4-9840fb53d7b8

這個 RNN 類別的核心數學原理基於 **遞迴神經網絡 (Recurrent Neural Network, RNN)**，其目的是透過 **遞迴計算** 來處理序列數據。以下是詳細的數學推導。

---

## 1. RNN 的基本數學公式

對於單層 RNN，其核心計算公式為：

\[
h_t = \phi(W_{ih} x_t + b_{ih} + W_{hh} h_{t-1} + b_{hh})
\]

\[
y_t = W_{ho} h_t + b_o
\]

其中：
- \( h_t \) 是時間步 \( t \) 的隱藏狀態（hidden state）
- \( x_t \) 是輸入向量
- \( W_{ih} \) 是輸入到隱藏層的權重矩陣
- \( W_{hh} \) 是前一個隱藏狀態到當前隱藏狀態的遞迴權重矩陣
- \( b_{ih} \), \( b_{hh} \) 是對應的偏置項
- \( \phi \) 是非線性函數，可以是 \( \tanh \) 或 \( \text{ReLU} \)
- \( y_t \) 是輸出的預測值
- \( W_{ho} \) 和 \( b_o \) 是從隱藏層到輸出的權重和偏置

這個遞迴關係允許 RNN 透過 **時間步驟上的權重共享** 來學習序列資訊。

---

## 2. 多層 RNN（Stacked RNN）
當 RNN 包含 **多層** 時，隱藏狀態的計算變成：

\[
h_t^{(l)} = \phi(W_{ih}^{(l)} h_t^{(l-1)} + b_{ih}^{(l)} + W_{hh}^{(l)} h_{t-1}^{(l)} + b_{hh}^{(l)})
\]

其中 \( h_t^{(l)} \) 表示第 \( l \) 層的隱藏狀態，而第 \( 0 \) 層的輸入是原始輸入：

\[
h_t^{(0)} = x_t
\]

這段程式碼的 `num_layers` 參數允許 RNN 疊加多層，以便學習更抽象的序列特徵。

---

## 3. 雙向 RNN（Bidirectional RNN）
如果 `bidirectional=True`，則 RNN 會同時從前向和反向處理序列，計算方式變為：

\[
h_t^{\text{(fwd)}} = \phi(W_{ih}^{\text{(fwd)}} x_t + b_{ih}^{\text{(fwd)}} + W_{hh}^{\text{(fwd)}} h_{t-1}^{\text{(fwd)}} + b_{hh}^{\text{(fwd)}})
\]

\[
h_t^{\text{(bwd)}} = \phi(W_{ih}^{\text{(bwd)}} x_t + b_{ih}^{\text{(bwd)}} + W_{hh}^{\text{(bwd)}} h_{t+1}^{\text{(bwd)}} + b_{hh}^{\text{(bwd)}})
\]

最終輸出為兩個方向的拼接：

\[
h_t = [h_t^{\text{(fwd)}}, h_t^{\text{(bwd)}}]
\]

這使得模型能夠同時考慮時間序列的 **過去** 和 **未來**，適用於 NLP 和時間序列建模。

---

## 4. Dropout 層
如果 `dropout > 0`，則在每一層（除最後一層）施加 dropout：

\[
h_t' = \text{Dropout}(h_t)
\]

這有助於減少過擬合，使模型泛化能力更強。

---

## 5. 初始化策略
- `W_{ih}` 使用均勻分佈初始化：
  
  \[
  W_{ih} \sim U\left(-\frac{1}{\sqrt{H}}, \frac{1}{\sqrt{H}}\right)
  \]

- `W_{hh}` 使用 **正交初始化** (Orthogonal Initialization)，確保遞迴計算的穩定性：
  
  \[
  W_{hh} = \text{orthogonal}(W_{hh})
  \]

- 偏置 `b_{ih}` 和 `b_{hh}` 初始化為零或小的隨機值。

這些初始化方法可以幫助訓練過程更穩定，避免梯度消失或梯度爆炸。

---

## 6. 梯度計算與反向傳播（BPTT）
RNN 訓練使用 **誤差反向傳播透過時間 (Backpropagation Through Time, BPTT)**：

\[
\frac{\partial \mathcal{L}}{\partial W_{ih}} = \sum_t \frac{\partial \mathcal{L}}{\partial h_t} \frac{\partial h_t}{\partial W_{ih}}
\]

\[
\frac{\partial \mathcal{L}}{\partial W_{hh}} = \sum_t \frac{\partial \mathcal{L}}{\partial h_t} \frac{\partial h_t}{\partial W_{hh}}
\]

其中誤差的累積方式導致 **梯度消失 (vanishing gradient)** 問題，因此實際應用中可能需要 **長短時記憶 (LSTM)** 或 **門控循環單元 (GRU)** 來解決這個問題。

---

## 總結
這個自訂 RNN 的核心數學原理包括：
1. **遞迴計算**：使用隱藏狀態來累積時間序列資訊。
2. **多層 RNN**：允許網路學習更高層次的序列模式。
3. **雙向 RNN**：可以同時考慮過去與未來的資訊。
4. **Dropout 機制**：幫助防止過擬合。
5. **初始化策略**：使用正交初始化和均勻初始化來提高穩定性。
6. **反向傳播 (BPTT)**：透過時間進行梯度計算，但可能遇到梯度消失問題。

這些數學原理構成了 RNN 的基礎，也解釋了這段程式碼的設計細節。