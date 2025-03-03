https://chatgpt.com/c/67c58c5c-a578-8012-b5a5-a017baac46e7

這段程式碼是 GPT-2 的簡化實作，主要包含多層注意力機制（multi-head attention）、前饋神經網路（feedforward network）、層正規化（layer normalization）、殘差連接（residual connection）以及因果注意力遮罩（causal attention mask）。以下是這段程式背後的數學原理。

---

## **1. 內嵌層（Embedding Layer）**
GPT-2 需要將輸入的 token 轉換為向量表示。這透過詞嵌入（word embedding）和位置嵌入（position embedding）來完成。

令：
- \( x \) 為輸入 token ID（形狀為 \([B, T]\)，其中 \( B \) 是 batch size，\( T \) 是序列長度）。
- 詞嵌入矩陣 \( W_e \) 為 \(\mathbb{R}^{V \times d}\)（詞彙表大小 \( V \)，詞向量維度 \( d \)）。
- 位置嵌入矩陣 \( W_p \) 為 \(\mathbb{R}^{T \times d}\)。

則：
\[
h_0 = W_e x + W_p
\]

---

## **2. 多頭自注意力機制（Multi-Head Self-Attention）**
在 GPT-2 的 **注意力層**（GPT2Attention），首先會將輸入投影到查詢（Query）、鍵（Key）、值（Value）矩陣：

\[
Q = X W_Q, \quad K = X W_K, \quad V = X W_V
\]

其中：
- \( W_Q, W_K, W_V \in \mathbb{R}^{d \times d} \) 為學習到的參數。
- \( X \) 為輸入的詞向量矩陣（形狀為 \([B, T, d]\)）。

接下來計算注意力分數：

\[
A = \frac{QK^T}{\sqrt{d_k}}
\]

其中：
- \( d_k = d / h \)（head_dim），是單個注意力頭的維度。
- \( A \) 形狀為 \([B, h, T, T]\)，其中 \( h \) 是注意力頭數量。

若有注意力遮罩（causal mask），則會應用：

\[
A_{ij} =
\begin{cases}
    A_{ij}, & \text{if } j \leq i \\
    -\infty, & \text{otherwise}
\end{cases}
\]

然後通過 softmax 取得注意力權重：

\[
\alpha = \text{softmax}(A)
\]

最後，計算注意力輸出：

\[
O = \alpha V
\]

並經過線性變換：

\[
Y = O W_O
\]

---

## **3. 殘差連接與層正規化**
Transformer Block 內，每個子層（Self-Attention / MLP）都有殘差連接與層正規化：

\[
X' = \text{LayerNorm}(X + \text{SelfAttention}(X))
\]

\[
X'' = \text{LayerNorm}(X' + \text{MLP}(X'))
\]

---

## **4. 前饋神經網路（MLP 層）**
MLP 由兩層線性變換與 GELU 非線性函數組成：

\[
H = \text{GELU}(X W_1 + b_1)
\]

\[
Y = H W_2 + b_2
\]

其中：
- \( W_1 \in \mathbb{R}^{d \times 4d} \)，\( W_2 \in \mathbb{R}^{4d \times d} \)。

---

## **5. 語言模型頭（Language Model Head）**
輸出經過最終的線性層：

\[
\text{logits} = Y W_{LM}
\]

其中 \( W_{LM} \) 與詞嵌入矩陣 \( W_e \) 共享權重。

---

## **6. 生成機制（Text Generation）**
在 `generate()` 方法中，每次根據當前輸出選擇下一個 token，使用 **溫度調整（temperature scaling）** 和 **Top-p 取樣**：

\[
P_i = \frac{\exp(l_i / T)}{\sum_j \exp(l_j / T)}
\]

若使用 Top-p（核取樣），則將較低機率的 token 濾除，使累積機率剛好超過 \( p \)。

---

這段程式碼主要基於 Transformer 架構，透過自注意力機制與殘差學習來捕捉序列資訊，並用來實作 GPT-2 的語言模型。