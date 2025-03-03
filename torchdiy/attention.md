* https://chatgpt.com/c/67c5055b-8884-8012-98b4-9840fb53d7b8

這段程式碼實現的是 **多頭注意力機制 (Multihead Attention, MHA)**，它是 Transformer 架構中的核心組件。背後的數學原理主要來自於 **縮放點積注意力 (Scaled Dot-Product Attention)**，並透過多個注意力頭來增強學習表達能力。  

---

## **1. 縮放點積注意力 (Scaled Dot-Product Attention)**  
給定查詢 (Query)、鍵 (Key) 和值 (Value)，注意力機制的計算方式如下：  

\[
\text{Attention}(Q, K, V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V
\]

其中：
- \( Q \in \mathbb{R}^{T_q \times d_k} \) 是查詢矩陣，\( T_q \) 是查詢的序列長度，\( d_k \) 是鍵的維度。
- \( K \in \mathbb{R}^{T_k \times d_k} \) 是鍵矩陣，\( T_k \) 是鍵值的序列長度。
- \( V \in \mathbb{R}^{T_k \times d_v} \) 是值矩陣，\( d_v \) 是值的維度。
- \( \frac{1}{\sqrt{d_k}} \) 是縮放因子，防止數值過大導致 softmax 過於尖銳。

這段程式碼的這一行對應這個數學運算：

```python
attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
```

這表示：
1. 先計算 \( QK^T \)，這代表每個查詢與所有鍵的相似度。
2. 除以 \( \sqrt{d_k} \) 來防止梯度消失或過大。
3. 套用 softmax，使結果轉為機率分佈：
   ```python
   attn_weights = F.softmax(attn_scores, dim=-1)
   ```

這些注意力權重 \( \alpha \) 用來加權求和值矩陣 \( V \)：

```python
output = torch.matmul(attn_weights, V)
```

這對應數學公式：
\[
\text{Output} = \sum_{i} \alpha_{ij} V_j
\]

---

## **2. 多頭注意力 (Multihead Attention)**  
單頭注意力可能會限制模型的學習能力，因此 Transformer 提出**多頭注意力**，即將嵌入維度 (embed_dim) **分割成多個較小的頭**，並讓每個頭獨立學習不同的關係：

1. 將輸入維度 \( d_{\text{model}} \) 分成 **num_heads** 個小維度 \( d_k \)：
   \[
   d_k = \frac{d_{\text{model}}}{\text{num\_heads}}
   \]

2. 每個頭都有獨立的線性轉換：
   \[
   Q_i = W_q^i X, \quad K_i = W_k^i X, \quad V_i = W_v^i X
   \]
   在程式碼中對應：
   ```python
   self.W_q = nn.Linear(embed_dim, embed_dim, bias=bias)
   self.W_k = nn.Linear(embed_dim, embed_dim, bias=bias)
   self.W_v = nn.Linear(embed_dim, embed_dim, bias=bias)
   ```

3. 並行計算多個頭的注意力：
   ```python
   Q = Q.view(seq_len_q, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
   K = K.view(seq_len_k, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
   V = V.view(seq_len_k, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
   ```
   這裡的 `.view(...).transpose(0, 1)` 讓計算可以對不同的頭進行獨立運算。

4. 計算多頭注意力後，將所有頭的輸出拼接回來：
   ```python
   output = output.transpose(0, 1).contiguous().view(seq_len_q, batch_size, embed_dim)
   ```
   並用最終的線性層轉換回輸出維度：
   ```python
   output = self.W_o(output)
   ```

數學公式表達：
\[
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O
\]

---

## **3. Masking 技術**  
為了避免模型關注到填充 (padding) 的部分，程式碼中有兩種 masking 技術：
1. **填充 Mask (Padding Mask)**：
   ```python
   if key_padding_mask is not None:
       attn_scores = attn_scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
   ```
   這可以讓模型忽略填充部分，不影響注意力計算。

2. **未來 Mask (Causal Mask)**：
   ```python
   if attn_mask is not None:
       attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))
   ```
   這是用於自回歸模型 (如 GPT)，確保未來的詞不會影響當前的輸出。

---

## **4. 直覺理解**
- **點積注意力**：計算每個詞與其他詞的關聯性，然後加權求和。
- **多頭機制**：讓不同的頭學習不同類型的關聯資訊。
- **線性轉換**：確保維度匹配，並增加模型的學習能力。

這段程式碼本質上就是 Transformer 中的 **Multihead Attention Layer**，它可以應用於 NLP、視覺 (ViT) 等深度學習任務。

---

### **小結**
1. **核心公式**：
   \[
   \text{Attention}(Q, K, V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V
   \]
2. **多頭注意力**：
   \[
   \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O
   \]
3. **Masking 技術**確保合理的計算方式：
   - **填充 Mask** 忽略 padding token
   - **未來 Mask** 避免使用未來資訊 (for 自回歸)

這是 Transformer 的核心機制之一，它能夠幫助模型有效學習序列關聯性，並提高表達能力！ 🚀