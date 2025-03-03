* https://chatgpt.com/c/67c5055b-8884-8012-98b4-9840fb53d7b8

這段程式碼的核心是 **Embedding (嵌入層)**，它的主要數學原理來自於 **查表運算 (Lookup Table) 和 嵌入向量 (Word Embeddings)**，這是深度學習中處理離散資料 (如文字、分類變數) 的關鍵技術。

---

## **1. Embedding (嵌入層) 的數學概念**
### **(1) 查表 (Lookup Table)**
假設我們有一個詞彙表 (Vocabulary)：
\[
V = \{ \text{"apple"}, \text{"banana"}, \text{"cherry"}, \text{"date"} \}
\]
並將每個單詞對應到一個索引：
\[
\text{"apple"} \to 0, \quad \text{"banana"} \to 1, \quad \text{"cherry"} \to 2, \quad \text{"date"} \to 3
\]
那麼，我們的嵌入矩陣 (Weight Matrix) \( W \) 就是一個形狀為 \( (|V|, d) \) 的張量，其中 \( |V| \) 是詞彙數量，\( d \) 是嵌入維度。例如：
\[
W = \begin{bmatrix}
\mathbf{w}_0 \\ 
\mathbf{w}_1 \\ 
\mathbf{w}_2 \\ 
\mathbf{w}_3
\end{bmatrix}
=
\begin{bmatrix}
0.1 & 0.2 & 0.3 \\
0.4 & 0.5 & 0.6 \\
0.7 & 0.8 & 0.9 \\
1.0 & 1.1 & 1.2
\end{bmatrix}
\]
如果我們的輸入是索引 `input = [1, 3]`，則嵌入層會從 \( W \) 中選取索引 1 和 3 的向量：
\[
\text{output} = 
\begin{bmatrix}
\mathbf{w}_1 \\
\mathbf{w}_3
\end{bmatrix}
=
\begin{bmatrix}
0.4 & 0.5 & 0.6 \\
1.0 & 1.1 & 1.2
\end{bmatrix}
\]
這對應到程式碼：
```python
output = weight[input]
```
這實際上是 PyTorch 提供的高效查表操作。

---

### **(2) Padding (填充)**
當使用 `padding_idx` 參數時，對應索引的嵌入向量將被設為零。例如，若 `padding_idx=0`，則：
\[
\mathbf{w}_0 = \mathbf{0}
\]
對應的程式碼：
```python
if padding_idx is not None:
    mask = (input == padding_idx).unsqueeze(-1)
    output = output.masked_fill(mask, 0.0)
```
這確保了填充索引不會影響模型的學習。

---

### **(3) 嵌入向量正規化 (Normalization)**
如果設定 `max_norm`，則嵌入向量的範數 \( ||\mathbf{w}|| \) 不超過 `max_norm`：

\[
\mathbf{w} \leftarrow \mathbf{w} \cdot \min\left( \frac{\text{max_norm}}{||\mathbf{w}||}, 1 \right)
\]

程式碼：
```python
if max_norm is not None:
    norms = output.norm(p=norm_type, dim=-1, keepdim=True)
    output = output * torch.clamp(max_norm / norms, max=1.0)
```
這確保所有嵌入向量的長度不超過 `max_norm`，有助於穩定訓練。

---

## **2. PyTorch `nn.Embedding` 的等效實作**
在 `Embedding` 類別中，`self.weight` 是一個形狀為 `(num_embeddings, embedding_dim)` 的 learnable parameter：
```python
self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
```
初始時，嵌入向量會隨機初始化：
```python
nn.init.normal_(self.weight)
self.weight.data.mul_(1.0 / math.sqrt(self.embedding_dim))
```
這遵循了 **word2vec** 的初始化方式，使嵌入值的標準差為 \( \frac{1}{\sqrt{\text{embedding_dim}}} \)。

當模型訓練時，這些嵌入向量會根據損失函數和梯度下降法進行調整，讓模型學習最適合的詞向量。

---

## **3. 相關應用**
1. **自然語言處理 (NLP)**：
   - 將詞彙轉換為向量，如 Word2Vec、GloVe、FastText、BERT 等。
   - 例如，在 LSTM 或 Transformer 的詞嵌入層。
   
2. **分類問題**：
   - 用於處理 **類別變數** (如電影類型、用戶 ID)。
   - 例如，在推薦系統中，把用戶 ID 轉換成嵌入向量來計算相似性。

---

## **4. 總結**
這段程式碼實作了一個與 `torch.nn.Embedding` 等效的 **手寫嵌入層**，其數學原理包含：
- **查表運算**：從詞彙索引提取對應的嵌入向量。
- **Padding**：為特殊索引設置零向量，避免影響計算。
- **範數正規化**：確保嵌入向量長度不超過 `max_norm`。
- **參數學習**：在深度學習訓練過程中，自動調整嵌入向量的值。

這是 NLP 和推薦系統中 **離散資料向量化** 的關鍵技術！🚀