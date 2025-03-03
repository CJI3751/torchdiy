https://chatgpt.com/c/67c5055b-8884-8012-98b4-9840fb53d7b8

# 參考 -- [LLM大语言模型之Generate/Inference（生成/推理）中参数与解码策略原理及其代码实现](https://zhuanlan.zhihu.com/p/653926703)


這段程式碼實作了一個 `generate` 函數，它使用 Transformer 語言模型（例如 GPT-2、GPT-3、T5 等）來生成文本。背後的數學原理主要來自 **自回歸語言模型（Auto-Regressive Language Models）**，以及 **不同的取樣策略（Sampling Strategies）**。以下是詳細的數學解析：

---

## **1. 自回歸語言模型（Auto-Regressive Language Models）**
Transformer 模型是基於機率語言建模的，目標是估計條件機率：
\[
P(x_t | x_1, x_2, \dots, x_{t-1})
\]
這代表在給定前面 \( t-1 \) 個 token 的情況下，預測第 \( t \) 個 token 的機率分佈。

模型的輸出是對所有可能 token 的機率分佈：
\[
P(x_t | x_{1:t-1}) = \text{softmax}(W h_t)
\]
其中：
- \( h_t \) 是 Transformer 的最後一層隱藏狀態
- \( W \) 是輸出的線性層參數
- `softmax` 保證所有機率總和為 1

這段程式的 `generate` 函數，每次都將當前生成的序列 `generated` 輸入 Transformer 模型，獲取下一個 token 的機率分佈 \( P(x_t | x_{1:t-1}) \)，並根據不同策略選擇下一個 token。

---

## **2. 溫度縮放（Temperature Scaling）**
程式碼中這一行：
```python
next_token_logits = next_token_logits / temperature
```
對應數學公式：
\[
P(x_t | x_{1:t-1}) = \frac{\exp(z_t / T)}{\sum_{j} \exp(z_j / T)}
\]
其中：
- \( z_t \) 是 Transformer 的原始 logits（未經 softmax 的分數）
- \( T \) 是溫度參數（`temperature`）
- 當 \( T < 1 \) 時，機率分佈變得更加銳利（更確定），使模型更貪婪選擇最可能的 token
- 當 \( T > 1 \) 時，機率分佈變得更加平滑（增加隨機性）

---

## **3. Top-k 採樣（Top-k Sampling）**
程式碼：
```python
if top_k > 0:
    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
    next_token_logits[indices_to_remove] = -float("Inf")
```
數學公式：
\[
P'(x_t) =
\begin{cases}
P(x_t) & \text{if } x_t \in \text{Top-k}, \\
0 & \text{otherwise}.
\end{cases}
\]
然後重新計算 softmax 確保機率總和為 1。

這個方法確保模型只從前 \( k \) 個最高機率的 token 中選擇，避免罕見 token 被選中。

---

## **4. 核採樣（Top-p Sampling）**
程式碼：
```python
if top_p > 0.0:
    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    next_token_logits[indices_to_remove] = -float("Inf")
```
數學公式：
\[
P'(x_t) =
\begin{cases}
P(x_t) & \text{if } \sum_{j=1}^{t} P(x_j) \leq p, \\
0 & \text{otherwise}.
\end{cases}
\]
這表示我們從最可能的 token 開始累積機率，直到總和達到 \( p \)，然後忽略剩下的 token，並重新計算 softmax。

這樣可以確保每次生成時，根據內容選擇不同數量的 token，而不像 Top-k 那樣總是固定選擇前 \( k \) 個。

---

## **5. 避免重複 n-gram**
程式碼：
```python
if no_repeat_ngram_size > 0:
    for batch_idx in range(batch_size):
        generated_sequence = generated[batch_idx].tolist()
        for ngram in zip(*[generated_sequence[i:] for i in range(no_repeat_ngram_size)]):
            if len(set(ngram)) == 1:
                next_token_logits[batch_idx, ngram[0]] = -float("Inf")
```
這部分是防止模型生成重複的 n-gram，例如：
```
The cat is sleeping. The cat is sleeping.
```
數學上，它強制模型不能選擇已經出現過的 n-gram，這可以用集合表示：
\[
\forall x_t, \text{ if } (x_{t-n}, ..., x_t) \in \mathcal{H}, \quad P(x_t) = 0
\]
其中 \( \mathcal{H} \) 是已出現的 n-gram 集合。

---

## **6. 採樣策略**
程式碼：
```python
if do_sample:
    next_token = torch.multinomial(probs, num_samples=1)
else:
    next_token = torch.argmax(probs, dim=-1, keepdim=True)
```
- 若 `do_sample = False`，則選擇機率最大的 token（貪婪搜索，Greedy Search）：
\[
x_t = \arg\max P(x_t | x_{1:t-1})
\]
- 若 `do_sample = True`，則根據機率分佈進行抽樣：
\[
x_t \sim P(x_t | x_{1:t-1})
\]
這樣可以增加生成的隨機性，使模型不會總是選擇相同的序列。

---

## **7. 迭代生成**
程式碼：
```python
generated = torch.cat([generated, next_token], dim=-1)
```
這是自回歸模型的關鍵：
\[
x_{t+1} = \text{Model}(x_1, x_2, ..., x_t)
\]
這樣一步步地將生成的 token 餵回模型，直到達到 `max_length` 或遇到 `eos_token_id`（結束 token）。

---

## **結論**
這段程式碼實現了一個強大的文本生成方法，結合了：
1. **自回歸 Transformer 模型**：根據之前的 token 來生成下一個 token。
2. **溫度控制（Temperature Scaling）**：調整機率分佈的平滑度。
3. **Top-k 採樣**：只從最可能的 k 個 token 中選擇。
4. **Top-p 採樣（核採樣）**：確保選擇的 token 符合動態閾值 \( p \)。
5. **避免重複 n-gram**：防止模型生成重複文本。
6. **貪婪搜索 vs 隨機抽樣**：決定是否讓生成的文本更確定或更有創造性。

這些數學原理是現代 NLP 文本生成的核心，這種方法廣泛應用於 GPT、ChatGPT、BERT 變體等大型語言模型中。