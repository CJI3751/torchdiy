from tokenizer import SimpleTokenizer

# Training a tokenizer on a corpus
texts = [
    "這是一個中文句子，用來測試分詞器。",
    "The quick brown fox jumps over the lazy dog.",
    "我們可以處理多語言文本。This is multilingual text.",
    "自然語言處理是人工智能的重要分支。",
    "Machine learning models require large datasets for training.",
    "深度學習模型在圖像識別任務中表現出色。",
    "Transformers architecture revolutionized NLP tasks.",
    "數據預處理是機器學習流程中的關鍵步驟。",
    "GPT models generate text based on input prompts.",
    "語言模型可以用於翻譯、摘要和問答系統。",
    "Word embeddings capture semantic relationships between words.",
    "向量空間模型將詞語映射到高維空間中。",
    "Attention mechanisms help models focus on relevant parts of input.",
    "批處理可以提高計算效率和訓練速度。",
    "數據增強技術可以提高模型的泛化能力。",
    "Transfer learning reduces the need for large training datasets.",
    "模型評估需要適當的度量標準和測試數據。",
    "Hyperparameter tuning is crucial for optimal model performance.",
    "正則化技術可以減少過擬合問題。",
    "損失函數定義了模型的優化目標。",
    "梯度下降是優化神經網絡的常用方法。",
    "Backpropagation calculates gradients in neural networks.",
    "卷積神經網絡在圖像處理任務中表現優異。",
    "循環神經網絡適合處理序列數據。",
    "LSTM networks address the vanishing gradient problem.",
    "注意力機制使模型能夠關注輸入的相關部分。",
    "Transformer models rely on self-attention mechanisms.",
    "預訓練語言模型可以適應各種下游任務。",
    "微調是將預訓練模型應用於特定任務的過程。",
    "語言生成任務需要創造性和上下文理解。",
    "Text classification assigns categories to documents.",
    "命名實體識別從文本中提取特定實體。",
    "情感分析評估文本中表達的情感極性。",
    "文本摘要生成文本的簡短版本。",
    "機器翻譯在不同語言之間轉換文本。",
    "問答系統回答基於上下文的問題。",
    "對話系統模擬人類對話的交互方式。",
    "知識圖譜表示實體間的關係和屬性。",
    "邏輯推理是人工智能的基礎能力之一。",
    "自然語言理解關注文本的語義和意圖。",
    "Tokenizers break text into smaller units for processing.",
    "詞袋模型將文本表示為詞頻向量。",
    "TF-IDF weighting accounts for term frequency and document frequency.",
    "N-gram models capture sequences of adjacent words.",
    "主題模型發現文本集合中的主題結構。",
    "詞性標注為文本中的每個詞分配語法標籤。",
    "依存句法分析確定詞語之間的語法關係。",
    "語義角色標注識別句子中的語義關係。",
    "詞義消歧確定多義詞在特定上下文中的含義。",
    "指代消解確定代詞所指的實體。",
    "語篇分析研究文本的連貫性和結構。",
    "語言識別檢測文本的語言。",
    "拼寫校正識別和糾正拼寫錯誤。",
    "語法檢查識別和糾正語法錯誤。",
    "文本標準化將文本轉換為標準形式。",
    "文本分類將文本分到預定義類別中。",
    "實體鏈接將提及的實體鏈接到知識庫。",
    "關係抽取識別文本中實體之間的關係。",
    "事件抽取識別文本中描述的事件。",
    "時間表達式識別和標準化時間提及。",
    "數值表達式識別和標準化數值提及。",
    "Text generation creates new content based on patterns.",
    "摘要生成提取或抽象關鍵信息。",
    "機器翻譯在保留意義的同時轉換語言。",
    "文本補全預測缺失或後續的文本。",
    "對話生成創建連貫的交互式回應。",
    "文本改寫以不同方式表達相同的意思。",
    "Text simplification makes content more accessible.",
    "文本擴展添加細節或闡述概念。",
    "風格轉換改變文本的寫作風格。",
    "語法糾錯識別和修正語法錯誤。",
    "詞語建議提供上下文相關的詞語選擇。",
    "自動評分評估文本的質量或相關性。",
    "Plagiarism detection identifies copied content.",
    "情感分析確定文本的情感極性。",
    "觀點挖掘識別文本中表達的觀點。",
    "立場檢測確定文本對特定主題的立場。",
    "諷刺檢測識別諷刺或反諷表達。",
    "仇恨言論檢測識別有害或冒犯性內容。",
    "假新聞檢測識別誤導性或虛假信息。",
    "Natural language processing bridges humans and computers.",
    "人機交互設計創造直觀的用戶體驗。",
    "語音識別將語音轉換為文本。",
    "語音合成將文本轉換為自然語音。",
    "多模態學習結合文本、圖像和其他數據類型。",
    "強化學習通過試錯學習最優策略。",
    "無監督學習發現數據中的隱藏模式。",
    "監督學習從標記數據中學習模式。",
    "半監督學習結合標記和未標記的數據。",
    "主動學習選擇性地請求標記數據。",
    "遷移學習將知識從一個領域轉移到另一個領域。",
    "Few-shot learning generalizes from limited examples.",
    "聯邦學習在保護隱私的同時進行分佈式學習。",
    "可解釋人工智能提供模型決策的透明度。",
    "負責任的人工智能考慮倫理和社會影響。",
    "數據隱私保護個人信息和用戶權利。",
    "公平性評估確保人工智能系統的公平對待。",
    "Human-in-the-loop systems combine human and AI capabilities."
]

# Initialize and train tokenizer
tokenizer = SimpleTokenizer(vocab_size=1000)
tokenizer.train(texts)

# Encode a new text
text = "This is a test sentence."
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.encode(text)

print(f"Tokens: {tokens}")
print(f"Token IDs: {token_ids}")

# Decode back to text
decoded_text = tokenizer.decode(token_ids)
print(f"Decoded: {decoded_text}")

# Save and load the tokenizer
tokenizer.save_pretrained("./my_tokenizer")
loaded_tokenizer = SimpleTokenizer.from_pretrained("./my_tokenizer")

# Batch processing with padding
batch = ["Short text", "This is a longer example text to demonstrate padding"]
encoded_batch = tokenizer.batch_encode_plus(batch, max_length=10, padding=True, truncation=True)
print(encoded_batch)
