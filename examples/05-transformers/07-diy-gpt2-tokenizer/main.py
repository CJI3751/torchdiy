# example.py
from tokenizer import GPT2Tokenizer, download_gpt2_tokenizer_files
import os

def main():
    # 確保有必要的文件
    model_dir = "./models/gpt2"
    if not os.path.exists(os.path.join(model_dir, "vocab.json")) or \
       not os.path.exists(os.path.join(model_dir, "merges.txt")):
        print("下載必要的模型文件...")
        download_gpt2_tokenizer_files(model_dir)
    
    # 創建 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    
    # 測試單個文本
    text = "Hello, world! How are you doing today?"
    output = tokenizer(text)
    
    print("原始文本:", text)
    print("Token IDs:", output["input_ids"])
    print("Attention Mask:", output["attention_mask"])
    print("解碼回原文:", tokenizer.decode(output["input_ids"]))
    
    # 測試批量處理並使用 padding
    texts = [
        "Hello, world!",
        "GPT-2 is a large language model developed by OpenAI."
    ]
    
    print("\n批量處理:")
    batch_output = tokenizer(texts, padding=True)
    
    for i, t in enumerate(texts):
        print(f"\n文本 {i+1}: {t}")
        print(f"Token IDs: {batch_output['input_ids'][i]}")
        print(f"Attention Mask: {batch_output['attention_mask'][i]}")
        print(f"解碼回原文: {tokenizer.decode(batch_output['input_ids'][i])}")

if __name__ == "__main__":
    main()