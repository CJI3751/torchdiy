import requests
import os

def download_file_from_huggingface(repo_id: str, file_path: str, save_path: str):
    """
    從 Hugging Face 下載文件

    Args:
        repo_id (str): Hugging Face 模型的 repo_id，例如 "openai-community/gpt2"
        file_path (str): 文件在模型倉庫中的路徑，例如 "vocab.json"
        save_path (str): 文件保存的本地路徑
    """
    # Hugging Face 的原始文件 URL
    url = f"https://huggingface.co/{repo_id}/resolve/main/{file_path}"

    # 發送 HTTP 請求
    response = requests.get(url)
    if response.status_code == 200:
        # 確保保存目錄存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # 寫入文件
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"文件已下載並保存到: {save_path}")
    else:
        print(f"{file_path} 無法下載文件，HTTP 狀態碼: {response.status_code}")

# 使用示例
if __name__ == "__main__":
    # 模型 repo_id 和文件路徑
    repo_id = "openai-community/gpt2"  # GPT-2 模型的 repo_id
    # 下載文件
    download_file_from_huggingface(repo_id, "vocab.json", "./gpt2/vocab.json")
    # 下載 tokenizer_config.json
    download_file_from_huggingface(repo_id, "tokenizer_config.json", "./gpt2/tokenizer_config.json")
    # 下載 special_tokens_map.json
    download_file_from_huggingface(repo_id, "special_tokens_map.json", "./gpt2/special_tokens_map.json")
