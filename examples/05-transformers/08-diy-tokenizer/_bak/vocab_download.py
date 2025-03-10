import requests
import os
import json

def download_file_from_huggingface(repo_id: str, file_path: str, save_path: str) -> bool:
    """
    從 Hugging Face 下載文件

    Args:
        repo_id (str): Hugging Face 模型的 repo_id，例如 "openai-community/gpt2"
        file_path (str): 文件在模型倉庫中的路徑，例如 "vocab.json"
        save_path (str): 文件保存的本地路徑

    Returns:
        bool: 是否下載成功
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
        return True
    else:
        print(f"無法下載文件 {file_path}，HTTP 狀態碼: {response.status_code}")
        return False

def load_special_tokens(repo_id: str, save_dir: str) -> dict:
    """
    加載特殊 token 映射

    Args:
        repo_id (str): Hugging Face 模型的 repo_id
        save_dir (str): 文件保存的目錄

    Returns:
        dict: 特殊 token 映射
    """
    # 嘗試下載 special_tokens_map.json
    special_tokens_map_file = os.path.join(save_dir, "special_tokens_map.json")
    if download_file_from_huggingface(repo_id, "special_tokens_map.json", special_tokens_map_file):
        with open(special_tokens_map_file, 'r', encoding='utf-8') as f:
            print('download special_tokens_map.json success')
            return json.load(f)

    special_tokens = {
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
    }
    # 如果 special_tokens_map.json 不存在，嘗試從 tokenizer_config.json 中提取
    tokenizer_config_file = os.path.join(save_dir, "tokenizer_config.json")
    if download_file_from_huggingface(repo_id, "tokenizer_config.json", tokenizer_config_file):
        with open(tokenizer_config_file, 'r', encoding='utf-8') as f:
            tokenizer_config = json.load(f)
            # special_tokens = {}
            for key, value in tokenizer_config.items():
                if key.endswith("_token") and value is not None:
                    special_tokens[key] = value
            print('download config success')
            return special_tokens

    # 如果都沒有，返回默認的特殊 token
    print("無法下載 special_tokens_map.json 或 tokenizer_config.json，使用默認特殊 token")
    return special_tokens

# 使用示例
if __name__ == "__main__":
    # 模型 repo_id
    repo_id = "openai-community/gpt2"  # GPT-2 模型的 repo_id

    # 本地保存目錄
    save_dir = "./model_files"
    os.makedirs(save_dir, exist_ok=True)

    # 下載 vocab.json
    vocab_file = os.path.join(save_dir, "vocab.json")
    download_file_from_huggingface(repo_id, "vocab.json", vocab_file)

    # 加載特殊 token
    special_tokens = load_special_tokens(repo_id, save_dir)
    print("特殊 token 映射:", special_tokens)
    with open(f"{save_dir}/special_tokens_map.json", 'w') as f:
        json.dump(special_tokens, f, indent=2)
