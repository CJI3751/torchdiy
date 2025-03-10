import transformers
import json

class BaseTokenizer:
    """通用 Tokenizer 基類"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.token_to_id = {}
        self.id_to_token = {}

    def load_vocab(self):
        """通用詞彙表載入方法，需要各 tokenizer 自行實作"""
        raise NotImplementedError

    def encode(self, text: str):
        """將文字轉換為 token ID"""
        raise NotImplementedError

    def decode(self, token_ids: list):
        """將 token ID 還原為文字"""
        raise NotImplementedError


class WordPieceTokenizer(BaseTokenizer):
    """實作 WordPiece 分詞器"""

    def __init__(self, model_name: str = "bert-base-uncased"):
        super().__init__(model_name)
        self.vocab_url = f"https://huggingface.co/{model_name}/resolve/main/vocab.txt"
        self.load_vocab()

    def load_vocab(self):
        """從 Hugging Face 下載 vocab.txt"""
        import requests
        response = requests.get(self.vocab_url)
        if response.status_code != 200:
            raise ValueError(f"Failed to download vocab from {self.vocab_url}")
        self.token_to_id = {token.strip(): i for i, token in enumerate(response.text.split("\n"))}
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    def encode(self, text: str):
        """使用 WordPiece 分詞"""
        words = text.lower().split()
        tokens = []
        for word in words:
            if word in self.token_to_id:
                tokens.append(word)
            else:
                sub_tokens = []
                for i in range(len(word)):
                    sub_word = "##" + word[i:] if i > 0 else word[i:]
                    if sub_word in self.token_to_id:
                        sub_tokens.append(sub_word)
                tokens.extend(sub_tokens if sub_tokens else ["[UNK]"])
        return [self.token_to_id[token] for token in tokens]

    def decode(self, token_ids):
        """將 token ID 轉換回文字"""
        tokens = [self.id_to_token[id] for id in token_ids]
        return " ".join(tokens).replace(" ##", "")


class BPETokenizer(BaseTokenizer):
    """實作 Byte Pair Encoding (BPE) 分詞器"""

    def __init__(self, model_name: str = "gpt2"):
        super().__init__(model_name)
        self.vocab_url = f"https://huggingface.co/{model_name}/resolve/main/vocab.json"
        self.merges_url = f"https://huggingface.co/{model_name}/resolve/main/merges.txt"
        self.bpe_ranks = {}
        self.load_vocab()

    def load_vocab(self):
        """下載 vocab.json 和 merges.txt"""
        import requests
        response = requests.get(self.vocab_url)
        if response.status_code != 200:
            raise ValueError(f"Failed to download vocab from {self.vocab_url}")
        self.token_to_id = json.loads(response.text)
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

        response = requests.get(self.merges_url)
        if response.status_code != 200:
            raise ValueError(f"Failed to download merges from {self.merges_url}")
        merges = response.text.strip().split("\n")[1:]
        self.bpe_ranks = {tuple(pair.split()): i for i, pair in enumerate(merges)}

    def _bpe(self, word):
        """應用 BPE 分詞"""
        tokens = list(word)
        while len(tokens) > 1:
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            pair_ranks = {pair: self.bpe_ranks[pair] for pair in pairs if pair in self.bpe_ranks}
            if not pair_ranks:
                break
            best_pair = min(pair_ranks, key=pair_ranks.get)
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append("".join(best_pair))
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def encode(self, text):
        words = text.lower().split()
        tokens = [subword for word in words for subword in self._bpe(word)]
        return [self.token_to_id.get(token, self.token_to_id.get("<|endoftext|>")) for token in tokens]

    def decode(self, token_ids):
        tokens = [self.id_to_token[id] for id in token_ids]
        return "".join(tokens).replace("Ġ", " ")

class SentencePieceTokenizer(BaseTokenizer):
    """實作 SentencePiece Tokenizer"""

    def __init__(self, model_name: str = "t5-small"):
        super().__init__(model_name)
        self.tokenizer_url = f"https://huggingface.co/{model_name}/resolve/main/tokenizer.json"
        self.load_vocab()

    def load_vocab(self):
        """下載 tokenizer.json"""
        import requests
        response = requests.get(self.tokenizer_url)
        if response.status_code != 200:
            raise ValueError(f"Failed to download tokenizer.json from {self.tokenizer_url}")
        tokenizer_data = json.loads(response.text)
        self.token_to_id = tokenizer_data["model"]["vocab"]
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    def encode(self, text):
        """直接查找詞彙表"""
        tokens = text.lower().split()
        return [self.token_to_id.get(token, self.token_to_id.get("<unk>")) for token in tokens]

    def decode(self, token_ids):
        return " ".join([self.id_to_token[id] for id in token_ids])

def get_tokenizer(model_name: str):
    if "bert" in model_name or "roberta" in model_name:
        return WordPieceTokenizer(model_name)
    elif "gpt2" in model_name or "t5" in model_name:
        return BPETokenizer(model_name)
    elif "t5" in model_name or "albert" in model_name:
        return SentencePieceTokenizer(model_name)
    else:
        raise ValueError("Unsupported model type")

tokenizer = get_tokenizer("gpt2") # BPE
print(tokenizer.encode("Hello world! GPT-2 is amazing."))


# tokenizer = get_tokenizer("google-t5/t5-base") # Unigram SentencePiece
# print(tokenizer.encode("Hello world! GPT-2 is amazing."))

tokenizer = get_tokenizer("google-bert/bert-base-uncased")
print(tokenizer.encode("Hello world! GPT-2 is amazing."))