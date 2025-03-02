import torch
import torch.nn.functional as F
import transformers
from . import lm

GPT2Config = transformers.GPT2Config
GPT2Model = transformers.GPT2Model

# 接著要定義自己的 model , 而非使用 huggingface 的 GPT2LMHeadModel
# 參考 https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
# https://github.com/karpathy/minGPT/blob/master/generate.ipynb
class GPT2LMHeadModel:
    def __init__(self, model, model_name):
        self.model = model
        self.tokenizer = transformers.GPT2Tokenizer.from_pretrained(model_name)

    @staticmethod
    def from_pretrained(model_name):
        model = transformers.GPT2LMHeadModel.from_pretrained(model_name)
        return GPT2LMHeadModel(model, model_name)

    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_length=50,
        num_return_sequences=1,
        temperature=1.0,
        top_k=0,
        top_p=0.0,
        do_sample=False,
        pad_token_id=None,
        eos_token_id=None,
        no_repeat_ngram_size=0,
    ):
        return lm.generate(self.model, self.tokenizer, input_ids, attention_mask, max_length, num_return_sequences, temperature, top_k, top_p, do_sample, pad_token_id, eos_token_id, no_repeat_ngram_size)
