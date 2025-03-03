import transformers
import torch
import torch.nn.functional as F
from . import lm

AutoTokenizer = transformers.AutoTokenizer
AutoConfig = transformers.AutoConfig
# AutoModelForCausalLM = transformers.AutoModelForCausalLM

class AutoModelForCausalLM():
    def __init__(self, model, model_name):
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)

    @staticmethod
    def from_pretrained(model_name):
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
        return AutoModelForCausalLM(model, model_name)

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
