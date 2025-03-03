# import torchdiy as torch
from torchdiy.transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel

def main():
    config = GPT2Config.from_pretrained("gpt2")
    model = GPT2LMHeadModel(config)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # 設定輸入文本
    input_text = "Once upon a time"
    inputs = tokenizer(input_text, return_tensors="pt")

    # 讓模型生成文字
    output_tokens = model.generate(
        inputs["input_ids"],
        max_length=100,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

    # 解碼並輸出生成的文本
    generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    print("生成的文本：")
    print(generated_text)

if __name__ == "__main__":
    main()