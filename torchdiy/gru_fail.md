
```sh
(py310) cccimac@cccimacdeiMac 02-language % ./test_gru.sh
tokens= 5231
len(ids)= 5231
ids.size(0)= 5231
batch_size= 20
num_batches= 261
len(ids)= 5220
ids.shape= torch.Size([20, 261])
dictionary= {0: 'the', 1: 'little', 2: 'pig', 3: '<eos>', 4: 'every', 5: 'white', 6: 'cat', 7: 'chase', 8: 'a', 9: 'bite', 10: 'black', 11: 'dog', 12: 'love'}
vocab_size= 13
training ...
Traceback (most recent call last):
  File "/Users/cccimac/Desktop/ccc/code/py/torchdiy/examples/02-language/main.py", line 169, in <module>
    train(corpus, method)
  File "/Users/cccimac/Desktop/ccc/code/py/torchdiy/examples/02-language/main.py", line 89, in train
    outputs, states = model(inputs, states) # 用 model 計算預測詞
  File "/opt/homebrew/Caskroom/miniforge/base/envs/py310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/homebrew/Caskroom/miniforge/base/envs/py310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/Users/cccimac/Desktop/ccc/code/py/torchdiy/examples/02-language/main.py", line 57, in forward
    out, h = self.rnn(x, h)
  File "/opt/homebrew/Caskroom/miniforge/base/envs/py310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/homebrew/Caskroom/miniforge/base/envs/py310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/opt/homebrew/Caskroom/miniforge/base/envs/py310/lib/python3.10/site-packages/torchdiy/gru.py", line 84, in forward
    h = (1 - update_gate) * h + update_gate * new_gate
RuntimeError: The size of tensor a (96) must match the size of tensor b (32) at non-singleton dimension 1
```