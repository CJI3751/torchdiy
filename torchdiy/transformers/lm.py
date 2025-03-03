import transformers
import torch
import torch.nn.functional as F

class CausalLMModel(transformers.PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def generate(self,
                 input_ids,
                 max_length=50,
                 temperature=1.0,
                 top_p=1.0,
                 do_sample=False,
                 **kwargs):
        self.eval()
        generated = input_ids
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                outputs = self(input_ids=generated)
                next_token_logits = outputs.logits[:, -1, :] / temperature
                
                if do_sample and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    next_token_logits.scatter_(1, sorted_indices, 
                                             sorted_logits.masked_fill(sorted_indices_to_remove, -float('inf')))
                
                if do_sample:
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                generated = torch.cat([generated, next_token], dim=1)
                
                if next_token.item() == self.config.eos_token_id:
                    break
        
        return generated