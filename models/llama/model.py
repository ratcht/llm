import einops
import torch.nn as nn
import torch.nn.functional as F
from block import Block
from config import ModelConfig
from embedding import Embedding
from layers import Linear, RMSNorm

import torch as t


class LlamaModel(nn.Module):
  def __init__(self, config: ModelConfig):
    super().__init__()
    self.config = config
    self.embed_tokens = Embedding(config.vocab_size, config.embed_dim)
    self.layers = nn.ModuleList([
      Block(config.embed_dim, config.hidden_dim, config.num_heads, config.max_batch_size, config.max_seq_len, config.dropout) for _ in range(config.num_blocks)
    ])
    self.norm = RMSNorm((config.embed_dim,))

  def forward(self, x, start_pos: int):
    x = self.embed_tokens(x)
    for layer in self.layers:
      x = layer(x, start_pos)
    x = self.norm(x)
    return x


class Llama(nn.Module):
  def __init__(self, config: ModelConfig):
    super().__init__()
    self.config = config
    self.model = LlamaModel(config)
    self.lm_head = Linear(config.embed_dim, config.vocab_size, bias=False)

  def reset_cache(self):
    for layer in self.model.layers:
      layer.self_attn.reset_cache()

  def forward(self, idx, start_pos: int = 0, targets=None):
    batch_size, block_size = idx.shape

    x = self.model(idx, start_pos)
    logits = self.lm_head(x)

    if targets is not None:
      logits = einops.rearrange(logits, "B T V -> (B T) V")
      targets = einops.rearrange(targets, "B T -> (B T)")
      loss = F.cross_entropy(logits, targets)
    else:
      loss = None

    return logits, loss

  @t.no_grad()
  def generate(self, idx: t.Tensor, max_new_tokens=100, temperature=0.7, top_p=0.9, eos_token_id=None, stream=None):
    idx = idx.to(next(self.parameters()).device)
    prompt_len = idx.size(1)

    self.reset_cache()

    # prefill: process entire prompt
    logits, _ = self(idx, start_pos=0)
    logits = logits[:, -1, :] / temperature

    for cur_pos in range(prompt_len, prompt_len + max_new_tokens):

      # top-p sampling
      sorted_logits, sorted_indices = t.sort(logits, descending=True)
      probs = F.softmax(sorted_logits, dim=-1)
      cumulative_probs = t.cumsum(probs, dim=-1)

      sorted_mask = cumulative_probs > top_p
      sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
      sorted_mask[:, 0] = False

      mask = sorted_mask.scatter(1, sorted_indices, sorted_mask)
      logits[mask] = float('-inf')

      # sample
      probs = F.softmax(logits, dim=-1)
      idx_next = t.multinomial(probs, num_samples=1)
      idx = t.cat((idx, idx_next), dim=1)

      if stream:
        stream(idx_next.item())

      if eos_token_id and idx_next.item() == eos_token_id:
        break

      # decode: process only new token
      logits, _ = self(idx_next, start_pos=cur_pos)
      logits = logits[:, -1, :] / temperature

    return idx


if __name__ == "__main__":
  import utils
  from torchinfo import summary

  model = Llama(
    ModelConfig(
      num_heads=4,
      num_blocks=2,
    )
  )

  print("=== PARAM NAMES ===")
  for name, _ in model.named_parameters():
    print(name)

  print("\n=== MODEL SUMMARY ===")
  summary(model, input_data=t.randint(0, 32000, (1, 32)))

  print("\n=== PARAM COUNT ===")
  utils.print_param_count(model)
