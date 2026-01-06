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
      Block(config.embed_dim, config.hidden_dim, config.num_heads, config.dropout) for _ in range(config.num_blocks)
    ])
    self.norm = RMSNorm((config.embed_dim,))

  def forward(self, x):
    x = self.embed_tokens(x)
    for layer in self.layers:
      x = layer(x)
    x = self.norm(x)
    return x


class Llama(nn.Module):
  def __init__(self, config: ModelConfig):
    super().__init__()
    self.config = config
    self.model = LlamaModel(config)
    self.lm_head = Linear(config.embed_dim, config.vocab_size, bias=False)

  def forward(self, idx, targets=None):
    batch_size, block_size = idx.shape

    x = self.model(idx)
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

    for _ in range(max_new_tokens):
      # forward pass (truncate to max_seq_len)
      logits, _ = self(idx[:, -self.config.max_seq_len:])
      logits = logits[:, -1, :] / temperature

      # top-p sampling: only keep tokens with cumulative prob < top_p
      sorted_logits, sorted_indices = t.sort(logits, descending=True)
      probs = F.softmax(sorted_logits, dim=-1)
      cumulative_probs = t.cumsum(probs, dim=-1)

      # mark tokens to remove (cumulative prob exceeds top_p)
      sorted_mask = cumulative_probs > top_p
      sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
      sorted_mask[:, 0] = False

      # scatter mask back to original order and apply
      mask = sorted_mask.scatter(1, sorted_indices, sorted_mask)
      logits[mask] = float('-inf')

      # sample
      probs = F.softmax(logits, dim=-1)
      idx_next = t.multinomial(probs, num_samples=1)
      idx = t.cat((idx, idx_next), dim=1)

      # stream callback
      if stream:
        stream(idx_next.item())

      # stop on eos
      if eos_token_id and idx_next.item() == eos_token_id:
        break

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
