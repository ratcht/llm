import einops
import torch
import torch.nn as nn
from layers import Block, LayerNorm
from torch.nn import functional as F


class GPT(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config

    self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
    self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
    self.blocks = nn.Sequential(
      *[Block(config.n_embd, config.n_head, config.block_size, config.dropout) for _ in range(config.n_layer)],
      LayerNorm(config.n_embd),
    )

    self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

  def forward(self, idx, targets=None):
    batch_size, block_size = idx.shape
    device = idx.device

    tok_embeddings = self.token_embedding_table(idx) # (batch, block, n_embeddings)
    pos_embeddings = self.position_embedding_table(torch.arange(block_size, device=device)) # (block, n_embeddings)
    x = tok_embeddings + pos_embeddings
    x = self.blocks(x)
    logits = self.lm_head(x) # (batch, block, vocab_size)

    # assert logits.shape == (BATCH_SIZE, BLOCK_SIZE, self.vocab_size)

    if targets is not None:
      logits = einops.rearrange(logits, "B T V -> (B T) V")
      targets = einops.rearrange(targets, "B T -> (B T)")

      loss = F.cross_entropy(logits, targets)
    else:
      loss = None

    return logits, loss

  def generate(self, idx, max_new_tokens=500):
    idx = idx.to(next(self.parameters()).device)

    for _ in range(max_new_tokens):
      logits, _ = self(idx[:, -self.config.block_size:])

      logits = logits[:, -1, :] # (B, C)

      probs = F.softmax(logits, dim=-1)

      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)
    return idx
