import einops
import torch.nn as nn

import torch as t


class Embedding(nn.Module):
  def __init__(self, num_embed, embed_dim):
    super().__init__()
    self.weight = nn.Parameter(
      t.randn(num_embed, embed_dim)
    )

  def forward(self, x):
    return self.weight[x]


class RoPE(nn.Module):
# note, i validated this logic against LlamaRotaryEmbedding and it passed, so dont worry about this
  cos_cached: t.Tensor
  sin_cached: t.Tensor

  def __init__(self, dim, max_seq_len: int = 4096, base: int = 10000):
    super().__init__()
    self.dim = dim
    self.base = base

    # theta
    theta = self.base ** (-2 * t.arange(0, self.dim//2).float() / self.dim)
    assert theta.shape == (self.dim // 2,)

    # pos indices
    m = t.arange(max_seq_len).float()
    assert m.shape == (max_seq_len,)

    # m * theta
    emb = einops.repeat(t.outer(m, theta), "s d -> s (2 d)")

    self.register_buffer("cos_cached", emb.cos())
    self.register_buffer("sin_cached", emb.sin())

  def forward(self, x):
    *_, L, D = x.shape
    assert D == self.dim

    cos = self.cos_cached[:L, :]
    sin = self.sin_cached[:L, :]

    assert cos.shape == sin.shape == (L, D)

    return self.apply_rotary_emb(x, cos, sin)

  def apply_rotary_emb(self, x, cos, sin):
    x_1, x_2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    x_r = t.cat((-x_2, x_1), dim=-1)

    return (x * cos) + (x_r * sin)
