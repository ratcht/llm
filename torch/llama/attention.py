from typing import Annotated

import torch.nn as nn
import torch.nn.functional as F
from embedding import RoPE
from layers import Dropout, Linear

import torch as t


class MultiHeadAttention(nn.Module):
  mask: t.Tensor | None

  def __init__(self, embed_dim, num_heads, dropout=0.0):
    super().__init__()

    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.head_dim = embed_dim // num_heads

    self.q_proj = Linear(embed_dim, embed_dim, bias=False)
    self.k_proj = Linear(embed_dim, embed_dim, bias=False)
    self.v_proj = Linear(embed_dim, embed_dim, bias=False)
    self.o_proj = Linear(embed_dim, embed_dim, bias=False)

    self.rope = RoPE(self.head_dim)
    self.dropout = Dropout(dropout)

    self.register_buffer("mask", None)

  def forward(self, x: Annotated[t.Tensor, "batch seq embed_dim"]):
    batch, seq_len, _ = x.shape

    Q = self.q_proj(x)
    K = self.k_proj(x)
    V = self.v_proj(x)

    # reshape to (batch, num_heads, seq, head_dim)
    Q = Q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    K = K.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    V = V.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    Q = self.rope(Q)
    K = self.rope(K)

    att = Q @ K.transpose(-2, -1)
    att = att / (self.head_dim ** 0.5)

    if self.mask is None or self.mask.size(-1) < seq_len:
      mask = t.tril(t.ones(seq_len, seq_len, device=x.device))
      self.mask = mask

    att = att.masked_fill(self.mask[:seq_len, :seq_len] == 0, float('-inf'))
    att = F.softmax(att, dim=-1)
    att = self.dropout(att)

    out = att @ V

    # reshape back to (batch, seq, embed_dim)
    out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)

    return self.o_proj(out)
