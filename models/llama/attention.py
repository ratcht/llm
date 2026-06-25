from typing import Annotated

import torch.nn as nn
import torch.nn.functional as F
from embedding import RoPE
from layers import Dropout, Linear

import torch as t


class MultiHeadAttention(nn.Module):
  cache_k: t.Tensor
  cache_v: t.Tensor

  def __init__(self, embed_dim, num_heads, max_batch_size, max_seq_len, dropout=0.0):
    super().__init__()

    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.head_dim = embed_dim // num_heads
    self.max_batch_size = max_batch_size
    self.max_seq_len = max_seq_len

    # kv cache
    self.register_buffer("cache_k", t.zeros(max_batch_size, num_heads, max_seq_len, self.head_dim))
    self.register_buffer("cache_v", t.zeros(max_batch_size, num_heads, max_seq_len, self.head_dim))

    self.q_proj = Linear(embed_dim, embed_dim, bias=False)
    self.k_proj = Linear(embed_dim, embed_dim, bias=False)
    self.v_proj = Linear(embed_dim, embed_dim, bias=False)
    self.o_proj = Linear(embed_dim, embed_dim, bias=False)

    self.rope = RoPE(self.head_dim)
    self.dropout = Dropout(dropout)

  def reset_cache(self):
    self.cache_k.zero_()
    self.cache_v.zero_()

  def forward(self, x: Annotated[t.Tensor, "batch seq embed_dim"], start_pos: int):
    batch, seq_len, _ = x.shape

    Q = self.q_proj(x)
    K = self.k_proj(x)
    V = self.v_proj(x)

    # reshape to (batch, num_heads, seq, head_dim)
    Q = Q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    K = K.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    V = V.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    Q = self.rope(Q, start_pos)
    K = self.rope(K, start_pos)

    # write & read cache
    self.cache_k[:batch, :, start_pos : start_pos + seq_len] = K
    self.cache_v[:batch, :, start_pos : start_pos + seq_len] = V

    K = self.cache_k[:batch, :, : start_pos + seq_len]
    V = self.cache_v[:batch, :, : start_pos + seq_len]

    att = Q @ K.transpose(-2, -1)
    att = att / (self.head_dim ** 0.5)

    # mask only needed during prefill (seq_len > 1)
    if seq_len > 1:
      mask = t.full((seq_len, seq_len), float("-inf"), device=x.device)
      mask = t.triu(mask, diagonal=1)
      mask = t.hstack([t.zeros((seq_len, start_pos), device=x.device), mask])
      att = att + mask
    att = F.softmax(att, dim=-1)
    att = self.dropout(att)

    out = att @ V

    # reshape back to (batch, seq, embed_dim)
    out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)

    return self.o_proj(out)
