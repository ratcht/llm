import torch.nn as nn
from attention import MultiHeadAttention
from layers import FeedForwardSwiGLU, RMSNorm


class Block(nn.Module):
  def __init__(self, embed_dim, hidden_dim, num_heads, max_batch_size, max_seq_len, dropout=0.0):
    super().__init__()

    self.self_attn = MultiHeadAttention(embed_dim, num_heads, max_batch_size, max_seq_len, dropout)
    self.input_layernorm = RMSNorm((embed_dim,))
    self.mlp = FeedForwardSwiGLU(embed_dim, hidden_dim, dropout)
    self.post_attention_layernorm = RMSNorm((embed_dim,))

  def forward(self, x, start_pos: int):
    x = x + self.self_attn(self.input_layernorm(x), start_pos)
    x = x + self.mlp(self.post_attention_layernorm(x))

    return x
