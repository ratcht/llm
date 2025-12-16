import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(142)

class SelfAttention(nn.Module):
  def __init__(self, n_embeddings: int, block_size: int, head_size: int, dropout: float = 0.5):
    super().__init__()
    self.block_size = block_size
    self.n_embeddings = n_embeddings
    self.head_size = head_size

    self.query = nn.Linear(n_embeddings, head_size, bias=False)
    self.key = nn.Linear(n_embeddings, head_size, bias=False)
    self.value = nn.Linear(n_embeddings, head_size, bias=False)

    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    self.dropout = nn.Dropout(dropout)


  def forward(self, x: torch.Tensor):
    B, T, C = x.shape

    q = self.query(x) # (batch, block, head_size)
    k = self.key(x) # (batch, block, head_size)
    v = self.value(x)

    w = q @ k.transpose(-2, -1) * C**-0.5 # (batch, block, head_size) @ (batch, head_size, block) -> (batch, block, block)

    w = w.masked_fill(
      self.tril[:T, :T] == 0, float('-inf') # type: ignore
    )
    w = F.softmax(w, dim=-1)
    w = self.dropout(w)

    out = w @ v

    return out

class MultiHeadAttention(nn.Module):
  def __init__(self, num_heads, n_embeddings, block_size, head_size, dropout=0.5):
    super().__init__()

    self.heads = nn.ModuleList([
      SelfAttention(n_embeddings, block_size, head_size, dropout=dropout) for _ in range(num_heads)
    ])
    self.proj = nn.Linear(n_embeddings, n_embeddings)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], -1)
    out = self.proj(out)
    out = self.dropout(out)
    return out


if __name__ == "__main__":
  batch_size, block_size, n_embeddings = 4, 8, 32

  x = torch.randn((batch_size, block_size, n_embeddings))

  attention = SelfAttention(n_embeddings, block_size, head_size=16)
  print(attention(x))
