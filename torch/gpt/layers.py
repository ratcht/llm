import torch
import torch.nn as nn
from attention import MultiHeadAttention


class LayerNorm(nn.Module):
  def __init__(self, num_features, eps=1e-05):
    super().__init__()
    self.num_features = num_features
    self.eps = eps
    self.gamma = nn.Parameter(torch.ones(num_features))
    self.beta = nn.Parameter(torch.zeros(num_features))

  def forward(self, x: torch.Tensor):
    x_mean = x.mean(-1, keepdim=True) # batch mean
    x_var = x.var(-1, keepdim=True) # batch mean

    x_hat = (x - x_mean) / torch.sqrt(x_var + self.eps)
    out = self.gamma * x_hat + self.beta

    return out


class FeedForward(nn.Module):
  def __init__(self, n_embd, dropout):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embd, 4 * n_embd),
      nn.ReLU(),
      nn.Linear(4 * n_embd, n_embd), # proj layer
      nn.Dropout(dropout)
    )

  def forward(self, x):
    return self.net(x)


class Block(nn.Module):
  def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
    super().__init__()
    head_size = n_embd // n_head
    self.sa = MultiHeadAttention(n_head, n_embd, block_size, head_size, dropout=dropout)
    self.ffwd = FeedForward(n_embd, dropout)
    self.ln1 = LayerNorm(n_embd)
    self.ln2 = LayerNorm(n_embd)

  def forward(self, x: torch.Tensor):
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x
