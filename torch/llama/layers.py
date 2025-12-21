import math

import torch.nn as nn
import torch.nn.functional as F

import torch as t


class Linear(nn.Module):
  def __init__(self, in_features, out_features, bias=True):
    super().__init__()

    k = 1. / in_features

    def scale(x):
      return (2*x - 1)/(k**(0.5))

    self.weight = nn.Parameter(scale(t.rand(out_features, in_features))) # X @ W + b  (*, in) @ (in, out)
    self.bias = None
    if bias:
      self.bias = nn.Parameter(scale(t.rand(out_features)))

  def forward(self, x):
    w = x @ self.weight.transpose(-2, -1)

    if self.bias is not None:
      w += self.bias

    return w


class Dropout(nn.Module):
  def __init__(self, p=0.5):
    super().__init__()
    self.p = p

  def forward(self, x):
    if self.training:
      # randomly set some to 0
      mask = (t.rand_like(x) > self.p).float()
      return (x * mask) / (1 - self.p)
    return x



class FeedForwardSwiGLU(nn.Module):
  def __init__(self, embed_dim, hidden_dim, dropout=0.5):
    super().__init__()

    self.gate_proj = Linear(embed_dim, hidden_dim, bias=False)
    self.up_proj = Linear(embed_dim, hidden_dim, bias=False)
    self.down_proj = Linear(hidden_dim, embed_dim, bias=False)
    self.dropout = Dropout(dropout)

  def forward(self, x):
    gate = F.silu(self.gate_proj(x))
    x = gate * self.up_proj(x)
    x = self.down_proj(x)
    x = self.dropout(x)
    return x


class RMSNorm(nn.Module):
  def __init__(self, normalized_shape, eps=1e-5):
    super().__init__()
    self.normalized_shape = normalized_shape
    self.n = math.prod(self.normalized_shape)
    self.eps = eps

    self.weight = nn.Parameter(t.ones(self.normalized_shape))

  def forward(self, x):
    ms = x.square().sum(-1, keepdim=True) / self.n

    if self.eps:
      rms = t.sqrt(self.eps + ms)
    else:
      rms = t.sqrt(ms)

    return (x / rms) * self.weight
