import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(142)

batch_size, block_size, vocab_size = 4, 8, 32

def toy_self_attention(x: torch.Tensor):
  tril = torch.tril(
    torch.ones(block_size, block_size)
  )
  w = torch.zeros((block_size, block_size))
  w = w.masked_fill(
    tril == 0, float('-inf')
  )
  w = F.softmax(w, dim=-1)
  out = w @ x

  return out

class SelfAttentionHead(nn.Module):
  def __init__(self, head_size: int = 16):
    super().__init__()

    self.Q = nn.Linear(vocab_size, head_size, bias=False)
    self.K = nn.Linear(vocab_size, head_size, bias=False)
    self.V = nn.Linear(vocab_size, head_size, bias=False)

  def forward(self, x: torch.Tensor):
    q = self.Q(x) # (batch, block, head_size)
    k = self.K(x) # (batch, block, head_size)
    v = self.V(x)

    w = q @ k.transpose(-2, -1) # (batch, block, head_size) @ (batch, head_size, block) -> (batch, block, block)

    tril = torch.tril(
      torch.ones(block_size, block_size)
    )
    w = w.masked_fill(
      tril == 0, float('-inf')
    )
    w = F.softmax(w, dim=-1)

    print(w[0])

    out = w @ v

    return out

if __name__ == "__main__":
  x = torch.randn((batch_size, block_size, vocab_size))

  attention = SelfAttentionHead(16)
  print(attention(x))
