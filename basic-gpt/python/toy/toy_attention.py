# %%
import torch
import torch.nn.functional as F

torch.manual_seed(142)

# dummy inefficient self attention
# here, we just average the current + all previous tokens for each token (bow)
batch_size, block_size, vocab_size = 4, 8, 2
x = torch.randn((batch_size, block_size, vocab_size))
xbow = torch.zeros((batch_size, block_size, vocab_size))

for b in range(batch_size):
  for t in range(block_size):
    xprev = x[b, :t+1] # (t, vocab_size)
    assert xprev.shape == (t + 1, vocab_size)

    xbow[b, t] = torch.mean(xprev, dim=0)

print(x[0])
print(xbow[0])
print("")
print("============")
print("")

# %%
# matmul trick
# we should instead use matmul
a = torch.tril(torch.ones(3, 3)) # lower triangular
a = a / torch.sum(a, dim=-1, keepdim=True) # normalize rows to get average
b = torch.randint(0, 10, (2, 3)).float()
c = a @ b.T

print(f"a={a}")
print(f"b={b}")
print(f"c={c.T}")

print("")
print("============")
print("")

# %%

# now apply matmul trick to attention
w = torch.tril(torch.ones(block_size, block_size))
w = w / torch.sum(w, dim=-1, keepdim=True)
xbow2 = w @ x # (T, T) @ (B, T, C) -> (B, T, T) @ (B, T, C) --> (B, T, C)

print(xbow2[0])

print(torch.allclose(xbow, xbow2, atol=1e-5))

print("")
print("============")
print("")

# %%

# clean using softmax
tril = torch.tril(
  torch.ones(block_size, block_size)
)
w = torch.zeros((block_size, block_size))

w = w.masked_fill(
  tril == 0, float('-inf')
)

w = F.softmax(w, dim=-1) ## softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)

xbow3 = w @ x # (T, T) @ (B, T, C) -> (B, T, T) @ (B, T, C) --> (B, T, C)

print(w)
print(xbow3[0])
print(torch.allclose(xbow2, xbow3, atol=1e-5))
