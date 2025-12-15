from dataclasses import dataclass
from pathlib import Path

import einops
import torch
import torch.nn as nn
from attention import MultiHeadAttention
from data import get_batch, get_flattened
from tokenizer import Tokenizer
from torch.nn import functional as F
from tqdm import tqdm

from datasets import Dataset, DatasetDict, load_from_disk

torch.manual_seed(42)

dataset_path = "datasets/tokenized-wikitext-2"

ds = load_from_disk(dataset_path)

tokenizer = Tokenizer(str(Path(dataset_path) / "vocab.json"))

@dataclass
class ModelParams:
  n_embd=384
  n_head=6
  n_layer=6
  vocab_size=tokenizer.vocab_size
  block_size=256
  dropout=0.2

@dataclass
class TrainingParams:
  max_steps=5000
  eval_iters = 200
  eval_interval = 1000
  batch_size=64
  lr=3e-4
  device="cuda" if torch.cuda.is_available() else "cpu"

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
  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embd, 4 * n_embd),
      nn.ReLU(),
      nn.Linear(4 * n_embd, n_embd), # proj layer
      nn.Dropout(ModelParams.dropout)
    )

  def forward(self, x):
    return self.net(x)


class Block(nn.Module):
  def __init__(self, n_embd: int, n_head: int):
    super().__init__()
    head_size = n_embd // n_head
    self.sa = MultiHeadAttention(n_head, n_embd, ModelParams.block_size, head_size, dropout=ModelParams.dropout)
    self.ffwd = FeedForward(n_embd)
    self.ln1 = LayerNorm(n_embd)
    self.ln2 = LayerNorm(n_embd)

  def forward(self, x: torch.Tensor):
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x

class BigramLanguageModel(nn.Module):
  def __init__(self, n_embd: int, n_head: int, n_layer: int):
    super().__init__()
    self.n_embd = n_embd

    self.token_embedding_table = nn.Embedding(ModelParams.vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(ModelParams.block_size, n_embd)
    self.blocks = nn.Sequential(
      *[Block(ModelParams.n_embd, ModelParams.n_head) for _ in range(n_layer)],
      LayerNorm(n_embd),
    )

    self.lm_head = nn.Linear(n_embd, ModelParams.vocab_size)

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
      logits, _ = self(idx[:, -ModelParams.block_size:]) # only need last token

      logits = logits[:, -1, :] # (B, C)

      probs = F.softmax(logits, dim=-1)

      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)
    return idx



@torch.no_grad()
def estimate_loss(model: BigramLanguageModel, data: dict[str, torch.Tensor], params: TrainingParams):
  out = {}
  model.eval()
  for split in ["train", "test"]:
    losses = torch.zeros(params.eval_iters)
    for k in range(params.eval_iters):
      X, Y = get_batch(data[split], batch_size=params.batch_size, block_size=ModelParams.block_size, device=params.device)
      logits, loss = model(X, Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out

def train(model: BigramLanguageModel, ds: Dataset | DatasetDict, params: TrainingParams):
  optimizer = torch.optim.AdamW(model.parameters(), lr=params.lr)

  train_data = get_flattened(ds, "train")
  test_data = get_flattened(ds, "test")

  pbar = tqdm(range(params.max_steps), desc="training", leave=True)

  loss_history = []
  for step in pbar:
    if step % params.eval_interval == 0:
      loss_history.append(
        estimate_loss(model, {"train": train_data, "test": test_data}, params)
      )
      pbar.set_postfix(loss=f"train loss: {loss_history[-1]['train']:.4f}  test loss: {loss_history[-1]['test']:.4f}")

    xb, yb = get_batch(train_data, batch_size=params.batch_size, block_size=ModelParams.block_size, device=params.device)

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


    loss_history.append(loss.item())

  return model, loss_history


if __name__ == "__main__":


  params = TrainingParams()

  model = BigramLanguageModel(
    ModelParams.n_embd,
    ModelParams.n_head,
    ModelParams.n_layer
  ).to(params.device)
  print(next(model.parameters()).device)


  model, loss_history = train(model, ds, params)

  # Save the trained model
  torch.save(model.state_dict(), "trained_bigram_model.pth")
  print("Model saved as 'trained_bigram_model.pth'")

  print(f"loss history: start - {loss_history[0]}. end - {loss_history[-1]}")

  print(
    tokenizer.decode(
      model.generate(torch.zeros((1, 1), dtype=torch.long, device=params.device), max_new_tokens=1000)[0].tolist()
    )
  )
