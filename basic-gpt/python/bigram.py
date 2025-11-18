from dataclasses import dataclass

import einops
import torch
import torch.nn as nn
from data import get_batch, get_flattened
from datasets import Dataset, DatasetDict, load_from_disk
from tokenizer import Tokenizer
from torch.nn import functional as F
from tqdm import tqdm

torch.manual_seed(42)


class BigramLanguageModel(nn.Module):
  def __init__(self, vocab_size: int, block_size: int, n_embeddings: int):
    super().__init__()
    self.vocab_size = vocab_size
    self.token_embedding_table = nn.Embedding(vocab_size, n_embeddings)
    self.position_embedding_table = nn.Embedding(block_size, n_embeddings)

    self.lm_head = nn.Linear(n_embeddings, vocab_size)

  def forward(self, idx, targets=None):
    batch_size, block_size = idx.shape
    device = idx.device

    tok_embeddings = self.token_embedding_table(idx) # (batch, block, n_embeddings)
    pos_embeddings = self.position_embedding_table(torch.arange(block_size, device)) # (block, n_embeddings)

    logits = self.lm_head(tok_embeddings + pos_embeddings) # (batch, block, vocab_size)

    # assert logits.shape == (BATCH_SIZE, BLOCK_SIZE, self.vocab_size)

    if targets is not None:
      logits = einops.rearrange(logits, "B T V -> (B T) V")
      targets = einops.rearrange(targets, "B T -> (B T)")

      loss = F.cross_entropy(logits, targets)
    else:
      loss = None

    return logits, loss

  def generate(self, idx, max_new_tokens=100):
    idx = idx.to(next(self.parameters()).device)

    for _ in range(max_new_tokens):
      logits, _ = self(idx[:, -1]) # only need last token

      probs = F.softmax(logits, dim=-1)

      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)
    return idx

@dataclass
class TrainingParams:
  max_steps=10000
  eval_iters = 200
  eval_interval = 300
  batch_size=32
  n_embeddings=32
  block_size=8
  lr=1e-3
  device="cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def estimate_loss(model: BigramLanguageModel, data: dict[str, torch.Tensor], params: TrainingParams):
  out = {}
  model.eval()
  for split in ["train", "test"]:
    losses = torch.zeros(params.eval_iters)
    for k in range(params.eval_iters):
      X, Y = get_batch(data[split], batch_size=params.batch_size, block_size=params.block_size, device=params.device)
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

    xb, yb = get_batch(train_data, batch_size=params.batch_size, block_size=params.block_size, device=params.device)

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


    loss_history.append(loss.item())

  return model, loss_history


if __name__ == "__main__":
  ds = load_from_disk("tokenized-wikitext-2")
  vocab_path = "toy-gpt/python/vocab.json"

  tokenizer = Tokenizer("toy-gpt/python/vocab.json")
  params = TrainingParams()

  model = BigramLanguageModel(
    tokenizer.vocab_size,
    params.block_size,
    params.n_embeddings
  ).to(params.device)
  print(next(model.parameters()).device)


  ds = load_from_disk("tokenized-wikitext-2")

  model, loss_history = train(model, ds, params)

  print(f"loss history: start - {loss_history[0]}. end - {loss_history[-1]}")

  print(
    tokenizer.decode(
      model.generate(torch.zeros((1, 1), dtype=torch.long, device=params.device), max_new_tokens=100)[0].tolist()
    )
  )
