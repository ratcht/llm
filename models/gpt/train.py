from pathlib import Path

import torch
from config import ModelConfig, TrainingConfig
from data import get_batch, get_flattened
from model import GPT
from tokenizer import Tokenizer
from tqdm import tqdm

from datasets import Dataset, DatasetDict, load_from_disk

torch.manual_seed(42)


@torch.no_grad()
def estimate_loss(model: GPT, data: dict[str, torch.Tensor], model_config: ModelConfig, training_config: TrainingConfig):
  out = {}
  model.eval()
  for split in ["train", "test"]:
    losses = torch.zeros(training_config.eval_iters)
    for k in range(training_config.eval_iters):
      X, Y = get_batch(data[split], batch_size=training_config.batch_size, block_size=model_config.block_size, device=training_config.device)
      logits, loss = model(X, Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out

def train(model: GPT, ds: Dataset | DatasetDict, model_config: ModelConfig, training_config: TrainingConfig):
  optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.lr)

  train_data = get_flattened(ds, "train")
  test_data = get_flattened(ds, "test")

  pbar = tqdm(range(training_config.max_steps), desc="training", leave=True)

  loss_history = []
  for step in pbar:
    if step % training_config.eval_interval == 0:
      loss_history.append(
        estimate_loss(model, {"train": train_data, "test": test_data}, model_config, training_config)
      )
      pbar.set_postfix(loss=f"train loss: {loss_history[-1]['train']:.4f}  test loss: {loss_history[-1]['test']:.4f}")

    xb, yb = get_batch(train_data, batch_size=training_config.batch_size, block_size=model_config.block_size, device=training_config.device)

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


    loss_history.append(loss.item())

  return model, loss_history


if __name__ == "__main__":

  dataset_path = "datasets/tokenized-wikitext-2"

  ds = load_from_disk(dataset_path)

  tokenizer = Tokenizer(str(Path(dataset_path) / "vocab.json"))

  model_config = ModelConfig(vocab_size=tokenizer.vocab_size)
  training_config = TrainingConfig()

  model = GPT(model_config).to(training_config.device)
  print(next(model.parameters()).device)


  model, loss_history = train(model, ds, model_config, training_config)

  # Save the trained model
  torch.save(model.state_dict(), "trained_bigram_model.pth")
  print("Model saved as 'trained_bigram_model.pth'")

  print(f"loss history: start - {loss_history[0]}. end - {loss_history[-1]}")

  print(
    tokenizer.decode(
      model.generate(torch.zeros((1, 1), dtype=torch.long, device=training_config.device), max_new_tokens=1000)[0].tolist()
    )
  )
