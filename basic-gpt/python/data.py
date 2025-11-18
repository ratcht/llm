import torch

from datasets import Dataset, DatasetDict, load_from_disk

torch.manual_seed(42)


def get_flattened(ds: Dataset | DatasetDict, split: str) -> torch.Tensor:
  flattened = []
  for ids in ds[split]["ids"]: # type: ignore
    flattened.extend(ids)

  return torch.tensor(flattened, dtype=torch.long)

def get_batch(t: torch.Tensor, batch_size, block_size, device="cpu"):
  r = torch.randint(len(t) - block_size, (batch_size,))

  x = torch.stack([t[i:i+block_size] for i in r])
  y = torch.stack([t[i+1:i+block_size+1] for i in r])

  x, y = x.to(device), y.to(device)

  return x, y


if __name__ == "__main__":
  ds = load_from_disk("datasets/tokenized-wikitext-2")

  train = get_flattened(ds, "train")

  xb, yb = get_batch(train, 8, 8)

  print(xb.shape)
  print(xb)
  print("--------")
  print(yb.shape)
  print(yb)
