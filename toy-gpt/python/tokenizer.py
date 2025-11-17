# very simple character-level tokenizer
import json
import os


class Tokenizer:
  def __init__(self, vocab_path: str = "vocab.json"):
    if not os.path.exists(vocab_path):
      raise FileNotFoundError(f"Vocabulary file '{vocab_path}' not found.")

    self.vocab = json.load(open(vocab_path))
    self.vocab_size = len(self.vocab)

    self.stoi = {c: i for i, c in enumerate(self.vocab)}
    self.itos = {i: c for i, c in enumerate(self.vocab)}

  def encode(self, s: str) -> list[int]:
    return [self.stoi[c] for c in s]

  def decode(self, li: list[int]) -> str:
    return "".join(self.itos[i] for i in li)


if __name__ == "__main__":
  from datasets import load_dataset

  ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

  tokenizer = Tokenizer("toy-gpt/python/vocab.json")

  tokenized = ds.map(
    lambda x: {"ids": tokenizer.encode(x["text"])},
    batched=False
  )

  tokenized.save_to_disk("tokenized-wikitext-2") # type: ignore
