from config import ModelConfig
from model import Llama
from transformers import AutoTokenizer, LlamaForCausalLM

import torch as t


def load_models(device="cuda:1"):
  ref_model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", dtype=t.float16, device_map=device)
  ref_model.eval()

  model = Llama(ModelConfig())
  model.load_state_dict(ref_model.state_dict(), strict=False)
  model.half()
  model.to(device)
  model.eval()

  tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
  return ref_model, model, tokenizer


def compare_outputs(ref_model, model, tokenizer, device="cuda:1"):
  tokens = tokenizer("Hello, world!", return_tensors="pt")["input_ids"].to(device)

  with t.no_grad():
    ref_out = ref_model(tokens).logits
    out, _ = model(tokens)

  max_diff = (ref_out - out).abs().max().item()
  mean_diff = (ref_out - out).abs().mean().item()

  print(f"Max diff: {max_diff}")
  print(f"Mean diff: {mean_diff}")


if __name__ == "__main__":
  ref_model, model, tokenizer = load_models()

  print("=== OUTPUT COMPARISON ===")
  compare_outputs(ref_model, model, tokenizer)
