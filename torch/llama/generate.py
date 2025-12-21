from config import ModelConfig
from model import Llama
from transformers import AutoTokenizer, LlamaForCausalLM

import torch as t


def load_model(device="cuda:1"):
  ref_model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", dtype=t.float16, device_map=device)

  model = Llama(ModelConfig())
  model.load_state_dict(ref_model.state_dict(), strict=False)
  model.half()
  model.to(device)
  model.eval()

  del ref_model
  t.cuda.empty_cache()

  tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
  return model, tokenizer


def chat(model, tokenizer):
  print("Chat with LLaMA (type ':q' to exit)")
  print("-" * 40)

  def stream(token_id):
    print(tokenizer.decode(token_id), end="", flush=True)

  while True:
    try:
      prompt = input("\n> ")
    except (KeyboardInterrupt, EOFError):
      print("\nBye!")
      break

    if prompt.lower() in [":q"]:
      print("Bye!")
      break

    if not prompt.strip():
      continue

    print()
    tokens = tokenizer(prompt, return_tensors="pt")["input_ids"]
    model.generate(tokens, max_new_tokens=400, eos_token_id=tokenizer.eos_token_id, stream=stream)
    print()


if __name__ == "__main__":
  print("Loading model...")
  model, tokenizer = load_model()
  print("Model loaded!\n")
  chat(model, tokenizer)
