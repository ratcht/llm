# pyright: reportAttributeAccessIssue=false

if __name__=="__main__":
  import sentencepiece as spm

  sp = spm.SentencePieceProcessor()
  sp.load("torch/llama/tokenizer.model")

  tokens = sp.encode("Hello, world!")
  print(tokens)

  text = sp.decode(tokens)
  print(text)

  print(sp.get_piece_size())
