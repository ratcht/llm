from dataclasses import dataclass

import torch


@dataclass
class ModelConfig:
  vocab_size: int = 32000
  embed_dim: int = 4096
  hidden_dim: int = 11008
  num_heads: int = 32
  num_blocks: int = 32
  max_seq_len: int = 2048
  dropout: float = 0.0

@dataclass
class TrainingConfig:
  max_steps=5000
  eval_iters = 200
  eval_interval = 1000
  batch_size=64
  lr=3e-4
  device="cuda" if torch.cuda.is_available() else "cpu"
