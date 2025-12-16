from dataclasses import dataclass

import torch


@dataclass
class ModelConfig:
  vocab_size: int
  n_embd=384
  n_head=6
  n_layer=6
  block_size=256
  dropout=0.2

@dataclass
class TrainingConfig:
  max_steps=5000
  eval_iters = 200
  eval_interval = 1000
  batch_size=64
  lr=3e-4
  device="cuda" if torch.cuda.is_available() else "cpu"
