# LLM

Complete implementations of large language models including all sub-components.


## Repository Structure

```
pytorch/
├── gpt/               # GPT-1 style implementation
└── llama/             # LLaMA-1/2 implementation
```

## What's Implemented

**GPT** (`pytorch/gpt/`):
- Multi-head self-attention with causal masking
- Learned positional embeddings
- LayerNorm, feedforward blocks
- Training loop with loss estimation

**LLaMA** (`pytorch/llama/`):
- Multi-head attention with Rotary Position Embeddings (RoPE)
- RMSNorm (instead of LayerNorm)
- SwiGLU feedforward network
- Top-p sampling for generation
- SentencePiece tokenizer

## Usage

**GPT:**
```bash
cd pytorch/gpt
python train.py
```

**LLaMA:**
```bash
cd pytorch/llama
python generate.py
```

## Default Configurations

| Parameter | GPT | LLaMA |
|-----------|-----|-------|
| Embedding dim | 384 | 4096 |
| Hidden dim | - | 11008 |
| Heads | 6 | 32 |
| Layers | 6 | 32 |
| Context length | 256 | 2048 |
| Dropout | 0.2 | 0.0 |

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al., 2017
- [Layer Normalization](https://arxiv.org/abs/1607.06450) - Ba et al., 2016
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) - Zhang & Sennrich, 2019
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) - Su et al., 2021
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) - Touvron et al., 2023
- [LLaMA 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288) - Touvron et al., 2023
