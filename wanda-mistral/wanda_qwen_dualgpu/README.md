# Wanda Pruning for Qwen3.5 (Text-Only)

Adaptation of **Wanda** (Pruning by **W**eights **and A**ctivations) for the
**Qwen3.5** hybrid architecture (GatedDeltaNet + Attention).

Based on: *A Simple and Effective Pruning Approach for Large Language Models*  
Sun et al., 2023 — [Paper](https://arxiv.org/abs/2306.11695) | [Original repo](https://github.com/locuslab/wanda)

## Qwen3.5 Architecture Notes

Qwen3.5 uses a **hybrid** decoder stack (not uniform self-attention like LLaMA/Mistral):

```
Pattern (repeating 8 times for the 4B model):
  3 × GatedDeltaNet (linear attention)  →  1 × Standard GQA Attention
```

**GatedDeltaNet layers** contain:
- `linear_attn.in_proj_qkv` (2560 → 8192)
- `linear_attn.in_proj_z` (2560 → 4096)
- `linear_attn.in_proj_a` (2560 → 32)
- `linear_attn.in_proj_b` (2560 → 32)
- `linear_attn.out_proj` (4096 → 2560)
- `linear_attn.conv1d` ← **not pruned** (Conv1d, not Linear)

**Standard Attention layers** contain:
- `self_attn.q_proj` (2560 → 8192)
- `self_attn.k_proj` (2560 → 1024)
- `self_attn.v_proj` (2560 → 1024)
- `self_attn.o_proj` (4096 → 2560)

**All layers** share the same MLP:
- `mlp.gate_proj` (2560 → 9216)
- `mlp.up_proj` (2560 → 9216)
- `mlp.down_proj` (9216 → 2560)

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Wanda (fast, no gradient needed)

```bash
# 50% unstructured sparsity with C4 calibration
python main.py \
    --model ./Qwen3.5-4BText \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --calib_dataset c4 \
    --nsamples 128 \
    --eval \
    --save ./Qwen3.5-4BText-wanda50/

# 2:4 structured sparsity (for NVIDIA tensor core acceleration)
python main.py \
    --model ./Qwen3.5-4BText \
    --prune_method wanda \
    --sparsity_type 2:4 \
    --eval \
    --save ./Qwen3.5-4BText-wanda-2to4/
```

### SparseGPT (slower, more accurate)

```bash
python main.py \
    --model ./Qwen3.5-4BText \
    --prune_method sparsegpt \
    --sparsity_ratio 0.5 \
    --eval \
    --save ./Qwen3.5-4BText-sparsegpt50/
```

### Magnitude baseline

```bash
python main.py \
    --model ./Qwen3.5-4BText \
    --prune_method magnitude \
    --sparsity_ratio 0.5 \
    --eval
```

### Custom calibration data

```bash
# Use a text file (one document per line)
python main.py \
    --model ./Qwen3.5-4BText \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --calib_dataset /path/to/your_data.txt \
    --nsamples 128
```

## Files

```
wanda_qwen/
├── main.py              # CLI entry point
├── lib/
│   ├── prune.py         # Core: Wanda, SparseGPT, Magnitude pruning
│   ├── layerwrapper.py  # WrappedGPT for activation capture
│   ├── data.py          # Calibration data loaders (C4, WikiText-2, PTB, custom)
│   └── eval.py          # Perplexity evaluation
├── requirements.txt
└── README.md
```

## Methods

| Method | Speed | Quality | Needs Calibration Data | Notes |
|--------|-------|---------|----------------------|-------|
| `magnitude` | Fastest | Baseline | No | Weight magnitude only |
| `wanda` | Fast (~2-5 min) | Good | Yes | |W| × ‖X‖₂ importance |
| `sparsegpt` | Slow (~15-30 min) | Best | Yes | Hessian-based error compensation |

## Acknowledgments

- Original Wanda: [locuslab/wanda](https://github.com/locuslab/wanda)
- Qwen3.5: [Qwen Team](https://qwen.ai/blog?id=qwen3.5)
