# REAP-GPT-OSS: Expert Pruning for GPT-OSS Models

This package adapts the [REAP (Router-weighted Expert Activation Pruning)](https://github.com/CerebrasResearch/reap) method from Cerebras Research for use with OpenAI's GPT-OSS mixture-of-experts models.

## Overview

REAP is a one-shot expert pruning method that compresses Sparse Mixture-of-Experts (SMoE) models by removing low-impact experts while preserving model quality. This implementation is specifically designed for:

- **GPT-OSS 20B**: 24 layers, 32 experts, top-4 routing (21B total / 3.6B active)
- **GPT-OSS 120B**: 36 layers, 128 experts, top-4 routing (117B total / 5.1B active)

## Installation

```bash
# Install from source
pip install -e .

# Or with development dependencies
pip install -e ".[dev]"
```

### Requirements

- Python >= 3.10
- PyTorch >= 2.0
- Transformers >= 4.40
- CUDA-capable GPU (recommended: H100 for 120B, RTX 4090+ for 20B)

## Quick Start

### Command Line

```bash
# Prune GPT-OSS 20B with 50% compression
reap-gptoss \
    --model_name openai/gpt-oss-20b \
    --compression_ratio 0.5 \
    --output_dir ./gptoss-20b-reap-50

# Prune with custom calibration dataset
reap-gptoss \
    --model_name openai/gpt-oss-20b \
    --compression_ratio 0.3 \
    --dataset_name bigcode/the-stack-dedup \
    --calibration_samples 1024 \
    --output_dir ./gptoss-20b-reap-30
```

### Python API

```python
from reap_gptoss import REAPPruner, PruningConfig
from reap_gptoss.model_util import load_gptoss_model, print_model_info
from reap_gptoss.data import create_gptoss_calibration_loader

# Load model
model = load_gptoss_model("openai/gpt-oss-20b")
print_model_info(model)

# Create calibration dataloader
dataloader = create_gptoss_calibration_loader(
    model_name="openai/gpt-oss-20b",
    dataset_name="theblackcat102/evol-codealpaca-v1",
    batch_size=4,
)

# Configure pruning
config = PruningConfig(
    compression_ratio=0.5,  # Remove 50% of experts
    calibration_samples=512,
    method="reap",
)

# Run REAP pruning
pruner = REAPPruner(model, config)
saliency_scores = pruner.compute_saliency(dataloader)
result = pruner.select_experts_to_prune(saliency_scores)
pruned_model = pruner.prune_experts(result)

# Save pruned model
pruned_model.save_pretrained("./pruned_gptoss_20b")
```

## GPT-OSS Architecture Details

GPT-OSS models use a MoE architecture with the following characteristics:

| Parameter | GPT-OSS 20B | GPT-OSS 120B |
|-----------|-------------|--------------|
| Total Parameters | 21B | 117B |
| Active Parameters | 3.6B | 5.1B |
| Layers | 24 | 36 |
| Experts per Layer | 32 | 128 |
| Active Experts (top-k) | 4 | 4 |
| Hidden Dimension | 2880 | 2880 |
| Attention | GQA (64 heads, 8 KV) | GQA (64 heads, 8 KV) |
| Quantization | MXFP4 (MoE) | MXFP4 (MoE) |

## REAP Methodology

REAP computes expert saliency based on two factors:

1. **Router Gate Values**: How frequently and strongly the router activates each expert
2. **Expert Output Norms**: The magnitude of each expert's contribution to the layer output

The saliency score for expert $e$ is:

$$\text{REAP}(e) = \frac{1}{|T_e|} \sum_{t \in T_e} g_t^{(e)} \cdot \|E_e(x_t)\|$$

Where:
- $T_e$ is the set of tokens routed to expert $e$
- $g_t^{(e)}$ is the gate value for expert $e$ at token $t$
- $E_e(x_t)$ is the expert output for input $x_t$

Experts with the lowest REAP scores are pruned.

## Pruning Methods

The package supports three pruning methods:

| Method | Description |
|--------|-------------|
| `reap` | Full REAP with gate values and output norms (recommended) |
| `frequency` | Prune least frequently activated experts |
| `random` | Random expert selection (baseline) |

## Expected Results

Based on the original REAP paper, at 50% compression on code generation tasks:

| Model | Task | Baseline | REAP 50% | Retention |
|-------|------|----------|----------|-----------|
| Qwen3-30B-A3B | LiveCodeBench | 100% | 95.9% | 95.9% |
| GLM-4.5-Air | LiveCodeBench | 100% | 94.1% | 94.1% |

GPT-OSS should show similar behavior given its comparable MoE architecture.

## Running Inference

After pruning, use vLLM for efficient inference:

```bash
vllm serve ./gptoss-20b-reap-50 \
    --tensor-parallel-size 1 \
    --trust-remote-code
```

Or with the original GPT-OSS tools:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "./gptoss-20b-reap-50",
    device_map="auto",
    torch_dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained("./gptoss-20b-reap-50")

# Use as normal
```

## Handling MXFP4 Quantization

GPT-OSS uses MXFP4 quantization for MoE weights. This package:

1. Loads weights in BF16 for saliency computation
2. Preserves quantization format when saving (if supported by transformers)

For maximum compatibility, consider converting to BF16 before pruning:

```python
model = load_gptoss_model(
    "openai/gpt-oss-20b",
    use_bf16=True,  # Upcast MXFP4 to BF16
)
```

## Differences from Original REAP

This adaptation makes the following modifications for GPT-OSS compatibility:

1. **Model Attributes**: Added GPT-OSS specific attribute mappings for MoE layers
2. **Router Handling**: Adapted for GPT-OSS's router implementation
3. **Quantization**: Handles MXFP4 weight format used by GPT-OSS
4. **Data Loading**: Optimized for code-focused calibration matching GPT-OSS's training

## Citation

If you use this work, please cite the original REAP paper:

```bibtex
@misc{lasby-reap,
    title       = {{REAP the Experts: Why Pruning Prevails for One-Shot MoE compression}},
    author      = {Lasby, Mike and Lazarevich, Ivan and Sinnadurai, Nish and Lie, Sean and Ioannou, Yani and Thangarasa, Vithursan},
    year        = {2025},
    publisher   = {arXiv},
    note        = {arXiv:2510.13999v1 [cs]},
    url         = {https://arxiv.org/abs/2510.13999v1}, 
}
```

## License

Apache 2.0 (following the original REAP repository)

## Acknowledgments

- Original REAP implementation: [CerebrasResearch/reap](https://github.com/CerebrasResearch/reap)
- GPT-OSS models: [openai/gpt-oss](https://github.com/openai/gpt-oss)
