# LLM-Sieve: Task-Specific Pruning for Large Language Models

Implementation of the paper **"How Many Parameters Does Your Task Really Need?"** 
by Waleed Reda, Abhinav Jangda, Krishna Chintalapudi (Microsoft Research, 2025)

## Key Innovation

LLM-Sieve achieves 20-75% parameter reduction with minimal accuracy loss by:

1. **Output-aligned projections**: Instead of SVD (which creates orthogonal bases not aligned with task outputs), we learn non-orthogonal adapter matrices that minimize actual reconstruction error ||Y - Ỹ||²

2. **Task-specific calibration**: Pruning is calibrated on your specific task data, not generic text

3. **Adaptive per-layer compression**: Different layers have different redundancy for your task

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Layer Discovery (Important!)

First, verify that layer detection works for your model:

```bash
python diagnose.py gpt2
```

Expected output:
```
Total linear layers: 50
By type:
  attention   :  24 layers, 14,155,776 params (56.1%)
  mlp         :  24 layers, 9,437,184 params (37.4%)
  head        :   1 layers, 1,601,536 params (6.4%)
  embedding   :   0 layers, 0 params (0.0%)

Prunable layers (attention + mlp): 48
✓ Ready for pruning!
```

### 3. Run Pruning

```bash
# Quick test with GPT-2
python example.py --model gpt2 --strategy uniform --samples 50

# With more samples for better quality
python example.py --model gpt2 --strategy uniform --samples 200 --eval-samples 100

# Adaptive pruning (slower but better compression)
python example.py --model gpt2-medium --strategy adaptive --samples 100
```

## API Usage

### Basic Pruning

```python
from transformers import AutoModelForCausalLM
from llm_sieve import TransformerPruner, LLMSieveConfig

# Load model
model = AutoModelForCausalLM.from_pretrained('gpt2')

# Configure pruning
config = LLMSieveConfig(
    pruning_strategy="uniform",
    target_accuracy_drop=0.05,  # 5% accuracy tolerance
)

# Create pruner
pruner = TransformerPruner(model, config)

# Run pruning (you need calibration data and evaluation function)
factors = pruner.uniform_prune(calibration_loader, eval_fn, baseline_accuracy)

# Apply and save
pruner.apply_pruning()
pruner.save_pruned_model("./pruned")
```

### Diagnosing Model Structure

```python
from llm_sieve import diagnose_model

# Check if layer discovery works
results = diagnose_model(model, "gpt2")
print(f"Prunable layers: {results['prunable']}")
```

## File Structure

```
llm_sieve/
├── __init__.py          # Package exports
├── core.py              # Low-rank adapter learning (MatrixPruner, LowRankAdapter)
├── layer_discovery.py   # Robust layer detection (LayerDiscovery)
├── pruner.py            # High-level interface (TransformerPruner)
├── genetic.py           # Genetic algorithm for adaptive pruning
├── example.py           # Complete working example
├── diagnose.py          # Model diagnostics
└── requirements.txt     # Dependencies
```

## Configuration Options

### LLMSieveConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `calibration_tokens` | 200,000 | Tokens for calibration |
| `batch_size` | 5000 | Batch size for adapter training |
| `learning_rate` | 0.001 | Learning rate |
| `num_epochs` | 2 | Epochs for adapter training |
| `pruning_strategy` | "uniform" | "uniform" or "adaptive" |
| `target_accuracy_drop` | 0.05 | Acceptable accuracy loss |
| `prune_attention` | True | Prune attention layers |
| `prune_mlp` | True | Prune MLP layers |
| `ga_population_size` | 100 | GA population (adaptive) |
| `ga_generations` | 30 | GA generations (adaptive) |

## How It Works

### 1. Low-Rank Approximation

For a weight matrix W ∈ ℝ^{H×D}, we learn adapter A ∈ ℝ^{D×R}:

- Original: `Y = W @ X`
- Pruned: `Y ≈ (W @ A) @ (A^T @ X)`

The adapter A is trained to minimize ||Y - Ỹ||² on your calibration data.

### 2. Pruning Factor

The pruning factor p determines compression:

```
p = R(H + D) / (HD)
```

Where R is the target rank. Compression ratio = 1 - p.

### 3. Uniform vs Adaptive

- **Uniform**: Same pruning factor for all layers. Uses binary search to find the best factor.
- **Adaptive**: Different factors per layer. Uses genetic algorithm to find optimal configuration.

## Troubleshooting

### "No prunable layers found"

This means layer naming patterns don't match your model. Run `diagnose.py` to see the actual layer names, then check if they're being classified correctly in `layer_discovery.py`.

### Out of Memory

- Reduce `calibration_tokens` in config
- Use smaller `batch_size`
- Run on CPU with `--device cpu`

### Poor Compression Results

- Increase calibration samples
- Try adaptive pruning
- Larger models generally compress better

## References

- Paper: [How Many Parameters Does Your Task Really Need?](https://arxiv.org/abs/2505.18350)
- Authors: Waleed Reda, Abhinav Jangda, Krishna Chintalapudi (Microsoft Research)

## License

MIT License
