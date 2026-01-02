#!/usr/bin/env python3
"""
LLM-Sieve Example: Pruning HuggingFace Models

Complete working example demonstrating:
1. Loading a model
2. Preparing calibration data
3. Running uniform or adaptive pruning
4. Evaluating and saving results

Usage:
    python example.py --model gpt2 --strategy uniform
    python example.py --model gpt2-medium --strategy adaptive --samples 50
"""

import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import sys
import os

# Add parent to path for local import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_sieve import (
    TransformerPruner,
    LLMSieveConfig,
    diagnose_model,
)


def parse_args():
    parser = argparse.ArgumentParser(description="LLM-Sieve Pruning Example")
    parser.add_argument("--model", type=str, default="gpt2",
                       help="HuggingFace model name")
    parser.add_argument("--strategy", type=str, default="uniform",
                       choices=["uniform", "adaptive", "simple"],
                       help="Pruning strategy (simple recommended for large models)")
    parser.add_argument("--factor", type=float, default=0.5,
                       help="Pruning factor for simple strategy (0.5 = 50%% compression)")
    parser.add_argument("--samples", type=int, default=100,
                       help="Number of calibration samples")
    parser.add_argument("--eval-samples", type=int, default=50,
                       help="Number of evaluation samples")
    parser.add_argument("--max-length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--accuracy-drop", type=float, default=0.05,
                       help="Target accuracy drop tolerance")
    parser.add_argument("--output-dir", type=str, default="./pruned_model",
                       help="Output directory for pruned model")
    parser.add_argument("--diagnose-only", action="store_true",
                       help="Only run model diagnostics")
    parser.add_argument("--device", type=str, default=None,
                       help="Device (cuda/cpu)")
    
    # Quantization options
    parser.add_argument("--load-in-4bit", action="store_true",
                       help="Load model in 4-bit quantization (requires bitsandbytes)")
    parser.add_argument("--load-in-8bit", action="store_true",
                       help="Load model in 8-bit quantization (requires bitsandbytes)")
    
    return parser.parse_args()


def load_model_and_tokenizer(model_name: str, device: str, load_in_4bit: bool = False, load_in_8bit: bool = False):
    """Load model and tokenizer from HuggingFace.
    
    Args:
        model_name: HuggingFace model name
        device: Device to load on (ignored if quantizing)
        load_in_4bit: Load in 4-bit quantization (requires bitsandbytes)
        load_in_8bit: Load in 8-bit quantization (requires bitsandbytes)
    """
    print(f"\nLoading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure quantization if requested
    load_kwargs = {}
    
    if load_in_4bit:
        print("  Loading in 4-bit quantization...")
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs['device_map'] = 'auto'
        except ImportError:
            raise ImportError(
                "4-bit loading requires bitsandbytes. "
                "Install with: pip install bitsandbytes"
            )
    elif load_in_8bit:
        print("  Loading in 8-bit quantization...")
        load_kwargs['load_in_8bit'] = True
        load_kwargs['device_map'] = 'auto'
    else:
        load_kwargs['torch_dtype'] = torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    
    # Move to device if not using quantization (quantized models use device_map)
    if not (load_in_4bit or load_in_8bit):
        model = model.to(device)
    
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    if load_in_4bit:
        print("  Model loaded in 4-bit NF4 quantization")
    elif load_in_8bit:
        print("  Model loaded in 8-bit quantization")
    
    return model, tokenizer


def prepare_calibration_data(tokenizer, num_samples: int, max_length: int):
    """Prepare calibration dataset from WikiText-2."""
    print(f"\nPreparing calibration data ({num_samples} samples)...")
    
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    
    # Filter non-empty texts
    texts = [t for t in dataset["text"] if len(t.strip()) > 50][:num_samples * 2]
    
    # Tokenize
    encodings = tokenizer(
        texts[:num_samples],
        max_length=max_length,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )
    
    return encodings


def create_dataloader(encodings, batch_size: int = 4):
    """Create DataLoader from encodings."""
    class SimpleDataset:
        def __init__(self, encodings):
            self.encodings = encodings
        
        def __len__(self):
            return self.encodings['input_ids'].size(0)
        
        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.encodings.items()}
    
    dataset = SimpleDataset(encodings)
    return DataLoader(dataset, batch_size=batch_size)


def evaluate_perplexity(model, dataloader, device: str) -> float:
    """Compute perplexity on evaluation data."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Shift for next-token prediction
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            
            # Sum losses weighted by token count
            n_tokens = (labels != -100).sum().item()
            total_loss += outputs.loss.item() * n_tokens
            total_tokens += n_tokens
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return perplexity


def main():
    args = parse_args()
    
    # Determine device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    
    # Check for conflicting options
    if args.load_in_4bit and args.load_in_8bit:
        print("Error: Cannot use both --load-in-4bit and --load-in-8bit")
        return
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(
        args.model, device, 
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit
    )
    
    # Update device for quantized models
    if args.load_in_4bit or args.load_in_8bit:
        device = next(model.parameters()).device
        print(f"Model loaded on: {device}")
    
    # Diagnose model structure
    print("\n" + "=" * 60)
    print("RUNNING MODEL DIAGNOSTICS")
    print("=" * 60)
    diag = diagnose_model(model, args.model)
    
    if args.diagnose_only:
        print("\nDiagnostics complete. Use without --diagnose-only to run pruning.")
        return
    
    if diag["prunable"] == 0:
        print("\n❌ ERROR: No prunable layers found!")
        print("This likely means layer detection failed.")
        print("Please check the model architecture and layer naming.")
        return
    
    # Prepare data
    calibration_encodings = prepare_calibration_data(
        tokenizer, args.samples, args.max_length
    )
    eval_encodings = prepare_calibration_data(
        tokenizer, args.eval_samples, args.max_length
    )
    
    calibration_loader = create_dataloader(calibration_encodings)
    eval_loader = create_dataloader(eval_encodings)
    
    # Compute baseline perplexity
    print("\nComputing baseline perplexity...")
    baseline_ppl = evaluate_perplexity(model, eval_loader, device)
    print(f"Baseline perplexity: {baseline_ppl:.2f}")
    
    # Create evaluation function (convert perplexity to accuracy-like metric)
    # Lower perplexity is better, so we invert for the pruner
    def eval_fn(m):
        ppl = evaluate_perplexity(m, eval_loader, device)
        # Convert to accuracy-like metric (higher is better)
        # Using 1/log(ppl) so lower perplexity = higher "accuracy"
        import math
        return 1.0 / math.log(max(ppl, 1.01))
    
    baseline_accuracy = eval_fn(model)
    
    # Configure pruning
    config = LLMSieveConfig(
        pruning_strategy=args.strategy,
        target_accuracy_drop=args.accuracy_drop,
        calibration_tokens=args.samples * args.max_length,
        device=device,
    )
    
    # Create pruner
    pruner = TransformerPruner(model, config)
    
    # Run pruning
    print(f"\n{'=' * 60}")
    print(f"RUNNING {args.strategy.upper()} PRUNING")
    print(f"{'=' * 60}")
    
    if args.strategy == "simple":
        # Simple mode: just apply a fixed factor, most memory efficient
        factors = pruner.simple_prune(
            calibration_loader,
            pruning_factor=args.factor,
        )
    elif args.strategy == "uniform":
        factors = pruner.uniform_prune(
            calibration_loader,
            eval_fn,
            baseline_accuracy,
            memory_efficient=True,  # Use memory-efficient mode by default
        )
    else:
        factors = pruner.adaptive_prune(
            calibration_loader,
            eval_fn,
            baseline_accuracy,
        )
    
    # Apply pruning
    pruner.apply_pruning()
    
    # Evaluate final model
    print("\nEvaluating pruned model...")
    final_ppl = evaluate_perplexity(model, eval_loader, device)
    print(f"Final perplexity: {final_ppl:.2f}")
    print(f"Perplexity increase: {(final_ppl - baseline_ppl) / baseline_ppl * 100:.1f}%")
    
    # Print statistics
    pruner.print_statistics()
    
    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    pruner.save_pruned_model(args.output_dir)
    
    print(f"\n✅ Pruning complete! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
