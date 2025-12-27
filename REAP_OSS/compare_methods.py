#!/usr/bin/env python3
"""
Example: Compare Pruning Methods for GPT-OSS

This script compares REAP, frequency-based, and random pruning
on GPT-OSS 20B to demonstrate REAP's effectiveness.

Usage:
    python examples/compare_methods.py
"""

import torch
import json
from pathlib import Path
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reap_gptoss import (
    REAPPruner,
    PruningConfig,
    load_gptoss_model,
    create_gptoss_calibration_loader,
    compute_frequency_scores,
    compute_random_scores,
)


def evaluate_perplexity(model, dataloader, max_samples: int = 100):
    """Simple perplexity evaluation on calibration data."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_samples:
                break
                
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(model.device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            
            total_loss += outputs.loss.item() * input_ids.numel()
            total_tokens += input_ids.numel()
            
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity


def main():
    # Configuration
    MODEL_NAME = "openai/gpt-oss-20b"
    COMPRESSION_RATIO = 0.5
    OUTPUT_DIR = Path("./comparison_results")
    
    print("="*70)
    print("Pruning Method Comparison: REAP vs Frequency vs Random")
    print("="*70)
    
    # Load model
    print(f"\nLoading model: {MODEL_NAME}")
    model = load_gptoss_model(
        MODEL_NAME,
        device_map="auto",
        use_bf16=True,
    )
    
    # Create dataloader
    print("Loading calibration data...")
    dataloader = create_gptoss_calibration_loader(
        model_name=MODEL_NAME,
        batch_size=4,
        max_length=2048,
    )
    
    # Baseline perplexity
    print("\nComputing baseline perplexity...")
    baseline_ppl = evaluate_perplexity(model, dataloader)
    print(f"Baseline perplexity: {baseline_ppl:.2f}")
    
    results = {"baseline": baseline_ppl}
    
    # Test each method
    methods = ["reap", "frequency", "random"]
    
    for method in methods:
        print(f"\n{'='*70}")
        print(f"Testing method: {method.upper()}")
        print("="*70)
        
        # Compute saliency
        print("Computing saliency scores...")
        if method == "reap":
            config = PruningConfig(
                compression_ratio=COMPRESSION_RATIO,
                calibration_samples=512,
                method=method,
            )
            pruner = REAPPruner(model, config)
            saliency = pruner.compute_saliency(dataloader, use_full_reap=True)
        elif method == "frequency":
            config = PruningConfig(
                compression_ratio=COMPRESSION_RATIO,
                calibration_samples=512,
                method=method,
            )
            pruner = REAPPruner(model, config)
            saliency = compute_frequency_scores(model, dataloader, 512)
        else:  # random
            config = PruningConfig(
                compression_ratio=COMPRESSION_RATIO,
                method=method,
            )
            pruner = REAPPruner(model, config)
            saliency = compute_random_scores(model)
        
        # Select and prune
        print("Selecting experts to prune...")
        result = pruner.select_experts_to_prune(saliency)
        
        print(f"Pruning {result.compression_achieved:.1%} of experts...")
        pruned_model = pruner.prune_experts(result, inplace=False)
        
        # Evaluate
        print("Evaluating pruned model...")
        pruned_ppl = evaluate_perplexity(pruned_model, dataloader)
        
        retention = baseline_ppl / pruned_ppl if pruned_ppl > 0 else 0
        print(f"\nResults for {method.upper()}:")
        print(f"  Perplexity: {pruned_ppl:.2f}")
        print(f"  Degradation: {(pruned_ppl/baseline_ppl - 1)*100:.1f}%")
        
        results[method] = {
            "perplexity": pruned_ppl,
            "degradation": (pruned_ppl/baseline_ppl - 1)*100,
            "compression": result.compression_achieved,
        }
        
        # Clean up
        del pruned_model
        torch.cuda.empty_cache()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Method':<12} {'Perplexity':<12} {'Degradation':<12} {'Compression':<12}")
    print("-"*48)
    print(f"{'Baseline':<12} {baseline_ppl:<12.2f} {'-':<12} {'-':<12}")
    
    for method in methods:
        r = results[method]
        print(f"{method.upper():<12} {r['perplexity']:<12.2f} {r['degradation']:<+11.1f}% {r['compression']:<11.1%}")
    
    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {OUTPUT_DIR / 'comparison_results.json'}")
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    # Find best method
    best_method = min(methods, key=lambda m: results[m]["perplexity"])
    print(f"\nBest method: {best_method.upper()}")
    print(f"REAP typically outperforms other methods, especially at high compression.")


if __name__ == "__main__":
    main()
