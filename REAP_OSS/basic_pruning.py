#!/usr/bin/env python3
"""
Example: Basic REAP Pruning for GPT-OSS 20B

This script demonstrates how to prune GPT-OSS 20B using REAP
with 50% compression ratio.

Requirements:
    - GPU with at least 48GB VRAM (for BF16 processing)
    - Or use device_map="auto" for multi-GPU setup
    
Usage:
    python examples/basic_pruning.py
"""

import torch
from pathlib import Path

# Add parent directory to path for local development
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reap_gptoss import (
    REAPPruner,
    PruningConfig,
    load_gptoss_model,
    print_model_info,
    create_gptoss_calibration_loader,
)


def main():
    # Configuration
    MODEL_NAME = "openai/gpt-oss-20b"
    COMPRESSION_RATIO = 0.5  # Remove 50% of experts
    OUTPUT_DIR = "./pruned_gptoss_20b"
    
    print("="*60)
    print("REAP Expert Pruning Example")
    print("="*60)
    
    # Step 1: Load the model
    print(f"\n[1/5] Loading model: {MODEL_NAME}")
    model = load_gptoss_model(
        MODEL_NAME,
        device_map="auto",
        use_bf16=True,
    )
    print_model_info(model)
    
    # Step 2: Create calibration dataloader
    print("\n[2/5] Setting up calibration data...")
    dataloader = create_gptoss_calibration_loader(
        model_name=MODEL_NAME,
        dataset_name="theblackcat102/evol-codealpaca-v1",
        batch_size=4,
        max_length=2048,
    )
    
    # Step 3: Configure pruning
    print("\n[3/5] Configuring REAP pruning...")
    config = PruningConfig(
        compression_ratio=COMPRESSION_RATIO,
        calibration_samples=512,
        method="reap",
        preserve_min_experts=2,
        seed=42,
    )
    
    pruner = REAPPruner(model, config)
    
    # Step 4: Compute saliency and select experts to prune
    print("\n[4/5] Computing REAP saliency scores...")
    saliency_scores = pruner.compute_saliency(dataloader, use_full_reap=True)
    
    result = pruner.select_experts_to_prune(saliency_scores)
    
    print(f"\nPruning plan:")
    print(f"  Original experts: {result.original_num_experts} per layer")
    print(f"  After pruning: {result.pruned_num_experts} per layer")
    print(f"  Compression: {result.compression_achieved:.1%}")
    
    # Show top-5 least salient experts per layer
    print("\nLeast salient experts (to be pruned):")
    for layer_idx in sorted(result.experts_to_prune.keys())[:3]:
        experts = result.experts_to_prune[layer_idx][:5]
        scores = [result.saliency_scores[layer_idx][e] for e in experts]
        print(f"  Layer {layer_idx}: {list(zip(experts, [f'{s:.4f}' for s in scores]))}")
    print("  ...")
    
    # Step 5: Apply pruning and save
    print("\n[5/5] Applying pruning and saving model...")
    pruned_model = pruner.prune_experts(result, inplace=False)
    
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    pruned_model.save_pretrained(output_path)
    
    # Save tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(output_path)
    
    print(f"\nPruned model saved to: {output_path}")
    
    # Verify reduction
    original_params = sum(p.numel() for p in model.parameters())
    pruned_params = sum(p.numel() for p in pruned_model.parameters())
    print(f"\nParameter reduction:")
    print(f"  Original: {original_params:,}")
    print(f"  Pruned: {pruned_params:,}")
    print(f"  Reduction: {(1 - pruned_params/original_params):.1%}")
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()
