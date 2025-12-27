"""
Main entry point for REAP expert pruning on GPT-OSS models.

Example usage:
    python -m reap_gptoss.main \
        --model_name openai/gpt-oss-20b \
        --compression_ratio 0.5 \
        --output_dir ./pruned_model
"""

import argparse
import os
import json
from pathlib import Path
from typing import Optional
import torch

from .model_util import load_gptoss_model, print_model_info, get_model_attrs
from .data import create_gptoss_calibration_loader, DataConfig
from .prune import REAPPruner, PruningConfig, PruningResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="REAP Expert Pruning for GPT-OSS Models"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="openai/gpt-oss-20b",
        help="HuggingFace model identifier",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./pruned_model",
        help="Directory to save pruned model",
    )
    
    # Pruning arguments
    parser.add_argument(
        "--compression_ratio",
        type=float,
        default=0.5,
        help="Fraction of experts to prune (0.0-1.0)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="reap",
        choices=["reap", "frequency", "random"],
        help="Pruning method",
    )
    parser.add_argument(
        "--calibration_samples",
        type=int,
        default=512,
        help="Number of tokens for saliency estimation",
    )
    parser.add_argument(
        "--preserve_min_experts",
        type=int,
        default=2,
        help="Minimum experts to keep per layer",
    )
    
    # Data arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="theblackcat102/evol-codealpaca-v1",
        help="Calibration dataset",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for calibration",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    
    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Device placement strategy",
    )
    parser.add_argument(
        "--use_bf16",
        action="store_true",
        default=True,
        help="Use BF16 precision",
    )
    parser.add_argument(
        "--save_saliency",
        action="store_true",
        help="Save saliency scores to JSON",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n" + "="*60)
    print("REAP Expert Pruning for GPT-OSS")
    print("="*60 + "\n")
    
    # Load model
    print(f"Loading model: {args.model_name}")
    model = load_gptoss_model(
        args.model_name,
        device_map=args.device_map,
        use_bf16=args.use_bf16,
    )
    print_model_info(model)
    
    # Create calibration dataloader
    print(f"Loading calibration data from: {args.dataset_name}")
    dataloader = create_gptoss_calibration_loader(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        seed=args.seed,
    )
    
    # Create pruning config
    config = PruningConfig(
        compression_ratio=args.compression_ratio,
        calibration_samples=args.calibration_samples,
        method=args.method,
        preserve_min_experts=args.preserve_min_experts,
        seed=args.seed,
    )
    
    # Initialize pruner
    pruner = REAPPruner(model, config)
    
    # Compute saliency scores
    print(f"\nComputing {args.method.upper()} saliency scores...")
    if args.method == "reap":
        saliency_scores = pruner.compute_saliency(dataloader, use_full_reap=True)
    elif args.method == "frequency":
        from .prune import compute_frequency_scores
        saliency_scores = compute_frequency_scores(
            model, dataloader, args.calibration_samples
        )
    else:  # random
        from .prune import compute_random_scores
        saliency_scores = compute_random_scores(model, args.seed)
    
    # Select experts to prune
    print("\nSelecting experts to prune...")
    result = pruner.select_experts_to_prune(saliency_scores)
    
    print(f"\nPruning summary:")
    print(f"  Original experts per layer: {result.original_num_experts}")
    print(f"  Experts after pruning: {result.pruned_num_experts}")
    print(f"  Compression achieved: {result.compression_achieved:.1%}")
    
    # Print per-layer statistics
    print("\nPer-layer pruning:")
    for layer_idx, experts in sorted(result.experts_to_prune.items()):
        print(f"  Layer {layer_idx}: pruning {len(experts)} experts: {experts}")
    
    # Apply pruning
    print("\nApplying expert pruning...")
    pruned_model = pruner.prune_experts(result, inplace=False)
    
    # Save model
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving pruned model to: {output_path}")
    pruned_model.save_pretrained(output_path)
    
    # Save tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.save_pretrained(output_path)
    
    # Save pruning config and results
    config_path = output_path / "pruning_config.json"
    with open(config_path, "w") as f:
        json.dump({
            "model_name": args.model_name,
            "compression_ratio": args.compression_ratio,
            "method": args.method,
            "calibration_samples": args.calibration_samples,
            "original_num_experts": result.original_num_experts,
            "pruned_num_experts": result.pruned_num_experts,
            "compression_achieved": result.compression_achieved,
            "experts_pruned": {
                str(k): v for k, v in result.experts_to_prune.items()
            },
        }, f, indent=2)
    
    if args.save_saliency:
        saliency_path = output_path / "saliency_scores.json"
        with open(saliency_path, "w") as f:
            json.dump({
                str(layer): {str(exp): score for exp, score in experts.items()}
                for layer, experts in result.saliency_scores.items()
            }, f, indent=2)
        print(f"Saved saliency scores to: {saliency_path}")
    
    print("\n" + "="*60)
    print("Pruning complete!")
    print(f"Pruned model saved to: {output_path}")
    print("="*60 + "\n")
    
    return pruned_model


if __name__ == "__main__":
    main()
