"""
Wanda Pruning for Qwen3.5-4B (Text-Only)
==========================================

CLI entry point for pruning Qwen3.5 models using Wanda, SparseGPT,
or magnitude-based pruning.

Usage:
    # Wanda at 50% unstructured sparsity
    python main.py \
        --model ./Qwen3.5-4BText \
        --prune_method wanda \
        --sparsity_ratio 0.5 \
        --save ./Qwen3.5-4BText-pruned/

    # 2:4 structured sparsity (for tensor core acceleration)
    python main.py \
        --model ./Qwen3.5-4BText \
        --prune_method wanda \
        --sparsity_type 2:4 \
        --save ./Qwen3.5-4BText-2to4/

    # SparseGPT (slower but more accurate)
    python main.py \
        --model ./Qwen3.5-4BText \
        --prune_method sparsegpt \
        --sparsity_ratio 0.5 \
        --save ./Qwen3.5-4BText-sparsegpt/

Requirements:
    pip install torch transformers accelerate safetensors datasets
    # Latest transformers for Qwen3.5 support:
    pip install git+https://github.com/huggingface/transformers.git@main
"""

import argparse
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from importlib.metadata import version

from lib.prune import (
    prune_wanda,
    prune_magnitude,
    prune_sparsegpt,
    check_sparsity,
    find_layers,
    get_layer_type,
)
from lib.structured import prune_structured
from lib.distill import distill
from lib.eval import eval_ppl
from lib.data import get_loaders


def print_versions():
    """Print library versions for reproducibility."""
    for pkg in ("torch", "transformers", "accelerate", "safetensors"):
        try:
            print(f"  {pkg}: {version(pkg)}")
        except Exception:
            print(f"  {pkg}: not installed")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  MPS available: {torch.backends.mps.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPUs: {torch.cuda.device_count()}")


def get_device():
    """Detect the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_model(model_path, cache_dir=None):
    """Load Qwen3.5 text model and tokenizer."""
    print(f"\nLoading model: {model_path}")

    device = get_device()
    print(f"  Device: {device}")

    # device_map="auto" doesn't work reliably on MPS (leaves placeholder tensors).
    # With multi-GPU, pin to cuda:0 so distillation can later move student to cuda:1.
    if device.type == "cuda":
        num_gpus = torch.cuda.device_count()
        if num_gpus >= 2:
            # Pin to GPU 0 — distillation will move student to GPU 1 later
            print(f"  Multi-GPU detected ({num_gpus} GPUs), pinning model to cuda:0")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map={"": "cuda:0"},
                cache_dir=cache_dir,
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                cache_dir=cache_dir,
                trust_remote_code=True,
            )
    else:
        # MPS or CPU: load to CPU, then move
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,  # MPS has limited float16 support
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        model = model.to(device)

    # Set seqlen for calibration
    if hasattr(model.config, "max_position_embeddings"):
        model.seqlen = model.config.max_position_embeddings
    else:
        model.seqlen = 4096  # Conservative default
        print(f"  Warning: max_position_embeddings not found, using {model.seqlen}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    return model, tokenizer


def print_model_info(model):
    """Print model architecture summary relevant to pruning."""
    layers = model.model.layers
    n_layers = len(layers)

    gdn_count = sum(1 for l in layers if get_layer_type(l) == "gated_deltanet")
    attn_count = sum(1 for l in layers if get_layer_type(l) == "attention")

    print(f"\n  Architecture: Qwen3.5 Hybrid")
    print(f"  Total decoder layers: {n_layers}")
    print(f"    GatedDeltaNet layers: {gdn_count}")
    print(f"    Standard Attention layers: {attn_count}")
    print(f"  Pattern: {gdn_count // (n_layers // 4)}x GatedDeltaNet -> 1x Attention (repeating)")

    # Count prunable parameters
    total_params = 0
    layer_params = {}
    for idx, layer in enumerate(layers):
        subset = find_layers(layer)
        count = sum(m.weight.numel() for m in subset.values())
        total_params += count
        layer_params[idx] = count

    print(f"  Total prunable params: {total_params:,} ({total_params / 1e9:.2f}B)")


def main():
    parser = argparse.ArgumentParser(
        description="Wanda Pruning for Qwen3.5 (Text-Only)"
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to Qwen3.5 text-only model or HuggingFace model ID",
    )
    parser.add_argument("--cache_dir", type=str, default=None)

    # Pruning method
    parser.add_argument(
        "--prune_method",
        type=str,
        default="wanda",
        choices=["wanda", "magnitude", "sparsegpt", "structured"],
        help="Pruning method",
    )

    # Sparsity (for unstructured methods)
    parser.add_argument(
        "--sparsity_ratio",
        type=float,
        default=0.5,
        help="Target sparsity ratio (0-1). Default: 0.5",
    )
    parser.add_argument(
        "--sparsity_type",
        type=str,
        default="unstructured",
        choices=["unstructured", "2:4", "4:8"],
        help="Sparsity pattern. Default: unstructured",
    )

    # Structured pruning options
    parser.add_argument(
        "--mlp_ratio",
        type=float,
        default=0.5,
        help="Fraction of MLP neurons to REMOVE (structured only). Default: 0.5",
    )
    parser.add_argument(
        "--attn_ratio",
        type=float,
        default=0.0,
        help="Fraction of attention heads to REMOVE (structured only). Default: 0.0 (disabled)",
    )
    parser.add_argument(
        "--prune_attn",
        action="store_true",
        help="Also prune attention/deltanet heads (structured only)",
    )

    # Calibration
    parser.add_argument(
        "--calib_dataset",
        type=str,
        default="c4",
        help="Calibration dataset: c4, wikitext2, ptb, or path to text file",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=128,
        help="Number of calibration samples. Default: 128",
    )
    parser.add_argument(
        "--seqlen",
        type=int,
        default=2048,
        help="Calibration sequence length. Default: 2048",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    # Evaluation
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate perplexity after pruning",
    )
    parser.add_argument(
        "--eval_dataset",
        type=str,
        default="wikitext2",
        help="Evaluation dataset. Default: wikitext2",
    )

    # Output
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Directory to save the pruned model",
    )

    # Advanced
    parser.add_argument(
        "--exclude_layers",
        type=str,
        nargs="*",
        default=None,
        help="Layer name patterns to exclude from pruning (e.g. 'norm' 'embed')",
    )

    # Distillation (for structured pruning recovery)
    parser.add_argument(
        "--distill",
        action="store_true",
        help="Run KL distillation after pruning to recover quality",
    )
    parser.add_argument(
        "--teacher",
        type=str,
        default=None,
        help="Path to teacher (original unpruned) model for distillation. "
             "Defaults to --model if not specified.",
    )
    parser.add_argument(
        "--distill_steps",
        type=int,
        default=1000,
        help="Number of distillation steps. Default: 1000",
    )
    parser.add_argument(
        "--distill_lr",
        type=float,
        default=2e-5,
        help="Distillation learning rate. Default: 2e-5",
    )
    parser.add_argument(
        "--distill_temperature",
        type=float,
        default=2.0,
        help="KL temperature. Default: 2.0",
    )
    parser.add_argument(
        "--distill_alpha",
        type=float,
        default=0.7,
        help="KL vs CE balance (1.0=pure KL, 0.0=pure CE). Default: 0.7",
    )
    parser.add_argument(
        "--distill_seqlen",
        type=int,
        default=512,
        help="Sequence length for distillation (shorter=faster, less VRAM). Default: 512",
    )
    parser.add_argument(
        "--gradient_accumulation",
        type=int,
        default=4,
        help="Gradient accumulation steps. Default: 4",
    )

    args = parser.parse_args()

    # ── Setup ─────────────────────────────────────────────────────────────
    print("=" * 60)
    print("Wanda Pruning for Qwen3.5")
    print("=" * 60)
    print_versions()

    device = get_device()

    # ── Load model ────────────────────────────────────────────────────────
    model, tokenizer = load_model(args.model, args.cache_dir)
    print_model_info(model)

    # ── Pre-pruning evaluation ────────────────────────────────────────────
    if args.eval:
        print("\n--- Pre-pruning evaluation ---")
        ppl_before = eval_ppl(model, tokenizer, args.eval_dataset, args.seqlen, device)

    # ── Calibration data ──────────────────────────────────────────────────
    if args.prune_method in ("wanda", "sparsegpt", "structured"):
        print(f"\nLoading calibration data: {args.calib_dataset}")
        print(f"  nsamples={args.nsamples}, seqlen={args.seqlen}, seed={args.seed}")

        calib_data = get_loaders(
            args.calib_dataset,
            nsamples=args.nsamples,
            seed=args.seed,
            seqlen=args.seqlen,
            tokenizer=tokenizer,
        )
        print(f"  Loaded {len(calib_data)} calibration samples")

    # ── Prune ─────────────────────────────────────────────────────────────
    print(f"\n--- Pruning with {args.prune_method} ---")
    t0 = time.time()

    if args.prune_method == "wanda":
        prune_wanda(
            model,
            tokenizer,
            calib_data,
            device,
            sparsity_ratio=args.sparsity_ratio,
            sparsity_type=args.sparsity_type,
            exclude_layers=args.exclude_layers,
        )
    elif args.prune_method == "sparsegpt":
        prune_sparsegpt(
            model,
            tokenizer,
            calib_data,
            device,
            sparsity_ratio=args.sparsity_ratio,
            sparsity_type=args.sparsity_type,
        )
    elif args.prune_method == "magnitude":
        prune_magnitude(
            model,
            sparsity_ratio=args.sparsity_ratio,
            sparsity_type=args.sparsity_type,
        )
    elif args.prune_method == "structured":
        prune_structured(
            model,
            tokenizer,
            calib_data,
            device,
            mlp_ratio=args.mlp_ratio,
            attn_ratio=args.attn_ratio,
            prune_mlp=True,
            prune_attn=args.prune_attn,
        )

    elapsed = time.time() - t0
    print(f"\nPruning completed in {elapsed:.1f}s")

    # ── Post-pruning evaluation (before distillation) ─────────────────────
    if args.eval:
        print("\n--- Post-pruning evaluation (before distillation) ---")
        ppl_after_prune = eval_ppl(model, tokenizer, args.eval_dataset, args.seqlen, device)
        if 'ppl_before' in locals():
            print(f"\n  PPL before pruning: {ppl_before:.2f}")
        print(f"  PPL after pruning:  {ppl_after_prune:.2f}")

    # ── Distillation ──────────────────────────────────────────────────────
    if args.distill:
        teacher_path = args.teacher if args.teacher else args.model
        print(f"\n--- KL Divergence Distillation ---")

        distill(
            teacher_path=teacher_path,
            student_model=model,
            tokenizer=tokenizer,
            device=device,
            calib_dataset=args.calib_dataset,
            nsamples=max(args.nsamples, 256),
            seqlen=args.distill_seqlen,
            seed=args.seed,
            num_steps=args.distill_steps,
            lr=args.distill_lr,
            temperature=args.distill_temperature,
            alpha=args.distill_alpha,
            save_every=args.distill_steps // 2,  # Checkpoint halfway
            save_dir=args.save,
            gradient_accumulation=args.gradient_accumulation,
        )

        # Post-distillation evaluation
        # After multi-GPU distillation, model may be on cuda:1 not cuda:0
        if args.eval:
            print("\n--- Post-distillation evaluation ---")
            model_device = next(model.parameters()).device
            ppl_after_distill = eval_ppl(model, tokenizer, args.eval_dataset, args.seqlen, model_device)
            print(f"\n  PPL after pruning:       {ppl_after_prune:.2f}")
            print(f"  PPL after distillation:  {ppl_after_distill:.2f}")
            print(f"  Recovery:                {ppl_after_prune - ppl_after_distill:+.2f}")
    else:
        if args.eval and 'ppl_before' in locals():
            print(f"\n  PPL before: {ppl_before:.2f}")
            print(f"  PPL after:  {ppl_after_prune:.2f}")
            print(f"  Δ PPL:      {ppl_after_prune - ppl_before:+.2f}")

    # ── Save ──────────────────────────────────────────────────────────────
    if args.save:
        print(f"\nSaving pruned model to: {args.save}")
        os.makedirs(args.save, exist_ok=True)

        # Cast back to bfloat16 before saving — the original model is bf16,
        # but on MPS we load as float32 for compatibility. Without this,
        # the saved model would be 2x larger than the original.
        model = model.to(dtype=torch.bfloat16, device="cpu")

        model.save_pretrained(args.save)
        tokenizer.save_pretrained(args.save)
        print("  Done.")

    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
