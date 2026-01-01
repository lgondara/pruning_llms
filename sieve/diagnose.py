#!/usr/bin/env python3
"""
Quick Diagnostic: Verify Layer Discovery Works

Run this BEFORE attempting any pruning to confirm the model
structure is being correctly parsed.

Usage:
    python diagnose.py gpt2
    python diagnose.py meta-llama/Llama-2-7b-hf
"""

import sys
import torch
import torch.nn as nn
from collections import defaultdict


def recursive_find_linear(module: nn.Module, prefix: str = "") -> dict:
    """Recursively find all Linear layers."""
    layers = {}
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            layers[full_name] = child
        else:
            layers.update(recursive_find_linear(child, full_name))
    return layers


def classify_layer(name: str) -> str:
    """Classify layer by its role."""
    name_lower = name.lower()
    
    # Check parent context
    if '.attn.' in name_lower or '.attention.' in name_lower or '.self_attn.' in name_lower:
        return 'attention'
    if '.mlp.' in name_lower or '.ffn.' in name_lower:
        return 'mlp'
    
    # Check name patterns
    layer_name = name.split('.')[-1].lower()
    
    if layer_name in ['c_attn', 'q_proj', 'k_proj', 'v_proj']:
        return 'attention'
    if layer_name in ['c_fc', 'fc1', 'gate_proj', 'up_proj']:
        return 'mlp'
    if layer_name in ['c_proj', 'o_proj', 'fc2', 'down_proj']:
        # c_proj can be either - check parent
        if 'attn' in name_lower:
            return 'attention'
        elif 'mlp' in name_lower:
            return 'mlp'
        return 'mlp'  # Default for projection
    
    if 'lm_head' in name_lower or 'classifier' in name_lower:
        return 'head'
    if 'embed' in name_lower or 'wte' in name_lower:
        return 'embedding'
    
    return 'other'


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
    
    print(f"\n{'=' * 60}")
    print(f"DIAGNOSING MODEL: {model_name}")
    print('=' * 60)
    
    # Load model
    print(f"\nLoading model...")
    from transformers import AutoModelForCausalLM
    
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Method 1: named_modules
    print("\n[Method 1] named_modules():")
    nm_layers = {n: m for n, m in model.named_modules() if isinstance(m, nn.Linear)}
    print(f"  Found: {len(nm_layers)} Linear layers")
    
    # Method 2: recursive
    print("\n[Method 2] Recursive traversal:")
    rec_layers = recursive_find_linear(model)
    print(f"  Found: {len(rec_layers)} Linear layers")
    
    # Compare
    if len(nm_layers) != len(rec_layers):
        print(f"\n⚠️  MISMATCH! This might indicate a problem.")
    else:
        print(f"\n✓ Both methods found the same number of layers")
    
    # Classify layers
    print("\n[Classification]")
    type_counts = defaultdict(int)
    type_params = defaultdict(int)
    
    layers = rec_layers if rec_layers else nm_layers
    
    for name, module in layers.items():
        ltype = classify_layer(name)
        type_counts[ltype] += 1
        type_params[ltype] += module.weight.numel()
    
    total_params = sum(type_params.values())
    
    print(f"\nTotal linear layers: {len(layers)}")
    print(f"Total parameters: {total_params:,}")
    
    print("\nBy type:")
    for ltype in ['attention', 'mlp', 'embedding', 'head', 'other']:
        count = type_counts[ltype]
        params = type_params[ltype]
        pct = 100 * params / total_params if total_params else 0
        print(f"  {ltype:12s}: {count:3d} layers, {params:12,} params ({pct:5.1f}%)")
    
    # Prunable count
    prunable = type_counts['attention'] + type_counts['mlp']
    print(f"\nPrunable layers (attention + mlp): {prunable}")
    
    if prunable == 0:
        print("\n❌ WARNING: No prunable layers detected!")
        print("   This means layer naming patterns don't match.")
        print("   Check the layer names below and update classify_layer()")
    else:
        print(f"\n✓ Ready for pruning!")
    
    # Show sample layers
    print("\n[Sample Layers]")
    for i, (name, module) in enumerate(list(layers.items())[:15]):
        ltype = classify_layer(name)
        shape = f"({module.out_features}, {module.in_features})"
        print(f"  [{ltype:5s}] {name}: {shape}")
    
    if len(layers) > 15:
        print(f"  ... and {len(layers) - 15} more layers")
    
    print("\n" + "=" * 60)
    print("DIAGNOSIS COMPLETE")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
