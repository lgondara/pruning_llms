"""
Diagnostic script to check if layer extraction will work for your model.

Run this BEFORE trying to prune, to see what layers will be found.
"""

def check_model_compatibility(model_name="gpt2"):
    """
    Check if a model's layer names are compatible with LLM-Sieve.
    
    This loads the model and prints all linear layer names, showing which
    ones would be selected for pruning.
    """
    try:
        from transformers import AutoModelForCausalLM
        import torch.nn as nn
    except ImportError:
        print("ERROR: transformers or torch not installed.")
        print("Install with: pip install transformers torch")
        return
    
    print("="*70)
    print(f"Diagnostic: Checking layer names for '{model_name}'")
    print("="*70)
    
    print(f"\n1. Loading model '{model_name}'...")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        print(f"   ✅ Model loaded successfully")
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        return
    
    # Collect all linear layers
    print("\n2. Scanning for linear layers...")
    all_linear = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            all_linear.append((name, module.in_features, module.out_features))
    
    print(f"   Found {len(all_linear)} total linear layers")
    
    if len(all_linear) == 0:
        print("\n   ❌ No linear layers found. This model may not be compatible.")
        return
    
    # Show all layers
    print("\n3. All linear layers in model:")
    print(f"   {'Layer Name':<60s} {'Shape':>15s}")
    print("   " + "-"*77)
    for name, in_f, out_f in all_linear:
        print(f"   {name:<60s} {in_f:>6d} -> {out_f:<6d}")
    
    # Categorize layers
    print("\n4. Categorizing layers...")
    
    def should_skip(name):
        """Check if layer should be skipped (embeddings, norms, etc.)"""
        skip_patterns = ['embed', 'wte', 'wpe', 'position', 'token', 
                        'ln', 'layernorm', 'norm', 'lm_head', 'score']
        return any(p in name.lower() for p in skip_patterns)
    
    def is_attention(name):
        """Check if layer is in attention block"""
        name_lower = name.lower()
        # Use .attn. with dots to match module hierarchy
        return any(block in name_lower for block in ['.attn.', '.attention.', '.self_attn.'])
    
    def is_mlp(name):
        """Check if layer is in MLP/FFN block"""
        name_lower = name.lower()
        # Use .mlp. with dots to match module hierarchy
        return any(block in name_lower for block in ['.mlp.', '.ffn.', '.feed_forward.'])
    
    skipped = []
    attention = []
    mlp = []
    other = []
    
    for name, in_f, out_f in all_linear:
        if should_skip(name):
            skipped.append(name)
        elif is_attention(name):
            attention.append(name)
        elif is_mlp(name):
            mlp.append(name)
        else:
            other.append(name)
    
    print(f"\n   📊 Categorization results:")
    print(f"   - Skipped (embeddings/norms): {len(skipped)}")
    print(f"   - Attention layers: {len(attention)}")
    print(f"   - MLP/FFN layers: {len(mlp)}")
    print(f"   - Other: {len(other)}")
    
    # Show what would be pruned
    print("\n5. Layers that WOULD be pruned (prune_attention=True, prune_ffn=True):")
    prunable = attention + mlp
    if len(prunable) > 0:
        print(f"\n   ✅ SUCCESS! {len(prunable)} layers would be pruned:")
        for name in prunable[:10]:  # Show first 10
            layer_type = "ATTN" if name in attention else "MLP "
            print(f"   [{layer_type}] {name}")
        if len(prunable) > 10:
            print(f"   ... and {len(prunable)-10} more")
    else:
        print("\n   ❌ WARNING! No layers would be pruned.")
        print("\n   This means the layer naming doesn't match expected patterns.")
        print("   Please share the layer names above for help adding support.")
    
    # Show skipped layers
    if len(skipped) > 0:
        print(f"\n6. Layers that would be SKIPPED:")
        for name in skipped[:5]:
            print(f"   [SKIP] {name}")
        if len(skipped) > 5:
            print(f"   ... and {len(skipped)-5} more")
    
    # Show unhandled layers
    if len(other) > 0:
        print(f"\n⚠️  Warning: {len(other)} layers don't match any category:")
        for name in other:
            print(f"   [????] {name}")
        print("\n   These might need custom handling. Please report if important.")
    
    print("\n" + "="*70)
    print("Diagnostic complete!")
    print("="*70)
    
    # Give recommendation
    print("\n📋 RECOMMENDATION:")
    if len(prunable) > 0:
        print(f"   ✅ This model should work with LLM-Sieve!")
        print(f"   {len(prunable)} layers can be pruned.")
        print("\n   Next step: Run the actual pruning with:")
        print(f"   python example_huggingface.py --model {model_name}")
    else:
        print("   ⚠️  This model may not work out-of-the-box.")
        print("   Please share the output above to get help adding support.")


if __name__ == "__main__":
    import sys
    
    # Get model name from command line or use default
    model_name = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
    
    print("\n🔍 LLM-Sieve Model Compatibility Checker")
    print()
    print("Usage: python diagnose_model.py [model_name]")
    print("Example: python diagnose_model.py gpt2")
    print()
    
    check_model_compatibility(model_name)
