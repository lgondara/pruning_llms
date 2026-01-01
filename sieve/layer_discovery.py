"""
Layer Discovery for Transformer Models

Handles the proper discovery of linear layers in HuggingFace transformers.
This module addresses the key issue of nested layer detection that was 
causing only lm_head to be found in GPT-2.

Supports multiple architectures: GPT-2, LLaMA, Mistral, Phi, etc.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class LayerInfo:
    """Information about a discovered linear layer."""
    name: str
    module: nn.Linear
    parent_name: str
    layer_type: str  # 'attention', 'mlp', 'embedding', 'head', 'other'
    layer_index: Optional[int]  # Which transformer block (0, 1, 2, ...)
    
    @property
    def shape(self) -> Tuple[int, int]:
        return (self.module.out_features, self.module.in_features)
    
    @property
    def num_params(self) -> int:
        return self.module.weight.numel()


class LayerDiscovery:
    """
    Discovers prunable linear layers in transformer models.
    
    Key insight: We need to recursively traverse the model tree,
    not just use named_modules() which may not properly iterate
    nested ModuleLists in some configurations.
    """
    
    # Patterns for identifying layer types
    ATTENTION_PATTERNS = {
        'q_proj', 'k_proj', 'v_proj', 'o_proj',  # LLaMA style
        'c_attn', 'c_proj',                       # GPT-2 style
        'query', 'key', 'value', 'dense',         # BERT style
        'qkv_proj', 'out_proj',                   # Some unified patterns
        'Wqkv', 'out_proj',                       # Phi style
    }
    
    MLP_PATTERNS = {
        'c_fc', 'c_proj',           # GPT-2 MLP
        'fc1', 'fc2',               # Standard naming
        'gate_proj', 'up_proj', 'down_proj',  # LLaMA MLP
        'fc_in', 'fc_out',          # Alternative
        'dense_h_to_4h', 'dense_4h_to_h',  # Some models
        'w1', 'w2', 'w3',           # Simplified naming
    }
    
    EMBEDDING_PATTERNS = {
        'wte', 'wpe',               # GPT-2 embeddings
        'embed_tokens', 'embed_positions',
        'word_embeddings', 'position_embeddings',
    }
    
    HEAD_PATTERNS = {
        'lm_head', 'cls', 'classifier', 'score',
    }
    
    SKIP_PATTERNS = {
        'layernorm', 'layer_norm', 'ln_', 'rmsnorm',
        'norm', 'embed', 'wte', 'wpe',
    }
    
    def __init__(self, model: nn.Module, verbose: bool = False):
        self.model = model
        self.verbose = verbose
        self.layers: Dict[str, LayerInfo] = {}
        self._discover_layers()
    
    def _discover_layers(self):
        """Discover all linear layers through recursive traversal."""
        self._recursive_find(self.model, "")
        
        if self.verbose:
            self._print_discovery_summary()
    
    def _recursive_find(self, module: nn.Module, prefix: str):
        """
        Recursively traverse model to find all Linear layers.
        
        This is more reliable than named_modules() for nested structures.
        """
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            if isinstance(child, nn.Linear):
                layer_info = self._classify_layer(full_name, child)
                self.layers[full_name] = layer_info
            else:
                # Recursively search children
                self._recursive_find(child, full_name)
    
    def _classify_layer(self, name: str, module: nn.Linear) -> LayerInfo:
        """Classify a linear layer by its role in the model."""
        name_lower = name.lower()
        parts = name.split('.')
        
        # Determine parent name
        parent_name = '.'.join(parts[:-1]) if len(parts) > 1 else ""
        
        # Extract layer index if present
        layer_index = None
        for part in parts:
            if part.isdigit():
                layer_index = int(part)
                break
        
        # Classify by layer type
        layer_type = self._get_layer_type(name_lower, parent_name.lower())
        
        return LayerInfo(
            name=name,
            module=module,
            parent_name=parent_name,
            layer_type=layer_type,
            layer_index=layer_index
        )
    
    def _get_layer_type(self, name: str, parent: str) -> str:
        """Determine the type of layer based on naming patterns."""
        # Check if in attention block
        if '.attn.' in name or '.attention.' in name or '.self_attn.' in name:
            return 'attention'
        
        # Check if in MLP block
        if '.mlp.' in name or '.ffn.' in name or '.feed_forward.' in name:
            return 'mlp'
        
        # Check layer name patterns
        layer_name = name.split('.')[-1].lower()
        
        # GPT-2 specific: c_proj can be in either attn or mlp
        if layer_name == 'c_proj':
            if 'attn' in parent:
                return 'attention'
            elif 'mlp' in parent:
                return 'mlp'
        
        if layer_name == 'c_attn':
            return 'attention'
        
        if layer_name == 'c_fc':
            return 'mlp'
        
        # General pattern matching
        if any(p in name for p in self.ATTENTION_PATTERNS):
            return 'attention'
        
        if any(p in name for p in self.MLP_PATTERNS):
            return 'mlp'
        
        if any(p in name for p in self.EMBEDDING_PATTERNS):
            return 'embedding'
        
        if any(p in name for p in self.HEAD_PATTERNS):
            return 'head'
        
        return 'other'
    
    def get_prunable_layers(
        self,
        include_attention: bool = True,
        include_mlp: bool = True,
        include_other: bool = False,
        exclude_names: Optional[Set[str]] = None
    ) -> Dict[str, LayerInfo]:
        """
        Get layers that should be pruned.
        
        Args:
            include_attention: Include attention projection layers
            include_mlp: Include MLP/FFN layers
            include_other: Include unclassified linear layers
            exclude_names: Set of layer names to exclude
        
        Returns:
            Dictionary of layer_name -> LayerInfo
        """
        exclude_names = exclude_names or set()
        prunable = {}
        
        for name, info in self.layers.items():
            # Skip excluded names
            if name in exclude_names:
                continue
            
            # Skip by pattern
            name_lower = name.lower()
            if any(skip in name_lower for skip in self.SKIP_PATTERNS):
                continue
            
            # Filter by type
            if info.layer_type == 'attention' and include_attention:
                prunable[name] = info
            elif info.layer_type == 'mlp' and include_mlp:
                prunable[name] = info
            elif info.layer_type == 'other' and include_other:
                prunable[name] = info
        
        return prunable
    
    def get_attention_layers(self) -> Dict[str, LayerInfo]:
        """Get only attention layers."""
        return {n: l for n, l in self.layers.items() if l.layer_type == 'attention'}
    
    def get_mlp_layers(self) -> Dict[str, LayerInfo]:
        """Get only MLP layers."""
        return {n: l for n, l in self.layers.items() if l.layer_type == 'mlp'}
    
    def get_layers_by_block(self) -> Dict[int, List[LayerInfo]]:
        """Group layers by transformer block index."""
        by_block = defaultdict(list)
        for info in self.layers.values():
            if info.layer_index is not None:
                by_block[info.layer_index].append(info)
        return dict(by_block)
    
    def _print_discovery_summary(self):
        """Print summary of discovered layers."""
        print("\n" + "=" * 60)
        print("LAYER DISCOVERY SUMMARY")
        print("=" * 60)
        
        type_counts = defaultdict(int)
        type_params = defaultdict(int)
        
        for info in self.layers.values():
            type_counts[info.layer_type] += 1
            type_params[info.layer_type] += info.num_params
        
        total_params = sum(type_params.values())
        
        print(f"\nTotal linear layers found: {len(self.layers)}")
        print(f"Total parameters: {total_params:,}")
        print("\nBreakdown by type:")
        
        for ltype in ['attention', 'mlp', 'embedding', 'head', 'other']:
            count = type_counts[ltype]
            params = type_params[ltype]
            pct = 100 * params / total_params if total_params > 0 else 0
            print(f"  {ltype:12s}: {count:3d} layers, {params:12,} params ({pct:5.1f}%)")
        
        # Show first few layers as examples
        print("\nFirst 10 layers:")
        for i, (name, info) in enumerate(list(self.layers.items())[:10]):
            print(f"  [{info.layer_type:5s}] {name}: {info.shape}")
        
        if len(self.layers) > 10:
            print(f"  ... and {len(self.layers) - 10} more")
        
        print("=" * 60 + "\n")


def diagnose_model(model: nn.Module, model_name: str = "model") -> Dict:
    """
    Run diagnostics on a model to verify layer discovery works.
    
    Returns a diagnostic report dict.
    """
    print(f"\n{'=' * 60}")
    print(f"DIAGNOSING MODEL: {model_name}")
    print('=' * 60)
    
    # Method 1: named_modules
    print("\n[Method 1] Using named_modules():")
    nm_linear = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    print(f"  Found {len(nm_linear)} Linear layers")
    
    # Method 2: named_children recursive
    print("\n[Method 2] Using recursive named_children():")
    discovery = LayerDiscovery(model, verbose=False)
    print(f"  Found {len(discovery.layers)} Linear layers")
    
    # Compare
    if len(nm_linear) != len(discovery.layers):
        print(f"\n⚠️  MISMATCH: named_modules found {len(nm_linear)}, "
              f"recursive found {len(discovery.layers)}")
    
    # Show prunable layers
    prunable = discovery.get_prunable_layers()
    print(f"\n[Result] Prunable layers: {len(prunable)}")
    print(f"  Attention: {len(discovery.get_attention_layers())}")
    print(f"  MLP: {len(discovery.get_mlp_layers())}")
    
    # Show structure
    print("\n[Structure] Layer hierarchy:")
    by_block = discovery.get_layers_by_block()
    for block_idx in sorted(by_block.keys())[:3]:
        print(f"  Block {block_idx}:")
        for info in by_block[block_idx]:
            print(f"    {info.layer_type:5s}: {info.name.split('.')[-1]} {info.shape}")
    if len(by_block) > 3:
        print(f"  ... {len(by_block) - 3} more blocks")
    
    print('=' * 60 + "\n")
    
    return {
        "model_name": model_name,
        "total_linear": len(discovery.layers),
        "prunable": len(prunable),
        "attention": len(discovery.get_attention_layers()),
        "mlp": len(discovery.get_mlp_layers()),
        "discovery": discovery
    }
