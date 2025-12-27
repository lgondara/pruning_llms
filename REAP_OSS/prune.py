"""
Expert pruning module implementing REAP for GPT-OSS models.

REAP (Router-weighted Expert Activation Pruning) selects experts to prune
based on their contribution to layer outputs, considering both:
1. Router gate values (how often and strongly the router activates experts)
2. Expert activation norms (the magnitude of expert outputs)

This implementation is adapted for GPT-OSS 20B and 120B models.
"""

from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import torch
import torch.nn as nn
from tqdm import tqdm
import copy

from .model_util import (
    get_model_attrs, 
    get_moe_layers, 
    get_num_experts,
    get_experts_module,
    get_router,
)
from .observer import MoEObserver, ExpertOutputObserver


@dataclass
class PruningConfig:
    """Configuration for expert pruning."""
    compression_ratio: float = 0.5  # Fraction of experts to remove
    calibration_samples: int = 512  # Number of tokens for saliency estimation
    method: str = "reap"  # "reap", "frequency", "random"
    uniform_across_layers: bool = True  # Same pruning ratio per layer
    preserve_min_experts: int = 2  # Minimum experts to keep per layer
    seed: int = 42


@dataclass
class PruningResult:
    """Result of expert pruning."""
    experts_to_prune: Dict[int, List[int]]  # layer_idx -> list of expert indices
    saliency_scores: Dict[int, Dict[int, float]]  # layer_idx -> expert_idx -> score
    original_num_experts: int
    pruned_num_experts: int
    compression_achieved: float


class REAPPruner:
    """
    REAP expert pruning for MoE models.
    
    Usage:
        pruner = REAPPruner(model, config)
        result = pruner.compute_saliency(dataloader)
        pruned_model = pruner.prune_experts(result)
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        config: PruningConfig,
    ):
        self.model = model
        self.config = config
        self.attrs = get_model_attrs(model)
        self.moe_layers = get_moe_layers(model)
        self.num_experts = get_num_experts(model)
        
        # Set random seed
        torch.manual_seed(config.seed)
        
    def compute_saliency(
        self,
        dataloader: torch.utils.data.DataLoader,
        use_full_reap: bool = True,
    ) -> Dict[int, Dict[int, float]]:
        """
        Compute REAP saliency scores by running calibration data through model.
        
        Args:
            dataloader: DataLoader providing calibration samples
            use_full_reap: If True, capture expert output norms (more accurate
                          but higher memory usage)
                          
        Returns:
            Dict mapping layer_idx -> expert_idx -> saliency_score
        """
        # Create observer
        if use_full_reap:
            observer = ExpertOutputObserver(self.model, self.attrs)
        else:
            observer = MoEObserver(self.model, self.attrs)
            
        observer.register_hooks()
        
        # Run calibration
        self.model.eval()
        token_count = 0
        
        with torch.no_grad(), observer.observe():
            for batch in tqdm(dataloader, desc="Computing saliency"):
                if token_count >= self.config.calibration_samples:
                    break
                    
                input_ids = batch["input_ids"].to(self.model.device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.model.device)
                    
                _ = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                
                token_count += input_ids.numel()
                
        observer.remove_hooks()
        
        # Get scores
        if use_full_reap:
            scores = observer.get_full_reap_scores()
        else:
            scores = observer.get_reap_scores()
            
        return scores
    
    def select_experts_to_prune(
        self,
        saliency_scores: Dict[int, Dict[int, float]],
    ) -> PruningResult:
        """
        Select which experts to prune based on saliency scores.
        
        Args:
            saliency_scores: Per-expert saliency from compute_saliency()
            
        Returns:
            PruningResult with experts to prune and statistics
        """
        experts_to_prune: Dict[int, List[int]] = {}
        
        if self.config.uniform_across_layers:
            # Same number of experts pruned per layer
            num_to_prune = int(
                self.num_experts * self.config.compression_ratio
            )
            num_to_prune = min(
                num_to_prune, 
                self.num_experts - self.config.preserve_min_experts
            )
            
            for layer_idx, layer_scores in saliency_scores.items():
                # Sort by saliency (lowest first = prune first)
                sorted_experts = sorted(
                    layer_scores.items(), 
                    key=lambda x: x[1]
                )
                experts_to_prune[layer_idx] = [
                    expert_idx for expert_idx, _ in sorted_experts[:num_to_prune]
                ]
        else:
            # Global pruning across all layers
            all_scores = []
            for layer_idx, layer_scores in saliency_scores.items():
                for expert_idx, score in layer_scores.items():
                    all_scores.append((layer_idx, expert_idx, score))
                    
            all_scores.sort(key=lambda x: x[2])
            
            total_experts = len(all_scores)
            num_to_prune = int(total_experts * self.config.compression_ratio)
            
            # Track per-layer counts to respect minimum
            layer_prune_counts: Dict[int, int] = {
                layer_idx: 0 for layer_idx, _ in self.moe_layers
            }
            
            for layer_idx, expert_idx, _ in all_scores[:num_to_prune]:
                if layer_prune_counts[layer_idx] >= (
                    self.num_experts - self.config.preserve_min_experts
                ):
                    continue
                    
                if layer_idx not in experts_to_prune:
                    experts_to_prune[layer_idx] = []
                experts_to_prune[layer_idx].append(expert_idx)
                layer_prune_counts[layer_idx] += 1
                
        # Compute statistics
        total_pruned = sum(len(v) for v in experts_to_prune.values())
        total_original = len(self.moe_layers) * self.num_experts
        
        return PruningResult(
            experts_to_prune=experts_to_prune,
            saliency_scores=saliency_scores,
            original_num_experts=self.num_experts,
            pruned_num_experts=self.num_experts - (
                total_pruned // len(self.moe_layers) if self.moe_layers else 0
            ),
            compression_achieved=total_pruned / total_original if total_original > 0 else 0,
        )
    
    def prune_experts(
        self,
        result: PruningResult,
        inplace: bool = False,
    ) -> nn.Module:
        """
        Apply expert pruning to the model.
        
        This modifies the model by:
        1. Removing pruned experts from each MoE layer's expert list
        2. Updating the router to handle reduced expert count
        3. Adjusting expert indices in the routing logic
        
        Args:
            result: PruningResult from select_experts_to_prune()
            inplace: If True, modify model in place; else return copy
            
        Returns:
            Pruned model
        """
        if not inplace:
            model = copy.deepcopy(self.model)
        else:
            model = self.model
            
        attrs = get_model_attrs(model)
        
        for layer_idx, moe_module in get_moe_layers(model):
            if layer_idx not in result.experts_to_prune:
                continue
                
            experts_to_remove = set(result.experts_to_prune[layer_idx])
            if not experts_to_remove:
                continue
                
            # Get current experts
            experts = getattr(moe_module, attrs.experts)
            
            # Create new ModuleList with remaining experts
            new_experts = nn.ModuleList([
                expert for idx, expert in enumerate(experts)
                if idx not in experts_to_remove
            ])
            
            # Replace experts
            setattr(moe_module, attrs.experts, new_experts)
            
            # Update router output dimension
            router = getattr(moe_module, attrs.router)
            old_out_features = router.out_features
            new_out_features = old_out_features - len(experts_to_remove)
            
            # Create new router with correct dimensions
            new_router = self._create_pruned_router(
                router, 
                experts_to_remove,
                new_out_features,
            )
            setattr(moe_module, attrs.router, new_router)
            
            # Store mapping for index translation
            self._store_expert_mapping(
                moe_module, 
                experts_to_remove,
                old_out_features,
            )
            
        # Update config
        model.config.num_experts = result.pruned_num_experts
        
        return model
    
    def _create_pruned_router(
        self,
        router: nn.Linear,
        removed_experts: Set[int],
        new_out_features: int,
    ) -> nn.Linear:
        """Create a new router with pruned expert outputs removed."""
        # Get indices to keep
        keep_indices = [
            i for i in range(router.out_features) 
            if i not in removed_experts
        ]
        
        # Create new linear layer
        new_router = nn.Linear(
            router.in_features,
            new_out_features,
            bias=router.bias is not None,
            device=router.weight.device,
            dtype=router.weight.dtype,
        )
        
        # Copy weights for kept experts
        with torch.no_grad():
            new_router.weight.copy_(router.weight[keep_indices])
            if router.bias is not None:
                new_router.bias.copy_(router.bias[keep_indices])
                
        return new_router
    
    def _store_expert_mapping(
        self,
        moe_module: nn.Module,
        removed_experts: Set[int],
        original_num_experts: int,
    ) -> None:
        """Store mapping from old to new expert indices."""
        old_to_new = {}
        new_idx = 0
        for old_idx in range(original_num_experts):
            if old_idx not in removed_experts:
                old_to_new[old_idx] = new_idx
                new_idx += 1
                
        moe_module._expert_index_mapping = old_to_new


def compute_frequency_scores(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_samples: int = 512,
) -> Dict[int, Dict[int, float]]:
    """
    Compute frequency-based saliency (baseline comparison for REAP).
    
    Simply counts how often each expert is selected by the router.
    """
    attrs = get_model_attrs(model)
    observer = MoEObserver(model, attrs)
    observer.register_hooks()
    
    model.eval()
    token_count = 0
    
    with torch.no_grad(), observer.observe():
        for batch in tqdm(dataloader, desc="Computing frequencies"):
            if token_count >= num_samples:
                break
            input_ids = batch["input_ids"].to(model.device)
            _ = model(input_ids=input_ids)
            token_count += input_ids.numel()
            
    observer.remove_hooks()
    
    # Convert counts to scores (higher = more important)
    frequencies = observer.get_expert_frequencies()
    return {
        layer_idx: {
            expert_idx: float(count)
            for expert_idx, count in layer_counts.items()
        }
        for layer_idx, layer_counts in frequencies.items()
    }


def compute_random_scores(
    model: nn.Module,
    seed: int = 42,
) -> Dict[int, Dict[int, float]]:
    """
    Assign random saliency scores (baseline comparison for REAP).
    """
    torch.manual_seed(seed)
    attrs = get_model_attrs(model)
    moe_layers = get_moe_layers(model)
    num_experts = get_num_experts(model)
    
    return {
        layer_idx: {
            expert_idx: torch.rand(1).item()
            for expert_idx in range(num_experts)
        }
        for layer_idx, _ in moe_layers
    }
