"""
Observer module for collecting expert activations and router logits.

This module provides hooks to observe the internal behavior of MoE layers,
specifically designed for REAP saliency computation on GPT-OSS models.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from contextlib import contextmanager


@dataclass
class ExpertObservation:
    """Container for observations from a single MoE layer forward pass."""
    layer_idx: int
    router_logits: torch.Tensor  # (batch, seq_len, num_experts)
    router_weights: torch.Tensor  # (batch, seq_len, top_k) normalized weights
    selected_experts: torch.Tensor  # (batch, seq_len, top_k) expert indices
    expert_outputs: Optional[Dict[int, torch.Tensor]] = None  # expert_idx -> outputs


@dataclass  
class ActivationStats:
    """Aggregated statistics for expert activations."""
    layer_idx: int
    expert_idx: int
    activation_count: int = 0
    gate_value_sum: float = 0.0
    activation_norm_sum: float = 0.0
    
    @property
    def avg_gate_value(self) -> float:
        if self.activation_count == 0:
            return 0.0
        return self.gate_value_sum / self.activation_count
    
    @property
    def avg_activation_norm(self) -> float:
        if self.activation_count == 0:
            return 0.0
        return self.activation_norm_sum / self.activation_count
    
    @property
    def reap_score(self) -> float:
        """
        Compute REAP saliency score.
        
        REAP = avg(gate_value * activation_norm) for active tokens
        """
        if self.activation_count == 0:
            return 0.0
        # Note: This is a simplified version; full REAP uses
        # sum(gate * norm) / num_active_tokens
        return self.avg_gate_value * self.avg_activation_norm


class MoEObserver:
    """
    Observer for collecting MoE layer activations and router decisions.
    
    Usage:
        observer = MoEObserver(model)
        observer.register_hooks()
        
        with observer.observe():
            outputs = model.generate(...)
            
        stats = observer.get_statistics()
    """
    
    def __init__(self, model: nn.Module, model_attrs: Any):
        self.model = model
        self.attrs = model_attrs
        self.observations: List[ExpertObservation] = []
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._observing = False
        
        # Get MoE layers
        self.moe_layers = self._get_moe_layers()
        
    def _get_moe_layers(self) -> List[Tuple[int, nn.Module]]:
        """Extract MoE layers from the model."""
        moe_layers = []
        
        if hasattr(self.model, "model"):
            base_model = self.model.model
        else:
            base_model = self.model
            
        if hasattr(base_model, "layers"):
            layers = base_model.layers
        elif hasattr(base_model, "decoder"):
            layers = base_model.decoder.layers
        else:
            raise AttributeError("Cannot find decoder layers")
            
        for idx, layer in enumerate(layers):
            if hasattr(layer, self.attrs.moe_block):
                moe_module = getattr(layer, self.attrs.moe_block)
                if hasattr(moe_module, self.attrs.experts):
                    moe_layers.append((idx, moe_module))
                    
        return moe_layers
    
    def _create_router_hook(self, layer_idx: int):
        """Create a hook to capture router logits."""
        def hook(module, inputs, outputs):
            if not self._observing:
                return
                
            # GPT-OSS router output structure may vary
            # Typically: router_logits = router(hidden_states)
            if isinstance(outputs, tuple):
                router_logits = outputs[0]
            else:
                router_logits = outputs
                
            # Get top-k routing
            num_experts_per_tok = getattr(
                self.model.config, 
                self.attrs.num_experts_per_tok
            )
            
            # Compute softmax over selected experts
            topk_logits, topk_indices = torch.topk(
                router_logits, 
                num_experts_per_tok, 
                dim=-1
            )
            topk_weights = torch.softmax(topk_logits, dim=-1)
            
            observation = ExpertObservation(
                layer_idx=layer_idx,
                router_logits=router_logits.detach().cpu(),
                router_weights=topk_weights.detach().cpu(),
                selected_experts=topk_indices.detach().cpu(),
            )
            self.observations.append(observation)
            
        return hook
    
    def register_hooks(self) -> None:
        """Register forward hooks on router modules."""
        self.remove_hooks()
        
        for layer_idx, moe_module in self.moe_layers:
            router = getattr(moe_module, self.attrs.router)
            hook = router.register_forward_hook(
                self._create_router_hook(layer_idx)
            )
            self.hooks.append(hook)
            
    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        
    @contextmanager
    def observe(self):
        """Context manager for observation mode."""
        self.observations.clear()
        self._observing = True
        try:
            yield self
        finally:
            self._observing = False
            
    def get_statistics(
        self, 
        num_experts: Optional[int] = None
    ) -> Dict[int, Dict[int, ActivationStats]]:
        """
        Aggregate observations into per-expert statistics.
        
        Returns:
            Dict mapping layer_idx -> expert_idx -> ActivationStats
        """
        if num_experts is None:
            num_experts = getattr(self.model.config, self.attrs.num_experts)
            
        # Initialize stats containers
        stats: Dict[int, Dict[int, ActivationStats]] = {}
        for layer_idx, _ in self.moe_layers:
            stats[layer_idx] = {
                expert_idx: ActivationStats(layer_idx, expert_idx)
                for expert_idx in range(num_experts)
            }
            
        # Aggregate observations
        for obs in self.observations:
            layer_stats = stats[obs.layer_idx]
            
            # Flatten batch and sequence dimensions
            weights_flat = obs.router_weights.view(-1, obs.router_weights.size(-1))
            experts_flat = obs.selected_experts.view(-1, obs.selected_experts.size(-1))
            
            for token_idx in range(weights_flat.size(0)):
                for k in range(weights_flat.size(1)):
                    expert_idx = experts_flat[token_idx, k].item()
                    gate_value = weights_flat[token_idx, k].item()
                    
                    layer_stats[expert_idx].activation_count += 1
                    layer_stats[expert_idx].gate_value_sum += gate_value
                    
        return stats
    
    def get_expert_frequencies(self) -> Dict[int, Dict[int, int]]:
        """
        Get activation frequency per expert.
        
        Returns:
            Dict mapping layer_idx -> expert_idx -> activation_count
        """
        stats = self.get_statistics()
        return {
            layer_idx: {
                expert_idx: expert_stats.activation_count
                for expert_idx, expert_stats in layer_stats.items()
            }
            for layer_idx, layer_stats in stats.items()
        }
    
    def get_reap_scores(self) -> Dict[int, Dict[int, float]]:
        """
        Compute REAP saliency scores for all experts.
        
        Note: This requires expert output norms which need additional
        instrumentation. This version uses gate values as proxy.
        
        Returns:
            Dict mapping layer_idx -> expert_idx -> reap_score
        """
        stats = self.get_statistics()
        return {
            layer_idx: {
                expert_idx: expert_stats.avg_gate_value
                for expert_idx, expert_stats in layer_stats.items()
            }
            for layer_idx, layer_stats in stats.items()
        }


class ExpertOutputObserver(MoEObserver):
    """
    Extended observer that also captures expert output norms.
    
    This is needed for full REAP score computation:
    REAP(e) = (1/|T_e|) * sum_{t in T_e}(gate_value_t * ||expert_output_t||)
    
    Warning: This significantly increases memory usage as it stores
    expert outputs for each token.
    """
    
    def __init__(self, model: nn.Module, model_attrs: Any):
        super().__init__(model, model_attrs)
        self.expert_output_norms: Dict[int, Dict[int, List[float]]] = {}
        
    def _create_expert_hook(self, layer_idx: int, expert_idx: int):
        """Create a hook to capture expert output norms."""
        def hook(module, inputs, outputs):
            if not self._observing:
                return
                
            # Compute L2 norm of expert output per token
            if isinstance(outputs, tuple):
                output = outputs[0]
            else:
                output = outputs
                
            # output shape: (batch * num_selected, hidden_size) typically
            norms = output.norm(dim=-1).detach().cpu().tolist()
            
            if layer_idx not in self.expert_output_norms:
                self.expert_output_norms[layer_idx] = {}
            if expert_idx not in self.expert_output_norms[layer_idx]:
                self.expert_output_norms[layer_idx][expert_idx] = []
                
            self.expert_output_norms[layer_idx][expert_idx].extend(norms)
            
        return hook
    
    def register_hooks(self) -> None:
        """Register hooks on both routers and individual experts."""
        super().register_hooks()
        
        for layer_idx, moe_module in self.moe_layers:
            experts = getattr(moe_module, self.attrs.experts)
            for expert_idx, expert in enumerate(experts):
                # Hook on the down projection (final output)
                down_proj = getattr(expert, self.attrs.down_proj)
                hook = down_proj.register_forward_hook(
                    self._create_expert_hook(layer_idx, expert_idx)
                )
                self.hooks.append(hook)
                
    @contextmanager
    def observe(self):
        """Context manager for observation mode."""
        self.observations.clear()
        self.expert_output_norms.clear()
        self._observing = True
        try:
            yield self
        finally:
            self._observing = False
            
    def get_full_reap_scores(self) -> Dict[int, Dict[int, float]]:
        """
        Compute full REAP scores using both gate values and output norms.
        
        Returns:
            Dict mapping layer_idx -> expert_idx -> reap_score
        """
        stats = self.get_statistics()
        scores = {}
        
        for layer_idx, layer_stats in stats.items():
            scores[layer_idx] = {}
            for expert_idx, expert_stats in layer_stats.items():
                if expert_stats.activation_count == 0:
                    scores[layer_idx][expert_idx] = 0.0
                    continue
                    
                # Get average output norm
                output_norms = self.expert_output_norms.get(layer_idx, {}).get(expert_idx, [])
                if not output_norms:
                    avg_norm = 1.0  # Default if not captured
                else:
                    avg_norm = sum(output_norms) / len(output_norms)
                    
                # REAP score = avg(gate * norm)
                scores[layer_idx][expert_idx] = (
                    expert_stats.avg_gate_value * avg_norm
                )
                
        return scores
