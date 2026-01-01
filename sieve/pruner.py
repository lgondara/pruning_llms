"""
TransformerPruner: High-Level Interface for LLM-Sieve

Provides an easy-to-use interface for pruning HuggingFace transformer models.
Supports both uniform and adaptive (genetic algorithm) pruning strategies.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Callable, Optional, Tuple, Any
from dataclasses import dataclass, field
from tqdm import tqdm
import copy

from .core import MatrixPruner, PrunedLinear, PruningConfig
from .layer_discovery import LayerDiscovery, LayerInfo


@dataclass
class LLMSieveConfig:
    """Configuration for LLM-Sieve pruning."""
    # Calibration settings
    calibration_tokens: int = 200_000
    batch_size: int = 5000
    learning_rate: float = 0.001
    num_epochs: int = 2
    
    # Pruning strategy
    pruning_strategy: str = "uniform"  # "uniform" or "adaptive"
    target_accuracy_drop: float = 0.05
    
    # What to prune
    prune_attention: bool = True
    prune_mlp: bool = True
    
    # Genetic algorithm settings (for adaptive)
    ga_population_size: int = 100
    ga_generations: int = 30
    ga_crossover_prob: float = 0.5
    ga_mutation_prob: float = 0.2
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Pruning factors to try (for uniform)
    pruning_factors: List[float] = field(
        default_factory=lambda: [0.3, 0.4, 0.5, 0.6, 0.7]
    )


class ActivationCollector:
    """Collects activations from linear layers during forward pass."""
    
    def __init__(self):
        self.activations: Dict[str, List[torch.Tensor]] = {}
        self.hooks = []
    
    def register_hooks(self, layers: Dict[str, LayerInfo]):
        """Register forward hooks on all target layers."""
        for name, info in layers.items():
            self.activations[name] = []
            hook = info.module.register_forward_hook(
                self._make_hook(name)
            )
            self.hooks.append(hook)
    
    def _make_hook(self, name: str):
        def hook(module, input, output):
            # input is a tuple, get the first element
            x = input[0].detach()
            # Flatten to [tokens, dim]
            if x.dim() == 3:
                x = x.reshape(-1, x.size(-1))
            self.activations[name].append(x.cpu())
        return hook
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_activations(self, name: str) -> torch.Tensor:
        """Get concatenated activations for a layer."""
        if name not in self.activations or not self.activations[name]:
            return None
        return torch.cat(self.activations[name], dim=0)
    
    def clear(self):
        """Clear collected activations."""
        for name in self.activations:
            self.activations[name] = []


class TransformerPruner:
    """
    Main interface for pruning transformer models with LLM-Sieve.
    
    Usage:
        pruner = TransformerPruner(model, config)
        pruning_factors = pruner.uniform_prune(calibration_data, eval_fn, baseline)
        pruner.apply_pruning()
        pruner.save_pruned_model("./pruned")
    """
    
    def __init__(self, model: nn.Module, config: Optional[LLMSieveConfig] = None):
        self.model = model
        self.config = config or LLMSieveConfig()
        self.device = self.config.device
        
        # Discover layers
        self.discovery = LayerDiscovery(model, verbose=True)
        self.prunable_layers = self.discovery.get_prunable_layers(
            include_attention=self.config.prune_attention,
            include_mlp=self.config.prune_mlp
        )
        
        # Pruner for each layer
        self.pruners: Dict[str, MatrixPruner] = {}
        self.pruning_factors: Dict[str, float] = {}
        
        # Track state
        self.calibrated = False
        self.pruning_applied = False
    
    def collect_activations(
        self,
        dataloader,
        max_tokens: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Collect activations from calibration data.
        
        Args:
            dataloader: Iterable yielding input tensors
            max_tokens: Maximum tokens to collect (default: config.calibration_tokens)
        
        Returns:
            Dict mapping layer names to activation tensors
        """
        max_tokens = max_tokens or self.config.calibration_tokens
        collector = ActivationCollector()
        collector.register_hooks(self.prunable_layers)
        
        self.model.eval()
        total_tokens = 0
        
        pbar = tqdm(desc="Collecting activations", unit="tok")
        
        try:
            with torch.no_grad():
                for batch in dataloader:
                    # Handle different input formats
                    if isinstance(batch, dict):
                        inputs = {k: v.to(self.device) for k, v in batch.items() 
                                 if isinstance(v, torch.Tensor)}
                        self.model(**inputs)
                        n_tokens = batch.get('input_ids', batch.get('inputs')).numel()
                    elif isinstance(batch, torch.Tensor):
                        self.model(batch.to(self.device))
                        n_tokens = batch.numel()
                    else:
                        # Assume tuple (input_ids, attention_mask, ...)
                        input_ids = batch[0].to(self.device)
                        self.model(input_ids)
                        n_tokens = input_ids.numel()
                    
                    total_tokens += n_tokens
                    pbar.update(n_tokens)
                    
                    if total_tokens >= max_tokens:
                        break
        finally:
            collector.remove_hooks()
            pbar.close()
        
        print(f"Collected {total_tokens:,} tokens of activations")
        
        # Build activation dict
        activations = {}
        for name in self.prunable_layers:
            acts = collector.get_activations(name)
            if acts is not None:
                activations[name] = acts
        
        return activations
    
    def calibrate_pruners(
        self,
        activations: Dict[str, torch.Tensor],
        pruning_factors: Dict[str, float]
    ):
        """
        Calibrate pruners for each layer with given pruning factors.
        
        Args:
            activations: Layer activations from collect_activations()
            pruning_factors: Dict mapping layer name to pruning factor
        """
        self.pruning_factors = pruning_factors
        
        pruning_config = PruningConfig(
            learning_rate=self.config.learning_rate,
            num_epochs=self.config.num_epochs,
            batch_size=self.config.batch_size,
            device=self.device
        )
        
        print("Calibrating pruners for each layer...")
        
        for name, info in tqdm(self.prunable_layers.items(), desc="Calibrating"):
            if name not in activations:
                print(f"  Warning: No activations for {name}, skipping")
                continue
            
            factor = pruning_factors.get(name, 0.5)  # Default 50% compression
            
            pruner = MatrixPruner(
                weight=info.module.weight.data,
                pruning_factor=factor,
                config=pruning_config
            )
            
            pruner.calibrate(activations[name])
            self.pruners[name] = pruner
        
        self.calibrated = True
        print(f"Calibrated {len(self.pruners)} pruners")
    
    def uniform_prune(
        self,
        dataloader,
        eval_fn: Callable[[nn.Module], float],
        baseline_accuracy: float,
        max_tokens: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Find optimal uniform pruning factor via binary search.
        
        Args:
            dataloader: Calibration data
            eval_fn: Function that evaluates model and returns accuracy
            baseline_accuracy: Original model accuracy
            max_tokens: Max tokens for calibration
        
        Returns:
            Dict of pruning factors (same factor for all layers)
        """
        threshold = baseline_accuracy * (1 - self.config.target_accuracy_drop)
        
        print(f"\n{'=' * 50}")
        print(f"UNIFORM PRUNING")
        print(f"Baseline: {baseline_accuracy:.4f}")
        print(f"Threshold: {threshold:.4f} ({self.config.target_accuracy_drop:.0%} drop)")
        print(f"{'=' * 50}\n")
        
        # Collect activations once
        activations = self.collect_activations(dataloader, max_tokens)
        
        # Try each pruning factor
        best_factor = 1.0  # No pruning
        
        for factor in sorted(self.config.pruning_factors):
            print(f"\nTrying pruning factor: {factor}")
            
            # Create uniform factors
            uniform_factors = {name: factor for name in self.prunable_layers}
            
            # Calibrate with this factor
            self.calibrate_pruners(activations, uniform_factors)
            
            # Apply pruning to a copy
            test_model = copy.deepcopy(self.model)
            self._apply_to_model(test_model)
            
            # Evaluate
            accuracy = eval_fn(test_model)
            compression = 1 - factor
            
            print(f"  Accuracy: {accuracy:.4f}, Compression: {compression:.1%}")
            
            if accuracy >= threshold:
                best_factor = factor
                print(f"  ✓ Meets threshold!")
            else:
                print(f"  ✗ Below threshold, stopping search")
                break
            
            # Clean up
            del test_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Final calibration with best factor
        print(f"\nBest pruning factor: {best_factor} ({1-best_factor:.1%} compression)")
        final_factors = {name: best_factor for name in self.prunable_layers}
        self.calibrate_pruners(activations, final_factors)
        
        return final_factors
    
    def adaptive_prune(
        self,
        dataloader,
        eval_fn: Callable[[nn.Module], float],
        baseline_accuracy: float,
        max_tokens: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Find optimal per-layer pruning factors using genetic algorithm.
        
        Args:
            dataloader: Calibration data
            eval_fn: Function that evaluates model and returns accuracy
            baseline_accuracy: Original model accuracy
            max_tokens: Max tokens for calibration
        
        Returns:
            Dict mapping layer names to optimal pruning factors
        """
        from .genetic import GeneticOptimizer
        
        threshold = baseline_accuracy * (1 - self.config.target_accuracy_drop)
        
        print(f"\n{'=' * 50}")
        print(f"ADAPTIVE PRUNING (Genetic Algorithm)")
        print(f"Baseline: {baseline_accuracy:.4f}")
        print(f"Threshold: {threshold:.4f}")
        print(f"{'=' * 50}\n")
        
        # Collect activations
        activations = self.collect_activations(dataloader, max_tokens)
        layer_names = list(self.prunable_layers.keys())
        
        def fitness_fn(factors: List[float]) -> float:
            """Evaluate fitness of a pruning configuration."""
            factor_dict = dict(zip(layer_names, factors))
            
            # Calibrate
            self.calibrate_pruners(activations, factor_dict)
            
            # Apply to copy
            test_model = copy.deepcopy(self.model)
            self._apply_to_model(test_model)
            
            # Evaluate
            accuracy = eval_fn(test_model)
            
            # Cleanup
            del test_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Compute compression
            avg_compression = 1 - sum(factors) / len(factors)
            
            # Fitness: reward compression when accuracy is above threshold
            if accuracy >= threshold:
                fitness = avg_compression
            else:
                # Exponential penalty below threshold
                penalty = 50 * (threshold - accuracy)
                fitness = avg_compression * (1 / (1 + penalty))
            
            return fitness
        
        # Run genetic algorithm
        optimizer = GeneticOptimizer(
            n_genes=len(layer_names),
            population_size=self.config.ga_population_size,
            generations=self.config.ga_generations,
            crossover_prob=self.config.ga_crossover_prob,
            mutation_prob=self.config.ga_mutation_prob
        )
        
        best_factors = optimizer.optimize(fitness_fn)
        final_factors = dict(zip(layer_names, best_factors))
        
        # Final calibration
        self.calibrate_pruners(activations, final_factors)
        
        return final_factors
    
    def apply_pruning(self):
        """Apply calibrated pruning to the model."""
        if not self.calibrated:
            raise RuntimeError("Must calibrate pruners before applying")
        
        self._apply_to_model(self.model)
        self.pruning_applied = True
        print("Pruning applied to model")
    
    def _apply_to_model(self, model: nn.Module):
        """Replace linear layers with pruned versions in given model."""
        for name, pruner in self.pruners.items():
            # Get the original module
            info = self.prunable_layers[name]
            original = info.module
            
            # Get pruned weights
            W_pruned, A_t = pruner.get_pruned_weights()
            
            # Create pruned module
            pruned = PrunedLinear(W_pruned, A_t, original.bias)
            
            # Replace in model
            self._replace_module(model, name, pruned)
    
    def _replace_module(self, model: nn.Module, name: str, new_module: nn.Module):
        """Replace a module by name in the model hierarchy."""
        parts = name.split('.')
        parent = model
        
        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)
        
        final_name = parts[-1]
        if final_name.isdigit():
            parent[int(final_name)] = new_module
        else:
            setattr(parent, final_name, new_module)
    
    def save_pruned_model(self, path: str):
        """Save the pruned model."""
        if not self.pruning_applied:
            print("Warning: Pruning not yet applied, saving original model")
        
        torch.save(self.model.state_dict(), f"{path}/model.pt")
        
        # Save pruning info
        import json
        info = {
            "pruning_factors": self.pruning_factors,
            "config": {
                "pruning_strategy": self.config.pruning_strategy,
                "target_accuracy_drop": self.config.target_accuracy_drop
            },
            "compression_stats": self.get_compression_stats()
        }
        with open(f"{path}/pruning_info.json", "w") as f:
            json.dump(info, f, indent=2)
        
        print(f"Saved pruned model to {path}")
    
    def get_compression_stats(self) -> Dict:
        """Get overall compression statistics."""
        total_original = 0
        total_pruned = 0
        
        for name, pruner in self.pruners.items():
            stats = pruner.get_compression_stats()
            total_original += stats["original_params"]
            total_pruned += stats["pruned_params"]
        
        return {
            "total_original_params": total_original,
            "total_pruned_params": total_pruned,
            "overall_compression": 1 - total_pruned / total_original if total_original > 0 else 0,
            "num_layers_pruned": len(self.pruners)
        }
    
    def print_statistics(self):
        """Print detailed pruning statistics."""
        print("\n" + "=" * 60)
        print("PRUNING STATISTICS")
        print("=" * 60)
        
        print(f"\nLayers discovered: {len(self.discovery.layers)}")
        print(f"Layers to prune: {len(self.prunable_layers)}")
        
        if self.pruners:
            stats = self.get_compression_stats()
            print(f"\nOriginal params: {stats['total_original_params']:,}")
            print(f"Pruned params: {stats['total_pruned_params']:,}")
            print(f"Compression: {stats['overall_compression']:.1%}")
        
        print("=" * 60 + "\n")
