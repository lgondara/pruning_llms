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
        
        NOTE: This method is kept for backward compatibility but is memory-intensive.
        For large models, use calibrate_memory_efficient() instead.
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
    
    def _collect_single_layer_activations(
        self,
        layer_name: str,
        dataloader,
        max_tokens: int
    ) -> torch.Tensor:
        """
        Collect activations for a single layer only.
        
        Memory-efficient: only stores activations for one layer at a time.
        """
        layer_info = self.prunable_layers[layer_name]
        activations = []
        
        def hook(module, input, output):
            x = input[0].detach()
            if x.dim() == 3:
                x = x.reshape(-1, x.size(-1))
            # Store as float16 to save memory
            activations.append(x.half().cpu())
        
        handle = layer_info.module.register_forward_hook(hook)
        
        self.model.eval()
        total_tokens = 0
        
        try:
            with torch.no_grad():
                for batch in dataloader:
                    if isinstance(batch, dict):
                        inputs = {k: v.to(self.device) for k, v in batch.items() 
                                 if isinstance(v, torch.Tensor)}
                        self.model(**inputs)
                        n_tokens = batch.get('input_ids', batch.get('inputs')).numel()
                    elif isinstance(batch, torch.Tensor):
                        self.model(batch.to(self.device))
                        n_tokens = batch.numel()
                    else:
                        input_ids = batch[0].to(self.device)
                        self.model(input_ids)
                        n_tokens = input_ids.numel()
                    
                    total_tokens += n_tokens
                    if total_tokens >= max_tokens:
                        break
        finally:
            handle.remove()
        
        # Concatenate and convert back to float32 for calibration
        return torch.cat(activations, dim=0).float()
    
    def calibrate_memory_efficient(
        self,
        dataloader,
        pruning_factors: Dict[str, float],
        max_tokens: Optional[int] = None
    ):
        """
        Memory-efficient calibration: process one layer at a time.
        
        This is the recommended method for large models (7B+).
        Instead of storing all activations, we:
        1. For each layer, collect its activations
        2. Calibrate that layer's pruner
        3. Free the activations before moving to next layer
        
        Args:
            dataloader: Calibration data (will be iterated multiple times)
            pruning_factors: Dict mapping layer name to pruning factor
            max_tokens: Max tokens per layer (default: config.calibration_tokens)
        """
        max_tokens = max_tokens or self.config.calibration_tokens
        self.pruning_factors = pruning_factors
        
        pruning_config = PruningConfig(
            learning_rate=self.config.learning_rate,
            num_epochs=self.config.num_epochs,
            batch_size=self.config.batch_size,
            device=self.device
        )
        
        layer_names = list(self.prunable_layers.keys())
        print(f"\nMemory-efficient calibration: {len(layer_names)} layers")
        print(f"Processing one layer at a time to minimize memory usage\n")
        
        for i, name in enumerate(layer_names):
            info = self.prunable_layers[name]
            factor = pruning_factors.get(name, 0.5)
            
            print(f"[{i+1}/{len(layer_names)}] {name}")
            print(f"  Collecting activations...", end=" ", flush=True)
            
            # Collect activations for this layer only
            activations = self._collect_single_layer_activations(
                name, dataloader, max_tokens
            )
            print(f"{activations.shape[0]:,} tokens")
            
            # Create and calibrate pruner
            print(f"  Calibrating (factor={factor:.2f})...", end=" ", flush=True)
            pruner = MatrixPruner(
                weight_or_module=info.module,
                pruning_factor=factor,
                config=pruning_config
            )
            pruner.calibrate(activations)
            self.pruners[name] = pruner
            print("done")
            
            # Free memory
            del activations
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        self.calibrated = True
        print(f"\nCalibration complete: {len(self.pruners)} layers")
    
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
        
        Handles both standard and quantized (4-bit/8-bit) models.
        """
        self.pruning_factors = pruning_factors
        
        pruning_config = PruningConfig(
            learning_rate=self.config.learning_rate,
            num_epochs=self.config.num_epochs,
            batch_size=self.config.batch_size,
            device=self.device
        )
        
        print("Calibrating pruners for each layer...")
        
        # Check if any layers are quantized
        n_quantized = sum(1 for info in self.prunable_layers.values() if info.is_quantized)
        if n_quantized > 0:
            print(f"  Note: {n_quantized} layers are quantized (4-bit/8-bit)")
            print("  Dequantizing weights for calibration...")
        
        for name, info in tqdm(self.prunable_layers.items(), desc="Calibrating"):
            if name not in activations:
                print(f"  Warning: No activations for {name}, skipping")
                continue
            
            factor = pruning_factors.get(name, 0.5)  # Default 50% compression
            
            # Pass the module directly - MatrixPruner handles dequantization
            pruner = MatrixPruner(
                weight_or_module=info.module,
                pruning_factor=factor,
                config=pruning_config
            )
            
            pruner.calibrate(activations[name])
            self.pruners[name] = pruner
        
        self.calibrated = True
        print(f"Calibrated {len(self.pruners)} pruners")
    
    def simple_prune(
        self,
        dataloader,
        pruning_factor: float = 0.5,
        max_tokens: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Simple single-pass pruning with a fixed factor.
        
        Use this when you know what pruning factor you want, or for
        memory-constrained situations where you can't afford multiple
        evaluation passes.
        
        Args:
            dataloader: Calibration data
            pruning_factor: Compression factor (0.5 = keep 50% of params)
            max_tokens: Max tokens for calibration
        
        Returns:
            Dict of pruning factors (same factor for all layers)
        """
        print(f"\n{'=' * 50}")
        print(f"SIMPLE PRUNING")
        print(f"Pruning factor: {pruning_factor} ({1-pruning_factor:.0%} compression)")
        print(f"{'=' * 50}\n")
        
        # Create uniform factors
        uniform_factors = {name: pruning_factor for name in self.prunable_layers}
        
        # Calibrate (memory efficient)
        self.calibrate_memory_efficient(dataloader, uniform_factors, max_tokens)
        
        return uniform_factors

    def uniform_prune(
        self,
        dataloader,
        eval_fn: Callable[[nn.Module], float],
        baseline_accuracy: float,
        max_tokens: Optional[int] = None,
        memory_efficient: bool = True
    ) -> Dict[str, float]:
        """
        Find optimal uniform pruning factor via binary search.
        
        Args:
            dataloader: Calibration data
            eval_fn: Function that evaluates model and returns accuracy
            baseline_accuracy: Original model accuracy
            max_tokens: Max tokens for calibration
            memory_efficient: Use memory-efficient mode (recommended for 7B+ models)
        
        Returns:
            Dict of pruning factors (same factor for all layers)
        """
        threshold = baseline_accuracy * (1 - self.config.target_accuracy_drop)
        
        print(f"\n{'=' * 50}")
        print(f"UNIFORM PRUNING")
        print(f"Baseline: {baseline_accuracy:.4f}")
        print(f"Threshold: {threshold:.4f} ({self.config.target_accuracy_drop:.0%} drop)")
        print(f"Memory efficient: {memory_efficient}")
        print(f"{'=' * 50}\n")
        
        # Try each pruning factor
        best_factor = 1.0  # No pruning
        
        for factor in sorted(self.config.pruning_factors):
            print(f"\nTrying pruning factor: {factor}")
            
            # Create uniform factors
            uniform_factors = {name: factor for name in self.prunable_layers}
            
            # Calibrate with this factor
            if memory_efficient:
                self.calibrate_memory_efficient(dataloader, uniform_factors, max_tokens)
            else:
                activations = self.collect_activations(dataloader, max_tokens)
                self.calibrate_pruners(activations, uniform_factors)
                del activations
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
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
        
        if memory_efficient:
            self.calibrate_memory_efficient(dataloader, final_factors, max_tokens)
        else:
            activations = self.collect_activations(dataloader, max_tokens)
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
