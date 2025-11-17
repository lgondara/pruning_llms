"""
High-level interface for LLM-Sieve that works with Hugging Face transformers.

This module provides an easy-to-use API for pruning transformer models.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass
import json
from pathlib import Path

from llm_sieve import MatrixPruner, PruningConfig
from genetic_pruning import GeneticPruningOptimizer, GeneticConfig


@dataclass
class LLMSieveConfig:
    """Configuration for LLM-Sieve pruning pipeline."""
    # Calibration settings
    calibration_tokens: int = 200_000
    batch_size: int = 5000
    learning_rate: float = 0.001
    num_epochs: int = 2
    
    # Pruning strategy
    pruning_strategy: str = "uniform"  # "uniform" or "adaptive"
    target_accuracy_drop: float = 0.05  # 5% accuracy drop tolerance
    
    # GA settings (for adaptive pruning)
    ga_population_size: int = 100
    ga_generations: int = 30
    ga_crossover_prob: float = 0.5
    ga_mutation_prob: float = 0.2
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Which matrices to prune
    prune_attention: bool = True
    prune_ffn: bool = True
    prune_embeddings: bool = False  # Usually keep embeddings


class TransformerPruner:
    """
    High-level interface for pruning transformer models.
    
    Supports both uniform and adaptive pruning strategies.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 config: Optional[LLMSieveConfig] = None):
        """
        Args:
            model: Transformer model to prune (e.g., from Hugging Face)
            config: Pruning configuration
        """
        self.model = model
        self.config = config or LLMSieveConfig()
        
        # Extract all linear layers to prune
        self.linear_layers = self._extract_linear_layers()
        self.layer_dims = self._compute_layer_dims()
        
        # Storage for pruners
        self.pruners: Dict[str, MatrixPruner] = {}
        self.pruning_factors: Dict[str, float] = {}
        
        print(f"Found {len(self.linear_layers)} linear layers to potentially prune")
    
    def _extract_linear_layers(self) -> Dict[str, nn.Linear]:
        """
        Extract all linear layers from the model that should be pruned.
        
        Returns dictionary mapping layer names to nn.Linear modules.
        """
        linear_layers = {}
        
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            
            # Determine if this layer should be pruned
            should_prune = False
            
            # Attention layers
            if self.config.prune_attention:
                if any(x in name.lower() for x in ['q_proj', 'k_proj', 'v_proj', 
                                                     'qkv', 'o_proj', 'out_proj',
                                                     'attention']):
                    should_prune = True
            
            # FFN layers
            if self.config.prune_ffn:
                if any(x in name.lower() for x in ['mlp', 'fc1', 'fc2', 'ffn',
                                                     'gate_proj', 'up_proj', 'down_proj']):
                    should_prune = True
            
            # Skip embeddings unless explicitly enabled
            if not self.config.prune_embeddings:
                if any(x in name.lower() for x in ['embed', 'wte', 'wpe']):
                    should_prune = False
            
            # Skip layer norm and final output layers
            if any(x in name.lower() for x in ['ln', 'norm', 'lm_head']):
                should_prune = False
            
            if should_prune:
                linear_layers[name] = module
        
        return linear_layers
    
    def _compute_layer_dims(self) -> Dict[str, Tuple[int, int]]:
        """Compute dimensions (H, D) for each layer."""
        return {
            name: (module.out_features, module.in_features)
            for name, module in self.linear_layers.items()
        }
    
    def _determine_projection_type(self, name: str, module: nn.Linear) -> str:
        """
        Determine if layer is up-projection or down-projection.
        
        Up-projection: increases dimensionality (e.g., Q/K/V, FFN up)
        Down-projection: decreases dimensionality (e.g., attention out, FFN down)
        """
        out_dim = module.out_features
        in_dim = module.in_features
        
        # Common patterns
        if any(x in name.lower() for x in ['o_proj', 'out_proj', 'down_proj', 'fc2']):
            return "down"
        elif any(x in name.lower() for x in ['q_proj', 'k_proj', 'v_proj', 
                                               'qkv', 'up_proj', 'gate_proj', 'fc1']):
            return "up"
        
        # Default based on dimensions
        return "up" if out_dim > in_dim else "down"
    
    def collect_activations(self,
                           calibration_data: List[Dict],
                           max_samples: Optional[int] = None) -> Dict[str, List[torch.Tensor]]:
        """
        Run forward passes to collect activations for calibration.
        
        Args:
            calibration_data: List of input samples (e.g., tokenized text)
            max_samples: Maximum number of samples to use
            
        Returns:
            Dictionary mapping layer names to lists of activation tensors
        """
        activations = {name: [] for name in self.linear_layers.keys()}
        
        # Register hooks to capture activations
        handles = []
        
        def make_hook(layer_name):
            def hook(module, input, output):
                # Input is a tuple, get first element
                if isinstance(input, tuple):
                    input_tensor = input[0]
                else:
                    input_tensor = input
                
                # Detach and move to CPU to save memory
                activations[layer_name].append(input_tensor.detach().cpu())
            return hook
        
        for name, module in self.linear_layers.items():
            handle = module.register_forward_hook(make_hook(name))
            handles.append(handle)
        
        # Run forward passes
        self.model.eval()
        with torch.no_grad():
            samples_processed = 0
            for batch in calibration_data:
                if max_samples and samples_processed >= max_samples:
                    break
                
                # Forward pass
                if isinstance(batch, dict):
                    _ = self.model(**{k: v.to(self.config.device) 
                                     for k, v in batch.items()})
                else:
                    _ = self.model(batch.to(self.config.device))
                
                samples_processed += 1
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        print(f"Collected activations from {samples_processed} samples")
        return activations
    
    def calibrate_pruners(self,
                         activations: Dict[str, List[torch.Tensor]],
                         pruning_factors: Optional[Dict[str, float]] = None,
                         verbose: bool = True) -> None:
        """
        Learn adapter matrices for all layers.
        
        Args:
            activations: Activations collected from calibration data
            pruning_factors: Pruning factor for each layer (or None for uniform 0.5)
            verbose: Print progress
        """
        if pruning_factors is None:
            pruning_factors = {name: 0.5 for name in self.linear_layers.keys()}
        
        self.pruning_factors = pruning_factors
        
        if verbose:
            print(f"\nCalibrating {len(self.linear_layers)} layers...")
        
        for name, module in self.linear_layers.items():
            if name not in activations or len(activations[name]) == 0:
                print(f"Warning: No activations for {name}, skipping")
                continue
            
            pruning_factor = pruning_factors.get(name, 0.5)
            projection_type = self._determine_projection_type(name, module)
            
            if verbose:
                print(f"\n{name}:")
                print(f"  Shape: {module.weight.shape}")
                print(f"  Type: {projection_type}-projection")
                print(f"  Pruning factor: {pruning_factor:.3f}")
            
            # Create pruner
            pruning_config = PruningConfig(
                rank=0,  # Will be computed
                learning_rate=self.config.learning_rate,
                num_epochs=self.config.num_epochs,
                batch_size=self.config.batch_size,
                device=self.config.device
            )
            
            pruner = MatrixPruner(
                weight=module.weight.data,
                pruning_factor=pruning_factor,
                projection_type=projection_type,
                config=pruning_config
            )
            
            # Calibrate
            loss = pruner.calibrate(activations[name], verbose=verbose)
            
            if verbose:
                print(f"  Final loss: {loss:.6f}")
                print(f"  Compression: {pruner.compute_pruning_ratio():.2%}")
            
            self.pruners[name] = pruner
    
    def uniform_prune(self,
                     calibration_data: List[Dict],
                     evaluate_fn: Callable[[nn.Module], float],
                     baseline_accuracy: float,
                     verbose: bool = True) -> Dict[str, float]:
        """
        Perform uniform pruning with binary search.
        
        Args:
            calibration_data: Data for collecting activations
            evaluate_fn: Function that evaluates model accuracy
            baseline_accuracy: Accuracy of unpruned model
            verbose: Print progress
            
        Returns:
            Dictionary of pruning factors
        """
        target_accuracy = baseline_accuracy * (1 - self.config.target_accuracy_drop)
        
        if verbose:
            print(f"Starting uniform pruning")
            print(f"Baseline accuracy: {baseline_accuracy:.2%}")
            print(f"Target accuracy: {target_accuracy:.2%}")
        
        # Collect activations once
        activations = self.collect_activations(calibration_data)
        
        # Binary search for best uniform factor
        def evaluate_pruning(pruning_factors):
            self.calibrate_pruners(activations, pruning_factors, verbose=False)
            self.apply_pruning()
            accuracy = evaluate_fn(self.model)
            self.restore_original_weights()
            return accuracy
        
        from llm_sieve import uniform_pruning_search
        
        pruning_factors = uniform_pruning_search(
            evaluate_fn=evaluate_pruning,
            layer_dims=self.layer_dims,
            target_accuracy=target_accuracy,
            baseline_accuracy=baseline_accuracy,
            verbose=verbose
        )
        
        # Final calibration with best factors
        self.calibrate_pruners(activations, pruning_factors, verbose=verbose)
        
        return pruning_factors
    
    def adaptive_prune(self,
                      calibration_data: List[Dict],
                      evaluate_fn: Callable[[nn.Module], float],
                      baseline_accuracy: float,
                      verbose: bool = True) -> Dict[str, float]:
        """
        Perform adaptive pruning with genetic algorithm.
        
        Args:
            calibration_data: Data for collecting activations  
            evaluate_fn: Function that evaluates model accuracy
            baseline_accuracy: Accuracy of unpruned model
            verbose: Print progress
            
        Returns:
            Dictionary of pruning factors
        """
        target_accuracy = baseline_accuracy * (1 - self.config.target_accuracy_drop)
        
        if verbose:
            print(f"Starting adaptive pruning with GA")
            print(f"Baseline accuracy: {baseline_accuracy:.2%}")
            print(f"Target accuracy: {target_accuracy:.2%}")
        
        # Collect activations once
        activations = self.collect_activations(calibration_data)
        
        # Evaluation function for GA
        def evaluate_pruning(pruning_factors):
            self.calibrate_pruners(activations, pruning_factors, verbose=False)
            self.apply_pruning()
            accuracy = evaluate_fn(self.model)
            self.restore_original_weights()
            return accuracy
        
        # Run genetic algorithm
        ga_config = GeneticConfig(
            population_size=self.config.ga_population_size,
            num_generations=self.config.ga_generations,
            crossover_prob=self.config.ga_crossover_prob,
            mutation_prob=self.config.ga_mutation_prob
        )
        
        optimizer = GeneticPruningOptimizer(
            matrix_names=list(self.linear_layers.keys()),
            layer_dims=self.layer_dims,
            evaluate_fn=evaluate_pruning,
            target_accuracy=target_accuracy,
            config=ga_config
        )
        
        best_chromosome = optimizer.optimize(verbose=verbose)
        pruning_factors = best_chromosome.pruning_factors
        
        # Final calibration with best factors
        self.calibrate_pruners(activations, pruning_factors, verbose=verbose)
        
        return pruning_factors
    
    def apply_pruning(self) -> None:
        """Apply learned pruning to the model by replacing weights."""
        for name, pruner in self.pruners.items():
            module = self.linear_layers[name]
            W_pruned, A_pseudo = pruner.get_pruned_weights()
            
            # Store original weight for restoration
            if not hasattr(module, '_original_weight'):
                module._original_weight = module.weight.data.clone()
            
            # Replace weight with pruned version
            # Note: This is a simplified version. In practice, you'd want to
            # modify the forward pass to use both W_pruned and A_pseudo
            module.weight.data = W_pruned.to(module.weight.device)
    
    def restore_original_weights(self) -> None:
        """Restore original unpruned weights."""
        for name, module in self.linear_layers.items():
            if hasattr(module, '_original_weight'):
                module.weight.data = module._original_weight.clone()
    
    def save_pruned_model(self, save_path: str) -> None:
        """Save pruned model and pruning configuration."""
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        torch.save(self.model.state_dict(), save_dir / "pruned_model.pt")
        
        # Save pruning configuration
        config_dict = {
            'pruning_factors': self.pruning_factors,
            'layer_dims': {k: list(v) for k, v in self.layer_dims.items()},
            'compression_ratio': self.get_compression_ratio()
        }
        
        with open(save_dir / "pruning_config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"Saved pruned model to {save_dir}")
    
    def get_compression_ratio(self) -> float:
        """Compute overall compression ratio."""
        from llm_sieve import compute_compression_ratio
        return compute_compression_ratio(self.pruning_factors, self.layer_dims)
    
    def print_statistics(self) -> None:
        """Print pruning statistics."""
        print("\n" + "=" * 60)
        print("PRUNING STATISTICS")
        print("=" * 60)
        
        total_original = sum(h * d for h, d in self.layer_dims.values())
        print(f"Original parameters: {total_original:,}")
        
        compression = self.get_compression_ratio()
        total_pruned = int(total_original * (1 - compression))
        print(f"Pruned parameters: {total_pruned:,}")
        print(f"Overall compression: {compression:.2%}")
        
        # Group by matrix type
        from collections import defaultdict
        type_stats = defaultdict(list)
        
        for name, factor in self.pruning_factors.items():
            matrix_type = name.split('.')[-1]  # Get last part
            type_stats[matrix_type].append(factor)
        
        print("\nAverage pruning factor by matrix type:")
        for mtype in sorted(type_stats.keys()):
            factors = type_stats[mtype]
            avg_factor = sum(factors) / len(factors)
            avg_compression = 1.0 - avg_factor
            print(f"  {mtype:20s}: {avg_factor:.3f} ({avg_compression:.1%} compressed)")


# Example usage
if __name__ == "__main__":
    print("LLM-Sieve High-Level Interface")
    print("=" * 60)
    
    # This is a simplified example with a dummy model
    # In practice, you'd use an actual transformer model
    
    class DummyTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.ModuleDict({
                    'q_proj': nn.Linear(512, 512),
                    'k_proj': nn.Linear(512, 512),
                    'v_proj': nn.Linear(512, 512),
                    'o_proj': nn.Linear(512, 512),
                    'fc1': nn.Linear(512, 2048),
                    'fc2': nn.Linear(2048, 512),
                })
                for _ in range(4)
            ])
        
        def forward(self, x):
            for layer in self.layers:
                # Simplified forward pass
                q = layer['q_proj'](x)
                k = layer['k_proj'](x)
                v = layer['v_proj'](x)
                attn_out = layer['o_proj'](v)
                ffn_out = layer['fc2'](torch.relu(layer['fc1'](attn_out)))
                x = x + ffn_out
            return x
    
    model = DummyTransformer()
    
    config = LLMSieveConfig(
        pruning_strategy="uniform",
        target_accuracy_drop=0.05
    )
    
    pruner = TransformerPruner(model, config)
    pruner.print_statistics()
