"""
LLM-Sieve: Task-Specific Pruning for Large Language Models

Implementation of the paper "How Many Parameters Does Your Task Really Need?"
by Waleed Reda, Abhinav Jangda, Krishna Chintalapudi (Microsoft Research)

Key innovations:
1. Output-aligned non-orthogonal projections for low-rank approximation
2. Adaptive pruning via Genetic Algorithm to find optimal compression per matrix
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Dict, List, Optional
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class PruningConfig:
    """Configuration for LLM-Sieve pruning."""
    rank: int  # Reduced rank for low-rank approximation
    learning_rate: float = 0.001
    num_epochs: int = 2
    batch_size: int = 5000  # tokens per batch
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    

class LowRankAdapter(nn.Module):
    """
    Learns a low-rank adapter matrix for output-aligned projection.
    
    For up-projections (expanding dim): Y ≈ (WA)(A†X)
    For down-projections (reducing dim): Y ≈ A†((AW)X)
    """
    
    def __init__(self, rank: int, original_dim: int, projection_type: str = "up"):
        """
        Args:
            rank: Target rank for compression
            original_dim: Original dimension (H for up-projection, D for down-projection)
            projection_type: "up" or "down" projection
        """
        super().__init__()
        self.rank = rank
        self.original_dim = original_dim
        self.projection_type = projection_type
        
        # Initialize adapter matrix A
        # For up-projection: A ∈ R^(R×H)
        # For down-projection: A ∈ R^(R×D)
        self.A = nn.Parameter(torch.randn(rank, original_dim) * 0.01)
        
    def forward(self, X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """
        Apply low-rank approximation.
        
        Args:
            X: Input activations [batch, seq_len, in_dim]
            W: Original weight matrix [out_dim, in_dim]
            
        Returns:
            Approximated output [batch, seq_len, out_dim]
        """
        if self.projection_type == "up":
            # Up-projection: Y ≈ (WA)(A†X)
            # First project input down: X_proj = A† @ X
            X_proj = torch.matmul(X, self.A.T)  # [batch, seq_len, rank]
            # Then project through compressed weight: Y = (WA) @ X_proj
            W_compressed = torch.matmul(W, self.A.T)  # [out_dim, rank]
            Y_approx = torch.matmul(X_proj, W_compressed.T)  # [batch, seq_len, out_dim]
        else:
            # Down-projection: Y ≈ A†((AW)X)
            # First compute in compressed space: Y_compressed = (AW) @ X
            W_compressed = torch.matmul(self.A, W)  # [rank, in_dim]
            Y_compressed = torch.matmul(X, W_compressed.T)  # [batch, seq_len, rank]
            # Project back up: Y = A† @ Y_compressed
            Y_approx = torch.matmul(Y_compressed, self.A)  # [batch, seq_len, out_dim]
            
        return Y_approx
    
    def get_pruned_weights(self, W: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the pruned weight matrices for inference.
        
        Returns:
            For up-projection: (W_pruned, A†) where W_pruned = WA
            For down-projection: (W_pruned, A†) where W_pruned = AW
        """
        with torch.no_grad():
            if self.projection_type == "up":
                W_pruned = torch.matmul(W, self.A.T)  # [out_dim, rank]
                A_pseudo = self.A.T  # [in_dim, rank] for X @ A†
            else:
                W_pruned = torch.matmul(self.A, W)  # [rank, in_dim]
                A_pseudo = self.A  # [rank, out_dim] for Y @ A
                
        return W_pruned, A_pseudo


class MatrixPruner:
    """
    Handles pruning of a single matrix in the transformer.
    """
    
    def __init__(self, 
                 weight: torch.Tensor, 
                 pruning_factor: float,
                 projection_type: str = "up",
                 config: Optional[PruningConfig] = None):
        """
        Args:
            weight: Original weight matrix to prune
            pruning_factor: Fraction of parameters to keep (0 < p <= 1)
            projection_type: "up" or "down" projection
            config: Pruning configuration
        """
        self.weight = weight.detach()
        self.pruning_factor = pruning_factor
        self.projection_type = projection_type
        self.config = config or PruningConfig(rank=self._compute_rank())
        
        # Compute target rank
        self.rank = self._compute_rank()
        
        # Initialize adapter
        original_dim = weight.shape[1] if projection_type == "up" else weight.shape[0]
        self.adapter = LowRankAdapter(self.rank, original_dim, projection_type)
        self.adapter.to(self.config.device)
        
    def _compute_rank(self) -> int:
        """
        Compute target rank R from pruning factor p.
        
        For W ∈ R^(H×D), pruned form has R(H+D) parameters.
        Pruning factor: p = R(H+D) / (HD)
        Therefore: R = p*HD / (H+D)
        """
        H, D = self.weight.shape
        R = int(self.pruning_factor * H * D / (H + D))
        R = max(1, min(R, min(H, D)))  # Ensure valid rank
        return R
    
    def calibrate(self, 
                  activations: List[torch.Tensor],
                  verbose: bool = False) -> float:
        """
        Learn the adapter matrix by minimizing reconstruction error.
        
        Args:
            activations: List of input activations [batch, seq_len, in_dim]
            verbose: Print training progress
            
        Returns:
            Final reconstruction loss
        """
        self.adapter.train()
        optimizer = optim.Adam(self.adapter.parameters(), 
                              lr=self.config.learning_rate)
        
        best_loss = float('inf')
        best_state = None
        
        pbar = tqdm(range(self.config.num_epochs), desc="Calibrating adapter") if verbose else range(self.config.num_epochs)
        
        for epoch in pbar:
            epoch_loss = 0.0
            num_batches = 0
            
            for X in activations:
                X = X.to(self.config.device)
                
                # Compute true output
                with torch.no_grad():
                    Y_true = torch.matmul(X, self.weight.T.to(self.config.device))
                
                # Compute approximated output
                Y_approx = self.adapter(X, self.weight.to(self.config.device))
                
                # L2 reconstruction loss
                loss = torch.mean((Y_true - Y_approx) ** 2)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = {k: v.cpu().clone() for k, v in self.adapter.state_dict().items()}
            
            if verbose:
                pbar.set_postfix({"loss": f"{avg_loss:.6f}"})
        
        # Load best parameters
        if best_state is not None:
            self.adapter.load_state_dict(best_state)
            self.adapter.to(self.config.device)
        
        return best_loss
    
    def get_pruned_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get pruned weight matrices for inference."""
        return self.adapter.get_pruned_weights(self.weight.to(self.config.device))
    
    def compute_pruning_ratio(self) -> float:
        """Compute actual parameter reduction ratio."""
        H, D = self.weight.shape
        original_params = H * D
        pruned_params = self.rank * (H + D)
        return 1.0 - (pruned_params / original_params)


def compute_compression_ratio(pruning_factors: Dict[str, float],
                             layer_dims: Dict[str, Tuple[int, int]]) -> float:
    """
    Compute overall compression ratio for a set of pruning factors.
    
    Args:
        pruning_factors: Dictionary mapping matrix names to pruning factors
        layer_dims: Dictionary mapping matrix names to (H, D) dimensions
        
    Returns:
        Overall compression ratio (fraction of parameters removed)
    """
    total_original = 0
    total_pruned = 0
    
    for name, (H, D) in layer_dims.items():
        p = pruning_factors.get(name, 1.0)
        R = int(p * H * D / (H + D))
        R = max(1, min(R, min(H, D)))
        
        total_original += H * D
        total_pruned += R * (H + D)
    
    return 1.0 - (total_pruned / total_original)


def uniform_pruning_search(evaluate_fn,
                           layer_dims: Dict[str, Tuple[int, int]],
                           target_accuracy: float,
                           baseline_accuracy: float,
                           max_iterations: int = 10,
                           verbose: bool = True) -> Dict[str, float]:
    """
    Binary search for optimal uniform pruning factor.
    
    Args:
        evaluate_fn: Function that takes pruning_factors dict and returns accuracy
        layer_dims: Dictionary mapping matrix names to dimensions
        target_accuracy: Minimum acceptable accuracy
        baseline_accuracy: Original model accuracy
        max_iterations: Maximum binary search iterations
        verbose: Print search progress
        
    Returns:
        Dictionary of pruning factors (all equal)
    """
    p_low = 0.0
    p_high = 1.0
    best_p = 1.0
    
    if verbose:
        print(f"Starting binary search for uniform pruning...")
        print(f"Target accuracy: {target_accuracy:.2%} (baseline: {baseline_accuracy:.2%})")
    
    for iteration in range(max_iterations):
        p_mid = (p_low + p_high) / 2
        
        # Create uniform pruning factors
        pruning_factors = {name: p_mid for name in layer_dims.keys()}
        
        # Evaluate
        accuracy = evaluate_fn(pruning_factors)
        compression = compute_compression_ratio(pruning_factors, layer_dims)
        
        if verbose:
            print(f"Iteration {iteration+1}: p={p_mid:.3f}, "
                  f"accuracy={accuracy:.2%}, compression={compression:.2%}")
        
        if accuracy >= target_accuracy:
            # Can compress more
            best_p = p_mid
            p_high = p_mid
        else:
            # Need less compression
            p_low = p_mid
        
        # Convergence check
        if abs(p_high - p_low) < 0.01:
            break
    
    final_factors = {name: best_p for name in layer_dims.keys()}
    
    if verbose:
        final_compression = compute_compression_ratio(final_factors, layer_dims)
        print(f"\nFinal uniform pruning factor: {best_p:.3f}")
        print(f"Final compression ratio: {final_compression:.2%}")
    
    return final_factors


# Simple example usage
if __name__ == "__main__":
    print("LLM-Sieve Implementation")
    print("=" * 50)
    
    # Example: Prune a single linear layer
    print("\nExample: Pruning a single matrix")
    
    # Create a dummy weight matrix
    H, D = 1024, 512
    weight = torch.randn(H, D)
    
    # Create dummy activations
    batch_size, seq_len = 4, 128
    activations = [torch.randn(batch_size, seq_len, D) for _ in range(5)]
    
    # Prune with 50% pruning factor
    pruning_factor = 0.5
    config = PruningConfig(rank=0)  # Will be computed automatically
    
    pruner = MatrixPruner(weight, pruning_factor, projection_type="up", config=config)
    
    print(f"Original weight shape: {weight.shape}")
    print(f"Original parameters: {H * D:,}")
    print(f"Target rank: {pruner.rank}")
    print(f"Pruned parameters: {pruner.rank * (H + D):,}")
    print(f"Compression ratio: {pruner.compute_pruning_ratio():.2%}")
    
    # Calibrate
    print("\nCalibrating adapter...")
    final_loss = pruner.calibrate(activations, verbose=True)
    print(f"Final reconstruction loss: {final_loss:.6f}")
    
    # Get pruned weights
    W_pruned, A_pseudo = pruner.get_pruned_weights()
    print(f"\nPruned weight shape: {W_pruned.shape}")
    print(f"Adapter shape: {A_pseudo.shape}")
