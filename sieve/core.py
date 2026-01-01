"""
LLM-Sieve Core: Low-Rank Adapter Learning

Implements output-aligned non-orthogonal projections as described in:
"How Many Parameters Does Your Task Really Need?" (Microsoft Research, 2025)

The key insight: SVD gives orthogonal bases that don't align with task-specific
outputs. Instead, we learn a non-orthogonal adapter A that minimizes ||Y - WXA||²
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional, List
from dataclasses import dataclass
import math


@dataclass
class PruningConfig:
    """Configuration for low-rank approximation."""
    learning_rate: float = 0.001
    num_epochs: int = 2
    batch_size: int = 5000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class LowRankAdapter(nn.Module):
    """
    Learns a low-rank adapter matrix A for output-aligned projection.
    
    Given weight W ∈ R^{H×D} and target rank R:
    - We learn A ∈ R^{D×R} that projects inputs to low-rank space
    - Pruned weight W' ∈ R^{H×R} = W @ A
    - Approximate Y ≈ W' @ A^T @ X
    
    This is better than SVD because:
    - SVD bases are orthogonal but not output-aligned
    - Learned A minimizes actual reconstruction error ||Y - Ỹ||²
    """
    
    def __init__(self, input_dim: int, target_rank: int, device: str = "cuda"):
        super().__init__()
        self.input_dim = input_dim
        self.target_rank = target_rank
        self.device = device
        
        # Adapter matrix: projects D-dim input to R-dim
        self.adapter = nn.Parameter(
            torch.randn(input_dim, target_rank, device=device) * 0.01
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input through adapter: X @ A"""
        return x @ self.adapter
    
    def get_projection_matrices(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (A, A^T) for use in pruning."""
        A = self.adapter.data
        return A, A.t()


class MatrixPruner:
    """
    Handles pruning for a single weight matrix.
    
    Steps:
    1. Collect activations X from calibration data
    2. Train adapter A to minimize ||WX - W'A^TX|| where W' = WA
    3. Replace W with factorized form (W', A^T)
    """
    
    def __init__(
        self,
        weight: torch.Tensor,
        pruning_factor: float,
        config: Optional[PruningConfig] = None
    ):
        """
        Args:
            weight: Original weight matrix [H, D]
            pruning_factor: Target compression (0.5 = keep 50% of params)
            config: Training configuration
        """
        self.config = config or PruningConfig()
        self.device = self.config.device
        
        # Store original weight
        self.W = weight.to(self.device).float()
        self.H, self.D = self.W.shape
        
        # Compute target rank from pruning factor
        # pruning_factor = R(H + D) / (HD)
        # R = pruning_factor * HD / (H + D)
        self.target_rank = self._compute_rank(pruning_factor)
        
        # Initialize adapter
        self.adapter = LowRankAdapter(self.D, self.target_rank, self.device)
        
        # Storage for calibration
        self.activations: List[torch.Tensor] = []
        self.calibrated = False
    
    def _compute_rank(self, pruning_factor: float) -> int:
        """Compute target rank from pruning factor."""
        # R = p * HD / (H + D)
        rank = int(pruning_factor * self.H * self.D / (self.H + self.D))
        # Clamp to valid range
        return max(1, min(rank, min(self.H, self.D)))
    
    def add_activations(self, x: torch.Tensor):
        """Collect input activations for calibration."""
        # x shape: [batch, seq_len, D] or [tokens, D]
        if x.dim() == 3:
            x = x.reshape(-1, x.size(-1))
        self.activations.append(x.detach().cpu())
    
    def calibrate(self, activations: Optional[torch.Tensor] = None):
        """
        Train the low-rank adapter using collected activations.
        
        Minimizes: ||WX - WA @ A^T @ X||²
        """
        if activations is not None:
            if activations.dim() == 3:
                activations = activations.reshape(-1, activations.size(-1))
            X = activations.to(self.device).float()
        elif self.activations:
            X = torch.cat(self.activations, dim=0).to(self.device).float()
        else:
            raise ValueError("No activations provided for calibration")
        
        # Compute target outputs: Y = WX^T (Y: [H, N])
        Y_target = self.W @ X.t()
        
        # Train adapter
        optimizer = optim.Adam(self.adapter.parameters(), lr=self.config.learning_rate)
        n_samples = X.size(0)
        batch_size = min(self.config.batch_size, n_samples)
        
        for epoch in range(self.config.num_epochs):
            indices = torch.randperm(n_samples)
            total_loss = 0.0
            n_batches = 0
            
            for i in range(0, n_samples, batch_size):
                batch_idx = indices[i:i + batch_size]
                X_batch = X[batch_idx]  # [B, D]
                Y_batch = Y_target[:, batch_idx]  # [H, B]
                
                optimizer.zero_grad()
                
                # Forward: X_batch @ A -> [B, R]
                X_proj = self.adapter(X_batch)
                
                # Pruned weight: W @ A -> [H, R]
                W_pruned = self.W @ self.adapter.adapter
                
                # Reconstruct: W' @ (A^T @ X^T) = W' @ X_proj^T
                Y_pred = W_pruned @ X_proj.t()  # [H, B]
                
                # Loss: ||Y - Y_pred||²
                loss = torch.mean((Y_batch - Y_pred) ** 2)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            avg_loss = total_loss / n_batches
        
        self.calibrated = True
        # Clear stored activations to free memory
        self.activations = []
    
    def get_pruned_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return factorized weights (W', A^T).
        
        Original: Y = W @ X
        Pruned:   Y ≈ W' @ A^T @ X  where W' = W @ A
        
        Returns:
            W_pruned: [H, R] - compressed weight
            A_transpose: [R, D] - projection matrix
        """
        if not self.calibrated:
            raise RuntimeError("Must calibrate before getting pruned weights")
        
        A, A_t = self.adapter.get_projection_matrices()
        W_pruned = self.W @ A  # [H, R]
        
        return W_pruned.detach(), A_t.detach()
    
    def get_compression_stats(self) -> dict:
        """Return compression statistics."""
        original_params = self.H * self.D
        pruned_params = self.target_rank * (self.H + self.D)
        compression = 1.0 - (pruned_params / original_params)
        
        return {
            "original_shape": (self.H, self.D),
            "target_rank": self.target_rank,
            "original_params": original_params,
            "pruned_params": pruned_params,
            "compression_ratio": compression,
            "pruning_factor": pruned_params / original_params
        }


class PrunedLinear(nn.Module):
    """
    Replacement module for a pruned linear layer.
    
    Stores factorized weights and performs efficient forward pass.
    """
    
    def __init__(self, W_pruned: torch.Tensor, A_transpose: torch.Tensor, bias: Optional[torch.Tensor] = None):
        """
        Args:
            W_pruned: [H, R] compressed weight
            A_transpose: [R, D] projection matrix
            bias: Optional bias term [H]
        """
        super().__init__()
        self.register_buffer('W_pruned', W_pruned)
        self.register_buffer('A_transpose', A_transpose)
        if bias is not None:
            self.register_buffer('bias', bias)
        else:
            self.bias = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Y = W' @ A^T @ X + b
        
        Computed as: (X @ A) @ W'^T + b for efficiency
        """
        # x: [..., D]
        # A^T: [R, D], so x @ A^T.T = x @ A -> [..., R]
        # W': [H, R], so (x @ A) @ W'^T -> [..., H]
        x_proj = x @ self.A_transpose.t()  # [..., R]
        out = x_proj @ self.W_pruned.t()   # [..., H]
        
        if self.bias is not None:
            out = out + self.bias
        
        return out
    
    @property
    def weight(self) -> torch.Tensor:
        """Reconstruct full weight for compatibility checks."""
        return (self.W_pruned @ self.A_transpose).t()


def compute_pruning_factor_for_rank(H: int, D: int, rank: int) -> float:
    """Compute pruning factor given target rank."""
    return rank * (H + D) / (H * D)


def compute_rank_for_pruning_factor(H: int, D: int, pruning_factor: float) -> int:
    """Compute target rank from pruning factor."""
    rank = int(pruning_factor * H * D / (H + D))
    return max(1, min(rank, min(H, D)))
