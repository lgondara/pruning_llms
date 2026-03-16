"""
Layer Wrapper for Wanda Pruning
===============================

Wraps linear layers to capture input activations during calibration.
Used to compute the Wanda importance score: |W_ij| * ||X_j||_2
"""

import torch
import torch.nn as nn


class WrappedGPT:
    """
    Wraps a linear layer to accumulate input activation norms
    across calibration samples.

    The Wanda score for weight W_ij is:
        S_ij = |W_ij| * ||X_j||_2

    where X_j is the j-th input feature across all calibration tokens.
    We accumulate ||X_j||^2 incrementally and take sqrt at the end.
    """

    def __init__(self, layer: nn.Linear, layer_id: int = 0, layer_name: str = ""):
        self.layer = layer
        self.layer_id = layer_id
        self.layer_name = layer_name
        self.device = layer.weight.device

        W = layer.weight.data
        self.rows = W.shape[0]  # output features
        self.columns = W.shape[1]  # input features

        # Accumulator for squared input norms per feature
        self.scaler_row = torch.zeros(self.columns, device=self.device)
        self.nsamples = 0

    def add_batch(self, inp: torch.Tensor, out: torch.Tensor = None):
        """
        Accumulate input activation statistics from one batch.

        Args:
            inp: Input tensor of shape (batch, seq_len, hidden) or (batch*seq, hidden)
            out: Output tensor (unused, kept for hook API compatibility)
        """
        if len(inp.shape) == 3:
            inp = inp.reshape(-1, inp.shape[-1])  # (B*T, D)

        inp = inp.float()
        tmp = inp.shape[0]  # number of tokens in this batch

        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        self.scaler_row += torch.sum(inp ** 2, dim=0) / self.nsamples

    def get_importance_scores(self) -> torch.Tensor:
        """
        Compute Wanda importance scores.

        Returns:
            Tensor of shape (rows, columns) with importance scores
        """
        W = self.layer.weight.data.float()
        # |W_ij| * sqrt(sum(X_j^2) / N)  = |W_ij| * RMS(X_j) * sqrt(N)
        # But since we're only ranking, the sqrt(N) cancels out
        importance = torch.abs(W) * torch.sqrt(self.scaler_row.unsqueeze(0))
        return importance
