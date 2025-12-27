"""
MXFP4 Quantization Utilities for GPT-OSS Models.

GPT-OSS uses MXFP4 (4-bit microscaling floating point) quantization 
for MoE expert weights. This module provides utilities for:
1. Detecting MXFP4 quantized weights
2. Dequantizing for REAP saliency computation
3. Preserving quantization format after pruning (when possible)

Note: Full MXFP4 support requires the `kernels` package from OpenAI.
This module provides fallback behavior when that package is unavailable.
"""

from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import warnings


def is_mxfp4_model(model: nn.Module) -> bool:
    """
    Check if a model uses MXFP4 quantization.
    
    MXFP4 tensors in GPT-OSS are stored as:
    - tensor.blocks: FP4 values packed as uint8 (2 values per byte)
    - tensor.scales: Block scale factors
    """
    for name, param in model.named_parameters():
        # MXFP4 weights are typically stored differently
        if hasattr(param, 'is_mxfp4') and param.is_mxfp4:
            return True
        # Alternative detection: check for unusual dtypes
        if param.dtype == torch.uint8 and 'experts' in name:
            return True
    return False


def estimate_memory_reduction(
    original_num_experts: int,
    pruned_num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    num_layers: int,
    is_mxfp4: bool = True,
) -> Dict[str, float]:
    """
    Estimate memory reduction from expert pruning.
    
    Args:
        original_num_experts: Number of experts before pruning
        pruned_num_experts: Number of experts after pruning
        hidden_size: Model hidden dimension
        intermediate_size: MoE intermediate dimension (typically 4x hidden)
        num_layers: Number of MoE layers
        is_mxfp4: Whether model uses MXFP4 quantization
        
    Returns:
        Dict with memory estimates in GB
    """
    # Parameters per expert: gate_proj + up_proj + down_proj
    # gate/up: hidden -> intermediate (each)
    # down: intermediate -> hidden
    params_per_expert = (
        2 * hidden_size * intermediate_size +  # gate + up
        intermediate_size * hidden_size  # down
    )
    
    # Bytes per parameter
    if is_mxfp4:
        # MXFP4 is ~4.25 bits per parameter with block scales
        bytes_per_param = 4.25 / 8
    else:
        # BF16 is 2 bytes per parameter
        bytes_per_param = 2
    
    original_expert_memory = (
        original_num_experts * params_per_expert * 
        bytes_per_param * num_layers
    ) / (1024**3)  # Convert to GB
    
    pruned_expert_memory = (
        pruned_num_experts * params_per_expert * 
        bytes_per_param * num_layers
    ) / (1024**3)
    
    return {
        "original_gb": original_expert_memory,
        "pruned_gb": pruned_expert_memory,
        "reduction_gb": original_expert_memory - pruned_expert_memory,
        "reduction_pct": 1 - (pruned_num_experts / original_num_experts),
    }


class MXFP4Handler:
    """
    Handler for MXFP4 quantized weights in GPT-OSS.
    
    This class provides methods to work with MXFP4 weights during pruning,
    including dequantization for saliency computation and requantization
    after pruning.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self._has_mxfp4 = None
        self._kernels_available = self._check_kernels()
        
    def _check_kernels(self) -> bool:
        """Check if OpenAI kernels package is available."""
        try:
            import kernels
            return True
        except ImportError:
            return False
    
    @property
    def has_mxfp4(self) -> bool:
        """Check if model has MXFP4 quantized weights."""
        if self._has_mxfp4 is None:
            self._has_mxfp4 = is_mxfp4_model(self.model)
        return self._has_mxfp4
    
    def dequantize_experts(self, moe_layer: nn.Module) -> None:
        """
        Dequantize expert weights to BF16 for saliency computation.
        
        Note: This modifies the layer in-place. Use with caution.
        """
        if not self.has_mxfp4:
            return
            
        if not self._kernels_available:
            warnings.warn(
                "MXFP4 detected but 'kernels' package not available. "
                "Assuming weights are already in BF16."
            )
            return
            
        # Use OpenAI kernels for dequantization
        try:
            from kernels import mxfp4_dequantize
            
            for expert in moe_layer.experts:
                for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                    proj = getattr(expert, proj_name)
                    if hasattr(proj, 'weight_blocks'):
                        # Dequantize MXFP4 weight
                        weight_bf16 = mxfp4_dequantize(
                            proj.weight_blocks,
                            proj.weight_scales,
                        )
                        proj.weight = nn.Parameter(weight_bf16)
                        # Remove MXFP4 specific attributes
                        delattr(proj, 'weight_blocks')
                        delattr(proj, 'weight_scales')
        except Exception as e:
            warnings.warn(f"MXFP4 dequantization failed: {e}")
    
    def quantize_experts(self, moe_layer: nn.Module) -> None:
        """
        Re-quantize expert weights to MXFP4 after pruning.
        
        Note: This is optional and requires the kernels package.
        """
        if not self._kernels_available:
            warnings.warn(
                "Cannot quantize to MXFP4: 'kernels' package not available. "
                "Experts will remain in BF16."
            )
            return
            
        try:
            from kernels import mxfp4_quantize
            
            for expert in moe_layer.experts:
                for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                    proj = getattr(expert, proj_name)
                    if hasattr(proj, 'weight'):
                        # Quantize to MXFP4
                        blocks, scales = mxfp4_quantize(proj.weight)
                        proj.weight_blocks = nn.Parameter(blocks, requires_grad=False)
                        proj.weight_scales = nn.Parameter(scales, requires_grad=False)
                        delattr(proj, 'weight')
        except Exception as e:
            warnings.warn(f"MXFP4 quantization failed: {e}")


def get_gptoss_memory_footprint(
    model_variant: str = "20b",
) -> Dict[str, float]:
    """
    Get memory footprint estimates for GPT-OSS models.
    
    Args:
        model_variant: Either "20b" or "120b"
        
    Returns:
        Dict with memory estimates
    """
    configs = {
        "20b": {
            "num_layers": 24,
            "num_experts": 32,
            "hidden_size": 2880,
            "intermediate_size": 2880 * 4,  # Approximate
            "total_params": 21e9,
            "active_params": 3.6e9,
        },
        "120b": {
            "num_layers": 36,
            "num_experts": 128,
            "hidden_size": 2880,
            "intermediate_size": 2880 * 4,
            "total_params": 117e9,
            "active_params": 5.1e9,
        },
    }
    
    config = configs.get(model_variant)
    if config is None:
        raise ValueError(f"Unknown variant: {model_variant}. Use '20b' or '120b'")
    
    # MXFP4 is ~0.53 bytes per parameter for quantized weights
    # Non-MoE weights are BF16 (~2 bytes per parameter)
    moe_params = config["total_params"] * 0.9  # ~90% is MoE
    non_moe_params = config["total_params"] * 0.1
    
    moe_memory_gb = moe_params * 0.53 / (1024**3)
    non_moe_memory_gb = non_moe_params * 2 / (1024**3)
    
    return {
        "model": f"gpt-oss-{model_variant}",
        "total_memory_gb": moe_memory_gb + non_moe_memory_gb,
        "moe_memory_gb": moe_memory_gb,
        "non_moe_memory_gb": non_moe_memory_gb,
        "bf16_equivalent_gb": config["total_params"] * 2 / (1024**3),
        "compression_vs_bf16": (moe_memory_gb + non_moe_memory_gb) / (config["total_params"] * 2 / (1024**3)),
    }
