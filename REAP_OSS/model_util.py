"""
Model utility module for REAP adaptation to GPT-OSS models.

This module extends the original REAP model_util.py with support for 
OpenAI's GPT-OSS 20B and 120B mixture-of-experts models.

GPT-OSS Architecture Summary:
- GPT-OSS 20B: 24 layers, 32 experts, top-4 routing, 21B total / 3.6B active
- GPT-OSS 120B: 36 layers, 128 experts, top-4 routing, 117B total / 5.1B active
- Uses gated SwiGLU activation in MoE blocks
- MXFP4 quantized weights for MoE layers
- Alternating dense and banded sparse attention patterns
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig


@dataclass
class ModelAttributes:
    """Configuration for model-specific attribute names."""
    moe_block: str
    gate_proj: str
    up_proj: str
    down_proj: str
    experts: str
    router: str
    num_experts: str
    num_experts_per_tok: str
    fused: bool = False
    shared_experts: Optional[str] = None
    
    
# Model attribute registry - maps model class names to their configurations
MODEL_ATTRS: Dict[str, ModelAttributes] = {
    # GPT-OSS models (OpenAI)
    "GptOssForCausalLM": ModelAttributes(
        moe_block="moe",
        gate_proj="gate_proj",
        up_proj="up_proj",
        down_proj="down_proj",
        experts="experts",
        router="router",
        num_experts="num_experts",
        num_experts_per_tok="num_experts_per_tok",
        fused=False,
    ),
    "GptOssModel": ModelAttributes(
        moe_block="moe",
        gate_proj="gate_proj",
        up_proj="up_proj",
        down_proj="down_proj",
        experts="experts",
        router="router",
        num_experts="num_experts",
        num_experts_per_tok="num_experts_per_tok",
        fused=False,
    ),
    # Qwen3 models (for reference/comparison)
    "Qwen3ForCausalLM": ModelAttributes(
        moe_block="mlp",
        gate_proj="gate_proj",
        up_proj="up_proj",
        down_proj="down_proj",
        experts="experts",
        router="gate",
        num_experts="num_experts",
        num_experts_per_tok="num_experts_per_tok",
        fused=False,
        shared_experts="shared_expert",
    ),
    # DeepSeek models
    "DeepseekV3ForCausalLM": ModelAttributes(
        moe_block="mlp",
        gate_proj="gate_proj",
        up_proj="up_proj",
        down_proj="down_proj",
        experts="experts",
        router="gate",
        num_experts="n_routed_experts",
        num_experts_per_tok="num_experts_per_tok",
        fused=False,
        shared_experts="shared_experts",
    ),
}


def get_model_attrs(model: nn.Module) -> ModelAttributes:
    """
    Get model attributes configuration based on model class name.
    
    Args:
        model: The loaded HuggingFace model
        
    Returns:
        ModelAttributes configuration for the model
        
    Raises:
        ValueError: If model type is not supported
    """
    model_class = model.__class__.__name__
    if model_class not in MODEL_ATTRS:
        raise ValueError(
            f"Model class '{model_class}' not supported. "
            f"Supported models: {list(MODEL_ATTRS.keys())}"
        )
    return MODEL_ATTRS[model_class]


def get_moe_layers(model: nn.Module) -> List[Tuple[int, nn.Module]]:
    """
    Extract all MoE layers from a model.
    
    Args:
        model: The loaded model
        
    Returns:
        List of (layer_idx, moe_module) tuples
    """
    attrs = get_model_attrs(model)
    moe_layers = []
    
    # Navigate to decoder layers
    if hasattr(model, "model"):
        base_model = model.model
    else:
        base_model = model
        
    if hasattr(base_model, "layers"):
        layers = base_model.layers
    elif hasattr(base_model, "decoder") and hasattr(base_model.decoder, "layers"):
        layers = base_model.decoder.layers
    else:
        raise AttributeError("Cannot find decoder layers in model")
    
    for idx, layer in enumerate(layers):
        if hasattr(layer, attrs.moe_block):
            moe_module = getattr(layer, attrs.moe_block)
            # Check if this is actually a MoE layer (has experts)
            if hasattr(moe_module, attrs.experts):
                moe_layers.append((idx, moe_module))
                
    return moe_layers


def get_num_experts(model: nn.Module) -> int:
    """Get the number of experts per MoE layer from model config."""
    attrs = get_model_attrs(model)
    return getattr(model.config, attrs.num_experts)


def get_num_experts_per_tok(model: nn.Module) -> int:
    """Get the number of experts activated per token."""
    attrs = get_model_attrs(model)
    return getattr(model.config, attrs.num_experts_per_tok)


def get_experts_module(moe_layer: nn.Module, model: nn.Module) -> nn.ModuleList:
    """Get the experts ModuleList from a MoE layer."""
    attrs = get_model_attrs(model)
    return getattr(moe_layer, attrs.experts)


def get_router(moe_layer: nn.Module, model: nn.Module) -> nn.Module:
    """Get the router/gate module from a MoE layer."""
    attrs = get_model_attrs(model)
    return getattr(moe_layer, attrs.router)


def get_expert_projections(
    expert: nn.Module, 
    model: nn.Module
) -> Tuple[nn.Module, nn.Module, nn.Module]:
    """
    Get the gate, up, and down projections from an expert.
    
    Returns:
        Tuple of (gate_proj, up_proj, down_proj)
    """
    attrs = get_model_attrs(model)
    gate = getattr(expert, attrs.gate_proj)
    up = getattr(expert, attrs.up_proj)
    down = getattr(expert, attrs.down_proj)
    return gate, up, down


def load_gptoss_model(
    model_name_or_path: str,
    device_map: str = "auto",
    torch_dtype: str = "auto",
    trust_remote_code: bool = True,
    use_bf16: bool = True,
) -> nn.Module:
    """
    Load a GPT-OSS model with appropriate settings for REAP processing.
    
    Note: GPT-OSS models use MXFP4 quantization by default. For REAP 
    processing, we may need to work with BF16 weights depending on
    the operation.
    
    Args:
        model_name_or_path: HuggingFace model identifier
        device_map: Device placement strategy
        torch_dtype: Data type for model weights
        trust_remote_code: Whether to trust remote code
        use_bf16: If True, upcast MXFP4 weights to BF16
        
    Returns:
        Loaded model ready for REAP processing
    """
    # GPT-OSS specific loading
    if torch_dtype == "auto":
        dtype = torch.bfloat16 if use_bf16 else None
    else:
        dtype = getattr(torch, torch_dtype) if isinstance(torch_dtype, str) else torch_dtype
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map=device_map,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    
    return model


def count_expert_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count parameters in experts vs. non-expert components.
    
    Returns:
        Dictionary with 'expert_params', 'non_expert_params', 'total_params'
    """
    attrs = get_model_attrs(model)
    expert_params = 0
    non_expert_params = 0
    
    for name, param in model.named_parameters():
        if attrs.experts in name:
            expert_params += param.numel()
        else:
            non_expert_params += param.numel()
            
    return {
        "expert_params": expert_params,
        "non_expert_params": non_expert_params,
        "total_params": expert_params + non_expert_params,
        "expert_ratio": expert_params / (expert_params + non_expert_params),
    }


def print_model_info(model: nn.Module) -> None:
    """Print summary information about a MoE model."""
    attrs = get_model_attrs(model)
    moe_layers = get_moe_layers(model)
    param_info = count_expert_parameters(model)
    
    print(f"\n{'='*60}")
    print(f"Model: {model.__class__.__name__}")
    print(f"{'='*60}")
    print(f"Number of MoE layers: {len(moe_layers)}")
    print(f"Experts per layer: {get_num_experts(model)}")
    print(f"Active experts per token: {get_num_experts_per_tok(model)}")
    print(f"\nParameter counts:")
    print(f"  Expert parameters: {param_info['expert_params']:,}")
    print(f"  Non-expert parameters: {param_info['non_expert_params']:,}")
    print(f"  Total parameters: {param_info['total_params']:,}")
    print(f"  Expert ratio: {param_info['expert_ratio']:.2%}")
    print(f"{'='*60}\n")
