"""
REAP-GPT-OSS: Router-weighted Expert Activation Pruning for GPT-OSS models.

This package adapts the REAP expert pruning method from Cerebras Research
for use with OpenAI's GPT-OSS mixture-of-experts models.

Example usage:
    from reap_gptoss import REAPPruner, PruningConfig
    from reap_gptoss.model_util import load_gptoss_model
    from reap_gptoss.data import create_gptoss_calibration_loader
    
    # Load model and data
    model = load_gptoss_model("openai/gpt-oss-20b")
    dataloader = create_gptoss_calibration_loader("openai/gpt-oss-20b")
    
    # Configure and run pruning
    config = PruningConfig(compression_ratio=0.5)
    pruner = REAPPruner(model, config)
    
    saliency = pruner.compute_saliency(dataloader)
    result = pruner.select_experts_to_prune(saliency)
    pruned_model = pruner.prune_experts(result)
    
    # Save
    pruned_model.save_pretrained("./pruned_gptoss")
"""

__version__ = "0.1.0"

from .model_util import (
    ModelAttributes,
    MODEL_ATTRS,
    get_model_attrs,
    get_moe_layers,
    get_num_experts,
    get_num_experts_per_tok,
    get_experts_module,
    get_router,
    load_gptoss_model,
    print_model_info,
)

from .observer import (
    ExpertObservation,
    ActivationStats,
    MoEObserver,
    ExpertOutputObserver,
)

from .prune import (
    PruningConfig,
    PruningResult,
    REAPPruner,
    compute_frequency_scores,
    compute_random_scores,
)

from .data import (
    DataConfig,
    CalibrationDataset,
    CodeCalibrationDataset,
    create_calibration_dataloader,
    create_gptoss_calibration_loader,
)

__all__ = [
    # Version
    "__version__",
    # Model utilities
    "ModelAttributes",
    "MODEL_ATTRS",
    "get_model_attrs",
    "get_moe_layers",
    "get_num_experts",
    "get_num_experts_per_tok",
    "get_experts_module",
    "get_router",
    "load_gptoss_model",
    "print_model_info",
    # Observer
    "ExpertObservation",
    "ActivationStats",
    "MoEObserver",
    "ExpertOutputObserver",
    # Pruning
    "PruningConfig",
    "PruningResult",
    "REAPPruner",
    "compute_frequency_scores",
    "compute_random_scores",
    # Data
    "DataConfig",
    "CalibrationDataset",
    "CodeCalibrationDataset",
    "create_calibration_dataloader",
    "create_gptoss_calibration_loader",
]
