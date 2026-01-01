"""
LLM-Sieve: Task-Specific Pruning for Large Language Models

Implementation of "How Many Parameters Does Your Task Really Need?"
by Waleed Reda, Abhinav Jangda, Krishna Chintalapudi (Microsoft Research, 2025)

Key features:
- Output-aligned non-orthogonal projections for better low-rank approximation
- Uniform pruning with binary search
- Adaptive pruning with genetic algorithm
- Easy integration with HuggingFace transformers

Quick start:
    from llm_sieve import TransformerPruner, LLMSieveConfig
    
    pruner = TransformerPruner(model, LLMSieveConfig())
    factors = pruner.uniform_prune(data, eval_fn, baseline)
    pruner.apply_pruning()
"""

from .core import (
    PruningConfig,
    LowRankAdapter,
    MatrixPruner,
    PrunedLinear,
    compute_pruning_factor_for_rank,
    compute_rank_for_pruning_factor,
)

from .layer_discovery import (
    LayerInfo,
    LayerDiscovery,
    diagnose_model,
)

from .pruner import (
    LLMSieveConfig,
    TransformerPruner,
    ActivationCollector,
)

from .genetic import (
    GeneticOptimizer,
    AdaptivePruningSearch,
)

__version__ = "1.0.0"
__all__ = [
    # Core
    "PruningConfig",
    "LowRankAdapter", 
    "MatrixPruner",
    "PrunedLinear",
    "compute_pruning_factor_for_rank",
    "compute_rank_for_pruning_factor",
    # Discovery
    "LayerInfo",
    "LayerDiscovery",
    "diagnose_model",
    # Pruner
    "LLMSieveConfig",
    "TransformerPruner",
    "ActivationCollector",
    # Genetic
    "GeneticOptimizer",
    "AdaptivePruningSearch",
]
