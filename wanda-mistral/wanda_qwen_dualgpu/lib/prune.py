"""
Wanda Pruning for Qwen3.5 (Hybrid GatedDeltaNet + Attention)
=============================================================

Adapted from the original Wanda implementation (Sun et al., 2023).

Key adaptation: Qwen3.5 uses a hybrid architecture where each group of 4
layers consists of 3 GatedDeltaNet (linear attention) layers followed by
1 standard GQA attention layer. Both layer types contain Linear modules
that can be pruned with Wanda.

GatedDeltaNet layers have:
    - linear_attn.in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, out_proj
    - Also has Conv1d (skipped for pruning — not a Linear layer)
MLP layers (all decoder layers):
    - mlp.gate_proj, mlp.up_proj, mlp.down_proj
Standard Attention layers have:
    - self_attn.q_proj, k_proj, v_proj, o_proj

References:
    - Wanda: https://arxiv.org/abs/2306.11695
    - Qwen3.5: https://qwen.ai/blog?id=qwen3.5
"""

import torch
import torch.nn as nn
from tqdm import tqdm

from .layerwrapper import WrappedGPT


# ──────────────────────────────────────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────────────────────────────────────


def find_layers(module: nn.Module, layers=(nn.Linear,), name: str = ""):
    """
    Recursively find all layers of the specified types.

    Args:
        module: Root module to search
        layers: Tuple of layer types to find
        name: Current name prefix

    Returns:
        Dict mapping full layer name to module
    """
    if type(module) in layers:
        return {name: module}

    res = {}
    for child_name, child in module.named_children():
        full_name = f"{name}.{child_name}" if name else child_name
        res.update(find_layers(child, layers, full_name))
    return res


def check_sparsity(model) -> float:
    """
    Calculate the overall sparsity of the model's linear layers.

    Returns:
        Sparsity ratio (0-1)
    """
    total_zeros = 0
    total_params = 0

    layers = model.model.layers

    for layer_idx, layer in enumerate(layers):
        subset = find_layers(layer)
        for name, module in subset.items():
            W = module.weight.data
            total_zeros += torch.sum(W == 0).item()
            total_params += W.numel()

    sparsity = total_zeros / total_params if total_params > 0 else 0
    print(f"Model sparsity: {sparsity:.4f} ({total_zeros}/{total_params})")

    return sparsity


def get_layer_type(layer) -> str:
    """
    Determine whether a Qwen3.5 decoder layer uses GatedDeltaNet or Attention.

    Returns:
        "gated_deltanet" or "attention"
    """
    if hasattr(layer, "linear_attn"):
        return "gated_deltanet"
    elif hasattr(layer, "self_attn"):
        return "attention"
    else:
        return "unknown"


# ──────────────────────────────────────────────────────────────────────────────
# Calibration input preparation
# ──────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def prepare_calibration_input(model, dataloader, device, nsamples=128):
    """
    Run calibration samples through the embedding layer to get inputs
    for the first decoder layer.

    Args:
        model: The Qwen3.5 model
        dataloader: List of input_ids tensors
        device: Device to use
        nsamples: Number of samples to use

    Returns:
        inps: Tensor of shape (nsamples, seqlen, hidden_size)
        attention_mask: Attention mask tensor
        position_ids: Position IDs tensor
        position_embeddings: Tuple of (cos, sin) rotary embeddings
    """
    layers = model.model.layers

    # Use a hook to capture the input to the first decoder layer
    dtype = next(iter(model.parameters())).dtype
    inps = []
    attention_masks = []
    position_ids_list = []
    position_embeddings_list = []

    class CatchInputs(nn.Module):
        """Temporary wrapper to catch inputs to first layer."""

        def __init__(self, module):
            super().__init__()
            self.module = module
            self.captured = False

        def __getattr__(self, name):
            # Proxy attribute access to the wrapped module so that
            # Qwen3.5's forward loop can access .layer_type, etc.
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

        def forward(self, *args, **kwargs):
            # args[0] is hidden_states
            hidden = args[0] if args else kwargs.get("hidden_states")
            inps.append(hidden.detach().cpu())

            attn_mask = kwargs.get("attention_mask", None)
            pos_ids = kwargs.get("position_ids", None)
            pos_embeds = kwargs.get("position_embeddings", None)

            if attn_mask is not None:
                attention_masks.append(attn_mask.detach().cpu())
            if pos_ids is not None:
                position_ids_list.append(pos_ids.detach().cpu())
            if pos_embeds is not None:
                position_embeddings_list.append(
                    tuple(t.detach().cpu() for t in pos_embeds)
                )

            self.captured = True
            raise ValueError("Early exit after capturing input")

    # Temporarily wrap the first layer
    layers[0] = CatchInputs(layers[0])

    for batch_idx, inp in enumerate(dataloader):
        if batch_idx >= nsamples:
            break
        try:
            inp = inp.to(device)
            model(inp)
        except ValueError:
            pass  # Expected early exit

    # Restore original layer
    layers[0] = layers[0].module

    inps = torch.cat(inps, dim=0)  # (nsamples, seqlen, hidden)

    # Build default attention mask and position ids if not captured
    if attention_masks:
        attention_mask = attention_masks[0]  # Same shape for all
    else:
        attention_mask = None

    if position_ids_list:
        position_ids = position_ids_list[0]
    else:
        seqlen = inps.shape[1]
        position_ids = torch.arange(seqlen, dtype=torch.long).unsqueeze(0)

    # Rotary embeddings — same for all samples (depends only on seq length)
    if position_embeddings_list:
        position_embeddings = position_embeddings_list[0]
    else:
        position_embeddings = None

    return inps, attention_mask, position_ids, position_embeddings


# ──────────────────────────────────────────────────────────────────────────────
# Pruning methods
# ──────────────────────────────────────────────────────────────────────────────


def prune_wanda(
    model,
    tokenizer,
    dataloader,
    device,
    sparsity_ratio: float = 0.5,
    sparsity_type: str = "unstructured",
    prune_n: int = 0,
    prune_m: int = 0,
    exclude_layers: list = None,
):
    """
    Prune the model using Wanda (Weights AND Activations).

    For each linear layer, computes importance as |W_ij| * ||X_j||_2
    and prunes the least important weights.

    Args:
        model: Qwen3.5 text model (Qwen3_5TextForCausalLM / Qwen3_5ForCausalLM)
        tokenizer: Tokenizer
        dataloader: List of calibration input_ids tensors
        device: Device to run on
        sparsity_ratio: Target sparsity (0-1), e.g. 0.5 for 50%
        sparsity_type: "unstructured", "2:4", or "4:8"
        prune_n: N for N:M structured sparsity (overrides sparsity_type)
        prune_m: M for N:M structured sparsity
        exclude_layers: List of layer name patterns to exclude from pruning
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    # Parse structured sparsity
    if sparsity_type == "2:4":
        prune_n, prune_m = 2, 4
    elif sparsity_type == "4:8":
        prune_n, prune_m = 4, 8

    if exclude_layers is None:
        exclude_layers = []

    print(f"Wanda pruning: sparsity={sparsity_ratio}, type={sparsity_type}")
    print(f"  N:M = {prune_n}:{prune_m}" if prune_n > 0 else "  Unstructured")

    # ── Step 1: Prepare calibration inputs ────────────────────────────────
    print("\nPreparing calibration inputs...")
    inps, attention_mask, position_ids, position_embeddings = prepare_calibration_input(
        model, dataloader, device, nsamples=len(dataloader)
    )
    print(f"  Calibration shape: {inps.shape}")

    layers = model.model.layers
    n_layers = len(layers)

    # ── Step 2: Process each layer ────────────────────────────────────────
    for layer_idx in tqdm(range(n_layers), desc="Pruning layers"):
        layer = layers[layer_idx]
        layer_type = get_layer_type(layer)

        # Find all Linear modules in this decoder layer
        subset = find_layers(layer)

        # Filter out excluded layers
        subset = {
            name: mod
            for name, mod in subset.items()
            if not any(excl in name for excl in exclude_layers)
        }

        if not subset:
            continue

        # Create WrappedGPT instances for activation capture
        wrapped_layers = {}
        for name, module in subset.items():
            wrapped_layers[name] = WrappedGPT(module, layer_idx, name)

        # Register forward hooks
        handles = []

        def make_hook(name):
            def hook(module, inp, out):
                wrapped_layers[name].add_batch(inp[0], out)
            return hook

        for name, module in subset.items():
            handles.append(module.register_forward_hook(make_hook(name)))

        # Run calibration samples through this layer
        outs = []
        for j in range(inps.shape[0]):
            inp_j = inps[j : j + 1].to(device)

            kwargs = {}
            if attention_mask is not None:
                kwargs["attention_mask"] = attention_mask.to(device)
            if position_ids is not None:
                kwargs["position_ids"] = position_ids.to(device)
            if position_embeddings is not None:
                kwargs["position_embeddings"] = tuple(
                    t.to(device) for t in position_embeddings
                )

            with torch.no_grad():
                out = layer(inp_j, **kwargs)
                # out is a tuple: (hidden_states, ...)
                if isinstance(out, tuple):
                    outs.append(out[0].detach().cpu())
                else:
                    outs.append(out.detach().cpu())

        # Remove hooks
        for h in handles:
            h.remove()

        # ── Step 3: Prune each linear layer ───────────────────────────────
        for name, module in subset.items():
            W = module.weight.data.float()
            importance = wrapped_layers[name].get_importance_scores()

            if prune_n > 0:
                # N:M structured sparsity
                mask = _create_nm_mask(importance, prune_n, prune_m)
            else:
                # Unstructured sparsity
                mask = _create_unstructured_mask(importance, sparsity_ratio)

            # Apply mask
            module.weight.data[~mask] = 0.0

        # Update inputs for the next layer
        inps = torch.cat(outs, dim=0)

    model.config.use_cache = use_cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nPruning complete.")
    check_sparsity(model)


def prune_magnitude(
    model,
    sparsity_ratio: float = 0.5,
    sparsity_type: str = "unstructured",
    prune_n: int = 0,
    prune_m: int = 0,
):
    """
    Baseline: Prune by weight magnitude only (no activations).

    Args:
        model: Qwen3.5 text model
        sparsity_ratio: Target sparsity
        sparsity_type: "unstructured", "2:4", or "4:8"
        prune_n, prune_m: For N:M sparsity
    """
    if sparsity_type == "2:4":
        prune_n, prune_m = 2, 4
    elif sparsity_type == "4:8":
        prune_n, prune_m = 4, 8

    layers = model.model.layers

    for layer_idx in tqdm(range(len(layers)), desc="Magnitude pruning"):
        layer = layers[layer_idx]
        subset = find_layers(layer)

        for name, module in subset.items():
            W = module.weight.data.float()
            importance = torch.abs(W)

            if prune_n > 0:
                mask = _create_nm_mask(importance, prune_n, prune_m)
            else:
                mask = _create_unstructured_mask(importance, sparsity_ratio)

            module.weight.data[~mask] = 0.0

    check_sparsity(model)


def prune_sparsegpt(
    model,
    tokenizer,
    dataloader,
    device,
    sparsity_ratio: float = 0.5,
    sparsity_type: str = "unstructured",
    prune_n: int = 0,
    prune_m: int = 0,
    blocksize: int = 128,
    percdamp: float = 0.01,
):
    """
    Prune using SparseGPT (Frantar & Alistarh, 2023).

    Uses approximate Hessian information to compensate for pruning error.
    More accurate but slower than Wanda.

    Args:
        model: Qwen3.5 text model
        tokenizer: Tokenizer
        dataloader: Calibration data
        device: Device
        sparsity_ratio: Target sparsity
        sparsity_type: "unstructured", "2:4", or "4:8"
        prune_n, prune_m: For N:M sparsity
        blocksize: Column block size for Hessian computation
        percdamp: Damping factor for Hessian
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    if sparsity_type == "2:4":
        prune_n, prune_m = 2, 4
    elif sparsity_type == "4:8":
        prune_n, prune_m = 4, 8

    print(f"SparseGPT pruning: sparsity={sparsity_ratio}, type={sparsity_type}")

    # Prepare calibration inputs
    inps, attention_mask, position_ids, position_embeddings = prepare_calibration_input(
        model, dataloader, device, nsamples=len(dataloader)
    )

    layers = model.model.layers

    for layer_idx in tqdm(range(len(layers)), desc="SparseGPT pruning"):
        layer = layers[layer_idx]
        subset = find_layers(layer)

        # Collect inputs for Hessian estimation
        layer_inputs = []
        handles = []

        def make_input_hook(storage):
            def hook(module, inp, out):
                storage.append(inp[0].detach())
            return hook

        # We need per-sublayer inputs, so wrap each Linear
        sublayer_inputs = {name: [] for name in subset}

        for name, module in subset.items():
            handles.append(
                module.register_forward_hook(make_input_hook(sublayer_inputs[name]))
            )

        # Forward pass through this layer
        outs = []
        for j in range(inps.shape[0]):
            inp_j = inps[j : j + 1].to(device)
            kwargs = {}
            if attention_mask is not None:
                kwargs["attention_mask"] = attention_mask.to(device)
            if position_ids is not None:
                kwargs["position_ids"] = position_ids.to(device)
            if position_embeddings is not None:
                kwargs["position_embeddings"] = tuple(
                    t.to(device) for t in position_embeddings
                )

            with torch.no_grad():
                out = layer(inp_j, **kwargs)
                if isinstance(out, tuple):
                    outs.append(out[0].detach().cpu())
                else:
                    outs.append(out.detach().cpu())

        for h in handles:
            h.remove()

        # Prune each sublayer using SparseGPT
        for name, module in subset.items():
            inputs_list = sublayer_inputs[name]
            _sparsegpt_prune_layer(
                module, inputs_list, sparsity_ratio,
                prune_n, prune_m, blocksize, percdamp
            )

        inps = torch.cat(outs, dim=0)

    model.config.use_cache = use_cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nSparseGPT pruning complete.")
    check_sparsity(model)


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────


def _create_unstructured_mask(importance: torch.Tensor, sparsity: float) -> torch.Tensor:
    """
    Create a binary mask for unstructured pruning.

    Keeps the top (1 - sparsity) fraction of weights by importance.

    Returns:
        Boolean mask (True = keep, False = prune)
    """
    threshold = torch.sort(importance.flatten())[0][
        int(importance.numel() * sparsity)
    ]
    mask = importance >= threshold
    return mask


def _create_nm_mask(importance: torch.Tensor, n: int, m: int) -> torch.Tensor:
    """
    Create a mask for N:M structured sparsity.

    In every contiguous group of M weights (along the output dim),
    keeps the top (M - N) weights and prunes N.

    Returns:
        Boolean mask (True = keep, False = prune)
    """
    mask = torch.zeros_like(importance, dtype=torch.bool)
    for i in range(0, importance.shape[1], m):
        block = importance[:, i : i + m]
        if block.shape[1] < m:
            # Handle edge case: last block smaller than m
            keep_k = max(1, int(block.shape[1] * (m - n) / m))
            _, topk_idx = torch.topk(block, keep_k, dim=1)
        else:
            _, topk_idx = torch.topk(block, m - n, dim=1)
        mask[:, i : i + m].scatter_(1, topk_idx, True)
    return mask


def _sparsegpt_prune_layer(
    layer: nn.Linear,
    inputs: list,
    sparsity_ratio: float,
    prune_n: int,
    prune_m: int,
    blocksize: int = 128,
    percdamp: float = 0.01,
):
    """
    Apply SparseGPT pruning to a single linear layer.

    Uses the Hessian (H = X^T X) to perform optimal weight rounding
    after pruning, compensating for the error introduced.
    """
    W = layer.weight.data.clone().float()

    # Compute Hessian approximation: H = (1/N) * sum(X^T X)
    H = torch.zeros((W.shape[1], W.shape[1]), device=W.device, dtype=torch.float32)

    for inp in inputs:
        inp = inp.float()
        if len(inp.shape) == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        H += inp.T @ inp

    H /= len(inputs)

    # Add damping for numerical stability
    damp = percdamp * torch.mean(torch.diag(H))
    H += damp * torch.eye(H.shape[0], device=H.device)

    # Compute inverse Hessian
    try:
        Hinv = torch.linalg.inv(H)
    except Exception:
        Hinv = torch.linalg.pinv(H)

    # Prune column by column in blocks
    for i1 in range(0, W.shape[1], blocksize):
        i2 = min(i1 + blocksize, W.shape[1])

        W_block = W[:, i1:i2].clone()
        Hinv_block = Hinv[i1:i2, i1:i2]

        # Importance for this block
        importance = W_block ** 2 / torch.diag(Hinv_block).unsqueeze(0)

        # Create mask
        if prune_n > 0:
            mask = _create_nm_mask(importance, prune_n, prune_m)
        else:
            mask = _create_unstructured_mask(importance, sparsity_ratio)

        # Compute error from pruned weights
        err = (W_block * (~mask).float()) @ Hinv_block

        # Apply pruning
        W[:, i1:i2] = W_block * mask.float()

        # Propagate error to remaining columns
        if i2 < W.shape[1]:
            W[:, i2:] -= err @ Hinv[i1:i2, i2:]

    layer.weight.data = W.to(layer.weight.dtype)
