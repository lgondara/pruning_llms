"""
Structured Pruning for Qwen3.5 (Physically Smaller Models)
==========================================================

Unlike unstructured/semi-structured pruning (which zeros individual weights
but keeps matrix dimensions the same), structured pruning removes entire
neurons or attention heads, physically shrinking weight matrices.

Example — MLP neuron pruning at 50%:
    gate_proj: (9216, 2560) → (4608, 2560)   # remove rows
    up_proj:   (9216, 2560) → (4608, 2560)   # remove same rows (SwiGLU pair)
    down_proj: (2560, 9216) → (2560, 4608)   # remove corresponding columns

The result is a genuinely smaller model with real speedups on any hardware.

Importance is computed Wanda-style: for each neuron, we combine weight
magnitude with activation norms from calibration data.

References:
    - Wanda: https://arxiv.org/abs/2306.11695
    - LLM-Pruner: https://arxiv.org/abs/2305.11627
    - Minitron: https://arxiv.org/abs/2407.14679
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Optional

from .prune import (
    find_layers,
    get_layer_type,
    prepare_calibration_input,
    check_sparsity,
)


# ──────────────────────────────────────────────────────────────────────────────
# Activation capture for structured importance
# ──────────────────────────────────────────────────────────────────────────────


class ActivationCapture:
    """
    Captures intermediate activations for computing per-neuron importance.

    For MLP: captures the output of gate_proj * up_proj (the intermediate
    activation) to measure how "active" each neuron is across calibration data.
    """

    def __init__(self):
        self.activations = []

    def hook_fn(self, module, inp, out):
        # out shape: (batch, seq, intermediate_size)
        self.activations.append(out.detach().cpu())

    def get_mean_activation_norm(self) -> torch.Tensor:
        """
        Compute mean L2 norm per neuron across all captured activations.

        Returns:
            Tensor of shape (intermediate_size,) with per-neuron importance
        """
        # Concatenate all: (total_tokens, intermediate_size)
        all_acts = torch.cat(
            [a.reshape(-1, a.shape[-1]) for a in self.activations], dim=0
        ).float()

        # Per-neuron L2 norm averaged across tokens
        importance = torch.sqrt(torch.mean(all_acts ** 2, dim=0))
        return importance

    def clear(self):
        self.activations = []


# ──────────────────────────────────────────────────────────────────────────────
# MLP Structured Pruning
# ──────────────────────────────────────────────────────────────────────────────


def compute_mlp_neuron_importance(
    layer: nn.Module,
    inps: torch.Tensor,
    attention_mask,
    position_ids,
    position_embeddings,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute importance score for each MLP intermediate neuron.

    Importance = weight_magnitude × activation_norm (Wanda-style)

    For SwiGLU: intermediate = SiLU(gate_proj(x)) * up_proj(x)
    We capture the gate_proj output to get per-neuron activation norms,
    then combine with weight norms.

    Args:
        layer: A Qwen3_5DecoderLayer
        inps: Calibration inputs (nsamples, seqlen, hidden)
        attention_mask, position_ids, position_embeddings: Forward kwargs
        device: Device

    Returns:
        importance: Tensor of shape (intermediate_size,)
    """
    mlp = layer.mlp

    # Capture gate_proj activations (before SiLU, but after projection)
    capture = ActivationCapture()
    handle = mlp.gate_proj.register_forward_hook(capture.hook_fn)

    # Run calibration through this layer
    for j in range(inps.shape[0]):
        inp_j = inps[j:j+1].to(device)
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
            layer(inp_j, **kwargs)

    handle.remove()

    # Activation-based importance: RMS activation per neuron
    act_importance = capture.get_mean_activation_norm()

    # Weight-based importance: L2 norm of each neuron's weights
    # Combine gate_proj and up_proj (SwiGLU pair) and down_proj
    gate_w = mlp.gate_proj.weight.data.float().cpu()  # (intermediate, hidden)
    up_w = mlp.up_proj.weight.data.float().cpu()       # (intermediate, hidden)
    down_w = mlp.down_proj.weight.data.float().cpu()   # (hidden, intermediate)

    # Per-neuron weight norm: combine input-side and output-side
    gate_norm = torch.norm(gate_w, dim=1)   # (intermediate,)
    up_norm = torch.norm(up_w, dim=1)       # (intermediate,)
    down_norm = torch.norm(down_w, dim=0)   # (intermediate,)

    weight_importance = (gate_norm + up_norm + down_norm) / 3.0

    # Wanda-style: weight × activation
    importance = weight_importance * act_importance

    capture.clear()
    return importance


def prune_mlp_neurons(
    layer: nn.Module,
    keep_indices: torch.Tensor,
):
    """
    Physically remove MLP neurons by slicing weight matrices.

    For SwiGLU MLP:
        gate_proj: (intermediate, hidden) → keep rows
        up_proj:   (intermediate, hidden) → keep same rows
        down_proj: (hidden, intermediate) → keep corresponding columns

    Args:
        layer: Decoder layer containing the MLP
        keep_indices: 1D tensor of neuron indices to keep (sorted)
    """
    mlp = layer.mlp
    device = mlp.gate_proj.weight.device
    dtype = mlp.gate_proj.weight.dtype
    keep = keep_indices.to(device)

    old_intermediate = mlp.gate_proj.weight.shape[0]
    new_intermediate = len(keep_indices)
    hidden_size = mlp.gate_proj.weight.shape[1]

    # Slice gate_proj: keep selected rows
    new_gate = nn.Linear(hidden_size, new_intermediate, bias=False, device=device, dtype=dtype)
    new_gate.weight.data = mlp.gate_proj.weight.data[keep]

    # Slice up_proj: keep same rows
    new_up = nn.Linear(hidden_size, new_intermediate, bias=False, device=device, dtype=dtype)
    new_up.weight.data = mlp.up_proj.weight.data[keep]

    # Slice down_proj: keep corresponding columns
    new_down = nn.Linear(new_intermediate, hidden_size, bias=False, device=device, dtype=dtype)
    new_down.weight.data = mlp.down_proj.weight.data[:, keep]

    # Replace
    mlp.gate_proj = new_gate
    mlp.up_proj = new_up
    mlp.down_proj = new_down

    return old_intermediate, new_intermediate


# ──────────────────────────────────────────────────────────────────────────────
# Attention Head Structured Pruning
# ──────────────────────────────────────────────────────────────────────────────


def compute_attention_head_importance(
    layer: nn.Module,
    inps: torch.Tensor,
    attention_mask,
    position_ids,
    position_embeddings,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute importance for each attention head (standard attention layers only).

    For GQA in Qwen3.5-4B:
        q_proj: 32 heads × 256 dim = 8192
        k_proj: 4 heads × 256 dim = 1024
        v_proj: 4 heads × 256 dim = 1024
        o_proj: 32 × 256 → hidden

    We compute importance per Q-head group (each group of 8 Q-heads shares 1 KV-head).
    Pruning removes entire KV-head groups.

    Returns:
        importance: Tensor of shape (num_kv_heads,) — one score per KV-head group
    """
    attn = layer.self_attn

    # Capture attention output (before o_proj)
    capture = ActivationCapture()
    handle = attn.o_proj.register_forward_hook(capture.hook_fn)

    for j in range(min(inps.shape[0], 32)):  # Limit samples for speed
        inp_j = inps[j:j+1].to(device)
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
            layer(inp_j, **kwargs)

    handle.remove()

    # Weight-based importance per KV-head group
    q_weight = attn.q_proj.weight.data.float().cpu()  # (num_q_heads*head_dim, hidden)
    k_weight = attn.k_proj.weight.data.float().cpu()  # (num_kv_heads*head_dim, hidden)
    v_weight = attn.v_proj.weight.data.float().cpu()  # (num_kv_heads*head_dim, hidden)
    o_weight = attn.o_proj.weight.data.float().cpu()  # (hidden, num_q_heads*head_dim)

    num_q_heads = q_weight.shape[0] // 256  # head_dim = 256
    num_kv_heads = k_weight.shape[0] // 256
    heads_per_group = num_q_heads // num_kv_heads  # Q heads per KV head

    head_dim = 256
    importance = torch.zeros(num_kv_heads)

    for kv_idx in range(num_kv_heads):
        # KV head weights
        k_head = k_weight[kv_idx * head_dim : (kv_idx + 1) * head_dim]
        v_head = v_weight[kv_idx * head_dim : (kv_idx + 1) * head_dim]

        # Corresponding Q heads
        q_start = kv_idx * heads_per_group * head_dim
        q_end = (kv_idx + 1) * heads_per_group * head_dim
        q_heads = q_weight[q_start:q_end]

        # O_proj corresponding columns
        o_heads = o_weight[:, q_start:q_end]

        importance[kv_idx] = (
            torch.norm(q_heads) + torch.norm(k_head) +
            torch.norm(v_head) + torch.norm(o_heads)
        )

    capture.clear()
    return importance


def prune_attention_heads(
    layer: nn.Module,
    keep_kv_indices: torch.Tensor,
    head_dim: int = 256,
):
    """
    Physically remove attention heads by slicing weight matrices.

    Removes entire KV-head groups (each KV head + its associated Q heads).

    Args:
        layer: Decoder layer with standard attention
        keep_kv_indices: KV head indices to keep (sorted)
        head_dim: Dimension per head (256 for Qwen3.5-4B)
    """
    attn = layer.self_attn
    device = attn.q_proj.weight.device
    dtype = attn.q_proj.weight.dtype

    num_q_heads = attn.q_proj.weight.shape[0] // head_dim
    num_kv_heads = attn.k_proj.weight.shape[0] // head_dim
    heads_per_group = num_q_heads // num_kv_heads
    hidden_size = attn.q_proj.weight.shape[1]

    new_num_kv = len(keep_kv_indices)
    new_num_q = new_num_kv * heads_per_group

    # Build index masks
    q_indices = []
    for kv_idx in keep_kv_indices:
        start = kv_idx * heads_per_group * head_dim
        end = (kv_idx + 1) * heads_per_group * head_dim
        q_indices.extend(range(start, end))
    q_indices = torch.tensor(q_indices, device=device)

    kv_indices = []
    for kv_idx in keep_kv_indices:
        start = kv_idx * head_dim
        end = (kv_idx + 1) * head_dim
        kv_indices.extend(range(start, end))
    kv_indices = torch.tensor(kv_indices, device=device)

    # Slice Q
    new_q = nn.Linear(hidden_size, new_num_q * head_dim, bias=False, device=device, dtype=dtype)
    new_q.weight.data = attn.q_proj.weight.data[q_indices]

    # Slice K
    new_k = nn.Linear(hidden_size, new_num_kv * head_dim, bias=False, device=device, dtype=dtype)
    new_k.weight.data = attn.k_proj.weight.data[kv_indices]

    # Slice V
    new_v = nn.Linear(hidden_size, new_num_kv * head_dim, bias=False, device=device, dtype=dtype)
    new_v.weight.data = attn.v_proj.weight.data[kv_indices]

    # Slice O (columns correspond to Q heads)
    new_o = nn.Linear(new_num_q * head_dim, hidden_size, bias=False, device=device, dtype=dtype)
    new_o.weight.data = attn.o_proj.weight.data[:, q_indices]

    # Replace
    attn.q_proj = new_q
    attn.k_proj = new_k
    attn.v_proj = new_v
    attn.o_proj = new_o

    # Update config attributes on the attention module if they exist
    if hasattr(attn, "num_heads"):
        attn.num_heads = new_num_q
    if hasattr(attn, "num_key_value_heads"):
        attn.num_key_value_heads = new_num_kv
    if hasattr(attn, "num_key_value_groups"):
        attn.num_key_value_groups = heads_per_group

    return num_kv_heads, new_num_kv


# ──────────────────────────────────────────────────────────────────────────────
# GatedDeltaNet Structured Pruning
# ──────────────────────────────────────────────────────────────────────────────


def compute_deltanet_head_importance(
    layer: nn.Module,
    inps: torch.Tensor,
    attention_mask,
    position_ids,
    position_embeddings,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute importance for each GatedDeltaNet head.

    GatedDeltaNet in Qwen3.5-4B:
        in_proj_qkv: (8192, 2560) — 32 heads × 256 (QKV interleaved or concatenated)
        in_proj_z:   (4096, 2560) — 32 heads × 128
        out_proj:    (2560, 4096) — reverse
        in_proj_a:   (32, 2560) — one scalar per head
        in_proj_b:   (32, 2560) — one scalar per head

    We compute per-head importance from weight norms.

    Returns:
        importance: Tensor of shape (num_heads,)
    """
    lin_attn = layer.linear_attn

    # Weight norms per head
    qkv_w = lin_attn.in_proj_qkv.weight.data.float().cpu()  # (8192, 2560)
    z_w = lin_attn.in_proj_z.weight.data.float().cpu()       # (4096, 2560)
    out_w = lin_attn.out_proj.weight.data.float().cpu()      # (2560, 4096)

    num_heads = 32  # For 4B model
    qkv_head_dim = qkv_w.shape[0] // num_heads  # 256
    z_head_dim = z_w.shape[0] // num_heads       # 128

    importance = torch.zeros(num_heads)
    for h in range(num_heads):
        qkv_slice = qkv_w[h * qkv_head_dim : (h + 1) * qkv_head_dim]
        z_slice = z_w[h * z_head_dim : (h + 1) * z_head_dim]
        out_slice = out_w[:, h * z_head_dim : (h + 1) * z_head_dim]

        importance[h] = (
            torch.norm(qkv_slice) + torch.norm(z_slice) + torch.norm(out_slice)
        )

    return importance


def prune_deltanet_heads(
    layer: nn.Module,
    keep_indices: torch.Tensor,
    num_heads: int = 32,
):
    """
    Physically remove GatedDeltaNet heads.

    Args:
        layer: Decoder layer with GatedDeltaNet
        keep_indices: Head indices to keep (sorted)
        num_heads: Original number of heads
    """
    lin_attn = layer.linear_attn
    device = lin_attn.in_proj_qkv.weight.device
    dtype = lin_attn.in_proj_qkv.weight.dtype
    hidden_size = lin_attn.in_proj_qkv.weight.shape[1]

    qkv_head_dim = lin_attn.in_proj_qkv.weight.shape[0] // num_heads  # 256
    z_head_dim = lin_attn.in_proj_z.weight.shape[0] // num_heads       # 128

    new_num_heads = len(keep_indices)

    # Build index masks for QKV
    qkv_indices = []
    for h in keep_indices:
        start = h * qkv_head_dim
        end = (h + 1) * qkv_head_dim
        qkv_indices.extend(range(start, end))
    qkv_idx = torch.tensor(qkv_indices, device=device)

    # Build index masks for Z and out
    z_indices = []
    for h in keep_indices:
        start = h * z_head_dim
        end = (h + 1) * z_head_dim
        z_indices.extend(range(start, end))
    z_idx = torch.tensor(z_indices, device=device)

    # Slice in_proj_qkv
    new_qkv = nn.Linear(hidden_size, new_num_heads * qkv_head_dim, bias=False, device=device, dtype=dtype)
    new_qkv.weight.data = lin_attn.in_proj_qkv.weight.data[qkv_idx]

    # Slice in_proj_z
    new_z = nn.Linear(hidden_size, new_num_heads * z_head_dim, bias=False, device=device, dtype=dtype)
    new_z.weight.data = lin_attn.in_proj_z.weight.data[z_idx]

    # Slice out_proj (columns)
    new_out = nn.Linear(new_num_heads * z_head_dim, hidden_size, bias=False, device=device, dtype=dtype)
    new_out.weight.data = lin_attn.out_proj.weight.data[:, z_idx]

    # Slice in_proj_a and in_proj_b (one row per head)
    keep_t = keep_indices.to(device)
    new_a = nn.Linear(hidden_size, new_num_heads, bias=False, device=device, dtype=dtype)
    new_a.weight.data = lin_attn.in_proj_a.weight.data[keep_t]

    new_b = nn.Linear(hidden_size, new_num_heads, bias=False, device=device, dtype=dtype)
    new_b.weight.data = lin_attn.in_proj_b.weight.data[keep_t]

    # Replace
    lin_attn.in_proj_qkv = new_qkv
    lin_attn.in_proj_z = new_z
    lin_attn.out_proj = new_out
    lin_attn.in_proj_a = new_a
    lin_attn.in_proj_b = new_b

    # Handle conv1d — it operates on the QKV dimension
    old_conv = lin_attn.conv1d
    if old_conv is not None and hasattr(old_conv, 'weight'):
        new_conv_channels = new_num_heads * qkv_head_dim
        new_conv = nn.Conv1d(
            new_conv_channels, new_conv_channels,
            kernel_size=old_conv.kernel_size, stride=old_conv.stride,
            padding=old_conv.padding, groups=new_conv_channels,
            bias=old_conv.bias is not None, device=device, dtype=dtype,
        )
        new_conv.weight.data = old_conv.weight.data[qkv_idx]
        if old_conv.bias is not None:
            new_conv.bias.data = old_conv.bias.data[qkv_idx]
        lin_attn.conv1d = new_conv

    return num_heads, new_num_heads


# ──────────────────────────────────────────────────────────────────────────────
# Main structured pruning entry point
# ──────────────────────────────────────────────────────────────────────────────


def prune_structured(
    model,
    tokenizer,
    dataloader,
    device,
    mlp_ratio: float = 0.5,
    attn_ratio: float = 0.0,
    prune_mlp: bool = True,
    prune_attn: bool = False,
):
    """
    Structured pruning: physically remove neurons and heads.

    Args:
        model: Qwen3.5 text model
        tokenizer: Tokenizer
        dataloader: Calibration data
        device: Device
        mlp_ratio: Fraction of MLP neurons to REMOVE (0.5 = remove 50%)
        attn_ratio: Fraction of attention heads to REMOVE (0.25 = remove 25%)
        prune_mlp: Whether to prune MLP neurons
        prune_attn: Whether to prune attention/deltanet heads
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print(f"Structured pruning:")
    if prune_mlp:
        print(f"  MLP neuron removal: {mlp_ratio:.0%}")
    if prune_attn:
        print(f"  Attention head removal: {attn_ratio:.0%}")

    # Prepare calibration inputs
    print("\nPreparing calibration inputs...")
    inps, attention_mask, position_ids, position_embeddings = prepare_calibration_input(
        model, dataloader, device, nsamples=len(dataloader)
    )
    print(f"  Calibration shape: {inps.shape}")

    layers = model.model.layers
    n_layers = len(layers)

    total_neurons_removed = 0
    total_neurons_original = 0
    total_heads_removed = 0
    total_heads_original = 0

    for layer_idx in tqdm(range(n_layers), desc="Structured pruning"):
        layer = layers[layer_idx]
        layer_type = get_layer_type(layer)

        # ── MLP pruning (all layers have MLPs) ────────────────────────────
        if prune_mlp:
            importance = compute_mlp_neuron_importance(
                layer, inps, attention_mask, position_ids,
                position_embeddings, device,
            )

            n_total = importance.shape[0]
            n_keep = int(n_total * (1 - mlp_ratio))
            # Round to multiple of 128 for efficiency
            n_keep = max(128, (n_keep // 128) * 128)

            _, top_indices = torch.topk(importance, n_keep)
            keep_indices = top_indices.sort()[0]

            old_size, new_size = prune_mlp_neurons(layer, keep_indices)
            total_neurons_removed += old_size - new_size
            total_neurons_original += old_size

        # ── Attention head pruning ────────────────────────────────────────
        if prune_attn and attn_ratio > 0:
            if layer_type == "attention":
                importance = compute_attention_head_importance(
                    layer, inps, attention_mask, position_ids,
                    position_embeddings, device,
                )
                n_heads = importance.shape[0]
                n_keep = max(1, int(n_heads * (1 - attn_ratio)))

                _, top_indices = torch.topk(importance, n_keep)
                keep_indices = top_indices.sort()[0]

                old_h, new_h = prune_attention_heads(layer, keep_indices)
                total_heads_removed += old_h - new_h
                total_heads_original += old_h

            elif layer_type == "gated_deltanet":
                importance = compute_deltanet_head_importance(
                    layer, inps, attention_mask, position_ids,
                    position_embeddings, device,
                )
                n_heads = importance.shape[0]
                n_keep = max(1, int(n_heads * (1 - attn_ratio)))

                _, top_indices = torch.topk(importance, n_keep)
                keep_indices = top_indices.sort()[0]

                old_h, new_h = prune_deltanet_heads(layer, keep_indices)
                total_heads_removed += old_h - new_h
                total_heads_original += old_h

        # Update inputs for next layer
        outs = []
        for j in range(inps.shape[0]):
            inp_j = inps[j:j+1].to(device)
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
        inps = torch.cat(outs, dim=0)

    # ── Summary ───────────────────────────────────────────────────────────
    model.config.use_cache = use_cache

    print("\n--- Structured Pruning Summary ---")
    if prune_mlp:
        print(f"  MLP neurons: {total_neurons_original:,} → {total_neurons_original - total_neurons_removed:,}")
        print(f"  MLP reduction: {total_neurons_removed / total_neurons_original:.1%}")
    if prune_attn:
        print(f"  Attn heads: {total_heads_original:,} → {total_heads_original - total_heads_removed:,}")
        print(f"  Head reduction: {total_heads_removed / total_heads_original:.1%}")

    # Count final parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Final model params: {total_params:,} ({total_params / 1e9:.2f}B)")

    # Update model config to reflect new dimensions
    if prune_mlp:
        new_intermediate = int(model.model.layers[0].mlp.gate_proj.weight.shape[0])
        if hasattr(model.config, "intermediate_size"):
            model.config.intermediate_size = new_intermediate
            print(f"  New intermediate_size: {new_intermediate}")
