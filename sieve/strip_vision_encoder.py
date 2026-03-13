import argparse
import json
import os
import shutil
from pathlib import Path
from collections import OrderedDict

try:
    from safetensors import safe_open
    from safetensors.torch import save_file
except ImportError:
    raise ImportError("Please install safetensors: pip install safetensors")


# ---------------------------------------------------------------------------
# Prefixes that belong to the vision encoder / multimodal projector.
# Any weight key starting with these prefixes will be REMOVED.
# Everything else (the language model) is kept.
# ---------------------------------------------------------------------------
VISION_PREFIXES = (
    "model.visual.",         # Vision Transformer (ViT) encoder + merger/projector
)

# Prefixes that are definitely part of the language model (kept).
# Used only for sanity-check logging, not for filtering.
LM_PREFIXES = (
    "model.language_model.",  # The main language model trunk (426 keys)
    "mtp.",                   # Multi-token prediction heads (14 keys, for speculative decoding)
)


def is_vision_weight(key: str) -> bool:
    """Return True if this weight key belongs to the vision encoder."""
    return any(key.startswith(p) for p in VISION_PREFIXES)


def remap_key(key: str) -> str:
    """
    Remap weight key from composite VLM naming to standalone text model naming.

    In the composite Qwen3_5ForConditionalGeneration:
        model.language_model.layers.0.mlp.down_proj.weight
        model.language_model.embed_tokens.weight
        model.language_model.norm.weight

    In the standalone Qwen3_5TextForCausalLM:
        model.layers.0.mlp.down_proj.weight
        model.embed_tokens.weight
        model.norm.weight

    MTP keys (mtp.*) are kept as-is.
    """
    if key.startswith("model.language_model."):
        return "model." + key[len("model.language_model."):]
    return key


def extract_text_config(config: dict) -> dict:
    """
    Build a standalone text-only config from the composite Qwen3.5 config.

    The original config.json has:
      - model_type: "qwen3_5"
      - text_config: { ... }       # Qwen3_5TextConfig
      - vision_config: { ... }     # Qwen3_5VisionConfig
      - image_token_id, video_token_id, etc.

    We promote text_config to be the root config and set model_type
    to "qwen3_5_text" so transformers loads it as a pure text model
    (Qwen3_5TextForCausalLM).
    """
    text_cfg = config.get("text_config", {})

    if not text_cfg:
        raise ValueError(
            "No 'text_config' found in config.json. "
            "Is this actually a Qwen3.5 composite model?"
        )

    # Start from text_config as the base
    new_config = dict(text_cfg)

    # Debug: show critical dimensions
    print(f"  text_config.hidden_size:      {text_cfg.get('hidden_size', 'NOT SET')}")
    print(f"  text_config.num_hidden_layers: {text_cfg.get('num_hidden_layers', 'NOT SET')}")
    print(f"  text_config.model_type:       {text_cfg.get('model_type', 'NOT SET')}")

    # Ensure model_type is set for standalone text model
    new_config["model_type"] = text_cfg.get("model_type", "qwen3_5_text")

    # Carry over shared fields from the parent config if not in text_config
    for key in [
        "torch_dtype",
        "transformers_version",
        "tie_word_embeddings",
    ]:
        if key in config and key not in new_config:
            new_config[key] = config[key]

    # Architectures: point to text-only causal LM class
    new_config["architectures"] = ["Qwen3_5TextForCausalLM"]

    # Remove any vision-related fields that may have leaked in
    for remove_key in [
        "vision_config", "image_token_id", "video_token_id",
        "vision_start_token_id", "vision_end_token_id",
        "spatial_merge_size", "vision_token_id",
    ]:
        new_config.pop(remove_key, None)

    return new_config


def process_safetensors_index(index_path: str) -> tuple[dict, list[str], dict]:
    """
    Parse the safetensors index file and separate vision vs text weights.

    Returns:
        new_index: Updated index with only text weights
        files_needed: List of safetensors shard files that contain text weights
        stats: Dictionary with counts/sizes for logging
    """
    with open(index_path, "r") as f:
        index = json.load(f)

    weight_map = index.get("weight_map", {})

    text_weight_map = {}
    vision_keys = []
    text_keys = []

    for key, shard_file in weight_map.items():
        if is_vision_weight(key):
            vision_keys.append(key)
        else:
            text_weight_map[key] = shard_file
            text_keys.append(key)

    # Determine which shard files are still needed
    files_needed = sorted(set(text_weight_map.values()))

    new_index = {
        "metadata": index.get("metadata", {}),
        "weight_map": text_weight_map,
    }

    stats = {
        "total_keys": len(weight_map),
        "vision_keys": len(vision_keys),
        "text_keys": len(text_keys),
        "original_shards": len(set(weight_map.values())),
        "needed_shards": len(files_needed),
    }

    return new_index, files_needed, stats


def strip_vision_from_shards(
    input_dir: str,
    output_dir: str,
    files_needed: list[str],
    text_weight_map: dict[str, str],
):
    """
    Re-save safetensors shard files with only text weights, remapping keys
    from composite VLM naming (model.language_model.*) to standalone text
    model naming (model.*).
    """
    print("\nLoading text weights from shard files...")

    all_text_tensors = {}
    text_keys_set = set(text_weight_map.keys())
    remapped_count = 0

    for shard_file in files_needed:
        shard_path = os.path.join(input_dir, shard_file)
        print(f"  Reading: {shard_file}")

        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in text_keys_set:
                    new_key = remap_key(key)
                    if new_key != key:
                        remapped_count += 1
                    all_text_tensors[new_key] = f.get_tensor(key)

    print(f"  Keys remapped (model.language_model.* -> model.*): {remapped_count}")

    # Save as a single file
    output_file = "model.safetensors"
    output_path = os.path.join(output_dir, output_file)

    print(f"\nSaving {len(all_text_tensors)} text-only tensors to {output_file}...")
    save_file(all_text_tensors, output_path)

    # Compute total size
    total_bytes = os.path.getsize(output_path)
    total_gb = total_bytes / (1024 ** 3)
    print(f"  Saved: {total_gb:.2f} GB")

    # Create a new index pointing to the single file (with remapped keys)
    new_weight_map = {key: output_file for key in all_text_tensors.keys()}
    new_index = {
        "metadata": {"total_size": total_bytes},
        "weight_map": new_weight_map,
    }

    return new_index, total_gb


def main():
    parser = argparse.ArgumentParser(
        description="Strip vision encoder from Qwen3.5-4B to create a text-only model"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input_dir",
        type=str,
        help="Path to local Qwen3.5-4B model directory",
    )
    group.add_argument(
        "--model_id",
        type=str,
        help="HuggingFace model ID (e.g., Qwen/Qwen3.5-4B). Will be downloaded first.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for the text-only model",
    )
    args = parser.parse_args()

    # If model_id provided, download first
    if args.model_id:
        from huggingface_hub import snapshot_download
        print(f"Downloading {args.model_id} from HuggingFace Hub...")
        args.input_dir = snapshot_download(
            args.model_id,
            local_dir=None,  # Use default cache
        )
        print(f"Downloaded to: {args.input_dir}")

    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 1: Parse config.json
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("Step 1: Processing config.json")
    print("=" * 60)

    config_path = os.path.join(input_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    print(f"  Original model_type: {config.get('model_type')}")
    print(f"  Has text_config: {'text_config' in config}")
    print(f"  Has vision_config: {'vision_config' in config}")

    new_config = extract_text_config(config)
    new_config_path = os.path.join(output_dir, "config.json")
    with open(new_config_path, "w") as f:
        json.dump(new_config, f, indent=2)
    print(f"  Saved text-only config -> {new_config_path}")
    print(f"  New model_type: {new_config.get('model_type')}")

    # -----------------------------------------------------------------------
    # Step 2: Process safetensors weights
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 2: Filtering vision encoder weights")
    print("=" * 60)

    index_path = os.path.join(input_dir, "model.safetensors.index.json")

    if os.path.exists(index_path):
        # Multi-shard model
        new_index, files_needed, stats = process_safetensors_index(index_path)

        print(f"  Total weight keys:  {stats['total_keys']}")
        print(f"  Vision keys (removing): {stats['vision_keys']}")
        print(f"  Text keys (keeping):    {stats['text_keys']}")
        print(f"  Original shard files:   {stats['original_shards']}")
        print(f"  Shard files needed:     {stats['needed_shards']}")

        # Re-save with only text weights
        final_index, total_gb = strip_vision_from_shards(
            input_dir, output_dir, files_needed, new_index["weight_map"]
        )

        # Save updated index
        index_out = os.path.join(output_dir, "model.safetensors.index.json")
        with open(index_out, "w") as f:
            json.dump(final_index, f, indent=2)

    else:
        # Single safetensors file
        single_path = os.path.join(input_dir, "model.safetensors")
        print(f"  Single safetensors file: {single_path}")

        text_tensors = {}
        vision_count = 0
        remapped_count = 0

        with safe_open(single_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if is_vision_weight(key):
                    vision_count += 1
                else:
                    new_key = remap_key(key)
                    if new_key != key:
                        remapped_count += 1
                    text_tensors[new_key] = f.get_tensor(key)

        print(f"  Vision keys (removing): {vision_count}")
        print(f"  Text keys (keeping):    {len(text_tensors)}")
        print(f"  Keys remapped:          {remapped_count}")

        out_path = os.path.join(output_dir, "model.safetensors")
        save_file(text_tensors, out_path)
        total_gb = os.path.getsize(out_path) / (1024 ** 3)
        print(f"  Saved: {total_gb:.2f} GB")

    # -----------------------------------------------------------------------
    # Step 3: Copy tokenizer and other essential files
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 3: Copying tokenizer and auxiliary files")
    print("=" * 60)

    copy_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "generation_config.json",
        "chat_template.jinja",
    ]

    for fname in copy_files:
        src = os.path.join(input_dir, fname)
        if os.path.exists(src):
            dst = os.path.join(output_dir, fname)
            shutil.copy2(src, dst)
            print(f"  Copied: {fname}")
        else:
            print(f"  Skipped (not found): {fname}")

    # -----------------------------------------------------------------------
    # Step 4: Clean up tokenizer_config (remove vision-specific chat template parts)
    # -----------------------------------------------------------------------
    tok_config_path = os.path.join(output_dir, "tokenizer_config.json")
    if os.path.exists(tok_config_path):
        with open(tok_config_path, "r") as f:
            tok_config = json.load(f)

        # Remove vision-specific special tokens from added_tokens if desired
        # (optional — keeping them doesn't hurt, they just won't be used)
        print("  Note: Vision-related special tokens kept in tokenizer (harmless)")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)

    original_size = sum(
        os.path.getsize(os.path.join(input_dir, f))
        for f in os.listdir(input_dir)
        if f.endswith(".safetensors")
    ) / (1024 ** 3)

    print(f"  Original model size (safetensors): ~{original_size:.2f} GB")
    print(f"  Text-only model size:              ~{total_gb:.2f} GB")
    print(f"  Reduction:                         ~{original_size - total_gb:.2f} GB")
    print(f"  Output directory:                  {output_dir}")



if __name__ == "__main__":
    main()