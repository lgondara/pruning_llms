"""
Data Distillation Pipeline: Teacher Sampling → Student SFT
===========================================================

Two-stage pipeline using ms-swift:
  Stage 1 (swift sample): Generate high-quality training data from a teacher model.
           Supports local models (via transformers/vLLM/lmdeploy) or remote APIs
           (OpenAI-compatible endpoints). Optional quality filtering via ORM/PRM.
  Stage 2 (swift sft):    Fine-tune a smaller student model on the distilled data.

This approach treats the teacher as a black box — no logit access required.
Suitable for distilling from closed-source APIs (GPT-4, Claude, DeepSeek-R1)
or when teacher and student architectures are incompatible.

Requirements:
    pip install ms-swift[llm] --break-system-packages
    # For vLLM-accelerated local sampling:
    pip install vllm --break-system-packages

Usage:
    # Full pipeline: sample from local teacher + SFT student
    python data_distillation_pipeline.py

    # Sample from API (e.g. DeepSeek-R1 via DashScope)
    python data_distillation_pipeline.py \
        --teacher_model deepseek-r1 \
        --sampler_engine client \
        --api_base "https://dashscope.aliyuncs.com/compatible-mode/v1" \
        --api_key "your_key"

    # Sample only (skip SFT)
    python data_distillation_pipeline.py --sample_only

    # SFT only (using previously generated data)
    python data_distillation_pipeline.py \
        --sft_only \
        --distilled_data_path sample_output/my_data.jsonl
"""

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from datetime import datetime


# ---------------------------------------------------------------------------
# Stage 1: Teacher Sampling Configuration
# ---------------------------------------------------------------------------

@dataclass
class SamplingConfig:
    """Configuration for teacher data generation (swift sample)."""

    # Teacher model: HuggingFace/ModelScope ID for local, or model name for API
    teacher_model: str = "Qwen/Qwen2.5-7B-Instruct"

    # Inference engine for sampling
    # "transformers" - default, no extra deps
    # "vllm"         - fastest for local models
    # "lmdeploy"     - alternative accelerator
    # "client"       - for OpenAI-compatible API endpoints
    sampler_engine: str = "vllm"

    # Sampler type: "sample" (rejection sampling) or "distill" (API distillation)
    sampler_type: str = "sample"

    # Dataset to generate responses for
    dataset: str = "AI-ModelScope/alpaca-gpt4-data-en#100"

    # Number of candidate responses per prompt
    num_return_sequences: int = 5

    # Keep only the top-N responses (after ORM/PRM scoring)
    n_best_to_keep: Optional[int] = 1

    # Sampling parameters
    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 2048

    # Output
    output_dir: str = "sample_output"
    output_file: Optional[str] = None  # None -> timestamp-based filename

    # ---- Quality filtering (optional) ----
    # Outcome Reward Model: verifies final answer correctness
    # Built-in: "math" for GSM8K-style; or a model ID
    orm_model: Optional[str] = None

    # Process Reward Model: scores intermediate reasoning steps
    prm_model: Optional[str] = None

    # ---- API-specific (sampler_engine="client") ----
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    stream: bool = True

    # ---- Batch processing ----
    num_sampling_batch_size: int = 50


def build_sample_args(config: SamplingConfig) -> list[str]:
    """Build CLI arguments for swift sample."""
    args = [
        "swift", "sample",
        "--model", config.teacher_model,
        "--sampler_engine", config.sampler_engine,
        "--sampler_type", config.sampler_type,
        "--dataset", config.dataset,
        "--num_return_sequences", str(config.num_return_sequences),
        "--temperature", str(config.temperature),
        "--top_p", str(config.top_p),
        "--max_new_tokens", str(config.max_new_tokens),
        "--output_dir", config.output_dir,
        "--num_sampling_batch_size", str(config.num_sampling_batch_size),
    ]

    if config.n_best_to_keep is not None:
        args += ["--n_best_to_keep", str(config.n_best_to_keep)]

    if config.output_file:
        args += ["--output_file", config.output_file]

    # Quality filtering
    if config.orm_model:
        args += ["--orm_model", config.orm_model]
    if config.prm_model:
        args += ["--prm_model", config.prm_model]

    # API-specific settings
    if config.sampler_engine == "client":
        if config.api_base:
            args += [
                "--engine_kwargs",
                f'{{"base_url":"{config.api_base}"}}',
            ]
        args += ["--stream", str(config.stream).lower()]

    return args


def run_sampling(config: SamplingConfig) -> str:
    """
    Execute Stage 1: generate distilled data from the teacher.
    Returns the path to the generated jsonl file.
    """
    # Set API key if using client engine
    env = os.environ.copy()
    if config.api_key and config.sampler_engine == "client":
        env["OPENAI_API_KEY"] = config.api_key

    args = build_sample_args(config)
    cmd_str = " \\\n    ".join(args)

    print("=" * 70)
    print("Stage 1: Teacher Sampling")
    print("=" * 70)
    print(f"  Teacher:    {config.teacher_model}")
    print(f"  Engine:     {config.sampler_engine}")
    print(f"  Dataset:    {config.dataset}")
    print(f"  Sequences:  {config.num_return_sequences} per prompt")
    print(f"  Keep best:  {config.n_best_to_keep}")
    if config.orm_model:
        print(f"  ORM:        {config.orm_model}")
    if config.prm_model:
        print(f"  PRM:        {config.prm_model}")
    print("=" * 70)
    print(f"\nCommand:\n{cmd_str}\n")

    subprocess.run(args, check=True, env=env)

    # Find the generated output file
    output_dir = Path(config.output_dir)
    if config.output_file:
        return str(output_dir / config.output_file)

    # Get the most recent jsonl file
    jsonl_files = sorted(output_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files found in {output_dir}")
    return str(jsonl_files[-1])


# ---------------------------------------------------------------------------
# Stage 2: Student SFT Configuration
# ---------------------------------------------------------------------------

@dataclass
class SFTConfig:
    """Configuration for student fine-tuning (swift sft)."""

    # Student model
    student_model: str = "Qwen/Qwen2.5-1.5B-Instruct"

    # Distilled dataset (output from Stage 1)
    dataset: str = ""  # Will be set to sampling output path

    # Tuning strategy
    tuner_type: str = "lora"
    lora_rank: int = 8
    lora_alpha: int = 32
    target_modules: str = "all-linear"

    # Training hyperparameters
    num_train_epochs: int = 3
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_length: int = 2048
    warmup_ratio: float = 0.05

    # Infrastructure
    torch_dtype: str = "bfloat16"
    output_dir: str = "output/data_distilled_student"
    deepspeed: Optional[str] = None
    padding_free: bool = False

    # Logging and saving
    logging_steps: int = 5
    save_steps: int = 200
    save_total_limit: int = 3
    report_to: str = "tensorboard"

    # Validation
    val_dataset: Optional[str] = None
    eval_steps: int = 200


def build_sft_args(config: SFTConfig) -> list[str]:
    """Build CLI arguments for swift sft."""
    args = [
        "swift", "sft",
        "--model", config.student_model,
        "--dataset", config.dataset,
        "--num_train_epochs", str(config.num_train_epochs),
        "--learning_rate", str(config.learning_rate),
        "--per_device_train_batch_size", str(config.per_device_train_batch_size),
        "--gradient_accumulation_steps", str(config.gradient_accumulation_steps),
        "--max_length", str(config.max_length),
        "--warmup_ratio", str(config.warmup_ratio),
        "--torch_dtype", config.torch_dtype,
        "--output_dir", config.output_dir,
        "--logging_steps", str(config.logging_steps),
        "--save_steps", str(config.save_steps),
        "--save_total_limit", str(config.save_total_limit),
        "--report_to", config.report_to,
    ]

    # Tuner
    if config.tuner_type == "lora":
        args += [
            "--tuner_type", "lora",
            "--lora_rank", str(config.lora_rank),
            "--lora_alpha", str(config.lora_alpha),
            "--target_modules", config.target_modules,
        ]
    else:
        args += ["--tuner_type", "full"]

    if config.deepspeed:
        args += ["--deepspeed", config.deepspeed]

    if config.padding_free:
        args += ["--padding_free", "true", "--attn_impl", "flash_attn"]

    if config.val_dataset:
        args += [
            "--val_dataset", config.val_dataset,
            "--eval_steps", str(config.eval_steps),
        ]

    return args


def run_sft(config: SFTConfig) -> None:
    """Execute Stage 2: fine-tune the student on distilled data."""
    args = build_sft_args(config)
    cmd_str = " \\\n    ".join(args)

    print("=" * 70)
    print("Stage 2: Student SFT on Distilled Data")
    print("=" * 70)
    print(f"  Student:   {config.student_model}")
    print(f"  Dataset:   {config.dataset}")
    print(f"  Tuner:     {config.tuner_type}")
    print(f"  Epochs:    {config.num_train_epochs}")
    print(f"  LR:        {config.learning_rate}")
    print("=" * 70)
    print(f"\nCommand:\n{cmd_str}\n")

    subprocess.run(args, check=True)


# ---------------------------------------------------------------------------
# Two-stage quality filtering (avoids OOM from loading sampler + RM together)
# ---------------------------------------------------------------------------

def run_two_stage_sampling(config: SamplingConfig) -> str:
    """
    Memory-efficient sampling: separates generation from reward scoring
    to avoid loading both the teacher and reward model simultaneously.

    Stage 1a: Generate samples (teacher model only)
    Stage 1b: Score and filter (reward model only, no teacher)
    """
    # 1a: Generate without reward models
    config_gen = SamplingConfig(
        teacher_model=config.teacher_model,
        sampler_engine=config.sampler_engine,
        sampler_type=config.sampler_type,
        dataset=config.dataset,
        num_return_sequences=config.num_return_sequences,
        n_best_to_keep=None,  # Keep all for now
        temperature=config.temperature,
        top_p=config.top_p,
        max_new_tokens=config.max_new_tokens,
        output_dir=config.output_dir,
        output_file="raw_samples.jsonl",
        api_base=config.api_base,
        api_key=config.api_key,
        stream=config.stream,
    )

    print("[Two-stage] Phase 1a: Generating samples...")
    raw_path = run_sampling(config_gen)

    # 1b: Score and filter (no sampling, just RM evaluation)
    filter_args = [
        "swift", "sample",
        "--sampler_engine", "no",  # No sampling, just filtering
        "--cache_files", raw_path,
        "--n_best_to_keep", str(config.n_best_to_keep or 1),
        "--output_dir", config.output_dir,
        "--output_file", "filtered_samples.jsonl",
    ]
    if config.orm_model:
        filter_args += ["--orm_model", config.orm_model]
    if config.prm_model:
        filter_args += ["--prm_model", config.prm_model]

    print("[Two-stage] Phase 1b: Scoring and filtering...")
    subprocess.run(filter_args, check=True)

    filtered_path = str(Path(config.output_dir) / "filtered_samples.jsonl")
    print(f"[Two-stage] Filtered data saved to {filtered_path}")
    return filtered_path


# ---------------------------------------------------------------------------
# Preset pipeline configurations
# ---------------------------------------------------------------------------

def preset_local_reasoning_distillation() -> tuple[SamplingConfig, SFTConfig]:
    """Distill reasoning ability from a large local model.
    Uses math ORM for answer verification."""
    sample_cfg = SamplingConfig(
        teacher_model="Qwen/Qwen2.5-7B-Instruct",
        sampler_engine="vllm",
        sampler_type="sample",
        dataset="modelscope/gsm8k#5000",
        num_return_sequences=8,
        n_best_to_keep=1,
        temperature=0.7,
        orm_model="math",
    )
    sft_cfg = SFTConfig(
        student_model="Qwen/Qwen2.5-1.5B-Instruct",
        num_train_epochs=5,
        learning_rate=2e-5,
    )
    return sample_cfg, sft_cfg


def preset_api_distillation() -> tuple[SamplingConfig, SFTConfig]:
    """Distill from a closed-source API (e.g. DeepSeek-R1).
    No local teacher GPU required for sampling."""
    sample_cfg = SamplingConfig(
        teacher_model="deepseek-r1",
        sampler_engine="client",
        sampler_type="distill",
        dataset="AI-ModelScope/alpaca-gpt4-data-en#2000",
        num_return_sequences=1,
        temperature=0.6,
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    sft_cfg = SFTConfig(
        student_model="Qwen/Qwen2.5-1.5B-Instruct",
        num_train_epochs=3,
        learning_rate=2e-5,
    )
    return sample_cfg, sft_cfg


def preset_with_prm_filtering() -> tuple[SamplingConfig, SFTConfig]:
    """Distill with Process Reward Model filtering.
    Scores each reasoning step, not just the final answer."""
    sample_cfg = SamplingConfig(
        teacher_model="Qwen/Qwen2.5-7B-Instruct",
        sampler_engine="vllm",
        dataset="modelscope/gsm8k",
        num_return_sequences=10,
        n_best_to_keep=2,
        temperature=0.8,
        orm_model="math",
        prm_model="AI-ModelScope/GRM-llama3.2-3B-rewardmodel-ft",
    )
    sft_cfg = SFTConfig(
        student_model="Qwen/Qwen2.5-1.5B-Instruct",
        num_train_epochs=5,
    )
    return sample_cfg, sft_cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Data Distillation Pipeline")

    # Pipeline control
    parser.add_argument("--sample_only", action="store_true",
                        help="Run sampling only, skip SFT")
    parser.add_argument("--sft_only", action="store_true",
                        help="Run SFT only, requires --distilled_data_path")
    parser.add_argument("--two_stage", action="store_true",
                        help="Use two-stage sampling to avoid OOM with reward models")

    # Preset
    parser.add_argument("--preset", type=str, default=None,
                        choices=["local_reasoning", "api_distill", "prm_filtering"],
                        help="Use a preset configuration")

    # Sampling args
    parser.add_argument("--teacher_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--sampler_engine", type=str, default="vllm")
    parser.add_argument("--dataset", type=str, default="AI-ModelScope/alpaca-gpt4-data-en#100")
    parser.add_argument("--num_return_sequences", type=int, default=5)
    parser.add_argument("--n_best_to_keep", type=int, default=1)
    parser.add_argument("--api_base", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)

    # SFT args
    parser.add_argument("--student_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--distilled_data_path", type=str, default=None)
    parser.add_argument("--tuner_type", type=str, default="lora")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="output/data_distilled_student")

    # Post-training
    parser.add_argument("--merge_lora", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load preset or build from CLI args
    if args.preset == "local_reasoning":
        sample_cfg, sft_cfg = preset_local_reasoning_distillation()
    elif args.preset == "api_distill":
        sample_cfg, sft_cfg = preset_api_distillation()
    elif args.preset == "prm_filtering":
        sample_cfg, sft_cfg = preset_with_prm_filtering()
    else:
        sample_cfg = SamplingConfig(
            teacher_model=args.teacher_model,
            sampler_engine=args.sampler_engine,
            dataset=args.dataset,
            num_return_sequences=args.num_return_sequences,
            n_best_to_keep=args.n_best_to_keep,
            api_base=args.api_base,
            api_key=args.api_key,
            sampler_type="distill" if args.sampler_engine == "client" else "sample",
        )
        sft_cfg = SFTConfig(
            student_model=args.student_model,
            tuner_type=args.tuner_type,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            deepspeed=args.deepspeed,
            output_dir=args.output_dir,
        )

    # ---- Execute pipeline ----

    distilled_data_path = args.distilled_data_path

    # Stage 1: Sampling
    if not args.sft_only:
        if args.two_stage and (sample_cfg.orm_model or sample_cfg.prm_model):
            distilled_data_path = run_two_stage_sampling(sample_cfg)
        else:
            distilled_data_path = run_sampling(sample_cfg)
        print(f"\nDistilled data: {distilled_data_path}")

    if args.sample_only:
        print("Sampling complete. Exiting (--sample_only).")
        sys.exit(0)

    # Stage 2: Student SFT
    if distilled_data_path is None:
        print("ERROR: --distilled_data_path required with --sft_only")
        sys.exit(1)

    sft_cfg.dataset = distilled_data_path
    run_sft(sft_cfg)

    # Optional: merge and export
    if args.merge_lora and sft_cfg.tuner_type == "lora":
        output_path = Path(sft_cfg.output_dir)
        checkpoints = sorted(output_path.glob("*/checkpoint-*"))
        if checkpoints:
            latest_ckpt = str(checkpoints[-1])
            merge_args = [
                "swift", "export",
                "--adapters", latest_ckpt,
                "--merge_lora", "true",
                "--output_dir", f"{sft_cfg.output_dir}/merged",
            ]
            subprocess.run(merge_args, check=True)
            print(f"Merged model exported to {sft_cfg.output_dir}/merged")

    print("\nPipeline complete.")