"""
Logit-Level Knowledge Distillation via GKD (Generalized Knowledge Distillation)
================================================================================

Uses ms-swift's GKD implementation based on Agarwal et al., "On-Policy Distillation
of Language Models: Learning from Self-Generated Mistakes" (ICLR 2024).

Core idea: instead of training the student on fixed ground-truth sequences only,
GKD also trains on student-generated outputs using the teacher's distribution as
supervision. The divergence is generalized JSD, interpolating between forward KL
(mode-covering) and reverse KL (mode-seeking) via beta.

Loss:
    L_GKD(x, y) = sum_t D_JSD(P_teacher(.|x, y_<t), P_student(.|x, y_<t))

    D_JSD(beta)(P, Q) = beta * KL(P || M) + (1-beta) * KL(Q || M)
    where M = beta * P + (1-beta) * Q

Training mode selection per sample:
    if random() < lmbda:       -> on-policy: y ~ student.generate(x)
    elif seq_kd:               -> seq-KD: y ~ teacher.generate(x)
    else:                      -> offline: y = y_ground_truth

Requirements:
    pip install ms-swift[llm] --break-system-packages
    # For vLLM acceleration (recommended for on-policy sampling):
    pip install vllm --break-system-packages

Usage:
    # Single GPU, off-policy only (fastest, no student sampling)
    python logit_distillation_gkd.py

    # Multi-GPU with on-policy sampling
    CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 python logit_distillation_gkd.py

    # With vLLM-accelerated on-policy sampling
    python logit_distillation_gkd.py --use_vllm
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class GKDConfig:
    """Configuration for GKD-based logit distillation."""

    # ---- Model specification ----
    teacher_model: str = "Qwen/Qwen2.5-7B-Instruct"
    student_model: str = "Qwen/Qwen2.5-1.5B-Instruct"

    # ---- Dataset ----
    # ms-swift accepts HuggingFace/ModelScope IDs or local jsonl paths.
    # Append #N to subsample, e.g. "dataset_id#5000"
    dataset: str = "AI-ModelScope/alpaca-gpt4-data-en"

    # ---- GKD-specific parameters ----
    # lmbda: probability of on-policy (student-generated) samples per step.
    #   lmbda=0   -> purely off-policy (dataset only, fastest)
    #   lmbda=0.5 -> 50/50 on-policy vs dataset/teacher
    #   lmbda=1.0 -> fully on-policy (student generates all outputs)
    lmbda: float = 0.5

    # beta: JSD interpolation between forward KL and reverse KL.
    #   beta=0 -> reverse KL (mode-seeking, student concentrates on teacher peaks)
    #   beta=1 -> forward KL (mode-covering, student covers full teacher dist)
    #   beta=0.5 -> symmetric JSD
    beta: float = 0.5

    # seq_kd: when True and lmbda < 1, uses teacher-generated outputs
    # (sequential KD) with probability (1-lmbda) instead of ground truth.
    # Recommended: pre-generate teacher data offline and set seq_kd=False.
    seq_kd: bool = False

    # sft_alpha: weight of auxiliary SFT loss. final_loss = gkd_loss + sft_alpha * sft_loss
    # Small positive values (0.1-0.5) can stabilise early training.
    sft_alpha: float = 0.0

    # Number of top-k teacher logits to retain (saves memory).
    gkd_logits_topk: int = 64

    # ---- Tuning strategy ----
    # "lora" or "full". LoRA is default and memory-efficient.
    tuner_type: str = "lora"
    lora_rank: int = 8
    lora_alpha: int = 32

    # ---- Training hyperparameters ----
    num_train_epochs: int = 3
    learning_rate: float = 5e-5
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_length: int = 2048
    max_completion_length: int = 2048

    # ---- Infrastructure ----
    torch_dtype: str = "bfloat16"
    output_dir: str = "output/gkd_distillation"
    deepspeed: Optional[str] = None  # e.g. "zero2", "zero3"
    padding_free: bool = False       # requires flash_attn, saves memory

    # ---- vLLM acceleration for on-policy sampling ----
    use_vllm: bool = False
    vllm_gpu_memory_utilization: float = 0.5
    vllm_max_model_len: int = 4096
    vllm_mode: str = "colocate"      # "colocate" or "server"

    # ---- Logging ----
    logging_steps: int = 5
    save_steps: int = 200
    save_total_limit: int = 3
    report_to: str = "tensorboard"   # or "wandb"


def build_cli_args(config: GKDConfig) -> list[str]:
    """Convert config to ms-swift CLI arguments."""
    args = [
        "swift", "rlhf",
        "--rlhf_type", "gkd",
        "--model", config.student_model,
        "--teacher_model", config.teacher_model,
        "--dataset", config.dataset,

        # GKD parameters
        "--lmbda", str(config.lmbda),
        "--beta", str(config.beta),
        "--seq_kd", str(config.seq_kd).lower(),
        "--sft_alpha", str(config.sft_alpha),
        "--gkd_logits_topk", str(config.gkd_logits_topk),

        # Training
        "--num_train_epochs", str(config.num_train_epochs),
        "--learning_rate", str(config.learning_rate),
        "--per_device_train_batch_size", str(config.per_device_train_batch_size),
        "--gradient_accumulation_steps", str(config.gradient_accumulation_steps),
        "--max_length", str(config.max_length),
        "--max_completion_length", str(config.max_completion_length),
        "--torch_dtype", config.torch_dtype,
        "--output_dir", config.output_dir,

        # Logging
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
        ]
    else:
        args += ["--tuner_type", "full"]

    # DeepSpeed
    if config.deepspeed:
        args += ["--deepspeed", config.deepspeed]

    # padding_free (requires flash_attn + transformers>=4.44)
    if config.padding_free:
        args += ["--padding_free", "true", "--attn_impl", "flash_attn"]

    # vLLM for on-policy acceleration
    if config.use_vllm and config.lmbda > 0:
        args += [
            "--use_vllm", "true",
            "--vllm_mode", config.vllm_mode,
            "--vllm_gpu_memory_utilization", str(config.vllm_gpu_memory_utilization),
            "--vllm_max_model_len", str(config.vllm_max_model_len),
        ]

    return args


def pre_generate_teacher_data(
    teacher_model: str,
    dataset: str,
    output_path: str = "teacher_generated_data.jsonl",
    infer_backend: str = "vllm",
    max_new_tokens: int = 2048,
) -> str:
    """
    Optional: pre-generate teacher outputs offline for seq_kd.
    This avoids the teacher generating on-the-fly during training,
    which is significantly faster for large teacher models.

    Returns the path to the generated jsonl file.
    """
    print(f"[Pre-generation] Generating teacher data from {teacher_model}...")
    cmd = [
        "swift", "infer",
        "--model", teacher_model,
        "--infer_backend", infer_backend,
        "--val_dataset", dataset,
        "--max_new_tokens", str(max_new_tokens),
        "--result_path", output_path,
        "--write_batch_size", "500",
    ]
    subprocess.run(cmd, check=True)
    print(f"[Pre-generation] Teacher data saved to {output_path}")
    return output_path


def run_gkd_distillation(config: GKDConfig) -> None:
    """Launch GKD distillation training."""
    args = build_cli_args(config)
    cmd_str = " \\\n    ".join(args)
    print("=" * 70)
    print("GKD Distillation Configuration")
    print("=" * 70)
    print(f"  Teacher:  {config.teacher_model}")
    print(f"  Student:  {config.student_model}")
    print(f"  Dataset:  {config.dataset}")
    print(f"  lmbda:    {config.lmbda} (on-policy probability)")
    print(f"  beta:     {config.beta} (JSD interpolation)")
    print(f"  seq_kd:   {config.seq_kd}")
    print(f"  Tuner:    {config.tuner_type}")
    print(f"  vLLM:     {config.use_vllm}")
    print("=" * 70)
    print(f"\nCommand:\n{cmd_str}\n")

    subprocess.run(args, check=True)


def run_inference(checkpoint_dir: str) -> None:
    """Run interactive inference with the distilled student model."""
    cmd = [
        "swift", "infer",
        "--adapters", checkpoint_dir,
        "--stream", "true",
        "--temperature", "0",
        "--max_new_tokens", "2048",
    ]
    subprocess.run(cmd, check=True)


def merge_and_export(checkpoint_dir: str, output_dir: str) -> None:
    """Merge LoRA weights and export the final model."""
    cmd = [
        "swift", "export",
        "--adapters", checkpoint_dir,
        "--merge_lora", "true",
        "--output_dir", output_dir,
    ]
    subprocess.run(cmd, check=True)
    print(f"Merged model exported to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GKD Logit Distillation with ms-swift")

    # Model args
    parser.add_argument("--teacher_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--student_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--dataset", type=str, default="AI-ModelScope/alpaca-gpt4-data-en")

    # GKD args
    parser.add_argument("--lmbda", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--seq_kd", action="store_true")
    parser.add_argument("--sft_alpha", type=float, default=0.0)

    # Training args
    parser.add_argument("--tuner_type", type=str, default="lora", choices=["lora", "full"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="output/gkd_distillation")

    # vLLM
    parser.add_argument("--use_vllm", action="store_true")

    # Pre-generate teacher data
    parser.add_argument("--pre_generate", action="store_true",
                        help="Pre-generate teacher data before training (for seq_kd)")

    # Post-training
    parser.add_argument("--merge_lora", action="store_true",
                        help="Merge LoRA and export after training")

    return parser.parse_args()


# ---- Preset configurations for common distillation scenarios ----

def config_offpolicy_fast() -> GKDConfig:
    """Off-policy only: fastest training, uses dataset ground truth.
    Good baseline; no student sampling overhead."""
    return GKDConfig(
        lmbda=0.0,
        seq_kd=False,
        beta=0.5,
    )


def config_onpolicy_balanced() -> GKDConfig:
    """Balanced on-policy + off-policy with JSD.
    Best general-purpose setting per the GKD paper."""
    return GKDConfig(
        lmbda=0.5,
        seq_kd=False,
        beta=0.5,
        use_vllm=True,
    )


def config_onpolicy_mode_seeking() -> GKDConfig:
    """Fully on-policy with reverse KL (mode-seeking).
    Student focuses on high-probability teacher regions.
    Best for reasoning tasks where precision > coverage."""
    return GKDConfig(
        lmbda=1.0,
        seq_kd=False,
        beta=0.0,
        use_vllm=True,
    )


def config_seq_kd_pregenerated() -> GKDConfig:
    """Sequential KD with pre-generated teacher data.
    Teacher outputs replace ground truth. Set dataset to the
    pre-generated jsonl file."""
    return GKDConfig(
        lmbda=0.0,
        seq_kd=False,  # False because data is already teacher-generated
        beta=0.5,
        dataset="teacher_generated_data.jsonl",
    )


if __name__ == "__main__":
    args = parse_args()

    config = GKDConfig(
        teacher_model=args.teacher_model,
        student_model=args.student_model,
        dataset=args.dataset,
        lmbda=args.lmbda,
        beta=args.beta,
        seq_kd=args.seq_kd,
        sft_alpha=args.sft_alpha,
        tuner_type=args.tuner_type,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        deepspeed=args.deepspeed,
        output_dir=args.output_dir,
        use_vllm=args.use_vllm,
    )

    # Optional: pre-generate teacher data for seq_kd
    if args.pre_generate:
        teacher_data_path = pre_generate_teacher_data(
            teacher_model=config.teacher_model,
            dataset=config.dataset,
        )
        config.dataset = teacher_data_path
        config.seq_kd = False  # data is already teacher-generated

    # Run distillation
    run_gkd_distillation(config)

    # Optional: merge LoRA and export
    if args.merge_lora and config.tuner_type == "lora":
        # Find latest checkpoint
        output_path = Path(config.output_dir)
        checkpoints = sorted(output_path.glob("*/checkpoint-*"))
        if checkpoints:
            latest_ckpt = str(checkpoints[-1])
            merge_and_export(latest_ckpt, f"{config.output_dir}/merged")