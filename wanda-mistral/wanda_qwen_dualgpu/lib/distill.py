"""
KL Divergence Distillation for Post-Pruning Recovery
=====================================================

After aggressive structured pruning, the model loses quality. This module
recovers it by training the pruned (student) model to match the output
distribution of the original unpruned (teacher) model.

Loss = KL(teacher_logits || student_logits) + α * CE(student_logits, labels)

The teacher is frozen; only the student is updated.

Hardware strategies:
    - 2 GPUs: teacher on GPU 0, student on GPU 1 (full float16, best quality)
    - 1 GPU (40GB+): both on same GPU in float16
    - 1 GPU (24GB): 4-bit teacher + 8-bit Adam + gradient checkpointing
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from typing import Optional

from .data import get_loaders


@torch.no_grad()
def get_teacher_logits(teacher, input_ids, temperature: float = 1.0):
    """
    Get soft targets from the teacher model.

    Args:
        teacher: Original unpruned model (frozen)
        input_ids: Input token IDs (must be on same device as teacher)
        temperature: Softmax temperature (higher = softer distribution)

    Returns:
        Teacher logits (before softmax), shape (batch, seq, vocab)
    """
    outputs = teacher(input_ids)
    return outputs.logits / temperature


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.0,
    alpha: float = 0.5,
    chunk_size: int = 32,
    debug: bool = False,
):
    """
    Combined distillation loss in float32, chunked over sequence dim.

    KL is computed MANUALLY instead of using F.kl_div because:
    F.kl_div with log_target=False internally computes target * log(target).
    With 248K vocab, softmax produces exact zeros for rare tokens, and
    0 * log(0) = 0 * -inf = NaN.

    Instead we use: KL = sum(exp(t_log) * (t_log - s_log))
    where t_log = log_softmax(teacher). This is safe because:
    - log_softmax never produces -inf (it's x - logsumexp(x), always finite)
    - exp(very_negative) = 0.0 in float32, and 0.0 * finite = 0.0 (not NaN)

    Loss = α * KL(teacher || student) * T² + (1 - α) * CE(student, labels)
    """
    seq_len = student_logits.shape[1] - 1  # -1 for next-token shift
    n_chunks = 0
    loss = torch.tensor(0.0, device=student_logits.device, requires_grad=True)

    kl_log = 0.0
    ce_log = 0.0

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)

        # Cast to float32 and clamp
        s_chunk = student_logits[:, start:end, :].float().clamp(-1e4, 1e4)
        l_chunk = labels[:, start + 1:end + 1]

        # Student log probs (retains grad)
        s_log_probs = F.log_softmax(s_chunk / temperature, dim=-1)

        # Teacher log probs (no grad) — already float32 from training loop
        with torch.no_grad():
            t_chunk = teacher_logits[:, start:end, :].float()
            # Belt-and-suspenders: clamp before log_softmax
            t_chunk = t_chunk.clamp(-1e4, 1e4)
            t_log_probs = F.log_softmax(t_chunk / temperature, dim=-1)

        # Manual KL: sum_vocab(exp(t_log) * (t_log - s_log)), mean over tokens
        t_probs = t_log_probs.exp()
        kl_per_token = (t_probs * (t_log_probs - s_log_probs)).sum(dim=-1)
        chunk_kl = kl_per_token.mean() * (temperature ** 2)

        # CE
        chunk_ce = F.cross_entropy(
            s_chunk.reshape(-1, s_chunk.size(-1)),
            l_chunk.reshape(-1),
            ignore_index=-100,
        )

        if debug and start == 0:
            print(f"  [DEBUG LOSS] chunk 0:")
            print(f"    s_chunk: min={s_chunk.min().item():.2f}, max={s_chunk.max().item():.2f}, nan={s_chunk.isnan().any().item()}")
            print(f"    t_chunk: min={t_chunk.min().item():.2f}, max={t_chunk.max().item():.2f}, nan={t_chunk.isnan().any().item()}")
            print(f"    s_log_probs: min={s_log_probs.min().item():.2f}, max={s_log_probs.max().item():.4f}, nan={s_log_probs.isnan().any().item()}")
            print(f"    t_log_probs: min={t_log_probs.min().item():.2f}, max={t_log_probs.max().item():.4f}, nan={t_log_probs.isnan().any().item()}")
            print(f"    t_probs: min={t_probs.min().item():.6f}, max={t_probs.max().item():.4f}, nan={t_probs.isnan().any().item()}")
            print(f"    kl_per_token: min={kl_per_token.min().item():.4f}, max={kl_per_token.max().item():.4f}, nan={kl_per_token.isnan().any().item()}")
            print(f"    chunk_kl={chunk_kl.item():.4f}, chunk_ce={chunk_ce.item():.4f}")

        loss = loss + alpha * chunk_kl + (1 - alpha) * chunk_ce

        kl_log += chunk_kl.item()
        ce_log += chunk_ce.item()
        n_chunks += 1

        del s_log_probs, t_log_probs, t_probs, s_chunk, t_chunk

    loss = loss / n_chunks
    return loss, kl_log / n_chunks, ce_log / n_chunks


def distill(
    teacher_path: str,
    student_model,
    tokenizer,
    device,
    calib_dataset: str = "c4",
    nsamples: int = 256,
    seqlen: int = 1024,
    seed: int = 0,
    num_steps: int = 1000,
    batch_size: int = 1,
    lr: float = 2e-5,
    temperature: float = 2.0,
    alpha: float = 0.7,
    save_every: int = 500,
    save_dir: Optional[str] = None,
    gradient_accumulation: int = 4,
):
    """
    Distill knowledge from teacher (original) into student (pruned) model.

    Device strategy:
        - 2+ GPUs: teacher on cuda:0, student on cuda:1 (best quality)
        - 1 GPU 40GB+: both on same device
        - 1 GPU 24GB: 4-bit teacher + 8-bit Adam + grad checkpointing
    """
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    print("=" * 60)
    print("KL Divergence Distillation")
    print("=" * 60)
    print(f"  Teacher: {teacher_path}")
    print(f"  Device: {device}")
    print(f"  Steps: {num_steps}")
    print(f"  LR: {lr}")
    print(f"  Temperature: {temperature}")
    print(f"  Alpha (KL weight): {alpha}")
    print(f"  Seq length: {seqlen}")
    print(f"  Batch size: {batch_size} × {gradient_accumulation} accumulation")
    print(f"  Effective batch: {batch_size * gradient_accumulation}")

    # ── Determine device strategy ─────────────────────────────────────────
    num_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    print(f"  GPUs available: {num_gpus}")

    if device.type == "cuda" and num_gpus >= 2:
        # ── Strategy: Teacher on GPU 0, Student on GPU 1 ──────────────
        strategy = "multi_gpu"
        teacher_device = torch.device("cuda:0")
        student_device = torch.device("cuda:1")

        print(f"\n  Strategy: Multi-GPU")
        print(f"    Teacher → cuda:0 (float16, ~8GB)")
        print(f"    Student → cuda:1 (float16 + optimizer + grads)")

        # First, move student off GPU 0 to make room for teacher.
        # Remove device_map dispatch hooks (from device_map="auto" in main.py)
        # so .to() actually moves all tensors.
        if hasattr(student_model, "hf_device_map"):
            from accelerate.hooks import remove_hook_from_submodules
            remove_hook_from_submodules(student_model)
            delattr(student_model, "hf_device_map")
            print("    Removed device_map hooks from student")

        student_model = student_model.cpu()
        torch.cuda.empty_cache()

        # Load teacher on GPU 0
        teacher = AutoModelForCausalLM.from_pretrained(
            teacher_path,
            torch_dtype=torch.float16,
            device_map={"": teacher_device},
            trust_remote_code=True,
        )

        # Now move student to GPU 1
        student_model = student_model.to(student_device)

    elif device.type == "cuda":
        # ── Strategy: Single GPU — try 4-bit teacher ──────────────────
        strategy = "single_gpu"
        student_device = device

        print(f"\n  Strategy: Single GPU")

        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            teacher = AutoModelForCausalLM.from_pretrained(
                teacher_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            print(f"    Teacher: 4-bit NF4 (~2GB)")
        except (ImportError, Exception) as e:
            print(f"    4-bit failed ({e}), falling back to float16")
            print(f"    WARNING: May OOM on 24GB. Use 2 GPUs or reduce seqlen.")
            teacher = AutoModelForCausalLM.from_pretrained(
                teacher_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )

        teacher_device = next(teacher.parameters()).device

    else:
        # ── Strategy: CPU/MPS ─────────────────────────────────────────
        strategy = "cpu"
        student_device = device
        teacher = AutoModelForCausalLM.from_pretrained(
            teacher_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        teacher = teacher.to(device)
        teacher_device = device

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student_model.parameters())
    print(f"  Teacher params: {teacher_params / 1e9:.2f}B (on {teacher_device})")
    print(f"  Student params: {student_params / 1e9:.2f}B (on {student_device})")

    # ── Load training data ────────────────────────────────────────────────
    print(f"\nLoading distillation data: {calib_dataset}")
    data = get_loaders(
        calib_dataset,
        nsamples=nsamples,
        seed=seed,
        seqlen=seqlen,
        tokenizer=tokenizer,
    )
    print(f"  Loaded {len(data)} samples")

    # ── Setup optimizer ───────────────────────────────────────────────────
    student_model.train()

    # Enable gradient checkpointing to save activation memory
    if hasattr(student_model, "gradient_checkpointing_enable"):
        student_model.gradient_checkpointing_enable()
        print("  Gradient checkpointing: enabled")

    # Only train MLP parameters — structured pruning targets MLPs, so only
    # those need recovery. Freezing attention + embeddings cuts gradient
    # memory roughly in half, critical for fitting on 24GB.
    train_params = []
    frozen_count = 0
    for name, p in student_model.named_parameters():
        if "mlp" in name:
            p.requires_grad = True
            train_params.append(p)
        else:
            p.requires_grad = False
            frozen_count += 1

    trainable_size = sum(p.numel() for p in train_params)
    print(f"  Trainable (MLP only): {trainable_size / 1e6:.1f}M")
    print(f"  Frozen (attention + embeddings): {frozen_count} layers")

    # Use 8-bit Adam if bitsandbytes is available
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(train_params, lr=lr, weight_decay=0.01)
        print(f"  Optimizer: AdamW 8-bit (~{trainable_size * 6 / 1e9:.1f}GB saved vs float32)")
    except ImportError:
        print("  bitsandbytes not found, using standard AdamW")
        print("  Install with: pip install bitsandbytes")
        optimizer = AdamW(train_params, lr=lr, weight_decay=0.01)

    scheduler = CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=lr * 0.1)

    # ── Training loop ─────────────────────────────────────────────────────
    print(f"\nStarting distillation...")
    t0 = time.time()

    total_kl_loss = 0
    total_ce_loss = 0
    total_loss = 0
    log_interval = 50
    data_idx = 0

    optimizer.zero_grad()

    for step in range(1, num_steps + 1):
        step_loss = 0
        step_kl = 0
        step_ce = 0

        for accum_step in range(gradient_accumulation):
            # Cycle through data
            input_ids = data[data_idx % len(data)]
            data_idx += 1
            debug = (step == 1 and accum_step == 0)  # print debug info on first iteration

            # Teacher forward (no grad)
            # Send input_ids to TEACHER's device
            with torch.no_grad():
                teacher_input = input_ids.to(teacher_device)
                teacher_out = teacher(teacher_input)
                # Extract logits and IMMEDIATELY go to CPU to escape
                # all accelerate dispatch hooks. .to(device) on hooked
                # tensors returns zeros — only CPU detach is safe.
                raw_logits = teacher_out.logits
                
                if debug:
                    print(f"\n  [DEBUG] teacher raw logits: dtype={raw_logits.dtype}, "
                          f"device={raw_logits.device}, "
                          f"min={raw_logits.min().item():.2f}, "
                          f"max={raw_logits.max().item():.2f}, "
                          f"has_nan={raw_logits.isnan().any().item()}, "
                          f"has_inf={raw_logits.isinf().any().item()}")

                # Escape hooks: detach → CPU → float32 → clamp
                teacher_logits = raw_logits.detach().cpu().float().clamp(-1e4, 1e4)
                del raw_logits, teacher_out, teacher_input

                if debug:
                    print(f"  [DEBUG] teacher on CPU: "
                          f"min={teacher_logits.min().item():.2f}, "
                          f"max={teacher_logits.max().item():.2f}, "
                          f"has_nan={teacher_logits.isnan().any().item()}")

                # Now move clean tensor to student device
                teacher_logits = teacher_logits.to(student_device)

                if debug:
                    print(f"  [DEBUG] teacher on {student_device}: "
                          f"min={teacher_logits.min().item():.2f}, "
                          f"max={teacher_logits.max().item():.2f}, "
                          f"has_nan={teacher_logits.isnan().any().item()}")

            # Student forward
            # Send input_ids to STUDENT's device
            student_input = input_ids.to(student_device)
            student_outputs = student_model(student_input)
            student_logits = student_outputs.logits

            if debug:
                print(f"  [DEBUG] student logits: dtype={student_logits.dtype}, "
                      f"device={student_logits.device}, "
                      f"min={student_logits.min().item():.2f}, "
                      f"max={student_logits.max().item():.2f}, "
                      f"has_nan={student_logits.isnan().any().item()}, "
                      f"has_inf={student_logits.isinf().any().item()}")

            # Compute loss (everything now on student_device)
            loss, kl_val, ce_val = distillation_loss(
                student_logits, teacher_logits, student_input,
                temperature=temperature, alpha=alpha,
                debug=debug,
            )

            if debug:
                print(f"  [DEBUG] loss={loss.item():.4f}, kl={kl_val:.4f}, ce={ce_val:.4f}, "
                      f"loss_nan={loss.isnan().item()}")

            # Free logits before backward
            del teacher_logits, student_logits, student_outputs

            # Scale loss for gradient accumulation
            scaled_loss = loss / gradient_accumulation
            scaled_loss.backward()

            del loss, scaled_loss

            step_loss += kl_val * alpha / gradient_accumulation + ce_val * (1 - alpha) / gradient_accumulation
            step_kl += kl_val / gradient_accumulation
            step_ce += ce_val / gradient_accumulation

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(train_params, max_norm=1.0)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += step_loss
        total_kl_loss += step_kl
        total_ce_loss += step_ce

        # Logging
        if step % log_interval == 0:
            avg_loss = total_loss / log_interval
            avg_kl = total_kl_loss / log_interval
            avg_ce = total_ce_loss / log_interval
            elapsed = time.time() - t0
            steps_per_sec = step / elapsed
            eta = (num_steps - step) / steps_per_sec if steps_per_sec > 0 else 0

            # GPU memory reporting
            mem_info = ""
            if torch.cuda.is_available():
                for gpu_id in range(min(num_gpus, 2)):
                    used = torch.cuda.memory_allocated(gpu_id) / 1e9
                    total = torch.cuda.get_device_properties(gpu_id).total_memory / 1e9
                    mem_info += f" | GPU{gpu_id}: {used:.1f}/{total:.0f}GB"

            print(
                f"  Step {step:>5d}/{num_steps} | "
                f"Loss: {avg_loss:.4f} | "
                f"KL: {avg_kl:.4f} | "
                f"CE: {avg_ce:.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                f"{steps_per_sec:.2f} it/s | "
                f"ETA: {eta / 60:.1f}min"
                f"{mem_info}"
            )

            total_loss = 0
            total_kl_loss = 0
            total_ce_loss = 0

        # Checkpoint
        if save_dir and save_every > 0 and step % save_every == 0:
            ckpt_dir = os.path.join(save_dir, f"checkpoint-{step}")
            print(f"  Saving checkpoint: {ckpt_dir}")
            os.makedirs(ckpt_dir, exist_ok=True)
            # Temporarily move to CPU + bf16 to save at correct size
            original_device = student_device
            original_dtype = next(student_model.parameters()).dtype
            student_model.to(dtype=torch.bfloat16, device="cpu")
            student_model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            # Move back for continued training
            student_model.to(dtype=original_dtype, device=original_device)
            student_model.train()
            if hasattr(student_model, "gradient_checkpointing_enable"):
                student_model.gradient_checkpointing_enable()

    # ── Final stats ───────────────────────────────────────────────────────
    total_time = time.time() - t0
    print(f"\nDistillation complete!")
    print(f"  Total time: {total_time / 60:.1f} min")
    print(f"  Avg speed: {num_steps / total_time:.2f} steps/s")

    # Clean up teacher to free memory
    del teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    student_model.eval()
