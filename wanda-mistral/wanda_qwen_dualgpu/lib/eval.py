"""
Evaluation Utilities
====================

Perplexity evaluation on standard benchmarks.
"""

import torch
import torch.nn as nn
from tqdm import tqdm

from .data import get_test_data


@torch.no_grad()
def eval_ppl(
    model,
    tokenizer,
    dataset: str = "wikitext2",
    seqlen: int = 2048,
    device: str = None,
) -> float:
    """
    Evaluate perplexity on a test dataset.

    Args:
        model: The (possibly pruned) model
        tokenizer: Tokenizer
        dataset: One of "wikitext2", "c4", "ptb"
        seqlen: Sequence length for evaluation chunks
        device: Device to use (auto-detected if None)

    Returns:
        Perplexity (float)
    """
    if device is None:
        device = next(model.parameters()).device

    print(f"\nEvaluating perplexity on {dataset} (device={device})...")

    test_ids = get_test_data(dataset, tokenizer, seqlen)

    # Split into chunks of seqlen
    n_chunks = test_ids.shape[1] // seqlen
    test_ids = test_ids[:, : n_chunks * seqlen].view(n_chunks, seqlen)

    total_nll = 0.0
    total_tokens = 0

    model.eval()

    for i in tqdm(range(n_chunks), desc=f"PPL eval ({dataset})"):
        input_ids = test_ids[i : i + 1].to(device)

        outputs = model(input_ids)
        logits = outputs.logits.float()  # float32 for stable CE computation

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        loss_fn = nn.CrossEntropyLoss(reduction="sum")
        nll = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        total_nll += nll.item()
        total_tokens += shift_labels.numel()

    avg_nll = total_nll / total_tokens
    ppl = torch.exp(torch.tensor(avg_nll)).item()

    print(f"  {dataset} perplexity: {ppl:.2f}")
    return ppl
