"""
Data Loading for Wanda Calibration
===================================

Provides calibration data loaders. Supports C4, WikiText-2,
and custom text datasets.
"""

import torch
from datasets import load_dataset
import random


def get_loaders(
    dataset_name: str,
    nsamples: int = 128,
    seed: int = 0,
    seqlen: int = 2048,
    tokenizer=None,
):
    """
    Get calibration data as a list of tokenized sequences.

    Args:
        dataset_name: One of "c4", "wikitext2", "ptb", or a path to a text file
        nsamples: Number of calibration samples
        seed: Random seed
        seqlen: Sequence length for each sample
        tokenizer: HuggingFace tokenizer

    Returns:
        List of input_ids tensors, each of shape (1, seqlen)
    """
    random.seed(seed)
    torch.manual_seed(seed)

    if dataset_name == "c4":
        return _get_c4(nsamples, seed, seqlen, tokenizer)
    elif dataset_name == "wikitext2":
        return _get_wikitext2(nsamples, seed, seqlen, tokenizer)
    elif dataset_name == "ptb":
        return _get_ptb(nsamples, seed, seqlen, tokenizer)
    else:
        return _get_custom(dataset_name, nsamples, seed, seqlen, tokenizer)


def _get_c4(nsamples, seed, seqlen, tokenizer):
    """Load calibration data from C4 (allenai/c4)."""
    dataset = load_dataset(
        "allenai/c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
    )

    random.seed(seed)
    samples = []
    for _ in range(nsamples):
        while True:
            idx = random.randint(0, len(dataset) - 1)
            text = dataset[idx]["text"]
            enc = tokenizer(text, return_tensors="pt")
            if enc.input_ids.shape[1] >= seqlen:
                break
        start = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        samples.append(enc.input_ids[:, start : start + seqlen])

    return samples


def _get_wikitext2(nsamples, seed, seqlen, tokenizer):
    """Load calibration data from WikiText-2."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    # Concatenate all text into one long string
    text = "\n\n".join(dataset["text"])
    enc = tokenizer(text, return_tensors="pt")

    random.seed(seed)
    samples = []
    for _ in range(nsamples):
        start = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        samples.append(enc.input_ids[:, start : start + seqlen])

    return samples


def _get_ptb(nsamples, seed, seqlen, tokenizer):
    """Load calibration data from Penn Treebank."""
    dataset = load_dataset("ptb_text_only", "penn_treebank", split="validation")

    text = "\n\n".join(dataset["sentence"])
    enc = tokenizer(text, return_tensors="pt")

    random.seed(seed)
    samples = []
    for _ in range(nsamples):
        start = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        samples.append(enc.input_ids[:, start : start + seqlen])

    return samples


def _get_custom(path, nsamples, seed, seqlen, tokenizer):
    """Load calibration data from a custom text file (one doc per line)."""
    with open(path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    random.seed(seed)
    random.shuffle(lines)

    samples = []
    for line in lines:
        if len(samples) >= nsamples:
            break
        enc = tokenizer(line, return_tensors="pt")
        if enc.input_ids.shape[1] >= seqlen:
            start = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
            samples.append(enc.input_ids[:, start : start + seqlen])

    if len(samples) < nsamples:
        print(
            f"Warning: Only found {len(samples)}/{nsamples} samples "
            f"with seqlen >= {seqlen} in {path}"
        )

    return samples


def get_test_data(dataset_name: str, tokenizer, seqlen: int = 2048):
    """
    Get test data for perplexity evaluation.

    Returns:
        Tokenized test set as a single long tensor
    """
    if dataset_name == "wikitext2":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(dataset["text"])
    elif dataset_name == "c4":
        dataset = load_dataset(
            "allenai/c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
        )
        text = "\n\n".join(dataset["text"][:1100])
    elif dataset_name == "ptb":
        dataset = load_dataset("ptb_text_only", "penn_treebank", split="test")
        text = "\n\n".join(dataset["sentence"])
    else:
        raise ValueError(f"Unknown test dataset: {dataset_name}")

    enc = tokenizer(text, return_tensors="pt")
    return enc.input_ids
