"""
Data loading utilities for REAP calibration.

This module provides dataset loading and preprocessing for computing
expert saliency scores on GPT-OSS models.
"""

from typing import Optional, Iterator, Dict, Any
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import AutoTokenizer, PreTrainedTokenizer


@dataclass
class DataConfig:
    """Configuration for calibration data loading."""
    dataset_name: str = "theblackcat102/evol-codealpaca-v1"
    dataset_split: str = "train"
    text_column: str = "text"
    max_length: int = 2048
    batch_size: int = 4
    num_workers: int = 0
    streaming: bool = True
    seed: int = 42


class CalibrationDataset(IterableDataset):
    """
    Iterable dataset for calibration data.
    
    Supports HuggingFace datasets with streaming for memory efficiency.
    """
    
    def __init__(
        self,
        dataset_name: str,
        tokenizer: PreTrainedTokenizer,
        text_column: str = "text",
        max_length: int = 2048,
        split: str = "train",
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.max_length = max_length
        
        # Load dataset with streaming
        try:
            from datasets import load_dataset
            self.dataset = load_dataset(
                dataset_name,
                split=split,
                streaming=True,
            )
            self.dataset = self.dataset.shuffle(seed=seed)
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset {dataset_name}: {e}")
            
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        for example in self.dataset:
            text = example.get(self.text_column, "")
            if not text:
                continue
                
            # Tokenize
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            
            yield {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
            }


class CodeCalibrationDataset(CalibrationDataset):
    """
    Calibration dataset optimized for code-focused models like GPT-OSS.
    
    Uses code-specific datasets and formatting.
    """
    
    # Recommended datasets for code calibration
    RECOMMENDED_DATASETS = [
        "theblackcat102/evol-codealpaca-v1",
        "bigcode/the-stack-dedup",
        "codeparrot/github-code",
    ]
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        dataset_name: Optional[str] = None,
        max_length: int = 2048,
        seed: int = 42,
    ):
        if dataset_name is None:
            dataset_name = self.RECOMMENDED_DATASETS[0]
            
        # Dataset-specific column names
        text_columns = {
            "theblackcat102/evol-codealpaca-v1": "text",
            "bigcode/the-stack-dedup": "content",
            "codeparrot/github-code": "code",
        }
        text_column = text_columns.get(dataset_name, "text")
        
        super().__init__(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            text_column=text_column,
            max_length=max_length,
            seed=seed,
        )


def create_calibration_dataloader(
    tokenizer: PreTrainedTokenizer,
    config: Optional[DataConfig] = None,
) -> DataLoader:
    """
    Create a DataLoader for calibration.
    
    Args:
        tokenizer: Tokenizer for the model
        config: Data configuration (uses defaults if None)
        
    Returns:
        DataLoader yielding batches of tokenized samples
    """
    if config is None:
        config = DataConfig()
        
    dataset = CalibrationDataset(
        dataset_name=config.dataset_name,
        tokenizer=tokenizer,
        text_column=config.text_column,
        max_length=config.max_length,
        split=config.dataset_split,
        seed=config.seed,
    )
    
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )


def create_gptoss_calibration_loader(
    model_name: str = "openai/gpt-oss-20b",
    dataset_name: str = "theblackcat102/evol-codealpaca-v1",
    batch_size: int = 4,
    max_length: int = 2048,
    seed: int = 42,
) -> DataLoader:
    """
    Convenience function to create calibration loader for GPT-OSS.
    
    Args:
        model_name: HuggingFace model identifier
        dataset_name: Calibration dataset name
        batch_size: Batch size for calibration
        max_length: Maximum sequence length
        seed: Random seed
        
    Returns:
        Configured DataLoader
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # GPT-OSS may need padding token set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    config = DataConfig(
        dataset_name=dataset_name,
        batch_size=batch_size,
        max_length=max_length,
        seed=seed,
    )
    
    return create_calibration_dataloader(tokenizer, config)


# Pre-defined calibration configs for different use cases
CALIBRATION_PRESETS = {
    "code": DataConfig(
        dataset_name="theblackcat102/evol-codealpaca-v1",
        text_column="text",
        max_length=2048,
    ),
    "general": DataConfig(
        dataset_name="wikitext",
        dataset_split="train",
        text_column="text",
        max_length=1024,
    ),
    "math": DataConfig(
        dataset_name="competition_math",
        text_column="problem",
        max_length=512,
    ),
    "instruction": DataConfig(
        dataset_name="tatsu-lab/alpaca",
        text_column="text",
        max_length=1024,
    ),
}


def get_preset_dataloader(
    preset: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 4,
) -> DataLoader:
    """
    Get a pre-configured DataLoader for common calibration scenarios.
    
    Args:
        preset: One of "code", "general", "math", "instruction"
        tokenizer: Model tokenizer
        batch_size: Batch size
        
    Returns:
        Configured DataLoader
    """
    if preset not in CALIBRATION_PRESETS:
        raise ValueError(
            f"Unknown preset '{preset}'. "
            f"Available: {list(CALIBRATION_PRESETS.keys())}"
        )
        
    config = CALIBRATION_PRESETS[preset]
    config.batch_size = batch_size
    
    return create_calibration_dataloader(tokenizer, config)
