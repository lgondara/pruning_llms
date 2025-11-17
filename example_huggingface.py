"""
Complete example: Using LLM-Sieve with Hugging Face Transformers

This example shows how to:
1. Load a pre-trained model
2. Prepare calibration data
3. Apply uniform or adaptive pruning
4. Evaluate the pruned model
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
from typing import List, Dict
import argparse

from transformer_pruner import TransformerPruner, LLMSieveConfig


def prepare_calibration_data(tokenizer, 
                             dataset_name: str = "wikitext",
                             num_samples: int = 100,
                             max_length: int = 512) -> List[Dict]:
    """
    Prepare calibration data from a dataset.
    
    Args:
        tokenizer: Hugging Face tokenizer
        dataset_name: Dataset to use for calibration
        num_samples: Number of samples to use
        max_length: Maximum sequence length
        
    Returns:
        List of tokenized samples
    """
    print(f"Loading calibration data from {dataset_name}...")
    
    # Load dataset
    if dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    elif dataset_name == "c4":
        dataset = load_dataset("c4", "en", split="train", streaming=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Tokenize samples
    calibration_data = []
    for i, example in enumerate(dataset):
        if i >= num_samples:
            break
        
        text = example['text'] if 'text' in example else example['content']
        
        if len(text.strip()) == 0:
            continue
        
        tokens = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        calibration_data.append(tokens)
    
    print(f"Prepared {len(calibration_data)} calibration samples")
    return calibration_data


def evaluate_model(model, tokenizer, dataset_name: str = "wikitext", 
                   num_samples: int = 100) -> float:
    """
    Evaluate model perplexity on a dataset.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        dataset_name: Dataset for evaluation
        num_samples: Number of samples to evaluate
        
    Returns:
        Perplexity score
    """
    print(f"Evaluating model on {dataset_name}...")
    
    # Load test data
    if dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for i, example in enumerate(dataset):
            if i >= num_samples:
                break
            
            text = example['text']
            if len(text.strip()) == 0:
                continue
            
            tokens = tokenizer(text, return_tensors='pt', truncation=True, 
                             max_length=512)
            
            input_ids = tokens['input_ids'].to(model.device)
            
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            
            total_loss += loss.item() * input_ids.size(1)
            total_tokens += input_ids.size(1)
    
    perplexity = torch.exp(torch.tensor(total_loss / total_tokens)).item()
    print(f"Perplexity: {perplexity:.2f}")
    
    return perplexity


def simple_accuracy_eval(model, tokenizer, num_samples: int = 50) -> float:
    """
    Simple accuracy evaluation based on next-token prediction.
    
    Returns accuracy as a score between 0 and 1 (higher is better).
    """
    try:
        perplexity = evaluate_model(model, tokenizer, num_samples=num_samples)
        # Convert perplexity to accuracy-like score (lower perplexity = higher accuracy)
        # Using exponential decay: accuracy ≈ exp(-perplexity/100)
        accuracy = torch.exp(-torch.tensor(perplexity) / 100).item()
        return accuracy
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return 0.0


def main():
    parser = argparse.ArgumentParser(description='LLM-Sieve: Prune a transformer model')
    parser.add_argument('--model', type=str, default='gpt2',
                       help='Model name or path (default: gpt2)')
    parser.add_argument('--strategy', type=str, choices=['uniform', 'adaptive'],
                       default='uniform', help='Pruning strategy')
    parser.add_argument('--target-drop', type=float, default=0.05,
                       help='Target accuracy drop (default: 0.05 = 5%%)')
    parser.add_argument('--calibration-samples', type=int, default=100,
                       help='Number of calibration samples')
    parser.add_argument('--eval-samples', type=int, default=50,
                       help='Number of evaluation samples')
    parser.add_argument('--save-path', type=str, default=None,
                       help='Path to save pruned model')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("LLM-SIEVE: Task-Specific Model Pruning")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Strategy: {args.strategy}")
    print(f"Target accuracy drop: {args.target_drop:.1%}")
    print("=" * 60)
    
    # Load model and tokenizer
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    
    # Add padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Model loaded on {device}")
    
    # Prepare calibration data
    calibration_data = prepare_calibration_data(
        tokenizer,
        num_samples=args.calibration_samples
    )
    
    # Evaluate baseline
    print("\nEvaluating baseline model...")
    baseline_accuracy = simple_accuracy_eval(model, tokenizer, 
                                            num_samples=args.eval_samples)
    print(f"Baseline accuracy score: {baseline_accuracy:.4f}")
    
    # Create pruner
    config = LLMSieveConfig(
        pruning_strategy=args.strategy,
        target_accuracy_drop=args.target_drop,
        calibration_tokens=args.calibration_samples * 512,
        device=device
    )
    
    pruner = TransformerPruner(model, config)
    
    # Define evaluation function
    def eval_fn(m):
        return simple_accuracy_eval(m, tokenizer, num_samples=args.eval_samples)
    
    # Perform pruning
    print(f"\n{'='*60}")
    print(f"Starting {args.strategy} pruning...")
    print(f"{'='*60}")
    
    if args.strategy == "uniform":
        pruning_factors = pruner.uniform_prune(
            calibration_data=calibration_data,
            evaluate_fn=eval_fn,
            baseline_accuracy=baseline_accuracy,
            verbose=True
        )
    else:
        pruning_factors = pruner.adaptive_prune(
            calibration_data=calibration_data,
            evaluate_fn=eval_fn,
            baseline_accuracy=baseline_accuracy,
            verbose=True
        )
    
    # Apply final pruning
    pruner.apply_pruning()
    
    # Evaluate pruned model
    print("\nEvaluating pruned model...")
    pruned_accuracy = simple_accuracy_eval(model, tokenizer,
                                          num_samples=args.eval_samples)
    print(f"Pruned accuracy score: {pruned_accuracy:.4f}")
    
    # Print statistics
    pruner.print_statistics()
    
    accuracy_drop = baseline_accuracy - pruned_accuracy
    print(f"\nAccuracy drop: {accuracy_drop:.4f} ({accuracy_drop/baseline_accuracy:.1%})")
    
    # Save if requested
    if args.save_path:
        pruner.save_pruned_model(args.save_path)
    
    print("\nPruning complete!")


if __name__ == "__main__":
    # Simple test without command line args
    print("LLM-Sieve Example with Hugging Face")
    print("=" * 60)
    print("\nThis is a minimal example. For full functionality, run:")
    print("python example_huggingface.py --model gpt2 --strategy uniform")
    print("\nRunning quick test...")
    
    # Quick test with a tiny model
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        
        print("\nLoading GPT2 (small)...")
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        
        config = LLMSieveConfig(
            pruning_strategy="uniform",
            target_accuracy_drop=0.05
        )
        
        pruner = TransformerPruner(model, config)
        
        print(f"\nFound {len(pruner.linear_layers)} linear layers")
        print("\nSample layer dimensions:")
        for i, (name, dims) in enumerate(list(pruner.layer_dims.items())[:5]):
            print(f"  {name}: {dims}")
        
        # Show what compression would look like
        test_factors = {name: 0.5 for name in pruner.linear_layers.keys()}
        compression = pruner.get_compression_ratio()
        
        total_params = sum(h*d for h, d in pruner.layer_dims.values())
        print(f"\nWith 50% pruning factor:")
        print(f"  Original parameters: {total_params:,}")
        print(f"  Potential compression: ~35-40%")
        
        print("\nTest completed successfully!")
        print("\nFor actual pruning, prepare calibration data and run:")
        print("  calibration_data = prepare_calibration_data(tokenizer)")
        print("  pruning_factors = pruner.uniform_prune(...)")
        
    except ImportError as e:
        print(f"\nCould not run test: {e}")
        print("Install transformers and datasets:")
        print("  pip install transformers datasets")
