#!/bin/bash

# Wanda Pruning Script for Mistral 7B
# Usage: ./scripts/mistral_7b.sh

# Available Mistral models:
# - mistralai/Mistral-7B-v0.1 (base)
# - mistralai/Mistral-7B-v0.3 (latest base)
# - mistralai/Mistral-7B-Instruct-v0.1
# - mistralai/Mistral-7B-Instruct-v0.2
# - mistralai/Mistral-7B-Instruct-v0.3

MODEL="mistralai/Mistral-7B-v0.1"

# Unstructured 50% sparsity with Wanda
echo "Running Wanda pruning on ${MODEL} (50% unstructured sparsity)"
python main.py \
    --model ${MODEL} \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save out/mistral_7b/unstructured/wanda/

# 2:4 structured sparsity (for NVIDIA sparse tensor cores)
echo "Running Wanda pruning on ${MODEL} (2:4 structured sparsity)"
python main.py \
    --model ${MODEL} \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type 2:4 \
    --save out/mistral_7b/2-4/wanda/

# 4:8 structured sparsity
echo "Running Wanda pruning on ${MODEL} (4:8 structured sparsity)"
python main.py \
    --model ${MODEL} \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type 4:8 \
    --save out/mistral_7b/4-8/wanda/

# Magnitude pruning baseline
echo "Running Magnitude pruning baseline"
python main.py \
    --model ${MODEL} \
    --prune_method magnitude \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save out/mistral_7b/unstructured/magnitude/

# SparseGPT (slower but potentially better at high sparsity)
echo "Running SparseGPT pruning on ${MODEL}"
python main.py \
    --model ${MODEL} \
    --prune_method sparsegpt \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save out/mistral_7b/unstructured/sparsegpt/

echo "All pruning experiments complete!"
