#!/bin/bash
# REAP Pruning Script for GPT-OSS Models
# 
# Usage:
#   ./scripts/prune_gptoss.sh <CUDA_DEVICES> [MODEL_NAME] [COMPRESSION_RATIO] [METHOD]
#
# Examples:
#   ./scripts/prune_gptoss.sh 0                                    # Default: GPT-OSS 20B, 50% REAP
#   ./scripts/prune_gptoss.sh 0,1 openai/gpt-oss-120b 0.3 reap    # GPT-OSS 120B, 30% REAP
#   ./scripts/prune_gptoss.sh 0 openai/gpt-oss-20b 0.5 frequency  # Compare with frequency

set -e

# Parse arguments
CUDA_DEVICES="${1:-0}"
MODEL_NAME="${2:-openai/gpt-oss-20b}"
COMPRESSION_RATIO="${3:-0.5}"
METHOD="${4:-reap}"
SEED="${5:-42}"
DATASET="${6:-theblackcat102/evol-codealpaca-v1}"
CALIBRATION_SAMPLES="${7:-512}"

# Derive output directory name
MODEL_SHORT=$(echo "$MODEL_NAME" | sed 's/.*\///')
COMPRESSION_PCT=$(echo "$COMPRESSION_RATIO * 100" | bc | cut -d. -f1)
OUTPUT_DIR="./outputs/${MODEL_SHORT}-${METHOD}-${COMPRESSION_PCT}pct"

echo "============================================================"
echo "REAP Pruning for GPT-OSS"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  CUDA Devices:       $CUDA_DEVICES"
echo "  Model:              $MODEL_NAME"
echo "  Compression Ratio:  $COMPRESSION_RATIO (${COMPRESSION_PCT}%)"
echo "  Method:             $METHOD"
echo "  Calibration Data:   $DATASET"
echo "  Samples:            $CALIBRATION_SAMPLES"
echo "  Seed:               $SEED"
echo "  Output Directory:   $OUTPUT_DIR"
echo ""

# Set CUDA devices
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"

# Run pruning
echo "Starting pruning..."
python -m reap_gptoss.main \
    --model_name "$MODEL_NAME" \
    --compression_ratio "$COMPRESSION_RATIO" \
    --method "$METHOD" \
    --dataset_name "$DATASET" \
    --calibration_samples "$CALIBRATION_SAMPLES" \
    --seed "$SEED" \
    --output_dir "$OUTPUT_DIR" \
    --save_saliency \
    --use_bf16

echo ""
echo "============================================================"
echo "Pruning complete!"
echo "Output saved to: $OUTPUT_DIR"
echo "============================================================"

# Print model info
if [ -f "$OUTPUT_DIR/pruning_config.json" ]; then
    echo ""
    echo "Pruning results:"
    cat "$OUTPUT_DIR/pruning_config.json"
fi
