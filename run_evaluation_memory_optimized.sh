#!/bin/bash

# Memory optimized evaluation script for VSI Benchmark
# This script uses optimized parameters to minimize GPU memory usage

# Set memory optimization environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=1
export TOKENIZERS_PARALLELISM=false

# Model and data paths
MODEL_PATH="/cephfs/ylshare/zihan/4D-LLM/output/sft_stage2/checkpoint-10349"
OUTPUT_DIR="./src/eval_results/"
DATASET_NAME="vsibench"
DEEPSPEED_CONFIG="./deepspeed_inference_config.json"

# Memory optimized parameters
BATCH_SIZE=1  # Start with batch size 1
MAX_FRAMES=8  # Limit video frames
USE_DEEPSPEED=true
CPU_OFFLOAD=true

echo "Starting memory-optimized VSI evaluation..."
echo "Model: $MODEL_PATH"
echo "Output directory: $OUTPUT_DIR" 
echo "Dataset: $DATASET_NAME"
echo "DeepSpeed config: $DEEPSPEED_CONFIG"
echo "Batch size: $BATCH_SIZE"
echo "Max frames per video: $MAX_FRAMES"

# Check if DeepSpeed config exists
if [ ! -f "$DEEPSPEED_CONFIG" ]; then
    echo "Warning: DeepSpeed config file not found at $DEEPSPEED_CONFIG"
    echo "Using default DeepSpeed configuration..."
    DEEPSPEED_ARG=""
else
    echo "Using custom DeepSpeed configuration: $DEEPSPEED_CONFIG"
    DEEPSPEED_ARG="--deepspeed_config $DEEPSPEED_CONFIG"
fi

# Run the evaluation with optimized settings
python evaluate_vsi.py \
    --batch_size $BATCH_SIZE \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --dataset_name "$DATASET_NAME" \
    --use_deepspeed \
    --cpu_offload \
    --max_frames_per_video $MAX_FRAMES \
    $DEEPSPEED_ARG

echo "Evaluation completed!" 