#!/bin/bash

# Set default values
BATCH_SIZE=1
MODEL_PATH="/cephfs/ylshare/zihan/4D-LLM/output/sft_stage2/checkpoint-10349"
OUTPUT_DIR="./src/eval_results/"
DATASET_NAME="vsibench"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --batch_size=*)
      BATCH_SIZE="${1#*=}"
      shift
      ;;
    --model_path=*)
      MODEL_PATH="${1#*=}"
      shift
      ;;
    --output_dir=*)
      OUTPUT_DIR="${1#*=}"
      shift
      ;;
    --dataset_name=*)
      DATASET_NAME="${1#*=}"
      shift
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

# 仅在主进程打印日志
export ACCELERATE_LOG_LEVEL=CRITICAL
export PYTHONIOENCODING=utf-8

# Run evaluation with Accelerate
accelerate launch \
  --multi_gpu \
  --num_processes=$(nvidia-smi --list-gpus | wc -l) \
  evaluate_vsi.py \
  --batch_size $BATCH_SIZE \
  --model_path "$MODEL_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --dataset_name "$DATASET_NAME"

echo "Evaluation complete!" 