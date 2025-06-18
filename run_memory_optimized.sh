#!/bin/bash

# 设置内存优化环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=1
export TOKENIZERS_PARALLELISM=false

# 清理GPU内存
nvidia-smi --gpu-reset || true

echo "Starting memory-optimized evaluation..."
echo "GPU Memory before start:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits

# 运行评估，批处理大小为1
python evaluate_vsi.py --batch_size 1

echo "Evaluation completed!"
echo "GPU Memory after completion:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 