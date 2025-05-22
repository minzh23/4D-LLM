#!/bin/bash

# 设置镜像地址前缀（你可以根据你用的是 Hugging Face 官方还是 hf-mirror 修改）
BASE_URL="https://hf-mirror.com/datasets/lmms-lab/LLaVA-Video-178K/resolve/main/0_30_s_youtube_v0_1"

# 循环下载第1到38个 part
for i in $(seq 1 20); do
    FILENAME="0_30_s_youtube_v0_1_videos_${i}.tar.gz"
    echo "⏬ Downloading $FILENAME ..."
    wget "${BASE_URL}/${FILENAME}" -c
done

echo "✅ All parts downloaded."