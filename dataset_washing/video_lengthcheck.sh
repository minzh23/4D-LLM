#!/bin/bash

VIDEO_DIR="/cephfs/shared/yicheng/4D-LLM/Video-R1/LLaVA-Video-178K/liwei_youtube_videos/videos/youtube_video_2024"

if ! command -v ffprobe &> /dev/null; then
    echo "ffprobe 未安装，请先安装 ffmpeg。"
    exit 1
fi

max_duration=0
max_file=""

# 遍历视频文件
find "$VIDEO_DIR" -type f \( -iname "*.mp4" -o -iname "*.avi" -o -iname "*.mkv" -o -iname "*.mov" \) | while read file; do
    duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$file")
    duration_int=$(printf "%.0f" "$duration")

    if (( duration_int > max_duration )); then
        max_duration=$duration_int
        max_file="$file"
    fi
done

# 输出最长视频信息
echo "最长视频: $max_file"
echo "时长（秒）: $max_duration"