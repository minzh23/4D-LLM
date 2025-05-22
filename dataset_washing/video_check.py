import os
import subprocess
from pathlib import Path
from tqdm import tqdm

# 设置视频目录
VIDEO_DIR = "/cephfs/shared/yicheng/4D-LLM/Video-R1/LLaVA-Video-178K/liwei_youtube_videos/videos/youtube_video_2024"

# 支持的视频格式
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov"}

def get_video_duration(file_path):
    try:
        # 调用 ffprobe 获取时长
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(file_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return float(result.stdout.strip())
    except Exception as e:
        print(f"无法处理文件 {file_path}: {e}")
        return 0.0

def find_longest_video(directory):
    video_files = [f for f in Path(directory).rglob("*") if f.suffix.lower() in VIDEO_EXTENSIONS]

    max_duration = 0
    max_file = None

    for video in tqdm(video_files, desc="正在处理视频"):
        duration = get_video_duration(video)
        if duration > max_duration:
            max_duration = duration
            max_file = video

    return max_file, max_duration

if __name__ == "__main__":
    longest_video, duration = find_longest_video(VIDEO_DIR)
    if longest_video:
        print("\n最长视频:")
        print(f"路径: {longest_video}")
        print(f"时长: {int(duration // 60)}分{int(duration % 60)}秒 ({duration:.2f} 秒)")
    else:
        print("未找到视频文件。")
