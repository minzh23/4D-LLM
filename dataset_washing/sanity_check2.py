import os
import subprocess

def is_video_corrupted(filepath):
    try:
        result = subprocess.run(
            ['ffmpeg', '-v', 'error', '-i', filepath, '-f', 'null', '-'],
            stderr=subprocess.PIPE,
            stdout=subprocess.DEVNULL
        )
        return result.stderr.decode() != ''
    except Exception as e:
        print(f"\n检测 {filepath} 时出错：{e}")
        return True

def check_videos_in_directory(directory, video_extensions=('.mp4', '.avi', '.mov', '.mkv')):
    corrupted_files = []
    total = 0
    checked = 0

    # 先统计总数
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(video_extensions):
                total += 1

    print(f"开始检测，共发现 {total} 个视频文件。\n")

    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(video_extensions):
                checked += 1
                filepath = os.path.join(root, file)
                print(f"({checked}/{total}) 正在检测：{filepath}")
                if is_video_corrupted(filepath):
                    print(f"⚠️  损坏视频：{filepath}")
                    corrupted_files.append(filepath)

    print(f"\n✅ 检测完成，共发现 {len(corrupted_files)} 个损坏文件。")
    return corrupted_files

# 用法
corrupted = check_videos_in_directory('./Video-R1/LLaVA-Video-178K/')

output_file = 'corrupted_videos.txt'
with open(output_file, 'w', encoding='utf-8') as f:
    for path in corrupted:
        f.write(path + '\n')

print(f"\n📄 损坏视频路径已保存到：{output_file}")