import subprocess
import csv

def get_video_duration(filepath):
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', filepath],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        duration = result.stdout.decode().strip()
        return float(duration) if duration else None
    except Exception as e:
        print(f"读取失败：{filepath}，错误：{e}")
        return None

def process_video_list(txt_file, output_csv='video_durations.csv'):
    results = []
    with open(txt_file, 'r', encoding='utf-8') as f:
        video_paths = [line.strip() for line in f if line.strip()]

    for path in video_paths:
        duration = get_video_duration(path)
        if duration is not None:
            print(f"{path} -> {duration:.2f} 秒")
            results.append([path, duration])
        else:
            print(f"⚠️ 无法读取：{path}")
            results.append([path, "ERROR"])

    # 保存为 CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['video_path', 'duration_seconds'])
        writer.writerows(results)

    print(f"\n✅ 已保存结果到 {output_csv}")

# 用法
process_video_list('corrupted_videos.txt')