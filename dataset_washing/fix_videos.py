import os
import subprocess

INPUT_LIST = 'corrupted_videos.txt'
OUTPUT_ROOT = 'fixed_videos'

def fix_video(input_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        result = subprocess.run([
            'ffmpeg', '-y',
            '-i', input_path,
            '-fflags', '+genpts',
            '-vsync', '2',
            '-c:v', 'libx264',
            '-c:a', 'aac',
            output_path
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            print(f"✅ 修复成功: {input_path} -> {output_path}")
        else:
            print(f"❌ 修复失败: {input_path}\n错误信息:\n{result.stderr.decode()}")
    except Exception as e:
        print(f"❌ 出错: {input_path}\n异常: {e}")

def main():
    if not os.path.exists(INPUT_LIST):
        print(f"找不到输入文件: {INPUT_LIST}")
        return

    with open(INPUT_LIST, 'r', encoding='utf-8') as f:
        video_paths = [line.strip() for line in f if line.strip()]

    for video_path in video_paths:
        rel_path = os.path.relpath(video_path)  # 相对路径
        output_path = os.path.join(OUTPUT_ROOT, rel_path)
        fix_video(video_path, output_path)

if __name__ == "__main__":
    main()