import os
import shutil

corrupted_list = 'corrupted_videos.txt'
fixed_root = 'fixed_videos'  # 修复后视频的根目录

def replace_video(original_path):
    rel_path = os.path.relpath(original_path)
    fixed_path = os.path.join(fixed_root, rel_path)

    if not os.path.exists(fixed_path):
        print(f"❌ 修复文件不存在：{fixed_path}")
        return

    if os.path.exists(original_path):
        try:
            os.remove(original_path)
            print(f"🗑️ 已删除：{original_path}")
        except Exception as e:
            print(f"⚠️ 删除失败：{original_path} -> {e}")
            return

    # 确保目标目录存在
    os.makedirs(os.path.dirname(original_path), exist_ok=True)

    try:
        shutil.copy2(fixed_path, original_path)
        print(f"✅ 已替换：{original_path}")
    except Exception as e:
        print(f"❌ 替换失败：{original_path} -> {e}")

def main():
    if not os.path.exists(corrupted_list):
        print(f"未找到文件：{corrupted_list}")
        return

    with open(corrupted_list, 'r', encoding='utf-8') as f:
        video_paths = [line.strip() for line in f if line.strip()]

    for path in video_paths:
        replace_video(path)

if __name__ == "__main__":
    main()
