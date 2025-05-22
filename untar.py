import os
import tarfile

def extract_tar_gz_files(root_dir='.'):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.tar.gz'):
                full_path = os.path.join(dirpath, filename)
                extract_dir = './'  # 去掉 .tar.gz
                print(f"正在解压：{full_path} 到 {extract_dir}")
                try:
                    with tarfile.open(full_path, 'r:gz') as tar:
                        tar.extractall(path=extract_dir, filter='data')
                except Exception as e:
                    print(f"解压失败：{full_path}，错误：{e}")

if __name__ == "__main__":
    extract_tar_gz_files()