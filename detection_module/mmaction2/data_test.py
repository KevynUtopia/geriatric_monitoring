#!/usr/bin/env python3
import os
from PIL import Image
from PIL import ImageFile
from tqdm import tqdm
import argparse

# 允许加载部分损坏的图片以便检测
ImageFile.LOAD_TRUNCATED_IMAGES = True


def check_image_integrity(filepath):
    """检查单个图片文件的完整性"""
    try:
        image =  Image.open(filepath)
        return True
    except (IOError, SyntaxError) as e:
        print(e)
        return False


def scan_folders_for_images(root_folder, image_extensions=('jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff')):
    """扫描文件夹中的图片文件"""
    corrupted_folders = set()

    # 获取所有子文件夹
    # folders = [root for root, _, _ in os.walk(root_folder)]
    folders = os.listdir(root_folder)
    folders.sort()  # 顺序
    # folders.sort(reverse=True)  # 逆序

    # 使用tqdm显示总进度
    for folder in folders:
        # 检查该文件夹下的所有图片文件
        for filename in tqdm(os.listdir(os.path.join(root_folder, folder)), desc=f"Scanning {folder}"):
            if filename.lower().endswith(image_extensions):
                filepath = os.path.join(root_folder, folder, filename)
                if not check_image_integrity(filepath):
                    corrupted_folders.add(folder)
                    print(folder)
                    break


    return corrupted_folders


def main():
    parser = argparse.ArgumentParser(description='检测文件夹中的损坏图片')
    parser.add_argument('--folder', help='要扫描的根文件夹路径')
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"错误: {args.folder} 不是一个有效的文件夹")
        return

    print(f"开始扫描 {args.folder} 中的图片文件...")
    corrupted = scan_folders_for_images(args.folder)

    print("\n检测到包含损坏图片的文件夹:")
    for folder in corrupted:
        print(folder)


if __name__ == "__main__":
    main()