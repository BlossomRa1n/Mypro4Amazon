"""数据校验: 验证 Amazon Reviews 2023 数据文件是否就绪"""
import sys
import os
import pandas as pd

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(curr_path)

sys.path.insert(0, curr_path)
import config

print(f"{'='*50}")
print(f" 数据集: Amazon Reviews 2023")
categories = ', '.join(config.AMAZON_CATEGORIES)
print(f" 品类: {categories}")
print(f" 数据路径: {config.DATA_PATH}")
print(f"{'='*50}\n")

# 检查第一个品类文件即可
ratings_file = os.path.join(config.DATA_PATH, f'{config.AMAZON_CATEGORIES[0]}.csv')

print(f"正在读取: {ratings_file}")
try:
    df = pd.read_csv(ratings_file, nrows=5)
    print("[OK] Amazon benchmark CSV 读取成功！前5行:")
    print(df.to_string(index=False))
    total = sum(1 for _ in open(ratings_file, encoding='utf-8'))
    print(f"\n  总行数: {total:,} 条评分")
    if 'rating' in df.columns:
        ratings_sample = pd.read_csv(ratings_file, usecols=['rating'], nrows=100000)
        pos_ratio = (ratings_sample['rating'] >= 4).mean()
        print(f"  评分 >= 4 (正反馈) 比例: {pos_ratio:.1%} (基于前100k条)")
except FileNotFoundError:
    print(f"[ERROR] 找不到 {ratings_file}")
    print(f"从以下地址下载 (hf-mirror.com 国内镜像):")
    for cat in config.AMAZON_CATEGORIES:
        print(f"  https://hf-mirror.com/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/benchmark/5core/rating_only/{cat}.csv")
    print(f"全部放到: {config.DATA_PATH}/")
