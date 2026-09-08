"""
HSTU 全库 HR@100 评估 — 对比 V2 基线 (8.66%) / ItemCF (10.62%) / 四路并集 (16.89%)。

采样 10 万 val 用户, 用 HSTU 双塔 (user_tower_type='hstu') 做 top-100 向量召回, 算命中率。
复用 recall_fusion.embedding_recall (已支持 user_tower_type)。
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import pickle, time
import numpy as np

import config
from data_loader import get_all_click_df, get_item_topk_click
from data_loader_ext import (load_raw_meta, build_extended_encoders,
                             build_extended_item_features,
                             build_extended_user_features, merge_jsonl_features)
from evaluate import split_train_val
from recall_fusion import embedding_recall

SAMPLE_USERS = 100000


def main():
    t0 = time.time()
    print(">>> Loading data...")
    click_df = get_all_click_df(config.DATA_PATH, offline=config.OFFLINE_MODE)
    raw_meta = load_raw_meta(config.DATA_PATH, config.AMAZON_CATEGORIES)
    click_df = merge_jsonl_features(click_df, config.DATA_PATH, config.AMAZON_CATEGORIES)

    train_click, val_click = split_train_val(click_df, config.EVAL_SPLIT_RATIO)

    print(">>> Building encoders & features...")
    encoders = build_extended_encoders(click_df, raw_meta, config.EXT_ENCODER_PKL)
    user_features = build_extended_user_features(
        train_click, raw_meta, encoders, hist_len=config.HIST_LEN)
    # 归一化 patch (embedding_recall 需要 user_avg/std_rating_norm)
    if 'hist_items_trunc' not in user_features.columns:
        user_features['hist_items_trunc'] = user_features['hist_items']
        user_features['user_avg_rating_norm'] = user_features['user_avg_rating'] / 5.0
        s = user_features['user_std_rating'].astype(float)
        user_features['user_std_rating_norm'] = (s - s.min()) / (s.max() - s.min() + 1e-8)
    if 'user_avg_rating_norm' not in user_features.columns:
        user_rating_stats = train_click.groupby('user_id')['rating'].agg(['mean', 'std']).fillna(0)
        user_rating_stats.columns = ['user_avg_rating', 'user_std_rating']
        user_rating_stats = user_rating_stats.reset_index()
        user_rating_stats['user_avg_rating_norm'] = user_rating_stats['user_avg_rating'] / 5.0
        user_rating_stats['user_std_rating_norm'] = (user_rating_stats['user_std_rating'] / 2.0).clip(0, 1)
        user_features = user_features.merge(
            user_rating_stats[['user_id', 'user_avg_rating_norm', 'user_std_rating_norm']],
            on='user_id', how='left')
        user_features['user_avg_rating_norm'] = user_features['user_avg_rating_norm'].fillna(0)
        user_features['user_std_rating_norm'] = user_features['user_std_rating_norm'].fillna(0)
    item_features = build_extended_item_features(train_click, raw_meta, encoders)
    item_topk_click = get_item_topk_click(click_df, k=200)

    print(">>> Loading HSTU embeddings...")
    with open(config.HSTU_EMBED_PKL, 'rb') as f:
        all_item_vecs = pickle.load(f)
    print(f">>> HSTU embeddings shape: {all_item_vecs.shape}")

    val_gt = val_click.groupby('user_id')['click_article_id'].apply(set).to_dict()
    all_val_users = list(val_gt.keys())
    rng = np.random.default_rng(42)
    if len(all_val_users) > SAMPLE_USERS:
        sample_users = rng.choice(all_val_users, SAMPLE_USERS, replace=False).tolist()
    else:
        sample_users = all_val_users
    print(f">>> Sampled {len(sample_users):,} val users")

    print(">>> HSTU embedding recall (top-100)...")
    hstu_dict = embedding_recall(
        sample_users, {}, train_click, user_features, item_features,
        encoders, all_item_vecs, item_topk_click,
        hist_len=config.HIST_LEN, recall_item_num=config.RECALL_NUM, weight=1.0,
        model_path=config.HSTU_BEST_FILE, channel_label="HSTU"
    )

    n = len(sample_users)
    hit = 0
    for u in sample_users:
        gt = val_gt.get(u, set())
        if not gt:
            continue
        pool = set(hstu_dict.get(u, {}).keys())
        if gt & pool:
            hit += 1
    hr100 = hit / max(n, 1)
    print("\n" + "=" * 60)
    print(f">>> HSTU 全库 HR@100 = {hr100*100:.2f}% ({hit:,}/{n:,})")
    print(f">>> 对比基线: V2=8.66% | ItemCF=10.62% | 四路并集=16.89%")
    print(f">>> Total time: {time.time()-t0:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
