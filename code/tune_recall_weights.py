"""
自适应召回权重搜索: 在验证集上 grid search 最优四路召回权重

使用方式:
  1. 先跑 train.py 生成 ItemCF 相似度
  2. 跑 train_v2.py 生成双塔 embedding
  3. 跑本脚本搜索最优权重
  4. 将结果写入 config.py 的 RECALL_WEIGHTS
"""
import os, sys, pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import product

sys.path.insert(0, os.path.dirname(__file__))
import config
from data_loader import (
    get_all_click_df, get_positive_click_df, load_articles,
    build_encoders, build_user_features, build_item_features,
    get_user_item_time, get_item_topk_click
)
from recall_fusion import itemcf_recall, embedding_recall, category_preference_recall, hot_recall, merge_recall_results
from evaluate import split_train_val


def evaluate_recall_merged(user_recall_dict, val_df, train_df, k=20):
    """对合并后的召回结果计算 HR@K"""
    val_items = defaultdict(set)
    for _, row in val_df.iterrows():
        if 'click_label' in val_df.columns and row.get('click_label') != 1:
            continue
        val_items[row['user_id']].add(row['click_article_id'])

    train_items = defaultdict(set)
    for _, row in train_df.iterrows():
        train_items[row['user_id']].add(row['click_article_id'])

    hr_total = 0
    for uid, gt_set in val_items.items():
        if uid not in user_recall_dict:
            continue
        recs = [item_id for item_id, _ in user_recall_dict[uid][:k]]
        if gt_set.intersection(recs):
            hr_total += 1

    n_users = len(val_items)
    return hr_total / n_users if n_users else 0


def main():
    print("=" * 60)
    print("Adaptive Recall Weight Search")
    print("=" * 60)

    # 1. 加载数据 & 时序分割
    print("\n[1/5] Loading data...")
    full_click = get_all_click_df(config.DATA_PATH, offline=config.OFFLINE_MODE)
    click_df = get_positive_click_df(full_click)

    print("[2/5] Temporal split...")
    train_click, val_click = split_train_val(click_df, config.EVAL_SPLIT_RATIO)

    articles_df = load_articles(config.DATA_PATH)

    # 2. 构建编码器 & 特征
    print("[3/5] Building features...")
    encoders = build_encoders(click_df, articles_df, config.ENCODER_PKL)
    user_features = build_user_features(train_click, encoders, hist_len=config.HIST_LEN)
    item_features = build_item_features(articles_df, train_click, encoders)

    target_users = val_click['user_id'].unique()
    print(f"    Search on {len(target_users):,} val users")

    # 3. 加载或计算 ItemCF
    user_item_time_dict = get_user_item_time(train_click)
    item_topk_click = get_item_topk_click(train_click, k=50)

    if os.path.exists(config.ITEMCF_SIM_PKL):
        with open(config.ITEMCF_SIM_PKL, 'rb') as f:
            i2i_sim = pickle.load(f)
    else:
        from itemcf import itemcf_sim
        i2i_sim = itemcf_sim(user_item_time_dict)
        os.makedirs(config.MODEL_PATH, exist_ok=True)
        with open(config.ITEMCF_SIM_PKL, 'wb') as f:
            pickle.dump(i2i_sim, f)

    # 4. 加载 embedding
    all_item_vecs = None
    for path in [config.V2_EMBED_PKL, config.EMBED_PKL]:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                all_item_vecs = pickle.load(f)
            print(f"    Loaded embeddings from {os.path.basename(path)}, shape={all_item_vecs.shape}")
            break

    if all_item_vecs is None:
        print("[ERROR] No embeddings found. Run train_v2.py first.")
        return

    # 5. 分别计算四条召回通道
    print("[4/5] Computing 4 recall channels...")
    print(f"    Users: {len(target_users)}, Items: {len(item_features)}")

    ch_itemcf = itemcf_recall(target_users, user_item_time_dict, i2i_sim, item_topk_click,
                               sim_item_topk=10, recall_item_num=config.RECALL_NUM)
    print(f"    Ch1 ItemCF:      {len(ch_itemcf)} users")

    ch_embed = embedding_recall(target_users, {}, train_click, user_features,
                                 item_features, encoders, all_item_vecs, item_topk_click,
                                 hist_len=config.HIST_LEN, recall_item_num=config.RECALL_NUM, weight=1.0)
    print(f"    Ch2 Embedding:   {len(ch_embed)} users")

    ch_cat = category_preference_recall(target_users, train_click, articles_df, item_topk_click,
                                          recall_item_num=20, weight=0.5)
    print(f"    Ch3 Category:    {len(ch_cat)} users")

    ch_hot = hot_recall(target_users, item_topk_click, recall_item_num=5, weight=0.1)
    print(f"    Ch4 Hot:         {len(ch_hot)} users")

    channels = [ch_itemcf, ch_embed, ch_cat, ch_hot]
    channel_names = ['itemcf', 'embedding', 'category', 'hot']

    # 6. Grid search
    print("[5/5] Grid searching weights...")
    candidates = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    best_hr = -1
    best_weights = None
    results = []

    for w1, w2, w3, w4 in product(candidates, repeat=4):
        merged = merge_recall_results(channels, [w1, w2, w3, w4])
        hr = evaluate_recall_merged(merged, val_click, train_click, k=config.EVAL_K)

        results.append((hr, (w1, w2, w3, w4)))
        if hr > best_hr:
            best_hr = hr
            best_weights = (w1, w2, w3, w4)
            print(f"    ★ New best: {best_weights} → HR@{config.EVAL_K}={best_hr:.4f}")

    # 7. 输出结果
    print(f"\n{'='*60}")
    print(f"Best weights found:")
    print(f"  itemcf={best_weights[0]}, embedding={best_weights[1]}, "
          f"category={best_weights[2]}, hot={best_weights[3]}")
    print(f"  HR@{config.EVAL_K}={best_hr:.4f} ({best_hr*100:.1f}%)")
    print(f"\nCopy to config.py:")
    print(f"  RECALL_WEIGHTS = {{")
    print(f"      'itemcf':    {best_weights[0]},")
    print(f"      'embedding': {best_weights[1]},")
    print(f"      'category':  {best_weights[2]},")
    print(f"      'hot':       {best_weights[3]},")
    print(f"  }}")

    # Top-10 for reference
    results.sort(key=lambda x: -x[0])
    print(f"\nTop-10 weight combinations:")
    for i, (hr, w) in enumerate(results[:10]):
        print(f"  {i+1}. HR={hr:.4f}  w={w}")

    print(f"\nTotal combos tested: {len(results)}")


if __name__ == "__main__":
    main()
