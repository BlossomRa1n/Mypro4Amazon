"""
Quick baseline: ItemCF-only recall + NDCG evaluation
不需要任何模型训练, 纯统计方法.
"""
import os, sys, pickle
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, 'code')
import config
from data_loader import (
    get_all_click_df, get_positive_click_df, load_articles,
    build_encoders, get_user_item_time, get_item_topk_click
)
from recall_fusion import itemcf_recall
from evaluate import split_train_val

ITEMCF_CACHE = os.path.join(config.MODEL_PATH, 'itemcf_i2i_sim.pkl')

print("=" * 60)
print("ItemCF Baseline Test")
print("=" * 60)

# 1. Load data
full_click = get_all_click_df(config.DATA_PATH, offline=config.OFFLINE_MODE)
click_df = get_positive_click_df(full_click)
train_click, val_click = split_train_val(click_df, config.EVAL_SPLIT_RATIO)
print(f"Train: {len(train_click):,}, Val: {len(val_click):,}")

# 2. Build ItemCF
user_item_time = get_user_item_time(train_click)
item_topk = get_item_topk_click(train_click, k=50)

if os.path.exists(ITEMCF_CACHE):
    with open(ITEMCF_CACHE, 'rb') as f:
        i2i_sim = pickle.load(f)
    print("Loaded cached ItemCF")
else:
    from itemcf import itemcf_sim
    i2i_sim = itemcf_sim(user_item_time)
    os.makedirs(config.MODEL_PATH, exist_ok=True)
    with open(ITEMCF_CACHE, 'wb') as f:
        pickle.dump(i2i_sim, f)
    print("Computed ItemCF")

# 3. Run recall on validation users
val_users = val_click['user_id'].unique()
recall = itemcf_recall(val_users, user_item_time, i2i_sim, item_topk,
                       sim_item_topk=10, recall_item_num=50)

# 4. Evaluate NDCG@20
val_gt = defaultdict(set)
for _, row in val_click.iterrows():
    val_gt[row['user_id']].add(row['click_article_id'])

train_set = defaultdict(set)
for _, row in train_click.iterrows():
    train_set[row['user_id']].add(row['click_article_id'])

hr_total = 0
ndcgs = []
n_eval = 0
for uid, gt in val_gt.items():
    if uid not in recall:
        continue
    user_recs = recall[uid]
    if isinstance(user_recs, dict):
        user_recs = sorted(user_recs.items(), key=lambda x: -x[1])
        user_recs = [item_id for item_id, _ in user_recs[:20]]
    else:
        user_recs = [item_id for item_id, _ in user_recs[:20]]
    recs = [r for r in user_recs if r not in train_set.get(uid, set())][:20]
    hits = gt.intersection(recs)
    if hits:
        hr_total += 1
    for pos, item_id in enumerate(recs):
        if item_id in gt:
            ndcgs.append(1.0 / np.log2(pos + 2))
    n_eval += 1

hr = hr_total / n_eval if n_eval else 0
ndcg = sum(ndcgs) / n_eval if n_eval else 0

print(f"\n=== RESULTS ===")
print(f"Evaluated users: {n_eval}")
print(f"HR@20:  {hr:.4f} ({hr*100:.1f}%)")
print(f"NDCG@20: {ndcg:.4f}")

# Popularity baseline for comparison
pop_items = train_click['click_article_id'].value_counts().index[:20].tolist()
pop_hr = 0
pop_ndcgs = []
pop_n = 0
for uid, gt in val_gt.items():
    recs = [r for r in pop_items if r not in train_set.get(uid, set())][:20]
    hits = gt.intersection(recs)
    if hits:
        pop_hr += 1
    for pos, item_id in enumerate(recs):
        if item_id in gt:
            pop_ndcgs.append(1.0 / np.log2(pos + 2))
    pop_n += 1

print(f"\nPopularity baseline:")
print(f"HR@20:  {pop_hr/pop_n:.4f} ({pop_hr/pop_n*100:.1f}%)")
print(f"NDCG@20: {sum(pop_ndcgs)/pop_n:.4f}")
