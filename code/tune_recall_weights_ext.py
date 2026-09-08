"""
召回融合权重 grid search (扩展 pipeline) — 10 万 val 用户上搜索最优四路召回权重。

与旧 tune_recall_weights.py 的区别:
  1. 用扩展 encoder (num_items=379152, 含 brand), 与 inference_full.py 完全一致
  2. 优化 HR@100 (召回天花板, 直接决定 DIN 精排输入), 副指标 NDCG@20
  3. 预排序各通道一次 + 快速融合, grid search 分钟级

对标 baseline: 四路并集 HR@100=16.89% (ItemCF 10.62% / V2 8.66% / Category 3.13%)
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import pickle, time
import numpy as np
from itertools import product

import config
from data_loader import (get_all_click_df, load_articles,
                         get_user_item_time, get_item_topk_click)
from data_loader_ext import (load_raw_meta, build_extended_encoders,
                             build_extended_item_features,
                             build_extended_user_features, merge_jsonl_features)
from evaluate import split_train_val
from recall_fusion import (itemcf_recall, embedding_recall,
                           category_preference_recall, hot_recall)

SAMPLE_USERS = 100000


def presort(channel_dict):
    """{user: {item:score}} → {user: [(item,score),...]} 降序预排序 (只做一次)."""
    out = {}
    for u, items in channel_dict.items():
        if isinstance(items, dict):
            out[u] = sorted(items.items(), key=lambda x: -x[1])
        else:
            out[u] = sorted(items, key=lambda x: -x[1])
    return out


def merge_fast(channels_sorted, weights, final_num=100):
    """按权重配额取各通道 top-quota, 加权求和排序取 top-final_num."""
    n = len(channels_sorted)
    total_w = sum(w for w in weights if w > 0) or 1.0
    floor = 3
    quotas = []
    for w in weights:
        quotas.append(0 if w <= 0 else max(floor, int(final_num * w / total_w)))

    all_users = set()
    for ch in channels_sorted:
        all_users.update(ch.keys())

    merged = {}
    for u in all_users:
        acc = {}
        for ch_sorted, w, q in zip(channels_sorted, weights, quotas):
            if w <= 0:
                continue
            for item, score in ch_sorted.get(u, [])[:q]:
                acc[item] = acc.get(item, 0) + score * w
        merged[u] = sorted(acc.items(), key=lambda x: -x[1])[:final_num]
    return merged


def eval_hr_ndcg(merged, val_gt, sample_users, k=100):
    hit = 0
    ndcg = 0.0
    n = len(sample_users)
    for u in sample_users:
        gt = val_gt.get(u, set())
        if not gt:
            continue
        items = merged.get(u, [])
        topk = [i for i, _ in items[:k]]
        if gt & set(topk):
            hit += 1
        for pos, it in enumerate(items[:20]):
            if it[0] in gt:
                ndcg += 1.0 / np.log2(pos + 2)
    return hit / max(n, 1), ndcg / max(n, 1)


def main():
    t0 = time.time()
    print("=" * 60)
    print("Recall Fusion Weight Grid Search (extended pipeline)")
    print("=" * 60)

    # ---- 1. 数据 ----
    print("\n[1/6] Loading data...")
    click_df = get_all_click_df(config.DATA_PATH, offline=config.OFFLINE_MODE)
    raw_meta = load_raw_meta(config.DATA_PATH, config.AMAZON_CATEGORIES)
    click_df = merge_jsonl_features(click_df, config.DATA_PATH, config.AMAZON_CATEGORIES)
    train_click, val_click = split_train_val(click_df, config.EVAL_SPLIT_RATIO)

    print("[2/6] Building extended encoders & features...")
    encoders = build_extended_encoders(click_df, raw_meta, config.EXT_ENCODER_PKL)
    user_features = build_extended_user_features(
        train_click, raw_meta, encoders, hist_len=config.HIST_LEN)
    # 归一化 patch (embedding_recall 需要 *_norm 列)
    if 'hist_items_trunc' not in user_features.columns:
        user_features['hist_items_trunc'] = user_features['hist_items']
        user_features['user_avg_rating_norm'] = user_features['user_avg_rating'] / 5.0
        s = user_features['user_std_rating'].astype(float)
        user_features['user_std_rating_norm'] = (s - s.min()) / (s.max() - s.min() + 1e-8)
    if 'user_avg_rating_norm' not in user_features.columns:
        us = train_click.groupby('user_id')['rating'].agg(['mean', 'std']).fillna(0)
        us.columns = ['user_avg_rating', 'user_std_rating']
        us = us.reset_index()
        us['user_avg_rating_norm'] = us['user_avg_rating'] / 5.0
        us['user_std_rating_norm'] = (us['user_std_rating'] / 2.0).clip(0, 1)
        user_features = user_features.merge(
            us[['user_id', 'user_avg_rating_norm', 'user_std_rating_norm']],
            on='user_id', how='left')
        user_features['user_avg_rating_norm'] = user_features['user_avg_rating_norm'].fillna(0)
        user_features['user_std_rating_norm'] = user_features['user_std_rating_norm'].fillna(0)
    item_features = build_extended_item_features(train_click, raw_meta, encoders)
    articles_df = load_articles(config.DATA_PATH)

    # ---- 2. ItemCF ----
    print("[3/6] Loading ItemCF similarity...")
    user_item_time_dict = get_user_item_time(
        train_click[train_click['click_label'] == 1])
    if os.path.exists(config.ITEMCF_SIM_PKL):
        with open(config.ITEMCF_SIM_PKL, 'rb') as f:
            i2i_sim = pickle.load(f)
    else:
        from itemcf import itemcf_sim
        i2i_sim = itemcf_sim(user_item_time_dict)
        with open(config.ITEMCF_SIM_PKL, 'wb') as f:
            pickle.dump(i2i_sim, f)
    item_topk_click = get_item_topk_click(click_df, k=200)

    # ---- 3. V2 embeddings ----
    print("[4/6] Loading V2 embeddings...")
    all_item_vecs = None
    if os.path.exists(config.V2_EMBED_PKL):
        with open(config.V2_EMBED_PKL, 'rb') as f:
            all_item_vecs = pickle.load(f)
        print(f">>> V2 embeddings: {all_item_vecs.shape}")
    if all_item_vecs is None:
        print("[ERROR] No V2 embeddings. Run train_v2.py first.")
        return

    # ---- 4. 采样 val 用户 + ground truth ----
    print("[5/6] Sampling val users...")
    val_gt = val_click.groupby('user_id')['click_article_id'].apply(set).to_dict()
    all_val_users = list(val_gt.keys())
    rng = np.random.default_rng(42)
    if len(all_val_users) > SAMPLE_USERS:
        sample_users = rng.choice(all_val_users, SAMPLE_USERS, replace=False).tolist()
    else:
        sample_users = all_val_users
    print(f">>> Sampled {len(sample_users):,} val users")

    # ---- 5. 四路召回 (一次) ----
    print("[6/6] Computing 4 recall channels...")
    ch_itemcf = itemcf_recall(sample_users, user_item_time_dict, i2i_sim,
                              item_topk_click, sim_item_topk=10,
                              recall_item_num=config.RECALL_NUM)
    print(f"    Ch1 ItemCF:   {len(ch_itemcf):,} users")

    ch_v2 = embedding_recall(
        sample_users, {}, train_click, user_features, item_features,
        encoders, all_item_vecs, item_topk_click,
        hist_len=config.HIST_LEN, recall_item_num=config.RECALL_NUM, weight=1.0,
        model_path=config.V2_BEST_FILE, channel_label="V2 SASRec")
    print(f"    Ch2 V2:       {len(ch_v2):,} users")

    ch_cat = category_preference_recall(
        sample_users, train_click, articles_df, item_topk_click,
        recall_item_num=20, weight=0.5)
    print(f"    Ch3 Category: {len(ch_cat):,} users")

    ch_hot = hot_recall(sample_users, item_topk_click, recall_item_num=5, weight=0.1)
    print(f"    Ch4 Hot:      {len(ch_hot):,} users")

    channels = [presort(c) for c in [ch_itemcf, ch_v2, ch_cat, ch_hot]]

    # ---- 并集天花板 ----
    union = set()
    for u in sample_users:
        gt = val_gt.get(u, set())
        if not gt:
            continue
        pool = set()
        for ch in channels:
            pool.update(i for i, _ in ch.get(u, []))
        if gt & pool:
            union.add(u)
    union_hr = len(union) / max(len(sample_users), 1)
    print(f"\n>>> 四路并集 HR@100 (天花板) = {union_hr*100:.2f}%")

    # ---- 当前权重基线 ----
    cur = [config.RECALL_WEIGHTS.get(k) for k in ['itemcf', 'v2_sasrec', 'category', 'hot']]
    cur_merged = merge_fast(channels, cur, final_num=config.RECALL_NUM)
    cur_hr, cur_ndcg = eval_hr_ndcg(cur_merged, val_gt, sample_users, k=config.RECALL_NUM)
    print(f">>> 当前权重 {cur} → HR@100={cur_hr*100:.2f}% | NDCG@20={cur_ndcg:.4f}")

    # ---- Grid search ----
    print("\nGrid searching fusion weights (optimize HR@100)...")
    cands_itemcf = [0.5, 1.0, 1.2, 1.5, 2.0, 3.0]
    cands_v2 = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    cands_cat = [0.1, 0.3, 0.5, 0.7, 1.0]
    cands_hot = [0.05, 0.1, 0.2]

    best_hr = -1
    best_ndcg = 0
    best_w = None
    results = []

    total = len(cands_itemcf) * len(cands_v2) * len(cands_cat) * len(cands_hot)
    for i, (w1, w2, w3, w4) in enumerate(
            product(cands_itemcf, cands_v2, cands_cat, cands_hot)):
        w = [w1, w2, w3, w4]
        merged = merge_fast(channels, w, final_num=config.RECALL_NUM)
        hr, ndcg = eval_hr_ndcg(merged, val_gt, sample_users, k=config.RECALL_NUM)
        results.append((hr, ndcg, w))
        if hr > best_hr:
            best_hr, best_ndcg, best_w = hr, ndcg, w
            print(f"  [{i+1}/{total}] ★ New best HR@100={hr*100:.2f}% "
                  f"(NDCG@20={ndcg:.4f})  w={w}")
        elif (i + 1) % 100 == 0:
            print(f"  [{i+1}/{total}] scanned... best HR@100={best_hr*100:.2f}%")

    print(f"\n{'='*60}")
    print(f"Best weights: itemcf={best_w[0]}, v2={best_w[1]}, "
          f"category={best_w[2]}, hot={best_w[3]}")
    print(f"  HR@100 = {best_hr*100:.2f}%  (当前 {cur_hr*100:.2f}%, "
          f"并集天花板 {union_hr*100:.2f}%)")
    print(f"  NDCG@20 = {best_ndcg:.4f}")
    print(f"\nCopy to config.py RECALL_WEIGHTS:")
    print(f"  {{'itemcf': {best_w[0]}, 'v2_sasrec': {best_w[1]}, "
          f"'category': {best_w[2]}, 'hot': {best_w[3]}}}")

    results.sort(key=lambda x: -x[0])
    print(f"\nTop-10 combos (by HR@100):")
    for j, (hr, ndcg, w) in enumerate(results[:10]):
        print(f"  {j+1}. HR@100={hr*100:.2f}% NDCG@20={ndcg:.4f}  w={w}")

    print(f"\n>>> Total time: {time.time()-t0:.2f}s")


if __name__ == "__main__":
    main()
