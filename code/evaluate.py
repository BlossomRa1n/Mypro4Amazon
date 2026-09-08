"""
离线评估模块 — 供训练脚本在每个 epoch 后调用，也可独立运行做完整评估。

评估策略（时序分割）：
- 对每个用户：前 80% 交互作历史，后 20% 作验证正样本
- 两个协议:
  1) Full-rank: 验证正样本在推荐列表中的命中率 (供初步诊断)
  2) Sampled (SASRec protocol): 1 gt vs N 随机负 → 模型打分 → HR@K / NDCG@K
     对齐 academic benchmark
"""
import pandas as pd
import numpy as np
import torch
import os, sys
from collections import defaultdict
from tqdm import tqdm


# ============================================================
# 数据分割
# ============================================================

def split_train_val(click_df, split_ratio=0.8):
    """
    按用户时序分割训练/验证集。保留全量交互用于训练（含显式负样本）。
    验证集只保留正样本 (4-5 分)，即我们只评估模型对用户喜欢的商品的命中率。

    Optimized: sort once globally (by user then timestamp), then vectorized
    cumulative-count to find split point — avoids per-group sort_values.
    """
    click_df = click_df.sort_values(['user_id', 'click_timestamp'])
    # Build per-user cumulative count (finds split index without per-group loop)
    user_groups = click_df.groupby('user_id', sort=False)
    cumcounts = user_groups.cumcount()  # 0, 1, 2, ... within each user
    counts = user_groups['click_timestamp'].transform('size')
    split_point = (counts * split_ratio).astype(int).clip(lower=1)

    # train_mask: rows where cumcount < split_point
    train_mask = cumcounts < split_point
    train_df = click_df[train_mask]

    # val_mask: rows where cumcount >= split_point AND click_label == 1
    val_mask = (cumcounts >= split_point) & (click_df.get('click_label', 1) == 1)
    val_df = click_df[val_mask]

    print(f">>> Train: {len(train_df):,} interactions, {train_df['user_id'].nunique():,} users")
    print(f">>> Val:   {len(val_df):,} interactions (pos only), {val_df['user_id'].nunique():,} users")
    return train_df, val_df


# ============================================================
# 物品向量批量计算 (TwoTower 共用)
# ============================================================

def _build_item_batch_for_indices(item_indices, item_features, device):
    """向量化构建物品特征 batch — 避免逐行 .loc 循环 (支持扩展特征)"""
    subset = item_features.reindex(item_indices, fill_value=0)
    batch = {
        'item_id': torch.LongTensor(item_indices).to(device),
        'category_id': torch.LongTensor(subset['category_idx'].values.astype(int).copy()).to(device),
        'item_click_count': torch.FloatTensor(subset['item_click_count_norm'].values.copy()).to(device),
        'created_at_ts': torch.FloatTensor(subset['created_at_ts_norm'].values.copy()).to(device),
    }
    # 扩展特征: brand + quality (仅在 DataFrame 中存在时添加)
    if 'brand_idx' in item_features.columns:
        batch['brand_id'] = torch.LongTensor(subset['brand_idx'].values.astype(int).copy()).to(device)
    if 'item_avg_rating_norm' in item_features.columns:
        batch['item_avg_rating'] = torch.FloatTensor(subset['item_avg_rating_norm'].values.copy()).to(device)
    if 'item_rating_number_norm' in item_features.columns:
        batch['item_rating_number'] = torch.FloatTensor(subset['item_rating_number_norm'].values.copy()).to(device)
    return batch


def compute_item_embeddings(model, num_items, item_features, device, batch_size=2048):
    """预计算所有物品的 embedding 向量 [num_items, embed_dim]"""
    model.eval()
    all_vecs = []
    with torch.no_grad():
        for start in range(0, num_items, batch_size):
            end = min(start + batch_size, num_items)
            indices = np.arange(start, end)
            item_batch = _build_item_batch_for_indices(indices, item_features, device)
            vecs = model.get_item_embedding(item_batch)
            all_vecs.append(vecs)
    return torch.cat(all_vecs, dim=0)  # [num_items, D]


# ============================================================
# TwoTower 模型评估 (train_v2 用)
# ============================================================

def evaluate_two_tower(model, val_df, train_df, user_features, item_features,
                       encoders, device, k=20, max_users=3000):
    """
    对 TwoTower 系列模型做 Recall@K / NDCG@K 评估。
    流程: 预计算所有 item embedding → 逐用户算 user_vec → dot product → top-K → 检查命中
    """
    model.eval()
    item_le = encoders['item_id']
    user_le = encoders['user_id']
    raw_to_idx = encoders.get('raw_to_idx', None)
    num_items = len(item_le.classes_)

    # 1. 预计算所有物品向量
    all_item_vecs = compute_item_embeddings(model, num_items, item_features, device)

    # --- 预构建 ID 映射 (只做一次) ---
    raw_to_item_enc = {}
    for i, cls in enumerate(item_le.classes_):
        raw_to_item_enc[cls] = i

    # 2. 构建训练集已交互物品集合 (向量化 groupby, 避免 iterrows)
    # 先映射 item 编码
    train_enc = train_df['click_article_id'].map(raw_to_item_enc).dropna().astype(int)
    train_pairs = pd.DataFrame({
        'user_id': train_df.loc[train_enc.index, 'user_id'],
        'item_enc': train_enc.values
    })
    train_items = train_pairs.groupby('user_id')['item_enc'].apply(set).to_dict()

    # 3. 构建验证集 ground truth (同样向量化)
    val_users_set = set(user_le.classes_)
    val_enc = val_df['click_article_id'].map(raw_to_item_enc).dropna().astype(int)
    val_pairs = pd.DataFrame({
        'user_id': val_df.loc[val_enc.index, 'user_id'],
        'item_enc': val_enc.values
    })
    # 过滤有效用户
    val_pairs = val_pairs[val_pairs['user_id'].isin(val_users_set)]
    val_items = val_pairs.groupby('user_id')['item_enc'].apply(set).to_dict()

    # 4. 采样用户 (加速)
    val_users = list(val_items.keys())
    if max_users and len(val_users) > max_users:
        rng = np.random.default_rng(42)
        val_users = list(rng.choice(val_users, max_users, replace=False))

    # --- 预构建用户特征字典 (O(1) 查询, 替代逐行 DataFrame 扫描) ---
    user_feat_dict = {}
    for _, row in user_features.iterrows():
        uid = row['user_id']
        hist = row.get('hist_items_trunc', row.get('hist_items', []))
        user_feat_dict[uid] = {
            'hist': hist,
            'hist_brands': row.get('hist_brands', []),
            'hist_time_deltas': row.get('hist_time_deltas', []),
            'click_norm': row['click_count_norm'],
            'span_norm': row['time_span_norm'],
            'avg_rating': float(row.get('user_avg_rating_norm', 0) or 0),
            'std_rating': float(row.get('user_std_rating_norm', 0) or 0),
        }

    # 5. 逐用户评估
    hr_total, ndcgs = 0, []
    user_batch_size = 256

    for start in range(0, len(val_users), user_batch_size):
        chunk_users = val_users[start:start + user_batch_size]
        # 组装用户 batch
        user_ids, histories, hist_lens, click_counts, time_spans, avg_ratings, std_ratings = [], [], [], [], [], [], []
        brands_list, times_list = [], []
        valid_users_in_chunk = []
        for raw_uid in chunk_users:
            feat = user_feat_dict.get(raw_uid)
            if feat is None:
                continue
            if raw_to_idx is not None:
                uidx = raw_to_idx.get(raw_uid, 0)
            else:
                uidx = user_le.transform([raw_uid])[0]

            hist_len = 50
            hist = feat['hist']
            hist_brands = feat['hist_brands'] or []
            hist_times = feat['hist_time_deltas'] or []
            hl = len(hist)
            if hl > hist_len:
                hist = hist[-hist_len:]
                hist_brands = hist_brands[-hist_len:]
                hist_times = hist_times[-hist_len:]
                hl = hist_len
            padded = hist + [0] * (hist_len - hl)
            padded_brands = hist_brands + [0] * (hist_len - len(hist_brands))
            padded_times = hist_times + [0.0] * (hist_len - len(hist_times))

            user_ids.append(uidx)
            histories.append(padded)
            brands_list.append(padded_brands)
            times_list.append(padded_times)
            hist_lens.append(hl)
            click_counts.append(feat['click_norm'])
            time_spans.append(feat['span_norm'])
            avg_ratings.append(feat['avg_rating'])
            std_ratings.append(feat['std_rating'])
            valid_users_in_chunk.append(raw_uid)

        if not valid_users_in_chunk:
            continue

        user_batch = {
            'user_id': torch.LongTensor(user_ids).to(device),
            'hist_items': torch.LongTensor(histories).to(device),
            'hist_brands': torch.LongTensor(brands_list).to(device),
            'hist_time_deltas': torch.FloatTensor(times_list).to(device),
            'hist_len': torch.LongTensor(hist_lens).to(device),
            'click_count': torch.FloatTensor(click_counts).to(device),
            'time_span': torch.FloatTensor(time_spans).to(device),
            'user_avg_rating': torch.FloatTensor(avg_ratings).to(device),
            'user_std_rating': torch.FloatTensor(std_ratings).to(device),
        }

        with torch.no_grad():
            user_vecs = model.get_user_embedding(user_batch)  # [B, D]
            scores = torch.matmul(user_vecs, all_item_vecs.t())  # [B, num_items]

            # 排除训练集已交互物品 (GPU 上批量 mask, 替代逐用户 CPU 赋值)
            mask_rows, mask_cols = [], []
            for i, raw_uid in enumerate(valid_users_in_chunk):
                for e in train_items.get(raw_uid, ()):
                    if 0 <= e < num_items:
                        mask_rows.append(i)
                        mask_cols.append(e)
            if mask_rows:
                rows = torch.as_tensor(mask_rows, dtype=torch.long, device=device)
                cols = torch.as_tensor(mask_cols, dtype=torch.long, device=device)
                scores[rows, cols] = -1e9

            # GPU topk (只取 top-k, 无需全量 argsort)
            top_indices = torch.topk(scores, k, dim=1).indices  # [B, k]

        top_indices = top_indices.cpu().numpy()

        for i, raw_uid in enumerate(valid_users_in_chunk):
            topk_idx = top_indices[i].tolist()
            gt_set = val_items.get(raw_uid, set())
            hits = gt_set.intersection(topk_idx)
            if hits:
                hr_total += 1
            for pos, idx in enumerate(topk_idx):
                if idx in gt_set:
                    ndcgs.append(1.0 / np.log2(pos + 2))

    hr = hr_total / len(val_users) if val_users else 0
    expected_ndcg = sum(ndcgs) / len(val_users) if val_users else 0

    return {
        'hr': hr,
        'ndcg': expected_ndcg,
        'n_users': len(val_users),
    }


# ============================================================
# Sampled 评估 (与 SASRec 论文对齐: 1正 vs N随机负)
# ============================================================

def evaluate_two_tower_sampled(model, val_df, train_df, user_features, item_features,
                                encoders, device, k=20, max_users=3000, num_negatives=100):
    """
    Sampled metrics 评估 (SASRec 论文标准):
    对每用户, 1 个正样本 vs `num_negatives` 个随机负样本排名 → HR@K / NDCG@K.
    """
    model.eval()
    item_le = encoders['item_id']
    user_le = encoders['user_id']
    raw_to_idx = encoders.get('raw_to_idx', None)
    num_items = len(item_le.classes_)

    # 预构建映射
    raw_to_item_enc = {cls: i for i, cls in enumerate(item_le.classes_)}

    # 训练集已交互集合 (排除负样本候选)
    train_enc = train_df['click_article_id'].map(raw_to_item_enc).dropna().astype(int)
    train_pairs = pd.DataFrame({
        'user_id': train_df.loc[train_enc.index, 'user_id'],
        'item_enc': train_enc.values
    })
    train_items = train_pairs.groupby('user_id')['item_enc'].apply(set).to_dict()

    # 验证集 ground truth (每用户取最近一次正交互)
    val_users_set = set(user_le.classes_)
    val_enc = val_df['click_article_id'].map(raw_to_item_enc).dropna().astype(int)
    val_pairs = pd.DataFrame({
        'user_id': val_df.loc[val_enc.index, 'user_id'],
        'item_enc': val_enc.values
    })
    val_pairs = val_pairs[val_pairs['user_id'].isin(val_users_set)]
    # 每个用户取最后一个作为 ground truth (leave-one-out)
    val_items = val_pairs.groupby('user_id')['item_enc'].last().to_dict()

    # 采样用户
    val_users = list(val_items.keys())
    if max_users and len(val_users) > max_users:
        rng = np.random.default_rng(42)
        val_users = list(rng.choice(val_users, max_users, replace=False))

    # 预构建用户特征字典
    user_feat_dict = {}
    for _, row in user_features.iterrows():
        uid = row['user_id']
        user_feat_dict[uid] = {
            'hist': row['hist_items_trunc'],
            'hist_brands': row.get('hist_brands', []),
            'hist_time_deltas': row.get('hist_time_deltas', []),
            'click_norm': row['click_count_norm'],
            'span_norm': row['time_span_norm'],
            'avg_rating': float(row.get('user_avg_rating_norm', 0) or 0),
            'std_rating': float(row.get('user_std_rating_norm', 0) or 0),
        }

    # 预计算所有物品向量 (仍然需要, 但评估时只取 N+1 个)
    all_item_vecs = compute_item_embeddings(model, num_items, item_features, device)

    # 逐用户 sampled 评估
    hr_total, ndcgs = 0, []
    rng = np.random.default_rng(42)

    for raw_uid in val_users:
        feat = user_feat_dict.get(raw_uid)
        if feat is None:
            continue
        gt_item = val_items.get(raw_uid)
        if gt_item is None:
            continue

        if raw_to_idx is not None:
            uidx = raw_to_idx.get(raw_uid, 0)
        else:
            uidx = user_le.transform([raw_uid])[0]

        hist_len = 50
        hist = feat['hist']
        hist_brands = feat['hist_brands'] or []
        hist_times = feat['hist_time_deltas'] or []
        hl = len(hist)
        if hl > hist_len:
            hist = hist[-hist_len:]
            hist_brands = hist_brands[-hist_len:]
            hist_times = hist_times[-hist_len:]
            hl = hist_len
        padded = hist + [0] * (hist_len - hl)
        padded_brands = hist_brands + [0] * (hist_len - len(hist_brands))
        padded_times = hist_times + [0.0] * (hist_len - len(hist_times))

        user_batch = {
            'user_id': torch.LongTensor([uidx]).to(device),
            'hist_items': torch.LongTensor([padded]).to(device),
            'hist_brands': torch.LongTensor([padded_brands]).to(device),
            'hist_time_deltas': torch.FloatTensor([padded_times]).to(device),
            'hist_len': torch.LongTensor([hl]).to(device),
            'click_count': torch.FloatTensor([feat['click_norm']]).to(device),
            'time_span': torch.FloatTensor([feat['span_norm']]).to(device),
            'user_avg_rating': torch.FloatTensor([feat['avg_rating']]).to(device),
            'user_std_rating': torch.FloatTensor([feat['std_rating']]).to(device),
        }

        with torch.no_grad():
            user_vec = model.get_user_embedding(user_batch)  # (1, D)

        # 采样 N 个负样本 (排除训练集和正样本)
        excluded = train_items.get(raw_uid, set()) | {gt_item}
        candidates = []
        while len(candidates) < num_negatives:
            neg = rng.integers(1, num_items)
            if neg not in excluded:
                candidates.append(neg)
                excluded.add(neg)

        # 正样本索引放在第 0 位
        item_indices = [gt_item] + candidates
        item_vecs = all_item_vecs[torch.LongTensor(item_indices).to(device)]  # (N+1, D)

        score = torch.matmul(user_vec, item_vecs.t())  # (1, N+1)
        score = score.cpu().numpy()[0]

        # 排名: 0 号是正样本, 1..N 是负样本
        rank = (score[1:] >= score[0]).sum() + 1  # 1-indexed rank
        if rank <= k:
            hr_total += 1
            ndcgs.append(1.0 / np.log2(rank + 1))  # position = rank-1, so rank+1 = (pos+1)+1 = pos+2

    n_eval = len(val_users)
    hr = hr_total / n_eval if n_eval else 0
    ndcg = sum(ndcgs) / n_eval if n_eval else 0

    return {
        'hr': hr,
        'ndcg': ndcg,
        'n_users': n_eval,
        'n_negatives': num_negatives,
        'method': 'sampled',
    }


# ============================================================
# DIN 评估
# ============================================================

def evaluate_din(model, val_df, user_features, item_features,
                 encoders, device, max_users=2000):
    """
    DIN 评估: 对验证集中每个正样本计算点击概率，对比随机负样本的概率。
    返回 AUC (近似) 和平均正样本预测概率。
    """
    model.eval()
    item_le = encoders['item_id']
    user_le = encoders['user_id']
    raw_to_idx = encoders.get('raw_to_idx', None)
    num_items = len(item_le.classes_)

    # --- 预构建用户特征字典 (O(1) 查询) ---
    user_feat_dict = {}
    for _, row in user_features.iterrows():
        uid = row['user_id']
        user_feat_dict[uid] = {
            'hist': row['hist_items_trunc'],
            'click_norm': row['click_count_norm'],
            'span_norm': row['time_span_norm'],
        }

    # --- 预构建物品特征 (numpy 数组, O(1) 下标访问) ---
    item_cat_arr = np.zeros(num_items, dtype=np.int64)
    item_click_arr = np.zeros(num_items, dtype=np.float32)
    item_created_arr = np.zeros(num_items, dtype=np.float32)
    for idx in item_features.index:
        item_cat_arr[idx] = int(item_features.loc[idx].get('category_idx', 0))
        item_click_arr[idx] = float(item_features.loc[idx].get('item_click_count_norm', 0))
        item_created_arr[idx] = float(item_features.loc[idx].get('created_at_ts_norm', 0))

    # --- 预构建验证集: user_id → [item_idx, ...] (O(1) 查询, 替代逐行扫 DataFrame) ---
    raw_to_item_enc = {cls: i for i, cls in enumerate(item_le.classes_)}

    # Filter val_df to valid users/items and encode
    val_enc = val_df['click_article_id'].map(raw_to_item_enc).dropna().astype(int)
    val_pairs = pd.DataFrame({
        'user_id': val_df.loc[val_enc.index, 'user_id'],
        'item_enc': val_enc.values
    })
    # Only keep users that are in user_feat_dict + user_le
    valid_uids = set(user_feat_dict.keys()) & set(user_le.classes_)
    val_pairs = val_pairs[val_pairs['user_id'].isin(valid_uids)]
    val_by_user = val_pairs.groupby('user_id')['item_enc'].apply(list).to_dict()

    # 采样用户
    val_users = list(val_by_user.keys())
    if max_users and len(val_users) > max_users:
        rng = np.random.default_rng(42)
        val_users = list(rng.choice(val_users, max_users, replace=False))

    pos_scores, neg_scores = [], []
    for raw_uid in val_users:
        feat = user_feat_dict[raw_uid]
        if raw_to_idx is not None:
            uidx = raw_to_idx.get(raw_uid, 0)
        else:
            uidx = user_le.transform([raw_uid])[0]

        hist_len = 50
        hist = feat['hist']
        hl = len(hist)
        if hl > hist_len:
            hist = hist[-hist_len:]
            hl = hist_len
        padded = hist + [0] * (hist_len - hl)

        # 该用户的验证集正样本 (从预建字典取, O(1))
        for item_idx in val_by_user.get(raw_uid, []):
            # 正样本分数
            batch = {
                'user_id': torch.LongTensor([uidx]).to(device),
                'hist_items': torch.LongTensor([padded]).to(device),
                'hist_len': torch.LongTensor([hl]).to(device),
                'click_count': torch.FloatTensor([feat['click_norm']]).to(device),
                'time_span': torch.FloatTensor([feat['span_norm']]).to(device),
                'item_id': torch.LongTensor([item_idx]).to(device),
                'category_id': torch.LongTensor([item_cat_arr[item_idx]]).to(device),
                'item_click_count': torch.FloatTensor([item_click_arr[item_idx]]).to(device),
                'created_at_ts': torch.FloatTensor([item_created_arr[item_idx]]).to(device),
            }
            with torch.no_grad():
                pos_scores.append(torch.sigmoid(model(batch)).item())

            # 随机负样本分数 (4 个)
            for _ in range(4):
                neg_idx = np.random.randint(1, num_items)
                neg_batch = {
                    'user_id': batch['user_id'],
                    'hist_items': batch['hist_items'],
                    'hist_len': batch['hist_len'],
                    'click_count': batch['click_count'],
                    'time_span': batch['time_span'],
                    'item_id': torch.LongTensor([neg_idx]).to(device),
                    'category_id': torch.LongTensor([item_cat_arr[neg_idx]]).to(device),
                    'item_click_count': torch.FloatTensor([item_click_arr[neg_idx]]).to(device),
                    'created_at_ts': torch.FloatTensor([item_created_arr[neg_idx]]).to(device),
                }
                with torch.no_grad():
                    neg_scores.append(torch.sigmoid(model(neg_batch)).item())

    if not pos_scores:
        return {'auc': 0.5, 'pos_mean': 0.5, 'n_samples': 0}

    pos_arr = np.array(pos_scores)
    neg_arr = np.array(neg_scores)

    # 近似 AUC: pairwise 比较
    auc = np.mean(pos_arr[:, None] > neg_arr[None, :])

    return {
        'auc': auc,
        'pos_mean': float(pos_arr.mean()),
        'n_users': len(val_users),
    }


# ============================================================
# 独立运行：完整离线评估
# ============================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, 'code')
    import config
    from data_loader_ext import load_raw_meta, merge_jsonl_features
    from data_loader import get_all_click_df

    import glob, os as _os
    # Auto-find the latest prediction CSV
    patterns = ['preds_*.csv', 'submission_*.csv']
    candidates = []
    for pat in patterns:
        candidates = sorted(glob.glob(_os.path.join(config.RESULT_PATH, pat)), key=_os.path.getmtime, reverse=True)
        if candidates:
            break
    result_path = candidates[0] if candidates else ''

    if not result_path or not _os.path.exists(result_path):
        print(f"[WARN] No prediction CSV found in {config.RESULT_PATH}, skipping eval.")
        print("Run inference_full.py first to generate results.")
        exit(0)

    print("=" * 60)
    print("Offline Evaluation (Temporal Split)")
    print("=" * 60)

    # ---- 使用和 inference_full.py 完全相同的数据管道 (CSV 5-core) ----
    print("Loading data (same CSV pipeline as inference_full.py)...")
    click_df = get_all_click_df(config.DATA_PATH, offline=config.OFFLINE_MODE)
    # Merge JSONL extra features (verified/helpful) into CSV click_df
    click_df = merge_jsonl_features(click_df, config.DATA_PATH, config.AMAZON_CATEGORIES)
    print(f">>> click_df: {len(click_df):,} interactions, "
          f"{click_df['user_id'].nunique():,} users, "
          f"{click_df['click_article_id'].nunique():,} items")

    # 时序分割
    train_df, test_df = split_train_val(click_df, split_ratio=config.EVAL_SPLIT_RATIO)

    # 读取已生成的推荐结果
    df = pd.read_csv(result_path)
    user_recs = {}
    for _, row in df.iterrows():
        user_recs[row['user_id']] = [row[f'item_{i+1}'] for i in range(5)]
    print(f">>> Loaded {len(user_recs)} users' recommendations")

    # ============================================================
    # Protocol 1: Full-rank (ground truth set → Hit/NDCG)
    # ============================================================
    test_items = defaultdict(set)
    for uid, g in test_df.groupby('user_id'):
        test_items[uid] = set(g['click_article_id'].values)
    val_users_fr = [u for u in test_items if u in user_recs]
    print(f">>> Full-rank: {len(val_users_fr):,} users with predictions "
          f"(/ {len(test_items):,} val users)")

    if val_users_fr:
        for ks in [(5, 5), (20, 20)]:
            hr_count, ndcg_vals = 0, []
            for uid in val_users_fr:
                gt_set = test_items[uid]
                recs = user_recs[uid][:ks[0]]
                hits = gt_set.intersection(recs)
                if hits:
                    hr_count += 1
                for pos, item_id in enumerate(recs):
                    if item_id in gt_set:
                        ndcg_vals.append(1 / np.log2(pos + 2))
            ndcg = sum(ndcg_vals) / len(val_users_fr) if ndcg_vals else 0
            print(f"  K={ks[0]:2d}: HR={hr_count/len(val_users_fr):.4f} "
                  f"({hr_count/len(val_users_fr)*100:5.1f}%), NDCG={ndcg:.4f}")

        pop_items = train_df['click_article_id'].value_counts().index.tolist()
        hr_pop = sum(1 for uid in val_users_fr
                     if pop_items[:5] and test_items[uid].intersection(pop_items[:5]))
        print(f"  Popularity K=5: HR={hr_pop/len(val_users_fr):.4f} "
              f"({hr_pop/len(val_users_fr)*100:.1f}%)")
    else:
        print("  [skip] No validation users with predictions")

    # ============================================================
    # Protocol 2: Sampled (SASRec protocol — model scoring)
    # 1 gt vs N random negatives → model scores → HR@K / NDCG@K
    # ============================================================
    print("\n" + "=" * 60)
    print("Sampled Evaluation (SASRec protocol: 1 gt vs N random negs)")
    print("=" * 60)

    # Load DIN-Ext model
    from model_ext import DINExtendedModel
    from data_loader_ext import build_extended_encoders, build_extended_item_features, \
        build_extended_user_features

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Using device: {device}")

    # Load encoders (must match inference)
    raw_meta = load_raw_meta(config.DATA_PATH, config.AMAZON_CATEGORIES)
    encoders = build_extended_encoders(click_df, raw_meta, config.EXT_ENCODER_PKL)
    item_le = encoders['item_id']
    user_le = encoders['user_id']
    num_items = len(item_le.classes_)
    num_users = len(user_le.classes_)

    # Load DIN-Ext checkpoint
    din_best = config.DIN_EXT_BEST_FILE if os.path.exists(config.DIN_EXT_BEST_FILE) else config.DIN_EXT_MODEL_FILE
    print(f">>> Loading DIN-Ext from: {din_best}")
    ckpt = torch.load(din_best, map_location=device, weights_only=False)
    cfg = ckpt['config']
    model = DINExtendedModel(
        num_users=cfg['num_users'], num_items=cfg['num_items'],
        num_brands=cfg['num_brands'], num_categories=cfg['num_categories'],
        embed_dim=cfg.get('embed_dim', 256), brand_embed_dim=cfg.get('brand_embed_dim', 64),
        hidden_dims=cfg['hidden_dims'], hist_len=cfg['hist_len'],
        dropout=cfg.get('dropout', 0.1),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"    model: users={cfg['num_users']}, items={cfg['num_items']}, "
          f"brands={cfg['num_brands']}, cats={cfg['num_categories']}")

    # Build features (train window only — same as inference)
    train_click, _ = split_train_val(click_df, config.EVAL_SPLIT_RATIO)
    user_features = build_extended_user_features(train_click, raw_meta, encoders, hist_len=config.HIST_LEN)
    item_features = build_extended_item_features(train_click, raw_meta, encoders)

    # --- Run evaluate_ext (from train_din_ext.py) ---
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ''))
    # Evaluate standalone sampled eval: import evaluate_ext from train_din_ext
    from train_din_ext import evaluate_ext as din_evaluate_ext

    # Leave-last-out from validation set
    val_click = test_df[test_df['click_label'] == 1]

    for neg_n in [50, 100, 200]:
        print(f"\n>>> neg={neg_n} (leave-last-out, 1 gt vs {neg_n} random)...")
        metrics = din_evaluate_ext(
            model, val_click, user_features, item_features,
            encoders, device, max_users=config.EVAL_MAX_USERS,
            hist_len=5, num_negatives=neg_n
        )
        auc = metrics['auc']
        pos_mean = metrics['pos_mean']
        print(f"    AUC={auc:.4f} (pairwise) | Pos Mean={pos_mean:.4f} | "
              f"Users={metrics['n_users']}")
