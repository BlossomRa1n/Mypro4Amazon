"""
离线评估模块 — 供训练脚本在每个 epoch 后调用，也可独立运行做完整评估。

评估策略（时序分割）：
- 对每个用户：前 80% 交互作历史，后 20% 作验证正样本
- 检查验证正样本有多少被推荐列表命中 → HR@K / NDCG@K
"""
import pandas as pd
import numpy as np
import torch
import os
from collections import defaultdict


# ============================================================
# 数据分割
# ============================================================

def split_train_val(click_df, split_ratio=0.8):
    """
    按用户时序分割训练/验证集。保留全量交互用于训练（含显式负样本）。
    验证集只保留正样本 (4-5 分)，即我们只评估模型对用户喜欢的商品的命中率。
    """
    click_df = click_df.sort_values('click_timestamp')
    train_rows, val_rows = [], []
    for _, grp in click_df.groupby('user_id'):
        grp = grp.sort_values('click_timestamp')
        split_point = max(1, int(len(grp) * split_ratio))
        train_rows.append(grp.iloc[:split_point])
        # 验证集只取正样本
        val_candidates = grp.iloc[split_point:]
        val_pos = val_candidates[val_candidates.get('click_label', 1) == 1]
        if len(val_pos) > 0:
            val_rows.append(val_pos)

    train_df = pd.concat(train_rows)
    val_df = pd.concat(val_rows) if val_rows else pd.DataFrame(columns=click_df.columns)

    print(f">>> Train: {len(train_df):,} interactions, {train_df['user_id'].nunique():,} users")
    print(f">>> Val:   {len(val_df):,} interactions (pos only), {val_df['user_id'].nunique():,} users")
    return train_df, val_df


# ============================================================
# 物品向量批量计算 (TwoTower 共用)
# ============================================================

def _build_item_batch_for_indices(item_indices, item_features, device):
    """向量化构建物品特征 batch — 避免逐行 .loc 循环"""
    subset = item_features.reindex(item_indices, fill_value=0)
    return {
        'item_id': torch.LongTensor(item_indices).to(device),
        'category_id': torch.LongTensor(subset['category_idx'].values.astype(int).copy()).to(device),
        'item_click_count': torch.FloatTensor(subset['item_click_count_norm'].values.copy()).to(device),
        'created_at_ts': torch.FloatTensor(subset['created_at_ts_norm'].values.copy()).to(device),
    }


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
# TwoTower 模型评估 (train_deep / train_v2 共用)
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
        val_users = list(np.random.choice(val_users, max_users, replace=False))

    # --- 预构建用户特征字典 (O(1) 查询, 替代逐行 DataFrame 扫描) ---
    user_feat_dict = {}
    for _, row in user_features.iterrows():
        uid = row['user_id']
        user_feat_dict[uid] = {
            'hist': row['hist_items_trunc'],
            'click_norm': row['click_count_norm'],
            'span_norm': row['time_span_norm'],
        }

    # 5. 逐用户评估
    hr_total, ndcgs = 0, []
    user_batch_size = 256

    for start in range(0, len(val_users), user_batch_size):
        chunk_users = val_users[start:start + user_batch_size]
        # 组装用户 batch
        user_ids, histories, hist_lens, click_counts, time_spans = [], [], [], [], []
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
            hl = len(hist)
            if hl > hist_len:
                hist = hist[-hist_len:]
                hl = hist_len
            padded = hist + [0] * (hist_len - hl)

            user_ids.append(uidx)
            histories.append(padded)
            hist_lens.append(hl)
            click_counts.append(feat['click_norm'])
            time_spans.append(feat['span_norm'])
            valid_users_in_chunk.append(raw_uid)

        if not valid_users_in_chunk:
            continue

        user_batch = {
            'user_id': torch.LongTensor(user_ids).to(device),
            'hist_items': torch.LongTensor(histories).to(device),
            'hist_len': torch.LongTensor(hist_lens).to(device),
            'click_count': torch.FloatTensor(click_counts).to(device),
            'time_span': torch.FloatTensor(time_spans).to(device),
        }

        with torch.no_grad():
            user_vecs = model.get_user_embedding(user_batch)  # [B, D]
            scores = torch.matmul(user_vecs, all_item_vecs.t())  # [B, num_items]

        scores = scores.cpu().numpy()

        for i, raw_uid in enumerate(valid_users_in_chunk):
            s = scores[i]
            # 排除训练集已交互物品
            excluded = train_items.get(raw_uid, set())
            for e in excluded:
                if 0 <= e < num_items:
                    s[e] = -1e9
            # Top-K
            top_indices = np.argsort(s)[::-1][:k]
            gt_set = val_items.get(raw_uid, set())
            hits = gt_set.intersection(set(top_indices))
            if hits:
                hr_total += 1
            for pos, idx in enumerate(top_indices):
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
        val_users = list(np.random.choice(val_users, max_users, replace=False))

    # 预构建用户特征字典
    user_feat_dict = {}
    for _, row in user_features.iterrows():
        uid = row['user_id']
        user_feat_dict[uid] = {
            'hist': row['hist_items_trunc'],
            'click_norm': row['click_count_norm'],
            'span_norm': row['time_span_norm'],
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
        hl = len(hist)
        if hl > hist_len:
            hist = hist[-hist_len:]
            hl = hist_len
        padded = hist + [0] * (hist_len - hl)

        user_batch = {
            'user_id': torch.LongTensor([uidx]).to(device),
            'hist_items': torch.LongTensor([padded]).to(device),
            'hist_len': torch.LongTensor([hl]).to(device),
            'click_count': torch.FloatTensor([feat['click_norm']]).to(device),
            'time_span': torch.FloatTensor([feat['span_norm']]).to(device),
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
        val_users = list(np.random.choice(val_users, max_users, replace=False))

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
    from data_loader import get_all_click_df

    result_path = os.path.join(config.RESULT_PATH, 'result_full_pipeline.csv')

    print("=" * 60)
    print("Offline Evaluation (Temporal Split)")
    print("=" * 60)

    train_df, test_df = split_train_val(
        get_all_click_df(config.DATA_PATH, offline=config.OFFLINE_MODE),
        split_ratio=config.EVAL_SPLIT_RATIO
    )

    # 读取已生成的推荐结果
    if os.path.exists(result_path):
        df = pd.read_csv(result_path)
        user_recs = {}
        for _, row in df.iterrows():
            user_recs[row['user_id']] = [row[f'item_{i+1}'] for i in range(5)]

        print(f">>> Loaded {len(user_recs)} users' recommendations")

        # 计算指标
        test_items = defaultdict(set)
        for uid, g in test_df.groupby('user_id'):
            test_items[uid] = set(g['click_article_id'].values)

        for ks in [(5, 5), (20, 20)]:
            hr_count = 0
            ndcg_vals = []
            total = 0
            for uid, gt_set in test_items.items():
                if uid not in user_recs:
                    continue
                recs = user_recs[uid][:ks[0]]
                hits = gt_set.intersection(recs)
                total += 1
                if hits:
                    hr_count += 1
                for pos, item_id in enumerate(recs):
                    if item_id in gt_set:
                        ndcg_vals.append(1 / np.log2(pos + 2))
            expected_ndcg = sum(ndcg_vals) / total if ndcg_vals else 0
            print(f"  K={ks[0]:2d}: HR={hr_count/total:.4f} ({hr_count/total*100:5.1f}%), NDCG={expected_ndcg:.4f}")

        # Popularity baseline
        pop_items = train_df['click_article_id'].value_counts().index.tolist()
        pop_recs = {uid: pop_items[:5] for uid in user_recs}
        hr_count = sum(1 for uid, gt_set in test_items.items()
                       if uid in pop_recs and gt_set.intersection(pop_recs[uid]))
        print(f"\n  Popularity K=5: HR={hr_count/len(user_recs):.4f}")
    else:
        print(f"[WARN] No result file at {result_path}, skipping eval.")
        print("Run inference_full.py first to generate results.")
