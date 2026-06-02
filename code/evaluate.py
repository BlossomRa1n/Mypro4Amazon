"""
离线评估模块 — 供训练脚本在每个 epoch 后调用，也可独立运行做完整评估。

评估策略（时序分割）：
- 对每个用户：前 80% 交互作历史，后 20% 作验证正样本
- 检查验证正样本有多少被推荐列表命中 → HR@K / NDCG@K
"""
import pandas as pd
import numpy as np
import torch
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
        'category_id': torch.LongTensor(subset['category_idx'].values.astype(int)).to(device),
        'item_click_count': torch.FloatTensor(subset['item_click_count_norm'].values).to(device),
        'created_at_ts': torch.FloatTensor(subset['created_at_ts_norm'].values).to(device),
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

    # 2. 构建训练集已交互物品集合 (评估时排除)
    train_items = defaultdict(set)
    for _, row in train_df.iterrows():
        uid = row['user_id']
        iid = row['click_article_id']
        if iid in item_le.classes_:
            train_items[uid].add(item_le.transform([iid])[0])

    # 3. 构建验证集 ground truth
    val_items = defaultdict(set)
    for _, row in val_df.iterrows():
        uid = row['user_id']
        iid = row['click_article_id']
        if iid in item_le.classes_ and uid in user_le.classes_:
            val_items[uid].add(item_le.transform([iid])[0])

    # 4. 采样用户 (加速)
    val_users = list(val_items.keys())
    if max_users and len(val_users) > max_users:
        val_users = list(np.random.choice(val_users, max_users, replace=False))

    # 5. 逐用户评估
    hr_total, ndcgs = 0, []
    user_batch_size = 256

    for start in range(0, len(val_users), user_batch_size):
        chunk_users = val_users[start:start + user_batch_size]
        # 组装用户 batch
        user_ids, histories, hist_lens, click_counts, time_spans = [], [], [], [], []
        valid_users_in_chunk = []
        for raw_uid in chunk_users:
            row = user_features[user_features['user_id'] == raw_uid]
            if len(row) == 0:
                continue
            row = row.iloc[0]
            if raw_to_idx is not None:
                uidx = raw_to_idx.get(raw_uid, 0)
            else:
                uidx = user_le.transform([raw_uid])[0]

            hist_len = 50
            hist = row['hist_items_trunc']
            hl = len(hist)
            if hl > hist_len:
                hist = hist[-hist_len:]
                hl = hist_len
            padded = hist + [0] * (hist_len - hl)

            user_ids.append(uidx)
            histories.append(padded)
            hist_lens.append(hl)
            click_counts.append(row['click_count_norm'])
            time_spans.append(row['time_span_norm'])
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

    # 采样用户
    val_users = val_df['user_id'].unique()
    if max_users and len(val_users) > max_users:
        val_users = list(np.random.choice(val_users, max_users, replace=False))

    pos_scores, neg_scores = [], []
    for raw_uid in val_users:
        row = user_features[user_features['user_id'] == raw_uid]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        if raw_to_idx is not None:
            uidx = raw_to_idx.get(raw_uid, 0)
        else:
            uidx = user_le.transform([raw_uid])[0]

        hist_len = 50
        hist = row['hist_items_trunc']
        hl = len(hist)
        if hl > hist_len:
            hist = hist[-hist_len:]
            hl = hist_len
        padded = hist + [0] * (hist_len - hl)

        # 该用户验证集正样本
        user_val = val_df[val_df['user_id'] == raw_uid]
        for _, vrow in user_val.iterrows():
            item_raw = vrow['click_article_id']
            if item_raw not in item_le.classes_:
                continue
            item_idx = item_le.transform([item_raw])[0]
            item_data = item_features.loc[item_idx] if item_idx in item_features.index else None
            if item_data is None:
                continue

            # 正样本分数
            batch = {
                'user_id': torch.LongTensor([uidx]).to(device),
                'hist_items': torch.LongTensor([padded]).to(device),
                'hist_len': torch.LongTensor([hl]).to(device),
                'click_count': torch.FloatTensor([row['click_count_norm']]).to(device),
                'time_span': torch.FloatTensor([row['time_span_norm']]).to(device),
                'item_id': torch.LongTensor([item_idx]).to(device),
                'category_id': torch.LongTensor([int(item_data.get('category_idx', 0))]).to(device),
                'item_click_count': torch.FloatTensor([float(item_data.get('item_click_count_norm', 0))]).to(device),
                'created_at_ts': torch.FloatTensor([float(item_data.get('created_at_ts_norm', 0))]).to(device),
            }
            with torch.no_grad():
                pos_scores.append(torch.sigmoid(model(batch)).item())

            # 随机负样本分数 (4 个)
            num_items = len(item_le.classes_)
            for _ in range(4):
                neg_idx = np.random.randint(1, num_items)
                nd = item_features.loc[neg_idx] if neg_idx in item_features.index else None
                if nd is None:
                    continue
                neg_batch = {
                    'user_id': batch['user_id'],
                    'hist_items': batch['hist_items'],
                    'hist_len': batch['hist_len'],
                    'click_count': batch['click_count'],
                    'time_span': batch['time_span'],
                    'item_id': torch.LongTensor([neg_idx]).to(device),
                    'category_id': torch.LongTensor([int(nd.get('category_idx', 0))]).to(device),
                    'item_click_count': torch.FloatTensor([float(nd.get('item_click_count_norm', 0))]).to(device),
                    'created_at_ts': torch.FloatTensor([float(nd.get('created_at_ts_norm', 0))]).to(device),
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
            user_recs[row['user_id']] = [row[f'article_{i+1}'] for i in range(5)]

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
