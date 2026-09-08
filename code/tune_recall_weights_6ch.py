"""
召回融合权重 grid search (6 路: ItemCF / V2 / Category / Hot / HSTU / MIND)。

并集增益测量已证 HSTU (+3.26pp 独有) 与 MIND (+1.59pp 独有) 都是有效通路,
故在四路基础上加 HSTU / MIND 两路, 重调融合权重。

两阶段 (近可加性 → 贪心足够):
  Stage 1: 四路 grid search (Phase 2 V2 已从 8.66% 变强到 10.13%, 权重会漂移)
  Stage 2: 固定 Stage 1 最优四路权重, 联合 grid search HSTU × MIND 权重

复用 tune_recall_weights_ext.py 的 presort/merge_fast/eval_hr_ndcg 与
recall_union_gain.py 的 mind_recall (多兴趣召回)。
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import pickle, time, json
import numpy as np
from itertools import product
import torch
from tqdm import tqdm

import config
from data_loader import (get_all_click_df, load_articles,
                         get_user_item_time, get_item_topk_click)
from data_loader_ext import (load_raw_meta, build_extended_encoders,
                             build_extended_item_features,
                             build_extended_user_features, merge_jsonl_features)
from evaluate import split_train_val
from recall_fusion import (itemcf_recall, embedding_recall,
                           category_preference_recall, hot_recall)
from model import MINDModel

SAMPLE_USERS = 100000


def mind_recall(sample_users, user_feat_dict, train_items, model, all_item_vecs,
                num_items, hist_len, K, per_interest, recall_num, encoders, device):
    """MIND 多兴趣召回 → {raw_uid: {raw_item_id: score}} (K 兴趣各 top-N 合并去重)."""
    item_le = encoders['item_id']
    user_le = encoders['user_id']
    raw_to_idx = encoders.get('raw_to_idx', None)
    idx_to_raw_item = item_le.classes_.tolist()
    all_item_vecs_t = torch.FloatTensor(all_item_vecs).to(device)

    recall_dict = {}
    batch_size = 256
    for start in tqdm(range(0, len(sample_users), batch_size), desc="MIND Recall"):
        chunk = sample_users[start:start + batch_size]
        histories, hist_lens, click_counts, time_spans = [], [], [], []
        valid = []
        for raw_uid in chunk:
            feat = user_feat_dict.get(raw_uid)
            if feat is None:
                continue
            uidx = (raw_to_idx.get(raw_uid, 0) if raw_to_idx is not None
                    else user_le.transform([raw_uid])[0])
            hist = feat['hist']
            hl = len(hist)
            if hl > hist_len:
                hist = hist[-hist_len:]
                hl = hist_len
            padded = hist + [0] * (hist_len - hl)
            histories.append(padded)
            hist_lens.append(hl)
            click_counts.append(feat['click_norm'])
            time_spans.append(feat['span_norm'])
            valid.append((raw_uid, uidx))
        if not valid:
            continue

        user_batch = {
            'user_id': torch.LongTensor([u for _, u in valid]).to(device),
            'hist_items': torch.LongTensor(histories).to(device),
            'hist_len': torch.LongTensor(hist_lens).to(device),
            'click_count': torch.FloatTensor(click_counts).to(device),
            'time_span': torch.FloatTensor(time_spans).to(device),
        }
        mask_rows, mask_cols = [], []
        for i, (raw_uid, _) in enumerate(valid):
            for e in train_items.get(raw_uid, ()):
                if 0 <= e < num_items:
                    mask_rows.append(i)
                    mask_cols.append(e)
        rows = torch.as_tensor(mask_rows, dtype=torch.long, device=device)
        cols = torch.as_tensor(mask_cols, dtype=torch.long, device=device)

        with torch.no_grad():
            interest_vectors = model.user_tower(
                user_batch['user_id'], user_batch['hist_items'],
                user_batch['hist_len'], user_batch['click_count'],
                user_batch['time_span'], target_item_vec=None)
            per_interest_topk = []
            for kk in range(K):
                ivec = interest_vectors[:, kk, :]
                s = torch.matmul(ivec, all_item_vecs_t.t())
                if mask_rows:
                    s[rows, cols] = -1e9
                per_interest_topk.append(torch.topk(s, per_interest, dim=1).indices)

        for i, (raw_uid, _) in enumerate(valid):
            merged, seen = [], set()
            for kk in range(K):
                for idx in per_interest_topk[kk][i].tolist():
                    if idx not in seen:
                        seen.add(idx)
                        merged.append(idx)
            merged = merged[:recall_num]
            recall_dict[raw_uid] = {idx_to_raw_item[idx]: 1.0 / (rank + 1)
                                    for rank, idx in enumerate(merged)}

    return recall_dict


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
    print("Recall Fusion Weight Grid Search (6-channel: +HSTU +MIND)")
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Device: {device}")

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
    with open(config.V2_EMBED_PKL, 'rb') as f:
        all_item_vecs = pickle.load(f)
    print(f">>> V2 embeddings: {all_item_vecs.shape}")

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

    # ---- 5. 六路召回 (一次) ----
    print("[6/6] Computing 6 recall channels...")
    ch_itemcf = itemcf_recall(sample_users, user_item_time_dict, i2i_sim,
                              item_topk_click, sim_item_topk=10,
                              recall_item_num=config.RECALL_NUM)
    print(f"    Ch1 ItemCF:   {len(ch_itemcf):,} users")

    ch_v2 = embedding_recall(
        sample_users, {}, train_click, user_features, item_features,
        encoders, all_item_vecs, item_topk_click,
        hist_len=config.HIST_LEN, recall_item_num=config.RECALL_NUM, weight=1.0,
        model_path=config.V2_BEST_FILE, channel_label="V2 Phase2")
    print(f"    Ch2 V2:       {len(ch_v2):,} users")

    ch_cat = category_preference_recall(
        sample_users, train_click, articles_df, item_topk_click,
        recall_item_num=20, weight=0.5)
    print(f"    Ch3 Category: {len(ch_cat):,} users")

    ch_hot = hot_recall(sample_users, item_topk_click, recall_item_num=5, weight=0.1)
    print(f"    Ch4 Hot:      {len(ch_hot):,} users")

    # HSTU
    with open(config.HSTU_EMBED_PKL, 'rb') as f:
        hstu_vecs = pickle.load(f)
    ch_hstu = embedding_recall(
        sample_users, {}, train_click, user_features, item_features,
        encoders, hstu_vecs, item_topk_click,
        hist_len=config.HIST_LEN, recall_item_num=config.RECALL_NUM, weight=1.0,
        model_path=config.HSTU_BEST_FILE, channel_label="HSTU")
    print(f"    Ch5 HSTU:     {len(ch_hstu):,} users")

    # MIND
    mind_ckpt = torch.load(config.MIND_MODEL_FILE, map_location=device, weights_only=False)
    mind_cfg = mind_ckpt['config']
    mind_model = MINDModel(**mind_cfg).to(device)
    mind_model.load_state_dict(mind_ckpt['model_state_dict'])
    mind_model.eval()
    with open(config.MIND_EMBED_PKL, 'rb') as f:
        mind_vecs = pickle.load(f)
    num_items_mind = min(mind_cfg['num_items'], mind_vecs.shape[0])

    user_feat_dict = {}
    for _, row in user_features.iterrows():
        uid = row['user_id']
        user_feat_dict[uid] = {
            'hist': row.get('hist_items_trunc', row.get('hist_items', [])),
            'click_norm': row['click_count_norm'],
            'span_norm': row['time_span_norm'],
        }
    item_le = encoders['item_id']
    raw_to_item_enc = {cls: i for i, cls in enumerate(item_le.classes_)}
    train_enc = train_click['click_article_id'].map(raw_to_item_enc).dropna().astype(int)
    train_pairs = train_click.loc[train_enc.index, ['user_id']].copy()
    train_pairs['item_enc'] = train_enc.values
    train_items = train_pairs.groupby('user_id')['item_enc'].apply(set).to_dict()

    K = mind_cfg['num_interests']
    per_interest = 40
    ch_mind = mind_recall(
        sample_users, user_feat_dict, train_items, mind_model, mind_vecs,
        num_items_mind, config.HIST_LEN, K, per_interest,
        config.RECALL_NUM, encoders, device)
    print(f"    Ch6 MIND:     {len(ch_mind):,} users")

    # 预排序
    ch_sorted = {
        'itemcf': presort(ch_itemcf),
        'v2': presort(ch_v2),
        'category': presort(ch_cat),
        'hot': presort(ch_hot),
        'hstu': presort(ch_hstu),
        'mind': presort(ch_mind),
    }

    # ---- 并集天花板 ----
    def union_hr(ch_names):
        union = set()
        for u in sample_users:
            gt = val_gt.get(u, set())
            if not gt:
                continue
            pool = set()
            for name in ch_names:
                pool.update(i for i, _ in ch_sorted[name].get(u, []))
            if gt & pool:
                union.add(u)
        return len(union) / max(len(sample_users), 1)

    union4 = union_hr(['itemcf', 'v2', 'category', 'hot'])
    union6 = union_hr(['itemcf', 'v2', 'category', 'hot', 'hstu', 'mind'])
    print(f"\n>>> 四路并集 HR@100 = {union4*100:.2f}%")
    print(f">>> 六路并集 HR@100 = {union6*100:.2f}%")

    # ============ Stage 1: 四路 grid search ============
    print("\n" + "=" * 60)
    print("Stage 1: 四路权重 grid search (Phase 2 V2)")
    print("=" * 60)
    cands = {
        'itemcf': [0.5, 1.0, 1.2, 1.5, 2.0, 3.0],
        'v2': [1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
        'category': [0.1, 0.3, 0.5, 0.7, 1.0],
        'hot': [0.05, 0.1, 0.2],
    }
    names4 = ['itemcf', 'v2', 'category', 'hot']
    best4_hr, best4_ndcg, best4_w = -1, 0, None
    total = (len(cands['itemcf']) * len(cands['v2'])
             * len(cands['category']) * len(cands['hot']))
    cnt = 0
    for w in product(cands['itemcf'], cands['v2'], cands['category'], cands['hot']):
        cnt += 1
        channels = [ch_sorted[n] for n in names4]
        merged = merge_fast(channels, list(w), final_num=config.RECALL_NUM)
        hr, ndcg = eval_hr_ndcg(merged, val_gt, sample_users, k=config.RECALL_NUM)
        if hr > best4_hr:
            best4_hr, best4_ndcg, best4_w = hr, ndcg, list(w)
            print(f"  [{cnt}/{total}] ★ best4 HR@100={hr*100:.2f}% w={list(w)}")
        elif cnt % 100 == 0:
            print(f"  [{cnt}/{total}] scanned... best4 HR@100={best4_hr*100:.2f}%")

    print(f"\n>>> Stage 1 最优四路: itemcf={best4_w[0]}, v2={best4_w[1]}, "
          f"category={best4_w[2]}, hot={best4_w[3]} → HR@100={best4_hr*100:.2f}%")

    # 当前权重基线
    cur = [config.RECALL_WEIGHTS.get(k) for k in ['itemcf', 'v2_sasrec', 'category', 'hot']]
    cur_merged = merge_fast([ch_sorted[n] for n in names4], cur, final_num=config.RECALL_NUM)
    cur_hr, cur_ndcg = eval_hr_ndcg(cur_merged, val_gt, sample_users, k=config.RECALL_NUM)
    print(f">>> 当前权重 {cur} → HR@100={cur_hr*100:.2f}% (Phase2 重调前后对比)")

    # ============ Stage 2: 固定四路最优, 调 HSTU × MIND ============
    print("\n" + "=" * 60)
    print("Stage 2: 固定四路最优, grid search HSTU × MIND")
    print("=" * 60)
    cands_hstu = [0.5, 1.0, 1.5, 2.0]
    cands_mind = [0.3, 0.5, 1.0, 1.5]
    best6_hr, best6_ndcg, best6_wh, best6_wm = -1, 0, None, None
    results6 = []
    for wh, wm in product(cands_hstu, cands_mind):
        w6 = best4_w + [wh, wm]
        channels6 = [ch_sorted[n] for n in names4 + ['hstu', 'mind']]
        merged = merge_fast(channels6, w6, final_num=config.RECALL_NUM)
        hr, ndcg = eval_hr_ndcg(merged, val_gt, sample_users, k=config.RECALL_NUM)
        results6.append((hr, ndcg, wh, wm))
        if hr > best6_hr:
            best6_hr, best6_ndcg, best6_wh, best6_wm = hr, ndcg, wh, wm
            print(f"  ★ best6 HR@100={hr*100:.2f}% (HSTU={wh}, MIND={wm})")

    best6_w = best4_w + [best6_wh, best6_wm]
    print(f"\n>>> Stage 2 最优六路: itemcf={best6_w[0]}, v2={best6_w[1]}, "
          f"category={best6_w[2]}, hot={best6_w[3]}, hstu={best6_w[4]}, mind={best6_w[5]}")
    print(f"    HR@100 = {best6_hr*100:.2f}% (四路最优 {best4_hr*100:.2f}%, "
          f"六路并集天花板 {union6*100:.2f}%)")
    print(f"    NDCG@20 = {best6_ndcg:.4f}")

    results6.sort(key=lambda x: -x[0])
    print(f"\n>>> Top-10 (HSTU × MIND) 组合:")
    for j, (hr, ndcg, wh, wm) in enumerate(results6):
        print(f"  {j+1}. HR@100={hr*100:.2f}% NDCG@20={ndcg:.4f} HSTU={wh} MIND={wm}")

    # 汇总
    summary = {
        'stage1_best_4ch': {'weights': best4_w, 'hr100': best4_hr, 'ndcg20': best4_ndcg},
        'stage2_best_6ch': {'weights': best6_w, 'hr100': best6_hr, 'ndcg20': best6_ndcg},
        'current_4ch': {'weights': cur, 'hr100': cur_hr, 'ndcg20': cur_ndcg},
        'union4': union4, 'union6': union6,
        'hstu_unique_pp': union6 - union4,
        'n_users': len(sample_users),
    }
    out = os.path.join(config.RESULT_PATH, 'fusion_weights_6ch.json')
    os.makedirs(config.RESULT_PATH, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n>>> Results saved to {out}")

    print("\n>>> 建议写入 config.py RECALL_WEIGHTS:")
    print(f"    {{'itemcf': {best6_w[0]}, 'v2_sasrec': {best6_w[1]}, "
          f"'category': {best6_w[2]}, 'hot': {best6_w[3]}, "
          f"'hstu': {best6_w[4]}, 'mind': {best6_w[5]}}}")

    print(f"\n>>> Total time: {time.time()-t0:.2f}s")


if __name__ == "__main__":
    main()
