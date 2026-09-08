"""
MIND / HSTU 并集增益测量 — 回答: 单路 HR 低 ≠ 无增量价值, 关键在独有命中。

对采样 100K val 用户, 复用 recall_fusion 真实召回, 计算:
  1. 四路基线并集 HR (ItemCF ∪ V2 ∪ Cat ∪ Hot)
  2. +HSTU 并集 HR 与增量
  3. +MIND 并集 HR 与增量
  4. 六路全并集 HR 与增量
  5. HSTU / MIND 独有命中 (仅该路命中, 四路基线都漏)

判读: 独有命中 ≥1% 用户 → 该路对并集有净增益, 值得接入融合。
纯推理测量 (复用现成 checkpoint + embedding, 不训练)。
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import pickle, time, json
import numpy as np
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


def main():
    t0 = time.time()
    print(">>> Loading data (与 recall_attribution.py 一致)...")
    click_df = get_all_click_df(config.DATA_PATH, offline=config.OFFLINE_MODE)
    raw_meta = load_raw_meta(config.DATA_PATH, config.AMAZON_CATEGORIES)
    click_df = merge_jsonl_features(click_df, config.DATA_PATH, config.AMAZON_CATEGORIES)
    train_click, val_click = split_train_val(click_df, config.EVAL_SPLIT_RATIO)

    user_item_time_dict = get_user_item_time(
        train_click[train_click['click_label'] == 1])

    if os.path.exists(config.ITEMCF_SIM_PKL):
        with open(config.ITEMCF_SIM_PKL, 'rb') as f:
            i2i_sim = pickle.load(f)
        print(">>> Loaded cached ItemCF similarity")
    else:
        from itemcf import itemcf_sim
        i2i_sim = itemcf_sim(user_item_time_dict)
        with open(config.ITEMCF_SIM_PKL, 'wb') as f:
            pickle.dump(i2i_sim, f)

    val_gt = val_click.groupby('user_id')['click_article_id'].apply(set).to_dict()
    print(f">>> val_gt: {len(val_gt):,} users")

    item_topk_click = get_item_topk_click(click_df, k=200)
    all_val_users = list(val_gt.keys())
    rng = np.random.default_rng(42)
    if len(all_val_users) > SAMPLE_USERS:
        sample_users = rng.choice(all_val_users, SAMPLE_USERS, replace=False).tolist()
    else:
        sample_users = all_val_users
    print(f">>> Sampled {len(sample_users):,} val users")

    print(">>> Building encoders & features...")
    encoders = build_extended_encoders(click_df, raw_meta, config.EXT_ENCODER_PKL)
    user_features = build_extended_user_features(
        train_click, raw_meta, encoders, hist_len=config.HIST_LEN)
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
    articles_df = load_articles(config.DATA_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Device: {device}")

    # --- 四路基线召回 ---
    print("\n>>> [Ch1] ItemCF Recall...")
    itemcf_dict = itemcf_recall(sample_users, user_item_time_dict, i2i_sim,
                                item_topk_click, sim_item_topk=10,
                                recall_item_num=config.RECALL_NUM)

    with open(config.V2_EMBED_PKL, 'rb') as f:
        v2_vecs = pickle.load(f)
    print(">>> [Ch2] V2 (Phase 2) Embedding Recall...")
    v2_dict = embedding_recall(
        sample_users, {}, train_click, user_features, item_features,
        encoders, v2_vecs, item_topk_click,
        hist_len=config.HIST_LEN, recall_item_num=config.RECALL_NUM, weight=1.0,
        model_path=config.V2_BEST_FILE, channel_label="V2 Phase2")

    print(">>> [Ch3] Category Recall...")
    cat_dict = category_preference_recall(
        sample_users, train_click, articles_df, item_topk_click,
        recall_item_num=20, weight=0.5)
    print(">>> [Ch4] Hot Recall...")
    hot_dict = hot_recall(sample_users, item_topk_click, recall_item_num=5, weight=0.1)

    # --- HSTU 召回 ---
    with open(config.HSTU_EMBED_PKL, 'rb') as f:
        hstu_vecs = pickle.load(f)
    print(">>> [Ch5] HSTU Embedding Recall...")
    hstu_dict = embedding_recall(
        sample_users, {}, train_click, user_features, item_features,
        encoders, hstu_vecs, item_topk_click,
        hist_len=config.HIST_LEN, recall_item_num=config.RECALL_NUM, weight=1.0,
        model_path=config.HSTU_BEST_FILE, channel_label="HSTU")

    # --- MIND 召回 ---
    print(">>> [Ch6] MIND multi-interest Recall...")
    mind_ckpt = torch.load(config.MIND_MODEL_FILE, map_location=device, weights_only=False)
    mind_cfg = mind_ckpt['config']
    mind_model = MINDModel(**mind_cfg).to(device)
    mind_model.load_state_dict(mind_ckpt['model_state_dict'])
    mind_model.eval()
    print(f">>> MIND model: epoch {mind_ckpt.get('epoch', '?')}, K={mind_cfg['num_interests']}")
    with open(config.MIND_EMBED_PKL, 'rb') as f:
        mind_vecs = pickle.load(f)

    num_items_mind = min(mind_cfg['num_items'], mind_vecs.shape[0])
    if num_items_mind != mind_cfg['num_items']:
        print(f">>> MIND WARNING: cfg num_items={mind_cfg['num_items']} vs emb={mind_vecs.shape[0]}, using min")

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
    mind_dict = mind_recall(
        sample_users, user_feat_dict, train_items, mind_model, mind_vecs,
        num_items_mind, config.HIST_LEN, K, per_interest,
        config.RECALL_NUM, encoders, device)

    # --- 命中统计 ---
    def _pool(recall_dict, u):
        items = recall_dict.get(u, {})
        return set(items.keys()) if isinstance(items, dict) else set(i for i, _ in items)

    n = len(sample_users)
    pools = {'ItemCF': itemcf_dict, 'V2': v2_dict, 'Category': cat_dict,
             'Hot': hot_dict, 'HSTU': hstu_dict, 'MIND': mind_dict}
    hit_sets = {name: set() for name in pools}
    for u in sample_users:
        gt = val_gt.get(u, set())
        if not gt:
            continue
        for name, d in pools.items():
            if gt & _pool(d, u):
                hit_sets[name].add(u)

    four_union = hit_sets['ItemCF'] | hit_sets['V2'] | hit_sets['Category'] | hit_sets['Hot']
    five_hstu = four_union | hit_sets['HSTU']
    five_mind = four_union | hit_sets['MIND']
    six_union = four_union | hit_sets['HSTU'] | hit_sets['MIND']

    hstu_unique = hit_sets['HSTU'] - four_union
    mind_unique = hit_sets['MIND'] - four_union

    def pct(s):
        return f"{len(s)/n*100:.2f}% ({len(s):,}/{n:,})"

    print("\n" + "=" * 60)
    print(">>> 各路独立 HR@100:")
    for name in ['ItemCF', 'V2', 'Category', 'Hot', 'HSTU', 'MIND']:
        print(f"    {name:10s}: {pct(hit_sets[name])}")

    print("\n>>> 并集 HR 与增量:")
    print(f"    四路基线 (ItemCF∪V2∪Cat∪Hot): {pct(four_union)}")
    print(f"    +HSTU: {pct(five_hstu)}  (增量 +{len(five_hstu - four_union)/n*100:.2f}pp)")
    print(f"    +MIND: {pct(five_mind)}  (增量 +{len(five_mind - four_union)/n*100:.2f}pp)")
    print(f"    六路全并集: {pct(six_union)}  (增量 +{len(six_union - four_union)/n*100:.2f}pp)")

    print("\n>>> 独有命中 (仅该路命中, 四路基线都漏):")
    print(f"    HSTU: {pct(hstu_unique)}")
    print(f"    MIND: {pct(mind_unique)}")
    print(f"    HSTU∪MIND: {pct(hstu_unique | mind_unique)}")
    print(f"    HSTU∩MIND (重叠独有): {pct(hstu_unique & mind_unique)}")

    results = {
        'standalone': {k: len(v) for k, v in hit_sets.items()},
        'four_union': len(four_union),
        'plus_hstu': len(five_hstu),
        'plus_mind': len(five_mind),
        'six_union': len(six_union),
        'hstu_unique': len(hstu_unique),
        'mind_unique': len(mind_unique),
        'hstu_or_mind_unique': len(hstu_unique | mind_unique),
        'n_users': n,
    }
    out = os.path.join(config.RESULT_PATH, 'union_gain_results.json')
    os.makedirs(config.RESULT_PATH, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n>>> Results saved to {out}")
    print(f"\n>>> Total time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
