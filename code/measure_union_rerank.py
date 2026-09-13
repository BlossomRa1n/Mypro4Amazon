"""
六路并集池 vs 加权 top-100 的 DIN 精排对比测量。

回答: 扩大召回桶(纯并集合并) 能否提升最终 HR@5。

流程:
  1. 复用 recall_union_gain.py 的六路召回 (ItemCF/V2/HSTU/MIND/Cat/Hot) + 100K 采样用户 (seed42)
  2. 构造两个候选池:
     - baseline: 四路加权 top-100 (当前生产, 预期 recall HR@100 ≈ 13.76%)
     - union:    六路并集去重 (~300-400 候选/用户, 预期 recall ≈ 22.02%)
  3. 复用 inference_full.py 的 DIN 精排 (din_rerank_chunk) 分别精排 → top-5
  4. 量两个池的最终 HR@5 + 召回覆盖率 + 各阶段耗时

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
                           category_preference_recall, hot_recall,
                           merge_recall_results)
from recall_union_gain import mind_recall
from inference_full import load_din_ext_model, din_rerank_chunk

SAMPLE_USERS = 100000
BASELINE_WEIGHTS = [1.5, 3.0, 1.0, 0.05]   # 当前生产四路权重 (13.76%)
BASELINE_CHANNELS = ['itemcf', 'v2', 'category', 'hot']


def rank_score(channel_dict, u):
    """某路召回 dict → rank 分 {item: 1/(rank+1)}, 跨路尺度无关, 供并集合并取 max。"""
    items = channel_dict.get(u, {})
    if isinstance(items, dict):
        items = sorted(items.items(), key=lambda x: -x[1])
    return {item: 1.0 / (r + 1) for r, (item, _) in enumerate(items)}


def build_union_pool(channels, users):
    """六路并集去重, 每 item 取跨路最高 rank 分 → {u: {item: score}}。"""
    pool = {}
    for u in tqdm(users, desc="Union merge"):
        merged = {}
        for ch in channels:
            for item, s in rank_score(ch, u).items():
                if s > merged.get(item, -1.0):
                    merged[item] = s
        pool[u] = merged
    return pool


def pool_hit_rate(pool, val_gt, users):
    """召回覆盖率: 候选池里至少含一个 gt 的用户比例 (baseline=HR@100, union=HR@池大小)。"""
    hit = 0
    for u in users:
        gt = val_gt.get(u, set())
        if not gt:
            continue
        items = pool.get(u)
        if not items:
            continue
        pool_items = set(items.keys()) if isinstance(items, dict) else set(i for i, _ in items)
        if gt & pool_items:
            hit += 1
    return hit / max(len(users), 1)


def din_rerank_pool(pool, sample_users, user_le, din_model, din_limits,
                    user_feat_dict, item_arrays, raw_to_item_enc, hist_len,
                    item_topk_click, device, label, chunk_size=256):
    """对候选池做 DIN 精排 → {raw_uid: [(item_id, final_score), ...] 按分数降序, 取 top5}。

    复用 inference_full.py 的 din_rerank_chunk; 流程与其 Step 6 对齐。
    """
    (item_cat_arr, item_brand_arr, item_click_arr, item_created_arr,
     item_avg_rating_arr, item_rating_num_arr) = item_arrays

    din_max_user  = din_limits['num_users']     if din_limits else 2**31 - 1
    din_max_item  = din_limits['num_items']     if din_limits else 2**31 - 1
    din_max_brand = din_limits['num_brands']    if din_limits else 2**31 - 1
    din_max_cat   = din_limits['num_categories'] if din_limits else 2**31 - 1

    sample_users_arr = np.array(sample_users)
    user_idxs = np.clip(user_le.transform(sample_users_arr), 0, din_max_user - 1)

    # 第一遍: 轻量收集候选 (纯 Python, 无 GPU)
    records = []
    result = {}
    for raw_uid, uidx in zip(sample_users_arr, user_idxs):
        recalled = pool.get(raw_uid, {})
        if isinstance(recalled, dict):
            recalled = list(recalled.items())
        if len(recalled) == 0:
            topk = item_topk_click[:config.FINAL_RECOMMEND_NUM]
            result[raw_uid] = [(iid, -999.0) for iid in topk]
            continue
        feat = user_feat_dict.get(raw_uid)
        if din_model is None or feat is None:
            result[raw_uid] = sorted(recalled, key=lambda x: -x[1])[:config.FINAL_RECOMMEND_NUM]
            continue
        valid_cands, fallback_cands = [], []
        for item_id, rs in recalled:
            item_idx = raw_to_item_enc.get(item_id)
            if item_idx is None or item_idx >= din_max_item:
                fallback_cands.append((item_id, rs))
            else:
                valid_cands.append((item_idx, rs, item_id))
        if not valid_cands:
            result[raw_uid] = sorted(recalled, key=lambda x: -x[1])[:config.FINAL_RECOMMEND_NUM]
            continue
        records.append((int(uidx), feat, valid_cands, fallback_cands, raw_uid))

    # 第二遍: 分批 DIN 前向
    RERANK_CHUNK = chunk_size
    for i in tqdm(range(0, len(records), RERANK_CHUNK), desc=f"DIN Rerank [{label}]"):
        chunk = records[i:i + RERANK_CHUNK]
        din_scores_list = din_rerank_chunk(
            [(r[0], r[1], r[2]) for r in chunk], din_model, device, item_arrays, hist_len,
            din_max_item, din_max_brand, din_max_cat
        )
        for (_, _, valid_cands, fallback_cands, raw_uid), din_scores in zip(chunk, din_scores_list):
            merged = list(din_scores)
            if fallback_cands:
                merged.extend(fallback_cands)
            merged.sort(key=lambda x: -x[1])
            final_topk = merged[:config.FINAL_RECOMMEND_NUM]
            if len(final_topk) < config.FINAL_RECOMMEND_NUM:
                existing = set(x[0] for x in final_topk)
                for iid in item_topk_click:
                    if iid not in existing:
                        final_topk.append((iid, -999.0))
                        existing.add(iid)
                    if len(final_topk) >= config.FINAL_RECOMMEND_NUM:
                        break
            result[raw_uid] = final_topk
    return result


def final_hr5(result, val_gt, users):
    hit = 0
    for u in users:
        gt = val_gt.get(u, set())
        rec = result.get(u)
        if not gt or not rec:
            continue
        if gt & set(i for i, _ in rec):
            hit += 1
    return hit / max(len(users), 1)


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Device: {device}")

    # ================= Step 1: 数据加载 (与 recall_union_gain.py 一致) =================
    print("\n[Step 1] Loading data...")
    click_df = get_all_click_df(config.DATA_PATH, offline=config.OFFLINE_MODE)
    raw_meta = load_raw_meta(config.DATA_PATH, config.AMAZON_CATEGORIES)
    click_df = merge_jsonl_features(click_df, config.DATA_PATH, config.AMAZON_CATEGORIES)
    train_click, val_click = split_train_val(click_df, config.EVAL_SPLIT_RATIO)

    user_item_time_dict = get_user_item_time(train_click[train_click['click_label'] == 1])
    if os.path.exists(config.ITEMCF_SIM_PKL):
        with open(config.ITEMCF_SIM_PKL, 'rb') as f:
            i2i_sim = pickle.load(f)
        print("    Loaded cached ItemCF similarity")
    else:
        from itemcf import itemcf_sim
        i2i_sim = itemcf_sim(user_item_time_dict)
        with open(config.ITEMCF_SIM_PKL, 'wb') as f:
            pickle.dump(i2i_sim, f)

    val_gt = val_click.groupby('user_id')['click_article_id'].apply(set).to_dict()
    item_topk_click = get_item_topk_click(click_df, k=200)
    all_val_users = list(val_gt.keys())
    rng = np.random.default_rng(42)
    if len(all_val_users) > SAMPLE_USERS:
        sample_users = rng.choice(all_val_users, SAMPLE_USERS, replace=False).tolist()
    else:
        sample_users = all_val_users
    n = len(sample_users)
    print(f"    Sampled {n:,} val users | val_gt {len(val_gt):,} users")

    # ================= Step 2: encoders + 特征 (含 DIN 需要全字段) =================
    print("\n[Step 2] Building encoders & features...")
    encoders = build_extended_encoders(click_df, raw_meta, config.EXT_ENCODER_PKL)
    user_features = build_extended_user_features(train_click, raw_meta, encoders, hist_len=config.HIST_LEN)
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

    # DIN 需要的物品特征数组 (O(1) 查表)
    item_le = encoders['item_id']
    num_items = len(item_le.classes_)
    item_cat_arr = np.zeros(num_items, dtype=np.int64)
    item_brand_arr = np.zeros(num_items, dtype=np.int64)
    item_click_arr = np.zeros(num_items, dtype=np.float32)
    item_created_arr = np.zeros(num_items, dtype=np.float32)
    item_avg_rating_arr = np.zeros(num_items, dtype=np.float32)
    item_rating_num_arr = np.zeros(num_items, dtype=np.float32)
    for idx in item_features.index:
        item_cat_arr[idx] = int(item_features.loc[idx].get('category_idx', 0))
        item_brand_arr[idx] = int(item_features.loc[idx].get('brand_idx', 0))
        item_click_arr[idx] = float(item_features.loc[idx].get('item_click_count_norm', 0))
        item_created_arr[idx] = float(item_features.loc[idx].get('created_at_ts_norm', 0))
        item_avg_rating_arr[idx] = float(item_features.loc[idx].get('item_avg_rating_norm', 0))
        item_rating_num_arr[idx] = float(item_features.loc[idx].get('item_rating_number_norm', 0))

    # DIN 需要的完整用户特征 dict (与 inference_full.py Step 1 对齐)
    user_feat_dict = {}
    for _, row in user_features.iterrows():
        uid = row['user_id']
        h = row.get('hist_items', [])
        user_feat_dict[uid] = {
            'hist': h,
            'hist_brands': row.get('hist_brands', [0] * len(h) if isinstance(h, list) else []),
            'hist_ratings': row.get('hist_ratings', [0] * len(h) if isinstance(h, list) else []),
            'hist_time_deltas': row.get('hist_time_deltas', [0] * len(h) if isinstance(h, list) else []),
            'hist_verified': row.get('hist_verified', [0] * len(h) if isinstance(h, list) else []),
            'click_norm': float(row.get('click_count_norm', 0) or 0),
            'span_norm': float(row.get('time_span_norm', 0) or 0),
            'avg_rating': float(row.get('user_avg_rating_norm', 0) or 0),
            'std_rating': float(row.get('user_std_rating_norm', 0) or 0),
            'user_verified_ratio': float(row.get('user_verified_ratio', 0) or 0),
            'user_avg_helpful': float(row.get('user_avg_helpful_norm', 0) or 0),
        }
    raw_to_item_enc = {cls: i for i, cls in enumerate(item_le.classes_)}
    item_arrays = (item_cat_arr, item_brand_arr, item_click_arr, item_created_arr,
                   item_avg_rating_arr, item_rating_num_arr)

    # ================= Step 3: 六路召回 =================
    print("\n[Step 3] Six-channel recall...")
    t_recall = time.time()

    itemcf_dict = itemcf_recall(sample_users, user_item_time_dict, i2i_sim,
                                item_topk_click, sim_item_topk=10,
                                recall_item_num=config.RECALL_NUM)
    with open(config.V2_EMBED_PKL, 'rb') as f:
        v2_vecs = pickle.load(f)
    v2_dict = embedding_recall(sample_users, {}, train_click, user_features, item_features,
                               encoders, v2_vecs, item_topk_click,
                               hist_len=config.HIST_LEN, recall_item_num=config.RECALL_NUM,
                               weight=1.0, model_path=config.V2_BEST_FILE, channel_label="V2 Phase2")
    cat_dict = category_preference_recall(sample_users, train_click, articles_df, item_topk_click,
                                          recall_item_num=20, weight=0.5)
    hot_dict = hot_recall(sample_users, item_topk_click, recall_item_num=5, weight=0.1)

    with open(config.HSTU_EMBED_PKL, 'rb') as f:
        hstu_vecs = pickle.load(f)
    hstu_dict = embedding_recall(sample_users, {}, train_click, user_features, item_features,
                                 encoders, hstu_vecs, item_topk_click,
                                 hist_len=config.HIST_LEN, recall_item_num=config.RECALL_NUM,
                                 weight=1.0, model_path=config.HSTU_BEST_FILE, channel_label="HSTU")

    mind_ckpt = torch.load(config.MIND_MODEL_FILE, map_location=device, weights_only=False)
    mind_cfg = mind_ckpt['config']
    from model import MINDModel
    mind_model = MINDModel(**mind_cfg).to(device)
    mind_model.load_state_dict(mind_ckpt['model_state_dict'])
    mind_model.eval()
    with open(config.MIND_EMBED_PKL, 'rb') as f:
        mind_vecs = pickle.load(f)
    num_items_mind = min(mind_cfg['num_items'], mind_vecs.shape[0])
    K = mind_cfg['num_interests']
    per_interest = 40
    # MIND 召回需要 minimal 用户特征 (hist/click_norm/span_norm)
    mind_user_feat = {}
    for _, row in user_features.iterrows():
        uid = row['user_id']
        mind_user_feat[uid] = {
            'hist': row.get('hist_items_trunc', row.get('hist_items', [])),
            'click_norm': row['click_count_norm'],
            'span_norm': row['time_span_norm'],
        }
    train_enc = train_click['click_article_id'].map(raw_to_item_enc).dropna().astype(int)
    train_pairs = train_click.loc[train_enc.index, ['user_id']].copy()
    train_pairs['item_enc'] = train_enc.values
    train_items = train_pairs.groupby('user_id')['item_enc'].apply(set).to_dict()
    mind_dict = mind_recall(sample_users, mind_user_feat, train_items, mind_model, mind_vecs,
                            num_items_mind, config.HIST_LEN, K, per_interest,
                            config.RECALL_NUM, encoders, device)
    del mind_model
    torch.cuda.empty_cache()

    print(f"    [Recall done] {time.time() - t_recall:.1f}s")

    # ================= Step 4: 构造两个候选池 =================
    print("\n[Step 4] Building candidate pools...")
    baseline_pool = merge_recall_results(
        [itemcf_dict, v2_dict, cat_dict, hot_dict], BASELINE_WEIGHTS,
        final_recall_num=config.RECALL_NUM)
    union_pool = build_union_pool(
        [itemcf_dict, v2_dict, cat_dict, hot_dict, hstu_dict, mind_dict], sample_users)

    b_sizes = [len(v) for v in baseline_pool.values()]
    u_sizes = [len(v) for v in union_pool.values()]
    print(f"    baseline pool: avg {np.mean(b_sizes):.1f} items/user (min {min(b_sizes)}, max {max(b_sizes)})")
    print(f"    union pool:    avg {np.mean(u_sizes):.1f} items/user (min {min(u_sizes)}, max {max(u_sizes)})")

    recall_hit_base = pool_hit_rate(baseline_pool, val_gt, sample_users)
    recall_hit_union = pool_hit_rate(union_pool, val_gt, sample_users)
    print(f"    [召回覆盖] baseline HR@100 = {recall_hit_base:.4f}")
    print(f"    [召回覆盖] union HR@池   = {recall_hit_union:.4f}")

    # ================= Step 5: 加载 DIN + 精排 =================
    print("\n[Step 5] Loading DIN-Ext model...")
    din_model, din_limits = load_din_ext_model(device)
    hist_len = config.HIST_LEN

    print("\n[Step 6] DIN rerank: baseline pool...")
    t_base = time.time()
    base_top5 = din_rerank_pool(
        baseline_pool, sample_users, encoders['user_id'], din_model, din_limits,
        user_feat_dict, item_arrays, raw_to_item_enc, hist_len, item_topk_click,
        device, "baseline")
    t_base_din = time.time() - t_base
    torch.cuda.empty_cache()

    print("\n[Step 7] DIN rerank: union pool...")
    t_union = time.time()
    union_top5 = din_rerank_pool(
        union_pool, sample_users, encoders['user_id'], din_model, din_limits,
        user_feat_dict, item_arrays, raw_to_item_enc, hist_len, item_topk_click,
        device, "union", chunk_size=48)
    t_union_din = time.time() - t_union

    # ================= Step 8: 最终 HR@5 =================
    hr5_base = final_hr5(base_top5, val_gt, sample_users)
    hr5_union = final_hr5(union_top5, val_gt, sample_users)

    print("\n" + "=" * 60)
    print(">>> 结果汇总 (100K 采样用户, seed42)")
    print("=" * 60)
    print(f"  召回覆盖  baseline (加权 top-100): {recall_hit_base:.4f}")
    print(f"  召回覆盖  union    (六路并集)    : {recall_hit_union:.4f}")
    print(f"  最终 HR@5  baseline: {hr5_base:.4f}")
    print(f"  最终 HR@5  union   : {hr5_union:.4f}")
    print(f"  增益: HR@5 {hr5_base:.4f} -> {hr5_union:.4f} "
          f"({(hr5_union - hr5_base) * 100:+.2f}pp)")
    print(f"  DIN 精排耗时: baseline {t_base_din:.1f}s | union {t_union_din:.1f}s")
    print(f"  总耗时: {time.time() - t0:.1f}s")

    results = {
        'recall_hit_baseline': recall_hit_base,
        'recall_hit_union': recall_hit_union,
        'hr5_baseline': hr5_base,
        'hr5_union': hr5_union,
        'hr5_gain_pp': (hr5_union - hr5_base) * 100,
        'baseline_avg_pool': float(np.mean(b_sizes)),
        'union_avg_pool': float(np.mean(u_sizes)),
        'din_time_baseline_s': t_base_din,
        'din_time_union_s': t_union_din,
        'total_time_s': time.time() - t0,
        'n_users': n,
    }
    out = os.path.join(config.RESULT_PATH, 'union_rerank_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n>>> Results saved to {out}")


if __name__ == "__main__":
    main()
