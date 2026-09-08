"""
完整推理流程: 多路召回 + DIN-Extended 精排 (batch 推理) + 提交生成

DIN 推理优化: 每用户所有候选打包成单 batch，N 次 forward → 1 次 forward
"""
# 必须在 import numpy / torch 之前设置, 避免 DataLoader 多线程与 OpenBLAS 冲突
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import pickle, time
import numpy as np
import torch
from tqdm import tqdm

import config
from data_loader import (
    get_all_click_df, load_articles, build_user_features,
    get_user_item_time, get_item_topk_click
)
from data_loader_ext import (
    load_raw_meta, build_extended_encoders, build_extended_item_features,
    build_extended_user_features, merge_jsonl_features,
)
from model import TwoTowerV2Model
from model_ext import DINExtendedModel
from recall_fusion import multi_channel_recall
from evaluate import split_train_val


def _save_csv(user_recall_items_dict, out_dir):
    """Save {user_id: [(item_id, score), ...]} → auto-named CSV."""
    import pandas as pd, os, re
    from datetime import datetime
    cats = ','.join(str(c) for c in getattr(config, 'AMAZON_CATEGORIES', ('multi',)))
    recall = getattr(config, 'RECALL_NUM', 50)
    topk = getattr(config, 'FINAL_RECOMMEND_NUM', 5)
    auto = re.sub(r'[\\/:*?"<>|]', '_',
                  f'preds_{cats}_recall{recall}_top{topk}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    rows = [[uid] + [it[0] for it in items] for uid, items in user_recall_items_dict.items()]
    n = len(rows[0]) - 1 if rows else 0
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, auto)
    pd.DataFrame(rows, columns=['user_id'] + [f'item_{i+1}' for i in range(n)]).to_csv(path, index=False)
    print(f"[OK] {len(rows):,} users x {n} → {path}")
    return path


def load_din_ext_model(device):
    """加载 DINExtendedModel (优先 best, 回退 default)"""
    candidates = [
        (config.DIN_EXT_BEST_FILE, "DIN-Ext best"),
        (config.DIN_EXT_MODEL_FILE, "DIN-Ext default"),
    ]
    for path, label in candidates:
        if os.path.exists(path):
            print(f"    Loading {label} from: {path}")
            ckpt = torch.load(path, map_location=device, weights_only=False)
            cfg = ckpt['config']
            model = DINExtendedModel(
                num_users=cfg['num_users'], num_items=cfg['num_items'],
                num_brands=cfg['num_brands'], num_categories=cfg['num_categories'],
                embed_dim=cfg.get('embed_dim', 256),
                brand_embed_dim=cfg.get('brand_embed_dim', 64),
                hidden_dims=cfg['hidden_dims'], hist_len=cfg['hist_len'],
                dropout=cfg.get('dropout', 0.1),
            ).to(device)
            model.load_state_dict(ckpt['model_state_dict'])
            model.eval()
            limits = {
                'num_users': cfg['num_users'],
                'num_items': cfg['num_items'],
                'num_brands': cfg['num_brands'],
                'num_categories': cfg['num_categories'],
            }
            print(f"    Loaded epoch {ckpt.get('epoch', '?')}, metrics: {ckpt.get('metrics', {})}")
            print(f"    Limits: users={limits['num_users']}, items={limits['num_items']}, "
                  f"brands={limits['num_brands']}, cats={limits['num_categories']}")
            return model, limits
    print("    No DIN-Ext model found, using recall scores only")
    return None, None


def din_rerank_batch(user_batch, item_indices, din_model, device,
                     item_cat_arr, item_brand_arr, item_click_arr, item_created_arr,
                     item_avg_rating_arr, item_rating_num_arr,
                     din_max_item=None, din_max_brand=None, din_max_cat=None):
    """对单个用户的全部候选商品做批量 DINExtended 推理。"""
    N = len(item_indices)
    idx_arr = np.array(item_indices, dtype=np.int64)
    if din_max_item is not None:
        idx_arr = np.clip(idx_arr, 0, din_max_item - 1)

    def _safe(arr):
        a = arr[idx_arr].copy()
        return a

    item_cat  = np.clip(_safe(item_cat_arr), 0, din_max_cat - 1) if din_max_cat else _safe(item_cat_arr)
    item_brand = np.clip(_safe(item_brand_arr), 0, din_max_brand - 1) if din_max_brand else _safe(item_brand_arr)

    B_user = user_batch['user_id'].size(0)
    batch = {
        # User side (repeat per candidate)
        'user_id':       user_batch['user_id'].repeat(N),
        'hist_items':    user_batch['hist_items'].repeat(N, 1),
        'hist_brands':   user_batch['hist_brands'].repeat(N, 1),
        'hist_ratings':  user_batch['hist_ratings'].repeat(N, 1),
        'hist_time_deltas': user_batch['hist_time_deltas'].repeat(N, 1),
        'hist_verified': user_batch['hist_verified'].repeat(N, 1),
        'hist_len':      user_batch['hist_len'].repeat(N),
        'click_count':   user_batch['click_count'].repeat(N),
        'time_span':     user_batch['time_span'].repeat(N),
        'user_avg_rating':    user_batch['user_avg_rating'].repeat(N),
        'user_std_rating':    user_batch['user_std_rating'].repeat(N),
        'user_verified_ratio': user_batch['user_verified_ratio'].repeat(N),
        'user_avg_helpful':    user_batch['user_avg_helpful'].repeat(N),
        # Item side
        'item_id':           torch.from_numpy(idx_arr).to(device),
        'category_id':       torch.from_numpy(item_cat).to(device),
        'brand_id':          torch.from_numpy(item_brand).to(device),
        'item_click_count':  torch.from_numpy(_safe(item_click_arr)).to(device),
        'created_at_ts':     torch.from_numpy(_safe(item_created_arr)).to(device),
        'item_avg_rating':   torch.from_numpy(_safe(item_avg_rating_arr)).to(device),
        'item_rating_number': torch.from_numpy(_safe(item_rating_num_arr)).to(device),
    }

    with torch.no_grad():
        logits = din_model(batch)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


def din_rerank_chunk(records, din_model, device, item_arrays, hist_len,
                     din_max_item, din_max_brand, din_max_cat):
    """
    批量精排: 一次前向处理多个用户的全部候选 (替代逐用户循环, GPU 不再空转)。

    records: list of (user_idx, feat, valid_cands)
      valid_cands = [(item_idx, recall_score, item_id), ...]  (item_idx 已确认在 embedding 范围内)
    item_arrays: (item_cat_arr, item_brand_arr, item_click_arr, item_created_arr,
                  item_avg_rating_arr, item_rating_num_arr)
    Returns: list of [(item_id, final_score), ...] (已按分数降序), 与 records 顺序对齐
    """
    (item_cat_arr, item_brand_arr, item_click_arr, item_created_arr,
     item_avg_rating_arr, item_rating_num_arr) = item_arrays

    user_ids, hist_items_l, hist_brands_l, hist_ratings_l, hist_deltas_l, hist_verified_l = [], [], [], [], [], []
    hist_lens, click_counts, time_spans, u_avg_ratings, u_std_ratings, u_verified_ratios, u_avg_helpfuls = [], [], [], [], [], [], []
    item_ids, item_cats, item_brands, item_clicks, item_createds, item_avg_rs, item_rating_nums = [], [], [], [], [], [], []
    recall_scores, raw_item_ids, split_sizes = [], [], []

    for user_idx, feat, valid_cands in records:
        # ---- 用户历史序列: 裁剪 + padding 到 hist_len (与 _build_user_batch 语义一致) ----
        h  = feat['hist']
        hb = feat.get('hist_brands', []) or []
        hr = feat.get('hist_ratings', []) or []
        hd = feat.get('hist_time_deltas', []) or []
        hv = feat.get('hist_verified', []) or []
        if din_max_item is not None:
            h = [x if x < din_max_item else 0 for x in h]
        if din_max_brand is not None:
            hb = [b if b < din_max_brand else 0 for b in hb]
        h = h[-hist_len:]; hb = hb[-hist_len:]; hr = hr[-hist_len:]
        hd = hd[-hist_len:]; hv = hv[-hist_len:]
        hl = len(h)
        h  = h  + [0] * (hist_len - len(h));  hb = hb + [0] * (hist_len - len(hb))
        hr = hr + [0] * (hist_len - len(hr)); hd = hd + [0] * (hist_len - len(hd))
        hv = hv + [0] * (hist_len - len(hv))

        cn = feat['click_norm']; sn = feat['span_norm']
        ar = feat.get('avg_rating', 0); sr = feat.get('std_rating', 0)
        vr = feat.get('user_verified_ratio', 0); ah = feat.get('user_avg_helpful', 0)

        n = len(valid_cands)
        split_sizes.append(n)
        for _ in range(n):
            user_ids.append(user_idx)
            hist_items_l.append(h); hist_brands_l.append(hb); hist_ratings_l.append(hr)
            hist_deltas_l.append(hd); hist_verified_l.append(hv)
            hist_lens.append(hl)
            click_counts.append(cn); time_spans.append(sn)
            u_avg_ratings.append(ar); u_std_ratings.append(sr)
            u_verified_ratios.append(vr); u_avg_helpfuls.append(ah)

        for item_idx, rs, item_id in valid_cands:
            item_ids.append(item_idx)
            item_cats.append(int(item_cat_arr[item_idx]))
            item_brands.append(int(item_brand_arr[item_idx]))
            item_clicks.append(float(item_click_arr[item_idx]))
            item_createds.append(float(item_created_arr[item_idx]))
            item_avg_rs.append(float(item_avg_rating_arr[item_idx]))
            item_rating_nums.append(float(item_rating_num_arr[item_idx]))
            recall_scores.append(rs)
            raw_item_ids.append(item_id)

    M = len(user_ids)
    item_ids_arr = np.clip(np.asarray(item_ids, dtype=np.int64), 0, din_max_item - 1) if din_max_item else np.asarray(item_ids, dtype=np.int64)
    item_cats_arr = np.clip(np.asarray(item_cats, dtype=np.int64), 0, din_max_cat - 1) if din_max_cat else np.asarray(item_cats, dtype=np.int64)
    item_brands_arr = np.clip(np.asarray(item_brands, dtype=np.int64), 0, din_max_brand - 1) if din_max_brand else np.asarray(item_brands, dtype=np.int64)

    batch = {
        'user_id':           torch.LongTensor(user_ids).to(device),
        'hist_items':        torch.LongTensor(hist_items_l).to(device),
        'hist_brands':       torch.LongTensor(hist_brands_l).to(device),
        'hist_ratings':      torch.FloatTensor(hist_ratings_l).to(device),
        'hist_time_deltas':  torch.FloatTensor(hist_deltas_l).to(device),
        'hist_verified':     torch.LongTensor(hist_verified_l).to(device),
        'hist_len':          torch.LongTensor(hist_lens).to(device),
        'click_count':       torch.FloatTensor(click_counts).to(device),
        'time_span':         torch.FloatTensor(time_spans).to(device),
        'user_avg_rating':    torch.FloatTensor(u_avg_ratings).to(device),
        'user_std_rating':    torch.FloatTensor(u_std_ratings).to(device),
        'user_verified_ratio': torch.FloatTensor(u_verified_ratios).to(device),
        'user_avg_helpful':    torch.FloatTensor(u_avg_helpfuls).to(device),
        'item_id':           torch.LongTensor(item_ids_arr).to(device),
        'category_id':       torch.LongTensor(item_cats_arr).to(device),
        'brand_id':          torch.LongTensor(item_brands_arr).to(device),
        'item_click_count':  torch.FloatTensor(item_clicks).to(device),
        'created_at_ts':     torch.FloatTensor(item_createds).to(device),
        'item_avg_rating':   torch.FloatTensor(item_avg_rs).to(device),
        'item_rating_number': torch.FloatTensor(item_rating_nums).to(device),
    }

    with torch.no_grad():
        logits = din_model(batch)
        probs = torch.sigmoid(logits).cpu().numpy()

    recall_arr = np.asarray(recall_scores, dtype=np.float64)
    alpha = 0.7
    final = alpha * probs + (1 - alpha) * recall_arr / (np.abs(recall_arr) + 1.0)

    results = []
    offset = 0
    for n in split_sizes:
        ids = raw_item_ids[offset:offset + n]
        fs = final[offset:offset + n]
        results.append([(i, float(s)) for i, s in sorted(zip(ids, fs), key=lambda x: -x[1])])
        offset += n
    return results


def _build_user_batch(user_idx, feat, hist_len, device,
                       din_max_item=None, din_max_brand=None):
    """构造单用户的基础 extended batch (含 brand/rating/time_delta/verified)."""
    if isinstance(feat, dict):
        hist         = feat['hist']
        hist_brands  = feat.get('hist_brands', [0]*len(hist))
        hist_ratings = feat.get('hist_ratings', [0]*len(hist))
        hist_deltas  = feat.get('hist_time_deltas', [0]*len(hist))
        hist_verified = feat.get('hist_verified', [0]*len(hist))
        click_norm   = feat['click_norm']
        span_norm    = feat['span_norm']
        avg_rating   = feat.get('avg_rating', 0)
        std_rating   = feat.get('std_rating', 0)
        verified_r   = feat.get('user_verified_ratio', 0)
        avg_helpful  = feat.get('user_avg_helpful', 0)
    else:
        # DataFrame row fallback
        hist         = feat['hist_items']
        hist_brands  = feat.get('hist_brands', [0]*len(hist))
        hist_ratings = feat.get('hist_ratings', [0]*len(hist))
        hist_deltas  = feat.get('hist_time_deltas', [0]*len(hist))
        hist_verified = feat.get('hist_verified', [0]*len(hist))
        click_norm   = float(feat.get('click_count_norm', 0) or 0)
        span_norm    = float(feat.get('time_span_norm', 0) or 0)
        avg_rating   = float(feat.get('user_avg_rating', 0) or 0)
        std_rating   = float(feat.get('user_std_rating', 0) or 0)
        verified_r   = float(feat.get('user_verified_ratio', 0) or 0)
        avg_helpful  = float(feat.get('user_avg_helpful', 0) or 0)

    # 安全裁剪: 索引不超过 model embedding 范围
    if din_max_item is not None:
        mask = [h < din_max_item for h in hist]
        hist = [h if ok else 0 for h, ok in zip(hist, mask)]
        hist_brands  = [b if ok else 0 for b, ok in zip(hist_brands, mask)]
        hist_ratings = [r if ok else 0 for r, ok in zip(hist_ratings, mask)]
        hist_deltas  = [d if ok else 0 for d, ok in zip(hist_deltas, mask)]
        hist_verified = [v if ok else 0 for v, ok in zip(hist_verified, mask)]
    if din_max_brand is not None:
        hist_brands = [b if b < din_max_brand else 0 for b in hist_brands]

    hl = len(hist)
    if hl > hist_len:
        hist = hist[-hist_len:]; hist_brands = hist_brands[-hist_len:]
        hist_ratings = hist_ratings[-hist_len:]; hist_deltas = hist_deltas[-hist_len:]
        hist_verified = hist_verified[-hist_len:]; hl = hist_len

    pad = lambda seq: seq + [0] * (hist_len - len(seq))
    return {
        'user_id':        torch.LongTensor([user_idx]).to(device),
        'hist_items':     torch.LongTensor([pad(hist)]).to(device),
        'hist_brands':    torch.LongTensor([pad(hist_brands)]).to(device),
        'hist_ratings':   torch.FloatTensor([pad(hist_ratings)]).to(device),
        'hist_time_deltas': torch.FloatTensor([pad(hist_deltas)]).to(device),
        'hist_verified':  torch.LongTensor([pad(hist_verified)]).to(device),
        'hist_len':        torch.LongTensor([hl]).to(device),
        'click_count':     torch.FloatTensor([click_norm]).to(device),
        'time_span':       torch.FloatTensor([span_norm]).to(device),
        'user_avg_rating':    torch.FloatTensor([avg_rating]).to(device),
        'user_std_rating':    torch.FloatTensor([std_rating]).to(device),
        'user_verified_ratio': torch.FloatTensor([verified_r]).to(device),
        'user_avg_helpful':    torch.FloatTensor([avg_helpful]).to(device),
    }, hl


def inference():
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Using device: {device}")

    # ================================================================
    # Step 1: 加载数据 (CSV 5-core, 131K 用户 + raw_meta brand)
    # ================================================================
    print("\n" + "=" * 60)
    print("Step 1: Loading data...")
    print("=" * 60)

    click_df = get_all_click_df(config.DATA_PATH, offline=config.OFFLINE_MODE)
    raw_meta = load_raw_meta(config.DATA_PATH, config.AMAZON_CATEGORIES)
    # Merge JSONL extra features (verified/helpful) into CSV click_df
    click_df = merge_jsonl_features(click_df, config.DATA_PATH, config.AMAZON_CATEGORIES)

    # --- 共享 encoder (全量数据, 包含 brand_id) ---
    encoders = build_extended_encoders(click_df, raw_meta, config.EXT_ENCODER_PKL)
    user_le = encoders['user_id']
    item_le = encoders['item_id']
    num_users = len(user_le.classes_)
    num_items = len(item_le.classes_)
    num_categories = len(encoders['category_id'].classes_)
    num_brands = len(encoders['brand_id'].classes_)
    print(f">>> num_users={num_users:,}, num_items={num_items:,}, "
          f"num_categories={num_categories}, num_brands={num_brands:,}")

    # --- V2 Recall: 使用正样本部分 (click_label==1) ---
    pos_click = click_df[click_df['click_label'] == 1].copy()
    articles_df = load_articles(config.DATA_PATH)

    # ================================================================
    # Key: 时序分割 — 用户历史 & ItemCF 只能用训练窗口内的数据，
    # 不能包含验证集未来交互，否则验证目标会被精确排除。
    # ================================================================
    print(">>> Temporal split: train window for user history & ItemCF...")
    train_click, val_click = split_train_val(click_df, config.EVAL_SPLIT_RATIO)

    # --- 用户特征: 只用训练集构建 (避免验证目标泄露进历史) ---
    print(">>> Building extended user features (train window only)...")
    user_features = build_extended_user_features(
        train_click, raw_meta, encoders, hist_len=config.HIST_LEN
    )
    # 额外补充 V2 recall 需要的 user rating stats
    # build_extended_user_features 的列名: hist_items, click_count_norm, time_span_norm, user_avg_rating, user_std_rating
    # embedding_recall 需要: hist_items_trunc, click_count_norm, time_span_norm, user_avg_rating_norm, user_std_rating_norm
    if 'hist_items_trunc' not in user_features.columns:
        user_features['hist_items_trunc'] = user_features['hist_items']
        user_features['user_avg_rating_norm'] = user_features['user_avg_rating'] / 5.0
        s = user_features['user_std_rating'].astype(float)
        user_features['user_std_rating_norm'] = (s - s.min()) / (s.max() - s.min() + 1e-8)
    if 'user_avg_rating_norm' not in user_features.columns:
        print(">>> Merging user rating stats for recall...")
        user_rating_stats = train_click.groupby('user_id')['rating'].agg(['mean', 'std']).fillna(0)
        user_rating_stats.columns = ['user_avg_rating', 'user_std_rating']
        user_rating_stats = user_rating_stats.reset_index()
        user_rating_stats['user_avg_rating_norm'] = user_rating_stats['user_avg_rating'] / 5.0
        user_rating_stats['user_std_rating_norm'] = (user_rating_stats['user_std_rating'] / 2.0).clip(0, 1)
        user_features = user_features.merge(
            user_rating_stats[['user_id', 'user_avg_rating_norm', 'user_std_rating_norm']],
            on='user_id', how='left'
        )
        user_features['user_avg_rating_norm'] = user_features['user_avg_rating_norm'].fillna(0)
        user_features['user_std_rating_norm'] = user_features['user_std_rating_norm'].fillna(0)

    # --- 物品特征 (扩展版, 含 brand + quality) ---
    item_features = build_extended_item_features(train_click, raw_meta, encoders)

    # ================================================================
    # --- 预构建数组 (O(1) 查表, 避免逐行扫 DataFrame) ---
    # ================================================================
    # 用户特征字典 (DIN-Ext 需要的全字段)
    user_feat_dict = {}
    for _, row in user_features.iterrows():
        uid = row['user_id']
        h  = row.get('hist_items', [])
        user_feat_dict[uid] = {
            'hist': h,
            'hist_brands':  row.get('hist_brands', [0]*len(h) if isinstance(h, list) else []),
            'hist_ratings': row.get('hist_ratings', [0]*len(h) if isinstance(h, list) else []),
            'hist_time_deltas': row.get('hist_time_deltas', [0]*len(h) if isinstance(h, list) else []),
            'hist_verified': row.get('hist_verified', [0]*len(h) if isinstance(h, list) else []),
            'click_norm':   float(row.get('click_count_norm', 0) or 0),
            'span_norm':    float(row.get('time_span_norm', 0) or 0),
            'avg_rating':   float(row.get('user_avg_rating_norm', 0) or 0),
            'std_rating':   float(row.get('user_std_rating_norm', 0) or 0),
            'user_verified_ratio': float(row.get('user_verified_ratio', 0) or 0),
            'user_avg_helpful':    float(row.get('user_avg_helpful_norm', 0) or 0),
        }
    # V2 recall 也需要两层 key 的 (hist/click_norm/span_norm/avg_rating/std_rating)
    # user_feat_dict 直接用上面已包含这些字段的 dict 即可

    # 物品特征数组
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

    raw_to_item_enc = {cls: i for i, cls in enumerate(item_le.classes_)}

    # 只对验证集用户做召回+精排 (提交目标即 val 用户, 避免对 83 万全量用户做无用功)
    target_users = val_click['user_id'].unique()
    print(f">>> Target users (val only): {len(target_users):,} | Items: {num_items:,}")

    # ================================================================
    # Step 2: ItemCF (只用训练窗口, 不能包含验证交互)
    # ================================================================
    print("\nStep 2: Building ItemCF similarity (train window only)...")
    # ItemCF 只用正样本 (rating≥4), 1-2 星负样本不应贡献 item-item 共现信号
    user_item_time_dict = get_user_item_time(train_click[train_click['click_label'] == 1])
    # 用全量 click_df 生成 item_topk_click，不用 train_click (避免热门计算偏差)
    # 全量统计流行度不会泄露验证目标 (Popularity 是全局属性不是个性化)
    item_topk_click = get_item_topk_click(click_df, k=200)

    if os.path.exists(config.ITEMCF_SIM_PKL):
        with open(config.ITEMCF_SIM_PKL, 'rb') as f:
            i2i_sim = pickle.load(f)
        print(">>> Loaded cached ItemCF similarity")
    else:
        from itemcf import itemcf_sim
        i2i_sim = itemcf_sim(user_item_time_dict)
        os.makedirs(config.MODEL_PATH, exist_ok=True)
        with open(config.ITEMCF_SIM_PKL, 'wb') as f:
            pickle.dump(i2i_sim, f)
        print(">>> Computed and cached ItemCF similarity")

    # ================================================================
    # Step 3: Load embeddings
    # ================================================================
    print("\nStep 3: Loading item embeddings...")
    all_item_vecs = None
    for path, label in [(config.V2_EMBED_PKL, "V2")]:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                all_item_vecs = pickle.load(f)
            print(f">>> Using {label} embeddings, shape: {all_item_vecs.shape}")
            break
    if all_item_vecs is None:
        print(">>> No embeddings found, dual-tower recall will be skipped")

    # ================================================================
    # Step 4: Multi-Channel Recall
    # ================================================================
    print("\nStep 4: Multi-Channel Recall...")
    recall_results = multi_channel_recall(
        target_users, train_click, user_item_time_dict, i2i_sim,
        item_topk_click, articles_df, user_features, item_features,
        encoders, all_item_vecs, hist_len=config.HIST_LEN,
        final_recall_num=config.RECALL_NUM
    )

    # ---- 召回天花板诊断: 用验证集 ground truth 测候选池覆盖率 ----
    # HR@100 = 候选池里至少含一个正样本的用户比例 = 最终 HR@5 的理论上限
    val_gt = val_click.groupby('user_id')['click_article_id'].apply(set).to_dict()

    def _pool_items(pool):
        # recall_results 的值可能是 {item: score} 或 [(item, score)]，两种都兼容
        return set(pool.keys()) if isinstance(pool, dict) else set(i for i, _ in pool)

    diag_hit100, diag_cov_sum, diag_n_val = 0, 0.0, 0
    for _u, _gt in val_gt.items():
        _pool = recall_results.get(_u)
        if not _pool:
            continue
        _inter = _gt & _pool_items(_pool)
        if _inter:
            diag_hit100 += 1
        diag_cov_sum += len(_inter) / len(_gt)
        diag_n_val += 1
    ceiling_hr100 = diag_hit100 / max(diag_n_val, 1)
    ceiling_recall = diag_cov_sum / max(diag_n_val, 1)
    print(f">>> [诊断] 召回天花板 HR@100={ceiling_hr100:.4f} ({diag_hit100}/{diag_n_val}) "
          f"| Recall@100={ceiling_recall:.4f}")

    # ================================================================
    # Step 5: Load DIN-Ext model
    # ================================================================
    print("\nStep 5: Loading DIN-Ext model...")
    din_model, din_limits = load_din_ext_model(device)

    # ================================================================
    # Step 6: DIN-Ext Batch Reranking
    # ================================================================
    print("\nStep 6: Final ranking (DIN-Ext batch inference)...")
    user_recall_items_dict = {}
    hist_len = config.HIST_LEN

    din_max_user  = din_limits['num_users']     if din_limits else 2**31 - 1
    din_max_item  = din_limits['num_items']     if din_limits else 2**31 - 1
    din_max_brand = din_limits['num_brands']    if din_limits else 2**31 - 1
    din_max_cat   = din_limits['num_categories'] if din_limits else 2**31 - 1

    # --- 预计算 user_id → user_idx (一次性向量化 transform, 替代逐用户 searchsorted) ---
    target_users_arr = np.array(target_users)
    target_user_idxs = np.clip(user_le.transform(target_users_arr), 0, din_max_user - 1)

    # --- 第一遍: 轻量收集每个用户的候选 (纯 Python, 无 GPU/无 tensor, 秒级) ---
    # records: (user_idx, feat, valid_cands, fallback_cands, raw_user_id)
    #   valid_cands   = [(item_idx, recall_score, item_id), ...]
    #   fallback_cands = [(item_id, recall_score), ...]  (item_idx 越界/缺失, 保留原始 recall 分数)
    records = []
    for raw_user_id, uidx in zip(target_users_arr, target_user_idxs):
        recalled_items = recall_results.get(raw_user_id, [])
        if isinstance(recalled_items, dict):
            recalled_items = list(recalled_items.items())

        if len(recalled_items) == 0:
            topk = item_topk_click[:config.FINAL_RECOMMEND_NUM]
            user_recall_items_dict[raw_user_id] = [(iid, -999.0) for iid in topk]
            continue

        feat = user_feat_dict.get(raw_user_id)
        if din_model is None or feat is None:
            sorted_items = sorted(recalled_items, key=lambda x: -x[1])[:config.FINAL_RECOMMEND_NUM]
            user_recall_items_dict[raw_user_id] = sorted_items
            continue

        valid_cands = []
        fallback_cands = []
        for item_id, recall_score in recalled_items:
            item_idx = raw_to_item_enc.get(item_id)
            if item_idx is None or item_idx >= din_max_item:
                fallback_cands.append((item_id, recall_score))
            else:
                valid_cands.append((item_idx, recall_score, item_id))

        if not valid_cands:
            sorted_items = sorted(recalled_items, key=lambda x: -x[1])[:config.FINAL_RECOMMEND_NUM]
            user_recall_items_dict[raw_user_id] = sorted_items
            continue

        records.append((int(uidx), feat, valid_cands, fallback_cands, raw_user_id))

    # --- 第二遍: 分批做 DIN 前向, 每批打包所有 (user, candidate) 对 ---
    item_arrays = (item_cat_arr, item_brand_arr, item_click_arr, item_created_arr,
                   item_avg_rating_arr, item_rating_num_arr)
    RERANK_CHUNK = 256  # 每批用户数 (~256*50=12800 行/前向, 显存安全且 GPU 打满)
    for i in tqdm(range(0, len(records), RERANK_CHUNK), desc="DIN Rerank (batched)"):
        chunk = records[i:i + RERANK_CHUNK]
        din_scores_list = din_rerank_chunk(
            [(r[0], r[1], r[2]) for r in chunk], din_model, device, item_arrays, hist_len,
            din_max_item, din_max_brand, din_max_cat
        )
        for (_, _, valid_cands, fallback_cands, raw_user_id), din_scores in zip(chunk, din_scores_list):
            merged = list(din_scores)
            if fallback_cands:
                merged.extend(fallback_cands)
            merged.sort(key=lambda x: -x[1])
            final_topk = merged[:config.FINAL_RECOMMEND_NUM]

            # 补齐热门兜底
            if len(final_topk) < config.FINAL_RECOMMEND_NUM:
                existing = set(x[0] for x in final_topk)
                for iid in item_topk_click:
                    if iid not in existing:
                        final_topk.append((iid, -999.0))
                        existing.add(iid)
                    if len(final_topk) >= config.FINAL_RECOMMEND_NUM:
                        break

            user_recall_items_dict[raw_user_id] = final_topk

    # ---- 最终命中 + 精排损失诊断 ----
    diag_hit5 = 0
    for _u, _gt in val_gt.items():
        _rec = user_recall_items_dict.get(_u)
        if not _rec:
            continue
        if _gt & set(_it for _it, _ in _rec):
            diag_hit5 += 1
    final_hr5 = diag_hit5 / max(diag_n_val, 1)
    _gap = ceiling_hr100 - final_hr5
    print(f">>> [诊断] 最终 HR@5={final_hr5:.4f} ({diag_hit5}/{diag_n_val}) "
          f"| 精排损失 Gap={_gap:.4f}")
    if ceiling_hr100 < 0.9:
        _bottleneck = "召回侧(天花板低)"
    elif _gap > 0.1:
        _bottleneck = "精排侧(Gap大)"
    else:
        _bottleneck = "均衡"
    print(f">>> [诊断] 瓶颈判定: {_bottleneck}")

    print("\nStep 7: Saving predictions...")
    _save_csv(user_recall_items_dict, config.RESULT_PATH)
    print(f">>> Full pipeline complete! Total time: {time.time() - start_time:.2f}s")

    # --- Quick stats dump for the user ---
    scores = []
    n_users = len(user_recall_items_dict)
    for uid, items in user_recall_items_dict.items():
        scores.extend([s for _, s in items if s > -999.0])
    print(f"\n{'='*60}")
    print(f"  Stats: {n_users} users, {len(scores)/max(n_users,1):.1f} items/user avg")
    if scores:
        print(f"  Scores: min={min(scores):.4f}  max={max(scores):.4f}  mean={sum(scores)/len(scores):.4f}  median={sorted(scores)[len(scores)//2]:.4f}")
    # Show top-5 for first 3 users
    for uid in list(user_recall_items_dict.keys())[:3]:
        items = user_recall_items_dict[uid][:5]
        print(f"  {uid}: {[(str(i), round(s,4)) for i,s in items]}")
    print(f"{'='*60}")


if __name__ == "__main__":
    inference()
