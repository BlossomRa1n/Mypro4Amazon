"""
Extended data loader for Amazon Reviews 2023 Raw (All_Beauty).
Uses raw JSONL data (review + meta) to build richer features:

离散特征: user_id, item_id, brand_id, verified_purchase
稠密特征: user_avg_rating, user_std_rating, user_verified_ratio, user_avg_helpful,
         item_avg_rating, item_rating_number, hist_len, click_count, time_span
序列特征: hist_items, hist_brands, hist_ratings, hist_time_deltas
"""
import pandas as pd
import numpy as np
import json
import os
import pickle
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from utils import reduce_mem
import config


# ============================================================
# Raw Data Loading
# ============================================================

def load_raw_reviews(data_path, categories, offline=False):
    """
    Load raw review JSONL with all fields: user_id, parent_asin, asin,
    rating, timestamp, helpful_vote, verified_purchase.

    Memory-optimized: builds typed column lists then constructs DataFrame
    with minimal dtypes to avoid 2× memory from list-of-dicts + DataFrame.
    """
    all_rows = []
    for cat in categories:
        path_candidates = [
            os.path.join(data_path, 'raw', 'review_categories', f'{cat}.jsonl'),
            os.path.join(data_path, f'{cat}_reviews.jsonl'),
        ]
        path = None
        for p in path_candidates:
            if os.path.exists(p):
                path = p
                break
        if path is None:
            raise FileNotFoundError(
                f"Raw review JSONL not found for {cat}. Tried: {path_candidates}\n"
                f"Run: python -c \"from huggingface_hub import hf_hub_download; "
                f"hf_hub_download('McAuley-Lab/Amazon-Reviews-2023', "
                f"'raw/review_categories/{cat}.jsonl', repo_type='dataset', "
                f"local_dir='{data_path}')\""
            )
        print(f">>> Loading raw reviews: {os.path.relpath(path, data_path)}")
        nrows = 10000 if offline else None

        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, desc=f"  {cat} reviews")):
                if nrows and i >= nrows:
                    break
                try:
                    obj = json.loads(line.strip())
                    all_rows.append((
                        obj['user_id'],
                        obj['parent_asin'],
                        obj.get('asin', obj['parent_asin']),
                        float(obj['rating']),
                        int(obj['timestamp']),
                        int(obj.get('helpful_vote', 0)),
                        int(obj.get('verified_purchase', False)),
                    ))
                except (json.JSONDecodeError, KeyError):
                    continue

    # Build DataFrame from typed lists — far less memory than list-of-dicts
    user_ids, parent_asins, asins, ratings, timestamps, helpfuls, verifieds = zip(*all_rows)
    del all_rows  # free intermediate list immediately

    df = pd.DataFrame({
        'user_id': pd.array(user_ids, dtype='string'),
        'parent_asin': pd.array(parent_asins, dtype='string'),
        'asin': pd.array(asins, dtype='string'),
        'rating': np.array(ratings, dtype='float32'),
        'timestamp': np.array(timestamps, dtype='int64'),
        'helpful_vote': np.array(helpfuls, dtype='int32'),
        'verified_purchase': np.array(verifieds, dtype='int8'),
    })
    del user_ids, parent_asins, asins, ratings, timestamps, helpfuls, verifieds
    print(f">>> Raw reviews loaded: {len(df):,} rows "
          f"(users={df['user_id'].nunique():,}, items={df['parent_asin'].nunique():,})")
    return df


def load_raw_meta(data_path, categories):
    """
    Load raw meta JSONL: parent_asin → store/brand, avg_rating, rating_number,
    price, skin_type, item_form.

    Memory-optimized: typed column arrays, immediate DataFrame construction.
    """
    meta_rows = []
    for cat in categories:
        path_candidates = [
            os.path.join(data_path, 'raw', 'meta_categories', f'meta_{cat}.jsonl'),
            os.path.join(data_path, f'{cat}_meta.jsonl'),
        ]
        path = None
        for p in path_candidates:
            if os.path.exists(p):
                path = p
                break
        if path is None:
            raise FileNotFoundError(
                f"Raw meta JSONL not found for {cat}. Tried: {path_candidates}\n"
                f"Run: python -c \"from huggingface_hub import hf_hub_download; "
                f"hf_hub_download('McAuley-Lab/Amazon-Reviews-2023', "
                f"'raw/meta_categories/meta_{cat}.jsonl', repo_type='dataset', "
                f"local_dir='{data_path}')\""
            )
        print(f">>> Loading raw meta: {os.path.relpath(path, data_path)}")

        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"  {cat} meta"):
                try:
                    obj = json.loads(line.strip())

                    # Extract brand: prefer 'store', fallback to details.Brand
                    store = str(obj.get('store', '')).strip()
                    details = obj.get('details', {})
                    details_brand = ''
                    skin_type = ''
                    item_form = ''
                    if isinstance(details, dict):
                        details_brand = str(details.get('Brand', '')).strip()
                        skin_type = str(details.get('Skin Type', '')).strip()
                        item_form = str(details.get('Item Form', '')).strip()
                    brand = store if store else details_brand

                    # Parse price
                    price_raw = obj.get('price')
                    try:
                        price = float(price_raw) if price_raw is not None else -1.0
                    except (ValueError, TypeError):
                        price = -1.0

                    meta_rows.append((
                        obj['parent_asin'],
                        brand if brand else 'Unknown',
                        obj.get('main_category', cat),
                        float(obj.get('average_rating', 0)),
                        int(obj.get('rating_number', 0)),
                        price,
                        skin_type,
                        item_form,
                    ))
                except (json.JSONDecodeError, KeyError):
                    continue

    if not meta_rows:
        return pd.DataFrame()

    parent_asins, brands, main_cats, avg_ratings, rating_nums, prices, skin_types, item_forms = zip(*meta_rows)
    del meta_rows

    df = pd.DataFrame({
        'parent_asin': pd.array(parent_asins, dtype='string'),
        'brand': pd.array(brands, dtype='string'),
        'main_category': pd.array(main_cats, dtype='string'),
        'average_rating': np.array(avg_ratings, dtype='float32'),
        'rating_number': np.array(rating_nums, dtype='int32'),
        'price': np.array(prices, dtype='float32'),
        'skin_type': pd.array(skin_types, dtype='string'),
        'item_form': pd.array(item_forms, dtype='string'),
    })
    del parent_asins, brands, main_cats, avg_ratings, rating_nums, prices, skin_types, item_forms

    print(f">>> Raw meta loaded: {len(df):,} items")
    print(f"    brands: {df['brand'].nunique():,} unique")
    has_price = (df['price'] > 0).sum()
    print(f"    price available: {has_price:,}/{len(df):,} "
          f"({has_price/max(len(df),1)*100:.0f}%)")
    print(f"    avg_rating available: {(df['average_rating'] > 0).sum():,}/{len(df):,}")
    return df


# ============================================================
# Data Preparation — Rating → Label + Filter
# ============================================================

RATING_MAP = {5.0: (1, 1.0), 4.0: (1, 0.5), 2.0: (0, 0.8), 1.0: (0, 0.8)}


def prepare_click_df(raw_reviews_df, min_user_inter=5, min_item_inter=5):
    """
    Convert raw reviews to labeled click dataframe with density filtering.

    Args:
        raw_reviews_df: from load_raw_reviews()
        min_user_inter: minimum interactions per user (default 5 → "5-core")
        min_item_inter: minimum interactions per item

    Returns:
        click_df with columns: user_id, click_article_id, click_timestamp,
                                click_label, click_weight, helpful_vote,
                                verified_purchase
    """
    df = raw_reviews_df.copy()

    # Rating → label & weight (3分丢弃) — vectorized, was per-row map+zip
    # Pre-build label/weight arrays for all 5 rating values
    rating_bins = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    label_lookup = np.array([0, 0, -1, 1, 1], dtype=np.int8)  # 3→-1 sentinel
    weight_lookup = np.array([0.8, 0.8, -1.0, 0.5, 1.0], dtype=np.float32)
    # Digitize ratings to bin indices (1.0→0, 2.0→1, ..., 5.0→4)
    r_idx = np.searchsorted(rating_bins, df['rating'].values, side='right') - 1
    r_idx = np.clip(r_idx, 0, 4)
    click_label = label_lookup[r_idx]
    click_weight = weight_lookup[r_idx]
    # Drop 3-star (label==-1)
    keep = click_label >= 0
    df = df.loc[keep].copy()
    df['click_label'] = click_label[keep]
    df['click_weight'] = click_weight[keep]

    df = df.rename(columns={
        'parent_asin': 'click_article_id',
        'timestamp': 'click_timestamp'
    })

    # Keep: user_id, click_article_id, click_timestamp, click_label,
    #       click_weight, helpful_vote, verified_purchase
    cols = ['user_id', 'click_article_id', 'click_timestamp',
            'click_label', 'click_weight', 'rating', 'helpful_vote', 'verified_purchase']
    df = df[[c for c in cols if c in df.columns]]

    # Density filtering — iterative (like 5-core)
    print(f">>> Before density filter: {len(df):,} interactions, "
          f"users={df['user_id'].nunique():,}, items={df['click_article_id'].nunique():,}")
    while True:
        prev = len(df)
        user_counts = df.groupby('user_id').size()
        valid_users = user_counts[user_counts >= min_user_inter].index
        df = df[df['user_id'].isin(valid_users)]
        item_counts = df.groupby('click_article_id').size()
        valid_items = item_counts[item_counts >= min_item_inter].index
        df = df[df['click_article_id'].isin(valid_items)]
        if len(df) == prev:
            break
    print(f">>> After density filter: {len(df):,} interactions, "
          f"users={df['user_id'].nunique():,}, items={df['click_article_id'].nunique():,}")

    df = df.drop_duplicates(['user_id', 'click_article_id', 'click_timestamp'])
    df = reduce_mem(df)

    pos = df[df['click_label'] == 1]
    neg = df[df['click_label'] == 0]
    print(f"    正样本 (4-5分): {len(pos):,} ({len(pos)/max(len(df),1)*100:.0f}%) | "
          f"显式负样本 (1-2分): {len(neg):,} ({len(neg)/max(len(df),1)*100:.0f}%)")
    return df


# ============================================================
# Extended Encoders
# ============================================================

def build_extended_encoders(click_df, meta_df, encoder_path):
    """
    Build LabelEncoders for user_id, item_id, brand_id, category_id.
    """
    if os.path.exists(encoder_path):
        print(f">>> Loading extended encoders from {encoder_path}...")
        with open(encoder_path, 'rb') as f:
            return pickle.load(f)

    print(">>> Building extended ID encoders...")
    encoders = {}

    # user_id
    user_le = LabelEncoder()
    user_le.fit(click_df['user_id'].unique())
    encoders['user_id'] = user_le
    encoders['raw_to_idx'] = {raw: idx for idx, raw in enumerate(user_le.classes_)}

    # item_id
    item_le = LabelEncoder()
    all_items = np.union1d(
        click_df['click_article_id'].unique(),
        meta_df['parent_asin'].unique()
    )
    item_le.fit(all_items)
    encoders['item_id'] = item_le

    # brand_id — map known brands from meta, map unseen to 0
    brand_le = LabelEncoder()
    brands_raw = meta_df['brand'].unique()
    brands = [str(b) for b in brands_raw if b is not None and str(b) != '<NA>']
    brand_le.fit(['__PAD__', '__UNKNOWN__'] + brands)
    encoders['brand_id'] = brand_le

    # category_id — from main_category in meta
    cat_le = LabelEncoder()
    cats_raw = meta_df['main_category'].unique()
    cats = [str(c) for c in cats_raw if c is not None and str(c) != '<NA>']
    cat_le.fit(['__PAD__'] + cats)
    encoders['category_id'] = cat_le

    os.makedirs(os.path.dirname(encoder_path), exist_ok=True)
    with open(encoder_path, 'wb') as f:
        pickle.dump(encoders, f)
    print(f">>> Extended encoders saved to {encoder_path}")
    print(f"    users={len(user_le.classes_):,}, items={len(item_le.classes_):,}, "
          f"brands={len(brand_le.classes_):,}, categories={len(cat_le.classes_):,}")
    return encoders


# ============================================================
# Extended Feature Building
# ============================================================

def build_extended_item_features(click_df, meta_df, encoders):
    """
    Build item feature table indexed by encoded item_idx.
    Columns: category_idx, brand_idx, item_click_count_norm, created_at_ts_norm,
             item_avg_rating_norm, item_rating_number_norm

    Optimized: replaces two Python for-loops over 112K+ items with
    pandas vectorized map + numpy advanced-indexing scatter.
    """
    print(">>> Building extended item features...")
    item_le = encoders['item_id']
    cat_le = encoders['category_id']
    brand_le = encoders['brand_id']

    num_items = len(item_le.classes_)
    item_cat_arr = np.zeros(num_items, dtype=np.int64)
    item_brand_arr = np.zeros(num_items, dtype=np.int64)
    item_click_arr = np.zeros(num_items, dtype=np.float32)
    item_created_arr = np.zeros(num_items, dtype=np.float32)
    item_avg_rating_arr = np.zeros(num_items, dtype=np.float32)
    item_rating_num_arr = np.zeros(num_items, dtype=np.float32)

    # --- Pre-build raw_id → item_idx mapping ---
    raw_to_item_enc = {cls: i for i, cls in enumerate(item_le.classes_)}

    # --- Category: vectorized map (was Python for-loop) ---
    cat_map = {'__PAD__': 0}
    for i, cls in enumerate(cat_le.classes_):
        cat_map[cls] = i

    # --- Brand: vectorized map ---
    brand_map = {'__PAD__': 0}
    for i, cls in enumerate(brand_le.classes_):
        brand_map[cls] = i

    # --- Fill from meta: vectorized scatter (was for-loop over 112K rows) ---
    meta_idx = meta_df['parent_asin'].map(raw_to_item_enc)
    is_valid = meta_idx.notna() & (meta_idx >= 0) & (meta_idx < num_items)
    meta_valid_idx = meta_idx[is_valid].astype(np.int64).values

    item_cat_arr[meta_valid_idx] = (
        meta_df.loc[is_valid, 'main_category'].map(cat_map).fillna(0).astype(np.int64).values
    )
    item_brand_arr[meta_valid_idx] = (
        meta_df.loc[is_valid, 'brand'].map(brand_map).fillna(0).astype(np.int64).values
    )
    item_avg_rating_arr[meta_valid_idx] = (
        meta_df.loc[is_valid, 'average_rating'].fillna(0).astype(np.float32).values
    )
    item_rating_num_arr[meta_valid_idx] = (
        meta_df.loc[is_valid, 'rating_number'].fillna(0).astype(np.float32).values
    )

    # --- Click count + created_at: vectorized scatter (was for-loop over 112K items) ---
    ts_min = click_df['click_timestamp'].min()
    ts_range = click_df['click_timestamp'].max() - ts_min + 1e-8

    click_counts = click_df.groupby('click_article_id').size()
    created_at = click_df.groupby('click_article_id')['click_timestamp'].min()

    click_min = click_counts.min()
    click_range = click_counts.max() - click_min + 1e-8

    # Map raw IDs → item indices (both Series share same index)
    cc_idx = click_counts.index.map(raw_to_item_enc)
    cc_valid = cc_idx.notna() & (cc_idx >= 0) & (cc_idx < num_items)
    cc_valid_idx = cc_idx[cc_valid].astype(np.int64).values

    item_click_arr[cc_valid_idx] = (
        (click_counts[cc_valid].values - click_min) / click_range
    ).astype(np.float32)
    item_created_arr[cc_valid_idx] = (
        (created_at[cc_valid].fillna(ts_min).values - ts_min) / ts_range
    ).astype(np.float32)

    # --- Normalize meta-derived features ---
    ar_max = item_avg_rating_arr.max()
    if ar_max > 0:
        item_avg_rating_arr = item_avg_rating_arr / ar_max

    rn_max = item_rating_num_arr.max()
    if rn_max > 0:
        item_rating_num_arr = np.log1p(item_rating_num_arr) / np.log1p(rn_max + 1e-8)

    # --- Build final DataFrame ---
    item_features = pd.DataFrame({
        'category_idx': item_cat_arr,
        'brand_idx': item_brand_arr,
        'item_click_count_norm': item_click_arr,
        'created_at_ts_norm': item_created_arr,
        'item_avg_rating_norm': item_avg_rating_arr,
        'item_rating_number_norm': item_rating_num_arr,
    }, index=range(num_items))

    print(f">>> Item features: {len(item_features)} items × 6 features")
    return item_features


def build_extended_user_features(click_df, meta_df, encoders, hist_len=50):
    """
    Build extended user features with:
    - History sequences: hist_items, hist_brands, hist_ratings, hist_time_deltas
    - Stats: click_count_norm, time_span_norm, user_avg_rating_norm,
             user_std_rating_norm, user_verified_ratio, user_avg_helpful_norm

    Optimized: pre-builds item→brand lookup array (O(1) per item instead of
    O(n_brands) LabelEncoder transform), uses numpy vectorized ops for time
    deltas, and sorts once globally instead of per-group.
    """
    print(">>> Building extended user features...")
    item_le = encoders['item_id']
    brand_le = encoders['brand_id']
    num_items = len(item_le.classes_)

    # --- Pre-build item_idx → brand_idx array (vectorized, one-shot) ---
    # This replaces the per-item get_brand_idx() which did a linear
    # brand_le.transform() scan — the #1 bottleneck for large datasets.
    meta = meta_df.copy()
    meta['item_idx'] = item_le.transform(meta['parent_asin'])
    # Map brand strings → label indices via dict (O(1) per unique brand)
    brand_map = {b: i for i, b in enumerate(brand_le.classes_)}
    meta['brand_idx'] = meta['brand'].map(brand_map).fillna(0).astype(np.int64)

    item_to_brand_arr = np.zeros(num_items, dtype=np.int64)
    meta_item_idx = meta['item_idx'].values
    meta_brand_idx_col = meta['brand_idx'].values
    # Vectorized scatter: only valid indices
    valid = (meta_item_idx >= 0) & (meta_item_idx < num_items)
    item_to_brand_arr[meta_item_idx[valid]] = meta_brand_idx_col[valid]

    # --- Sort once globally, then groupby preserves order ---
    click_df = click_df.copy()
    click_df['item_idx'] = item_le.transform(click_df['click_article_id'])
    click_df = click_df.sort_values(['user_id', 'click_timestamp'])

    # --- Per-user aggregation (loop unavoidable, but inner ops are vectorized) ---
    user_data = []
    for uid, grp in tqdm(click_df.groupby('user_id', sort=False),
                         desc="  building user features"):
        # Already sorted by timestamp — no per-group sort needed
        item_seq = grp['item_idx'].to_numpy(dtype=np.int64)
        rating_seq = grp['rating'].to_numpy(dtype=np.float32)
        ts_seq = grp['click_timestamp'].to_numpy(dtype=np.int64)
        verified_seq = grp['verified_purchase'].to_numpy(dtype=np.int64)
        helpful_seq = grp['helpful_vote'].to_numpy(dtype=np.float32)

        n = len(item_seq)
        stats_n = n  # original click_count uses full group size

        # Truncate to last hist_len
        if n > hist_len:
            item_seq = item_seq[-hist_len:]
            rating_seq = rating_seq[-hist_len:]
            ts_seq = ts_seq[-hist_len:]
            verified_seq = verified_seq[-hist_len:]
            helpful_seq = helpful_seq[-hist_len:]
            n = hist_len

        # Brand sequence — vectorized O(1) array lookup (was per-item O(n_brands))
        brand_seq = item_to_brand_arr[item_seq]

        # Time deltas — vectorized numpy (was Python for-loop)
        if n > 1:
            time_deltas = np.diff(ts_seq) / (1000.0 * 3600 * 24)  # ms → days
            np.clip(time_deltas, None, 365, out=time_deltas)       # cap at 1 year
            time_deltas = np.insert(time_deltas, 0, 0)              # first delta = 0
        else:
            time_deltas = np.array([0], dtype=np.float64)
        max_delta = float(time_deltas.max() or 1)
        time_deltas_norm = (time_deltas / (max_delta + 1e-8)).tolist()

        user_data.append({
            'user_id': uid,
            'hist_items': item_seq.tolist(),
            'hist_brands': brand_seq.tolist(),
            'hist_ratings': rating_seq.tolist(),
            'hist_time_deltas': time_deltas_norm,
            'hist_verified': verified_seq.tolist(),
            'hist_len': n,
            'click_count': stats_n,
            'time_span': int(ts_seq[-1] - ts_seq[0]) if n > 1 else 0,
            'user_avg_rating': float(rating_seq.mean()),
            'user_std_rating': float(rating_seq.std()) if n > 1 else 0.0,
            'user_verified_ratio': float(verified_seq.mean()),
            'user_avg_helpful': float(helpful_seq.mean()),
        })

    user_features = pd.DataFrame(user_data)

    # --- Normalize dense features ---
    # click_count_norm
    cc = user_features['click_count'].astype(float)
    user_features['click_count_norm'] = ((cc - cc.min()) / (cc.max() - cc.min() + 1e-8))

    # time_span_norm
    ts = user_features['time_span'].astype(float)
    user_features['time_span_norm'] = ((ts - ts.min()) / (ts.max() - ts.min() + 1e-8))

    # user_avg_rating_norm (already 1-5 scale)
    user_features['user_avg_rating_norm'] = user_features['user_avg_rating'] / 5.0

    # user_std_rating_norm
    sr = user_features['user_std_rating'].astype(float)
    user_features['user_std_rating_norm'] = ((sr - sr.min()) /
                                               (sr.max() - sr.min() + 1e-8))

    # user_avg_helpful_norm (log scale for long tail)
    ah = user_features['user_avg_helpful'].astype(float)
    user_features['user_avg_helpful_norm'] = np.log1p(ah) / np.log1p(ah.max() + 1e-8)

    # user_verified_ratio — already [0, 1], no normalization needed

    print(f">>> User features: {len(user_features)} users")
    print(f"    avg hist_len: {user_features['hist_len'].mean():.1f}, "
          f"median: {user_features['hist_len'].median():.0f}")
    return user_features


# ============================================================
# Hard Negative Index (ItemCF-based)
# ============================================================

def build_hard_negative_index_ext(i2i_sim, encoders, num_hard_negatives=4):
    """Hard Negative Mining: 从 ItemCF 相似但用户未点击的 item 中挖掘困难负样本"""
    print(">>> Building hard negative index from ItemCF similarity...")
    item_le = encoders['item_id']
    raw_to_enc = {cls: i for i, cls in enumerate(item_le.classes_)}
    import heapq

    hard_neg_index = {}
    for raw_item, sim_items in tqdm(i2i_sim.items(), desc="  HardNeg Index"):
        if raw_item not in raw_to_enc:
            continue
        item_idx = raw_to_enc[raw_item]

        top_k = heapq.nlargest(
            num_hard_negatives * 3, sim_items.items(),
            key=lambda x: x[1]
        )
        hard_negs = []
        for sim_item, _ in top_k:
            enc = raw_to_enc.get(sim_item)
            if enc is not None:
                hard_negs.append(enc)
            if len(hard_negs) >= num_hard_negatives:
                break
        hard_neg_index[item_idx] = hard_negs

    print(f">>> Built hard negative index for {len(hard_neg_index):,} items")
    return hard_neg_index


# ============================================================
# Extended DIN Dataset (BPR Pairwise)
# ============================================================

class DINExtendedDataset(Dataset):
    """
    Extended DIN dataset with brand, verified_purchase, item quality signals.

    Each sample = (user features, positive item features, negative item features).
    User history includes: items, brands, ratings, time_deltas, verified.

    Optimized: vectorized BPR-pair negative sampling with ItemCF Hard Negative
    Mining — prioritizes similar-but-unclicked items as negatives to force the
    model to learn fine-grained preference distinctions.
    """
    def __init__(self, click_df, user_features, item_features, encoders,
                 hist_len=50, neg_ratio=4, hard_neg_index=None, num_hard_negatives=4):
        self.user_le = encoders['user_id']
        self.item_le = encoders['item_id']
        self.brand_le = encoders['brand_id']
        self.raw_to_idx = encoders.get('raw_to_idx', None)

        click_df = click_df.copy()
        click_df['user_idx'] = self.user_le.transform(click_df['user_id'])
        click_df['item_idx'] = self.item_le.transform(click_df['click_article_id'])

        pos_df = click_df[click_df['click_label'] == 1]

        # --- Build uid_to_idx (vectorized, was iterrows) ---
        user_enc_map = self.user_le.transform(user_features['user_id'].unique())
        uid_to_idx = dict(zip(user_features['user_id'].unique(), user_enc_map))

        # --- Pre-build user feature arrays (vectorized, was two iterrows loops) ---
        num_users = len(self.user_le.classes_)
        self.user_hist_arr = [None] * num_users
        self.user_brand_hist_arr = [None] * num_users
        self.user_rating_hist_arr = [None] * num_users
        self.user_delta_hist_arr = [None] * num_users
        self.user_verified_hist_arr = [None] * num_users
        self.user_click_cnt_arr = np.zeros(num_users, dtype=np.float32)
        self.user_time_span_arr = np.zeros(num_users, dtype=np.float32)
        self.user_avg_rating_arr = np.zeros(num_users, dtype=np.float32)
        self.user_std_rating_arr = np.zeros(num_users, dtype=np.float32)
        self.user_verified_ratio_arr = np.zeros(num_users, dtype=np.float32)
        self.user_avg_helpful_arr = np.zeros(num_users, dtype=np.float32)

        uf_user_ids = user_features['user_id'].values
        uf_to_idx = np.array([uid_to_idx.get(uid, -1) for uid in uf_user_ids], dtype=np.int64)
        uf_valid = uf_to_idx >= 0

        # Direct numpy-array copy: user_features columns → pre-built arrays
        for i in range(len(uf_user_ids)):
            if not uf_valid[i]:
                continue
            uidx = int(uf_to_idx[i])
            self.user_hist_arr[uidx] = user_features.iloc[i]['hist_items']
            self.user_brand_hist_arr[uidx] = user_features.iloc[i]['hist_brands']
            self.user_rating_hist_arr[uidx] = user_features.iloc[i]['hist_ratings']
            self.user_delta_hist_arr[uidx] = user_features.iloc[i]['hist_time_deltas']
            self.user_verified_hist_arr[uidx] = user_features.iloc[i]['hist_verified']
            self.user_click_cnt_arr[uidx] = user_features.iloc[i]['click_count_norm']
            self.user_time_span_arr[uidx] = user_features.iloc[i]['time_span_norm']
            self.user_avg_rating_arr[uidx] = user_features.iloc[i]['user_avg_rating_norm']
            self.user_std_rating_arr[uidx] = user_features.iloc[i]['user_std_rating_norm']
            self.user_verified_ratio_arr[uidx] = user_features.iloc[i]['user_verified_ratio']
            self.user_avg_helpful_arr[uidx] = user_features.iloc[i]['user_avg_helpful_norm']

        # Also build uid_to_hist from the same arrays (single pass, no extra iterrows)
        self.uid_to_hist = {}
        for i in range(len(uf_user_ids)):
            if not uf_valid[i]:
                continue
            uidx = int(uf_to_idx[i])
            self.uid_to_hist[uidx] = {
                'items': self.user_hist_arr[uidx],
                'brands': self.user_brand_hist_arr[uidx],
                'ratings': self.user_rating_hist_arr[uidx],
                'time_deltas': self.user_delta_hist_arr[uidx],
                'verified': self.user_verified_hist_arr[uidx],
            }

        # --- Build BPR pairs with Hard Negative Mining ---
        self.pairs = []
        rng = np.random.default_rng(42)
        num_all_items = len(self.item_le.classes_)
        pad_offset = 1

        # Pre-compute per-user positive sets for exclusion
        user_pos_items = pos_df.groupby('user_idx')['item_idx'].apply(set).to_dict()

        pos_uids = pos_df['user_idx'].to_numpy(dtype=np.int64)
        pos_iids = pos_df['item_idx'].to_numpy(dtype=np.int64)
        n_pos = len(pos_uids)

        hard_neg_hits = 0  # count how many successful hard-neg placements

        for k in tqdm(range(n_pos), desc="  building BPR pairs", total=n_pos):
            uid = int(pos_uids[k])
            pid = int(pos_iids[k])
            pos_set = user_pos_items.get(uid, set())

            # Collect negative candidates for this positive sample
            neg_candidates = []

            # 1) Hard negatives from ItemCF: items similar to what user interacted
            #    with, but not actually clicked — forces fine-grained discrimination
            if hard_neg_index is not None:
                hist_items = self.user_hist_arr[uid] or []
                seen_hn = set()
                for h in hist_items:
                    if h > 0 and h in hard_neg_index:
                        for hn in hard_neg_index[h][:num_hard_negatives]:
                            if hn not in seen_hn and hn not in pos_set:
                                seen_hn.add(hn)
                                neg_candidates.append(hn)

            # 2) Fill remaining slots with random negatives
            while len(neg_candidates) < neg_ratio:
                r = int(rng.integers(pad_offset, num_all_items))
                if r not in pos_set and r not in neg_candidates:
                    neg_candidates.append(r)

            # Take first neg_ratio candidates (hard negs come first = priority)
            selected_negs = neg_candidates[:neg_ratio]
            if selected_negs and selected_negs[0] != -1 and hard_neg_index is not None:
                # Check if at least one hard neg was used
                hist_items_for_check = self.user_hist_arr[uid] or []
                if hist_items_for_check:
                    # First candidate is from hard_neg pool if hist had hits
                    hard_neg_hits += 1

            for neg_iid in selected_negs:
                self.pairs.append((uid, pid, int(neg_iid)))

        if hard_neg_index is not None:
            print(f">>> Hard negative samples used: {hard_neg_hits:,} / {n_pos:,} ({hard_neg_hits/max(n_pos,1)*100:.0f}%)")

        print(f">>> DINExtendedDataset: {len(self.pairs):,} BPR pairs")
        self.hist_len = hist_len

        # --- Pre-build item feature arrays (vectorized, was loc-per-row loop) ---
        # Direct numpy-copy: item_features is indexed 0..num_items-1
        nums = min(len(item_features), num_all_items)
        self.item_cat_arr = item_features['category_idx'].iloc[:nums].to_numpy(dtype=np.int64)
        self.item_brand_arr = item_features['brand_idx'].iloc[:nums].to_numpy(dtype=np.int64)
        self.item_click_arr = item_features['item_click_count_norm'].iloc[:nums].to_numpy(dtype=np.float32)
        self.item_created_arr = item_features['created_at_ts_norm'].iloc[:nums].to_numpy(dtype=np.float32)
        self.item_avg_rating_arr = item_features['item_avg_rating_norm'].iloc[:nums].to_numpy(dtype=np.float32)
        self.item_rating_num_arr = item_features['item_rating_number_norm'].iloc[:nums].to_numpy(dtype=np.float32)

        # Pad to full length if item_features is shorter than num_all_items
        if nums < num_all_items:
            self.item_cat_arr = np.pad(self.item_cat_arr, (0, num_all_items - nums))
            self.item_brand_arr = np.pad(self.item_brand_arr, (0, num_all_items - nums))
            self.item_click_arr = np.pad(self.item_click_arr, (0, num_all_items - nums))
            self.item_created_arr = np.pad(self.item_created_arr, (0, num_all_items - nums))
            self.item_avg_rating_arr = np.pad(self.item_avg_rating_arr, (0, num_all_items - nums))
            self.item_rating_num_arr = np.pad(self.item_rating_num_arr, (0, num_all_items - nums))

    def __len__(self):
        return len(self.pairs)

    def _pad_seq(self, seq, dtype=torch.long):
        """Pad sequence to hist_len with zeros."""
        actual_len = len(seq)
        if actual_len < self.hist_len:
            padded = list(seq) + [0] * (self.hist_len - actual_len)
        else:
            padded = list(seq)[-self.hist_len:]
            actual_len = self.hist_len
        return torch.tensor(padded, dtype=dtype), actual_len

    def _pad_seq_float(self, seq):
        """Pad float sequence to hist_len with zeros."""
        actual_len = len(seq)
        if actual_len < self.hist_len:
            padded = list(seq) + [0.0] * (self.hist_len - actual_len)
        else:
            padded = list(seq)[-self.hist_len:]
            actual_len = self.hist_len
        return torch.tensor(padded, dtype=torch.float32), min(actual_len, self.hist_len)

    def _get_user_data(self, user_idx):
        hist_items = self.user_hist_arr[user_idx] or []
        hist_brands = self.user_brand_hist_arr[user_idx] or []
        hist_ratings = self.user_rating_hist_arr[user_idx] or []
        hist_deltas = self.user_delta_hist_arr[user_idx] or []
        hist_verified = self.user_verified_hist_arr[user_idx] or []

        items_padded, hl = self._pad_seq(hist_items)
        brands_padded, _ = self._pad_seq(hist_brands)
        ratings_padded, _ = self._pad_seq_float(hist_ratings)
        deltas_padded, _ = self._pad_seq_float(hist_deltas)
        verified_padded, _ = self._pad_seq_float(hist_verified)

        return {
            'hist_items': items_padded,
            'hist_brands': brands_padded,
            'hist_ratings': ratings_padded,
            'hist_time_deltas': deltas_padded,
            'hist_verified': verified_padded,
            'hist_len': torch.tensor(min(hl, self.hist_len), dtype=torch.long),
            'click_count': torch.tensor(self.user_click_cnt_arr[user_idx], dtype=torch.float32),
            'time_span': torch.tensor(self.user_time_span_arr[user_idx], dtype=torch.float32),
            'user_avg_rating': torch.tensor(self.user_avg_rating_arr[user_idx], dtype=torch.float32),
            'user_std_rating': torch.tensor(self.user_std_rating_arr[user_idx], dtype=torch.float32),
            'user_verified_ratio': torch.tensor(self.user_verified_ratio_arr[user_idx], dtype=torch.float32),
            'user_avg_helpful': torch.tensor(self.user_avg_helpful_arr[user_idx], dtype=torch.float32),
        }

    def _get_item_data(self, item_idx):
        return {
            'item_id': torch.tensor(item_idx, dtype=torch.long),
            'category_id': torch.tensor(self.item_cat_arr[item_idx], dtype=torch.long),
            'brand_id': torch.tensor(self.item_brand_arr[item_idx], dtype=torch.long),
            'item_click_count': torch.tensor(self.item_click_arr[item_idx], dtype=torch.float32),
            'created_at_ts': torch.tensor(self.item_created_arr[item_idx], dtype=torch.float32),
            'item_avg_rating': torch.tensor(self.item_avg_rating_arr[item_idx], dtype=torch.float32),
            'item_rating_number': torch.tensor(self.item_rating_num_arr[item_idx], dtype=torch.float32),
        }

    def __getitem__(self, idx):
        user_idx, pos_item_idx, neg_item_idx = self.pairs[idx]

        user_data = self._get_user_data(user_idx)
        pos_data = self._get_item_data(pos_item_idx)
        neg_data = self._get_item_data(neg_item_idx)

        return {
            'user_id': torch.tensor(user_idx, dtype=torch.long),
            **user_data,
            # Positive item
            'pos_item_id': pos_data['item_id'],
            'pos_category_id': pos_data['category_id'],
            'pos_brand_id': pos_data['brand_id'],
            'pos_item_click_count': pos_data['item_click_count'],
            'pos_created_at_ts': pos_data['created_at_ts'],
            'pos_item_avg_rating': pos_data['item_avg_rating'],
            'pos_item_rating_number': pos_data['item_rating_number'],
            # Negative item
            'neg_item_id': neg_data['item_id'],
            'neg_category_id': neg_data['category_id'],
            'neg_brand_id': neg_data['brand_id'],
            'neg_item_click_count': neg_data['item_click_count'],
            'neg_created_at_ts': neg_data['created_at_ts'],
            'neg_item_avg_rating': neg_data['item_avg_rating'],
            'neg_item_rating_number': neg_data['item_rating_number'],
        }
