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
    """
    all_rows = []
    for cat in categories:
        # huggingface_hub downloads to raw/review_categories/<cat>.jsonl
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
                    all_rows.append({
                        'user_id': obj['user_id'],
                        'parent_asin': obj['parent_asin'],
                        'asin': obj.get('asin', obj['parent_asin']),
                        'rating': float(obj['rating']),
                        'timestamp': int(obj['timestamp']),
                        'helpful_vote': int(obj.get('helpful_vote', 0)),
                        'verified_purchase': int(obj.get('verified_purchase', False)),
                    })
                except (json.JSONDecodeError, KeyError):
                    continue

    df = pd.DataFrame(all_rows)
    print(f">>> Raw reviews loaded: {len(df):,} rows "
          f"(users={df['user_id'].nunique():,}, items={df['parent_asin'].nunique():,})")
    return df


def load_raw_meta(data_path, categories):
    """
    Load raw meta JSONL: parent_asin → store/brand, avg_rating, rating_number,
    price, skin_type, item_form.
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
                        price = float(price_raw) if price_raw is not None else None
                    except (ValueError, TypeError):
                        price = None

                    meta_rows.append({
                        'parent_asin': obj['parent_asin'],
                        'brand': brand if brand else 'Unknown',
                        'main_category': obj.get('main_category', cat),
                        'average_rating': float(obj.get('average_rating', 0)),
                        'rating_number': int(obj.get('rating_number', 0)),
                        'price': price,
                        'skin_type': skin_type,
                        'item_form': item_form,
                    })
                except (json.JSONDecodeError, KeyError):
                    continue

    df = pd.DataFrame(meta_rows)
    print(f">>> Raw meta loaded: {len(df):,} items")
    print(f"    brands: {df['brand'].nunique():,} unique")
    has_price = (df['price'].notna()).sum()
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

    # Rating → label & weight (3分丢弃)
    df['click_label'], df['click_weight'] = zip(*df['rating'].map(
        lambda r: RATING_MAP.get(r, (None, None))
    ))
    df = df.dropna(subset=['click_label']).copy()
    df['click_label'] = df['click_label'].astype('int8')
    df['click_weight'] = df['click_weight'].astype('float32')

    df = df.rename(columns={
        'parent_asin': 'click_article_id',
        'timestamp': 'click_timestamp'
    })

    # Keep: user_id, click_article_id, click_timestamp, click_label,
    #       click_weight, helpful_vote, verified_purchase
    cols = ['user_id', 'click_article_id', 'click_timestamp',
            'click_label', 'click_weight', 'helpful_vote', 'verified_purchase']
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
    brands = meta_df['brand'].unique()
    brand_le.fit(['__PAD__', '__UNKNOWN__'] + list(brands))
    encoders['brand_id'] = brand_le

    # category_id — from main_category in meta
    cat_le = LabelEncoder()
    cats = meta_df['main_category'].unique()
    cat_le.fit(['__PAD__'] + list(cats))
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
    """
    print(">>> Building extended item features...")
    item_le = encoders['item_id']
    cat_le = encoders['category_id']

    num_items = len(item_le.classes_)
    item_cat_arr = np.zeros(num_items, dtype=np.int64)
    item_brand_arr = np.zeros(num_items, dtype=np.int64)
    item_click_arr = np.zeros(num_items, dtype=np.float32)
    item_created_arr = np.zeros(num_items, dtype=np.float32)
    item_avg_rating_arr = np.zeros(num_items, dtype=np.float32)
    item_rating_num_arr = np.zeros(num_items, dtype=np.float32)

    # --- Category (fast: map via dict) ---
    cat_map = {'__PAD__': 0}
    for i, cls in enumerate(cat_le.classes_):
        cat_map[cls] = i
    meta_cat_series = meta_df['main_category'].map(cat_map).fillna(0).astype(np.int64).values
    meta_parent_asin = meta_df['parent_asin'].values

    # Build parent_asin → item_idx mapping
    raw_to_item_enc = {cls: i for i, cls in enumerate(item_le.classes_)}

    # --- Fill from meta in one pass ---
    for i in range(len(meta_df)):
        parent = meta_parent_asin[i]
        idx = raw_to_item_enc.get(parent, -1)
        if idx < 0 or idx >= num_items:
            continue
        item_cat_arr[idx] = meta_cat_series[i]
        item_avg_rating_arr[idx] = float(meta_df.iloc[i].get('average_rating', 0) or 0)
        item_rating_num_arr[idx] = float(meta_df.iloc[i].get('rating_number', 0) or 0)

    # --- Click count + created_at per item ---
    ts_min = click_df['click_timestamp'].min()
    ts_max = click_df['click_timestamp'].max()
    ts_range = ts_max - ts_min + 1e-8

    click_counts = click_df.groupby('click_article_id').size()
    click_min = click_counts.min()
    click_max = click_counts.max()
    click_range = click_max - click_min + 1e-8

    created_at = click_df.groupby('click_article_id')['click_timestamp'].min()

    for raw_id, count_val in click_counts.items():
        idx = raw_to_item_enc.get(raw_id, -1)
        if idx < 0 or idx >= num_items:
            continue
        item_click_arr[idx] = (count_val - click_min) / click_range
        ts_val = created_at.get(raw_id, ts_min)
        item_created_arr[idx] = (ts_val - ts_min) / ts_range

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
    """
    print(">>> Building extended user features...")
    item_le = encoders['item_id']
    brand_le = encoders['brand_id']

    click_df = click_df.copy()
    click_df['item_idx'] = item_le.transform(click_df['click_article_id'])
    click_df = click_df.sort_values('click_timestamp')

    # --- Build item→brand mapping ---
    meta = meta_df.copy()
    meta['item_idx'] = item_le.transform(meta['parent_asin'])
    item_to_brand = dict(zip(meta['item_idx'], meta['brand']))

    def get_brand_idx(item_idx):
        brand = item_to_brand.get(item_idx, '__UNKNOWN__')
        return brand_le.transform([brand])[0] if brand in brand_le.classes_ else 0

    # --- Per-user aggregation ---
    user_data = []
    for uid, grp in tqdm(click_df.groupby('user_id'), desc="  building user features"):
        grp = grp.sort_values('click_timestamp')

        item_seq = grp['item_idx'].tolist()
        rating_seq = grp['rating'].tolist()
        ts_seq = grp['click_timestamp'].tolist()
        verified_seq = grp['verified_purchase'].tolist()
        helpful_seq = grp['helpful_vote'].tolist()

        # Truncate to last hist_len
        if len(item_seq) > hist_len:
            item_seq = item_seq[-hist_len:]
            rating_seq = rating_seq[-hist_len:]
            ts_seq = ts_seq[-hist_len:]
            verified_seq = verified_seq[-hist_len:]
            helpful_seq = helpful_seq[-hist_len:]

        # Time deltas
        time_deltas = [0]
        for i in range(1, len(ts_seq)):
            delta = (ts_seq[i] - ts_seq[i - 1]) / (1000 * 3600 * 24)  # days
            time_deltas.append(min(delta, 365))  # cap at 1 year
        # Normalize to [0, 1]
        max_delta = max(time_deltas) if time_deltas else 1
        time_deltas_norm = [d / (max_delta + 1e-8) for d in time_deltas]

        # Brand sequence
        brand_seq = [get_brand_idx(ii) for ii in item_seq]

        # Stats
        ratings_arr = np.array(rating_seq)
        verified_arr = np.array(verified_seq)
        helpful_arr = np.array(helpful_seq)

        user_data.append({
            'user_id': uid,
            'hist_items': item_seq,
            'hist_brands': brand_seq,
            'hist_ratings': rating_seq,
            'hist_time_deltas': time_deltas_norm,
            'hist_verified': verified_seq,
            'hist_len': len(item_seq),
            'click_count': len(grp),
            'time_span': ts_seq[-1] - ts_seq[0] if len(ts_seq) > 1 else 0,
            'user_avg_rating': float(ratings_arr.mean()),
            'user_std_rating': float(ratings_arr.std()) if len(ratings_arr) > 1 else 0.0,
            'user_verified_ratio': float(verified_arr.mean()),
            'user_avg_helpful': float(helpful_arr.mean()),
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
# Extended DIN Dataset (BPR Pairwise)
# ============================================================

class DINExtendedDataset(Dataset):
    """
    Extended DIN dataset with brand, verified_purchase, item quality signals.

    Each sample = (user features, positive item features, negative item features).
    User history includes: items, brands, ratings, time_deltas, verified.
    """
    def __init__(self, click_df, user_features, item_features, encoders,
                 hist_len=50, neg_ratio=4):
        self.user_le = encoders['user_id']
        self.item_le = encoders['item_id']
        self.brand_le = encoders['brand_id']
        self.raw_to_idx = encoders.get('raw_to_idx', None)

        click_df = click_df.copy()
        click_df['user_idx'] = self.user_le.transform(click_df['user_id'])
        click_df['item_idx'] = self.item_le.transform(click_df['click_article_id'])

        pos_df = click_df[click_df['click_label'] == 1]

        # --- Build lookup dicts ---
        # user_id → encoded user index
        uid_to_idx = {}
        for _, row in user_features.iterrows():
            raw_uid = row['user_id']
            if self.raw_to_idx is not None:
                uid_to_idx[raw_uid] = self.raw_to_idx.get(raw_uid)
            else:
                uid_to_idx[raw_uid] = self.user_le.transform([raw_uid])[0]

        # user history lookup
        self.uid_to_hist = {}
        for _, row in user_features.iterrows():
            raw_uid = row['user_id']
            uidx = uid_to_idx.get(raw_uid)
            if uidx is None:
                continue
            self.uid_to_hist[uidx] = {
                'items': row['hist_items'],
                'brands': row['hist_brands'],
                'ratings': row['hist_ratings'],
                'time_deltas': row['hist_time_deltas'],
                'verified': row['hist_verified'],
            }

        # User's positive items (for negative sampling exclusion)
        user_pos_items = pos_df.groupby('user_idx')['item_idx'].apply(set).to_dict()

        # --- Build BPR pairs ---
        self.pairs = []
        rng = np.random.default_rng(42)
        all_items = set(range(len(self.item_le.classes_)))

        for _, row in tqdm(pos_df.iterrows(), desc="  building BPR pairs", total=len(pos_df)):
            uid = row['user_idx']
            pos_iid = row['item_idx']
            pos_set = user_pos_items.get(uid, set())

            for _ in range(neg_ratio):
                neg_iid = rng.integers(1, len(self.item_le.classes_))
                while neg_iid in pos_set:
                    neg_iid = rng.integers(1, len(self.item_le.classes_))
                self.pairs.append((uid, pos_iid, int(neg_iid)))

        print(f">>> DINExtendedDataset: {len(self.pairs):,} BPR pairs")
        self.hist_len = hist_len

        # --- Pre-build user feature arrays ---
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

        for _, row in user_features.iterrows():
            raw_uid = row['user_id']
            uidx = uid_to_idx.get(raw_uid)
            if uidx is None:
                continue
            self.user_hist_arr[uidx] = row['hist_items']
            self.user_brand_hist_arr[uidx] = row['hist_brands']
            self.user_rating_hist_arr[uidx] = row['hist_ratings']
            self.user_delta_hist_arr[uidx] = row['hist_time_deltas']
            self.user_verified_hist_arr[uidx] = row['hist_verified']
            self.user_click_cnt_arr[uidx] = row['click_count_norm']
            self.user_time_span_arr[uidx] = row['time_span_norm']
            self.user_avg_rating_arr[uidx] = row['user_avg_rating_norm']
            self.user_std_rating_arr[uidx] = row['user_std_rating_norm']
            self.user_verified_ratio_arr[uidx] = row['user_verified_ratio']
            self.user_avg_helpful_arr[uidx] = row['user_avg_helpful_norm']

        # --- Pre-build item feature arrays ---
        num_all_items = len(self.item_le.classes_)
        self.item_cat_arr = np.zeros(num_all_items, dtype=np.int64)
        self.item_brand_arr = np.zeros(num_all_items, dtype=np.int64)
        self.item_click_arr = np.zeros(num_all_items, dtype=np.float32)
        self.item_created_arr = np.zeros(num_all_items, dtype=np.float32)
        self.item_avg_rating_arr = np.zeros(num_all_items, dtype=np.float32)
        self.item_rating_num_arr = np.zeros(num_all_items, dtype=np.float32)

        for idx in item_features.index:
            self.item_cat_arr[idx] = int(item_features.loc[idx].get('category_idx', 0))
            self.item_brand_arr[idx] = int(item_features.loc[idx].get('brand_idx', 0))
            self.item_click_arr[idx] = float(item_features.loc[idx].get('item_click_count_norm', 0))
            self.item_created_arr[idx] = float(item_features.loc[idx].get('created_at_ts_norm', 0))
            self.item_avg_rating_arr[idx] = float(item_features.loc[idx].get('item_avg_rating_norm', 0))
            self.item_rating_num_arr[idx] = float(item_features.loc[idx].get('item_rating_number_norm', 0))

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
