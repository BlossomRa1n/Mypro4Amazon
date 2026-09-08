"""
MIND 全库 HR@100 评估 — 对比 V2 基线 (8.66%) / ItemCF (10.62%) / 四路并集 (16.89%)。

采样 10 万 val 用户, 用 MIND 多兴趣网络 (K=3 兴趣胶囊) 做 top-100 多兴趣召回:
  每个兴趣向量独立 top-N 召回, 合并去重取 top-100 (覆盖「第二兴趣」)。
复用 train_mind.py 的 mind_evaluate 逻辑 + 加载训练后保存的 item_embeddings_mind.pkl。
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import pickle, time
import numpy as np
import torch
from tqdm import tqdm

import config
from data_loader import get_all_click_df, get_item_topk_click
from data_loader_ext import (load_raw_meta, build_extended_encoders,
                             build_extended_item_features,
                             build_extended_user_features, merge_jsonl_features)
from evaluate import split_train_val
from model import MINDModel

SAMPLE_USERS = 100000


def main():
    t0 = time.time()
    print(">>> Loading data...")
    click_df = get_all_click_df(config.DATA_PATH, offline=config.OFFLINE_MODE)
    raw_meta = load_raw_meta(config.DATA_PATH, config.AMAZON_CATEGORIES)
    click_df = merge_jsonl_features(click_df, config.DATA_PATH, config.AMAZON_CATEGORIES)
    train_click, val_click = split_train_val(click_df, config.EVAL_SPLIT_RATIO)

    print(">>> Building encoders & features...")
    encoders = build_extended_encoders(click_df, raw_meta, config.EXT_ENCODER_PKL)
    user_features = build_extended_user_features(
        train_click, raw_meta, encoders, hist_len=config.HIST_LEN)
    if 'hist_items_trunc' not in user_features.columns:
        user_features['hist_items_trunc'] = user_features['hist_items']
    item_features = build_extended_item_features(train_click, raw_meta, encoders)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Device: {device}")

    print(">>> Loading MIND model & item embeddings...")
    ckpt = torch.load(config.MIND_MODEL_FILE, map_location=device, weights_only=False)
    model_cfg = ckpt['config']
    model = MINDModel(**model_cfg).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f">>> MIND model: epoch {ckpt.get('epoch', '?')}, "
          f"best NDCG@20={ckpt.get('best_ndcg', 0):.4f}, "
          f"K={model_cfg['num_interests']}")

    with open(config.MIND_EMBED_PKL, 'rb') as f:
        all_item_vecs = pickle.load(f)
    print(f">>> MIND item embeddings: {all_item_vecs.shape}")

    num_items = model_cfg['num_items']
    hist_len = config.HIST_LEN
    K = model_cfg['num_interests']
    per_interest = 40   # 3 × 40 = 120 → 去重后 ~100

    # ground truth + 采样
    val_gt = val_click.groupby('user_id')['click_article_id'].apply(set).to_dict()
    all_val_users = list(val_gt.keys())
    rng = np.random.default_rng(42)
    if len(all_val_users) > SAMPLE_USERS:
        sample_users = rng.choice(all_val_users, SAMPLE_USERS, replace=False).tolist()
    else:
        sample_users = all_val_users
    print(f">>> Sampled {len(sample_users):,} val users")

    # 用户特征字典
    item_le = encoders['item_id']
    user_le = encoders['user_id']
    raw_to_idx = encoders.get('raw_to_idx', None)
    idx_to_raw_item = item_le.classes_.tolist()

    user_feat_dict = {}
    for _, row in user_features.iterrows():
        uid = row['user_id']
        user_feat_dict[uid] = {
            'hist': row.get('hist_items_trunc', row.get('hist_items', [])),
            'click_norm': row['click_count_norm'],
            'span_norm': row['time_span_norm'],
        }

    # 训练集已交互物品 (召回时过滤)
    raw_to_item_enc = {cls: i for i, cls in enumerate(item_le.classes_)}
    train_enc = train_click['click_article_id'].map(raw_to_item_enc).dropna().astype(int)
    train_pairs = train_click.loc[train_enc.index, ['user_id']].copy()
    train_pairs['item_enc'] = train_enc.values
    train_items = train_pairs.groupby('user_id')['item_enc'].apply(set).to_dict()

    all_item_vecs_t = torch.FloatTensor(all_item_vecs).to(device)

    print(">>> MIND multi-interest recall (top-100)...")
    hit = 0
    batch_size = 256
    n_users = len(sample_users)

    for start in tqdm(range(0, n_users, batch_size), desc="MIND Recall"):
        chunk = sample_users[start:start + batch_size]
        histories, hist_lens, click_counts, time_spans = [], [], [], []
        valid = []
        for raw_uid in chunk:
            feat = user_feat_dict.get(raw_uid)
            if feat is None:
                continue
            if raw_to_idx is not None:
                uidx = raw_to_idx.get(raw_uid, 0)
            else:
                uidx = user_le.transform([raw_uid])[0]
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
                user_batch['time_span'], target_item_vec=None
            )  # (B, K, D)

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
            merged = merged[:config.RECALL_NUM]
            gt = val_gt.get(raw_uid, set())
            if gt & set(idx_to_raw_item[j] for j in merged):
                hit += 1

    hr100 = hit / max(n_users, 1)
    print("\n" + "=" * 60)
    print(f">>> MIND 全库 HR@100 = {hr100*100:.2f}% ({hit:,}/{n_users:,})")
    print(f">>> 对比基线: V2=8.66% | ItemCF=10.62% | 四路并集=16.89%")
    print(f">>> Total time: {time.time()-t0:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
