"""
Test script: Extended DIN + ALS pre-training + rating_only CSV + raw meta
Verified model architecture against old DIN (which reached AUC=0.66 with BPR).
"""

import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from data_loader import get_all_click_df, load_articles
from data_loader_ext import (
    load_raw_meta, prepare_click_df,
    build_extended_encoders, build_extended_item_features,
    build_extended_user_features, DINExtendedDataset,
)
from model_ext import DINExtendedModel
from evaluate import split_train_val
from train_din_ext import evaluate_ext, collate_fn
from als_init import init_model_with_als


def load_rating_only_as_raw(data_path, categories):
    """
    Load rating_only CSV and convert to the format expected by
    prepare_click_df (same output shape as load_raw_reviews).
    Adds dummy columns: helpful_vote=0, verified_purchase=0, asin=parent_asin.
    """
    import pandas as pd

    dfs = []
    for cat in categories:
        path = os.path.join(data_path, f'{cat}.csv')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Rating-only CSV not found: {path}")
        df = pd.read_csv(path)
        df['helpful_vote'] = 0
        df['verified_purchase'] = 0
        df['asin'] = df['parent_asin']
        dfs.append(df)
        print(f"  {cat}: {len(df)} rows")

    return pd.concat(dfs, ignore_index=True)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Device: {device}")

    # ============================================================
    # Step 1: Load data
    # ============================================================
    print("\n=== Step 1: Load data ===")
    raw_reviews = load_rating_only_as_raw(config.DATA_PATH, config.EXT_CATEGORIES)
    raw_meta = load_raw_meta(config.DATA_PATH, config.EXT_CATEGORIES)
    click_df = prepare_click_df(raw_reviews,
                                 min_user_inter=config.EXT_MIN_USER_INTER,
                                 min_item_inter=config.EXT_MIN_ITEM_INTER)
    train_click, val_click = split_train_val(click_df, config.EVAL_SPLIT_RATIO)

    # ============================================================
    # Step 2: Encoders & features
    # ============================================================
    print("\n=== Step 2: Encoders & features ===")
    # Force rebuild
    if os.path.exists(config.EXT_ENCODER_PKL):
        os.remove(config.EXT_ENCODER_PKL)

    encoders = build_extended_encoders(click_df, raw_meta, config.EXT_ENCODER_PKL)
    nu = len(encoders['user_id'].classes_)
    ni = len(encoders['item_id'].classes_)
    nb = len(encoders['brand_id'].classes_)
    nc = len(encoders['category_id'].classes_)
    print(f"Users={nu}, Items={ni}, Brands={nb}, Categories={nc}")

    item_features = build_extended_item_features(click_df, raw_meta, encoders)
    user_features = build_extended_user_features(click_df, raw_meta, encoders,
                                                  hist_len=config.HIST_LEN)
    print(f"Avg hist_len: {user_features['hist_len'].mean():.1f}")

    # ============================================================
    # Step 3: Dataset
    # ============================================================
    print("\n=== Step 3: Dataset ===")
    ds = DINExtendedDataset(train_click, user_features, item_features, encoders,
                             hist_len=config.HIST_LEN, neg_ratio=config.DIN_NEG_RATIO)
    print(f"BPR pairs: {len(ds):,}")
    dl = DataLoader(ds, batch_size=config.DIN_BATCH_SIZE, shuffle=True,
                     num_workers=0, collate_fn=collate_fn,
                     persistent_workers=False)

    # ============================================================
    # Step 4: Model + ALS
    # ============================================================
    print("\n=== Step 4: Model + ALS ===")
    model_cfg = {
        'num_users': nu,
        'num_items': ni,
        'num_brands': nb,
        'num_categories': nc,
        'embed_dim': config.DIN_EMBED_DIM,
        'brand_embed_dim': config.DIN_BRAND_EMBED_DIM,
        'hidden_dims': config.DIN_HIDDEN_DIMS,
        'hist_len': config.HIST_LEN,
        'dropout': config.DIN_DROPOUT,
    }
    model = DINExtendedModel(**model_cfg).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    # ALS pre-training
    pos_click = click_df[click_df['click_label'] == 1]
    model, als_ok = init_model_with_als(
        model, pos_click, encoders['user_id'], encoders['item_id'],
        fix_embeddings=False,
    )
    print(f"ALS: {'OK' if als_ok else 'SKIPPED (no ALS cache)'}")

    optimizer = optim.Adam(model.parameters(), lr=config.DIN_LEARNING_RATE,
                            weight_decay=config.DIN_WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    # ============================================================
    # Step 5: Train
    # ============================================================
    print("\n=== Step 5: Train ===")
    user_keys = ['user_id', 'hist_items', 'hist_brands', 'hist_ratings',
                 'hist_time_deltas', 'hist_verified', 'hist_len',
                 'click_count', 'time_span', 'user_avg_rating',
                 'user_std_rating', 'user_verified_ratio', 'user_avg_helpful']
    pos_keys = ['item_id', 'category_id', 'brand_id', 'item_click_count',
                 'created_at_ts', 'item_avg_rating', 'item_rating_number']

    best_auc = 0.5
    history = []

    for epoch in range(min(config.DIN_NUM_EPOCHS, 5)):
        model.train()
        total_loss, n_batches = 0.0, 0
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{config.DIN_NUM_EPOCHS}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            ub = {k: batch[k] for k in user_keys if k in batch}
            pb = {k.replace('pos_', ''): batch[f'pos_{k}'] for k in pos_keys if f'pos_{k}' in batch}
            nb = {k.replace('neg_', ''): batch[f'neg_{k}'] for k in pos_keys if f'neg_{k}' in batch}

            pos_score, neg_score = model.forward_pairwise(ub, pb, nb)
            loss = model.compute_bpr_loss(pos_score, neg_score)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            acc = (pos_score > neg_score).float().mean().item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.4f}")

        scheduler.step()
        train_loss = total_loss / max(n_batches, 1)

        # Eval
        metrics = evaluate_ext(model, val_click, user_features, item_features,
                                encoders, device, max_users=config.EVAL_MAX_USERS)
        auc, pm = metrics['auc'], metrics['pos_mean']
        print(f"Epoch {epoch+1}: Loss={train_loss:.4f} AUC={auc:.4f} PosMean={pm:.4f}")

        history.append({'epoch': epoch + 1, 'train_loss': train_loss,
                         'val_auc': auc, 'val_pos_mean': pm})

        if auc > best_auc:
            best_auc = auc
            print(f"  * Best!")

    print(f"\n{'='*60}")
    print(f"Done! Best AUC={best_auc:.4f}")
    print(f"History: {json.dumps(history, indent=2)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    train()
