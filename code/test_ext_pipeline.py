"""End-to-end smoke test for Extended DIN pipeline (All_Beauty raw data)."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code'))

import numpy as np
import torch

import config
from data_loader_ext import (
    load_raw_reviews, load_raw_meta, prepare_click_df,
    build_extended_encoders, build_extended_item_features,
    build_extended_user_features, DINExtendedDataset
)
from model_ext import DINExtendedModel
from evaluate import split_train_val

def main():
    t_total = time.time()

    # ============================================================
    # Step 1: Load raw data
    # ============================================================
    print('=' * 60)
    print('STEP 1: Load raw data (offline mode)')
    print('=' * 60)
    t0 = time.time()
    reviews = load_raw_reviews(config.DATA_PATH, config.EXT_CATEGORIES, offline=True)
    meta = load_raw_meta(config.DATA_PATH, config.EXT_CATEGORIES)
    print(f'  Reviews: {len(reviews):,} rows, {reviews.user_id.nunique():,} users, '
          f'{reviews.parent_asin.nunique():,} items')
    print(f'  Meta: {len(meta):,} items, {meta.brand.nunique():,} brands')
    print(f'  Time: {time.time()-t0:.1f}s')

    # Step 1b: Prepare click df
    t0 = time.time()
    click = prepare_click_df(reviews, min_user_inter=2, min_item_inter=2)
    train_click, val_click = split_train_val(click, 0.8)
    print(f'  Click: {len(click):,} rows → train={len(train_click):,}, val={len(val_click):,}')
    print(f'  Time: {time.time()-t0:.1f}s')

    # ============================================================
    # Step 2: Build encoders & features
    # ============================================================
    print()
    print('=' * 60)
    print('STEP 2: Build encoders & features')
    print('=' * 60)
    t0 = time.time()

    # Clear cache for fresh test
    if os.path.exists(config.EXT_ENCODER_PKL):
        os.remove(config.EXT_ENCODER_PKL)

    encoders = build_extended_encoders(click, meta, config.EXT_ENCODER_PKL)
    nu = len(encoders['user_id'].classes_)
    ni = len(encoders['item_id'].classes_)
    nb = len(encoders['brand_id'].classes_)
    nc = len(encoders['category_id'].classes_)
    print(f'  Users={nu:,}, Items={ni:,}, Brands={nb:,}, Categories={nc}')

    item_feat = build_extended_item_features(click, meta, encoders)
    print(f'  Item features: {item_feat.shape}')
    user_feat = build_extended_user_features(click, meta, encoders, hist_len=50)
    print(f'  User features: {user_feat.shape}')
    print(f'  Time: {time.time()-t0:.1f}s')

    # ============================================================
    # Step 3: Build dataset
    # ============================================================
    print()
    print('=' * 60)
    print('STEP 3: Build dataset')
    print('=' * 60)
    t0 = time.time()
    ds = DINExtendedDataset(train_click, user_feat, item_feat, encoders,
                             hist_len=50, neg_ratio=2)
    print(f'  Dataset: {len(ds):,} BPR pairs')
    sample = ds[0]
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f'    {k:30s} shape={list(v.shape)}')
    print(f'  Time: {time.time()-t0:.1f}s')

    # ============================================================
    # Step 4: Model forward + backward
    # ============================================================
    print()
    print('=' * 60)
    print('STEP 4: Model forward + backward')
    print('=' * 60)
    t0 = time.time()

    model = DINExtendedModel(nu, ni, nb, nc, embed_dim=64, brand_embed_dim=32,
                             hidden_dims=[128, 64], hist_len=50, dropout=0.1)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'  Params: {total_params:,}')

    # Build batch from sample
    batch = {k: v.unsqueeze(0) for k, v in sample.items() if isinstance(v, torch.Tensor)}

    user_keys = ['user_id', 'hist_items', 'hist_brands', 'hist_ratings',
                 'hist_time_deltas', 'hist_verified', 'hist_len',
                 'click_count', 'time_span', 'user_avg_rating',
                 'user_std_rating', 'user_verified_ratio', 'user_avg_helpful']
    pos_keys = ['item_id', 'category_id', 'brand_id', 'item_click_count',
                'created_at_ts', 'item_avg_rating', 'item_rating_number']

    user_batch = {k: batch[k] for k in user_keys if k in batch}
    pos_batch = {k: batch[f'pos_{k}'] for k in pos_keys if f'pos_{k}' in batch}
    neg_batch = {k: batch[f'neg_{k}'] for k in pos_keys if f'neg_{k}' in batch}

    pos_score, neg_score = model.forward_pairwise(user_batch, pos_batch, neg_batch)
    loss = model.compute_bpr_loss(pos_score, neg_score)
    print(f'  Pos score: {pos_score.item():.4f}')
    print(f'  Neg score: {neg_score.item():.4f}')
    print(f'  BPR loss:  {loss.item():.4f}')

    loss.backward()
    print(f'  Backward: OK')
    print(f'  Time: {time.time()-t0:.1f}s')

    # ============================================================
    # Step 5: Evaluation
    # ============================================================
    print()
    print('=' * 60)
    print('STEP 5: Evaluation')
    print('=' * 60)
    t0 = time.time()

    from train_din_ext import evaluate_ext
    device = torch.device('cpu')
    metrics = evaluate_ext(model, val_click, user_feat, item_feat,
                            encoders, device, max_users=50)
    print(f'  AUC:     {metrics["auc"]:.4f}')
    print(f'  PosMean: {metrics["pos_mean"]:.4f}')
    print(f'  Users:   {metrics["n_users"]}')
    print(f'  Time: {time.time()-t0:.1f}s')

    # ============================================================
    # Done
    # ============================================================
    print()
    print('=' * 60)
    print(f'ALL CHECKS PASSED  (total: {time.time()-t_total:.1f}s)')
    print('=' * 60)


if __name__ == '__main__':
    main()
