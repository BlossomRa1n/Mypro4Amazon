"""Full pipeline smoke test (V2 SASRec → Extended DIN)."""
import sys, os, time

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

import config

# Override config for local offline smoke test
config.OFFLINE_MODE = True
config.SKIP_EVAL = True
config.AMAZON_CATEGORIES = ['Video_Games']
config.EXT_CATEGORIES = ['Video_Games']

# Auto-detect local data path (config defaults to server /root/autodl-tmp)
import os as _os
if not _os.path.exists(config.DATA_PATH):
    config.DATA_PATH = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'amazon_reviews')
    print(f'[Auto-detect] DATA_PATH → {config.DATA_PATH}')


def step(label, fn):
    print(); print('=' * 60)
    print(f'STEP: {label}')
    print('=' * 60)
    t0 = time.time()
    result = fn()
    print(f'  Time: {time.time()-t0:.1f}s')
    return result


def main():
    t_total = time.time()

    # ============================================================
    # Step 1: Load benchmark CSV + raw meta
    # ============================================================
    def load_data():
        from data_loader import get_all_click_df, load_articles
        from data_loader_ext import load_raw_meta

        click_df = get_all_click_df(config.DATA_PATH, offline=True)
        articles_df = load_articles(config.DATA_PATH)
        raw_meta = load_raw_meta(config.DATA_PATH, config.EXT_CATEGORIES)
        print(f'  CSV: {len(click_df):,} rows | '
              f'Articles: {len(articles_df):,} | '
              f'Meta: {len(raw_meta):,} (brands={raw_meta.brand.nunique():,})')
        return click_df, articles_df, raw_meta

    click_df, articles_df, raw_meta = step('Load benchmark CSV + raw meta', load_data)

    # ============================================================
    # Step 2: Build encoders + features (ext)
    # ============================================================
    def build_features():
        from evaluate import split_train_val
        from data_loader import build_encoders, build_enhanced_user_features
        from data_loader_ext import build_extended_encoders, build_extended_item_features

        train_click, val_click = split_train_val(click_df, 0.8)

        # V1/V2 encoders (from benchmark CSV)
        if os.path.exists(config.ENCODER_PKL):
            os.remove(config.ENCODER_PKL)
        encoders = build_encoders(click_df, articles_df, config.ENCODER_PKL)

        # Extended encoders (with brand from raw meta)
        if os.path.exists(config.EXT_ENCODER_PKL):
            os.remove(config.EXT_ENCODER_PKL)
        ext_encoders = build_extended_encoders(click_df, raw_meta, config.EXT_ENCODER_PKL)

        nu = len(encoders['user_id'].classes_)
        ni = len(encoders['item_id'].classes_)
        nc = len(ext_encoders['category_id'].classes_)
        nb = len(ext_encoders['brand_id'].classes_)
        print(f'  Users={nu:,} Items={ni:,} Categories={nc} Brands={nb}')

        # Features
        item_features = build_extended_item_features(train_click, raw_meta, ext_encoders)
        user_features_v2 = build_enhanced_user_features(
            train_click[train_click['click_label'] == 1], articles_df, encoders, hist_len=config.HIST_LEN
        )
        # Merge user rating stats into V2
        pos_click = train_click[train_click['click_label'] == 1]
        stats = pos_click.groupby('user_id')['rating'].agg(['mean', 'std']).fillna(0)
        stats.columns = ['user_avg_rating', 'user_std_rating']
        stats = stats.reset_index()
        stats['user_avg_rating_norm'] = stats['user_avg_rating'] / 5.0
        stats['user_std_rating_norm'] = (stats['user_std_rating'] / 2.0).clip(0, 1)
        user_features_v2 = user_features_v2.merge(
            stats[['user_id', 'user_avg_rating_norm', 'user_std_rating_norm']],
            on='user_id', how='left'
        ).fillna(0)

        print(f'  Item features: {item_features.shape}')
        print(f'  V2 User features: {user_features_v2.shape}')
        return train_click, val_click, encoders, ext_encoders, item_features, \
               user_features_v2, nu, ni, nc, nb

    (train_click, val_click, encoders, ext_encoders, item_features,
     user_features_v2, nu, ni, nc, nb) = step(
        'Build encoders + features', build_features)

    # ============================================================
    # Step 3: V2 TwoTowerV2Model — 1 batch fwd/back
    # ============================================================
    def test_v2():
        from data_loader import TwoTowerV2Dataset
        from model import TwoTowerV2Model
        from torch.utils.data import DataLoader

        ds = TwoTowerV2Dataset(train_click, user_features_v2, item_features, ext_encoders,
                               num_items=ni, hist_len=config.HIST_LEN, num_brands=nb)
        dl = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=step.collate)
        batch = next(iter(dl))

        model = TwoTowerV2Model(nu, ni, nc, num_brands=nb, embed_dim=64,
                                hidden_dims=[128, 64], hist_len=config.HIST_LEN,
                                brand_embed_dim=32)
        params = sum(p.numel() for p in model.parameters())
        user_vec, pos_vec = model(batch)
        loss = model.compute_infonce_loss(user_vec, pos_vec)
        loss.backward()
        print(f'  Params={params:,} | Loss={loss.item():.4f}  ✓')

    step('V2 TwoTowerV2Model fwd+back', test_v2)

    # ============================================================
    # Step 4: Extended DIN — 1 batch fwd/back
    # ============================================================
    def test_ext_din():
        from data_loader_ext import build_extended_user_features, DINExtendedDataset
        from model_ext import DINExtendedModel
        from torch.utils.data import DataLoader

        user_feat = build_extended_user_features(train_click, raw_meta, ext_encoders, hist_len=50)

        ds = DINExtendedDataset(train_click, user_feat, item_features, ext_encoders,
                                hist_len=50, neg_ratio=2)
        sample = ds[0]

        model = DINExtendedModel(nu, ni, nb, nc, embed_dim=64, brand_embed_dim=32,
                                 hidden_dims=[128, 64], hist_len=50, dropout=0.1)
        params = sum(p.numel() for p in model.parameters())

        # Single-sample forward
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
        loss.backward()
        print(f'  Params={params:,} | Pos={pos_score.item():.3f} | '
              f'Neg={neg_score.item():.3f} | Loss={loss.item():.4f}  ✓')

    step('Extended DIN fwd+back', test_ext_din)

    # ============================================================
    print(); print('=' * 60)
    print(f'ALL 4 STEPS PASSED  (total: {time.time()-t_total:.1f}s)')
    print('=' * 60)


def _collate(batch):
    keys = batch[0].keys()
    result = {}
    for key in keys:
        values = [item[key] for item in batch]
        if values and isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values)
        else:
            result[key] = torch.tensor(values)
    return result


step.collate = _collate

if __name__ == '__main__':
    main()