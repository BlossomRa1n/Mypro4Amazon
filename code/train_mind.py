"""
MIND 多兴趣网络训练脚本 (第 6 路召回) — 动态路由聚类 K 个兴趣胶囊 + InfoNCE + SVD warm-start

与 train_v2.py 的区别:
  1. 用户塔 = MINDUserTower (K=3 兴趣胶囊, 动态路由), 而非 SASRec 单向量
  2. 训练用 Label-Aware Attention 选最相关兴趣算 loss
  3. 评估用「多兴趣召回」: 每个兴趣向量独立 top-N 召回, 合并 (召回「第二兴趣」)
已修复: 移除 user_embedding (背答案捷径); ItemTower 升级为扩展版 (brand/quality);
        forward/get_item_embedding 位置参数错位。
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import time, pickle, json
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from data_loader import TwoTowerV2Dataset, get_all_click_df
from data_loader_ext import (
    load_raw_meta, build_extended_encoders, build_extended_item_features,
    build_extended_user_features, merge_jsonl_features,
)
from model import MINDModel
from evaluate import split_train_val, compute_item_embeddings
from utils import set_seed

set_seed(42)


def collate_fn(batch):
    keys = batch[0].keys()
    result = {}
    for key in keys:
        result[key] = torch.stack([item[key] for item in batch])
    return result


def _save_checkpoint(model, optimizer, scheduler, scaler, model_cfg, filepath, epoch, metrics, best_ndcg, patience):
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model_cfg,
        'epoch': epoch,
        'metrics': metrics,
        'best_ndcg': best_ndcg,
        'patience_counter': patience,
    }, filepath)


def _find_latest_checkpoint(prefix):
    import glob
    pattern = os.path.join(config.CHECKPOINT_DIR, f'{prefix}_epoch*.pth')
    files = glob.glob(pattern)
    if not files:
        return None
    def extract_epoch(p):
        import re
        m = re.search(r'epoch(\d+)', os.path.basename(p))
        return int(m.group(1)) if m else 0
    files.sort(key=extract_epoch)
    return files[-1]


def mind_evaluate(model, val_click, train_click, user_features, item_features,
                  encoders, device, k=20, max_users=3000, per_interest=20):
    """
    多兴趣召回评估: 每个用户 K 个兴趣向量, 每个兴趣独立 top-N 召回, 合并 (去重 + 过滤已交互)。
    这是 MIND 区别于单向量模型的评估口径 — 覆盖「第二兴趣」。
    """
    model.eval()
    item_le = encoders['item_id']
    user_le = encoders['user_id']
    raw_to_idx = encoders.get('raw_to_idx', None)
    num_items = len(item_le.classes_)

    all_item_vecs = compute_item_embeddings(model, num_items, item_features, device)

    raw_to_item_enc = {cls: i for i, cls in enumerate(item_le.classes_)}
    train_enc = train_click['click_article_id'].map(raw_to_item_enc).dropna().astype(int)
    train_pairs = pd.DataFrame({
        'user_id': train_click.loc[train_enc.index, 'user_id'],
        'item_enc': train_enc.values
    })
    train_items = train_pairs.groupby('user_id')['item_enc'].apply(set).to_dict()

    val_users_set = set(user_le.classes_)
    val_enc = val_click['click_article_id'].map(raw_to_item_enc).dropna().astype(int)
    val_pairs = pd.DataFrame({
        'user_id': val_click.loc[val_enc.index, 'user_id'],
        'item_enc': val_enc.values
    })
    val_pairs = val_pairs[val_pairs['user_id'].isin(val_users_set)]
    val_items = val_pairs.groupby('user_id')['item_enc'].apply(set).to_dict()

    val_users = list(val_items.keys())
    if max_users and len(val_users) > max_users:
        val_users = list(np.random.choice(val_users, max_users, replace=False))

    user_feat_dict = {}
    for _, row in user_features.iterrows():
        uid = row['user_id']
        user_feat_dict[uid] = {
            'hist': row.get('hist_items_trunc', row.get('hist_items', [])),
            'click_norm': row['click_count_norm'],
            'span_norm': row['time_span_norm'],
        }

    num_interests = config.MIND_NUM_INTERESTS
    hist_len = config.HIST_LEN
    hr_total, ndcgs = 0, []
    batch_size = 256

    for start in range(0, len(val_users), batch_size):
        chunk = val_users[start:start + batch_size]
        histories, hist_lens, click_counts, time_spans = [], [], [], []
        valid = []
        for raw_uid in chunk:
            feat = user_feat_dict.get(raw_uid)
            if feat is None:
                continue
            uidx = raw_to_idx.get(raw_uid, 0) if raw_to_idx is not None else user_le.transform([raw_uid])[0]
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

        # 预构建训练集已交互物品的 mask 索引 (向量化, 一次复用)
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
                user_batch['user_id'], user_batch['hist_items'], user_batch['hist_len'],
                user_batch['click_count'], user_batch['time_span'], target_item_vec=None
            )  # (B, K, D)

            # 逐兴趣打分 (避免 (B,K,N) 大矩阵), 每个兴趣独立 top-N
            per_interest_topk = []
            for kk in range(num_interests):
                ivec = interest_vectors[:, kk, :]  # (B, D)
                s = torch.matmul(ivec, all_item_vecs.t())  # (B, N)
                if mask_rows:
                    s[rows, cols] = -1e9
                per_interest_topk.append(torch.topk(s, per_interest, dim=1).indices)

        for i, (raw_uid, _) in enumerate(valid):
            merged, seen = [], set()
            for kk in range(num_interests):
                for idx in per_interest_topk[kk][i].tolist():
                    if idx not in seen:
                        seen.add(idx)
                        merged.append(idx)
            merged = merged[:k]
            gt_set = val_items.get(raw_uid, set())
            if gt_set.intersection(merged):
                hr_total += 1
            for pos, idx in enumerate(merged):
                if idx in gt_set:
                    ndcgs.append(1.0 / np.log2(pos + 2))

    hr = hr_total / len(val_users) if val_users else 0
    ndcg = sum(ndcgs) / len(val_users) if val_users else 0
    return {'hr': hr, 'ndcg': ndcg, 'n_users': len(val_users), 'method': 'mind_multi_interest'}


def train():
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Using device: {device}")

    print("Step 1: Loading data...")
    click_df = get_all_click_df(config.DATA_PATH, offline=config.OFFLINE_MODE)
    raw_meta = load_raw_meta(config.DATA_PATH, config.AMAZON_CATEGORIES)
    click_df = merge_jsonl_features(click_df, config.DATA_PATH, config.AMAZON_CATEGORIES)

    print("\n--- Temporal Train/Val Split ---")
    train_click, val_click = split_train_val(click_df, config.EVAL_SPLIT_RATIO)

    print("\nStep 2: Building extended encoders & features...")
    encoders = build_extended_encoders(click_df, raw_meta, config.EXT_ENCODER_PKL)
    num_users = len(encoders['user_id'].classes_)
    num_items = len(encoders['item_id'].classes_)
    num_categories = len(encoders['category_id'].classes_)
    num_brands = len(encoders['brand_id'].classes_)
    print(f">>> num_users={num_users}, num_items={num_items}, "
          f"num_categories={num_categories}, num_brands={num_brands}")

    item_features = build_extended_item_features(train_click, raw_meta, encoders)
    user_features = build_extended_user_features(
        train_click, raw_meta, encoders, hist_len=config.HIST_LEN
    )
    if 'hist_items_trunc' not in user_features.columns:
        user_features['hist_items_trunc'] = user_features['hist_items']

    print("\nStep 3: Building training dataset (MIND)...")
    dataset = TwoTowerV2Dataset(
        train_click, user_features, item_features, encoders,
        num_items=num_items, hist_len=config.HIST_LEN,
        hard_neg_index=None, num_hard_negatives=0,
        num_brands=num_brands
    )
    dataloader = DataLoader(
        dataset, batch_size=config.MIND_BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, collate_fn=collate_fn, persistent_workers=True
    )

    print("\nStep 4: Initializing MINDModel (multi-interest)...")
    model_cfg = {
        'num_users': num_users,
        'num_items': num_items,
        'num_categories': num_categories,
        'num_brands': num_brands,
        'brand_embed_dim': getattr(config, 'V2_BRAND_EMBED_DIM', 64),
        'embed_dim': config.EMBED_DIM,
        'hidden_dims': config.HIDDEN_DIMS,
        'hist_len': config.HIST_LEN,
        'temperature': config.MIND_TEMPERATURE,
        'num_interests': config.MIND_NUM_INTERESTS,
        'num_routing_iterations': config.MIND_ROUTING_ITERS,
        'dropout': config.SASREC_DROPOUT,
    }
    model = MINDModel(**model_cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f">>> Total parameters: {total_params:,}")

    if getattr(config, 'USE_ALS_INIT', False):
        from als_init import init_model_with_als
        pos_click = click_df[click_df['click_label'] == 1]
        model, als_ok = init_model_with_als(
            model, pos_click, encoders['user_id'], encoders['item_id'],
            fix_embeddings=getattr(config, 'ALS_FIX_EMBEDDINGS', False)
        )
        if als_ok:
            print(">>> SVD pre-training loaded — MIND starts from collaborative-filtering quality")
        else:
            print(">>> SVD unavailable, falling back to random init")

    optimizer = optim.AdamW(model.parameters(), lr=config.MIND_LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.MIND_NUM_EPOCHS, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler() if config.USE_AMP else None

    os.makedirs(config.MODEL_PATH, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    start_epoch = 0
    best_ndcg = 0.0
    patience_counter = 0
    history = []
    resume_path = os.path.join(config.MODEL_PATH, 'mind_latest.pth')
    if not os.path.exists(resume_path):
        resume_path = _find_latest_checkpoint('mind')
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model_cfg = ckpt['config']
        model = MINDModel(**model_cfg).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        best_ndcg = ckpt.get('best_ndcg', 0.0)
        patience_counter = ckpt.get('patience_counter', 0)
        print(f">>> Resumed from epoch {start_epoch}, best NDCG@{config.EVAL_K}={best_ndcg:.4f}")
        history_path = os.path.join(config.MODEL_PATH, 'mind_history.json')
        if os.path.exists(history_path):
            with open(history_path) as f:
                history = json.load(f)
            history = history[:start_epoch]

    print(f"\nStep 5: Training from epoch {start_epoch+1}...")
    best_epoch = start_epoch if start_epoch > 0 else -1

    for epoch in range(start_epoch, config.MIND_NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.MIND_NUM_EPOCHS}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=config.USE_AMP):
                interest_vectors, pos_item_vec = model(batch)
                loss = model.compute_infonce_loss(interest_vectors, pos_item_vec)

            if torch.isnan(loss) or torch.isinf(loss):
                if scaler is not None:
                    scaler.update()
                continue

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        train_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1}/{config.MIND_NUM_EPOCHS} | "
              f"Train Loss={train_loss:.4f} | LR={scheduler.get_last_lr()[0]:.6f}")

        if config.SKIP_EVAL:
            metrics = {'hr': 0.0, 'ndcg': 0.0, 'n_users': 0}
        else:
            print("  Running validation (multi-interest recall)...")
            metrics = mind_evaluate(
                model, val_click, train_click, user_features, item_features,
                encoders, device, k=config.EVAL_K, max_users=config.EVAL_MAX_USERS,
                per_interest=config.EVAL_K
            )

        hr = metrics['hr']
        ndcg = metrics['ndcg']
        print(f"  → Val HR@{config.EVAL_K}={hr:.4f} ({hr*100:.1f}%) | "
              f"NDCG@{config.EVAL_K}={ndcg:.4f} | Users={metrics['n_users']}")

        history.append({'epoch': epoch + 1, 'train_loss': train_loss,
                        'val_hr': hr, 'val_ndcg': ndcg})

        epoch_path = os.path.join(config.CHECKPOINT_DIR, f'mind_epoch{epoch+1:02d}.pth')
        _save_checkpoint(model, optimizer, scheduler, scaler, model_cfg,
                        epoch_path, epoch + 1, metrics, best_ndcg, patience_counter)
        if epoch >= 1:
            prev_path = os.path.join(config.CHECKPOINT_DIR, f'mind_epoch{epoch:02d}.pth')
            if os.path.exists(prev_path):
                os.remove(prev_path)

        latest_path = os.path.join(config.MODEL_PATH, 'mind_latest.pth')
        _save_checkpoint(model, optimizer, scheduler, scaler, model_cfg,
                        latest_path, epoch + 1, metrics, best_ndcg, patience_counter)

        is_better = config.SKIP_EVAL or ndcg > best_ndcg
        if is_better:
            best_ndcg = ndcg
            best_epoch = epoch + 1
            patience_counter = 0
            best_path = config.MIND_MODEL_FILE
            _save_checkpoint(model, optimizer, scheduler, scaler, model_cfg,
                            best_path, epoch + 1, metrics, best_ndcg, patience_counter)
            print(f"  ★ New best! NDCG@{config.EVAL_K}={ndcg:.4f}")
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epochs "
                  f"(best NDCG@{config.EVAL_K}={best_ndcg:.4f} at epoch {best_epoch})")

        if patience_counter >= config.EARLY_STOP_PATIENCE:
            print(f"\n>>> Early stop triggered after {epoch+1} epochs")
            break

    history_path = os.path.join(config.MODEL_PATH, 'mind_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"MIND training complete!")
    print(f"  Best epoch: {best_epoch}, Best NDCG@{config.EVAL_K}={best_ndcg:.4f}")
    print(f"  User tower: MIND (K={config.MIND_NUM_INTERESTS} interests, "
          f"routing_iters={config.MIND_ROUTING_ITERS})")
    print(f"{'='*60}")

    results_path = os.path.join(config.RESULT_PATH, 'mind_final_results.json')
    os.makedirs(config.RESULT_PATH, exist_ok=True)
    final_results = {
        'model': 'MINDModel',
        'data': config.AMAZON_CATEGORIES,
        'best_epoch': best_epoch,
        'best_ndcg': best_ndcg,
        'hyperparams': {
            'num_interests': config.MIND_NUM_INTERESTS,
            'routing_iters': config.MIND_ROUTING_ITERS,
            'temperature': config.MIND_TEMPERATURE,
            'lr': config.MIND_LEARNING_RATE,
            'batch_size': config.MIND_BATCH_SIZE,
            'num_epochs': config.MIND_NUM_EPOCHS,
        },
        'history': history,
    }
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f">>> Final results saved to {results_path}")

    best_path = config.MIND_MODEL_FILE
    if not os.path.exists(best_path):
        print("[WARN] No best model saved, using last checkpoint.")
        best_path = os.path.join(config.CHECKPOINT_DIR, f'mind_epoch{len(history):02d}.pth')

    print("\nStep 6: Loading best model for item embeddings...")
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model = MINDModel(**ckpt['config']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])

    print("  Pre-computing item embeddings...")
    all_item_vecs = compute_item_embeddings(model, num_items, item_features, device, batch_size=2048)
    all_item_vecs = all_item_vecs.cpu().numpy()
    with open(config.MIND_EMBED_PKL, 'wb') as f:
        pickle.dump(all_item_vecs, f)
    print(f">>> Item embeddings saved to {config.MIND_EMBED_PKL}, shape={all_item_vecs.shape}")

    print(f"\nAll done! Total time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    train()
