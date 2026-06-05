"""
Extended DIN training script for Amazon Reviews 2023 Raw (All_Beauty).

Uses raw review JSONL (10 columns) + raw meta JSONL (14 columns) to build
rich features: brand/verified/helpful/price/item_quality signals.

Key differences from train_din.py:
  - Loads raw JSONL data instead of rating-only CSV
  - Extended features: brand, verified_purchase, item_avg_rating, etc.
  - Extended user history: brand_seq, rating_seq, time_delta_seq, verified_seq
  - New DINExtendedModel with brand_embedding and multi-signal attention
"""
import os, time, json
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from data_loader_ext import (
    load_raw_reviews, load_raw_meta, prepare_click_df,
    build_extended_encoders, build_extended_item_features,
    build_extended_user_features, DINExtendedDataset
)
from model_ext import DINExtendedModel
from evaluate import split_train_val


def collate_fn(batch):
    """Stack batch dicts into tensors."""
    keys = batch[0].keys()
    result = {}
    for key in keys:
        values = [item[key] for item in batch]
        if values and isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values)
        else:
            result[key] = torch.tensor(values)
    return result


def _save_checkpoint(model, optimizer, scheduler, model_cfg, filepath,
                      epoch, metrics, best_auc, patience, save_optimizer=True):
    data = {
        'model_state_dict': model.state_dict(),
        'config': model_cfg,
        'epoch': epoch,
        'metrics': metrics,
        'best_auc': best_auc,
        'patience_counter': patience,
    }
    if save_optimizer:
        data['optimizer_state_dict'] = optimizer.state_dict()
        data['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(data, filepath)


def _find_latest_checkpoint(prefix):
    import glob, re
    pattern = os.path.join(config.CHECKPOINT_DIR, f'{prefix}_epoch*.pth')
    files = glob.glob(pattern)
    if not files:
        return None
    def extract_epoch(p):
        m = re.search(r'epoch(\d+)', os.path.basename(p))
        return int(m.group(1)) if m else 0
    files.sort(key=extract_epoch)
    return files[-1]


def _load_checkpoint(filepath, model_class, optimizer, scheduler, device):
    ckpt = torch.load(filepath, map_location=device, weights_only=False)
    model_cfg = ckpt['config']
    model = model_class(**model_cfg).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    if 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    return (model, model_cfg,
            ckpt.get('epoch', 0),
            ckpt.get('best_auc', 0.5),
            ckpt.get('patience_counter', 0),
            ckpt.get('metrics', {}))


def evaluate_ext(model, val_df, user_features, item_features,
                  encoders, device, max_users=2000):
    """
    DIN evaluation: AUC via pairwise comparison (pos vs random neg).
    """
    model.eval()
    item_le = encoders['item_id']
    user_le = encoders['user_id']
    brand_le = encoders['brand_id']
    raw_to_idx = encoders.get('raw_to_idx', None)
    num_items = len(item_le.classes_)

    # --- Pre-build user feature dict ---
    user_feat_dict = {}
    for _, row in user_features.iterrows():
        uid = row['user_id']
        user_feat_dict[uid] = {c: row[c] for c in user_features.columns}

    # --- Pre-build item feature arrays ---
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

    # --- val: user → [item_enc, ...] ---
    raw_to_item_enc = {cls: i for i, cls in enumerate(item_le.classes_)}
    val_enc = val_df['click_article_id'].map(raw_to_item_enc).dropna().astype(int)
    val_pairs = val_df.loc[val_enc.index].copy()
    val_pairs['item_enc'] = val_enc.values
    valid_uids = set(user_feat_dict.keys()) & set(user_le.classes_)
    val_pairs = val_pairs[val_pairs['user_id'].isin(valid_uids)]
    val_by_user = val_pairs.groupby('user_id')['item_enc'].apply(list).to_dict()

    val_users = list(val_by_user.keys())
    if max_users and len(val_users) > max_users:
        val_users = list(np.random.choice(val_users, max_users, replace=False))

    H = config.HIST_LEN
    pos_scores, neg_scores = [], []

    for raw_uid in tqdm(val_users, desc="  evaluating"):
        feat = user_feat_dict[raw_uid]
        if raw_to_idx is not None:
            uidx = raw_to_idx.get(raw_uid, 0)
        else:
            uidx = user_le.transform([raw_uid])[0]

        # Pad user history
        for arr_name in ['hist_items', 'hist_brands', 'hist_ratings',
                          'hist_time_deltas', 'hist_verified']:
            arr = feat.get(arr_name, [])
            hl = len(arr)
            if hl > H:
                arr = arr[-H:]
                hl = H
            feat[arr_name] = arr

        hist_items = feat['hist_items'] + [0] * (H - len(feat['hist_items']))
        hist_brands = feat['hist_brands'] + [0] * (H - len(feat['hist_brands']))
        hist_ratings = feat['hist_ratings'] + [0.0] * (H - len(feat['hist_ratings']))
        hist_deltas = feat['hist_time_deltas'] + [0.0] * (H - len(feat['hist_time_deltas']))
        hist_verified = feat['hist_verified'] + [0] * (H - len(feat['hist_verified']))
        hl = min(len(feat['hist_items']), H)

        for item_idx in val_by_user.get(raw_uid, []):
            # --- Positive ---
            batch = {
                'user_id': torch.LongTensor([uidx]).to(device),
                'hist_items': torch.LongTensor([hist_items]).to(device),
                'hist_brands': torch.LongTensor([hist_brands]).to(device),
                'hist_ratings': torch.FloatTensor([hist_ratings]).to(device),
                'hist_time_deltas': torch.FloatTensor([hist_deltas]).to(device),
                'hist_verified': torch.FloatTensor([hist_verified]).to(device),
                'hist_len': torch.LongTensor([hl]).to(device),
                'click_count': torch.FloatTensor([feat['click_count_norm']]).to(device),
                'time_span': torch.FloatTensor([feat['time_span_norm']]).to(device),
                'user_avg_rating': torch.FloatTensor([feat['user_avg_rating_norm']]).to(device),
                'user_std_rating': torch.FloatTensor([feat['user_std_rating_norm']]).to(device),
                'user_verified_ratio': torch.FloatTensor([feat['user_verified_ratio']]).to(device),
                'user_avg_helpful': torch.FloatTensor([feat['user_avg_helpful_norm']]).to(device),
                'item_id': torch.LongTensor([item_idx]).to(device),
                'category_id': torch.LongTensor([item_cat_arr[item_idx]]).to(device),
                'brand_id': torch.LongTensor([item_brand_arr[item_idx]]).to(device),
                'item_click_count': torch.FloatTensor([item_click_arr[item_idx]]).to(device),
                'created_at_ts': torch.FloatTensor([item_created_arr[item_idx]]).to(device),
                'item_avg_rating': torch.FloatTensor([item_avg_rating_arr[item_idx]]).to(device),
                'item_rating_number': torch.FloatTensor([item_rating_num_arr[item_idx]]).to(device),
            }
            with torch.no_grad():
                pos_scores.append(torch.sigmoid(model(batch)).item())

            # --- 4 random negatives ---
            for _ in range(4):
                neg_idx = np.random.randint(1, num_items)
                neg_batch = dict(batch)
                neg_batch['item_id'] = torch.LongTensor([neg_idx]).to(device)
                neg_batch['category_id'] = torch.LongTensor([item_cat_arr[neg_idx]]).to(device)
                neg_batch['brand_id'] = torch.LongTensor([item_brand_arr[neg_idx]]).to(device)
                neg_batch['item_click_count'] = torch.FloatTensor([item_click_arr[neg_idx]]).to(device)
                neg_batch['created_at_ts'] = torch.FloatTensor([item_created_arr[neg_idx]]).to(device)
                neg_batch['item_avg_rating'] = torch.FloatTensor([item_avg_rating_arr[neg_idx]]).to(device)
                neg_batch['item_rating_number'] = torch.FloatTensor([item_rating_num_arr[neg_idx]]).to(device)
                with torch.no_grad():
                    neg_scores.append(torch.sigmoid(model(neg_batch)).item())

    if not pos_scores:
        return {'auc': 0.5, 'pos_mean': 0.5, 'n_users': 0}

    pos_arr = np.array(pos_scores)
    neg_arr = np.array(neg_scores)
    auc = np.mean(pos_arr[:, None] > neg_arr[None, :])

    return {
        'auc': auc,
        'pos_mean': float(pos_arr.mean()),
        'n_users': len(val_users),
    }


def train():
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Using device: {device}")

    # ================================================================
    # Step 1: Load Raw Data
    # ================================================================
    print("Step 1: Loading raw All_Beauty data...")
    raw_reviews = load_raw_reviews(
        config.DATA_PATH, config.EXT_CATEGORIES, offline=config.OFFLINE_MODE
    )
    raw_meta = load_raw_meta(config.DATA_PATH, config.EXT_CATEGORIES)

    # Prepare click_df with labels
    click_df = prepare_click_df(raw_reviews,
                                 min_user_inter=config.EXT_MIN_USER_INTER,
                                 min_item_inter=config.EXT_MIN_ITEM_INTER)

    print("\n--- Temporal Train/Val Split ---")
    train_click, val_click = split_train_val(click_df, config.EVAL_SPLIT_RATIO)

    # ================================================================
    # Step 2: Build Encoders & Features
    # ================================================================
    print("\nStep 2: Building extended encoders & features...")
    encoders = build_extended_encoders(click_df, raw_meta, config.EXT_ENCODER_PKL)
    num_users = len(encoders['user_id'].classes_)
    num_items = len(encoders['item_id'].classes_)
    num_brands = len(encoders['brand_id'].classes_)
    num_categories = len(encoders['category_id'].classes_)
    print(f">>> num_users={num_users:,}, num_items={num_items:,}, "
          f"num_brands={num_brands:,}, num_categories={num_categories}")

    item_features = build_extended_item_features(click_df, raw_meta, encoders)
    user_features = build_extended_user_features(
        click_df, raw_meta, encoders, hist_len=config.HIST_LEN
    )

    # ================================================================
    # Step 3: Build Dataset & DataLoader
    # ================================================================
    print("\nStep 3: Building extended DIN dataset...")
    dataset = DINExtendedDataset(
        train_click, user_features, item_features, encoders,
        hist_len=config.HIST_LEN, neg_ratio=config.DIN_NEG_RATIO
    )
    print(f">>> Dataset size: {len(dataset):,} BPR pairs")
    dataloader = DataLoader(
        dataset, batch_size=config.DIN_BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, collate_fn=collate_fn,
        persistent_workers=True,
    )

    # ================================================================
    # Step 4: Initialize Model
    # ================================================================
    print("\nStep 4: Initializing Extended DIN model...")
    model_cfg = {
        'num_users': num_users,
        'num_items': num_items,
        'num_brands': num_brands,
        'num_categories': num_categories,
        'embed_dim': config.DIN_EMBED_DIM,
        'brand_embed_dim': config.DIN_BRAND_EMBED_DIM,
        'hidden_dims': config.DIN_HIDDEN_DIMS,
        'hist_len': config.HIST_LEN,
        'dropout': config.DIN_DROPOUT,
    }
    model = DINExtendedModel(**model_cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f">>> Total parameters: {total_params:,}")

    optimizer = optim.Adam(model.parameters(),
                            lr=config.DIN_LEARNING_RATE,
                            weight_decay=config.DIN_WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    os.makedirs(config.MODEL_PATH, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # --- Resume ---
    start_epoch, best_auc, patience_counter = 0, 0.5, 0
    history = []

    resume_path = os.path.join(config.MODEL_PATH, 'din_ext_latest.pth')
    if not os.path.exists(resume_path):
        resume_path = _find_latest_checkpoint('din_ext')
    if resume_path:
        model, model_cfg, start_epoch, best_auc, patience_counter, prev_metrics = \
            _load_checkpoint(resume_path, DINExtendedModel, optimizer, scheduler, device)
        print(f">>> Resumed from epoch {start_epoch}, best AUC={best_auc:.4f}")
        hist_path = os.path.join(config.MODEL_PATH, 'din_ext_history.json')
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                history = json.load(f)[:start_epoch]

    # ================================================================
    # Step 5: Training Loop
    # ================================================================
    print(f"\nStep 5: Training extended DIN from epoch {start_epoch+1}...")
    best_epoch = start_epoch

    for epoch in range(start_epoch, config.DIN_NUM_EPOCHS):
        # --- Train ---
        model.train()
        total_loss = 0.0
        num_batches = 0

        user_keys = ['user_id', 'hist_items', 'hist_brands', 'hist_ratings',
                     'hist_time_deltas', 'hist_verified', 'hist_len',
                     'click_count', 'time_span', 'user_avg_rating',
                     'user_std_rating', 'user_verified_ratio', 'user_avg_helpful']
        pos_keys = ['item_id', 'category_id', 'brand_id', 'item_click_count',
                     'created_at_ts', 'item_avg_rating', 'item_rating_number']

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.DIN_NUM_EPOCHS}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}

            user_batch = {k: batch[k] for k in user_keys if k in batch}
            pos_batch = {k.replace('pos_', ''): batch[f'pos_{k}']
                         for k in pos_keys if f'pos_{k}' in batch}
            neg_batch = {k.replace('neg_', ''): batch[f'neg_{k}']
                         for k in pos_keys if f'neg_{k}' in batch}

            pos_score, neg_score = model.forward_pairwise(user_batch, pos_batch, neg_batch)
            loss = model.compute_bpr_loss(pos_score, neg_score)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            correct = (pos_score > neg_score).float().mean().item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct:.4f}")

        scheduler.step()
        train_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1}/{config.DIN_NUM_EPOCHS} | Train Loss={train_loss:.4f}")

        # --- Validate ---
        if config.SKIP_EVAL:
            metrics = {'auc': 0.5, 'pos_mean': 0.5, 'n_users': 0}
        else:
            print("  Running validation...")
            metrics = evaluate_ext(
                model, val_click, user_features, item_features,
                encoders, device, max_users=config.EVAL_MAX_USERS
            )

        auc = metrics['auc']
        pos_mean = metrics['pos_mean']
        print(f"  → Val AUC={auc:.4f} | Pos Mean={pos_mean:.4f} | Users={metrics['n_users']}")

        history.append({'epoch': epoch + 1, 'train_loss': train_loss,
                        'val_auc': auc, 'val_pos_mean': pos_mean})

        # --- Save checkpoint per epoch ---
        epoch_path = os.path.join(config.CHECKPOINT_DIR, f'din_ext_epoch{epoch+1:02d}.pth')
        _save_checkpoint(model, optimizer, scheduler, model_cfg,
                         epoch_path, epoch + 1, metrics, best_auc, patience_counter,
                         save_optimizer=False)

        # --- Cleanup: keep only last 3 epoch checkpoints to save disk ---
        import glob as _glob
        all_epoch_files = sorted(_glob.glob(
            os.path.join(config.CHECKPOINT_DIR, 'din_ext_epoch*.pth')
        ))
        for old_file in all_epoch_files[:-3]:
            os.remove(old_file)

        # --- Save latest (with optimizer for resume) ---
        latest_path = os.path.join(config.MODEL_PATH, 'din_ext_latest.pth')
        _save_checkpoint(model, optimizer, scheduler, model_cfg,
                         latest_path, epoch + 1, metrics, best_auc, patience_counter,
                         save_optimizer=True)

        # --- Track best ---
        is_better = config.SKIP_EVAL or auc > best_auc
        if is_better:
            best_auc = auc
            best_epoch = epoch + 1
            patience_counter = 0
            best_path = os.path.join(config.MODEL_PATH, 'din_ext_best.pth')
            _save_checkpoint(model, optimizer, scheduler, model_cfg,
                             best_path, epoch + 1, metrics, best_auc, patience_counter,
                             save_optimizer=True)
            print(f"  ★ New best! AUC={auc:.4f}")
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epochs "
                  f"(best AUC={best_auc:.4f} at epoch {best_epoch})")

        if patience_counter >= config.EARLY_STOP_PATIENCE:
            print(f"\n>>> Early stop triggered after {epoch+1} epochs")
            break

    # --- Save history ---
    history_path = os.path.join(config.MODEL_PATH, 'din_ext_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Extended DIN training complete!")
    print(f"  Best epoch: {best_epoch}, Best AUC={best_auc:.4f}")
    print(f"  Total time: {time.time() - start_time:.2f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    train()
