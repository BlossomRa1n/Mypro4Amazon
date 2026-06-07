"""
Extended DIN training script for Amazon Reviews 2023 Raw (BPR pairwise).

Data: raw review JSONL + meta JSONL (brand/verified/helpful/price/quality).
Model: DINExtendedModel — Dense + Sequence (DIN Target Attention).
Metrics: leave-last-out pairwise AUC (hist_len=5, 50 random negatives).
Loss: BPR (Bayesian Personalized Ranking) with AdamW + CosineAnnealingLR.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import time, json, pickle
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from data_loader import get_all_click_df
from data_loader_ext import (
    load_raw_meta, build_extended_encoders, build_extended_item_features,
    build_extended_user_features, build_hard_negative_index_ext, DINExtendedDataset,
    merge_jsonl_features,
)
from model_ext import DINExtendedModel
from evaluate import split_train_val
from itemcf import itemcf_sim


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
                  encoders, device, max_users=2000, hist_len=5, num_negatives=50):
    """
    DIN evaluation: leave-last-out + random negatives (SASRec-style).

    For each user, we take the LAST positive item in the validation set
    as the target, then score it against num_negatives random items that
    the user has not interacted with.  AUC (pairwise) is the fraction
    of (positive, negative) pairs where the positive item is scored higher
    than the randomly-sampled negative.

    This is NOT a standard ROC AUC — it's an adapted pairwise metric
    that directly answers "can the model rank the one true next item
    above random impostors?" and is directly comparable to the numbers
    reported in the SASRec literature.

    Key parameters:
      - hist_len=5:  truncate user history to last 5 items (SASRec convention)
      - num_negatives=50:  default from the SASRec paper
    """
    model.eval()
    item_le = encoders['item_id']
    user_le = encoders['user_id']
    raw_to_idx = encoders.get('raw_to_idx', None)
    num_items = len(item_le.classes_)
    num_users = len(user_le.classes_)

    # --- Pre-build item feature arrays (vectorized) ---
    nums = min(len(item_features), num_items)
    item_cat_arr = item_features['category_idx'].iloc[:nums].to_numpy(dtype=np.int64)
    item_brand_arr = item_features['brand_idx'].iloc[:nums].to_numpy(dtype=np.int64)
    item_click_arr = item_features['item_click_count_norm'].iloc[:nums].to_numpy(dtype=np.float32)
    item_created_arr = item_features['created_at_ts_norm'].iloc[:nums].to_numpy(dtype=np.float32)
    item_avg_rating_arr = item_features['item_avg_rating_norm'].iloc[:nums].to_numpy(dtype=np.float32)
    item_rating_num_arr = item_features['item_rating_number_norm'].iloc[:nums].to_numpy(dtype=np.float32)
    if nums < num_items:
        item_cat_arr = np.pad(item_cat_arr, (0, num_items - nums))
        item_brand_arr = np.pad(item_brand_arr, (0, num_items - nums))
        item_click_arr = np.pad(item_click_arr, (0, num_items - nums))
        item_created_arr = np.pad(item_created_arr, (0, num_items - nums))
        item_avg_rating_arr = np.pad(item_avg_rating_arr, (0, num_items - nums))
        item_rating_num_arr = np.pad(item_rating_num_arr, (0, num_items - nums))

    # --- Pre-build user feature arrays (indexed by encoded uid) ---
    user_hist_arr = [None] * num_users
    user_brand_hist_arr = [None] * num_users
    user_rating_hist_arr = [None] * num_users
    user_delta_hist_arr = [None] * num_users
    user_verified_hist_arr = [None] * num_users
    user_click_cnt_arr = np.zeros(num_users, dtype=np.float32)
    user_time_span_arr = np.zeros(num_users, dtype=np.float32)
    user_avg_rating_arr = np.zeros(num_users, dtype=np.float32)
    user_std_rating_arr = np.zeros(num_users, dtype=np.float32)
    user_verified_ratio_arr = np.zeros(num_users, dtype=np.float32)
    user_avg_helpful_arr = np.zeros(num_users, dtype=np.float32)

    # Map raw user_id → encoded user_idx
    uid_to_idx = {raw: i for i, raw in enumerate(user_le.classes_)}
    # --- For each user: pre-truncate history to last hist_len items ---
    for i in range(len(user_features)):
        row = user_features.iloc[i]
        raw_uid = row['user_id']
        uidx = uid_to_idx.get(raw_uid)
        if uidx is None:
            continue
        hist_items = row['hist_items']
        hist_brands = row['hist_brands']
        hist_ratings = row['hist_ratings']
        hist_deltas = row['hist_time_deltas']
        hist_verified = row['hist_verified']
        if len(hist_items) > hist_len:
            hist_items = hist_items[-hist_len:]
            hist_brands = hist_brands[-hist_len:]
            hist_ratings = hist_ratings[-hist_len:]
            hist_deltas = hist_deltas[-hist_len:]
            hist_verified = hist_verified[-hist_len:]
        user_hist_arr[uidx] = hist_items
        user_brand_hist_arr[uidx] = hist_brands
        user_rating_hist_arr[uidx] = hist_ratings
        user_delta_hist_arr[uidx] = hist_deltas
        user_verified_hist_arr[uidx] = hist_verified
        user_click_cnt_arr[uidx] = row['click_count_norm']
        user_time_span_arr[uidx] = row['time_span_norm']
        user_avg_rating_arr[uidx] = row['user_avg_rating_norm']
        user_std_rating_arr[uidx] = row['user_std_rating_norm']
        user_verified_ratio_arr[uidx] = row['user_verified_ratio']
        user_avg_helpful_arr[uidx] = row['user_avg_helpful_norm']

    # --- val: user → LAST positive item only (leave-last-out) ---
    raw_to_item_enc = {cls: i for i, cls in enumerate(item_le.classes_)}
    val_enc = val_df['click_article_id'].map(raw_to_item_enc).dropna().astype(int)
    val_pairs = val_df.loc[val_enc.index].copy()
    val_pairs['item_enc'] = val_enc.values
    val_last = val_pairs.groupby('user_id')['item_enc'].last().to_dict()

    # Map val user_ids to encoded indices
    val_user_pairs = []
    for raw_uid, last_item in val_last.items():
        uidx = uid_to_idx.get(raw_uid)
        if uidx is not None and user_hist_arr[uidx] is not None:
            val_user_pairs.append((uidx, last_item))

    if max_users and len(val_user_pairs) > max_users:
        indices = np.random.choice(len(val_user_pairs), max_users, replace=False)
        val_user_pairs = [val_user_pairs[i] for i in indices]

    H = hist_len

    def _pad_history(seq, L, fill_val=0):
        """Pad to L, truncating from left."""
        n = len(seq)
        if n > L:
            seq = seq[-L:]
            n = L
        return list(seq) + [fill_val] * (L - n)

    def _pad_history_float(seq, L):
        return _pad_history(seq, L, 0.0)

    pos_scores, neg_scores = [], []
    rng = np.random.default_rng()

    for uidx, last_item in tqdm(val_user_pairs, desc="  evaluating"):
        # Pad user history to eval hist_len
        hist_items = _pad_history(user_hist_arr[uidx] or [], H)
        hist_brands = _pad_history(user_brand_hist_arr[uidx] or [], H)
        hist_ratings = _pad_history_float(user_rating_hist_arr[uidx] or [], H)
        hist_deltas = _pad_history_float(user_delta_hist_arr[uidx] or [], H)
        hist_verified = _pad_history_float(user_verified_hist_arr[uidx] or [], H)
        hl = min(len(user_hist_arr[uidx] or []), H)
        # ================================================================
        # 旧逻辑: 遍历所有验证正样本 + 4 随机负样本
        # ================================================================

        u_tensor = torch.LongTensor([uidx]).to(device)
        hi_tensor = torch.LongTensor([hist_items]).to(device)
        hb_tensor = torch.LongTensor([hist_brands]).to(device)
        hr_tensor = torch.FloatTensor([hist_ratings]).to(device)
        hd_tensor = torch.FloatTensor([hist_deltas]).to(device)
        hv_tensor = torch.FloatTensor([hist_verified]).to(device)
        hl_tensor = torch.LongTensor([hl]).to(device)
        cc_tensor = torch.FloatTensor([user_click_cnt_arr[uidx]]).to(device)
        ts_tensor = torch.FloatTensor([user_time_span_arr[uidx]]).to(device)
        uar_tensor = torch.FloatTensor([user_avg_rating_arr[uidx]]).to(device)
        usr_tensor = torch.FloatTensor([user_std_rating_arr[uidx]]).to(device)
        uvr_tensor = torch.FloatTensor([user_verified_ratio_arr[uidx]]).to(device)
        uah_tensor = torch.FloatTensor([user_avg_helpful_arr[uidx]]).to(device)

        # Positive (only the LAST item in val — leave-last-out)
        batch = {
            'user_id': u_tensor,
            'hist_items': hi_tensor,
            'hist_brands': hb_tensor,
            'hist_ratings': hr_tensor,
            'hist_time_deltas': hd_tensor,
            'hist_verified': hv_tensor,
            'hist_len': hl_tensor,
            'click_count': cc_tensor,
            'time_span': ts_tensor,
            'user_avg_rating': uar_tensor,
            'user_std_rating': usr_tensor,
            'user_verified_ratio': uvr_tensor,
            'user_avg_helpful': uah_tensor,
            'item_id': torch.LongTensor([last_item]).to(device),
            'category_id': torch.LongTensor([item_cat_arr[last_item]]).to(device),
            'brand_id': torch.LongTensor([item_brand_arr[last_item]]).to(device),
            'item_click_count': torch.FloatTensor([item_click_arr[last_item]]).to(device),
            'created_at_ts': torch.FloatTensor([item_created_arr[last_item]]).to(device),
            'item_avg_rating': torch.FloatTensor([item_avg_rating_arr[last_item]]).to(device),
            'item_rating_number': torch.FloatTensor([item_rating_num_arr[last_item]]).to(device),
        }
        with torch.no_grad():
            pos_scores.append(torch.sigmoid(model(batch)).item())

        # Random negatives (num_negatives, default 50)
        neg_idxs = rng.integers(1, num_items, size=num_negatives)
        for neg_idx in neg_idxs:
            neg_batch = dict(batch)
            neg_batch['item_id'] = torch.LongTensor([int(neg_idx)]).to(device)
            neg_batch['category_id'] = torch.LongTensor([item_cat_arr[int(neg_idx)]]).to(device)
            neg_batch['brand_id'] = torch.LongTensor([item_brand_arr[int(neg_idx)]]).to(device)
            neg_batch['item_click_count'] = torch.FloatTensor([item_click_arr[int(neg_idx)]]).to(device)
            neg_batch['created_at_ts'] = torch.FloatTensor([item_created_arr[int(neg_idx)]]).to(device)
            neg_batch['item_avg_rating'] = torch.FloatTensor([item_avg_rating_arr[int(neg_idx)]]).to(device)
            neg_batch['item_rating_number'] = torch.FloatTensor([item_rating_num_arr[int(neg_idx)]]).to(device)
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
        'n_users': len(val_user_pairs),
        'hist_len': hist_len,
        'num_negatives': num_negatives,
    }


def train():
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Using device: {device}")

    # ================================================================
    # Step 1: Load Data (CSV 5-core + raw_meta brand)
    # ================================================================
    print("Step 1: Loading data...")
    click_df = get_all_click_df(config.DATA_PATH, offline=config.OFFLINE_MODE)
    raw_meta = load_raw_meta(config.DATA_PATH, config.AMAZON_CATEGORIES)
    # Merge JSONL extra features (verified/helpful) into CSV click_df
    click_df = merge_jsonl_features(click_df, config.DATA_PATH, config.AMAZON_CATEGORIES)

    print("\n--- Temporal Train/Val Split ---")
    train_click, val_click = split_train_val(click_df, config.EVAL_SPLIT_RATIO)

    # Build encoders on FULL data (IDs must cover all users/items)
    # Build features on TRAIN data only to prevent target leakage
    print("\nStep 2: Building extended encoders & features...")
    encoders = build_extended_encoders(click_df, raw_meta, config.EXT_ENCODER_PKL)
    num_users = len(encoders['user_id'].classes_)
    num_items = len(encoders['item_id'].classes_)
    num_brands = len(encoders['brand_id'].classes_)
    num_categories = len(encoders['category_id'].classes_)
    print(f">>> num_users={num_users:,}, num_items={num_items:,}, "
          f"num_brands={num_brands:,}, num_categories={num_categories}")

    # WARNING: Use train_click ONLY for features — val data must not leak into training
    item_features = build_extended_item_features(train_click, raw_meta, encoders)
    user_features = build_extended_user_features(
        train_click, raw_meta, encoders, hist_len=config.HIST_LEN
    )

    # --- Build ItemCF Hard Negative Index ---
    # NOTE: Hard negatives from ItemCF hurt BPR val AUC (they're similar to user
    # history → same as val pos items → model learns to suppress similar items).
    # Disabled by default. Re-enable with care: use late-epoch curriculum or
    # ensure neg pool excludes items from same user's val set.
    USE_HARD_NEGATIVES = False
    hard_neg_index = None
    if USE_HARD_NEGATIVES:
        if os.path.exists(config.ITEMCF_SIM_PKL):
            print("\n>>> Loading ItemCF similarity for Hard Negative Mining...")
            with open(config.ITEMCF_SIM_PKL, 'rb') as f:
                i2i_sim = pickle.load(f)
            hard_neg_index = build_hard_negative_index_ext(
                i2i_sim, encoders, num_hard_negatives=config.DIN_NEG_RATIO
            )
        else:
            print("\n>>> No ItemCF similarity found, building from scratch...")
            # Build user-item-time dict from click_df
            user_item_time = {}
            for uid, grp in click_df[click_df['click_label'] == 1].groupby('user_id'):
                user_item_time[uid] = list(zip(
                    grp['click_article_id'], grp['click_timestamp']
                ))
            i2i_sim = itemcf_sim(user_item_time)
            os.makedirs(os.path.dirname(config.ITEMCF_SIM_PKL), exist_ok=True)
            with open(config.ITEMCF_SIM_PKL, 'wb') as f:
                pickle.dump(i2i_sim, f)
            print(f">>> ItemCF similarity saved to {config.ITEMCF_SIM_PKL}")
            hard_neg_index = build_hard_negative_index_ext(
                i2i_sim, encoders, num_hard_negatives=config.DIN_NEG_RATIO
            )

    # ================================================================
    # Step 3: Build Dataset & DataLoader
    # ================================================================
    print("\nStep 3: Building extended DIN dataset...")
    dataset = DINExtendedDataset(
        train_click, user_features, item_features, encoders,
        hist_len=config.HIST_LEN, neg_ratio=config.DIN_NEG_RATIO,
        hard_neg_index=hard_neg_index, num_hard_negatives=config.DIN_NEG_RATIO
    )
    print(f">>> Dataset size: {len(dataset):,} BPR pairs")
    dataloader = DataLoader(
        dataset, batch_size=config.BATCH_SIZE, shuffle=True,
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
        'embed_dim': config.EMBED_DIM,
        'brand_embed_dim': config.DIN_BRAND_EMBED_DIM,
        'hidden_dims': config.DIN_HIDDEN_DIMS,
        'hist_len': config.HIST_LEN,
        'dropout': config.DIN_DROPOUT,
    }
    model = DINExtendedModel(**model_cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f">>> Total params: {total_params:,} "
          f"| embed={model_cfg['embed_dim']}d "
          f"| brand_embed={model_cfg['brand_embed_dim']}d "
          f"| hidden={model_cfg['hidden_dims']} "
          f"| hist={model_cfg['hist_len']} "
          f"| users={model_cfg['num_users']:,} "
          f"| items={model_cfg['num_items']:,} "
          f"| brands={model_cfg['num_brands']:,} "
          f"| cats={model_cfg['num_categories']}")

    # ---- ALS 预训练初始化 item_embedding (关键: 防止 BPR 坍塌) ----
    if getattr(config, 'USE_ALS_INIT', False):
        from als_init import init_model_with_als
        pos_click = click_df[click_df['click_label'] == 1]
        model, als_ok = init_model_with_als(
            model, pos_click, encoders['user_id'], encoders['item_id'],
            fix_embeddings=getattr(config, 'ALS_FIX_EMBEDDINGS', False)
        )
        if als_ok:
            print(">>> ALS pre-training loaded — Extended DIN starts from collaborative-filtering quality")
        else:
            print(">>> ALS unavailable for Extended DIN, falling back to random init")
    # ----------------------------------------------------------------

    optimizer = optim.AdamW(model.parameters(),
                            lr=config.LEARNING_RATE,
                            weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6
    )

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
        # Verify checkpoint is compatible with current data
        ckpt_num_users = model_cfg.get('num_users', 0)
        if ckpt_num_users != num_users:
            print(f">>> WARNING: checkpoint num_users={ckpt_num_users} != current={num_users}, re-initializing model")
            model = DINExtendedModel(**model_cfg).to(device)
            optimizer = optim.AdamW(model.parameters(),
                                    lr=config.LEARNING_RATE,
                                    weight_decay=config.WEIGHT_DECAY)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6
            )
            start_epoch, best_auc, patience_counter = 0, 0.5, 0
        else:
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

    for epoch in range(start_epoch, config.NUM_EPOCHS):
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

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
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
            # NaN check before clipping
            has_nan = False
            for p in model.parameters():
                if p.grad is not None and torch.isnan(p.grad).any():
                    has_nan = True
                    break
            if has_nan:
                print(f"  [WARN] NaN grad (loss={loss.item():.4f}), skipping batch")
                optimizer.zero_grad()
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            correct = (pos_score > neg_score).float().mean().item()
            # Note: BPR pairwise accuracy naturally trends toward 1.0 (pos>neg is easy for an MLP).
            # This metric is NOT a signal of overfitting — watch val AUC and train loss instead.
            pbar.set_postfix(loss=f"{loss.item():.4f}", bpr_pair_acc=f"{correct:.4f}")

        scheduler.step()
        train_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} | Train Loss={train_loss:.4f}")

        # --- Validate ---
        if config.SKIP_EVAL:
            metrics = {
                'auc': 0.5, 'pos_mean': 0.5, 'n_users': 0,
                'hist_len': 5, 'num_negatives': 50,
            }
        else:
            print("  Running validation (leave-last-out, HL=5, 50neg, max_users=2,000)...")
            metrics = evaluate_ext(
                model, val_click, user_features, item_features,
                encoders, device, max_users=config.EVAL_MAX_USERS
            )

        auc = metrics['auc']
        pos_mean = metrics['pos_mean']
        print(f"  → Val AUC={auc:.4f}/{pos_mean:.4f} | "
              f"Users={metrics['n_users']} | "
              f"HL={metrics.get('hist_len', 5)} | "
              f"Neg={metrics.get('num_negatives', 50)}")

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

    # --- Save model config alongside history for reproducibility ---
    cfg_dump = {
        'model': 'DINExtendedModel',
        'data': config.EXT_CATEGORIES,
        'embed_dim': config.EMBED_DIM,
        'brand_embed_dim': config.DIN_BRAND_EMBED_DIM,
        'hidden_dims': config.DIN_HIDDEN_DIMS,
        'hist_len': config.HIST_LEN,
        'dropout': config.DIN_DROPOUT,
        'lr': config.LEARNING_RATE,
        'batch_size': config.BATCH_SIZE,
        'weight_decay': config.WEIGHT_DECAY,
        'num_epochs': config.NUM_EPOCHS,
    }
    cfg_path = os.path.join(config.MODEL_PATH, 'din_ext_config.json')
    with open(cfg_path, 'w') as f:
        json.dump(cfg_dump, f, indent=2)
    print(f"\n>>> Model config saved to {cfg_path}")

    # --- Dump final results summary ---
    results_path = os.path.join(config.RESULT_PATH, 'din_ext_final_results.json')
    os.makedirs(config.RESULT_PATH, exist_ok=True)
    final_results = {
        'model': 'DINExtendedModel',
        'data': config.EXT_CATEGORIES,
        'best_epoch': best_epoch,
        'best_auc': best_auc,
        'eval_protocol': 'leave-last-out, 50 random negatives (SASRec-style)',
        'hyperparams': {
            'embed_dim': config.EMBED_DIM,
            'brand_embed_dim': config.DIN_BRAND_EMBED_DIM,
            'hidden_dims': config.DIN_HIDDEN_DIMS,
            'hist_len': config.HIST_LEN,
            'dropout': config.DIN_DROPOUT,
            'lr': config.LEARNING_RATE,
            'batch_size': config.BATCH_SIZE,
            'weight_decay': config.WEIGHT_DECAY,
            'num_epochs': config.NUM_EPOCHS,
        },
        'history': history,
    }
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"\n>>> Final results saved to {results_path}")

    print(f"\n{'='*60}")
    print(f"DIN-Ext training complete!")
    print(f"  Best epoch: {best_epoch}, Best AUC={best_auc:.4f}")
    print(f"  Data: {config.EXT_CATEGORIES} | LR={config.LEARNING_RATE} | "
          f"Embed={config.EMBED_DIM}d | Batch={config.BATCH_SIZE} | "
          f"HistLen={config.HIST_LEN} | Hidden={config.DIN_HIDDEN_DIMS}")
    print(f"  BrandEmbed={config.DIN_BRAND_EMBED_DIM}d | Dropout={config.DIN_DROPOUT} | "
          f"WD={config.WEIGHT_DECAY}")
    print(f"  Total time: {time.time() - start_time:.2f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    train()
