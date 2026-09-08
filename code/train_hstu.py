"""
HSTU 用户塔训练脚本 (第 5 路召回) — pointwise 序列传导 + InfoNCE + Hard Negative + SVD warm-start

与 train_v2.py 的唯一区别: 用户塔从 SASRec (softmax 自注意力) 换成 HSTU (pointwise 传导 + 因果 cumsum)。
其余完全一致 (同一 ItemTower / InfoNCE temp=0.07 / hard_neg=3 / SVD warm-start / 时序评估),
保证 HSTU vs SASRec 的唯一变量是「序列编码器」, 便于归因与互补性验证。
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import time, pickle, json
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from data_loader import (
    get_all_click_df, load_articles, build_hard_negative_index,
    TwoTowerV2Dataset
)
from data_loader_ext import (
    load_raw_meta, build_extended_encoders, build_extended_item_features,
    build_extended_user_features, build_hard_negative_index_ext,
    merge_jsonl_features,
)
from model import TwoTowerV2Model
from evaluate import split_train_val, evaluate_two_tower, evaluate_two_tower_sampled, compute_item_embeddings
from utils import set_seed

set_seed(42)


def collate_fn(batch):
    keys = batch[0].keys()
    result = {}
    for key in keys:
        result[key] = torch.stack([item[key] for item in batch])
    return result


def _save_checkpoint(model, optimizer, scheduler, scaler, model_cfg, filepath, epoch, metrics, best_ndcg, patience, save_optimizer=True):
    data = {
        'model_state_dict': model.state_dict(),
        'config': model_cfg,
        'epoch': epoch,
        'metrics': metrics,
        'best_ndcg': best_ndcg,
        'patience_counter': patience,
    }
    if save_optimizer:
        data['optimizer_state_dict'] = optimizer.state_dict()
        data['scheduler_state_dict'] = scheduler.state_dict()
        data['scaler_state_dict'] = scaler.state_dict() if scaler else None
    torch.save(data, filepath)


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
    latest = files[-1]
    print(f">>> Found checkpoint: {os.path.basename(latest)}")
    return latest


def _load_checkpoint(filepath, model_class, optimizer, scheduler, device):
    ckpt = torch.load(filepath, map_location=device, weights_only=False)
    model_cfg = ckpt['config']
    model = model_class(**model_cfg).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    if 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    start_epoch = ckpt.get('epoch', 0)
    best_ndcg = ckpt.get('best_ndcg', 0.0)
    patience_counter = ckpt.get('patience_counter', 0)
    metrics = ckpt.get('metrics', {})
    scaler_state = ckpt.get('scaler_state_dict')
    return model, model_cfg, start_epoch, best_ndcg, patience_counter, metrics, scaler_state


def train():
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Using device: {device}")

    print("Step 1: Loading data...")
    click_df = get_all_click_df(config.DATA_PATH, offline=config.OFFLINE_MODE)
    raw_meta = load_raw_meta(config.DATA_PATH, config.AMAZON_CATEGORIES)
    click_df = merge_jsonl_features(click_df, config.DATA_PATH, config.AMAZON_CATEGORIES)
    articles_df = load_articles(config.DATA_PATH)

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

    hard_neg_index = None
    if config.NUM_HARD_NEGATIVES > 0 and os.path.exists(config.ITEMCF_SIM_PKL):
        print(">>> Loading ItemCF similarity for hard negative mining...")
        with open(config.ITEMCF_SIM_PKL, 'rb') as f:
            i2i_sim = pickle.load(f)
        hard_neg_index = build_hard_negative_index(i2i_sim, encoders, config.NUM_HARD_NEGATIVES)

    print("\nStep 3: Building V2 training dataset (HSTU user tower)...")
    dataset = TwoTowerV2Dataset(
        train_click, user_features, item_features, encoders,
        num_items=num_items, hist_len=config.HIST_LEN,
        hard_neg_index=hard_neg_index, num_hard_negatives=config.NUM_HARD_NEGATIVES,
        num_brands=num_brands
    )
    dataloader = DataLoader(
        dataset, batch_size=config.HSTU_BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, collate_fn=collate_fn, persistent_workers=True
    )

    print("\nStep 4: Initializing TwoTowerV2 (HSTU user tower + InfoNCE)...")
    model_cfg = {
        'num_users': num_users,
        'num_items': num_items,
        'num_categories': num_categories,
        'num_brands': num_brands,
        'embed_dim': config.EMBED_DIM,
        'hidden_dims': config.HIDDEN_DIMS,
        'hist_len': config.HIST_LEN,
        'temperature': config.INFONCE_TEMPERATURE,
        'num_heads': config.SASREC_NUM_HEADS,
        'num_blocks': config.HSTU_NUM_BLOCKS,
        'dropout': config.SASREC_DROPOUT,
        'num_hard_negatives': config.NUM_HARD_NEGATIVES,
        'brand_embed_dim': getattr(config, 'V2_BRAND_EMBED_DIM', 64),
        'user_tower_type': 'hstu',
    }
    model = TwoTowerV2Model(**model_cfg).to(device)
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
            print(">>> SVD pre-training loaded — HSTU starts from collaborative-filtering quality")
        else:
            print(">>> SVD unavailable, falling back to random init")

    optimizer = optim.AdamW(model.parameters(), lr=config.HSTU_LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.HSTU_NUM_EPOCHS, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler() if config.USE_AMP else None

    os.makedirs(config.MODEL_PATH, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    start_epoch = 0
    best_ndcg = 0.0
    patience_counter = 0
    history = []
    resume_path = os.path.join(config.MODEL_PATH, 'hstu_latest.pth')
    if not os.path.exists(resume_path):
        resume_path = _find_latest_checkpoint('hstu')
    if resume_path:
        model, model_cfg, start_epoch, best_ndcg, patience_counter, prev_metrics, scaler_state = \
            _load_checkpoint(resume_path, TwoTowerV2Model, optimizer, scheduler, device)
        if scaler is not None and scaler_state is not None:
            scaler.load_state_dict(scaler_state)
        print(f">>> Resumed from epoch {start_epoch}, best NDCG@{config.EVAL_K}={best_ndcg:.4f}, "
              f"patience={patience_counter}")
        history_path = os.path.join(config.MODEL_PATH, 'hstu_history.json')
        if os.path.exists(history_path):
            with open(history_path) as f:
                history = json.load(f)
            history = history[:start_epoch]

    print(f"\nStep 5: Training from epoch {start_epoch+1}...")
    print(f"    AMP={config.USE_AMP}, Effective BS={config.HSTU_BATCH_SIZE}")

    best_epoch = start_epoch if start_epoch > 0 else -1

    for epoch in range(start_epoch, config.HSTU_NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.HSTU_NUM_EPOCHS}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=config.USE_AMP):
                user_vec, pos_item_vec = model(batch)

                hard_neg_vecs = None
                if config.NUM_HARD_NEGATIVES > 0 and 'hard_neg_ids' in batch and batch['hard_neg_ids'].numel() > 0:
                    bs = batch['hard_neg_ids'].shape[0]
                    nh = batch['hard_neg_ids'].shape[1]
                    hard_neg_batch = {
                        'item_id': batch['hard_neg_ids'].reshape(-1),
                        'category_id': batch['hard_neg_category_ids'].reshape(-1),
                        'brand_id': batch.get('hard_neg_brand_ids', batch['hard_neg_ids'] * 0).reshape(-1),
                        'item_click_count': batch['hard_neg_item_click_count'].reshape(-1),
                        'created_at_ts': batch['hard_neg_created_at_ts'].reshape(-1),
                        'item_avg_rating': batch.get('hard_neg_item_avg_rating', torch.zeros_like(batch['hard_neg_ids'].float())).reshape(-1),
                        'item_rating_number': batch.get('hard_neg_item_rating_number', torch.zeros_like(batch['hard_neg_ids'].float())).reshape(-1),
                    }
                    hard_neg_vecs = model.get_item_embedding(hard_neg_batch)
                    hard_neg_vecs = hard_neg_vecs.reshape(bs, nh, -1)

                loss = model.compute_infonce_loss(
                    user_vec, pos_item_vec, hard_neg_vecs,
                    click_weight=batch.get('click_weight'))

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
        print(f"Epoch {epoch+1}/{config.HSTU_NUM_EPOCHS} | "
              f"Train Loss={train_loss:.4f} | "
              f"LR={scheduler.get_last_lr()[0]:.6f}")

        if config.SKIP_EVAL:
            metrics = {'hr': 0.0, 'ndcg': 0.0, 'n_users': 0}
        elif getattr(config, 'EVAL_FULL_RANK', True):
            print("  Running validation (Full rank)...")
            metrics = evaluate_two_tower(
                model, val_click, train_click, user_features, item_features,
                encoders, device, k=config.EVAL_K, max_users=config.EVAL_MAX_USERS
            )
        else:
            print(f"  Running validation (Sampled, {config.EVAL_NUM_NEGATIVES} negatives)...")
            metrics = evaluate_two_tower_sampled(
                model, val_click, train_click, user_features, item_features,
                encoders, device, k=config.EVAL_K, max_users=config.EVAL_MAX_USERS,
                num_negatives=config.EVAL_NUM_NEGATIVES
            )

        hr = metrics['hr']
        ndcg = metrics['ndcg']
        method = metrics.get('method', 'full')
        n_neg = metrics.get('n_negatives', 0)
        info = f"(sampled {n_neg})" if method == 'sampled' else "(full)"
        print(f"  → Val HR@{config.EVAL_K}={hr:.4f} ({hr*100:.1f}%) | "
              f"NDCG@{config.EVAL_K}={ndcg:.4f} | "
              f"Users={metrics['n_users']} {info}")

        history.append({'epoch': epoch + 1, 'train_loss': train_loss,
                        'val_hr': hr, 'val_ndcg': ndcg})

        epoch_path = os.path.join(config.CHECKPOINT_DIR, f'hstu_epoch{epoch+1:02d}.pth')
        _save_checkpoint(model, optimizer, scheduler, scaler, model_cfg,
                        epoch_path, epoch + 1, metrics, best_ndcg, patience_counter,
                        save_optimizer=False)
        if epoch >= 1:
            prev_path = os.path.join(config.CHECKPOINT_DIR, f'hstu_epoch{epoch:02d}.pth')
            if os.path.exists(prev_path):
                os.remove(prev_path)

        latest_path = os.path.join(config.MODEL_PATH, 'hstu_latest.pth')
        _save_checkpoint(model, optimizer, scheduler, scaler, model_cfg,
                        latest_path, epoch + 1, metrics, best_ndcg, patience_counter,
                        save_optimizer=False)

        is_better = config.SKIP_EVAL or ndcg > best_ndcg
        if is_better:
            best_ndcg = ndcg
            best_epoch = epoch + 1
            patience_counter = 0
            best_path = config.HSTU_BEST_FILE
            _save_checkpoint(model, optimizer, scheduler, scaler, model_cfg,
                            best_path, epoch + 1, metrics, best_ndcg, patience_counter,
                            save_optimizer=False)
            print(f"  ★ New best! NDCG@{config.EVAL_K}={ndcg:.4f}")
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epochs "
                  f"(best NDCG@{config.EVAL_K}={best_ndcg:.4f} at epoch {best_epoch})")

        if patience_counter >= config.EARLY_STOP_PATIENCE:
            print(f"\n>>> Early stop triggered after {epoch+1} epochs")
            break

    history_path = os.path.join(config.MODEL_PATH, 'hstu_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n>>> Training history saved to {history_path}")

    print(f"\n{'='*60}")
    print(f"HSTU training complete!")
    print(f"  Best epoch: {best_epoch}, Best NDCG@{config.EVAL_K}={best_ndcg:.4f}")
    print(f"  User tower: HSTU (pointwise transduction, {config.HSTU_NUM_BLOCKS} blocks) | "
          f"Embed={config.EMBED_DIM}d | Batch={config.HSTU_BATCH_SIZE}")
    print(f"  Temperature={config.INFONCE_TEMPERATURE} | HardNeg={config.NUM_HARD_NEGATIVES} | "
          f"HistLen={config.HIST_LEN}")
    print(f"{'='*60}")

    results_path = os.path.join(config.RESULT_PATH, 'hstu_final_results.json')
    os.makedirs(config.RESULT_PATH, exist_ok=True)
    final_results = {
        'model': 'TwoTowerV2Model (HSTU user tower)',
        'data': config.AMAZON_CATEGORIES,
        'best_epoch': best_epoch,
        'best_ndcg': best_ndcg,
        'eval_protocol': f'NDCG@{config.EVAL_K}',
        'hyperparams': {
            'embed_dim': config.EMBED_DIM,
            'hidden_dims': config.HIDDEN_DIMS,
            'hist_len': config.HIST_LEN,
            'hstu_blocks': config.HSTU_NUM_BLOCKS,
            'dropout': config.SASREC_DROPOUT,
            'temperature': config.INFONCE_TEMPERATURE,
            'lr': config.HSTU_LEARNING_RATE,
            'batch_size': config.HSTU_BATCH_SIZE,
            'num_epochs': config.HSTU_NUM_EPOCHS,
        },
        'history': history,
    }
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f">>> Final results saved to {results_path}")

    best_path = config.HSTU_BEST_FILE
    if not os.path.exists(best_path):
        print("[WARN] No best model saved, using last checkpoint.")
        best_path = os.path.join(config.CHECKPOINT_DIR,
                                 f'hstu_epoch{len(history):02d}.pth')

    print("\nStep 6: Loading best model for item embeddings...")
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model = TwoTowerV2Model(**ckpt['config']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    best_ep = ckpt.get('epoch', -1)

    print("  Pre-computing item embeddings...")
    all_item_vecs = compute_item_embeddings(
        model, num_items, item_features, device, batch_size=2048
    )

    all_item_vecs = all_item_vecs.cpu().numpy()
    with open(config.HSTU_EMBED_PKL, 'wb') as f:
        pickle.dump(all_item_vecs, f)
    print(f">>> Item embeddings saved to {config.HSTU_EMBED_PKL}, shape={all_item_vecs.shape}")

    print(f"\nAll done! Total time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    train()
