"""
Tokenized DIN (Step 1) training — five semantic tokens + concat/MLP fusion.

Model: DINTokenizedModel (seq/user/item/cross/dense 五路 token)。
Loss: BPR pairwise (与 train_din_ext 一致)。
Warm-start: 截断 SVD 同时初始化 user_embedding + item_embedding (FM cross_out 需要两者 CF 语义)。
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import time, json, glob
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from data_loader import get_all_click_df
from data_loader_ext import (
    load_raw_meta, build_extended_encoders, build_extended_item_features,
    build_extended_user_features, DINExtendedDataset, merge_jsonl_features,
)
from model_ext import DINTokenizedModel
from evaluate import split_train_val
from als_init import init_rerank_with_svd
from utils import set_seed
from train_din_ext import (
    evaluate_ext, collate_fn, _save_checkpoint, _find_latest_checkpoint, _load_checkpoint,
)

set_seed(42)

MODEL_PREFIX = 'din_token'


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
    num_brands = len(encoders['brand_id'].classes_)
    num_categories = len(encoders['category_id'].classes_)
    print(f">>> num_users={num_users:,}, num_items={num_items:,}, "
          f"num_brands={num_brands:,}, num_categories={num_categories}")

    item_features = build_extended_item_features(train_click, raw_meta, encoders)
    user_features = build_extended_user_features(
        train_click, raw_meta, encoders, hist_len=config.HIST_LEN
    )

    print("\nStep 3: Building extended DIN dataset...")
    dataset = DINExtendedDataset(
        train_click, user_features, item_features, encoders,
        hist_len=config.HIST_LEN, neg_ratio=config.DIN_NEG_RATIO,
        hard_neg_index=None, num_hard_negatives=config.DIN_NEG_RATIO
    )
    print(f">>> Dataset size: {len(dataset):,} BPR pairs")
    dataloader = DataLoader(
        dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, collate_fn=collate_fn,
        persistent_workers=True,
    )

    print("\nStep 4: Initializing DINTokenizedModel...")
    model_cfg = {
        'num_users': num_users,
        'num_items': num_items,
        'num_brands': num_brands,
        'num_categories': num_categories,
        'embed_dim': config.EMBED_DIM,
        'brand_embed_dim': config.DIN_BRAND_EMBED_DIM,
        'hidden_dims': config.DIN_TOKEN_HIDDEN_DIMS,
        'hist_len': config.HIST_LEN,
        'dropout': config.DIN_TOKEN_DROPOUT,
    }
    model = DINTokenizedModel(**model_cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f">>> Total params: {total_params:,} | hidden={model_cfg['hidden_dims']} "
          f"| hist={model_cfg['hist_len']}")

    # ---- 截断 SVD warm-start: 同时初始化 user_embedding + item_embedding ----
    if getattr(config, 'USE_ALS_INIT', False):
        pos_click = train_click[train_click['click_label'] == 1]
        model, svd_ok = init_rerank_with_svd(
            model, pos_click, encoders['user_id'], encoders['item_id'],
            fix_embeddings=getattr(config, 'ALS_FIX_EMBEDDINGS', False)
        )
        if svd_ok:
            print(">>> SVD warm-start loaded — user+item embeddings start from CF quality")
        else:
            print(">>> SVD unavailable, falling back to random init")

    optimizer = optim.AdamW(model.parameters(),
                            lr=config.LEARNING_RATE,
                            weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6
    )

    os.makedirs(config.MODEL_PATH, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    start_epoch, best_auc, patience_counter = 0, 0.5, 0
    history = []

    resume_path = os.path.join(config.MODEL_PATH, f'{MODEL_PREFIX}_latest.pth')
    if not os.path.exists(resume_path):
        resume_path = _find_latest_checkpoint(MODEL_PREFIX)
    if resume_path:
        model, model_cfg, start_epoch, best_auc, patience_counter, prev_metrics = \
            _load_checkpoint(resume_path, DINTokenizedModel, optimizer, scheduler, device)
        ckpt_num_users = model_cfg.get('num_users', 0)
        if ckpt_num_users != num_users:
            print(f">>> WARNING: checkpoint num_users={ckpt_num_users} != current={num_users}, re-initializing model")
            model = DINTokenizedModel(**model_cfg).to(device)
            optimizer = optim.AdamW(model.parameters(),
                                    lr=config.LEARNING_RATE,
                                    weight_decay=config.WEIGHT_DECAY)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6
            )
            start_epoch, best_auc, patience_counter = 0, 0.5, 0
        else:
            print(f">>> Resumed from epoch {start_epoch}, best AUC={best_auc:.4f}")
            hist_path = os.path.join(config.MODEL_PATH, f'{MODEL_PREFIX}_history.json')
            if os.path.exists(hist_path):
                with open(hist_path) as f:
                    history = json.load(f)[:start_epoch]

    print(f"\nStep 5: Training DINTokenizedModel from epoch {start_epoch+1}...")
    best_epoch = start_epoch

    for epoch in range(start_epoch, config.NUM_EPOCHS):
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
            pbar.set_postfix(loss=f"{loss.item():.4f}", bpr_pair_acc=f"{correct:.4f}")

        scheduler.step()
        train_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} | Train Loss={train_loss:.4f}")

        if config.SKIP_EVAL:
            metrics = {
                'auc': 0.5, 'pos_mean': 0.5, 'n_users': 0,
                'hist_len': 5, 'num_negatives': 50,
            }
        else:
            print("  Running validation (leave-last-out, HL=5, 50neg)...")
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

        epoch_path = os.path.join(config.CHECKPOINT_DIR, f'{MODEL_PREFIX}_epoch{epoch+1:02d}.pth')
        _save_checkpoint(model, optimizer, scheduler, model_cfg,
                         epoch_path, epoch + 1, metrics, best_auc, patience_counter,
                         save_optimizer=False)

        all_epoch_files = sorted(glob.glob(
            os.path.join(config.CHECKPOINT_DIR, f'{MODEL_PREFIX}_epoch*.pth')
        ))
        for old_file in all_epoch_files[:-3]:
            os.remove(old_file)

        latest_path = os.path.join(config.MODEL_PATH, f'{MODEL_PREFIX}_latest.pth')
        _save_checkpoint(model, optimizer, scheduler, model_cfg,
                         latest_path, epoch + 1, metrics, best_auc, patience_counter,
                         save_optimizer=False)

        is_better = config.SKIP_EVAL or auc > best_auc
        if is_better:
            best_auc = auc
            best_epoch = epoch + 1
            patience_counter = 0
            best_path = os.path.join(config.MODEL_PATH, f'{MODEL_PREFIX}_best.pth')
            _save_checkpoint(model, optimizer, scheduler, model_cfg,
                             best_path, epoch + 1, metrics, best_auc, patience_counter,
                             save_optimizer=False)
            print(f"  ★ New best! AUC={auc:.4f}")
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epochs "
                  f"(best AUC={best_auc:.4f} at epoch {best_epoch})")

        if patience_counter >= config.EARLY_STOP_PATIENCE:
            print(f"\n>>> Early stop triggered after {epoch+1} epochs")
            break

    history_path = os.path.join(config.MODEL_PATH, f'{MODEL_PREFIX}_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    cfg_dump = {
        'model': 'DINTokenizedModel',
        'data': config.EXT_CATEGORIES,
        'embed_dim': config.EMBED_DIM,
        'brand_embed_dim': config.DIN_BRAND_EMBED_DIM,
        'hidden_dims': config.DIN_TOKEN_HIDDEN_DIMS,
        'hist_len': config.HIST_LEN,
        'dropout': config.DIN_TOKEN_DROPOUT,
        'lr': config.LEARNING_RATE,
        'batch_size': config.BATCH_SIZE,
        'weight_decay': config.WEIGHT_DECAY,
        'num_epochs': config.NUM_EPOCHS,
    }
    cfg_path = os.path.join(config.MODEL_PATH, f'{MODEL_PREFIX}_config.json')
    with open(cfg_path, 'w') as f:
        json.dump(cfg_dump, f, indent=2)
    print(f"\n>>> Model config saved to {cfg_path}")

    results_path = os.path.join(config.RESULT_PATH, f'{MODEL_PREFIX}_final_results.json')
    os.makedirs(config.RESULT_PATH, exist_ok=True)
    final_results = {
        'model': 'DINTokenizedModel',
        'data': config.EXT_CATEGORIES,
        'best_epoch': best_epoch,
        'best_auc': best_auc,
        'eval_protocol': 'leave-last-out, 50 random negatives (SASRec-style)',
        'history': history,
    }
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"\n>>> Final results saved to {results_path}")

    print(f"\n{'='*60}")
    print(f"DINTokenizedModel training complete!")
    print(f"  Best epoch: {best_epoch}, Best AUC={best_auc:.4f}")
    print(f"  Data: {config.EXT_CATEGORIES} | LR={config.LEARNING_RATE} | "
          f"Embed={config.EMBED_DIM}d | Batch={config.BATCH_SIZE} | "
          f"HistLen={config.HIST_LEN} | Hidden={config.DIN_TOKEN_HIDDEN_DIMS}")
    print(f"  Total time: {time.time() - start_time:.2f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    train()
