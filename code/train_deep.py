"""
TwoTowerModel 训练脚本 (基础双塔 + BPR loss)

新增: 时序验证 + Recall@K 评估 + Early Stop + 每 epoch 存档
"""
import os, time, pickle, json
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from data_loader import (
    get_all_click_df, get_positive_click_df, load_articles,
    build_encoders, build_item_features, build_user_features,
    TwoTowerDataset
)
from model import TwoTowerModel
from evaluate import split_train_val, evaluate_two_tower, evaluate_two_tower_sampled


def collate_fn(batch):
    keys = batch[0].keys()
    result = {}
    for key in keys:
        result[key] = torch.stack([item[key] for item in batch])
    return result


def _save_checkpoint(model, optimizer, scheduler, model_cfg, filepath, epoch, metrics, best_ndcg, patience, save_optimizer=True):
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
    print(f">>> Found checkpoint: {os.path.basename(files[-1])}")
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
    return model, model_cfg, ckpt.get('epoch', 0), ckpt.get('best_ndcg', 0.0), ckpt.get('patience_counter', 0), ckpt.get('metrics', {})


def _load_best_model(filepath, model_class, device):
    """加载最佳模型用于最终 embedding 计算"""
    ckpt = torch.load(filepath, map_location=device, weights_only=False)
    model_cfg = ckpt['config']
    model = model_class(**model_cfg).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    return model, ckpt.get('epoch', -1), ckpt.get('metrics', {})


def train():
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Using device: {device}")

    # ================================================================
    # Step 1: 加载数据 & 时序分割
    # ================================================================
    print("Step 1: Loading data...")
    click_df = get_all_click_df(config.DATA_PATH, offline=config.OFFLINE_MODE)
    articles_df = load_articles(config.DATA_PATH)

    print("\n--- Temporal Train/Val Split ---")
    train_click, val_click = split_train_val(click_df, config.EVAL_SPLIT_RATIO)

    # ================================================================
    # Step 2: 构建编码器 & 特征
    # ================================================================
    print("\nStep 2: Building encoders & features...")
    # 编码器用全量数据构建 (包含验证集用户/物品)
    encoders = build_encoders(click_df, articles_df, config.ENCODER_PKL)
    num_users = len(encoders['user_id'].classes_)
    num_items = len(encoders['item_id'].classes_)
    num_categories = len(encoders.get('category_id', __import__('sklearn').preprocessing.LabelEncoder()).classes_) if 'category_id' in encoders else 1
    print(f">>> num_users={num_users}, num_items={num_items}, num_categories={num_categories}")

    # 物品特征用全量数据构建 (物品属性不依赖训练/验证分割)
    item_features = build_item_features(articles_df, click_df, encoders)
    # 用户特征只用训练集正样本构建 (避免验证集信息泄露，且不包括不喜欢的)
    user_features = build_user_features(get_positive_click_df(train_click), encoders, hist_len=config.HIST_LEN)

    # ================================================================
    # Step 3: 构建 Dataset / DataLoader (仅训练集)
    # ================================================================
    print("\nStep 3: Building training dataset...")
    dataset = TwoTowerDataset(
        train_click, user_features, item_features, encoders,
        num_items=num_items, hist_len=config.HIST_LEN, num_negatives=config.NUM_NEGATIVES
    )
    dataloader = DataLoader(
        dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, collate_fn=collate_fn, persistent_workers=True
    )

    # ================================================================
    # Step 4: 初始化模型
    # ================================================================
    print("\nStep 4: Initializing TwoTowerModel...")
    model_cfg = {
        'num_users': num_users,
        'num_items': num_items,
        'num_categories': num_categories,
        'embed_dim': config.EMBED_DIM,
        'hidden_dims': config.HIDDEN_DIMS,
        'hist_len': config.HIST_LEN,
        'temperature': config.TEMPERATURE,
    }
    model = TwoTowerModel(**model_cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f">>> Total parameters: {total_params:,}")

    # ---- ALS 预训练初始化 (如果有) ----
    if getattr(config, 'USE_ALS_INIT', False):
        from als_init import init_model_with_als
        pos_click = get_positive_click_df(click_df)
        model, als_ok = init_model_with_als(
            model, pos_click, encoders['user_id'], encoders['item_id'],
            fix_embeddings=getattr(config, 'ALS_FIX_EMBEDDINGS', False)
        )
        if als_ok:
            print(">>> ALS pre-training loaded — model starts with collaborative-filtering quality embeddings")
        else:
            print(">>> ALS unavailable, falling back to random init")
    # ----------------------------------

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    os.makedirs(config.MODEL_PATH, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # ---- Resume (优先加载含优化器的 latest.pth, 否则从 epoch checkpoint 恢复) ----
    start_epoch, best_ndcg, patience_counter = 0, 0.0, 0
    history = []
    # 1) 尝试 latest.pth (含优化器, 可断点续训)
    resume_path = os.path.join(config.MODEL_PATH, 'two_tower_latest.pth')
    if not os.path.exists(resume_path):
        # 2) 回退到最新 epoch checkpoint (仅权重, 优化器从头初始化)
        resume_path = _find_latest_checkpoint('two_tower')
    if resume_path and os.path.exists(resume_path):
        model, model_cfg, start_epoch, best_ndcg, patience_counter, prev_metrics = \
            _load_checkpoint(resume_path, TwoTowerModel, optimizer, scheduler, device)
        print(f">>> Resumed from epoch {start_epoch}, best NDCG@{config.EVAL_K}={best_ndcg:.4f}")
        history_path = os.path.join(config.MODEL_PATH, 'two_tower_history.json')
        if os.path.exists(history_path):
            with open(history_path) as f:
                history = json.load(f)[:start_epoch]

    # ================================================================
    # Step 5: 训练循环
    # ================================================================
    print(f"\nStep 5: Training from epoch {start_epoch+1}...")
    best_epoch = start_epoch

    for epoch in range(start_epoch, config.NUM_EPOCHS):
        # ---- 训练 ----
        model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            pos_score, neg_score, _, _, _ = model(batch)
            loss = model.compute_loss(pos_score, neg_score,
                                       click_weight=batch.get('click_weight'))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        train_loss = total_loss / num_batches

        # ---- 验证 (本地验证模式可跳过) ----
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
        info = f"(samp)" if method == 'sampled' else ""
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} | "
              f"Train Loss={train_loss:.4f} | "
              f"Val HR@{config.EVAL_K}={hr:.4f} ({hr*100:.1f}%) | "
              f"NDCG@{config.EVAL_K}={ndcg:.4f} | "
              f"Users={metrics['n_users']} {info}")

        history.append({'epoch': epoch + 1, 'train_loss': train_loss,
                        'val_hr': hr, 'val_ndcg': ndcg})

        # ---- 每 epoch 存档 (仅权重, 不存优化器以节省磁盘) ----
        epoch_path = os.path.join(config.CHECKPOINT_DIR, f'two_tower_epoch{epoch+1:02d}.pth')
        _save_checkpoint(model, optimizer, scheduler, model_cfg,
                        epoch_path, epoch + 1, metrics, best_ndcg, patience_counter,
                        save_optimizer=False)

        # ---- 更新最新存档 (含优化器, 用于断点恢复) ----
        latest_path = os.path.join(config.MODEL_PATH, 'two_tower_latest.pth')
        _save_checkpoint(model, optimizer, scheduler, model_cfg,
                        latest_path, epoch + 1, metrics, best_ndcg, patience_counter,
                        save_optimizer=True)

        # ---- 更新最佳模型 (基于 NDCG, 含优化器) ----
        is_better = config.SKIP_EVAL or ndcg > best_ndcg
        if is_better:
            best_ndcg = ndcg
            best_epoch = epoch + 1
            patience_counter = 0
            best_path = os.path.join(config.MODEL_PATH, 'two_tower_best.pth')
            _save_checkpoint(model, optimizer, scheduler, model_cfg,
                            best_path, epoch + 1, metrics, best_ndcg, patience_counter,
                            save_optimizer=True)
            print(f"  ★ New best! NDCG@{config.EVAL_K}={ndcg:.4f}, saved to {best_path}")
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epochs "
                  f"(best NDCG@{config.EVAL_K}={best_ndcg:.4f} at epoch {best_epoch})")

        # ---- Early Stop ----
        if patience_counter >= config.EARLY_STOP_PATIENCE:
            print(f"\n>>> Early stop triggered after {epoch+1} epochs "
                  f"(no improvement for {config.EARLY_STOP_PATIENCE} epochs)")
            break

    # ---- 保存训练历史 ----
    history_path = os.path.join(config.MODEL_PATH, 'two_tower_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n>>> Training history saved to {history_path}")

    # ---- 总结 ----
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Best epoch: {best_epoch}, Best NDCG@{config.EVAL_K}={best_ndcg:.4f}")
    print(f"  Total epochs run: {len(history)}")
    print(f"  All checkpoints: {config.CHECKPOINT_DIR}/")
    print(f"{'='*60}")

    # ================================================================
    # Step 6: 加载最佳模型 → 预计算 item embeddings
    # ================================================================
    best_path = os.path.join(config.MODEL_PATH, 'two_tower_best.pth')
    if not os.path.exists(best_path):
        print("[WARN] No best model saved, using last checkpoint for embeddings.")
        best_path = os.path.join(config.CHECKPOINT_DIR,
                                 f'two_tower_epoch{len(history):02d}.pth')

    print("\nStep 6: Loading best model for item embeddings...")
    model, best_ep, best_metrics = _load_best_model(best_path, TwoTowerModel, device)

    print("  Pre-computing item embeddings...")
    model.eval()
    all_item_indices = np.arange(num_items)
    batch_size = 2048
    all_item_vecs = []

    with torch.no_grad():
        for start in range(0, num_items, batch_size):
            end = min(start + batch_size, num_items)
            indices = all_item_indices[start:end]
            subset = item_features.reindex(indices, fill_value=0)
            item_batch = {
                'item_id': torch.LongTensor(indices).to(device),
                'category_id': torch.LongTensor(subset['category_idx'].values.astype(int)).to(device),
                'item_click_count': torch.FloatTensor(subset['item_click_count_norm'].values).to(device),
                'created_at_ts': torch.FloatTensor(subset['created_at_ts_norm'].values).to(device),
            }
            item_vecs = model.get_item_embedding(item_batch)
            all_item_vecs.append(item_vecs.cpu().numpy())

    all_item_vecs = np.concatenate(all_item_vecs, axis=0)
    with open(config.EMBED_PKL, 'wb') as f:
        pickle.dump(all_item_vecs, f)
    print(f">>> Item embeddings saved to {config.EMBED_PKL}, shape={all_item_vecs.shape}")

    print(f"\nAll done! Total time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    train()
