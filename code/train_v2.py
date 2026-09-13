"""
TwoTowerV2 训练脚本 (SASRec 用户塔 + InfoNCE loss + Hard Negative Mining + 扩展特征)

新增: 时序验证 + Recall@K 评估 + Early Stop + 每 epoch 存档
扩展特征: brand/quality (ItemTower) + user_avg/std_rating (SASRecUserTower)
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
    build_same_category_hard_neg_index, TwoTowerV2Dataset
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
    """在 checkpoints 目录下找最新存档"""
    import glob
    pattern = os.path.join(config.CHECKPOINT_DIR, f'{prefix}_epoch*.pth')
    files = glob.glob(pattern)
    if not files:
        return None
    # 按 epoch 号排序 (文件名含 epoch05)
    def extract_epoch(p):
        import re
        m = re.search(r'epoch(\d+)', os.path.basename(p))
        return int(m.group(1)) if m else 0
    files.sort(key=extract_epoch)
    latest = files[-1]
    print(f">>> Found checkpoint: {os.path.basename(latest)}")
    return latest


def _load_checkpoint(filepath, model_class, optimizer, scheduler, device):
    """加载存档，恢复模型/优化器/scheduler 状态 (优化器可选, 仅权重存档时不报错)"""
    ckpt = torch.load(filepath, map_location=device, weights_only=False)
    model_cfg = ckpt['config']
    model = model_class(**model_cfg).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    from baseline_runtime import rebind_optimizer
    rebind_optimizer(optimizer, model)
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


def build_embedding_hard_neg_index(item_vecs, k=4, chunk=4096, device=None, item_cat_arr=None):
    """
    ANCE 式动态难负: 用当前模型 item embedding 的余弦相似度重挖难负。
    对每个 item 找 top-k 最近邻 (embedding 空间), 替代静态 ItemCF 相似度。
    item_vecs: [N, D] numpy array → 返回 {item_idx: [neighbor_idx, ...]}
    item_cat_arr: [N] category_idx 数组, 若提供则只保留同品类难负
                  (同品类不足 k 个时返回不足数, 由 Dataset 随机补齐)
    """
    F = torch.nn.functional
    vecs = torch.FloatTensor(item_vecs).to(device)
    vecs = F.normalize(vecs, p=2, dim=-1)
    N = vecs.shape[0]
    index = {}
    arange_dev = torch.arange(chunk, device=device)
    cat_tensor = None
    if item_cat_arr is not None:
        cat_tensor = torch.LongTensor(item_cat_arr).to(device)   # [N]
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        block = vecs[start:end]                          # (C, D)
        sim = torch.matmul(block, vecs.t())              # (C, N)
        # 排除自身 (block 内对角线)
        c = end - start
        sim[arange_dev[:c], torch.arange(start, end, device=device)] = -1e9
        # 品类约束: 非同品类 item 相似度置 -1e9 (Phase 1 ANCE)
        if cat_tensor is not None:
            block_cat = cat_tensor[start:end]            # (C,)
            cat_match = (block_cat[:, None] == cat_tensor[None, :])   # (C, N) bool
            sim = torch.where(cat_match, sim, torch.full_like(sim, -1e9))
        topk_vals, topk_idx = torch.topk(sim, k, dim=1)  # (C, k) values/indices
        topk_idx = topk_idx.cpu().tolist()
        topk_vals = topk_vals.cpu().tolist()
        for i in range(c):
            # 过滤无效难负 (相似度 -1e9 的填充), 同品类不足时由 Dataset 随机补齐
            valid = [idx for idx, val in zip(topk_idx[i], topk_vals[i]) if val > -1e8]
            index[start + i] = valid
    return index


def train():
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Using device: {device}")

    # ================================================================
    # Step 1: 加载数据 (CSV 5-core, 131K 用户 + raw_meta brand)
    # ================================================================
    print("Step 1: Loading data...")
    click_df = get_all_click_df(config.DATA_PATH, offline=config.OFFLINE_MODE)
    raw_meta = load_raw_meta(config.DATA_PATH, config.AMAZON_CATEGORIES)
    # Merge JSONL extra features (verified/helpful) into CSV click_df
    click_df = merge_jsonl_features(click_df, config.DATA_PATH, config.AMAZON_CATEGORIES)
    articles_df = load_articles(config.DATA_PATH)

    print("\n--- Temporal Train/Val Split ---")
    train_click, val_click = split_train_val(click_df, config.EVAL_SPLIT_RATIO)

    # ================================================================
    # Step 2: 构建编码器 & 特征 (扩展: brand + quality + user_stats)
    # ================================================================
    print("\nStep 2: Building extended encoders & features...")
    # Use a SEPARATE cache file from V1 — extended encoders include brand_id
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
    # V2 Dataset 用 hist_items_trunc (旧 build_enhanced_user_features 的列名)
    # build_extended_user_features 输出 hist_items — 兼容一下
    if 'hist_items_trunc' not in user_features.columns:
        user_features['hist_items_trunc'] = user_features['hist_items']
    # user_avg_rating_norm / user_std_rating_norm 已由 build_extended_user_features 生成
    # 确认列存在 (build_extended 已做 normalize)

    # ANCE 品类约束用: item_idx -> category_idx 数组 (Phase 1)
    item_cat_arr = item_features['category_idx'].reindex(
        np.arange(num_items), fill_value=0).to_numpy(dtype=np.int64)

    # Hard Negative Index (用 raw reviews 的 ItemCF)
    hard_neg_index = None
    if config.USE_SAME_CATEGORY_HARD_NEG:
        # 同品类难负 (替代 ItemCF 相似难负)
        hard_neg_index = build_same_category_hard_neg_index(
            item_features, num_items, config.NUM_HARD_NEGATIVES)
    elif config.NUM_HARD_NEGATIVES > 0 and os.path.exists(config.ITEMCF_SIM_PKL):
        print(">>> Loading ItemCF similarity for hard negative mining...")
        with open(config.ITEMCF_SIM_PKL, 'rb') as f:
            i2i_sim = pickle.load(f)
        hard_neg_index = build_hard_negative_index(i2i_sim, encoders, config.NUM_HARD_NEGATIVES)

    # ================================================================
    # Step 3: 构建 Dataset / DataLoader (仅训练集)
    # ================================================================
    print("\nStep 3: Building V2 training dataset...")
    dataset = TwoTowerV2Dataset(
        train_click, user_features, item_features, encoders,
        num_items=num_items, hist_len=config.HIST_LEN,
        hard_neg_index=hard_neg_index, num_hard_negatives=config.NUM_HARD_NEGATIVES,
        num_brands=num_brands,
        num_explicit_negatives=config.NUM_EXPLICIT_NEGATIVES
    )
    dataloader = DataLoader(
        dataset, batch_size=config.V2_BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, collate_fn=collate_fn, persistent_workers=True
    )

    # ================================================================
    # Step 4: 初始化模型
    # ================================================================
    print("\nStep 4: Initializing TwoTowerV2 (SASRec + InfoNCE)...")
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
        'num_blocks': config.SASREC_NUM_BLOCKS,
        'dropout': config.SASREC_DROPOUT,
        'num_hard_negatives': config.NUM_HARD_NEGATIVES,
        'brand_embed_dim': getattr(config, 'V2_BRAND_EMBED_DIM', 64),
        'use_brand_pref': getattr(config, 'V2_USE_BRAND_PREF', False),
        'use_time_decay': getattr(config, 'V2_USE_TIME_DECAY', False),
        'time_decay_lambda': getattr(config, 'V2_TIME_DECAY_LAMBDA', 0.3),
    }
    model = TwoTowerV2Model(**model_cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f">>> Total parameters: {total_params:,}")

    # ---- ALS 预训练初始化 (如果有) ----
    if getattr(config, 'USE_ALS_INIT', False):
        from als_init import init_model_with_als
        pos_click = train_click[train_click['click_label'] == 1]
        model, als_ok = init_model_with_als(
            model, pos_click, encoders['user_id'], encoders['item_id'],
            fix_embeddings=getattr(config, 'ALS_FIX_EMBEDDINGS', False)
        )
        if als_ok:
            print(">>> ALS pre-training loaded — InfoNCE training starts from collaborative-filtering quality")
        else:
            print(">>> ALS unavailable, falling back to random init")
    # ----------------------------------

    optimizer = optim.AdamW(model.parameters(), lr=config.V2_LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.V2_NUM_EPOCHS, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler() if config.USE_AMP else None

    os.makedirs(config.MODEL_PATH, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # ---- Resume: 自动检测最新 checkpoint 并恢复训练 ----
    start_epoch = 0
    best_ndcg = 0.0
    best_hr = 0.0
    patience_counter = 0
    history = []
    # ---- Resume (优先含优化器的 latest.pth, 回退到 epoch checkpoint) ----
    resume_path = os.path.join(config.MODEL_PATH, 'two_tower_v2_latest.pth')
    if not os.path.exists(resume_path):
        resume_path = _find_latest_checkpoint('two_tower_v2')
    if resume_path:
        model, model_cfg, start_epoch, best_ndcg, patience_counter, prev_metrics, scaler_state = \
            _load_checkpoint(resume_path, TwoTowerV2Model, optimizer, scheduler, device)
        # 恢复 AMP scaler
        if scaler is not None and scaler_state is not None:
            scaler.load_state_dict(scaler_state)
        print(f">>> Resumed from epoch {start_epoch}, best NDCG@{config.EVAL_K}={best_ndcg:.4f}, "
              f"patience={patience_counter}")
        # 从 history.json 恢复历史记录
        history_path = os.path.join(config.MODEL_PATH, 'two_tower_v2_history.json')
        if os.path.exists(history_path):
            with open(history_path) as f:
                history = json.load(f)
            history = history[:start_epoch]  # 截断到 resume epoch
            if history:
                best_hr = max((h.get('val_hr', 0.0) for h in history), default=0.0)

    # ================================================================
    # Step 5: 训练循环
    # ================================================================
    print(f"\nStep 5: Training from epoch {start_epoch+1}...")
    print(f"    AMP={config.USE_AMP}, GradAccum={config.V2_GRADIENT_ACCUM_STEPS}, "
          f"Effective BS={config.V2_BATCH_SIZE * config.V2_GRADIENT_ACCUM_STEPS}")

    best_epoch = start_epoch if start_epoch > 0 else -1

    for epoch in range(start_epoch, config.V2_NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.V2_NUM_EPOCHS}")
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
                    scaler.update()  # must call update() to keep scaler state consistent
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
        print(f"Epoch {epoch+1}/{config.V2_NUM_EPOCHS} | "
              f"Train Loss={train_loss:.4f} | "
              f"LR={scheduler.get_last_lr()[0]:.6f}")

        # ---- 验证 ----
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

        # ---- ANCE 动态难负: 每 N epoch 用当前模型 embedding 重挖难负 ----
        if config.DYNAMIC_HARD_NEG_EVERY > 0 and (epoch + 1) % config.DYNAMIC_HARD_NEG_EVERY == 0:
            print(f"  >>> ANCE: recomputing hard negatives from current embeddings...")
            model.eval()
            all_emb = compute_item_embeddings(
                model, num_items, item_features, device, batch_size=2048
            ).cpu().numpy()
            dataset.hard_neg_index = build_embedding_hard_neg_index(
                all_emb, k=config.NUM_HARD_NEGATIVES, device=device,
                item_cat_arr=item_cat_arr)
            # persistent_workers 持有独立数据集副本, 重建 DataLoader 以传播新 index
            dataloader = DataLoader(
                dataset, batch_size=config.V2_BATCH_SIZE, shuffle=True,
                num_workers=config.NUM_WORKERS, collate_fn=collate_fn,
                persistent_workers=True
            )
            model.train()

        # ---- 每 epoch 存档 (仅权重, 不存优化器以节省磁盘) ----
        epoch_path = os.path.join(config.CHECKPOINT_DIR, f'two_tower_v2_epoch{epoch+1:02d}.pth')
        _save_checkpoint(model, optimizer, scheduler, scaler, model_cfg,
                        epoch_path, epoch + 1, metrics, best_ndcg, patience_counter,
                        save_optimizer=False)
        # 释放磁盘: 单个 checkpoint 含 265万 item embedding (~3.5GB), 只保留最近一个 epoch 存档
        if epoch >= 1:
            prev_path = os.path.join(config.CHECKPOINT_DIR, f'two_tower_v2_epoch{epoch:02d}.pth')
            if os.path.exists(prev_path):
                os.remove(prev_path)

        # ---- 更新最新存档 (含优化器, 用于断点恢复) ----
        latest_path = os.path.join(config.MODEL_PATH, 'two_tower_v2_latest.pth')
        _save_checkpoint(model, optimizer, scheduler, scaler, model_cfg,
                        latest_path, epoch + 1, metrics, best_ndcg, patience_counter,
                        save_optimizer=False)

        # ---- 更新最佳 (HR 优先, NDCG 作 tie-breaker) ----
        is_better = config.SKIP_EVAL or (hr > best_hr) or (hr == best_hr and ndcg > best_ndcg)
        if is_better:
            best_hr = hr
            best_ndcg = ndcg
            best_epoch = epoch + 1
            patience_counter = 0
            best_path = os.path.join(config.MODEL_PATH, 'two_tower_v2_best.pth')
            _save_checkpoint(model, optimizer, scheduler, scaler, model_cfg,
                            best_path, epoch + 1, metrics, best_ndcg, patience_counter,
                            save_optimizer=False)
            print(f"  ★ New best! HR@{config.EVAL_K}={hr:.4f} (NDCG={ndcg:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epochs "
                  f"(best HR@{config.EVAL_K}={best_hr:.4f} at epoch {best_epoch})")

        if patience_counter >= config.EARLY_STOP_PATIENCE:
            print(f"\n>>> Early stop triggered after {epoch+1} epochs")
            break

    # ---- 保存训练历史 ----
    history_path = os.path.join(config.MODEL_PATH, 'two_tower_v2_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n>>> Training history saved to {history_path}")

    print(f"\n{'='*60}")
    print(f"V2 training complete!")
    print(f"  Best epoch: {best_epoch}, Best NDCG@{config.EVAL_K}={best_ndcg:.4f}")
    print(f"  Data: {config.AMAZON_CATEGORIES} | LR={config.V2_LEARNING_RATE} | "
          f"Embed={config.EMBED_DIM}d | Batch={config.V2_BATCH_SIZE} | "
          f"HistLen={config.HIST_LEN} | Hidden={config.HIDDEN_DIMS}")
    print(f"  Temperature={config.INFONCE_TEMPERATURE} | Heads={config.SASREC_NUM_HEADS} | "
          f"Blocks={config.SASREC_NUM_BLOCKS} | Dropout={config.SASREC_DROPOUT}")
    print(f"  Total epochs run: {len(history)} | All checkpoints: {config.CHECKPOINT_DIR}/")
    print(f"{'='*60}")

    # --- Dump final results summary ---
    results_path = os.path.join(config.RESULT_PATH, 'v2_final_results.json')
    os.makedirs(config.RESULT_PATH, exist_ok=True)
    final_results = {
        'model': 'TwoTowerV2Model',
        'data': config.AMAZON_CATEGORIES,
        'best_epoch': best_epoch,
        'best_ndcg': best_ndcg,
        'eval_protocol': f'NDCG@{config.EVAL_K}',
        'hyperparams': {
            'embed_dim': config.EMBED_DIM,
            'hidden_dims': config.HIDDEN_DIMS,
            'hist_len': config.HIST_LEN,
            'sasrec_heads': config.SASREC_NUM_HEADS,
            'sasrec_blocks': config.SASREC_NUM_BLOCKS,
            'sasrec_dropout': config.SASREC_DROPOUT,
            'temperature': config.INFONCE_TEMPERATURE,
            'lr': config.V2_LEARNING_RATE,
            'batch_size': config.V2_BATCH_SIZE,
            'weight_decay': config.WEIGHT_DECAY,
            'num_epochs': config.V2_NUM_EPOCHS,
        },
        'history': history,
    }
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f">>> Final results saved to {results_path}")

    # ================================================================
    # Step 6: 最佳模型 → 预计算 item embeddings
    # ================================================================
    best_path = os.path.join(config.MODEL_PATH, 'two_tower_v2_best.pth')
    if not os.path.exists(best_path):
        print("[WARN] No best model saved, using last checkpoint.")
        best_path = os.path.join(config.CHECKPOINT_DIR,
                                 f'two_tower_v2_epoch{len(history):02d}.pth')

    print("\nStep 6: Loading best model for item embeddings...")
    # 只加载模型权重，不需要优化器状态
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model = TwoTowerV2Model(**ckpt['config']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    best_ep = ckpt.get('epoch', -1)
    best_metrics = ckpt.get('metrics', {})

    print("  Pre-computing item embeddings...")
    all_item_vecs = compute_item_embeddings(
        model, num_items, item_features, device, batch_size=2048
    )

    all_item_vecs = all_item_vecs.cpu().numpy()
    with open(config.V2_EMBED_PKL, 'wb') as f:
        pickle.dump(all_item_vecs, f)
    print(f">>> Item embeddings saved to {config.V2_EMBED_PKL}, shape={all_item_vecs.shape}")

    # 也保存一份到 config 中 inference_full.py 可能用到的路径
    from evaluate import _build_item_batch_for_indices
    print(f"\nAll done! Total time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    train()
