"""
双塔模型推理脚本
================
作用：用已经训练好的双塔模型，为测试集中的每个用户生成 Top-5 新闻推荐。

前提：必须先运行过 train_deep.py，生成了模型文件和物品向量。

类比理解：
    训练阶段 = 教模型理解"用户喜欢什么样的文章"
    推理阶段 = 让模型给新用户推荐文章
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import config
from data_loader import (
    get_all_click_df, get_positive_click_df, load_articles,
    build_encoders, build_user_features, build_item_features
)
from model import TwoTowerModel
from submit import save_submission


def inference():
    """推理主函数：加载训练好的模型，为测试用户生成推荐结果。"""

    # ============================================================
    # 选择运行设备：优先用 GPU，没有就退回 CPU
    # ============================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Using device: {device}")

    # ============================================================
    # 检查：模型文件是否存在？（如果没训练过就跑推理，会在这里报错退出）
    # ============================================================
    if not os.path.exists(config.DEEP_MODEL_FILE):
        print(f"Error: Model file not found at {config.DEEP_MODEL_FILE}")
        print("Please run train_deep.py first!")
        return

    if not os.path.exists(config.EMBED_PKL):
        print(f"Error: Item embeddings not found at {config.EMBED_PKL}")
        print("Please run train_deep.py first!")
        return

    # ============================================================
    # Step 1: 加载训练好的模型
    #
    # 类比：训练时模型学会了"理解用户"和"理解文章"，
    # 现在要把这个"大脑"重新加载到内存里。
    # ============================================================
    print("Step 1: Loading model...")

    # torch.load 会把之前 save 的整个字典读出来，包括模型参数和配置
    checkpoint = torch.load(config.DEEP_MODEL_FILE, map_location=device)
    model_cfg = checkpoint['config']

    # 用训练时的配置重建一个"空壳"模型（参数还没填进去）
    model = TwoTowerModel(
        num_users=model_cfg['num_users'],
        num_items=model_cfg['num_items'],
        num_categories=model_cfg['num_categories'],
        embed_dim=model_cfg['embed_dim'],
        hidden_dims=model_cfg['hidden_dims'],
        hist_len=model_cfg['hist_len'],
        temperature=model_cfg['temperature']
    ).to(device)

    # 把训练好的权重"灌"进空壳模型 —— 现在模型恢复到了训练完的状态
    model.load_state_dict(checkpoint['model_state_dict'])

    # 切换到推理模式：关闭 Dropout 等训练时才有的随机行为
    model.eval()
    print(">>> Model loaded successfully")

    # ============================================================
    # Step 2: 加载数据和特征工程
    #
    # 模型不吃原始数据（如 user_id=10086），它吃的是"特征向量"。
    # 所以这里要做两件事：
    #   1. 把原始 ID 转成模型认识的数字编号（LabelEncoder）
    #   2. 构造用户特征（历史点击序列、活跃度等）
    #      和物品特征（类别、字数、热度等）
    # ============================================================
    print("Step 2: Loading encoders & features...")

    # 读取所有点击日志 + 文章信息
    full_click = get_all_click_df(config.DATA_PATH, offline=config.OFFLINE_MODE)
    click_df = get_positive_click_df(full_click)
    articles_df = load_articles(config.DATA_PATH)

    # build_encoders：
    #   把 user_id=10086 → user_idx=3
    #   把 article_id=205000 → item_idx=7
    #   这样模型才能用 embedding 层去"查表"
    encoders = build_encoders(click_df, articles_df, config.ENCODER_PKL)

    # build_user_features：
    #   对每个用户，提取：历史点击了哪些文章、点了多少次、活跃时间跨度
    #   最终得到一个 DataFrame，每行代表一个用户
    user_features = build_user_features(click_df, encoders, hist_len=config.HIST_LEN)

    # build_item_features：
    #   对每篇文章，提取：类别、字数、被点击次数、创建时间
    #   最终得到一个 DataFrame，每行代表一篇文章
    item_features = build_item_features(articles_df, click_df, encoders)

    # 保存两个编码器，后面要把"模型内部的数字编号"变回"原始 article_id"
    user_le = encoders['user_id']
    item_le = encoders['item_id']

    # ============================================================
    # Step 3: 加载预计算好的物品向量（item embeddings）
    #
    # 类比：每篇文章都有一个"指纹向量"，训练时已经全部算好存在磁盘上。
    # 推理时不需要重新算，直接拿来和用户向量做对比就行。
    # 向量越相似 → 文章越符合用户兴趣。
    # ============================================================
    print("Step 3: Loading pre-computed item embeddings...")

    with open(config.EMBED_PKL, 'rb') as f:
        all_item_vecs = pickle.load(f)

    # 转成 PyTorch 张量，放到 GPU/CPU 上，方便后续做矩阵乘法
    all_item_vecs_tensor = torch.FloatTensor(all_item_vecs).to(device)
    print(f">>> Item embeddings shape: {all_item_vecs.shape}")

    # ============================================================
    # Step 4: 读取测试用户列表
    #
    # testA_click_log.csv 里记录了测试集用户的历史点击行为，
    # 我们要给这些用户推荐未来的文章。
    # ============================================================
    print("Step 4: Reading test users...")

    # 全量用户作为推荐目标
    target_users = click_df['user_id'].unique()
    print(f">>> Number of target users: {len(target_users)}")

    # 全局最热的 50 篇文章，用作"冷启动兜底"
    # （如果模型不认识某个用户，就给他推热门文章）
    item_topk_click = click_df['click_article_id'].value_counts().index[:50].tolist()

    # ============================================================
    # Step 5: 为每个用户生成推荐
    #
    # 核心逻辑：
    #   1. 取出用户的历史行为特征
    #   2. 让模型的"用户塔"输出一个用户向量（代表这个用户的兴趣）
    #   3. 用用户向量和所有文章向量做点积 → 得到对每篇文章的"兴趣分数"
    #   4. 排除用户已经看过的文章
    #   5. 按分数从高到低排序，取 Top-5
    # ============================================================
    print("Step 5: Generating recommendations...")

    recall_item_num = 5  # 每个用户最终推荐的文章数量
    user_recall_items_dict = {}  # 结果容器：{user_id: [(article_id, score), ...]}

    for raw_user_id in tqdm(target_users, desc="Inference"):

        # ----------------------------------------------------------
        # 情况 A：模型从未见过这个用户（冷启动）
        # 处理：直接推热门文章，跳过模型推理
        # ----------------------------------------------------------
        if raw_user_id not in user_le.classes_:
            topk_items = item_topk_click[:recall_item_num]
            user_recall_items_dict[raw_user_id] = [(iid, -999.0) for iid in topk_items]
            continue

        # 把原始 user_id 转成模型内部的连续编号（比如 10086 → 3）
        user_idx = user_le.transform([raw_user_id])[0]

        # 从用户特征表里取出这一行的数据
        user_row = user_features[user_features['user_id'] == raw_user_id]

        # ----------------------------------------------------------
        # 情况 B：用户存在但没有特征数据（比如数据异常）
        # 处理：同样推热门文章兜底
        # ----------------------------------------------------------
        if len(user_row) == 0:
            topk_items = item_topk_click[:recall_item_num]
            user_recall_items_dict[raw_user_id] = [(iid, -999.0) for iid in topk_items]
            continue

        row = user_row.iloc[0]

        # 用户历史点击过的文章索引列表
        hist_items = row['hist_items_trunc']
        hist_len_actual = len(hist_items)  # 实际历史长度

        # ----------------------------------------------------------
        # 把历史序列填充/截断到固定长度 HIST_LEN=50
        # 太短 → 补 0 凑够 50；太长 → 只保留最近 50 条
        # 因为模型的输入形状是固定的，不能动态变
        # ----------------------------------------------------------
        if hist_len_actual < config.HIST_LEN:
            padded = hist_items + [0] * (config.HIST_LEN - hist_len_actual)
        else:
            padded = hist_items[-config.HIST_LEN:]
            hist_len_actual = config.HIST_LEN

        # 组装模型需要的输入批次（虽然只有一个用户，也要包装成 batch 形式）
        user_batch = {
            'user_id': torch.LongTensor([user_idx]).to(device),        # 用户编号
            'hist_items': torch.LongTensor([padded]).to(device),        # 历史点击序列（补0到50）
            'hist_len': torch.LongTensor([hist_len_actual]).to(device), # 实际历史长度
            'click_count': torch.FloatTensor([row['click_count_norm']]).to(device),  # 活跃度
            'time_span': torch.FloatTensor([row['time_span_norm']]).to(device),      # 时间跨度
        }

        # ----------------------------------------------------------
        # 核心推理：把用户输入送入"用户塔"，得到一个 64 维的用户兴趣向量
        # 不需要梯度计算（推理阶段不训练），节省显存
        # ----------------------------------------------------------
        with torch.no_grad():
            user_vec = model.get_user_embedding(user_batch)

        # ----------------------------------------------------------
        # 用用户向量和所有文章向量做矩阵乘法 → 得到用户对每篇文章的兴趣分数
        #
        # 类比：
        #   user_vec     = [1x64]  用户的"兴趣指纹"
        #   all_item_vecs = [Nx64] 所有文章的"内容指纹"
        #   点积结果     = [N]     用户和每篇文章的匹配度
        #
        # 向量越相似（方向越一致），点积越大 → 越推荐
        # ----------------------------------------------------------
        scores = torch.matmul(user_vec, all_item_vecs_tensor.t()).squeeze(0)
        scores = scores.cpu().numpy()  # 转回 NumPy，方便后续排序

        # ----------------------------------------------------------
        # 排除用户已经看过的文章：把历史点击的文章分数设为负无穷
        # 这样排序时它们自然会被排到最后，不会被推荐出来
        # ----------------------------------------------------------
        hist_set = set(hist_items)
        for idx in hist_set:
            if 0 <= idx < len(scores):
                scores[idx] = -1e9  # 负无穷，确保不会入选

        # 按分数从高到低排序，取前 5 个
        top_indices = np.argsort(scores)[::-1][:recall_item_num]

        # 把模型内部的数字编号变回原始 article_id
        recommended_items = []
        for idx in top_indices:
            raw_item_id = item_le.inverse_transform([idx])[0]
            recommended_items.append((raw_item_id, float(scores[idx])))

        # ----------------------------------------------------------
        # 兜底补齐：如果推荐不够 5 篇（比如用户只认识很少的文章），
        # 用热门文章补满，确保提交文件里每个用户都有 5 条推荐
        # ----------------------------------------------------------
        if len(recommended_items) < recall_item_num:
            # 把用户历史文章转回原始 ID，避免重复推荐
            hist_raw_set = set()
            for item_idx in hist_items:
                if item_idx in range(len(item_le.classes_)):
                    hist_raw_set.add(item_le.inverse_transform([item_idx])[0])
            for iid in item_topk_click:
                if iid not in hist_raw_set and iid not in [x[0] for x in recommended_items]:
                    recommended_items.append((iid, -999.0))  # -999 表示是兜底推荐
                if len(recommended_items) == recall_item_num:
                    break

        # 存入结果字典
        user_recall_items_dict[raw_user_id] = recommended_items

    # ============================================================
    # Step 6: 保存提交文件
    #
    # 生成格式：
    #   user_id,article_1,article_2,article_3,article_4,article_5
    #   10086,  205000,  205001,  205002,  205003,  205004
    # ============================================================
    print("Step 6: Saving submission...")
    save_submission(user_recall_items_dict, config.RESULT_PATH, file_name='result_deep.csv')
    print(">>> Inference complete!")


if __name__ == "__main__":
    # 直接运行 python inference.py 时进入推理流程
    inference()
