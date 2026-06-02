import math
import pickle
import os
from tqdm import tqdm
from collections import defaultdict

def itemcf_sim(user_item_time_dict):
    """
    计算物品相似度矩阵 (修正版)
    """
    i2i_sim = {}
    item_cnt = defaultdict(int)

    print(">>> Calculating Item-Item Similarity Matrix (Improved)...")

    for user, item_time_list in tqdm(user_item_time_dict.items()):
        
        # 1. IUF 权重
        # 打压热门物品（Popularity Bias）、提升推荐多样性和长尾物品曝光率
        iuf_weight = 1.0 / math.log(1 + len(item_time_list))
        
        for loc1, (i, i_click_time) in enumerate(item_time_list):
            item_cnt[i] += 1
            i2i_sim.setdefault(i, {})
            
            for loc2, (j, j_click_time) in enumerate(item_time_list):
                if i == j:
                    continue
                
                # 2. 位置权重
                loc_alpha = 1.0 if loc2 > loc1 else 0.7
                
                # 【!!! 修正点 !!!】这里必须是 ** (次方)，绝对不能是 * (乘法)
                loc_weight = loc_alpha * (0.9 ** (abs(loc1 - loc2) - 1))
                
                i2i_sim[i].setdefault(j, 0)
                i2i_sim[i][j] += iuf_weight * loc_weight
            
    # 3. 归一化
    print(">>> Normalizing...")
    for i, related_items in i2i_sim.items():
        for j, wij in related_items.items():
            i2i_sim[i][j] = wij / math.sqrt(item_cnt[i] * item_cnt[j])
            
    return i2i_sim

def item_based_recommend(user_id,user_item_time_dict,i2i_sim,sim_item_topk,recall_item_num,item_topk_click):
    """
    item_base_recommend 的 Docstring
    基于物品的协同过滤
    :param sim_item_topk: 选取最相似的k个物品进行计算
    :param recall_item_num: 最终召回多少个物品
    :param item_topk_click: 全局热门物品（用于补全）
    """
    #1.获取用户历史交互过的物品
    # 这里的 user_item_time_dict 是我们在 data_loader 里搞出来的那个字典
    user_hist_items = user_item_time_dict.get(user_id,[])
    user_hist_set = {item for item,_ in user_hist_items}
    item_rank = {}

    # 2. 遍历用户看过的每个物品 i
    # 【优化】考虑用户最近的行为权重更高
    # 我们用 list 的索引来代表时间远近，loc 越大代表越近
    hist_len = len(user_hist_items)
    for loc,(i,click_time) in enumerate(user_hist_items):
        # 【新增】近期行为权重 (Time Decay)
        # 越接近当前(loc 越大)，权重越大。
        # 0.9 ** (hist_len - 1 - loc)
        # 例如：总长5。第4个(最新): 0.9^0=1。第0个(最旧): 0.9^4=0.65
        time_decay = 0.9 ** (hist_len - loc - 1)
        for j,wij in sorted(i2i_sim.get(i,{}).items(),key = lambda x : x[1],reverse= True)[:sim_item_topk]:
            if j in user_hist_set:
                continue
            #计算推荐分数
            item_rank.setdefault(j,0)
            item_rank[j] += wij * time_decay
        
    #3.热门补全
    if len(item_rank) < recall_item_num:
        for item in item_topk_click:
            if item not in item_rank and item not in user_hist_set:
                item_rank[item] = -999
                if len(item_rank) == recall_item_num:
                    break
    # 4. 排序并返回前 recall_item_num 个
    # 返回格式: [(item_id, score), (item_id, score)...]
    return sorted(item_rank.items(),key= lambda x : x[1],reverse= True)[:recall_item_num]