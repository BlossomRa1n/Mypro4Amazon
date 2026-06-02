import pandas as pd
import pickle
import os
import time
from tqdm import tqdm
# 导入我们自己写的模块
import config
from data_loader import get_all_click_df, get_positive_click_df, get_user_item_time, get_item_topk_click
from itemcf import itemcf_sim, item_based_recommend
from submit import save_submission

# --- 全局配置 ---
# True: 只跑少量数据用来排查 Bug (Debug模式)
# False: 跑全量数据，生成最终提交文件
OFFLINE = config.OFFLINE_MODE

def main():
    start_time = time.time()
    
    # ---------------------------------------------------------
    # 1. 读取数据
    # ---------------------------------------------------------
    print("Step 1: Loading data...")
    # 这里会调用 config.DATA_PATH
    click_df = get_positive_click_df(get_all_click_df(config.DATA_PATH, offline=OFFLINE))
    
    # ---------------------------------------------------------
    # 2. 构建字典 (User-Item-Time) & 热门备胎
    # ---------------------------------------------------------
    print("Step 2: Constructing dictionaries...")
    user_item_time_dict = get_user_item_time(click_df)
    # 获取最热的 50 个文章，万一推荐不够 5 个，用这些补
    item_topk_click = get_item_topk_click(click_df, k=50)
    
    # ---------------------------------------------------------
    # 3. 计算物品相似度 (最耗时的一步)
    # ---------------------------------------------------------
    print("Step 3: Calculating Item Similarity...")
    # 检查有没有存好的 pickle 文件，有就直接读，没有就算
    if os.path.exists(config.ITEMCF_SIM_PKL) and not OFFLINE:
        print(f"Loading similarity matrix from {config.ITEMCF_SIM_PKL}...")
        i2i_sim = pickle.load(open(config.ITEMCF_SIM_PKL, 'rb'))
    else:
        print("No cache found, calculating from scratch...")
        i2i_sim = itemcf_sim(user_item_time_dict)
        # 保存下来，下次不用重算了
        if not os.path.exists(config.MODEL_PATH):
            os.makedirs(config.MODEL_PATH)
        #pickle模块是将i2i_sim对象序列化并保存到文件中的标准操作
        pickle.dump(i2i_sim, open(config.ITEMCF_SIM_PKL, 'wb'))
        
    # ---------------------------------------------------------
    # 4. 为测试集用户生成推荐
    # ---------------------------------------------------------
    print("Step 4: Generating recommendations...")
    
    # Amazon 数据无独立测试集，用全量用户
    target_users = click_df['user_id'].unique()
    
    # 结果容器
    user_recall_items_dict = {}
    
    # 【修改点】：这里加上 tqdm(target_users)，给循环加个进度条
    for user in tqdm(target_users, desc="Recommend"):
        topk_items = item_based_recommend(
            user, 
            user_item_time_dict, 
            i2i_sim, 
            sim_item_topk=10, #itemcf里截取该user的某个历史物品的最相似的10个物品进行计算
            recall_item_num=5, 
            item_topk_click=item_topk_click
        )
        user_recall_items_dict[user] = topk_items
        
    # ---------------------------------------------------------
    # 5. 生成提交文件
    # ---------------------------------------------------------
    print("Step 5: Saving result...")
    save_submission(user_recall_items_dict, config.RESULT_PATH)
    
    print(f"All done! Total time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
