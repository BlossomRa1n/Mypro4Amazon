import pandas as pd
import os
import config

def save_submission(user_recall_items_dict, submit_path, file_name='result.csv'):
    """
    将推荐结果字典转换为提交格式
    :param user_recall_items_dict: {user_id: [(item1, score1), (item2, score2)...]}
    """
    print(">>> Generating submission file...")
    
    # 1. 字典转列表
    data = []
    for user_id, items in user_recall_items_dict.items():
        # items 是 [(item_id, score), ...]
        # 我们只需要 item_id，不需要分数
        row = [user_id] + [item[0] for item in items]
        data.append(row)
    
    # 2. 转换为 DataFrame
    # 列名: user_id, article_1, article_2, ..., article_5
    columns = ['user_id'] + [f'article_{i+1}' for i in range(5)]
    df = pd.DataFrame(data, columns=columns)
    
    # 3. 保存
    if not os.path.exists(submit_path):
        os.makedirs(submit_path)
        
    final_path = os.path.join(submit_path, file_name)
    df.to_csv(final_path, index=False)
    print(f"[OK] Submission saved to: {final_path}")