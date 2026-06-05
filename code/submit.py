import pandas as pd
import os
import config

def save_submission(user_recall_items_dict, submit_path, file_name='result.csv'):
    """
    保存推荐结果为 CSV.

    格式:
      user_id, item_1, item_2, ..., item_K

    参数:
      user_recall_items_dict: {user_id: [(item_id, score), ...]}
      submit_path: 输出目录
      file_name: 输出文件名
    """
    print(">>> Saving recommendations...")

    data = []
    for user_id, items in user_recall_items_dict.items():
        # items 是 [(item_id, score), ...]，只需要 item_id
        row = [user_id] + [item[0] for item in items]
        data.append(row)

    # 动态列名: 有几个推荐就几列
    n_recs = len(data[0]) - 1 if data else 0
    columns = ['user_id'] + [f'item_{i+1}' for i in range(n_recs)]
    df = pd.DataFrame(data, columns=columns)

    os.makedirs(submit_path, exist_ok=True)
    final_path = os.path.join(submit_path, file_name)
    df.to_csv(final_path, index=False)
    print(f"[OK] Saved {len(data):,} users × {n_recs} recommendations to {final_path}")