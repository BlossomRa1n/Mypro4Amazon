import pandas as pd
import numpy as np
import time
import random
import torch


def set_seed(seed=42):
    """
    固定随机种子, 保证训练可复现 (A/B 对比基线的三前提之一)。
    seed 的唯一来源在此函数默认值, 召回/精排脚本统一调用 set_seed()。
    cudnn.deterministic=True 牺牲少量速度换取 bit 级确定性;
    若只做架构对比 (非 bit 级复现), 可改为 False 提速。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def reduce_mem(df):
    """
    【直接复制即可】
    通过降低数值类型的精度（如 int64 -> int32, float64 -> float32）来减少内存占用。
    这对天池这种大数据量的比赛至关重要，否则你的 4090 显存没爆，内存先爆了。
    """
    starttime = time.time()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if pd.isnull(c_min) or pd.isnull(c_max):
                continue
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    print('-- Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction),time spend:{:2.2f} min'.format(end_mem,100*(start_mem-end_mem)/start_mem,(time.time()-starttime)/60))
    return df