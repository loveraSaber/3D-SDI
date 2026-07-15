# import os
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# from torch.nn.utils.rnn import pad_sequence

# class SleepWindowDataset(Dataset):
#     def __init__(self, file_list, window_size=15, stride=1, cache_size=10):
#         self.file_list = file_list
#         self.window_size = window_size
#         self.stride = stride
#         self.cache_size = cache_size
#         self.cache = {}

#     def _load_file(self, file_path):
#         if file_path in self.cache:
#             return self.cache[file_path]

#         data = np.load(file_path)
#         features = data["features"]  # (T, F)
#         labels = data["labels"]      # (T,) 0..4 (W,N1,N2,N3,REM)

#         w = self.window_size
#         half = w // 2

#         windows = []
#         # i 为窗口中心点
#         for i in range(half, len(features) - half, self.stride):
#             win = features[i - half: i + half + 1]        # (w, F)
#             # 不在这里 reshape，模型里会把 (w,F) 展平成 (w*F)
#             label = int(labels[i])                        # 5 分类深度标签
#             nrem_label = 0 if label < 4 else 1            # 二分类：非REM=0, REM=1
#             windows.append((
#                 torch.tensor(win, dtype=torch.float32),        # (w, F)
#                 torch.tensor(label, dtype=torch.long),         # ()
#                 torch.tensor(nrem_label, dtype=torch.long),    # ()
#             ))

#         if len(self.cache) < self.cache_size:
#             self.cache[file_path] = windows

#         return windows

#     def __len__(self):
#         return len(self.file_list)

#     def __getitem__(self, idx):
#         file_path = self.file_list[idx]
#         windows = self._load_file(file_path)
#         feats = [w[0] for w in windows]           # [(w, F)] * L
#         depth_labels = [w[1] for w in windows]    # [()] * L
#         nrem_labels = [w[2] for w in windows]     # [()] * L
#         return torch.stack(feats), torch.stack(depth_labels), torch.stack(nrem_labels)

# def collate_fn_window(batch):
#     """
#     batch: list of tuples from __getitem__ of one file
#            features: (L, w, F), labels: (L,), nrem: (L,)
#     返回：
#       padded_features: (B, L_max, w, F)
#       padded_labels  : (B, L_max) with -100 as pad
#       padded_nrem    : (B, L_max) with -100 as pad
#     """
#     features, labels, nrem_labels = zip(*batch)
#     padded_features = pad_sequence(features, batch_first=True)  # (B, L_max, w, F)
#     padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)  # (B, L_max)
#     padded_nrem = pad_sequence(nrem_labels, batch_first=True, padding_value=-100)  # (B, L_max)
#     return padded_features, padded_labels, padded_nrem


'''不添加深度'''
# import os
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# from torch.nn.utils.rnn import pad_sequence

# class SleepWindowDataset(Dataset):
#     def __init__(self, file_list, window_size=15, stride=1, cache_size=10):
#         self.file_list = file_list
#         self.window_size = window_size
#         self.stride = stride
#         self.cache_size = cache_size
#         self.cache = {}

#     def _load_file(self, file_path):
#         if file_path in self.cache:
#             return self.cache[file_path]

#         data = np.load(file_path)
#         features = data["features"]  # (T, F)
#         labels = data["labels"]      # (T,) 0..4 (Wake=0, N1=1, N2=2, N3=3, REM=4)

#         w = self.window_size
#         half = w // 2

#         windows = []
#         # i 为窗口中心点
#         for i in range(half, len(features) - half, self.stride):
#             win = features[i - half: i + half + 1]        # (w, F)
#             # 不在这里 reshape，模型里会把 (w,F) 展平成 (w*F)
#             original_label = int(labels[i])                # 原始5分类标签（0-4）
            

#             # 1. Wake 标签：Wake=1，其他=0
#             wake_label = 1 if original_label == 1 else 0

#             # 2. REM 标签：REM=1，其他=0
#             rem_label = 1 if original_label == 4 else 0
            
#             windows.append((
#                 torch.tensor(win, dtype=torch.float32),    # (w, F)
#                 torch.tensor(wake_label, dtype=torch.long),# () 二分类Wake标签
#                 torch.tensor(rem_label, dtype=torch.long),# () 二分类NREM标签
#             ))

#         if len(self.cache) < self.cache_size:
#             self.cache[file_path] = windows

#         return windows

#     def __len__(self):
#         return len(self.file_list)

#     def __getitem__(self, idx):
#         file_path = self.file_list[idx]
#         windows = self._load_file(file_path)
#         feats = [w[0] for w in windows]           # [(w, F)] * L
#         wake_labels = [w[1] for w in windows]     # [()] * L （Wake标签）
#         nrem_labels = [w[2] for w in windows]     # [()] * L （NREM标签）
#         return torch.stack(feats), torch.stack(wake_labels), torch.stack(nrem_labels)


# def collate_fn_window(batch):
#     """
#     batch: list of tuples from __getitem__ of one file
#            features: (L, w, F), wake_labels: (L,), nrem_labels: (L,)
#     返回：
#       padded_features: (B, L_max, w, F)
#       padded_wake: (B, L_max) with -100 as pad （Wake标签）
#       padded_nrem: (B, L_max) with -100 as pad （NREM标签）
#     """
#     features, wake_labels, nrem_labels = zip(*batch)
#     padded_features = pad_sequence(features, batch_first=True)  # (B, L_max, w, F)
#     # 修复：移除 dtype 参数，用 .long() 强制转换为长整数类型
#     padded_wake = pad_sequence(wake_labels, batch_first=True, padding_value=-100).long()  # (B, L_max)
#     padded_nrem = pad_sequence(nrem_labels, batch_first=True, padding_value=-100).long()  # (B, L_max)
#     return padded_features, padded_wake, padded_nrem



'''添加深度'''
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple

class SleepWindowDataset(Dataset):
    def __init__(self, file_list, window_size=15, stride=1, cache_size=10):
        self.file_list = file_list
        self.window_size = window_size
        self.stride = stride
        self.cache_size = cache_size
        self.cache = {}

    def _load_file(self, file_path):
        if file_path in self.cache:
            return self.cache[file_path]

        try:
            with np.load(file_path, allow_pickle=True) as data_npz:
                features = data_npz["features"].copy()  # (T, F)
                labels = data_npz["labels"].copy()      # (T,) 
        except Exception as e:
            print(f"🚨 错误：无法加载文件 {file_path}. 详细信息: {e}")
            raise IOError(f"Failed to load data file: {file_path}") from e

        w = self.window_size
        half = w // 2

        windows = []
        # 窗口切分和标签派生
        for i in range(half, len(features) - half, self.stride):
            win = features[i - half: i + half + 1]  # (w, F)
            original_label = int(labels[i])         # 原始5分类标签（0-4）
            
            # --- 标签派生 ---
            # 1. Wake 标签：
            # 这里我们沿用您上一次提交的代码逻辑（假设它正确）。
            wake_label = 1 if original_label == 0 else 0

            # 2. REM 标签：REM=1，其他=0
            rem_label = 1 if original_label == 4 else 0
            
            # 3. 按照原标签进行深度排序
            depth_label = original_label
            
            windows.append((
                torch.tensor(win, dtype=torch.float32), 
                torch.tensor(wake_label, dtype=torch.long), 
                torch.tensor(rem_label, dtype=torch.long),
                torch.tensor(depth_label, dtype=torch.long), # 新增：深度排序标签 (0/1)
            ))

        if len(self.cache) < self.cache_size:
            self.cache[file_path] = windows

        return windows

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        windows = self._load_file(file_path)
        
        feats = [w[0] for w in windows] 
        wake_labels = [w[1] for w in windows]  
        rem_labels = [w[2] for w in windows]  
        depth_labels = [w[3] for w in windows] # 接收新增的深度标签
        
        # 返回四个堆叠的张量 (L, w, F), (L,), (L,), (L,)
        return torch.stack(feats), torch.stack(wake_labels), torch.stack(rem_labels), torch.stack(depth_labels)

# --- Collate Function ---
def collate_fn_window(batch):
    """
    batch: list of tuples from __getitem__ of one file
    返回：
      padded_features: (B, L_max, w, F)
      padded_wake: (B, L_max) Wake/Nwake 分类标签
      padded_rem: (B, L_max) REM/Non-REM 分类标签
      padded_depth: (B, L_max) 深度排序标签 (新增)
    """
    # 接收四个元素
    features, wake_labels, rem_labels, depth_labels = zip(*batch) 
    
    padded_features = pad_sequence(features, batch_first=True) 

    # 标签填充
    padding_value = -100 # 用于损失函数忽略计算的填充值
    padded_wake = pad_sequence(wake_labels, batch_first=True, padding_value=padding_value).long()
    padded_rem = pad_sequence(rem_labels, batch_first=True, padding_value=padding_value).long() 
    
    # 新增深度标签的填充
    padded_depth = pad_sequence(depth_labels, batch_first=True, padding_value=padding_value).long() 
    
    # 返回四个填充后的张量
    return padded_features, padded_wake, padded_rem, padded_depth