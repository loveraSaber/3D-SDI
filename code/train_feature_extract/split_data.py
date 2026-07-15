'''过滤mros  visit2数据'''
import os
import numpy as np
from sklearn.model_selection import train_test_split

def get_npz_files(directory):
    npz_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.npz'):
                npz_files.append(os.path.join(root, file))
    return npz_files

def save_data_splits(data_directories, base_path, seed, split_data, max_samples_per_dataset=None):
    train_paths = []
    val_paths = []
    test_paths = []

    for directory in data_directories:
        npz_file_paths = get_npz_files(directory)
        
        # ==========================================
        # 修改开始：针对 mros 数据集的特殊过滤逻辑
        # ==========================================
        if "mros" in directory:
            # 过滤：只保留文件名中包含 'visit1' 的文件
            # os.path.basename(f) 获取文件名，避免路径中其他部分干扰
            filtered_paths = [f for f in npz_file_paths if "visit1" in os.path.basename(f)]
            
            print(f"正在处理 mros: 原始文件数 {len(npz_file_paths)}, 过滤后(仅保留visit1)文件数 {len(filtered_paths)}")
            npz_file_paths = filtered_paths
        # ==========================================
        # 修改结束
        # ==========================================

        if max_samples_per_dataset:
            npz_file_paths = npz_file_paths[:max_samples_per_dataset]
        
        if split_data:
            # 按 6:2:2 划分训练、验证和测试集
            # 注意：如果文件数太少，split可能会报错，建议加个简单判断
            if len(npz_file_paths) > 0:
                train, test = train_test_split(npz_file_paths, test_size=0.4, random_state=seed)
                val, test = train_test_split(test, test_size=0.5, random_state=seed)
                
                train_paths.extend(train)
                val_paths.extend(val)
                test_paths.extend(test)
            else:
                print(f"Warning: Directory {directory} has no files after filtering.")
        else:
            train_paths.extend(npz_file_paths)

    if split_data:
        train_path = os.path.join(base_path, 'train_paths.txt')
        val_path = os.path.join(base_path, 'val_paths.txt')
        test_path = os.path.join(base_path, 'test_paths.txt')

        np.savetxt(train_path, train_paths, fmt='%s')
        np.savetxt(val_path, val_paths, fmt='%s')
        np.savetxt(test_path, test_paths, fmt='%s')

        print(f"Number of training samples: {len(train_paths)}")
        print(f"Number of validation samples: {len(val_paths)}")
        print(f"Number of testing samples: {len(test_paths)}")
    else:
        all_data_path = os.path.join(base_path, 'cfs_all_data_paths.txt')
        np.savetxt(all_data_path, train_paths, fmt='%s')
        print(f"Number of total samples: {len(train_paths)}")

    return train_paths, val_paths, test_paths

def main(data_directories, base_path, seed, split_data, max_samples_per_dataset=None):
    save_data_splits(data_directories, base_path, seed, split_data, max_samples_per_dataset)

if __name__ == '__main__':
    data_directories = [
        # "/DATA/disk2/xdl/processed_data/ss/step2_data/all_final_model",
        "/data/0shared/zhaoqingshuo/NSRR_data/all_data_features/cfs/",
        "/data/0shared/zhaoqingshuo/NSRR_data/all_data_features/mesa/",
        "/data/0shared/zhaoqingshuo/NSRR_data/all_data_features/mros/"
    ]

    base_path = "/home/zhaoqingshuo/SDI/ECG_sleepstage/transformer_W_NW_REM_NREM/data_split/inner"
    # base_path = "./test_split" # 本地测试路径
    os.makedirs(base_path, exist_ok=True)

    split_data = True  # 设置为 False 时，不进行数据集划分，而是将整个数据集提取出来
    max_samples_per_dataset = 10000  # 设置每个数据集的最大样本数

    main(data_directories, base_path, seed=42, split_data=split_data, max_samples_per_dataset=max_samples_per_dataset)



'''shhs全数据'''
# import os
# import glob

# def find_and_save_files(directory, prefix, output_filename="found_files.txt"):
#     """
#     查找指定目录下所有以特定前缀开头的文件，并将其完整路径保存到TXT文件中。

#     Args:
#         directory (str): 要搜索的目录路径。
#         prefix (str): 文件名的前缀（例如 'shhs1'）。
#         output_filename (str): 输出TXT文件的名称。
#     """
    
#     # 检查目录是否存在
#     if not os.path.isdir(directory):
#         print(f"错误：目录 '{directory}' 不存在或不是一个有效的目录。")
#         return

#     # 使用 glob 模式匹配所有以指定前缀开头的文件
#     # **/* 表示递归搜索子目录
#     search_pattern = os.path.join(directory, f"**/{prefix}*")
    
#     # 使用 glob.glob 查找匹配的文件。recursive=True 启用递归搜索。
#     # 注意：glob 库返回的路径已经是完整的路径
#     # 如果您只需要搜索当前目录，可以简化为：search_pattern = os.path.join(directory, f"{prefix}*")
#     file_paths = glob.glob(search_pattern, recursive=True)

#     # 将文件路径写入输出文件
#     try:
#         with open(output_filename, 'w') as f:
#             for path in file_paths:
#                 # 写入完整路径，并在末尾添加换行符
#                 f.write(path + '\n')
        
#         print(f"✅ 成功找到 {len(file_paths)} 个文件，并已将路径保存到 '{output_filename}' 中。")
#         print(f"👉 输出文件位于：{os.path.abspath(output_filename)}")

#     except Exception as e:
#         print(f"❌ 写入文件时发生错误: {e}")

# # --- 请根据您的实际情况修改以下参数 ---

# # 1. **指定要搜索的根目录** (请替换为您的实际路径)
# # 示例路径：/data/0shared/zhaoqingshuo/NSRR_data/all_data_features/shhs/
# search_directory = "/data/0shared/zhaoqingshuo/NSRR_data/all_data_features/shhs/"

# # 2. **指定文件名前缀**
# file_prefix = "shhs1" 

# # 3. **指定输出的TXT文件名**
# output_file = "/home/zhaoqingshuo/SDI/ECG_sleepstage/transformer_W_NW_REM_NREM/data_split/external/shhs1_file_paths.txt"

# # --- 运行函数 ---
# find_and_save_files(
#     directory=search_directory,
#     prefix=file_prefix,
#     output_filename=output_file
# )