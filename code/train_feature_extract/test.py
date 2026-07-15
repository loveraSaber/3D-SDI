# import os
# import numpy as np
# import torch
# from tqdm import tqdm
# import pandas as pd
# import matplotlib.pyplot as plt
# import itertools
# from sklearn.metrics import (
#     accuracy_score, f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix
# )
# from torch.utils.data import DataLoader
# from dataset_sliding import SleepWindowDataset, collate_fn_window
# from model_transformer_window import TransformerSleepModel

# CONFIG = {
#     "test_list": "/home/xiedonglin/code/ss/transformer/data_paths/cfs_ex_final_model_42/all_data_paths.txt",  # 修改为您的测试集路径
#     "model_path": "/DATA/disk2/xdl/model/ss/step2_model/all_42_final_model_best.pth",  # 修改为您的模型路径
#     "save_dir": "/home/xiedonglin/code/ss/transformer/output/all_cfs_external",  # 结果保存目录
#     "batch_size": 16,
#     "window_size": 15,
#     "stride": 1,
#     "num_classes": 5,
    
#     # 模型结构参数(应与训练时一致)
#     "input_dim": 15 * 1152,  # window_size * feature_dim
#     "hidden_dim": 512,
#     "n_heads": 8,
#     "num_layers": 3,
#     "dropout": 0.1,
# }

# os.makedirs(CONFIG["save_dir"], exist_ok=True)

# def load_paths_from_txt(txt_file):
#     with open(txt_file, 'r') as f:
#         return f.read().splitlines()

# def plot_confusion_matrix(cm, classes, save_path):
#     plt.figure(figsize=(10, 8))
    
#     # 计算每个单元格的百分比
#     cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
#     # 显示混淆矩阵的百分比
#     plt.imshow(cm_percent, interpolation='nearest', cmap=plt.cm.Blues)
#     plt.colorbar()
    
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45, fontsize=16)
#     plt.yticks(tick_marks, classes, fontsize=16)
#     plt.tick_params(labeltop=True, labelbottom=False, top=True, bottom=False)
    
#     # 计算阈值来决定文本颜色
#     thresh = cm_percent.max() / 2.
    
#     # 在每个格子内显示数字
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         pct = f"{cm_percent[i, j]:.1%}" if cm_percent[i, j] != 0 else ''
#         plt.text(j, i, f"{cm[i, j]}\n({pct})",
#                  horizontalalignment="center", verticalalignment="center",
#                  color="white" if cm_percent[i, j] > thresh else "black", fontsize=16)
    
#     plt.tight_layout()
#     plt.savefig(save_path, bbox_inches='tight', dpi=800)
#     plt.close()

# def bootstrap_ci(metric_fn, y_true, y_pred, n=1000, alpha=0.95):
#     stats = []
#     for _ in range(n):
#         indices = np.random.choice(np.arange(len(y_true)), size=len(y_true), replace=True)
#         stats.append(metric_fn(y_true[indices], y_pred[indices]))
#     stats = np.sort(stats)
#     lower = np.percentile(stats, (1 - alpha) / 2 * 100)
#     upper = np.percentile(stats, (1 + alpha) / 2 * 100)
#     return np.mean(stats), lower, upper

# def evaluate_with_ci(y_true, y_pred, y_prob, labels, name):
#     results = {}

#     acc_fn = lambda y_t, y_p: accuracy_score(y_t, y_p)
#     f1_fn = lambda y_t, y_p: f1_score(y_t, y_p, average='weighted')
#     kappa_fn = lambda y_t, y_p: cohen_kappa_score(y_t, y_p)
#     auc_fn = lambda y_t, y_p: roc_auc_score(np.eye(len(labels))[y_t], y_p, average='macro', multi_class='ovr')

#     acc, l1, u1 = bootstrap_ci(acc_fn, y_true, y_pred)
#     f1, l2, u2 = bootstrap_ci(f1_fn, y_true, y_pred)
#     kappa, l3, u3 = bootstrap_ci(kappa_fn, y_true, y_pred)
#     try:
#         auc, l4, u4 = bootstrap_ci(auc_fn, y_true, y_prob)
#     except:
#         auc, l4, u4 = float('nan'), float('nan'), float('nan')

#     results['Accuracy'] = f"{acc:.3f} [{l1:.3f}, {u1:.3f}]"
#     results['F1 Score'] = f"{f1:.3f} [{l2:.3f}, {u2:.3f}]"
#     results['Kappa'] = f"{kappa:.3f} [{l3:.3f}, {u3:.3f}]"
#     results['Macro AUC'] = f"{auc:.3f} [{l4:.3f}, {u4:.3f}]"

#     cm = confusion_matrix(y_true, y_pred)
#     plot_confusion_matrix(cm, classes=labels, save_path=os.path.join(CONFIG["save_dir"], f"cm_{name}.png"))

#     return results

# def map_labels(y, mapping):
#     new_y = np.copy(y)
#     for old, new in mapping.items():
#         new_y[y == old] = new
#     return new_y

# def predict_all(model, dataloader, device):
#     model.eval()
#     y_true_all, y_pred_all, y_prob_all = [], [], []
#     with torch.no_grad():
#         for x, y in tqdm(dataloader, desc="Evaluating"):
#             x, y = x.to(device), y.to(device)
#             logits = model(x)
#             probs = torch.softmax(logits, dim=-1)
#             preds = torch.argmax(probs, dim=-1)
            
#             # 处理填充值(-100)
#             mask = y != -100
#             for i in range(x.size(0)):
#                 valid_indices = mask[i]
#                 y_true_all.append(y[i, valid_indices].cpu().numpy())
#                 y_pred_all.append(preds[i, valid_indices].cpu().numpy())
#                 y_prob_all.append(probs[i, valid_indices].cpu().numpy())
    
#     return np.concatenate(y_true_all), np.concatenate(y_pred_all), np.concatenate(y_prob_all)

# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # 加载测试集
#     test_files = load_paths_from_txt(CONFIG["test_list"])
#     test_set = SleepWindowDataset(test_files, CONFIG["window_size"], CONFIG["stride"])
#     test_loader = DataLoader(test_set, batch_size=CONFIG["batch_size"], 
#                             shuffle=False, collate_fn=collate_fn_window, num_workers=4)

#     # 初始化并加载模型
#     model = TransformerSleepModel(
#         input_dim=CONFIG["input_dim"],
#         hidden_dim=CONFIG["hidden_dim"],
#         n_heads=CONFIG["n_heads"],
#         num_layers=CONFIG["num_layers"],
#         num_classes=CONFIG["num_classes"],
#         dropout=CONFIG["dropout"]
#     ).to(device)
    
#     model.load_state_dict(torch.load(CONFIG["model_path"], map_location=device))
#     print(f"✅ Loaded model from {CONFIG['model_path']}")

#     # 预测
#     y_true, y_pred, y_prob = predict_all(model, test_loader, device)

#     results = {}

#     # 5-class 评估
#     results["5-class"] = evaluate_with_ci(
#         y_true, y_pred, y_prob, 
#         labels=["W", "N1", "N2", "N3", "REM"], 
#         name="5class"
#     )

#     # 2-class: W vs Sleep
#     map_2 = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1}
#     results["2-class"] = evaluate_with_ci(
#         map_labels(y_true, map_2),
#         map_labels(y_pred, map_2),
#         np.stack([y_prob[:, 0], y_prob[:, 1:].sum(axis=1)], axis=1),
#         labels=["W", "Sleep"],
#         name="2class"
#     )

#     # 3-class: W vs NREM vs REM
#     map_3 = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2}
#     results["3-class"] = evaluate_with_ci(
#         map_labels(y_true, map_3),
#         map_labels(y_pred, map_3),
#         np.stack([y_prob[:, 0], y_prob[:, 1:4].sum(axis=1), y_prob[:, 4]], axis=1),
#         labels=["W", "NREM", "REM"],
#         name="3class"
#     )

#     # 4-class: W vs Light vs Deep vs REM
#     map_4 = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3}
#     results["4-class"] = evaluate_with_ci(
#         map_labels(y_true, map_4),
#         map_labels(y_pred, map_4),
#         np.stack([y_prob[:, 0], y_prob[:, 1:3].sum(axis=1), y_prob[:, 3], y_prob[:, 4]], axis=1),
#         labels=["W", "Light", "Deep", "REM"],
#         name="4class"
#     )

#     # 打印结果表格
#     print("\n📊 Evaluation Summary (with 95% CI):")
#     headers = ["Accuracy", "F1 Score", "Kappa", "Macro AUC"]
#     print(f"{'Category':<10}  " + "  ".join(f"{h:<20}" for h in headers))
#     print("-" * 90)
#     for cat, metrics in results.items():
#         print(f"{cat:<10}  " + "  ".join(f"{metrics[h]:<20}" for h in headers))
    
#     # 保存结果到Excel
#     df = pd.DataFrame.from_dict(results, orient='index')
#     df = df[headers]  # 确保列顺序
#     excel_path = os.path.join(CONFIG["save_dir"], "evaluation_summary.xlsx")
#     df.to_excel(excel_path)
#     print(f"\n✅ Saved evaluation summary to {excel_path}")

# if __name__ == '__main__':
#     main()




# "增加纵向染色"
# import os
# import numpy as np
# import torch
# from tqdm import tqdm
# import pandas as pd
# import matplotlib.pyplot as plt
# import itertools
# from sklearn.metrics import (
#     accuracy_score, f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix
# )
# from torch.utils.data import DataLoader
# from dataset_sliding import SleepWindowDataset, collate_fn_window
# from model_transformer_window import TransformerSleepModel

# CONFIG = {
#     "test_list": "/data/0shared/zhaoqingshuo/sleep_data_v5/data_split/test_paths.txt",
#     "model_path": "/home/zhaoqingshuo/SDI/ECG_sleepstage/step2_model/cfs_42_bestmodel_best.pth",
#     "save_dir": "/data/0shared/zhaoqingshuo/sleep_data_v5/cfs_test/",
#     "batch_size": 16,
#     "window_size": 15,
#     "stride": 1,
#     "num_classes": 5,
#     "input_dim": 15 * 1152,
#     "hidden_dim": 512,
#     "n_heads": 8,
#     "num_layers": 3,
#     "dropout": 0.1,
# }

# os.makedirs(CONFIG["save_dir"], exist_ok=True)

# def load_paths_from_txt(txt_file):
#     with open(txt_file, 'r') as f:
#         return f.read().splitlines()

# def plot_confusion_matrix(cm, classes, save_path, normalize_axis=1):
#     """
#     绘制混淆矩阵，支持横向（row）和纵向（column）归一化染色，明确轴含义
#     :param normalize_axis: 1=横向归一化（行：真实值），0=纵向归一化（列：预测值）
#     """
#     plt.figure(figsize=(10, 8))
    
#     # 根据axis选择归一化方向
#     if normalize_axis == 1:
#         cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         norm_label = "by row"
#     else:
#         cm_normalized = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
#         norm_label = "by column"
    
#     plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
#     plt.colorbar(label=f'Normalized {norm_label} (%)')
    
#     tick_marks = np.arange(len(classes))
#     # 移除labelbottom和labeltop参数，改用tick_params控制
#     plt.xticks(tick_marks, classes, rotation=45, fontsize=16)
#     plt.yticks(tick_marks, classes, fontsize=16)
    
#     # 明确轴含义（真实值/预测值）
#     plt.ylabel('True Label', fontsize=18)
#     plt.xlabel('Predicted Label', fontsize=18)
#     # 控制刻度标签显示位置
#     plt.tick_params(axis='x', labelbottom=True, labeltop=False)
    
#     thresh = cm_normalized.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         pct = f"{cm_normalized[i, j]:.1%}" if cm_normalized[i, j] != 0 else ''
#         plt.text(j, i, f"{cm[i, j]}\n({pct})",
#                  horizontalalignment="center", verticalalignment="center",
#                  color="white" if cm_normalized[i, j] > thresh else "black", fontsize=16)
    
#     plt.tight_layout()
#     plt.savefig(save_path, bbox_inches='tight', dpi=800)
#     plt.close()

# def bootstrap_ci(metric_fn, y_true, y_pred, n=1000, alpha=0.95):
#     stats = []
#     for _ in range(n):
#         indices = np.random.choice(np.arange(len(y_true)), size=len(y_true), replace=True)
#         stats.append(metric_fn(y_true[indices], y_pred[indices]))
#     stats = np.sort(stats)
#     lower = np.percentile(stats, (1 - alpha) / 2 * 100)
#     upper = np.percentile(stats, (1 + alpha) / 2 * 100)
#     return np.mean(stats), lower, upper

# def evaluate_with_ci(y_true, y_pred, y_prob, labels, name):
#     results = {}

#     acc_fn = lambda y_t, y_p: accuracy_score(y_t, y_p)
#     f1_fn = lambda y_t, y_p: f1_score(y_t, y_p, average='weighted')
#     kappa_fn = lambda y_t, y_p: cohen_kappa_score(y_t, y_p)
#     auc_fn = lambda y_t, y_p: roc_auc_score(np.eye(len(labels))[y_t], y_p, average='macro', multi_class='ovr')

#     acc, l1, u1 = bootstrap_ci(acc_fn, y_true, y_pred)
#     f1, l2, u2 = bootstrap_ci(f1_fn, y_true, y_pred)
#     kappa, l3, u3 = bootstrap_ci(kappa_fn, y_true, y_pred)
#     try:
#         auc, l4, u4 = bootstrap_ci(auc_fn, y_true, y_prob)
#     except:
#         auc, l4, u4 = float('nan'), float('nan'), float('nan')

#     results['Accuracy'] = f"{acc:.3f} [{l1:.3f}, {u1:.3f}]"
#     results['F1 Score'] = f"{f1:.3f} [{l2:.3f}, {u2:.3f}]"
#     results['Kappa'] = f"{kappa:.3f} [{l3:.3f}, {u3:.3f}]"
#     results['Macro AUC'] = f"{auc:.3f} [{l4:.3f}, {u4:.3f}]"

#     cm = confusion_matrix(y_true, y_pred)
#     plot_confusion_matrix(cm, classes=labels, 
#                          save_path=os.path.join(CONFIG["save_dir"], f"cm_{name}_row_norm.png"),
#                          normalize_axis=1)
#     plot_confusion_matrix(cm, classes=labels, 
#                          save_path=os.path.join(CONFIG["save_dir"], f"cm_{name}_col_norm.png"),
#                          normalize_axis=0)

#     return results

# def map_labels(y, mapping):
#     new_y = np.copy(y)
#     for old, new in mapping.items():
#         new_y[y == old] = new
#     return new_y

# def predict_all(model, dataloader, device):
#     model.eval()
#     y_true_all, y_pred_all, y_prob_all = [], [], []
#     with torch.no_grad():
#         for x, y in tqdm(dataloader, desc="Evaluating"):
#             x, y = x.to(device), y.to(device)
#             logits = model(x)
#             probs = torch.softmax(logits, dim=-1)
#             preds = torch.argmax(probs, dim=-1)
            
#             mask = y != -100
#             for i in range(x.size(0)):
#                 valid_indices = mask[i]
#                 y_true_all.append(y[i, valid_indices].cpu().numpy())
#                 y_pred_all.append(preds[i, valid_indices].cpu().numpy())
#                 y_prob_all.append(probs[i, valid_indices].cpu().numpy())
    
#     return np.concatenate(y_true_all), np.concatenate(y_pred_all), np.concatenate(y_prob_all)

# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     test_files = load_paths_from_txt(CONFIG["test_list"])
#     test_set = SleepWindowDataset(test_files, CONFIG["window_size"], CONFIG["stride"])
#     test_loader = DataLoader(test_set, batch_size=CONFIG["batch_size"], 
#                             shuffle=False, collate_fn=collate_fn_window, num_workers=4)

#     model = TransformerSleepModel(
#         input_dim=CONFIG["input_dim"],
#         hidden_dim=CONFIG["hidden_dim"],
#         n_heads=CONFIG["n_heads"],
#         num_layers=CONFIG["num_layers"],
#         num_classes=CONFIG["num_classes"],
#         dropout=CONFIG["dropout"]
#     ).to(device)
    
#     state_dict = torch.load(CONFIG["model_path"], map_location=device)
#     if list(state_dict.keys())[0].startswith("module."):
#         state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
#     model.load_state_dict(state_dict)
#     print(f"✅ Loaded model from {CONFIG['model_path']}")

#     y_true, y_pred, y_prob = predict_all(model, test_loader, device)

#     results = {}

#     results["5-class"] = evaluate_with_ci(
#         y_true, y_pred, y_prob, 
#         labels=["W", "N1", "N2", "N3", "REM"], 
#         name="5class"
#     )

#     map_2 = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1}
#     results["2-class"] = evaluate_with_ci(
#         map_labels(y_true, map_2),
#         map_labels(y_pred, map_2),
#         np.stack([y_prob[:, 0], y_prob[:, 1:].sum(axis=1)], axis=1),
#         labels=["W", "Sleep"],
#         name="2class"
#     )

#     map_3 = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2}
#     results["3-class"] = evaluate_with_ci(
#         map_labels(y_true, map_3),
#         map_labels(y_pred, map_3),
#         np.stack([y_prob[:, 0], y_prob[:, 1:4].sum(axis=1), y_prob[:, 4]], axis=1),
#         labels=["W", "NREM", "REM"],
#         name="3class"
#     )

#     map_4 = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3}
#     results["4-class"] = evaluate_with_ci(
#         map_labels(y_true, map_4),
#         map_labels(y_pred, map_4),
#         np.stack([y_prob[:, 0], y_prob[:, 1:3].sum(axis=1), y_prob[:, 3], y_prob[:, 4]], axis=1),
#         labels=["W", "Light", "Deep", "REM"],
#         name="4class"
#     )

#     print("\n📊 Evaluation Summary (with 95% CI):")
#     headers = ["Accuracy", "F1 Score", "Kappa", "Macro AUC"]
#     print(f"{'Category':<10}  " + "  ".join(f"{h:<20}" for h in headers))
#     print("-" * 90)
#     for cat, metrics in results.items():
#         print(f"{cat:<10}  " + "  ".join(f"{metrics[h]:<20}" for h in headers))
    
#     excel_path = os.path.join(CONFIG["save_dir"], "evaluation_summary.xlsx")
#     df = pd.DataFrame.from_dict(results, orient='index')
#     df = df[headers]
#     df.to_excel(excel_path)
#     print(f"\n✅ Saved evaluation summary to {excel_path}")
#     print(f"✅ Saved confusion matrices (row/column normalized) to {CONFIG['save_dir']}")

# if __name__ == '__main__':
#     main()




# "简化版本"
# import os
# import numpy as np
# import torch
# from tqdm import tqdm
# from sklearn.metrics import (
#     accuracy_score, cohen_kappa_score, f1_score,
#     roc_auc_score
# )
# from sklearn.preprocessing import label_binarize
# from torch.utils.data import DataLoader
# from dataset_sliding import SleepWindowDataset, collate_fn_window
# from model_transformer_window import TransformerSleepModel

# CONFIG = {
#     "test_list":"/data/0shared/zhaoqingshuo/sleep_data_v5/data_split/test_paths.txt",
#     "model_path": "/home/zhaoqingshuo/SDI/ECG_sleepstage/step2_model/all_42_final_model_best.pth",
#     "batch_size": 16,
#     "window_size": 15,
#     "stride": 1,
#     "num_classes": 5,
#     "input_dim": 15 * 1152,
#     "hidden_dim": 512,
#     "n_heads": 8,
#     "num_layers": 3,
#     "dropout": 0.1,
# }

# # 修复所有分类的标签映射逻辑，确保标签连续且不越界
# LABEL_MAPPINGS = {
#     "5class": {
#         "map": lambda x: x,  # 原始标签：0(Wake),1(N1),2(N2),3(N3),4(REM)
#         "names": ["Wake", "N1", "N2", "N3", "REM"],
#         "desc": "5分类（Wake, N1, N2, N3, REM）",
#         "calc_auc": False
#     },
#     "4class": {
#         # 修复映射：N1(1)、N2(2)→1(Light)；N3(3)→2(Deep)；REM(4)→3(REM)
#         "map": lambda x: {0:0, 1:1, 2:1, 3:2, 4:3}[x],
#         "names": ["Wake", "Light", "Deep", "REM"],
#         "desc": "4分类（Wake, Light, Deep, REM）",
#         "calc_auc": False
#     },
#     "3class": {
#         # 修复映射：N1(1)、N2(2)、N3(3)→1(NREM)；REM(4)→2(REM)
#         "map": lambda x: {0:0, 1:1, 2:1, 3:1, 4:2}[x],
#         "names": ["Wake", "NREM", "REM"],
#         "desc": "3分类（Wake, NREM, REM）",
#         "calc_auc": False
#     },
#     "2class_wake_sleep": {
#         # Wake(0)→0；其余(1-4)→1(Sleep)
#         "map": lambda x: 0 if x == 0 else 1,
#         "names": ["Wake", "Sleep"],
#         "desc": "2分类（Wake, Sleep）",
#         "calc_auc": True
#     },
#     "2class_nonrem_rem": {
#         # REM(4)→1；其余(0-3)→0(Non-REM)
#         "map": lambda x: 1 if x == 4 else 0,
#         "names": ["Non-REM", "REM"],
#         "desc": "2分类（Non-REM, REM）",
#         "calc_auc": True
#     }
# }

# def load_paths_from_txt(txt_file):
#     with open(txt_file, 'r') as f:
#         return f.read().splitlines()

# def predict_all(model, dataloader, device):
#     model.eval()
#     y_true_all, y_pred_all, y_prob_all = [], [], []
#     with torch.no_grad():
#         for x, y in tqdm(dataloader, desc="Evaluating"):
#             x, y = x.to(device), y.to(device)
#             logits = model(x)
#             preds = torch.argmax(logits, dim=-1)
#             probs = torch.softmax(logits, dim=-1)  # 计算类别概率
            
#             # 处理填充值(-100)
#             mask = y != -100
#             valid_y = y[mask]
#             valid_preds = preds[mask]
#             valid_probs = probs[mask]
            
#             # 收集结果
#             y_true_all.append(valid_y.cpu().numpy())
#             y_pred_all.append(valid_preds.cpu().numpy())
#             y_prob_all.append(valid_probs.cpu().numpy())
    
#     return (
#         np.concatenate(y_true_all),
#         np.concatenate(y_pred_all),
#         np.concatenate(y_prob_all)
#     )

# def calculate_metrics(y_true, y_pred, y_prob=None, class_names=None, calc_auc=False):
#     """计算基础指标，可选计算2分类ROCAUC"""
#     acc = accuracy_score(y_true, y_pred)
#     kappa = cohen_kappa_score(y_true, y_pred)
#     mf1 = f1_score(y_true, y_pred, average="macro")
#     metrics = {"Accuracy": acc, "Kappa": kappa, "Macro-F1": mf1}
    
#     # 仅2分类场景计算ROCAUC
#     if calc_auc and y_prob is not None and len(class_names) == 2:
#         # 取正类（第1类）的概率作为预测分数
#         pos_class_idx = 1
#         y_score = y_prob[:, pos_class_idx]
#         # 确保标签是0/1编码
#         y_true_bin = label_binarize(y_true, classes=[0, 1]).flatten()
#         # 处理极端情况（所有标签相同会导致AUC计算错误）
#         if len(np.unique(y_true_bin)) == 2:
#             auc = roc_auc_score(y_true_bin, y_score)
#         else:
#             auc = 0.0  # 或根据需求设为NaN
#         metrics["ROCAUC"] = auc
    
#     return metrics

# def remap_labels(y, mapping_func):
#     """标签映射转换（使用字典映射，确保正确性）"""
#     return np.array([mapping_func(label) for label in y])

# def remap_probs(y_prob, mapping_func, num_new_classes):
#     """根据标签映射合并概率（修复索引越界问题）"""
#     new_probs = np.zeros((y_prob.shape[0], num_new_classes), dtype=np.float32)
#     for old_cls in range(y_prob.shape[1]):
#         new_cls = mapping_func(old_cls)
#         # 确保new_cls在有效范围内（双重保险）
#         if 0 <= new_cls < num_new_classes:
#             new_probs[:, new_cls] += y_prob[:, old_cls]
#         else:
#             raise ValueError(f"映射后的标签{new_cls}超出目标类别数量{num_new_classes}的范围")
#     return new_probs

# def main():
#     # 设置设备
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
    
#     # 加载测试集
#     test_files = load_paths_from_txt(CONFIG["test_list"])
#     test_set = SleepWindowDataset(test_files, CONFIG["window_size"], CONFIG["stride"])
#     test_loader = DataLoader(
#         test_set, batch_size=CONFIG["batch_size"],
#         shuffle=False, collate_fn=collate_fn_window, num_workers=4,
#         pin_memory=True  # 加速GPU数据传输
#     )

#     # 初始化并加载模型
#     model = TransformerSleepModel(
#         input_dim=CONFIG["input_dim"],
#         hidden_dim=CONFIG["hidden_dim"],
#         n_heads=CONFIG["n_heads"],
#         num_layers=CONFIG["num_layers"],
#         num_classes=CONFIG["num_classes"],
#         dropout=CONFIG["dropout"]
#     ).to(device)
    
#     # 加载模型（处理DDP保存的模型）
#     state_dict = torch.load(CONFIG["model_path"], map_location=device)
#     # 如果是DDP保存的模型，去掉module.前缀
#     if list(state_dict.keys())[0].startswith("module."):
#         state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
#     model.load_state_dict(state_dict)
#     print(f"✅ Loaded model from {CONFIG['model_path']}\n")

#     # 预测（获取真实标签、预测标签、预测概率）
#     y_true_5class, y_pred_5class, y_prob_5class = predict_all(model, test_loader, device)
#     print("="*70)
#     print("📋 睡眠分期多分类体系完整评估结果（含2分类ROCAUC）")
#     print("="*70)

#     # 逐一分级计算并输出指标
#     for key, config in LABEL_MAPPINGS.items():
#         print(f"\n【{config['desc']}】")
#         try:
#             # 标签映射
#             y_true = remap_labels(y_true_5class, config["map"])
#             y_pred = remap_labels(y_pred_5class, config["map"])
#             # 概率映射（适配合并后的类别）
#             y_prob = remap_probs(y_prob_5class, config["map"], len(config["names"]))
#             # 计算指标
#             metrics = calculate_metrics(
#                 y_true, y_pred, y_prob, config["names"], config["calc_auc"]
#             )
#             # 输出核心指标（对齐格式）
#             for metric_name, value in metrics.items():
#                 print(f"{metric_name:<10}: {value:.4f}")
#             # 输出单个类别F1
#             class_f1 = f1_score(y_true, y_pred, average=None)
#             for i, (cls_name, f1) in enumerate(zip(config["names"], class_f1)):
#                 print(f"  - {cls_name:<8}: F1 = {f1:.4f}")
#         except Exception as e:
#             print(f"❌ 该分类体系计算失败：{str(e)}")

#     print("\n" + "="*70)

# if __name__ == '__main__':
#     main()




'''新代码'''
# import os
# import numpy as np
# import torch
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import (
#     accuracy_score, cohen_kappa_score, f1_score,
#     roc_auc_score, classification_report, confusion_matrix,
#     roc_curve, auc
# )
# from sklearn.preprocessing import label_binarize
# from scipy import stats
# from torch.utils.data import DataLoader
# # 假设 SleepWindowDataset, collate_fn_window, TransformerSleepModel, ThreeHeadSleepModel 已经导入
# from dataset_sliding import SleepWindowDataset, collate_fn_window
# from model_transformer import TransformerSleepModel, ThreeHeadSleepModel
# import pandas as pd

# # 测试配置（适配双二分类任务）
# CONFIG = {
#     "test_list": "/home/zhaoqingshuo/SDI/ECG_sleepstage/transformer_W_NW_REM_NREM/data_split/inner/test_paths.txt",
#     "model_path": "/data/0shared/zhaoqingshuo/SDI/model_W_NW_REM_NREM_all/best.pth",
#     "batch_size": 8,
#     "window_size": 15,
#     "stride": 1,
#     "input_dim": 15 * 1152,
#     "hidden_dim": 512,
#     "n_heads": 8,
#     "num_layers": 3,
#     "dropout": 0.1,
#     "output_dir":"/data/0shared/zhaoqingshuo/SDI/model_W_NW_REM_NREM_all/test/csv_exports/",
#     "n_bootstrap": 1000,
#     "random_seed": 42
# }

# # 双二分类任务配置（全部改为英文，避免中文乱码）
# TASKS = [
#     {
#         "name": "Wake_vs_NonWake",
#         "desc": "Binary Task: Wake vs Non-Wake",
#         "true_label_map": lambda x: 0 if x == 0 else 1,
#         "class_names": ["Wake", "Non-Wake"],
#         "calc_auc": True,
#         "pos_class": 1
#     },
#     {
#         "name": "NonREM_vs_REM",
#         "desc": "Binary Task: Non-REM vs REM",
#         "true_label_map": lambda x: 1 if x == 4 else 0,
#         "class_names": ["Non-REM", "REM"],
#         "calc_auc": True,
#         "pos_class": 1
#     }
# ]

# # 设置随机种子
# np.random.seed(CONFIG["random_seed"])
# torch.manual_seed(CONFIG["random_seed"])
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(CONFIG["random_seed"])

# # -------------------------- 修复字体警告：使用系统默认无衬线字体 --------------------------
# plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Heiti TC', 'sans-serif']
# plt.rcParams['axes.unicode_minus'] = False 
# # ------------------------------------------------------------------------------------------

# def create_output_dirs():
#     """创建输出目录"""
#     os.makedirs(CONFIG["output_dir"], exist_ok=True)
#     os.makedirs(os.path.join(CONFIG["output_dir"], "confusion_matrices"), exist_ok=True)
#     os.makedirs(os.path.join(CONFIG["output_dir"], "auroc_curves"), exist_ok=True)
#     return CONFIG["output_dir"]

# def load_paths_from_txt(txt_file):
#     with open(txt_file, 'r') as f:
#         return f.read().splitlines()

# def build_three_head_model(config, device):
#     feature_extractor = TransformerSleepModel(
#         input_dim=config["input_dim"],
#         hidden_dim=config["hidden_dim"],
#         n_heads=config["n_heads"],
#         num_layers=config["num_layers"],
#         num_classes=2,
#         dropout=config["dropout"]
#     )
#     model = ThreeHeadSleepModel(
#         feature_extractor=feature_extractor,
#         classes=2, 
#         hidden_dim=config["hidden_dim"]
#     ).to(device)
#     return model

# def predict_three_head(model, dataloader, device):
#     model.eval()
#     results = {
#         "Wake_vs_NonWake": {"y_true": [], "y_pred": [], "y_prob": []},
#         "NonREM_vs_REM": {"y_true": [], "y_pred": [], "y_prob": []}
#     }
    
#     with torch.no_grad():
#         for x, wake_labels, rem_labels, _ in tqdm(dataloader, desc="Evaluating Three-Head Model"): 
#             x = x.to(device, non_blocking=True)
#             wake_labels = wake_labels.to(device, non_blocking=True)
#             rem_labels = rem_labels.to(device, non_blocking=True)
            
#             rem_logits, wake_logits, _ = model(x) 
            
#             wake_preds = torch.argmax(wake_logits, dim=-1)
#             wake_probs = torch.softmax(wake_logits, dim=-1)
            
#             rem_preds = torch.argmax(rem_logits, dim=-1)
#             rem_probs = torch.softmax(rem_logits, dim=-1)
            
#             # 处理填充值
#             wake_mask = wake_labels != -100
#             results["Wake_vs_NonWake"]["y_true"].append(wake_labels[wake_mask].cpu().numpy())
#             results["Wake_vs_NonWake"]["y_pred"].append(wake_preds[wake_mask].cpu().numpy())
#             results["Wake_vs_NonWake"]["y_prob"].append(wake_probs[wake_mask].cpu().numpy())
            
#             rem_mask = rem_labels != -100
#             results["NonREM_vs_REM"]["y_true"].append(rem_labels[rem_mask].cpu().numpy())
#             results["NonREM_vs_REM"]["y_pred"].append(rem_preds[rem_mask].cpu().numpy())
#             results["NonREM_vs_REM"]["y_prob"].append(rem_probs[rem_mask].cpu().numpy())

#     for task in results.values():
#         task["y_true"] = np.concatenate(task["y_true"])
#         task["y_pred"] = np.concatenate(task["y_pred"])
#         task["y_prob"] = np.concatenate(task["y_prob"])
    
#     return results

# def calculate_bootstrap_ci(y_true, y_pred, y_prob, metric_func, n_bootstrap=1000, confidence=0.95):
#     """使用bootstrap方法计算指标的置信区间（强制返回3个值）"""
#     n_samples = len(y_true)
#     bootstrap_scores = []
    
#     for _ in tqdm(range(n_bootstrap), desc="Calculating Bootstrap CI", leave=False):
#         try:
#             indices = np.random.choice(n_samples, size=n_samples, replace=True)
#             y_true_sample = y_true[indices]
#             y_pred_sample = y_pred[indices]
#             y_prob_sample = y_prob[indices]
            
#             score = metric_func(y_true_sample, y_pred_sample, y_prob_sample)
#             bootstrap_scores.append(score)
#         except Exception as e:
#             # 当抽样导致指标计算失败时，使用完整数据集的指标作为近似值
#             try:
#                 bootstrap_scores.append(metric_func(y_true, y_pred, y_prob))
#             except:
#                 continue # 避免在极端情况下中断
    
#     if not bootstrap_scores:
#         default_val = 0.0
#         return (default_val, default_val, default_val)
    
#     bootstrap_scores = np.array(bootstrap_scores)
#     mean_score = np.mean(bootstrap_scores)
    
#     try:
#         lower = np.percentile(bootstrap_scores, (1 - confidence) * 100 / 2)
#         upper = np.percentile(bootstrap_scores, 100 - (1 - confidence) * 100 / 2)
#     except:
#         lower = mean_score
#         upper = mean_score
    
#     return mean_score, lower, upper

# def calculate_roc_ci(y_true, y_score, n_bootstrap, pos_class, rng_seed=CONFIG["random_seed"]):
#     """
#     通过 Bootstrap 方法计算 ROC 曲线的置信带和 AUC 的置信区间。
#     """
#     n_samples = len(y_true)
    
#     # 统一的FPR点，用于插值
#     mean_fpr = np.linspace(0, 1, 100)
#     tprs = []
#     aucs = []

#     # 设定随机种子以保证结果可复现
#     rng = np.random.RandomState(rng_seed)
    
#     for i in tqdm(range(n_bootstrap), desc="Calculating ROC Confidence Band", leave=False):
#         # 随机选择样本，带放回抽样
#         indices = rng.choice(n_samples, size=n_samples, replace=True)
        
#         y_true_sample = y_true[indices]
#         y_score_sample = y_score[indices]

#         # 确保每个子样本中正负类别都存在（否则 roc_auc_score 会失败）
#         if len(np.unique(y_true_sample)) < 2:
#             continue
            
#         fpr, tpr, _ = roc_curve(y_true_sample, y_score_sample)
        
#         # 将当前的TPR曲线插值到统一的mean_fpr点上
#         tprs.append(interp(mean_fpr, fpr, tpr))
#         tprs[-1][0] = 0.0  # 确保起点为 (0, 0)
        
#         aucs.append(auc(fpr, tpr))

#     # 如果没有成功计算任何曲线，返回默认值
#     if not tprs:
#         # 返回100个0的数组
#         return mean_fpr, np.zeros_like(mean_fpr), np.zeros_like(mean_fpr), np.zeros_like(mean_fpr), 0.0, 0.0, []

#     # 计算 ROC 曲线置信带 (2.5% 和 97.5% 分位数)
#     tprs_array = np.array(tprs)
#     tpr_lower = np.percentile(tprs_array, 2.5, axis=0)
#     tpr_upper = np.percentile(tprs_array, 97.5, axis=0)

#     # 计算 AUC 置信区间
#     auc_lower = np.percentile(aucs, 2.5)
#     auc_upper = np.percentile(aucs, 97.5)

#     # 计算平均 ROC (用于绘图中心线，虽然我们用原始曲线作为中心线)
#     mean_tpr = np.mean(tprs_array, axis=0)

#     return mean_fpr, mean_tpr, tpr_lower, tpr_upper, auc_lower, auc_upper, aucs


# # ========================== 🛠️ 混淆矩阵优化：改为纵向归一化和百分比标注 ==========================
# def plot_confusion_matrix(y_true, y_pred, class_names, task_name, output_dir):
#     """
#     【优化点】
#     1. 计算混淆矩阵的纵向归一化（按 True Label 归一化）。
#     2. 矩阵内部标注改为百分比（保留两位小数）。
#     3. 颜色条标签改为"百分比"。
#     """
#     cm_raw = confusion_matrix(y_true, y_pred)
#     # 原始数量矩阵
    
#     # 纵向归一化（按行/真实标签归一化）：每行之和为1
#     # 注意：如果某一行（真实标签）总和为0，需要避免除以零
#     row_sums = cm_raw.sum(axis=1)
#     cm_norm_vertical = np.divide(cm_raw.astype('float'), row_sums[:, np.newaxis], 
#                                  out=np.zeros_like(cm_raw, dtype=float), where=row_sums[:, np.newaxis]!=0)

    
#     # 将归一化结果转换为百分比字符串（保留两位小数）
#     # 使用 .2% 格式化为百分比
#     labels = np.asarray([f"{val:.2%}" for val in cm_norm_vertical.flatten()]).reshape(cm_raw.shape)
    
#     plt.figure(figsize=(8, 6))
    
#     # 使用纵向归一化矩阵作为颜色依据，百分比字符串作为标注
#     sns.heatmap(cm_norm_vertical, annot=labels, fmt='', cmap='Blues',
#                 xticklabels=class_names,
#                 yticklabels=class_names,
#                 cbar_kws={'label': 'Percentage (%)'}) 
    
#     plt.xlabel('Predicted Label')
#     plt.ylabel('True Label')
#     plt.title(f'Confusion Matrix (True Label Normalized Percentage) - {task_name}') 
#     plt.tight_layout()
    
#     save_path = os.path.join(output_dir, f'confusion_matrix_{task_name}_norm_true.png')
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.close()
    
#     # 保存原始数量矩阵和归一化矩阵
#     np.savez(os.path.join(output_dir, f'confusion_matrix_{task_name}.npz'),
#              count=cm_raw,
#              norm_true_label=cm_norm_vertical)
    
#     return cm_raw

# # ========================== 🛠️ AUROC曲线优化：增加95%置信带显示 ==========================
# def plot_auroc_curve(y_true, y_prob, class_names, task_name, pos_class, output_dir):
#     """
#     【优化点】
#     1. 绘制 AUROC 曲线的 95% 置信带（通过 calculate_roc_ci 计算）。
#     2. 将 AUC 值和其 95% 置信区间添加到图例中。
#     """
#     y_true_bin = label_binarize(y_true, classes=[0, 1]).flatten()
#     y_score = y_prob[:, pos_class]
    
#     # 1. 计算核心 AUC 和 ROC 曲线
#     fpr, tpr, _ = roc_curve(y_true_bin, y_score)
#     roc_auc = auc(fpr, tpr)
    
#     # 2. 计算 AUROC 95% 置信带和 AUC 置信区间 (使用新的辅助函数)
#     n_bootstrap = CONFIG["n_bootstrap"]
    
#     mean_fpr, mean_tpr, tpr_lower, tpr_upper, auc_lower, auc_upper, _ = \
#         calculate_roc_ci(y_true_bin, y_score, n_bootstrap, pos_class) # 调用新增的函数
    
#     # 3. 绘图
#     plt.figure(figsize=(8, 6))
    
#     # 绘制 95% 置信带 
#     plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='lightsalmon', alpha=.2,
#                      label='95% ROC Confidence Band')
    
#     # 绘制核心 AUROC 曲线（作为中心线）
#     plt.plot(fpr, tpr, color='darkorange', lw=2,
#              label=f'AUROC = {roc_auc:.4f} (95% CI: {auc_lower:.4f}-{auc_upper:.4f})')
    
#     # 绘制对角线
#     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate (FPR)')
#     plt.ylabel('True Positive Rate (TPR)')
#     plt.title(f'ROC Curve - {task_name} ({class_names[pos_class]} as Positive)')
#     plt.legend(loc="lower right")
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
    
#     save_path = os.path.join(output_dir, f'auroc_curve_ci_{task_name}.png')
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.close()
    
#     # 4. 保存 ROC 数据和置信带数据
#     np.savez(os.path.join(output_dir, f'roc_data_ci_{task_name}.npz'),
#              fpr=fpr, tpr=tpr, auroc=roc_auc, auc_lower=auc_lower, auc_upper=auc_upper,
#              # 新增置信带数据
#              mean_fpr=mean_fpr, tpr_lower=tpr_lower, tpr_upper=tpr_upper) 
    
#     return roc_auc, fpr, tpr

# # ========================== 未修改的核心函数保持不变 ==========================

# def calculate_binary_metrics_with_ci(y_true, y_pred, y_prob, task, output_dir):
#     """计算二分类任务的完整指标（含CI）"""
#     task_name = task["name"]
#     class_names = task["class_names"]
#     calc_auc = task["calc_auc"]
#     pos_class = task["pos_class"]
    
#     metrics = {}
    
#     # 定义指标函数
#     def acc_func(y_t, y_p, y_pr): return accuracy_score(y_t, y_p)
#     def kappa_func(y_t, y_p, y_pr): return cohen_kappa_score(y_t, y_p)
#     def macro_f1_func(y_t, y_p, y_pr): return f1_score(y_t, y_p, average="macro", zero_division=0)
#     def weighted_f1_func(y_t, y_p, y_pr): return f1_score(y_t, y_p, average="weighted", zero_division=0)
#     def auroc_func(y_t, y_p, y_pr):
#         y_t_bin = label_binarize(y_t, classes=[0, 1]).flatten()
#         y_sc = y_pr[:, pos_class]
#         if len(np.unique(y_t_bin)) < 2: return 0.0
#         return roc_auc_score(y_t_bin, y_sc)
    
#     # 计算核心指标（强制调用返回3个值的CI函数）
#     print(f"Calculating CI for {task_name}...")
#     metrics["Accuracy"] = calculate_bootstrap_ci(y_true, y_pred, y_prob, acc_func, CONFIG["n_bootstrap"])
#     metrics["Kappa"] = calculate_bootstrap_ci(y_true, y_pred, y_prob, kappa_func, CONFIG["n_bootstrap"])
#     metrics["Macro-F1"] = calculate_bootstrap_ci(y_true, y_pred, y_prob, macro_f1_func, CONFIG["n_bootstrap"])
#     metrics["Weighted-F1"] = calculate_bootstrap_ci(y_true, y_pred, y_prob, weighted_f1_func, CONFIG["n_bootstrap"])
    
#     # 计算AUROC
#     if calc_auc:
#         try:
#             # 这里的 AUROC CI 仍然通过 calculate_bootstrap_ci 计算，以保证与 core metrics 一致
#             metrics["ROCAUC"] = calculate_bootstrap_ci(y_true, y_pred, y_prob, auroc_func, CONFIG["n_bootstrap"])
            
#             y_t_bin = label_binarize(y_true, classes=[0, 1]).flatten()
#             if len(np.unique(y_t_bin)) >= 2:
#                 # 绘制 AUROC 曲线和置信带
#                 plot_auroc_curve(y_true, y_prob, class_names, task_name, pos_class, 
#                                  os.path.join(output_dir, "auroc_curves"))
#             else:
#                 print(f"⚠️ {task_name}: Only one class exists, skip AUROC curve plotting")
#         except Exception as e:
#             metrics["ROCAUC"] = (0.0, 0.0, 0.0)
#             print(f"⚠️ {task_name} AUROC calculation failed: {str(e)}")
#     else:
#         metrics["ROCAUC"] = (0.0, 0.0, 0.0)
    
#     # 类别详细指标
#     class_report = classification_report(
#         y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
#     )
#     metrics["Class_Detail"] = {
#         cls: {
#             "Precision": class_report[cls]["precision"],
#             "Recall": class_report[cls]["recall"],
#             "F1-Score": class_report[cls]["f1-score"],
#             "Support": class_report[cls]["support"]
#         } for cls in class_names
#     }
    
#     # 绘制并保存混淆矩阵（调用优化后的函数）
#     cm_raw = plot_confusion_matrix(y_true, y_pred, class_names, task_name,
#                                    os.path.join(output_dir, "confusion_matrices"))
#     metrics["Confusion_Matrix"] = {
#         "count": cm_raw 
#     }
    
#     # 强制校验所有指标格式
#     for metric_name in list(metrics.keys()):
#         if metric_name not in ["Class_Detail", "Confusion_Matrix"]:
#             val = metrics[metric_name]
#             if not isinstance(val, (tuple, list)) or len(val) != 3:
#                 print(f"⚠️ Fix metric format: {metric_name} returned {len(val) if isinstance(val, (tuple, list)) else 1} value(s), force to 3 values")
#                 metrics[metric_name] = (0.0, 0.0, 0.0)
    
#     return metrics

# def save_test_report(metrics_dict, task_configs, output_dir):
#     """保存完整的测试报告（纯英文，避免乱码）"""
#     report_path = os.path.join(output_dir, "test_report.txt")
    
#     with open(report_path, 'w', encoding='utf-8') as f:
#         f.write("="*80 + "\n")
#         f.write("📋 Two-Binary-Class Task Test Report\n")
#         f.write("="*80 + "\n")
#         f.write(f"Test Time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
#         f.write(f"Model Path: {CONFIG['model_path']}\n")
#         f.write(f"Test Set Window Count: {len(test_set)}\n")
#         f.write(f"Bootstrap Sampling Times: {CONFIG['n_bootstrap']}\n")
#         f.write(f"Confidence Level: 95%\n")
#         f.write("="*80 + "\n\n")
        
#         for task in task_configs:
#             task_name = task["name"]
#             f.write(f"【{task['desc']}】\n")
#             f.write("-" * 50 + "\n")
            
#             metrics = metrics_dict[task_name]
            
#             # 写入核心指标（含CI）
#             f.write("Core Metrics (95% Confidence Interval):\n")
#             for metric_name in ["Accuracy", "Kappa", "Macro-F1", "Weighted-F1", "ROCAUC"]:
#                 val = metrics.get(metric_name, (0.0, 0.0, 0.0))
#                 mean_val, lower_val, upper_val = val if len(val) == 3 else (0.0, 0.0, 0.0)
#                 f.write(f"  {metric_name:<12}: {mean_val:.4f} ({lower_val:.4f} - {upper_val:.4f})\n")
            
#             # 写入类别详细指标
#             f.write("\nClass-wise Metrics:\n")
#             for cls_name, cls_metrics in metrics["Class_Detail"].items():
#                 f.write(f"  {cls_name}:\n")
#                 f.write(f"    Precision: {cls_metrics['Precision']:.4f}\n")
#                 f.write(f"    Recall:    {cls_metrics['Recall']:.4f}\n")
#                 f.write(f"    F1-Score:  {cls_metrics['F1-Score']:.4f}\n")
#                 f.write(f"    Support:   {cls_metrics['Support']}\n")
            
#             # 写入混淆矩阵（只显示原始数量）
#             f.write("\nConfusion Matrix (Count):\n")
#             cm_count = metrics["Confusion_Matrix"]["count"]
#             for row in cm_count:
#                 f.write(f"    {row}\n") 
            
#             f.write("\n" + "-" * 50 + "\n\n")
    
#     print(f"📄 Test report saved to: {report_path}")
#     return report_path

# def save_metrics_to_npz(metrics_dict, output_dir):
#     """将所有指标保存为NPZ文件（便于后续分析）"""
#     npz_path = os.path.join(output_dir, "all_metrics.npz")
    
#     save_data = {}
#     for task_name, metrics in metrics_dict.items():
#         # 核心指标（强制处理3个值）
#         for metric_name in ["Accuracy", "Kappa", "Macro-F1", "Weighted-F1", "ROCAUC"]:
#             val = metrics.get(metric_name, (0.0, 0.0, 0.0))
#             mean_val, lower_val, upper_val = val if len(val) == 3 else (0.0, 0.0, 0.0)
#             save_data[f"{task_name}_{metric_name}_mean"] = mean_val
#             save_data[f"{task_name}_{metric_name}_lower"] = lower_val
#             save_data[f"{task_name}_{metric_name}_upper"] = upper_val
        
#         # 类别详细指标
#         for cls_idx, (cls_name, cls_metrics) in enumerate(metrics["Class_Detail"].items()):
#             for metric_key, metric_val in cls_metrics.items():
#                 save_data[f"{task_name}_{cls_name}_{metric_key}"] = metric_val
        
#         # 混淆矩阵（只保存数量）
#         save_data[f"{task_name}_confusion_matrix_count"] = metrics["Confusion_Matrix"]["count"]
    
#     np.savez_compressed(npz_path, **save_data)
#     print(f"📊 Metrics data saved to: {npz_path}")
#     return npz_path

# def main():
#     # 创建输出目录
#     output_dir = create_output_dirs()
    
#     # 设置设备
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#     print(f"Model path: {CONFIG['model_path']}")
#     print(f"Output directory: {output_dir}\n")
    
#     # 加载测试集
#     test_files = load_paths_from_txt(CONFIG["test_list"])
#     global test_set
#     test_set = SleepWindowDataset(test_files, CONFIG["window_size"], CONFIG["stride"])
#     test_loader = DataLoader(
#         test_set, batch_size=CONFIG["batch_size"],
#         shuffle=False, collate_fn=collate_fn_window, num_workers=4,
#         pin_memory=True
#     )
#     print(f"Test set size: {len(test_set)} windows")

#     # 构建并加载模型
#     model = build_three_head_model(CONFIG, device) 
#     try:
#         state_dict = torch.load(CONFIG["model_path"], map_location=device)
        
#         if list(state_dict.keys())[0].startswith("module."):
#             state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
#         model.load_state_dict(state_dict)
#         print("✅ Loaded three-head model successfully!\n") 
#     except Exception as e:
#         print(f"⚠️ Model loading failed. Running with random predictions (Error: {e})\n")
#         # 如果模型加载失败，将无法使用实际的model(x)进行预测，但predict_three_head中已加入模拟逻辑

#     # 预测
#     results = predict_three_head(model, test_loader, device)

#     # 计算所有指标（含CI）
#     all_metrics = {}
#     print("\n" + "="*80)
#     print("📈 Calculating Metrics and 95% Confidence Interval")
#     print("="*80)
    
#     for task in TASKS:
#         task_name = task["name"]
#         print(f"\nProcessing {task_name}...")
        
#         y_true = results[task_name]["y_true"]
#         y_pred = results[task_name]["y_pred"]
#         y_prob = results[task_name]["y_prob"]
        
#         # 确保数据不是空的 (模拟预测可能产生空数组)
#         if len(y_true) == 0:
#              print(f"⚠️ {task_name} data is empty, skipping metric calculation.")
#              all_metrics[task_name] = {}
#              continue

#         metrics = calculate_binary_metrics_with_ci(y_true, y_pred, y_prob, task, output_dir)
#         all_metrics[task_name] = metrics

#     # 输出评估结果到控制台（纯英文）
#     print("\n" + "="*80)
#     print("📋 Two-Binary-Class Task Evaluation Results (with 95% CI)")
#     print("="*80)

#     for task in TASKS:
#         task_name = task["name"]
#         if task_name not in all_metrics or not all_metrics[task_name]:
#             print(f"【{task['desc']}】: Metrics not available.")
#             continue

#         print(f"\n【{task['desc']}】")
#         print("-" * 50)
        
#         metrics = all_metrics[task_name]
        
#         # 核心指标
#         print("Core Metrics (95% CI):")
#         core_metrics = ["Accuracy", "Kappa", "Macro-F1", "Weighted-F1", "ROCAUC"]
#         for metric_name in core_metrics:
#             val = metrics.get(metric_name, (0.0, 0.0, 0.0))
#             mean_val, lower_val, upper_val = val if len(val) == 3 else (0.0, 0.0, 0.0)
#             print(f"  {metric_name:<12}: {mean_val:.4f} ({lower_val:.4f} - {upper_val:.4f})")
        
#         # 输出类别详细指标
#         print("\nClass-wise Metrics:")
#         for cls_name, cls_metrics in metrics["Class_Detail"].items():
#             print(f"  {cls_name}:")
#             print(f"    Precision: {cls_metrics['Precision']:.4f}")
#             print(f"    Recall:    {cls_metrics['Recall']:.4f}")
#             print(f"    F1-Score:  {cls_metrics['F1-Score']:.4f}")
#             print(f"    Support:   {cls_metrics['Support']}")

#     # 保存结果
#     print("\n" + "="*80)
#     print("💾 Saving Results")
#     print("="*80)
    
#     save_test_report(all_metrics, TASKS, output_dir)
#     save_metrics_to_npz(all_metrics, output_dir)
    
#     print("\n✅ Evaluation completed! All results saved to: {}".format(output_dir))

# if __name__ == '__main__':
#     main()

import os
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from scipy import stats
from torch.utils.data import DataLoader
# 假设 SleepWindowDataset, collate_fn_window, TransformerSleepModel, ThreeHeadSleepModel 已经导入
from dataset_sliding import SleepWindowDataset, collate_fn_window
from model_transformer import TransformerSleepModel, ThreeHeadSleepModel
import pandas as pd

# 测试配置
CONFIG = {
    "test_list": "/home/zhaoqingshuo/SDI/ECG_sleepstage/transformer_W_NW_REM_NREM/data_split/inner/test_paths.txt",
    "model_path": "/data/0shared/zhaoqingshuo/SDI/model_W_NW_REM_NREM_all/best.pth",
    "batch_size": 8,
    "window_size": 15,
    "stride": 1,
    "input_dim": 15 * 1152,
    "hidden_dim": 512,
    "n_heads": 8,
    "num_layers": 3,
    "dropout": 0.1,
    "output_dir":"/data/0shared/zhaoqingshuo/SDI/model_W_NW_REM_NREM_all/test/csv_exports_new/",
    "n_bootstrap": 1000,
    "random_seed": 42
}

# 任务配置：pos_class 设置为 1 对应各个二分类的正类
TASKS = [
    {
        "name": "Wake_vs_NonWake",
        "desc": "Binary Task: Wake vs Non-Wake",
        "class_names": ["Wake", "Non-Wake"],
        "calc_auc": True,
        "pos_class": 1  # 对应 Non-Wake (原标签 1,2,3,4)
    },
    {
        "name": "NonREM_vs_REM",
        "desc": "Binary Task: Non-REM vs REM",
        "class_names": ["Non-REM", "REM"],
        "calc_auc": True,
        "pos_class": 1  # 对应 REM (原标签 4)
    }
]

# 设置随机种子
np.random.seed(CONFIG["random_seed"])
torch.manual_seed(CONFIG["random_seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed(CONFIG["random_seed"])

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Heiti TC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False 

def create_output_dirs():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    os.makedirs(os.path.join(CONFIG["output_dir"], "confusion_matrices"), exist_ok=True)
    os.makedirs(os.path.join(CONFIG["output_dir"], "auroc_curves"), exist_ok=True)
    return CONFIG["output_dir"]

def load_paths_from_txt(txt_file):
    with open(txt_file, 'r') as f:
        return f.read().splitlines()

def build_three_head_model(config, device):
    feature_extractor = TransformerSleepModel(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        n_heads=config["n_heads"],
        num_layers=config["num_layers"],
        num_classes=2,
        dropout=config["dropout"]
    )
    model = ThreeHeadSleepModel(
        feature_extractor=feature_extractor,
        classes=2, 
        hidden_dim=config["hidden_dim"]
    ).to(device)
    return model

def predict_three_head(model, dataloader, device):
    model.eval()
    results = {
        "Wake_vs_NonWake": {"y_true": [], "y_pred": [], "y_prob": [], "y_raw": []},
        "NonREM_vs_REM": {"y_true": [], "y_pred": [], "y_prob": [], "y_raw": []}
    }
    
    with torch.no_grad():
        for x, wake_labels, rem_labels, depth_labels in tqdm(dataloader, desc="Evaluating Three-Head Model"): 
            x = x.to(device, non_blocking=True)
            wake_labels = wake_labels.to(device, non_blocking=True)
            rem_labels = rem_labels.to(device, non_blocking=True)
            depth_labels = depth_labels.to(device, non_blocking=True)
            
            rem_logits, wake_logits, _ = model(x) 
            
            # Wake 任务
            wake_mask = wake_labels != -100
            results["Wake_vs_NonWake"]["y_true"].append(wake_labels[wake_mask].cpu().numpy())
            results["Wake_vs_NonWake"]["y_pred"].append(torch.argmax(wake_logits, dim=-1)[wake_mask].cpu().numpy())
            results["Wake_vs_NonWake"]["y_prob"].append(torch.softmax(wake_logits, dim=-1)[wake_mask].cpu().numpy())
            results["Wake_vs_NonWake"]["y_raw"].append(depth_labels[wake_mask].cpu().numpy())
            
            # REM 任务
            rem_mask = rem_labels != -100
            results["NonREM_vs_REM"]["y_true"].append(rem_labels[rem_mask].cpu().numpy())
            results["NonREM_vs_REM"]["y_pred"].append(torch.argmax(rem_logits, dim=-1)[rem_mask].cpu().numpy())
            results["NonREM_vs_REM"]["y_prob"].append(torch.softmax(rem_logits, dim=-1)[rem_mask].cpu().numpy())
            results["NonREM_vs_REM"]["y_raw"].append(depth_labels[rem_mask].cpu().numpy())

    for task in results.values():
        for key in ["y_true", "y_pred", "y_prob", "y_raw"]:
            task[key] = np.concatenate(task[key])
    return results

def calculate_bootstrap_ci(y_true, y_pred, y_prob, y_raw, metric_func, n_bootstrap=1000, confidence=0.95):
    n_samples = len(y_true)
    bootstrap_scores = []
    rng = np.random.RandomState(CONFIG["random_seed"])
    
    for _ in tqdm(range(n_bootstrap), desc="Calculating Bootstrap CI", leave=False):
        try:
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            score = metric_func(y_true[indices], y_pred[indices], y_prob[indices], y_raw[indices])
            bootstrap_scores.append(score)
        except:
            continue
    
    if not bootstrap_scores: return (0.0, 0.0, 0.0)
    bootstrap_scores = np.array(bootstrap_scores)
    return np.mean(bootstrap_scores), np.percentile(bootstrap_scores, 2.5), np.percentile(bootstrap_scores, 97.5)

def calculate_roc_ci(y_true, y_score, n_bootstrap, pos_class, rng_seed=CONFIG["random_seed"]):
    n_samples = len(y_true)
    mean_fpr = np.linspace(0, 1, 100)
    tprs, aucs = [], []
    rng = np.random.RandomState(rng_seed)
    
    for _ in range(n_bootstrap):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        y_true_s, y_score_s = y_true[indices], y_score[indices]
        if len(np.unique(y_true_s)) < 2: continue
        fpr, tpr, _ = roc_curve(y_true_s, y_score_s)
        # 修复点：使用 np.interp 替代 scipy.interpolate.interp
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        aucs.append(auc(fpr, tpr))

    if not tprs: return mean_fpr, np.zeros(100), np.zeros(100), np.zeros(100), 0.0, 0.0, []
    tprs_array = np.array(tprs)
    return mean_fpr, np.mean(tprs_array, axis=0), np.percentile(tprs_array, 2.5, axis=0), \
           np.percentile(tprs_array, 97.5, axis=0), np.percentile(aucs, 2.5), np.percentile(aucs, 97.5), aucs

def plot_confusion_matrix(y_true, y_pred, class_names, task_name, output_dir):
    cm_raw = confusion_matrix(y_true, y_pred)
    row_sums = cm_raw.sum(axis=1)
    cm_norm = np.divide(cm_raw.astype('float'), row_sums[:, np.newaxis], 
                        out=np.zeros_like(cm_raw, dtype=float), where=row_sums[:, np.newaxis]!=0)
    labels = np.asarray([f"{val:.2%}" for val in cm_norm.flatten()]).reshape(cm_raw.shape)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=labels, fmt='', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix (True Label Normalized) - {task_name}') 
    plt.savefig(os.path.join(output_dir, f'cm_{task_name}.png'), dpi=300)
    plt.close()
    return cm_raw

def plot_auroc_curve(y_true, y_prob, class_names, task_name, pos_class, output_dir):
    y_true_bin = label_binarize(y_true, classes=[0, 1]).flatten()
    y_score = y_prob[:, pos_class]
    fpr, tpr, _ = roc_curve(y_true_bin, y_score)
    roc_auc = auc(fpr, tpr)
    mean_fpr, _, tpr_l, tpr_u, auc_l, auc_u, _ = calculate_roc_ci(y_true_bin, y_score, CONFIG["n_bootstrap"], pos_class)
    
    plt.figure(figsize=(8, 6))
    plt.fill_between(mean_fpr, tpr_l, tpr_u, color='lightsalmon', alpha=.2, label='95% ROC Band')
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUROC = {roc_auc:.4f} ({auc_l:.4f}-{auc_u:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.legend(loc="lower right"); plt.title(f'ROC - {task_name}')
    plt.savefig(os.path.join(output_dir, f'roc_{task_name}.png'), dpi=300)
    plt.close()

def calculate_binary_metrics_with_ci(y_true, y_pred, y_prob, y_raw, task, output_dir):
    tn = task["name"]; pc = task["pos_class"]
    metrics = {}
    
    def acc_f(yt, yp, ypr, yr): return accuracy_score(yt, yp)
    def kappa_f(yt, yp, ypr, yr): return cohen_kappa_score(yt, yp)
    def macro_f1_f(yt, yp, ypr, yr): return f1_score(yt, yp, average="macro", zero_division=0)
    def auroc_f(yt, yp, ypr, yr): return roc_auc_score(yt, ypr[:, pc])
    def spearman_f(yt, yp, ypr, yr):
        rho, _ = stats.spearmanr(yr, ypr[:, pc])
        return rho

    metrics["Accuracy"] = calculate_bootstrap_ci(y_true, y_pred, y_prob, y_raw, acc_f)
    metrics["Kappa"] = calculate_bootstrap_ci(y_true, y_pred, y_prob, y_raw, kappa_f)
    metrics["Macro-F1"] = calculate_bootstrap_ci(y_true, y_pred, y_prob, y_raw, macro_f1_f)
    metrics["ROCAUC"] = calculate_bootstrap_ci(y_true, y_pred, y_prob, y_raw, auroc_f)
    metrics["Spearman"] = calculate_bootstrap_ci(y_true, y_pred, y_prob, y_raw, spearman_f)
    
    plot_auroc_curve(y_true, y_prob, task["class_names"], tn, pc, os.path.join(output_dir, "auroc_curves"))
    cm_raw = plot_confusion_matrix(y_true, y_pred, task["class_names"], tn, os.path.join(output_dir, "confusion_matrices"))
    
    report = classification_report(y_true, y_pred, target_names=task["class_names"], output_dict=True, zero_division=0)
    metrics["Class_Detail"] = {cls: report[cls] for cls in task["class_names"]}
    metrics["Confusion_Matrix"] = {"count": cm_raw}
    return metrics

def save_test_report(metrics_dict, task_configs, output_dir, set_size):
    report_path = os.path.join(output_dir, "test_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Test Report - Size: {set_size}\n" + "="*50 + "\n")
        for task in task_configs:
            tn = task["name"]; m = metrics_dict[tn]
            f.write(f"Task: {tn}\n")
            for kn in ["Accuracy", "Kappa", "Macro-F1", "ROCAUC", "Spearman"]:
                val, low, high = m[kn]
                f.write(f"  {kn:<12}: {val:.4f} ({low:.4f} - {high:.4f})\n")
            f.write("\n")

def main():
    output_dir = create_output_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_files = load_paths_from_txt(CONFIG["test_list"])
    global test_set
    test_set = SleepWindowDataset(test_files, CONFIG["window_size"], CONFIG["stride"])
    test_loader = DataLoader(test_set, batch_size=CONFIG["batch_size"], collate_fn=collate_fn_window, num_workers=4)

    model = build_three_head_model(CONFIG, device) 
    state_dict = torch.load(CONFIG["model_path"], map_location=device)
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    results = predict_three_head(model, test_loader, device)
    all_metrics = {}
    
    for task in TASKS:
        tn = task["name"]
        all_metrics[tn] = calculate_binary_metrics_with_ci(
            results[tn]["y_true"], results[tn]["y_pred"], 
            results[tn]["y_prob"], results[tn]["y_raw"], 
            task, output_dir
        )

    save_test_report(all_metrics, TASKS, output_dir, len(test_set))
    print(f"✅ Finished. Results at: {output_dir}")

if __name__ == '__main__':
    main()