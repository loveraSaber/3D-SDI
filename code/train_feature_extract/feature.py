# import os
# import re
# import pandas as pd
# import numpy as np
# from scipy.stats import skew
# from sklearn.preprocessing import MinMaxScaler
# # 尝试导入 antropy，如果失败则禁用复杂度计算
# try:
#     import antropy as ant
#     ANTROPY_AVAILABLE = True
# except ImportError:
#     ANTROPY_AVAILABLE = False
#     print("警告：未找到 antropy 库，复杂度特征 (APPe, DETRf) 将返回 NaN。")

# # --------------------------------------------------------
# # 核心特征提取函数 (涵盖所有多变量和单变量特征)
# # --------------------------------------------------------
# def extract_final_multivar_features(df, filename):
    
#     # 1. 信号准备与归一化
#     try:
#         ts = df['wake_depth_score']  # 原始 SDI 信号 (W)
#         rem_raw = df['rem_logit_rem'].values.reshape(-1, 1) # 原始 REM 信号 (R)
#     except KeyError as e:
#         raise ValueError(f"CSV文件缺少必要列: {e}. 预期列为 'wake_depth_score' 和 'rem_logit_rem'.")

#     # Min-Max 归一化 (将数据缩放到 [0, 1] 范围)
#     scaler = MinMaxScaler()
#     ts_norm = scaler.fit_transform(ts.values.reshape(-1, 1)).flatten()
#     rem_norm = scaler.fit_transform(rem_raw).flatten()
    
#     # 2. 计算合成幅度信号 S(t) (多变量 AP/CV/Complexity 的基础)
#     S = np.sqrt(ts_norm**2 + rem_norm**2)

#     # 3. NSRRID 提取
#     match = re.search(r'-(\d+)_', filename)
#     nsrrid = match.group(1) if match else 'ID_NOT_FOUND'

#     # --- 统计和稳定性特征 ---
    
#     # 3.1 单变量 AP/CV/SK (基于原始 SDI)
#     ap_univar = np.mean(ts) 
#     cv_univar = np.std(ts) / np.mean(ts) if np.mean(ts) != 0 else 0
#     depth_skewness = skew(ts)
    
#     # 3.2 多变量 AP/CV/RHO (基于 S)
#     avg_mag = np.mean(S)
#     cv_mag = np.std(S) / avg_mag if avg_mag != 0 else 0
#     rho_wr = np.corrcoef(ts_norm, rem_norm)[0, 1]

#     # --- 睡眠片段化 (多变量联合阈值 RB) ---
#     ratio_below_02_multivar = ((ts_norm < 0.2) & (rem_norm < 0.2)).mean()
#     ratio_below_03_multivar = ((ts_norm < 0.3) & (rem_norm < 0.3)).mean() 

#     # --- 复杂度 (应用于合成幅度 S) ---
#     app = np.nan 
#     detrended = np.nan 
#     if ANTROPY_AVAILABLE:
#         try:
#             # 使用合成幅度 S 计算复杂度
#             app = ant.app_entropy(S)
#             detrended = ant.detrended_fluctuation(S)
#         except Exception:
#             pass
    
#     # 4. 结果汇总
#     manual_features = {
#         'NSRRID': nsrrid,
#         'AP_UNIVAR': ap_univar,
#         'AVG_MAG_MULTIVAR': avg_mag,
#         'CV_UNIVAR': cv_univar,
#         'CV_MAG_MULTIVAR': cv_mag,
#         'SK': depth_skewness,
#         'RHO_WR': rho_wr,
#         'RB_02': ratio_below_02_multivar,
#         'RB_03': ratio_below_03_multivar,
#         'APPe_MAG': app,
#         'DETRf_MAG': detrended,
#     }

#     return manual_features

# # --------------------------------------------------------
# # 批处理主函数
# # --------------------------------------------------------
# def process_csv_directory(input_dir, output_dir):
#     """
#     遍历输入目录中的所有 CSV 文件，提取特征并保存。
#     """
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#         print(f"已创建输出目录: {output_dir}")
    
#     files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
#     if not files:
#         print(f"错误: 在目录 {input_dir} 中未找到任何 CSV 文件。")
#         return

#     print(f"--- 找到 {len(files)} 个 CSV 文件进行处理 ---")
#     all_features = []

#     for i, file in enumerate(files):
#         print(f"正在处理 ({i+1}/{len(files)}): {file}")
#         file_path = os.path.join(input_dir, file)
        
#         try:
#             df = pd.read_csv(file_path)
#             if len(df) < 5: 
#                 print(f"跳过文件 {file}: 数据行太少。")
#                 continue
                
#             feats = extract_final_multivar_features(df, file)
#             all_features.append(feats)
            
#         except Exception as e:
#             print(f"‼️ 错误：处理文件 {file} 失败。原因: {e}")
#             continue

#     if all_features:
#         result_df = pd.DataFrame(all_features)
#         output_file_name = f'all_final_multivar_features_summary.csv'
#         output_path = os.path.join(output_dir, output_file_name)
#         result_df.to_csv(output_path, index=False)
#         print("\n==============================================")
#         print(f"✅ 所有特征已汇总并保存至:\n{output_path}")
#         print("==============================================")
#     else:
#         print("\n没有成功提取任何特征。")

# # --------------------------------------------------------
# # 运行配置 (请修改以下路径)
# # --------------------------------------------------------
# if __name__ == "__main__":
#     # ⚠️ 请将此处替换为您的实际 CSV 文件夹路径
#     input_directory = "/data/0shared/zhaoqingshuo/SDI/model_W_NW_REM_NREM/test/csv_exports/"
    
#     # ⚠️ 请将此处替换为您希望保存结果的输出目录
#     output_directory = "/data/0shared/zhaoqingshuo/SDI/model_W_NW_REM_NREM/test/feature/"
    
#     if input_directory == '/path/to/your/input_csv_folder':
#         print("请注意：请将 input_directory 和 output_directory 替换为您的实际路径，然后再次运行。")
#     else:
#         process_csv_directory(input_directory, output_directory)



# import os
# import re
# import pandas as pd
# import numpy as np
# from scipy.stats import skew
# from sklearn.preprocessing import MinMaxScaler
# from scipy.spatial import ConvexHull
# from scipy.fft import fft, fftfreq

# # 尝试导入 antropy
# try:
#     import antropy as ant
#     ANTROPY_AVAILABLE = True
# except ImportError:
#     ANTROPY_AVAILABLE = False
#     print("⚠️ 提示: 未检测到 antropy 库。复杂度特征将返回 NaN。")

# # --------------------------------------------------------
# # 全维度特征提取函数 (ID提取逻辑已更新)
# # --------------------------------------------------------
# def extract_master_features(df, filename):
    
#     # --- 1. ID 提取 (针对 SHHS 格式修改) ---
#     # 目标文件名示例: shhs1-200001_features_predictions.csv
#     # 提取目标: 200001
    
#     # 优先匹配 shhsX-XXXXXX 格式
#     match = re.search(r'shhs\d+-(\d+)', filename)
    
#     if match:
#         nsrrid = match.group(1)
#     else:
#         # 备用逻辑：如果没有找到 shhs 前缀，尝试提取文件名中的第一组 6 位数字
#         match_backup = re.search(r'(\d{6})', filename)
#         if match_backup:
#             nsrrid = match_backup.group(1)
#         else:
#             # 再次备用：尝试提取两个符号中间的数字 (如 -200001_)
#             match_last = re.search(r'-(\d+)_', filename)
#             nsrrid = match_last.group(1) if match_last else 'ID_NOT_FOUND'

#     # --- 2. 数据加载与预处理 ---
#     try:
#         ts = pd.to_numeric(df['wake_depth_score'], errors='coerce').fillna(0)
#         rem_raw = pd.to_numeric(df['rem_logit_rem'], errors='coerce').fillna(0)
#     except KeyError as e:
#         # 如果列名不对，打印错误并跳过
#         print(f"Skipping {filename}: {e}")
#         return None

#     # 归一化 [0, 1]
#     scaler = MinMaxScaler()
#     ts_norm = scaler.fit_transform(ts.values.reshape(-1, 1)).flatten()
#     rem_norm = scaler.fit_transform(rem_raw.values.reshape(-1, 1)).flatten()
    
#     N = len(ts_norm)
#     if N < 10: return None

#     # --- 3. 构建基础向量 ---
#     S = np.sqrt(ts_norm**2 + rem_norm**2) # 状态模长
#     diff_w = np.diff(ts_norm)
#     diff_r = np.diff(rem_norm)
#     velocity = np.sqrt(diff_w**2 + diff_r**2) # 速度
#     angles = np.degrees(np.arctan2(rem_norm, ts_norm)) # 极角

#     # --- 4. 特征计算 ---

#     # A. 基础统计 (Legacy)
#     ap_univar = np.mean(ts) 
#     cv_univar = np.std(ts) / ap_univar if ap_univar != 0 else 0
#     sk_univar = skew(ts)
#     rho_wr = np.corrcoef(ts_norm, rem_norm)[0, 1]

#     # B. 三维能量 (Energy)
#     total_energy = np.mean(S)
#     cv_energy = np.std(S) / total_energy if total_energy != 0 else 0

#     # C. 空间结构 (Structure)
#     core_sleep_mask = (ts_norm < 0.2) & (rem_norm < 0.2)
#     core_sleep_ratio = np.mean(core_sleep_mask)
    
#     ratio_below_03 = ((ts_norm < 0.3) & (rem_norm < 0.3)).mean()
    
#     rem_sector_mask = (angles >= 60) & (angles <= 120) & (S > 0.2)
#     rem_sector_ratio = np.mean(rem_sector_mask)
    
#     polar_mean_angle = np.mean(angles)

#     # D. 动力学与几何 (Kinematics)
#     traj_agitation = np.sum(velocity) / N
#     max_shock = np.max(velocity)
    
#     try:
#         points_2d = np.column_stack((ts_norm, rem_norm))
#         hull_area = ConvexHull(points_2d).volume
#     except:
#         hull_area = 0

#     # E. 复杂性 (Complexity)
#     try:
#         cov_mat = np.cov(np.vstack((ts_norm, rem_norm)))
#         dispersion = np.linalg.det(cov_mat)
#     except:
#         dispersion = 0

#     complexity_appe = np.nan
#     fractal_detrend = np.nan
#     if ANTROPY_AVAILABLE:
#         try:
#             complexity_appe = ant.sample_entropy(S)
#             fractal_detrend = ant.detrended_fluctuation(S)
#         except:
#             pass
            
#     try:
#         rem_detrend = rem_norm - np.mean(rem_norm)
#         yf = fft(rem_detrend)
#         xf = fftfreq(N, 1)
#         idx = np.argmax(np.abs(yf[1:N//2])) + 1
#         rem_period = 1 / xf[idx] if xf[idx] != 0 else 0
#     except:
#         rem_period = np.nan

#     # --- 5. 结果打包 ---
#     features = {
#         'NSRRID': nsrrid, # 提取出的 ID (如 200001)
        
#         # 核心五维特征
#         'TOTAL_SLEEP_ENERGY': total_energy,
#         'CORE_SLEEP_RATIO': core_sleep_ratio,
#         'REM_PURE_RATIO': rem_sector_ratio,
#         'STATE_DISPERSION': dispersion,
#         'ENTROPY_APPE': complexity_appe,
        
#         # 辅助详细特征
#         'AP_UNIVAR': ap_univar,
#         'CV_UNIVAR': cv_univar,
#         'SK_UNIVAR': sk_univar,
#         'RHO_WR': rho_wr,
#         'CV_SLEEP_ENERGY': cv_energy,
#         'DEEP_SLEEP_RATIO': ratio_below_03,
#         'POLAR_MEAN_ANGLE': polar_mean_angle,
#         'TRAJ_AGITATION': traj_agitation,
#         'MAX_SHOCK_VELOCITY': max_shock,
#         'STATE_HULL_AREA': hull_area,
#         'FRACTAL_DFA': fractal_detrend,
#         'REM_CYCLE_PERIOD': rem_period
#     }

#     return features

# # --------------------------------------------------------
# # 批处理逻辑
# # --------------------------------------------------------
# def process_directory(input_dir, output_dir):
#     if not os.path.exists(output_dir): os.makedirs(output_dir)
    
#     files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
#     print(f"--- 🚀 正在处理目录: {input_dir} ---")
#     print(f"--- 找到 {len(files)} 个 CSV 文件 ---")
    
#     results = []
#     for i, file in enumerate(files):
#         if (i+1) % 50 == 0: print(f"已处理 {i+1}/{len(files)}...")
#         try:
#             df = pd.read_csv(os.path.join(input_dir, file))
#             res = extract_master_features(df, file)
#             if res: results.append(res)
#         except Exception as e:
#             print(f"❌ 文件 {file} 出错: {e}")
            
#     if results:
#         res_df = pd.DataFrame(results)
#         # 确保 NSRRID 排在第一列
#         cols = ['NSRRID'] + [c for c in res_df.columns if c != 'NSRRID']
#         res_df = res_df[cols]
        
#         # 根据 NSRRID 排序 (可选)
#         try:
#             res_df['NSRRID'] = res_df['NSRRID'].astype(str)
#             res_df = res_df.sort_values('NSRRID')
#         except:
#             pass

#         out_file = 'SHHS_Master_Sleep_3D_Features.csv'
#         out_path = os.path.join(output_dir, out_file)
#         res_df.to_csv(out_path, index=False)
#         print(f"\n✅ 提取完成! \n📂 输出文件: {out_path}")
#     else:
#         print("⚠️ 未提取到任何特征，请检查输入目录或文件格式。")

# # --------------------------------------------------------
# # 执行部分
# # --------------------------------------------------------
# if __name__ == "__main__":
#     # ⚠️ 输入路径 (你提供的路径)
#     IN_DIR = "/data/0shared/zhaoqingshuo/SDI/model_W_NW_REM_NREM_all/test/csv_exports/csv/csv_exports/"
    
#     # ⚠️ 输出路径 (我在输入路径下创建了一个 feature_output 文件夹，你可以修改)
#     OUT_DIR = "/data/0shared/zhaoqingshuo/SDI/model_W_NW_REM_NREM_all/test/csv_exports_shhs/csv/feature_output/"
    
#     process_directory(IN_DIR, OUT_DIR)





# import os
# import re
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from scipy.stats import spearmanr, skew, entropy
# from scipy.signal import correlate, welch, savgol_filter
# from scipy.spatial import ConvexHull, distance_matrix

# # --------------------------------------------------------
# # 1. 基础工具函数
# # --------------------------------------------------------

# def get_entropy_weight(p_array):
#     """计算分类置信度权重 (1 - 归一化熵)"""
#     p = np.clip(p_array, 1e-7, 1 - 1e-7)
#     ent = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
#     return 1 - ent

# def box_counting_dimension(coords, k_max=6):
#     """简易盒计数法计算分形维度"""
#     # 归一化到 [0, 1] 空间
#     c_min = coords.min(axis=0)
#     c_max = coords.max(axis=0)
#     scaled = (coords - c_min) / (c_max - c_min + 1e-9)
    
#     counts = []
#     scales = [2**k for k in range(2, k_max)]
#     for s in scales:
#         # 将空间划分为 s^3 个格子，统计非空格子数
#         bins = np.linspace(0, 1, s + 1)
#         hist, _ = np.histogramdd(scaled, bins=(bins, bins, bins))
#         counts.append(np.sum(hist > 0))
    
#     # 对 log(1/scale) 和 log(counts) 进行线性拟合
#     coeffs = np.polyfit(np.log(scales), np.log(counts), 1)
#     return coeffs[0]

# # --------------------------------------------------------
# # 2. 核心提取函数
# # --------------------------------------------------------

# def extract_all_layer_features(df, filename, w_thresholds, r_thresholds):
#     try:
#         # 数据归一化与准备
#         scaler = MinMaxScaler()
#         body = scaler.fit_transform(pd.to_numeric(df['wake_depth_score'], errors='coerce').fillna(0).values.reshape(-1, 1)).flatten()
#         brain = scaler.fit_transform(pd.to_numeric(df['rem_logit_rem'], errors='coerce').fillna(0).values.reshape(-1, 1)).flatten()
#         N = len(body)
#         if N < 60: return None
        
#         # 构建三维坐标 (X: 大脑, Y: 身体, Z: 时间轴)
#         time_z = np.linspace(0, 1, N) # 时间轴归一化
#         coords = np.vstack([brain, body, time_z]).T
        
#     except Exception as e:
#         return None

#     f = {'NSRRID': filename}

#     # --- 第一层面：时域特征 ---
#     f['AP'] = np.mean(body)
#     f['RB'] = np.mean(body < w_thresholds[0])
#     is_rem = brain >= r_thresholds[3]
#     f['MDR'] = np.mean(body[is_rem]) if np.any(is_rem) else 0
#     f['PR'] = np.mean(is_rem)
#     f['CV'] = np.std(body) / (np.mean(body) + 1e-7)
#     f['SK'] = skew(body)
#     conf_w = get_entropy_weight(brain)
#     f['Conf_Weighted_AP'] = np.mean(body * conf_w)

#     # --- 第二层面：频域特征 ---
#     freqs, psd = welch(body, fs=1/30, nperseg=min(N, 256)) 
#     f['SDI_Peak_Freq'] = freqs[np.argmax(psd)]
#     f['SDI_Spec_Centroid'] = np.sum(freqs * psd) / np.sum(psd)

#     # --- 第三层面：非线性与几何特征 (新增三维特征) ---
#     # 1. 脑体相位延迟
#     corr = correlate(body - np.mean(body), brain - np.mean(brain), mode='full')
#     f['BB_Phase_Lag'] = np.arange(-N + 1, N)[np.argmax(corr)]
    
#     # 2. 空间占用熵 (Spatial Entropy)
#     hist_2d, _, _ = np.histogram2d(brain, body, bins=5, range=[[0, 1], [0, 1]])
#     f['Spatial_Entropy'] = entropy(hist_2d.flatten() + 1e-9)

#     # 3. 三维弯曲度 (Curvature)
#     # 使用 Savitzky-Golay 平滑减少微小噪声对导数的影响
#     smooth_coords = savgol_filter(coords, window_length=min(11, N-1 if N%2!=0 else N-2), polyorder=3, axis=0)
#     r_prime = np.gradient(smooth_coords, axis=0)
#     r_prime_prime = np.gradient(r_prime, axis=0)
#     # 计算曲率公式: kappa = |r' x r''| / |r'|^3
#     cross_prod = np.cross(r_prime, r_prime_prime)
#     f['3D_Mean_Curvature'] = np.mean(np.linalg.norm(cross_prod, axis=1) / (np.linalg.norm(r_prime, axis=1)**3 + 1e-6))

#     # 4. 状态空间重现率 (Recurrence Rate, RR)
#     # 设定距离阈值 epsilon (空间跨度的 5%)
#     xy_coords = coords[:, :2]
#     epsilon = 0.05
#     dist_mtx = distance_matrix(xy_coords[::5], xy_coords[::5]) # 降采样提高速度
#     f['Recurrence_Rate'] = np.mean(dist_mtx < epsilon)

#     # 5. 三维凸包体积比 (3D Convex Hull Ratio)
#     try:
#         hull = ConvexHull(coords)
#         v_hull = hull.volume
#         v_box = np.prod(coords.max(axis=0) - coords.min(axis=0))
#         f['3D_Convex_Hull_Ratio'] = v_hull / (v_box + 1e-9)
#     except:
#         f['3D_Convex_Hull_Ratio'] = 0

#     # 6. 分形维度 (Fractal Dimension)
#     f['Fractal_Dimension'] = box_counting_dimension(coords)

#     # --- 第四层面：结构特征 ---
#     f['Path_Length_2D'] = np.sum(np.sqrt(np.diff(body)**2 + np.diff(brain)**2)) / N
#     f['Transition_Rate'] = np.sum(np.abs(np.diff(body > w_thresholds[1]))) / (N * 30 / 3600)
    
#     accel = np.diff(body, n=2)
#     f['SDI_Collapse_Accel'] = np.min(accel) if len(accel) > 0 else 0
    
#     f['BB_Antagonistic_Ratio'] = np.mean((np.diff(body) * np.diff(brain)) < 0)
    
#     # 矢量方向熵 (球面坐标映射)
#     delta_r = np.diff(coords, axis=0)
#     norms = np.linalg.norm(delta_r, axis=1, keepdims=True)
#     u = delta_r / (norms + 1e-9)
#     theta = np.arccos(np.clip(u[:, 1], -1, 1)) # 映射到身体轴方向
#     phi = np.arctan2(u[:, 2], u[:, 0]) # 时间与大脑轴平面
#     hist_ang, _, _ = np.histogram2d(theta, phi, bins=8)
#     f['Transition_Vector_Entropy'] = entropy(hist_ang.flatten() + 1e-9)

#     return f

# # --------------------------------------------------------
# # 3. 批处理主程序 (保持不变)
# # --------------------------------------------------------

# def main_process(input_path, output_path):
#     w_ts = [0.1022, 0.1406, 0.9584]
#     r_ts = [0.2580, 0.3086, 0.6299, 0.7236]
    
#     if not os.path.exists(output_path):
#         os.makedirs(output_path)
    
#     results = []
#     files = [f for f in os.listdir(input_path) if f.endswith('.csv')]
#     print(f"--- 🚀 开始提取全维度特征 (n={len(files)}) ---")
    
#     for i, file in enumerate(files):
#         try:
#             df = pd.read_csv(os.path.join(input_path, file))
#             res = extract_all_layer_features(df, file, w_ts, r_ts)
#             if res: results.append(res)
#             if (i+1) % 100 == 0: print(f"进度: {i+1}/{len(files)}...")
#         except Exception as e:
#             print(f"跳过文件 {file}: {e}")

#     final_df = pd.DataFrame(results)
#     final_df.to_csv(os.path.join(output_path, 'SDI_3D_Full_Features.csv'), index=False)
#     print("✅ 全维度（含几何拓扑）特征提取完成！")

# if __name__ == "__main__":
#     IN ="/data/0shared/zhaoqingshuo/SDI/model_W_NW_REM_NREM_all/test/csv_exports_new/csv/mros/"
#     OUT ="/data/0shared/zhaoqingshuo/SDI/model_W_NW_REM_NREM_all/test/csv_exports_new/csv/mros_feature/"
#     main_process(IN, OUT)

import os
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr, skew, entropy
from scipy.signal import correlate, welch

# --------------------------------------------------------
# 1. 基础工具函数
# --------------------------------------------------------

def get_entropy_weight(p_array):
    """计算分类置信度权重 (1 - 归一化熵)"""
    p = np.clip(p_array, 1e-7, 1 - 1e-7)
    ent = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    return 1 - ent

def calculate_appe(x, m=2, r=0.2):
    """计算近似熵 (Approximate Entropy) 的简化实现 [cite: 704]"""
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])
    def _phi(m):
        x_split = [x[i:i + m] for i in range(len(x) - m + 1)]
        C = [len([1 for x_j in x_split if _maxdist(x_i, x_j) <= r]) / (len(x) - m + 1) for x_i in x_split]
        return sum(np.log(C)) / (len(x) - m + 1)
    # 对于大数据集建议降采样或使用优化库，此处提供逻辑参考
    try:
        return abs(_phi(m) - _phi(m + 1))
    except:
        return 0

# --------------------------------------------------------
# 2. 四大层面特征提取核心类
# --------------------------------------------------------

def extract_all_layer_features(df, filename, w_thresholds, r_thresholds):
    # 数据归一化与准备
    try:
        scaler = MinMaxScaler()
        body = scaler.fit_transform(pd.to_numeric(df['wake_depth_score'], errors='coerce').fillna(0).values.reshape(-1, 1)).flatten()
        brain = scaler.fit_transform(pd.to_numeric(df['rem_logit_rem'], errors='coerce').fillna(0).values.reshape(-1, 1)).flatten()
        N = len(body)
        if N < 60: return None
    except: return None

    f = {'NSRRID': filename} # 实际应用中建议使用正则提取ID

    # --- 第一层面：时域特征 (Time Domain) --- [cite: 120, 701, 702]
    f['AP'] = np.mean(body)                        # SDI面积占比 (修复效率) [cite: 180]
    f['RB'] = np.mean(body < w_thresholds[0])      # 浅睡率 (觉醒风险) [cite: 179]
    is_rem = brain >= r_thresholds[3]
    f['MDR'] = np.mean(body[is_rem]) if np.any(is_rem) else 0  # REM期间平均深度 [cite: 262]
    f['PR'] = np.mean(is_rem)                      # REM占比 [cite: 262]
    f['CV'] = np.std(body) / (np.mean(body) + 1e-7) # 变异系数 (波动性) [cite: 120]
    f['SK'] = skew(body)                           # 偏度 [cite: 120]
    conf_w = get_entropy_weight(brain)
    f['Conf_Weighted_AP'] = np.mean(body * conf_w) # 专属特征：置信加权深度面积

    # --- 第二层面：频域特征 (Frequency Domain) --- 
    # 注意：此处是对预测轨迹序列进行频谱分析，反映睡眠波动的周期性
    freqs, psd = welch(body, fs=1/30, nperseg=min(N, 256)) 
    f['SDI_Peak_Freq'] = freqs[np.argmax(psd)]     # SDI波动主频率
    f['SDI_Spec_Centroid'] = np.sum(freqs * psd) / np.sum(psd) # SDI频谱重心

    # --- 第三层面：非线性特征 (Non-linear) --- [cite: 120, 703]
    # f['APPe'] = calculate_appe(body[::5])        # 近似熵 (计算耗时，建议针对性开启) [cite: 182]
    # 专属特征：脑体相位延迟 (身体响应大脑的滞后Epoch数)
    corr = correlate(body - np.mean(body), brain - np.mean(brain), mode='full')
    f['BB_Phase_Lag'] = np.arange(-N + 1, N)[np.argmax(corr)]
    # 专属特征：空间占用熵 (2D平面分布的混乱度)
    hist_2d, _, _ = np.histogram2d(brain, body, bins=5, range=[[0, 1], [0, 1]])
    f['Spatial_Entropy'] = entropy(hist_2d.flatten() + 1e-9)
    # 专属特征：矢量方向熵 (轨迹移动方向的随机性)
    angles = np.arctan2(np.diff(body), np.diff(brain))
    hist_ang, _ = np.histogram(angles, bins=12, range=[-np.pi, np.pi])
    f['Directional_Entropy'] = entropy(hist_ang + 1e-9)

    # --- 第四层面：结构特征 (Structural) --- 
    f['Path_Length_2D'] = np.sum(np.sqrt(np.diff(body)**2 + np.diff(brain)**2)) / N # 轨迹总长
    f['Transition_Rate'] = np.sum(np.abs(np.diff(body > w_thresholds[1]))) / (N * 30 / 3600) # 切换率
    accel = np.diff(body, n=2)
    f['SDI_Collapse_Accel'] = np.min(accel) if len(accel) > 0 else 0 # 崩塌加速度 (微觉醒强度)
    f['BB_Antagonistic_Ratio'] = np.mean((np.diff(body) * np.diff(brain)) < 0) # 脑体拮抗系数
    # N3平均持续时间 (Bout Duration)
    is_n3 = body >= w_thresholds[2]
    bouts = np.diff(np.concatenate(([0], is_n3.view(np.int8), [0])))
    durs = np.where(bouts == -1)[0] - np.where(bouts == 1)[0]
    f['Bout_N3_Mean_Dur'] = np.mean(durs) if len(durs) > 0 else 0

    return f

# --------------------------------------------------------
# 3. 批处理与保存
# --------------------------------------------------------

def main_process(input_path, output_path):
    w_ts = [0.1022, 0.1406, 0.9584] # 身体轴阈值
    r_ts = [0.2580, 0.3086, 0.6299, 0.7236] # REM轴阈值
    
    results = []
    files = [f for f in os.listdir(input_path) if f.endswith('.csv')]
    for file in files:
        df = pd.read_csv(os.path.join(input_path, file))
        res = extract_all_layer_features(df, file, w_ts, r_ts)
        if res: results.append(res)
    
    final_df = pd.DataFrame(results)
    final_df.to_csv(os.path.join(output_path, 'SDI_Full_Layer_Features.csv'), index=False)
    print("✅ 四大层面特征提取完成！")

if __name__ == "__main__":
    IN ="/data/0shared/zhaoqingshuo/SDI/model_W_NW_REM_NREM_all/test/csv_exports_new/csv/mros/"
    OUT ="/data/0shared/zhaoqingshuo/SDI/model_W_NW_REM_NREM_all/test/csv_exports_new/csv/mros_feature/"
    main_process(IN, OUT)