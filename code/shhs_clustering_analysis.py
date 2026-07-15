import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy import stats

# ==========================================
# 1. 核心定义
# ==========================================
# 聚类使用的SDI特征（基于相关性分析选出的特征）
sdi_cluster_features = [
    'Conf_Weighted_AP',
    'Directional_Entropy',
    'Path_Length_2D',
    'Transition_Rate',
    'BB_Antagonistic_Ratio'
]

# 绘图映射表
feature_mapping = {
    'age_clean': 'Age', 'bmi_clean': 'BMI', 'se': 'SE', 'sl': 'SL',
    'AP': 'AP', 'RB': 'RB', 'MDR': 'MDR', 'PR': 'PR', 'CV': 'CV', 'SK': 'SK',
    'Conf_Weighted_AP': 'CWSD', 'SDI_Peak_Freq': 'SDI Peak Freq',
    'SDI_Spec_Centroid': 'SDI Spec Centroid', 'BB_Phase_Lag': 'BB Phase Lag',
    'Spatial_Entropy': 'Spatial Entropy', 'Directional_Entropy': 'STDE',
    '3D_Mean_Curvature': '3D Mean Curvature',
    'Recurrence_Rate': 'Recurrence Rate',
    '3D_Convex_Hull_Ratio': '3D Convex Hull Ratio',
    'Fractal_Dimension': 'Fractal Dimension',
    'Path_Length_2D': 'TDTL', 'Transition_Rate': 'SSTR',
    'SDI_Collapse_Accel': 'SDI Collapse Accel', 'BB_Antagonistic_Ratio': 'BBAI',
    'Transition_Vector_Entropy': 'Transition Vector Entropy',
    'sex_clean': 'Sex', 'ahi_binary': 'OSA', 
    'htn_binary': 'HTN', 'diabetes_binary': 'Diabetes', 'cvd_binary': 'CVD',
    'good_sleep': 'Good Sleep', 'poor_sleep': 'Poor Sleep', 'insomnia': 'Insomnia'
}

# ==========================================
# 2. 数据加载与清理 (仅使用SHHS1基线数据)
# ==========================================
def load_shhs_data(base_path):
    # 仅使用SHHS1基线数据 (Sleep Heart Health Study Visit 1)
    df_baseline = pd.read_csv(os.path.join(base_path, 'shhs1-dataset-0.20.0.csv'), low_memory=False)
    df_cvd = pd.read_csv(os.path.join(base_path, 'shhs-cvd-summary-dataset-0.19.0.csv'), low_memory=False)
    df_harmonized = pd.read_csv(os.path.join(base_path, 'shhs-harmonized-dataset-0.20.0.csv'))
    
    # 加载SDI 3D全特征文件
    df_features = pd.read_csv(os.path.join(base_path, 'SDI_Full_Layer_Features.csv'))
    
    # 从文件名中提取ID (格式: shhs1-202037_features_predictions.csv -> 202037)
    # 只匹配shhs1的数据
    def extract_id_from_filename(filename):
        """从文件名中提取nsrrid，只匹配shhs1的数据"""
        if pd.isna(filename):
            return None
        filename_str = str(filename)
        # 只匹配以shhs1-开头的文件名
        match = re.search(r'shhs1-(\d+)', filename_str)
        if match:
            return match.group(1)
        return None
    
    # 提取ID列，只保留shhs1的数据
    df_features['nsrrid'] = df_features['NSRRID'].apply(extract_id_from_filename)
    df_features = df_features.dropna(subset=['nsrrid'])
    
    # 将所有数据框的nsrrid统一转换为字符串以确保匹配
    df_baseline['nsrrid'] = df_baseline['nsrrid'].astype(str)
    df_cvd['nsrrid'] = df_cvd['nsrrid'].astype(str)
    df_features['nsrrid'] = df_features['nsrrid'].astype(str)
    
    print(f"特征文件提取ID后（仅shhs1）: {len(df_features)} 行")
    print(f"特征文件ID示例: {df_features['nsrrid'].head().tolist()}")
    print(f"基线文件ID示例: {df_baseline['nsrrid'].head().tolist()}")

    # 逐步合并
    df = pd.merge(df_baseline, df_cvd, on='nsrrid', how='inner')
    # 只使用visit 1的数据（基线数据），避免重复行
    df_harmonized_v1 = df_harmonized[df_harmonized['visitnumber'] == 1][['nsrrid', 'nsrr_ahi_hp4u_aasm15']].copy()
    df_harmonized_v1['nsrrid'] = df_harmonized_v1['nsrrid'].astype(str)
    df = pd.merge(df, df_harmonized_v1, on='nsrrid', how='inner')
    df = pd.merge(df, df_features, on='nsrrid', how='inner')

    # 处理合并产生的重复列 (解决 age_s1 报错)
    def fix_col(df, name):
        if f"{name}_x" in df.columns: return df[f"{name}_x"].fillna(df[f"{name}_y"])
        return df[name]

    # SHHS1基线变量 (所有变量均来自Visit 1)
    df['age_clean'] = fix_col(df, 'age_s1')  # SHHS1年龄
    df['bmi_clean'] = fix_col(df, 'bmi_s1')  # SHHS1 BMI
    # SE/SL 兼容原列和合并后 _x/_y 后缀
    se_raw = fix_col(df, 'slpeffp')
    sl_raw = fix_col(df, 'slplatp')
    if se_raw.isna().all():
        print("警告: 找不到SE列 slpeffp/slpeffp_x/slpeffp_y")
    if sl_raw.isna().all():
        print("警告: 找不到SL列 slplatp/slplatp_x/slplatp_y")
    df['se'] = pd.to_numeric(se_raw, errors='coerce')
    df['sl'] = pd.to_numeric(sl_raw, errors='coerce')

    df['sex_clean'] = np.where(df['gender_x'] == 1, 1, 0)  # SHHS1性别
    
    # 定义结局与疾病变量（均基于SHHS1基线数据）[cite: 471-478, 668-670]
    df['cvd_binary'] = df['any_cvd'].fillna(0).astype(int)  # CVD（来自cvd summary，但基于基线后的随访）
    df['htn_binary'] = df['htnderv_s1'].fillna(0).astype(int)  # SHHS1高血压
    df['diabetes_binary'] = df['parrptdiab'].fillna(0).astype(int)  # SHHS1糖尿病史
    df['ahi_binary'] = np.where(df['nsrr_ahi_hp4u_aasm15'] >= 5, 1, 0)  # AHI（来自harmonized数据集，SHHS1基线）
    df['poor_sleep'] = ((df['ltdp10'] <= 2) & (df['rest10'] <= 2)).astype(int)  # SHHS1差睡眠质量
    df['poor_sleep'] = df['poor_sleep'].fillna(0)
    df['good_sleep'] = ((df['ltdp10'] >= 4) & (df['rest10'] >= 4)).astype(int)  # SHHS1好睡眠质量
    df['good_sleep'] = df['good_sleep'].fillna(0)
    df['insomnia'] = df['diffa10'].fillna(0).astype(int)  # SHHS1失眠（diffa10为0/1值，1表示失眠）
    
    return df

# ==========================================
# 3. 统计计算 (Cohen's d & OR w/ Bootstrap) [cite: 405-412]
# ==========================================
def run_full_stats(df_c, n_boot=500):
    stats_list = []
    # 【对齐逻辑】: 确保 subtype=1 是紊乱组 (Disturbed) [cite: 13, 263, 400]
    if df_c.groupby('subtype')['nsrr_ahi_hp4u_aasm15'].mean().idxmax() == 0:
        df_c['subtype'] = 1 - df_c['subtype']
    
    # 连续变量: Cohen's d [cite: 405]
    cont_vars = ['age_clean', 'bmi_clean']
    if 'se' in df_c.columns:
        cont_vars.append('se')
    if 'sl' in df_c.columns:
        cont_vars.append('sl')
    cont_vars += sdi_cluster_features
    print(f"计算 {len(cont_vars)} 个连续变量的Cohen's d...")
    for i, var in enumerate(cont_vars):
        if (i + 1) % 5 == 0:
            print(f"  进度: {i+1}/{len(cont_vars)}", flush=True)
        g1 = df_c[df_c['subtype'] == 1][var].dropna().values
        g0 = df_c[df_c['subtype'] == 0][var].dropna().values
        d_orig = (np.mean(g1) - np.mean(g0)) / np.sqrt(((len(g1)-1)*np.var(g1)+(len(g0)-1)*np.var(g0))/(len(g1)+len(g0)-2))
        boot_ds = []
        for _ in range(n_boot):
            b1, b0 = np.random.choice(g1, len(g1), True), np.random.choice(g0, len(g0), True)
            boot_ds.append((np.mean(b1)-np.mean(b0))/np.sqrt(((len(b1)-1)*np.var(b1)+(len(b0)-1)*np.var(b0))/(len(b1)+len(b0)-2)))
        stats_list.append({'Variable': feature_mapping.get(var, var), 'Type': 'd', 'Val': d_orig, 'Low': np.percentile(boot_ds, 2.5), 'High': np.percentile(boot_ds, 97.5)})

    # 分类变量: Adjusted Odds Ratio [cite: 411]
    cat_vars = ['sex_clean', 'ahi_binary', 'htn_binary', 'diabetes_binary', 'cvd_binary', 'good_sleep', 'poor_sleep', 'insomnia']
    for i, var in enumerate(cat_vars):
        print(f"计算 {var} 的OR ({i+1}/{len(cat_vars)})...", end=' ', flush=True)
        # 避免重复列名：如果var已经在协变量中，单独提取
        covar_cols = ['age_clean', 'bmi_clean', 'sex_clean', 'subtype']
        if var in covar_cols:
            # 如果var是协变量之一（如sex_clean），需要从协变量列表中移除
            X_cols = [c for c in covar_cols if c != var]
            temp = df_c[X_cols + [var]].dropna()
        else:
            temp = df_c[covar_cols + [var]].dropna()
        
        if len(temp) < 10:
            stats_list.append({'Variable': feature_mapping.get(var, var), 'Type': 'OR', 'Val': 1.0, 'Low': 1.0, 'High': 1.0})
            continue
        
        # 确保y是一维数组
        y = temp[var].values.ravel()
        # 构建X：如果var是协变量之一，需要重新添加subtype
        if var in covar_cols:
            X = temp[X_cols].values
        else:
            X = temp[covar_cols].values
        
        try:
            # 检查y是否包含至少两个类别
            if len(np.unique(y)) < 2:
                print(f"警告: {var}只有一个类别，跳过OR计算")
                stats_list.append({'Variable': feature_mapping.get(var, var), 'Type': 'OR', 'Val': 1.0, 'Low': 1.0, 'High': 1.0})
                continue
            
            # X_cols已经包含了subtype（如果var != subtype），所以X中已经包含subtype作为最后一个特征
            # 如果var是协变量之一，X_cols = [c for c in covar_cols if c != var]，所以subtype仍然在X_cols中（除非var='subtype'）
            # 因此，X已经包含了所有需要的特征，包括subtype作为最后一个特征
            X_fit = X  # X已经包含了所有协变量（不包括var），且subtype是最后一个特征
            
            lr = LogisticRegression(max_iter=1000, random_state=42).fit(X_fit, y)
            or_orig = np.exp(lr.coef_[0][-1])  # 最后一个系数是subtype的OR
            
            boot_ors = []
            np.random.seed(42)
            for b_idx in range(n_boot):
                idx = np.random.choice(len(temp), len(temp), True)
                try:
                    temp_b = temp.iloc[idx]
                    if var in covar_cols:
                        X_b_fit = temp_b[X_cols].values
                    else:
                        X_b_fit = temp_b[covar_cols].values
                    y_b = temp_b[var].values.ravel()
                    # 检查y_b是否包含至少两个类别
                    if len(np.unique(y_b)) < 2:
                        continue  # 如果只有一个类别，跳过这次bootstrap
                    lr_b = LogisticRegression(max_iter=1000).fit(X_b_fit, y_b)
                    boot_ors.append(np.exp(lr_b.coef_[0][-1]))
                except: 
                    continue
                # 每100次显示进度
                if (b_idx + 1) % 100 == 0:
                    print(f"({b_idx+1}/{n_boot}, valid={len(boot_ors)})", end=' ', flush=True)
            if len(boot_ors) > 0:
                stats_list.append({'Variable': feature_mapping.get(var, var), 'Type': 'OR', 'Val': or_orig, 'Low': np.percentile(boot_ors, 2.5), 'High': np.percentile(boot_ors, 97.5)})
                print(f"完成 (OR={or_orig:.3f})")
            else:
                stats_list.append({'Variable': feature_mapping.get(var, var), 'Type': 'OR', 'Val': or_orig, 'Low': or_orig*0.9, 'High': or_orig*1.1})
                print(f"完成 (OR={or_orig:.3f}, 使用近似CI)")
        except Exception as e:
            print(f"错误: {e}")
            stats_list.append({'Variable': feature_mapping.get(var, var), 'Type': 'OR', 'Val': 1.0, 'Low': 1.0, 'High': 1.0})

    return pd.DataFrame(stats_list)

# ==========================================
# 4. 绘图函数
# ==========================================
def plot_forest(df_stats):
    fig, ax1 = plt.subplots(figsize=(10, 14))
    ax2 = ax1.twiny()
    df = df_stats[::-1].reset_index(drop=True)
    y_pos = np.arange(len(df))
    
    # Cohen's d (蓝点)
    dm = df['Type'] == 'd'
    ax2.errorbar(df.loc[dm, 'Val'], y_pos[dm], xerr=[df.loc[dm, 'Val']-df.loc[dm, 'Low'], df.loc[dm, 'High']-df.loc[dm, 'Val']], fmt='o', color='#6baed6', capsize=3, label="Cohen's d")
    
    # Odds Ratio (红星)
    om = df['Type'] == 'OR'
    ax1.errorbar(df.loc[om, 'Val'], y_pos[om], xerr=[df.loc[om, 'Val']-df.loc[om, 'Low'], df.loc[om, 'High']-df.loc[om, 'Val']], fmt='*', color='#e6550d', markersize=9, capsize=3, label="Odds Ratio")
    
    ax2.set_xlim(-3, 3); ax2.set_xlabel("Cohen's d"); ax2.axvline(0, color='gray', ls='--')
    ax1.set_xlim(0, 6); ax1.set_xlabel("Odds Ratio"); ax1.axvline(1.0, color='gray', ls='--')
    ax1.set_yticks(y_pos); ax1.set_yticklabels(df['Variable'])
    ax1.text(0.15, -0.05, '← Normal Sleep', transform=ax1.transAxes, ha='center')
    ax1.text(0.85, -0.05, 'Disturbed Sleep →', transform=ax1.transAxes, ha='center')
    plt.title("SHHS Subtype Analysis", pad=30)
    plt.tight_layout()
    # 使用相对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'shhs_forest_plot_full.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"完整特征森林图已保存至: {plot_path}")
    plt.show()

# ==========================================
# 5. 执行主流程
# ==========================================
if __name__ == "__main__":
    # 使用相对路径
    base_path = r"D:\Project\Python_Project\SDI\W_NW_REM_NREM\data\shhs"
    
    print("开始加载数据...")
    df_prep = load_shhs_data(base_path)
    
    print(f"数据加载完成，样本数: {len(df_prep)}")
    print("开始聚类...")
    df_c = df_prep.dropna(subset=sdi_cluster_features).copy()
    print(f"聚类前样本数: {len(df_c)}")
    
    X = StandardScaler().fit_transform(df_c[sdi_cluster_features])
    gmm = GaussianMixture(n_components=2, random_state=42)
    df_c['subtype'] = gmm.fit_predict(X)
    
    print("开始计算统计指标...")
    final_stats = run_full_stats(df_c)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    stats_path = os.path.join(output_dir, 'shhs_forest_stats.csv')
    final_stats.to_csv(stats_path, index=False, encoding='utf-8-sig')
    print(f"统计结果已保存至: {stats_path}")
    
    print("开始绘制森林图...")
    plot_forest(final_stats)
    
    print("\n分析完成！")