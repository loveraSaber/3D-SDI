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
# 聚类使用的SDI特征（从mros_SDI_Full_Layer_Features.csv文件中提取的特征）
sdi_cluster_features = [
    "Conf_Weighted_AP",
    "Directional_Entropy",
    "Path_Length_2D",
    "Transition_Rate",
    "BB_Antagonistic_Ratio",
]

# 绘图映射表
feature_mapping = {
    'age_clean': 'Age', 'bmi_clean': 'BMI', 'se': 'SE', 'sl': 'SL',
    'AP': 'AP', 'RB': 'RB', 'MDR': 'MDR', 'PR': 'PR', 'CV': 'CV', 'SK': 'SK',
    'Conf_Weighted_AP': 'CWSD', 'SDI_Peak_Freq': 'SDI Peak Freq',
    'SDI_Spec_Centroid': 'SDI Spec Centroid', 'BB_Phase_Lag': 'BB Phase Lag',
    'Spatial_Entropy': 'Spatial Entropy', 'Directional_Entropy': 'STDE',
    '3D_Mean_Curvature': '3D Mean Curvature', 'Recurrence_Rate': 'Recurrence Rate',
    '3D_Convex_Hull_Ratio': '3D Convex Hull Ratio', 'Fractal_Dimension': 'Fractal Dimension',
    'Path_Length_2D': 'TDTL', 'Transition_Rate': 'SSTR',
    'SDI_Collapse_Accel': 'SDI Collapse Accel', 'BB_Antagonistic_Ratio': 'BBAI',
    'Transition_Vector_Entropy': 'Transition Vector Entropy',
    'sex_clean': 'Sex', 'ahi_binary': 'OSA', 
    'htn_binary': 'HTN', 'diabetes_binary': 'Diabetes',
    'good_sleep': 'Good Sleep', 'poor_sleep': 'Poor Sleep', 'insomnia': 'Insomnia'
}

# ==========================================
# 2. 数据加载与清理
# ==========================================
def load_mros_data(base_path):
    # 加载MRoS Visit 1基线数据
    df_baseline = pd.read_csv(os.path.join(base_path, 'mros-visit1-dataset-0.6.0.csv'), low_memory=False)
    
    # 加载SDI Full Layer特征文件
    df_features = pd.read_csv(os.path.join(base_path, 'mros_SDI_Full_Layer_Features.csv'))
    
    # 从文件名中提取ID (格式: mros-visit1-aa1898_features_predictions.csv -> aa1898)
    def extract_id_from_filename(filename):
        """从文件名中提取nsrrid"""
        if pd.isna(filename):
            return None
        # 匹配模式: mros-visit1-aaXXXX
        match = re.search(r'mros-visit1-([a-z]{2}\d+)', str(filename).lower())
        if match:
            return match.group(1)
        return None
    
    # 提取ID列
    if 'NSRRID' in df_features.columns:
        df_features['nsrrid'] = df_features['NSRRID'].apply(extract_id_from_filename)
    elif 'filename' in df_features.columns:
        df_features['nsrrid'] = df_features['filename'].apply(extract_id_from_filename)
    else:
        # 尝试直接使用 filename 如果没有 NSRRID
        df_features['nsrrid'] = df_features['filename'].apply(extract_id_from_filename)

    df_features = df_features.dropna(subset=['nsrrid'])
    
    # 将基线数据中的nsrrid转换为小写以确保匹配
    df_baseline['nsrrid'] = df_baseline['nsrrid'].astype(str).str.lower()
    
    # 合并数据
    df = pd.merge(df_baseline, df_features, on='nsrrid', how='inner')

    # 处理合并产生的重复列
    def fix_col(df, name):
        if f"{name}_x" in df.columns: 
            return df[f"{name}_x"].fillna(df.get(f"{name}_y", pd.Series([np.nan]*len(df))))
        if f"{name}_y" in df.columns:
            return df[f"{name}_y"]
        return df.get(name, pd.Series([np.nan]*len(df)))

    # MRoS基线变量
    # 年龄：vsage1
    if 'vsage1' in df.columns:
        df['age_clean'] = pd.to_numeric(df['vsage1'], errors='coerce')
    else:
        raise ValueError("找不到年龄列 vsage1")
    
    # BMI：hwbmi
    if 'hwbmi' in df.columns:
        df['bmi_clean'] = pd.to_numeric(df['hwbmi'], errors='coerce')
    else:
        raise ValueError("找不到BMI列 hwbmi")

    # SE/SL：优先使用 PSG 列，找不到时回退问卷列
    se_candidates = ['poslpeff', 'pqpeffcy', 'pqpeffic']
    sl_candidates = ['posllatp', 'pqplaten']
    se_col = next((c for c in se_candidates if c in df.columns), None)
    sl_col = next((c for c in sl_candidates if c in df.columns), None)
    if se_col is None:
        print("警告: 找不到SE列 (poslpeff/pqpeffcy/pqpeffic)")
        df['se'] = np.nan
    else:
        df['se'] = pd.to_numeric(df[se_col], errors='coerce')
    if sl_col is None:
        print("警告: 找不到SL列 (posllatp/pqplaten)")
        df['sl'] = np.nan
    else:
        df['sl'] = pd.to_numeric(df[sl_col], errors='coerce')
    
    # 性别编码：1=男性，0=女性
    # 注意：MROS是男性队列研究，gender列值为2表示男性
    if 'gender' in df.columns:
        gender_numeric = pd.to_numeric(df['gender'], errors='coerce')
        df['sex_clean'] = np.where(gender_numeric == 2, 1, 0)
    elif 'sex' in df.columns:
        sex_numeric = pd.to_numeric(df['sex'], errors='coerce')
        df['sex_clean'] = np.where(sex_numeric == 1, 1, 0)
    else:
        # MROS全是男性，如果没有性别列，可以设为1
        print("警告: 找不到性别列，默认设为男性(1)")
        df['sex_clean'] = 1
    
    # 定义结局与疾病变量
    # 高血压：mhbp (1=是, 0=否, 可能包含其他值如'A'等)
    if 'mhbp' in df.columns:
        df['htn_binary'] = pd.to_numeric(df['mhbp'], errors='coerce').fillna(0)
        df['htn_binary'] = np.where(df['htn_binary'] == 1, 1, 0).astype(int)
    else:
        print("警告: 找不到高血压列 mhbp，将htn_binary设为0")
        df['htn_binary'] = 0
    
    # 糖尿病：mhdiab (1=是, 0=否, 可能包含其他值如'A'等)
    if 'mhdiab' in df.columns:
        df['diabetes_binary'] = pd.to_numeric(df['mhdiab'], errors='coerce').fillna(0)
        df['diabetes_binary'] = np.where(df['diabetes_binary'] == 1, 1, 0).astype(int)
    else:
        print("警告: 找不到糖尿病列 mhdiab，将diabetes_binary设为0")
        df['diabetes_binary'] = 0
    
    # OSA: 尝试使用nsrr_ahi_hp4u_aasm15，如果不存在则使用poahi3或poahi4
    ahi_col = None
    if 'nsrr_ahi_hp4u_aasm15' in df.columns:
        ahi_col = 'nsrr_ahi_hp4u_aasm15'
    elif 'poahi3' in df.columns:
        ahi_col = 'poahi3'
    elif 'poahi4' in df.columns:
        ahi_col = 'poahi4'
    elif 'pooahi3' in df.columns:
        ahi_col = 'pooahi3'
    elif 'pooahi4' in df.columns:
        ahi_col = 'pooahi4'
    
    if ahi_col:
        df['ahi_binary'] = np.where(pd.to_numeric(df[ahi_col], errors='coerce') >= 5, 1, 0)
        df['ahi_value'] = pd.to_numeric(df[ahi_col], errors='coerce').fillna(0)
    else:
        print("警告: 找不到AHI列，将ahi_binary设为0")
        df['ahi_binary'] = 0
        df['ahi_value'] = 0
    
    # 差睡眠质量：poxqual1 ≤ 2 AND poxqual3 ≤ 2
    # 将字符串转换为数值类型
    df['poxqual1'] = pd.to_numeric(df['poxqual1'], errors='coerce')
    df['poxqual3'] = pd.to_numeric(df['poxqual3'], errors='coerce')
    df['poxfall'] = pd.to_numeric(df['poxfall'], errors='coerce')
    
    df['poor_sleep'] = ((df['poxqual1'] <= 2) & (df['poxqual3'] <= 2)).astype(int)
    df['poor_sleep'] = df['poor_sleep'].fillna(0)
    
    # 好睡眠质量：poxqual1 ≥ 4 AND poxqual3 ≥ 4
    df['good_sleep'] = ((df['poxqual1'] >= 4) & (df['poxqual3'] >= 4)).astype(int)
    df['good_sleep'] = df['good_sleep'].fillna(0)
    
    # 失眠：poxfall > 40
    df['insomnia'] = np.where(df['poxfall'] > 40, 1, 0)
    df['insomnia'] = df['insomnia'].fillna(0)
    
    return df

# ==========================================
# 3. 统计计算 (Cohen's d & OR w/ Bootstrap)
# ==========================================
def run_full_stats(df_c, n_boot=500):
    stats_list = []
    # 【对齐逻辑】: 确保 subtype=1 是紊乱组 (Disturbed)
    # 使用AHI值进行对齐
    if 'ahi_value' in df_c.columns:
        if df_c.groupby('subtype')['ahi_value'].mean().idxmax() == 0:
            df_c['subtype'] = 1 - df_c['subtype']
    else:
        # 如果没有AHI值，使用睡眠质量得分对齐
        # 使用SDI_Full_Layer_Features中的特征进行对齐
        # 选择与睡眠质量相关的特征
        features_for_align = ['SDI_Collapse_Accel', 'Transition_Rate', 'Path_Length_2D', 'Bout_N3_Mean_Dur']
        # 检查对齐特征是否存在
        available_align_features = [f for f in features_for_align if f in df_c.columns]
        if len(available_align_features) == 0:
            # 如果都不存在，使用前几个聚类特征
            available_align_features = [f for f in sdi_cluster_features if f in df_c.columns][:4]
        
        if len(available_align_features) >= 2:
            scaler_align = StandardScaler()
            align_data = df_c[available_align_features].fillna(df_c[available_align_features].mean())
            align_data_scaled = scaler_align.fit_transform(align_data)
            # 使用前3个特征加，最后一个特征减（如果存在）
            if len(available_align_features) >= 4:
                df_c['sleep_quality_score'] = (
                    align_data_scaled[:, 0] + 
                    align_data_scaled[:, 1] + 
                    align_data_scaled[:, 2] - 
                    align_data_scaled[:, 3]
                )
            else:
                df_c['sleep_quality_score'] = np.sum(align_data_scaled, axis=1)
            
            if df_c.groupby('subtype')['sleep_quality_score'].mean().idxmax() == 0:
                df_c['subtype'] = 1 - df_c['subtype']
    
    # 连续变量: Cohen's d
    cont_vars = ['age_clean', 'bmi_clean']
    if 'se' in df_c.columns:
        cont_vars.append('se')
    if 'sl' in df_c.columns:
        cont_vars.append('sl')
    
    # 添加SDI特征
    sdi_order = [
        "BB_Antagonistic_Ratio",
        "Transition_Rate",
        "Path_Length_2D",
        "Directional_Entropy",
        "Conf_Weighted_AP"
    ]
    for feat in sdi_order:
        if feat in df_c.columns:
            cont_vars.append(feat)
    
    print(f"可用的SDI特征: {[f for f in sdi_order if f in df_c.columns]}")
    print(f"计算 {len(cont_vars)} 个连续变量的Cohen's d...")
    for i, var in enumerate(cont_vars):
        if (i + 1) % 5 == 0:
            print(f"  进度: {i+1}/{len(cont_vars)}", flush=True)
        # 确保转换为数值类型
        g1 = pd.to_numeric(df_c[df_c['subtype'] == 1][var], errors='coerce').dropna().values
        g0 = pd.to_numeric(df_c[df_c['subtype'] == 0][var], errors='coerce').dropna().values
        
        if len(g1) < 2 or len(g0) < 2:
            stats_list.append({'Variable': feature_mapping.get(var, var), 'Type': 'd', 'Val': 0.0, 'Low': 0.0, 'High': 0.0})
            continue
        
        # 计算Cohen's d
        mean1, mean0 = np.mean(g1), np.mean(g0)
        var1, var0 = np.var(g1, ddof=1), np.var(g0, ddof=1)
        n1, n0 = len(g1), len(g0)
        pooled_std = np.sqrt(((n1-1)*var1 + (n0-1)*var0) / (n1+n0-2))
        if pooled_std == 0:
            d_orig = 0.0
        else:
            d_orig = (mean1 - mean0) / pooled_std
        
        # Bootstrap
        boot_ds = []
        np.random.seed(42)
        for _ in range(n_boot):
            b1 = np.random.choice(g1, len(g1), True)
            b0 = np.random.choice(g0, len(g0), True)
            m1, m0 = np.mean(b1), np.mean(b0)
            v1, v0 = np.var(b1, ddof=1), np.var(b0, ddof=1)
            pstd = np.sqrt(((len(b1)-1)*v1 + (len(b0)-1)*v0) / (len(b1)+len(b0)-2))
            if pstd > 0:
                boot_ds.append((m1 - m0) / pstd)
        
        if len(boot_ds) > 0:
            stats_list.append({
                'Variable': feature_mapping.get(var, var), 
                'Type': 'd', 
                'Val': d_orig, 
                'Low': np.percentile(boot_ds, 2.5), 
                'High': np.percentile(boot_ds, 97.5)
            })
        else:
            stats_list.append({
                'Variable': feature_mapping.get(var, var), 
                'Type': 'd', 
                'Val': d_orig, 
                'Low': d_orig*0.9, 
                'High': d_orig*1.1
            })

    # 分类变量: Adjusted Odds Ratio
    cat_vars = ['sex_clean', 'ahi_binary', 'htn_binary', 'diabetes_binary',  'poor_sleep', 'insomnia']
    for i, var in enumerate(cat_vars):
        print(f"计算 {feature_mapping.get(var, var)} 的OR ({i+1}/{len(cat_vars)})... ", end='', flush=True)
        
        # 避免重复列名：如果var已经在协变量中，单独提取
        covar_cols = ['age_clean', 'bmi_clean', 'sex_clean', 'subtype']
        if var in covar_cols:
            X_cols = [c for c in covar_cols if c != var]
            temp = df_c[X_cols + [var]].copy()
        else:
            temp = df_c[covar_cols + [var]].copy()
        
        # 确保所有列都是数值类型
        for col in temp.columns:
            temp[col] = pd.to_numeric(temp[col], errors='coerce')
        
        # 删除包含 NaN 的行
        temp = temp.dropna()
        
        if len(temp) < 10:
            stats_list.append({'Variable': feature_mapping.get(var, var), 'Type': 'OR', 'Val': 1.0, 'Low': 1.0, 'High': 1.0})
            print(f"跳过 (样本数不足)")
            continue
        
        y = temp[var].values.ravel()
        X = temp[X_cols].values if var in covar_cols else temp[covar_cols].values
        
        # 检查y是否包含至少两个类别
        if len(np.unique(y)) < 2:
            stats_list.append({'Variable': feature_mapping.get(var, var), 'Type': 'OR', 'Val': 1.0, 'Low': 1.0, 'High': 1.0})
            print(f"跳过 (y只包含一个类别: {np.unique(y)})")
            continue

        try:
            lr = LogisticRegression(max_iter=1000, random_state=42).fit(X, y)
            or_orig = np.exp(lr.coef_[0][-1])
            
            boot_ors = []
            np.random.seed(42)
            for b_idx in range(n_boot):
                idx = np.random.choice(len(temp), len(temp), True)
                temp_b = temp.iloc[idx]
                X_b_fit = temp_b[X_cols].values if var in covar_cols else temp_b[covar_cols].values
                y_b = temp_b[var].values.ravel()
                
                # 检查bootstrap样本的y_b是否包含至少两个类别
                if len(np.unique(y_b)) < 2:
                    continue # 跳过此次bootstrap迭代
                
                try:
                    lr_b = LogisticRegression(max_iter=1000).fit(X_b_fit, y_b)
                    boot_ors.append(np.exp(lr_b.coef_[0][-1]))
                except Exception as e_inner: 
                    continue
                
                if (b_idx + 1) % 100 == 0:
                    print(f"({b_idx+1}/{n_boot}, valid={len(boot_ors)})", end=' ', flush=True)
            
            if len(boot_ors) > 0:
                stats_list.append({
                    'Variable': feature_mapping.get(var, var), 
                    'Type': 'OR', 
                    'Val': or_orig, 
                    'Low': np.percentile(boot_ors, 2.5), 
                    'High': np.percentile(boot_ors, 97.5)
                })
                print(f"完成 (OR={or_orig:.3f})")
            else:
                stats_list.append({
                    'Variable': feature_mapping.get(var, var), 
                    'Type': 'OR', 
                    'Val': or_orig, 
                    'Low': or_orig*0.9, 
                    'High': or_orig*1.1
                })
                print(f"完成 (OR={or_orig:.3f}, 使用近似CI)")
        except Exception as e:
            print(f"错误: {e}")
            stats_list.append({'Variable': feature_mapping.get(var, var), 'Type': 'OR', 'Val': 1.0, 'Low': 1.0, 'High': 1.0})

    return pd.DataFrame(stats_list)

# ==========================================
# 4. 绘图函数
# ==========================================
def plot_forest(df_stats):
    # 压缩图形尺寸（宽度和高度）
    fig, ax1 = plt.subplots(figsize=(8, 10))
    ax2 = ax1.twiny()
    
    # 创建分组标签函数
    def create_grouped_labels(df):
        labels = []
        # 从下到上的顺序：疾病在下，特征在上
        
        # 疾病部分
        # 注意：MROS是男性队列，无Sex
        outcome_order = ['HTN', 'CVD', 'Diabetes', 'Insomnia', 'Poor Sleep', 'Good Sleep', 'OSA']  # 从下到上的顺序
        outcomes_found = []
        for var in outcome_order:
            if var in df['Variable'].values:
                outcomes_found.append(var)
        
        if len(outcomes_found) > 0:
            for var in outcomes_found:
                labels.append('  ' + var)  # 缩进子项
            labels.append('Outcomes')  # 分组标题
        
        # SDI Features
        sdi_features_display = [
            'BBAI',
            'SSTR',
            'TDTL',
            'STDE',
            'CWSD'
        ]
        sdi_found = []
        for var in sdi_features_display:
            if var in df['Variable'].values:
                sdi_found.append(var)
        
        if len(sdi_found) > 0:
            for var in sdi_found:
                labels.append('  ' + var)  # 缩进子项
            labels.append('SDI Features')  # 分组标题
        
        # SL, SE, BMI, Age
        for var in ['SL', 'SE', 'BMI', 'Age']:
            if var in df['Variable'].values:
                labels.append(var)
        
        return labels
    
    # 重新组织数据
    ordered_vars = []
    y_positions = []
    current_pos = 0
    
    # ========== 疾病部分（在下方，先添加）==========
    or_df = df_stats[df_stats['Type'] == 'OR'].copy()
    # MROS无Sex
    outcome_order = ['HTN', 'CVD', 'Diabetes', 'Insomnia', 'Poor Sleep', 'Good Sleep', 'OSA'] # 从下到上
    outcomes_found = []
    for var in outcome_order:
        if var in or_df['Variable'].values:
            idx = or_df[or_df['Variable'] == var].index[0]
            outcomes_found.append(or_df.loc[idx])
    
    if len(outcomes_found) > 0:
        # 先添加Outcomes的数据
        for var_row in outcomes_found:
            ordered_vars.append(var_row)
            y_positions.append(current_pos)
            current_pos += 1
        # 添加分组标题行（空数据）
        ordered_vars.append(pd.Series({'Variable': 'Outcomes', 'Type': 'OR', 'Val': np.nan, 'Low': np.nan, 'High': np.nan}))
        y_positions.append(current_pos)
        current_pos += 1
    
    # ========== 特征部分（在上方，后添加）==========
    d_df = df_stats[df_stats['Type'] == 'd'].copy()
    
    # SDI Features（按照从下到上的顺序）
    sdi_features_display = [
        'BBAI',
        'SSTR',
        'TDTL',
        'STDE',
        'CWSD'
    ]
    sdi_found = []
    for var in sdi_features_display:
        if var in d_df['Variable'].values:
            idx = d_df[d_df['Variable'] == var].index[0]
            sdi_found.append(d_df.loc[idx])
    
    if len(sdi_found) > 0:
        # 添加SDI特征
        for var_row in sdi_found:
            ordered_vars.append(var_row)
            y_positions.append(current_pos)
            current_pos += 1
        # 添加分组标题行（空数据）
        ordered_vars.append(pd.Series({'Variable': 'SDI Features', 'Type': 'd', 'Val': np.nan, 'Low': np.nan, 'High': np.nan}))
        y_positions.append(current_pos)
        current_pos += 1
    
    # SL, SE, BMI, Age（从下到上的顺序）
    for var in ['SL', 'SE', 'BMI', 'Age']:
        if var in d_df['Variable'].values:
            idx = d_df[d_df['Variable'] == var].index[0]
            ordered_vars.append(d_df.loc[idx])
            y_positions.append(current_pos)
            current_pos += 1
    
    # 创建新的DataFrame
    df_ordered = pd.DataFrame(ordered_vars).reset_index(drop=True)
    y_pos = np.arange(len(df_ordered))
    
    # OR (橙星) - 在下方，压缩标记尺寸
    om = (df_ordered['Type'] == 'OR') & (~df_ordered['Variable'].isin(['SDI Features', 'Outcomes']))
    if om.sum() > 0:
        ax1.errorbar(df_ordered.loc[om, 'Val'], y_pos[om],
                     xerr=[df_ordered.loc[om, 'Val'] - df_ordered.loc[om, 'Low'],
                           df_ordered.loc[om, 'High'] - df_ordered.loc[om, 'Val']],
                     fmt='*', color='orange', capsize=2.5, capthick=1.2, markersize=8, label='Odds Ratio')
    
    # Cohen's d (蓝点) - 在上方，压缩标记尺寸
    dm = (df_ordered['Type'] == 'd') & (~df_ordered['Variable'].isin(['SDI Features', 'Outcomes']))
    if dm.sum() > 0:
        ax2.errorbar(df_ordered.loc[dm, 'Val'], y_pos[dm], 
                     xerr=[df_ordered.loc[dm, 'Val'] - df_ordered.loc[dm, 'Low'],
                           df_ordered.loc[dm, 'High'] - df_ordered.loc[dm, 'Val']],
                     fmt='o', color='lightblue', capsize=2.5, capthick=1.2, markersize=6, label='Cohen\'s d')
    
    # 设置X轴 - 压缩范围，字体14
    ax1.set_xlabel("Odds Ratio", fontsize=14, fontweight='bold')
    ax1.set_xlim(-1, 5)  # 压缩横坐标范围
    ax1.tick_params(axis='x', labelsize=14)
    
    ax2.set_xlabel("Cohen's d", fontsize=14, fontweight='bold')
    ax2.set_xlim(-3, 3)  # 压缩横坐标范围
    ax2.tick_params(axis='x', labelsize=14)
    
    # 设置Y轴标签（带缩进）- 字体14
    y_labels = []
    # 所有的子项标签
    sdi_sub_items = [
        'BBAI',
        'SSTR',
        'TDTL',
        'STDE',
        'CWSD'
    ]
    
    for idx, var in enumerate(df_ordered['Variable']):
        if var in ['SDI Features', 'Outcomes']:
            y_labels.append(var)  # 分组标题不加缩进，使用粗体
        elif var in sdi_sub_items:
            y_labels.append('  ' + var)  # SDI特征缩进
        elif var in outcome_order:
            y_labels.append('  ' + var)  # Outcomes缩进
        else:
            y_labels.append(var)  # 其他变量不缩进
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(y_labels, fontsize=14)
    ax1.set_ylabel('')
    
    # 为分组标题设置粗体
    for idx, label in enumerate(ax1.get_yticklabels()):
        if label.get_text() in ['SDI Features', 'Outcomes']:
            label.set_fontweight('bold')
    
    # 找到分隔点并绘制水平分隔线（疾病和特征之间）
    # 找到Outcomes标题的位置
    outcomes_idx = df_ordered[df_ordered['Variable'] == 'Outcomes'].index
    
    if len(outcomes_idx) > 0:
        # 在Outcomes标题位置绘制实线（对准outcomes行）
        separator_y = y_pos[outcomes_idx[0]]
        ax1.axhline(separator_y, color='black', linestyle='-', linewidth=2, alpha=0.9)
        ax2.axhline(separator_y, color='black', linestyle='-', linewidth=2, alpha=0.9)
        
        # 绘制垂直虚线，精确到分界线位置
        # 计算y轴的实际范围
        y_min_plot = ax1.get_ylim()[0]
        y_max_plot = ax1.get_ylim()[1]
        
        # 计算分界线在整个y轴范围中的精确位置（数据坐标）
        if y_max_plot > y_min_plot:
            separator_ratio = (separator_y - y_min_plot) / (y_max_plot - y_min_plot)
        else:
            separator_ratio = 0.5
        
        # OR 的1线（下方区域）- 从底部到分界线
        ax1.axvline(1, color='gray', linestyle='--', linewidth=1.2, alpha=0.7, 
                   ymin=0, ymax=separator_ratio)
        
        # Cohen's d 的0线（上方区域）- 从分界线到顶部
        ax2.axvline(0, color='gray', linestyle='--', linewidth=1.2, alpha=0.7, 
                   ymin=separator_ratio, ymax=1)
    else:
        # 如果没有分界线，则绘制完整的虚线
        ax1.axvline(1, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)
        ax2.axvline(0, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)
    
    # 底部指示箭头（左指向左，右指向右）- 字体14
    # 左侧箭头：指向左边（Normal Sleep，OR < 1）
    ax1.annotate('', xy=(0.05, -0.05), xycoords='axes fraction', xytext=(0.30, -0.05),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax1.text(0.175, -0.08, 'Normal Sleep', transform=ax1.transAxes, ha='center', fontsize=14, fontweight='bold')
    
    # 右侧箭头：指向右边（Disturbed Sleep，OR > 1）
    ax1.annotate('', xy=(0.95, -0.05), xycoords='axes fraction', xytext=(0.70, -0.05),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax1.text(0.825, -0.08, 'Disturbed Sleep', transform=ax1.transAxes, ha='center', fontsize=14, fontweight='bold')
    
    plt.title("MROS", fontsize=16, fontweight='bold', pad=20)
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax1.grid(False)
    ax2.grid(False)
    
    # 调整子图布局，减小边距以压缩空间
    plt.subplots_adjust(left=0.20, right=0.95, top=0.95, bottom=0.10)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'mros_forest_plot_full.png')
    plt.savefig(output_path, dpi=800, bbox_inches='tight')  # DPI设置为800
    print(f"\n森林图已保存至: {output_path}")
    plt.close()

# ==========================================
# 5. 执行主流程
# ==========================================
if __name__ == "__main__":
    # 设置中文字体（如果需要）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 使用相对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(script_dir, '..', 'data', 'mros')
    base_path = os.path.abspath(base_path)
    
    print("开始加载数据...")
    df_prep = load_mros_data(base_path)
    
    print(f"数据加载完成，样本数: {len(df_prep)}")
    print("开始聚类...")
    df_c = df_prep.dropna(subset=sdi_cluster_features).copy()
    print(f"聚类前样本数: {len(df_c)}")
    
    X = StandardScaler().fit_transform(df_c[sdi_cluster_features])
    gmm = GaussianMixture(n_components=2, random_state=42)
    df_c['subtype'] = gmm.fit_predict(X)
    
    print(f"正常睡眠亚型 (subtype=0): {len(df_c[df_c['subtype']==0])}")
    print(f"紊乱睡眠亚型 (subtype=1): {len(df_c[df_c['subtype']==1])}")
    
    print("开始计算统计指标...")
    final_stats = run_full_stats(df_c)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    stats_path = os.path.join(output_dir, 'mros_forest_stats.csv')
    final_stats.to_csv(stats_path, index=False, encoding='utf-8-sig')
    print(f"统计结果已保存至: {stats_path}")
    
    print("开始绘制森林图...")
    plot_forest(final_stats)
    
    print("\n分析完成！")
