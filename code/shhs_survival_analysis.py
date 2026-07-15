import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CSV 读取：兼容非 UTF-8 编码
# ==========================================
def read_csv_robust(path, **kwargs):
    """
    优先用 utf-8 读取；失败则自动回退到常见编码，避免 UnicodeDecodeError。
    """
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1", "gb18030"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError as e:
            last_err = e
            continue
    raise last_err

# 导入生存分析库
try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
    from lifelines.statistics import logrank_test
    HAS_LIFELINES = True
except ImportError:
    print("警告: 需要安装lifelines库: pip install lifelines")
    HAS_LIFELINES = False
    exit(1)

# ==========================================
# 1. 核心定义（与聚类代码保持一致）
# ==========================================
# 聚类使用的SDI特征（4个Full Layer特征）
sdi_cluster_features = [
    'Conf_Weighted_AP',
    'Path_Length_2D',
    'Transition_Rate',
    'BB_Antagonistic_Ratio'
]

# ==========================================
# 2. 数据加载与清理
# ==========================================
def load_shhs_data(base_path):
    """加载SHHS数据，与聚类代码保持一致"""
    df_baseline = pd.read_csv(os.path.join(base_path, 'shhs1-dataset-0.20.0.csv'), low_memory=False)
    df_cvd = pd.read_csv(os.path.join(base_path, 'shhs-cvd-summary-dataset-0.19.0.csv'), low_memory=False)
    df_harmonized = read_csv_robust(os.path.join(base_path, 'shhs-harmonized-dataset-0.20.0.csv'))
    
    # 加载SDI Full Layer特征文件
    df_features = pd.read_csv(os.path.join(base_path, 'SDI_Full_Layer_Features.csv'))
    
    # 从文件名中提取ID（只保留SHHS1数据）
    def extract_id_from_filename(filename):
        if pd.isna(filename):
            return None
        filename_str = str(filename).lower()
        # 排除SHHS2数据
        if "shhs2" in filename_str:
            return None
        match = re.search(r'shhs1-(\d+)', filename_str)
        if match:
            return match.group(1)
        return None
    
    df_features['nsrrid'] = df_features['NSRRID'].apply(extract_id_from_filename)
    df_features = df_features.dropna(subset=['nsrrid'])
    
    # 统一ID格式
    df_baseline['nsrrid'] = df_baseline['nsrrid'].astype(str)
    df_cvd['nsrrid'] = df_cvd['nsrrid'].astype(str)
    df_features['nsrrid'] = df_features['nsrrid'].astype(str)
    
    # 逐步合并
    df = pd.merge(df_baseline, df_cvd, on='nsrrid', how='inner')
    df_harmonized_v1 = df_harmonized[df_harmonized['visitnumber'] == 1][['nsrrid', 'nsrr_ahi_hp4u_aasm15']].copy()
    df_harmonized_v1['nsrrid'] = df_harmonized_v1['nsrrid'].astype(str)
    df = pd.merge(df, df_harmonized_v1, on='nsrrid', how='inner')
    df = pd.merge(df, df_features, on='nsrrid', how='inner')
    
    # 处理重复列
    def fix_col(df, name):
        if f"{name}_x" in df.columns:
            return df[f"{name}_x"].fillna(df[f"{name}_y"])
        return df[name]
    
    # 基线变量
    df['age_clean'] = fix_col(df, 'age_s1')
    df['bmi_clean'] = fix_col(df, 'bmi_s1')
    df['sex_clean'] = np.where(df['gender_x'] == 1, 1, 0)
    
    # 疾病变量
    df['htn_binary'] = df['htnderv_s1'].fillna(0).astype(int)
    df['diabetes_binary'] = df['parrptdiab'].fillna(0).astype(int)
    df['ahi_value'] = df['nsrr_ahi_hp4u_aasm15'].fillna(0)
    
    # 种族信息（从baseline数据中获取，1=White, 2=Black, 3=Other）
    if 'race' in df.columns:
        df['race_clean'] = df['race'].fillna(0).astype(int)
        print(f"种族分布: White(1)={len(df[df['race_clean']==1])}, Black(2)={len(df[df['race_clean']==2])}, Other(3)={len(df[df['race_clean']==3])}")
    else:
        # 尝试从fix_col获取
        try:
            df['race_clean'] = fix_col(df, 'race').fillna(0).astype(int)
            print(f"种族分布（从重复列获取）: White(1)={len(df[df['race_clean']==1])}, Black(2)={len(df[df['race_clean']==2])}, Other(3)={len(df[df['race_clean']==3])}")
        except:
            df['race_clean'] = 0
            print("警告: 未找到race列，模型1中将不包含种族")
    
    # 吸烟史（尝试从baseline数据中查找相关列）
    smoking_cols = [col for col in df.columns if 'smok' in col.lower() or 'pack' in col.lower() or 'cigar' in col.lower()]
    if smoking_cols:
        # 使用第一个找到的吸烟相关列
        df['smoking_binary'] = df[smoking_cols[0]].fillna(0).astype(int)
        print(f"使用 {smoking_cols[0]} 作为吸烟史变量")
    else:
        df['smoking_binary'] = 0
        print("警告: 未找到吸烟史列，模型2中将不包含吸烟史")
    
    # TST总睡眠时间计算
    # 方法1: 从睡眠阶段时长计算（timest1=N1, timest2=N2, timest34=N3, timerem=REM）
    # 根据用户说明，这些列相加即为总睡眠时间
    sleep_stage_cols = ['timest1', 'timest2', 'timest34', 'timerem']
    available_sleep_stage_cols = [col for col in sleep_stage_cols if col in df.columns]
    
    if len(available_sleep_stage_cols) == 4:
        # 所有睡眠阶段列都存在，直接相加计算TST
        df['tst_value'] = df[available_sleep_stage_cols].sum(axis=1)
        print(f"从睡眠阶段列计算TST: {', '.join(available_sleep_stage_cols)} 相加")
        print(f"  (timest1+N2+timest34+timerem = 总睡眠时长)")
    elif len(available_sleep_stage_cols) > 0:
        # 部分列存在，使用现有列相加
        df['tst_value'] = df[available_sleep_stage_cols].sum(axis=1)
        print(f"从睡眠阶段列计算TST（部分列）: {', '.join(available_sleep_stage_cols)} 相加")
        print(f"  缺失的列: {set(sleep_stage_cols) - set(available_sleep_stage_cols)}")
    else:
        # 方法2: 尝试旧的方法（nsrr_pctdursp系列）
        old_sleep_stage_cols = ['nsrr_pctdursp_s1', 'nsrr_pctdursp_s2', 'nsrr_pctdursp_s3', 'nsrr_pctdursp_sr']
        old_available_cols = [col for col in old_sleep_stage_cols if col in df.columns]
        if len(old_available_cols) == 4:
            df['tst_value'] = df[old_available_cols].sum(axis=1)
            print(f"从旧睡眠阶段列计算TST: {', '.join(old_available_cols)} 相加")
        else:
            # 方法3: 从harmonized数据集中查找TST列
            tst_cols = [col for col in df_harmonized.columns if 'tst' in col.lower() or ('total' in col.lower() and 'sleep' in col.lower() and 'time' in col.lower())]
            if tst_cols:
                df_harmonized_v1_tst = df_harmonized[df_harmonized['visitnumber'] == 1][['nsrrid'] + tst_cols[:1]].copy()
                df_harmonized_v1_tst['nsrrid'] = df_harmonized_v1_tst['nsrrid'].astype(str)
                df = pd.merge(df, df_harmonized_v1_tst, on='nsrrid', how='left')
                df['tst_value'] = df[tst_cols[0]].fillna(df[tst_cols[0]].median())
                print(f"使用 {tst_cols[0]} 作为TST变量（来自harmonized数据集）")
            else:
                df['tst_value'] = 0
                print("警告: 未找到TST相关列（timest1/timest2/timest34/timerem），模型3中将不包含TST")
    
    # 处理缺失值：用中位数填充
    if df['tst_value'].isna().any() or df['tst_value'].isnull().any():
        tst_median = df['tst_value'].median()
        if pd.isna(tst_median) or tst_median == 0:
            tst_median = df[df['tst_value'] > 0]['tst_value'].median()
        if not pd.isna(tst_median) and tst_median > 0:
            df['tst_value'] = df['tst_value'].fillna(tst_median)
        else:
            df['tst_value'] = df['tst_value'].fillna(0)
    
    # 输出TST统计信息
    tst_valid = df[df['tst_value'] > 0]['tst_value']
    if len(tst_valid) > 0:
        print(f"TST统计: 中位数={tst_valid.median():.1f} 分钟, 范围=[{tst_valid.min():.1f}, {tst_valid.max():.1f}] 分钟")
    else:
        print("警告: TST值为0或缺失，模型3中将无法使用TST")
    
    # 生存分析变量
    # vital: 1=存活, 0=死亡（根据原始数据）
    # 转换为生存分析常用格式：event=1表示发生事件（死亡）
    df['event_allcause'] = (1 - df['vital']).astype(int)  # 全因死亡事件
    df['event_chd'] = df['chd_death'].fillna(0).astype(int)  # CHD死亡事件
    df['event_cvd'] = df['cvd_death'].fillna(0).astype(int)  # CVD死亡事件
    
    # 生存时间（从censdate获取，单位可能是天数，转换为年）
    # 检查censdate的分布
    print(f"\n生存时间数据检查:")
    print(f"  censdate统计: 最小值={df['censdate'].min()}, 最大值={df['censdate'].max()}, 中位数={df['censdate'].median()}")
    print(f"  vital分布: 存活(vital=1)={len(df[df['vital']==1])}, 死亡(vital=0)={len(df[df['vital']==0])}")
    print(f"  事件分布: 全因死亡事件={df['event_allcause'].sum()}, CHD死亡事件={df['event_chd'].sum()}, CVD死亡事件={df['event_cvd'].sum()}")
    
    # 如果censdate的值看起来很大（>10000），可能是天数；如果较小（<100），可能是年
    if df['censdate'].max() > 1000:
        # 假设是天数，转换为年
        df['time_years'] = df['censdate'] / 365.25
        print(f"  将censdate视为天数，转换为年")
    else:
        # 假设已经是年
        df['time_years'] = df['censdate']
        print(f"  将censdate视为年（未转换）")
    
    print(f"  转换后生存时间: 最小值={df['time_years'].min():.2f}年, 最大值={df['time_years'].max():.2f}年, 中位数={df['time_years'].median():.2f}年")
    
    # 检查是否有异常值（生存时间过长或过短）
    if df['time_years'].max() > 50:
        print(f"  警告: 最大生存时间超过50年，可能存在数据错误")
    if df['time_years'].min() < 0:
        print(f"  警告: 存在负的生存时间，可能存在数据错误")
        df = df[df['time_years'] >= 0].copy()
    
    return df

# ==========================================
# 3. 聚类分析（与原始聚类代码一致）
# ==========================================
def perform_clustering(df):
    """执行聚类分析"""
    df_cluster = df.dropna(subset=sdi_cluster_features).copy()
    print(f"聚类前样本数: {len(df_cluster)}")
    
    # 标准化特征
    X = StandardScaler().fit_transform(df_cluster[sdi_cluster_features])
    
    # GMM聚类
    gmm = GaussianMixture(n_components=2, random_state=42)
    df_cluster['subtype'] = gmm.fit_predict(X)
    
    # 对齐逻辑：确保subtype=1是紊乱组（Disturbed）
    # 通过AHI均值判断
    if df_cluster.groupby('subtype')['ahi_value'].mean().idxmax() == 0:
        df_cluster['subtype'] = 1 - df_cluster['subtype']
    
    print(f"聚类后样本数: {len(df_cluster)}")
    print(f"Normal Sleep (subtype=0): {len(df_cluster[df_cluster['subtype']==0])} 人")
    print(f"Disturbed Sleep (subtype=1): {len(df_cluster[df_cluster['subtype']==1])} 人")
    
    return df_cluster

# ==========================================
# 4. 生存分析函数
# ==========================================
def perform_survival_analysis(df_cluster, outcome_type='allcause', title_suffix='All-cause Mortality'):
    """
    执行生存分析
    outcome_type: 'allcause', 'chd', 或 'cvd'
    """
    # 选择事件和生存时间
    if outcome_type == 'allcause':
        event_col = 'event_allcause'
        time_col = 'time_years'
    elif outcome_type == 'chd':
        event_col = 'event_chd'
        time_col = 'time_years'
    elif outcome_type == 'cvd':
        event_col = 'event_cvd'
        time_col = 'time_years'
    else:
        raise ValueError("outcome_type必须是'allcause', 'chd', 或 'cvd'")
    
    # 准备数据（去除缺失值）
    survival_cols = ['subtype', event_col, time_col, 'age_clean', 'sex_clean', 
                     'bmi_clean', 'diabetes_binary', 'ahi_value', 'tst_value']
    
    # 检查race和smoking是否存在且非全0
    if 'race_clean' in df_cluster.columns and df_cluster['race_clean'].nunique() > 1:
        survival_cols.append('race_clean')
    if 'smoking_binary' in df_cluster.columns and df_cluster['smoking_binary'].nunique() > 1:
        survival_cols.append('smoking_binary')
    
    df_surv = df_cluster[survival_cols].copy()
    # 只删除缺失的生存时间和事件，保留subtype可能有缺失的情况
    df_surv = df_surv.dropna(subset=[event_col, time_col])
    # 确保subtype不缺失
    df_surv = df_surv[df_surv['subtype'].notna()].copy()
    
    # 检查删除缺失值后的数据
    print(f"  删除缺失值前样本数: {len(df_cluster)}")
    print(f"  删除缺失值后样本数: {len(df_surv)}")
    print(f"  删除的样本数: {len(df_cluster) - len(df_surv)}")
    
    # 数据质量检查
    print(f"\n{title_suffix} - 数据质量检查:")
    print(f"  总样本数: {len(df_surv)}")
    print(f"  生存时间统计: 最小值={df_surv[time_col].min():.2f}年, 最大值={df_surv[time_col].max():.2f}年, 中位数={df_surv[time_col].median():.2f}年")
    print(f"  事件统计: 总事件数={df_surv[event_col].sum()}, 事件率={df_surv[event_col].mean()*100:.1f}%")
    print(f"  生存时间>10年的样本数: {len(df_surv[df_surv[time_col] > 10])}")
    print(f"  生存时间>10年且发生事件的样本数: {len(df_surv[(df_surv[time_col] > 10) & (df_surv[event_col] == 1)])}")
    
    print(f"\n{title_suffix} - 分组统计:")
    print(f"  Normal Sleep (subtype=0): {len(df_surv[df_surv['subtype']==0])} 人，事件数: {df_surv[df_surv['subtype']==0][event_col].sum()}")
    print(f"    - 生存时间范围: [{df_surv[df_surv['subtype']==0][time_col].min():.2f}, {df_surv[df_surv['subtype']==0][time_col].max():.2f}] 年")
    print(f"    - 生存时间>10年: {len(df_surv[(df_surv['subtype']==0) & (df_surv[time_col] > 10)])} 人")
    print(f"  Disturbed Sleep (subtype=1): {len(df_surv[df_surv['subtype']==1])} 人，事件数: {df_surv[df_surv['subtype']==1][event_col].sum()}")
    print(f"    - 生存时间范围: [{df_surv[df_surv['subtype']==1][time_col].min():.2f}, {df_surv[df_surv['subtype']==1][time_col].max():.2f}] 年")
    print(f"    - 生存时间>10年: {len(df_surv[(df_surv['subtype']==1) & (df_surv[time_col] > 10)])} 人")
    
    # 分组数据
    group_normal = df_surv[df_surv['subtype'] == 0].copy()
    group_disturbed = df_surv[df_surv['subtype'] == 1].copy()
    
    # 检查分组数据的生存时间分布
    print(f"\n分组生存时间详细统计:")
    print(f"  Normal组 - 生存时间分位数:")
    print(f"    25%: {group_normal[time_col].quantile(0.25):.2f}年, 50%: {group_normal[time_col].quantile(0.50):.2f}年, 75%: {group_normal[time_col].quantile(0.75):.2f}年")
    print(f"    事件分布: 0-5年={((group_normal[time_col] <= 5) & (group_normal[event_col] == 1)).sum()}, "
          f"5-10年={((group_normal[time_col] > 5) & (group_normal[time_col] <= 10) & (group_normal[event_col] == 1)).sum()}, "
          f">10年={(group_normal[time_col] > 10).sum()}")
    print(f"    最长随访时间: {group_normal[time_col].max():.2f}年")
    print(f"  Disturbed组 - 生存时间分位数:")
    print(f"    25%: {group_disturbed[time_col].quantile(0.25):.2f}年, 50%: {group_disturbed[time_col].quantile(0.50):.2f}年, 75%: {group_disturbed[time_col].quantile(0.75):.2f}年")
    print(f"    事件分布: 0-5年={((group_disturbed[time_col] <= 5) & (group_disturbed[event_col] == 1)).sum()}, "
          f"5-10年={((group_disturbed[time_col] > 5) & (group_disturbed[time_col] <= 10) & (group_disturbed[event_col] == 1)).sum()}, "
          f">10年={(group_disturbed[time_col] > 10).sum()}")
    print(f"    最长随访时间: {group_disturbed[time_col].max():.2f}年")
    
    # 检查是否有足够的未发生事件的观察值（censored observations）
    print(f"\n审查（censored）数据检查:")
    print(f"  Normal组 - 未发生事件（存活）数: {(group_normal[event_col] == 0).sum()}, 发生事件数: {(group_normal[event_col] == 1).sum()}")
    print(f"  Disturbed组 - 未发生事件（存活）数: {(group_disturbed[event_col] == 0).sum()}, 发生事件数: {(group_disturbed[event_col] == 1).sum()}")
    
    # Kaplan-Meier拟合
    kmf_normal = KaplanMeierFitter()
    kmf_disturbed = KaplanMeierFitter()
    
    # 确保使用正确的格式：duration_col是生存时间，event_col是事件指示（1=发生事件）
    kmf_normal.fit(group_normal[time_col], group_normal[event_col], label='Normal Sleep')
    kmf_disturbed.fit(group_disturbed[time_col], group_disturbed[event_col], label='Disturbed Sleep')
    
    # 检查KM曲线的最大时间点和生存概率
    max_time_normal = kmf_normal.timeline.max()
    max_time_disturbed = kmf_disturbed.timeline.max()
    print(f"\nKM曲线时间范围:")
    print(f"  Normal组 - 最大时间点: {max_time_normal:.2f}年")
    print(f"  Disturbed组 - 最大时间点: {max_time_disturbed:.2f}年")
    
    # 检查KM曲线的生存概率（使用interpolate=True确保能预测任意时间点）
    try:
        surv_5n = kmf_normal.predict(5) if max_time_normal >= 5 else kmf_normal.predict(max_time_normal)
        surv_10n = kmf_normal.predict(10) if max_time_normal >= 10 else kmf_normal.predict(max_time_normal)
        surv_15n = kmf_normal.predict(15) if max_time_normal >= 15 else kmf_normal.predict(max_time_normal)
        print(f"  Normal组 - 5年生存率: {surv_5n:.3f}, 10年生存率: {surv_10n:.3f}, 15年生存率: {surv_15n:.3f}")
    except:
        print(f"  Normal组 - 无法预测某些时间点的生存率")
    
    try:
        surv_5d = kmf_disturbed.predict(5) if max_time_disturbed >= 5 else kmf_disturbed.predict(max_time_disturbed)
        surv_10d = kmf_disturbed.predict(10) if max_time_disturbed >= 10 else kmf_disturbed.predict(max_time_disturbed)
        surv_15d = kmf_disturbed.predict(15) if max_time_disturbed >= 15 else kmf_disturbed.predict(max_time_disturbed)
        print(f"  Disturbed组 - 5年生存率: {surv_5d:.3f}, 10年生存率: {surv_10d:.3f}, 15年生存率: {surv_15d:.3f}")
    except:
        print(f"  Disturbed组 - 无法预测某些时间点的生存率")
    
    # Log-rank检验
    results = logrank_test(group_normal[time_col], group_disturbed[time_col],
                          group_normal[event_col], group_disturbed[event_col])
    p_value_lr = results.p_value
    
    # Cox比例风险模型1：年龄、性别、种族
    model1_vars = ['subtype', 'age_clean', 'sex_clean']
    if 'race_clean' in df_surv.columns and df_surv['race_clean'].nunique() > 1:
        model1_vars.append('race_clean')
    
    df_model1 = df_surv[model1_vars + [event_col, time_col]].dropna()
    cph1 = CoxPHFitter()
    cph1.fit(df_model1, duration_col=time_col, event_col=event_col)
    
    # 获取HR和置信区间
    hr1 = cph1.summary.loc['subtype', 'exp(coef)']
    ci1_low = cph1.summary.loc['subtype', 'exp(coef) lower 95%']
    ci1_high = cph1.summary.loc['subtype', 'exp(coef) upper 95%']
    p1 = cph1.summary.loc['subtype', 'p']
    
    # Cox比例风险模型2：模型1 + BMI、吸烟史、糖尿病（不纳入高血压）
    model2_vars = model1_vars + ['bmi_clean', 'diabetes_binary']
    if 'smoking_binary' in df_surv.columns and df_surv['smoking_binary'].nunique() > 1:
        model2_vars.append('smoking_binary')
    
    df_model2 = df_surv[model2_vars + [event_col, time_col]].dropna()
    cph2 = CoxPHFitter()
    cph2.fit(df_model2, duration_col=time_col, event_col=event_col)
    
    # 获取HR和置信区间
    hr2 = cph2.summary.loc['subtype', 'exp(coef)']
    ci2_low = cph2.summary.loc['subtype', 'exp(coef) lower 95%']
    ci2_high = cph2.summary.loc['subtype', 'exp(coef) upper 95%']
    p2 = cph2.summary.loc['subtype', 'p']
    
    # Cox比例风险模型3：模型2 + AHI + TST
    model3_vars = model2_vars + ['ahi_value']
    if 'tst_value' in df_surv.columns and df_surv['tst_value'].sum() > 0:
        model3_vars.append('tst_value')
    
    df_model3 = df_surv[model3_vars + [event_col, time_col]].dropna()
    cph3 = CoxPHFitter()
    cph3.fit(df_model3, duration_col=time_col, event_col=event_col)
    
    # 获取HR和置信区间
    hr3 = cph3.summary.loc['subtype', 'exp(coef)']
    ci3_low = cph3.summary.loc['subtype', 'exp(coef) lower 95%']
    ci3_high = cph3.summary.loc['subtype', 'exp(coef) upper 95%']
    p3 = cph3.summary.loc['subtype', 'p']
    
    # 打印结果
    print(f"\n{title_suffix} - 统计分析结果:")
    print(f"  Log-rank test: p={p_value_lr:.4f}")
    print(f"  模型1 (年龄+性别+种族): HR={hr1:.2f} (95% CI: {ci1_low:.2f}-{ci1_high:.2f}), p={p1:.4f}")
    print(f"  模型2 (+BMI+吸烟+DM): HR={hr2:.2f} (95% CI: {ci2_low:.2f}-{ci2_high:.2f}), p={p2:.4f}")
    print(f"  模型3 (+AHI+TST): HR={hr3:.2f} (95% CI: {ci3_low:.2f}-{ci3_high:.2f}), p={p3:.4f}")
    
    return {
        'kmf_normal': kmf_normal,
        'kmf_disturbed': kmf_disturbed,
        'p_value_lr': p_value_lr,
        'hr1': hr1,
        'ci1': (ci1_low, ci1_high),
        'p1': p1,
        'hr2': hr2,
        'ci2': (ci2_low, ci2_high),
        'p2': p2,
        'hr3': hr3,
        'ci3': (ci3_low, ci3_high),
        'p3': p3,
        'df_surv': df_surv
    }

# ==========================================
# 5. 绘图函数
# ==========================================
def plot_single_km_curve(ax, results_dict, title_suffix='All-cause Mortality'):
    """在给定的axes上绘制单个KM曲线（参考ECG-sdi格式）"""
    kmf_normal = results_dict['kmf_normal']
    kmf_disturbed = results_dict['kmf_disturbed']
    p_value_lr = results_dict['p_value_lr']
    hr1 = results_dict['hr1']
    ci1 = results_dict['ci1']
    p1 = results_dict['p1']
    hr2 = results_dict['hr2']
    ci2 = results_dict['ci2']
    p2 = results_dict['p2']
    hr3 = results_dict['hr3']
    ci3 = results_dict['ci3']
    p3 = results_dict['p3']
    df_surv = results_dict['df_surv']

    # 最大随访时间
    max_time_plot = 15

    # Normal组
    sf_normal = kmf_normal.survival_function_
    timeline_normal = sf_normal.index.values.copy()
    surv_values_normal = sf_normal.iloc[:, 0].values.copy()
    if timeline_normal[-1] < max_time_plot:
        timeline_normal = np.append(timeline_normal, max_time_plot)
        surv_values_normal = np.append(surv_values_normal, surv_values_normal[-1])

    # Disturbed组
    sf_disturbed = kmf_disturbed.survival_function_
    timeline_disturbed = sf_disturbed.index.values.copy()
    surv_values_disturbed = sf_disturbed.iloc[:, 0].values.copy()
    if timeline_disturbed[-1] < max_time_plot:
        timeline_disturbed = np.append(timeline_disturbed, max_time_plot)
        surv_values_disturbed = np.append(surv_values_disturbed, surv_values_disturbed[-1])

    ax.step(timeline_normal, surv_values_normal, where="post", color="#2166ac", linewidth=2, label="Normal Sleep")
    ax.step(timeline_disturbed, surv_values_disturbed, where="post", color="#b2182b", linewidth=2, label="Disturbed Sleep")

    # 置信区间
    ci_normal = kmf_normal.confidence_interval_
    if ci_normal is not None and len(ci_normal) > 0:
        ci_timeline = ci_normal.index.values.copy()
        ci_low = ci_normal.iloc[:, 0].values.copy()
        ci_high = ci_normal.iloc[:, 1].values.copy()
        if ci_timeline[-1] < max_time_plot:
            ci_timeline = np.append(ci_timeline, max_time_plot)
            ci_low = np.append(ci_low, ci_low[-1])
            ci_high = np.append(ci_high, ci_high[-1])
        ax.fill_between(ci_timeline, ci_low, ci_high, step="post", alpha=0.2, color="#2166ac")

    ci_disturbed = kmf_disturbed.confidence_interval_
    if ci_disturbed is not None and len(ci_disturbed) > 0:
        ci_timeline = ci_disturbed.index.values.copy()
        ci_low = ci_disturbed.iloc[:, 0].values.copy()
        ci_high = ci_disturbed.iloc[:, 1].values.copy()
        if ci_timeline[-1] < max_time_plot:
            ci_timeline = np.append(ci_timeline, max_time_plot)
            ci_low = np.append(ci_low, ci_low[-1])
            ci_high = np.append(ci_high, ci_high[-1])
        ax.fill_between(ci_timeline, ci_low, ci_high, step="post", alpha=0.2, color="#b2182b")

    ax.set_xlabel("Time to Event (Years)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Survival Probability (%)", fontsize=14, fontweight="bold")
    ax.set_title(title_suffix, fontsize=16, fontweight="bold", pad=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim(0, 15)
    
    # 设置刻度标签字体大小
    ax.tick_params(axis='both', labelsize=14)
    
    # 根据不同的图设置不同的y轴范围
    if "All-cause" in title_suffix:
        # a图：全因死亡，纵轴55%-100%
        ax.set_ylim(0.55, 1.0)
        ax.set_yticks([0.55, 0.65, 0.75, 0.85, 0.95, 1.0])
        ax.set_yticklabels(["55", "65", "75", "85", "95", "100"])
    else:
        # b图和c图：CHD和CVD死亡，纵轴85%-100%
        ax.set_ylim(0.85, 1.0)
        ax.set_yticks([0.85, 0.88, 0.91, 0.94, 0.97, 1.0])
        ax.set_yticklabels(["85", "88", "91", "94", "97", "100"])

    # 将统计信息移到左下角，避免与曲线重叠
    stats_text = f"Log-rank test: p={p_value_lr:.4f}" if p_value_lr >= 0.001 else "Log-rank test: p<0.001"
    ax.text(0.02, 0.20, stats_text, transform=ax.transAxes, fontsize=11,
            ha="left", va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray", linewidth=1.2))

    hr1_text = f"Model 1: HR={hr1:.2f} (95% CI: {ci1[0]:.2f}-{ci1[1]:.2f})"
    ax.text(0.02, 0.14, hr1_text, transform=ax.transAxes, fontsize=10,
            ha="left", va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray", linewidth=1.0))

    hr2_text = f"Model 2: HR={hr2:.2f} (95% CI: {ci2[0]:.2f}-{ci2[1]:.2f})"
    ax.text(0.02, 0.08, hr2_text, transform=ax.transAxes, fontsize=10,
            ha="left", va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray", linewidth=1.0))

    hr3_text = f"Model 3: HR={hr3:.2f} (95% CI: {ci3[0]:.2f}-{ci3[1]:.2f})"
    ax.text(0.02, 0.02, hr3_text, transform=ax.transAxes, fontsize=10,
            ha="left", va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray", linewidth=1.0))

    ax.legend(loc="upper right", fontsize=12)

def plot_km_curve_single(results_dict, title_suffix, save_path):
    """输出单图KM曲线"""
    fig, ax = plt.subplots(figsize=(10, 7))
    plot_single_km_curve(ax, results_dict, title_suffix=title_suffix)
    plt.tight_layout()
    plt.savefig(save_path, dpi=800, bbox_inches="tight")
    print(f"\n单图KM曲线已保存至: {save_path}")
    plt.close(fig)


# ==========================================
# 6. 主程序
# ==========================================
if __name__ == "__main__":
    # 数据路径
    base_path = r"D:\Project\Python_Project\SDI\W_NW_REM_NREM\data\shhs"
    
    print("=" * 60)
    print("SHHS 生存分析 - 基于Full Layer特征聚类")
    print("=" * 60)

    print("\n步骤1: 加载数据...")
    df = load_shhs_data(base_path)
    print(f"数据加载完成，总样本数: {len(df)}")

    print("\n步骤2: 执行聚类分析...")
    df_cluster = perform_clustering(df)

    print("\n步骤3: 全因死亡生存分析...")
    results_allcause = perform_survival_analysis(df_cluster, outcome_type='allcause',
                                                  title_suffix='All-cause Mortality')

    print("\n步骤4: 心血管疾病死亡生存分析...")
    results_cvd = perform_survival_analysis(df_cluster, outcome_type='cvd',
                                            title_suffix='Cardiovascular Disease Death')

    print("\n步骤5: 绘制KM曲线...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, '..', 'output')
    os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    plot_single_km_curve(ax1, results_allcause, title_suffix='All-cause Mortality')
    ax1.text(-0.05, 1.02, 'a', transform=ax1.transAxes, fontsize=16, fontweight='bold', va='bottom')
    plot_single_km_curve(ax2, results_cvd, title_suffix='Cardiovascular Disease Death')
    ax2.text(-0.05, 1.02, 'b', transform=ax2.transAxes, fontsize=16, fontweight='bold', va='bottom')

    plt.subplots_adjust(bottom=0.08, top=0.95, left=0.05, right=0.98, wspace=0.25)

    combined_path = os.path.join(output_dir, 'shhs_km_curves_combined_2plots_full_layer.png')
    plt.savefig(combined_path, dpi=800, bbox_inches='tight', facecolor='white')
    print(f"\n组合KM曲线图（2图）已保存至: {combined_path}")
    plt.show()

    # 单图输出
    plot_km_curve_single(
        results_allcause,
        title_suffix='All-cause Mortality',
        save_path=os.path.join(output_dir, 'shhs_km_curve_allcause_full_layer.png'),
    )
    plot_km_curve_single(
        results_cvd,
        title_suffix='Cardiovascular Disease Death',
        save_path=os.path.join(output_dir, 'shhs_km_curve_cvd_full_layer.png'),
    )

    print("\n" + "=" * 60)
    print("生存分析完成！")
    print("=" * 60)
