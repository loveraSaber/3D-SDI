import os
import xmltodict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.preprocessing import MinMaxScaler

def plot_combined_sleep_analysis(xml_path, csv_path):
    """
    上图：睡眠事件（左移0.5小时）
    中图：Wake & REM 概率（归一化）
    下图：睡眠阶段标签 —— 阶梯折线形式
    映射：0=wake, 1+2=N1+N2, 3=N3, 4=REM
    """
    # ===================== 1. 睡眠事件定义 =====================
    EVENT_MAP = {
        'Arousal': {'concepts': ['ASDA arousal|Arousal (ASDA)', 'Arousal|Arousal ()', 'Arousals|Arousals', 'Arousal (ASDA)'], 'color': '#FF4B4B', 'y_pos': 5},
        'Mixed Apnea': {'concepts': ['Mixed apnea|Mixed Apnea'], 'color': '#9B59B6', 'y_pos': 4},
        'Central Apnea': {'concepts': ['Central apnea|Central Apnea'], 'color': '#3498DB', 'y_pos': 3},
        'Obstructive Apnea': {'concepts': ['Obstructive apnea|Obstructive Apnea'], 'color': '#2ECC71', 'y_pos': 2},
        'Hypopnea': {'concepts': ['Hypopnea|Hypopnea'], 'color': '#F1C40F', 'y_pos': 1},
        'SpO2 Desat': {'concepts': ['SpO2 desaturation|SpO2 desaturation'], 'color': '#E67E22', 'y_pos': 0}
    }

    # ===================== 2. 读取 XML 睡眠事件 =====================
    with open(xml_path, 'r', encoding='utf-8') as f:
        data_dict = xmltodict.parse(f.read())
    events = data_dict['PSGAnnotation']['ScoredEvents']['ScoredEvent']
    if not isinstance(events, list):
        events = [events]

    # ===================== 3. 读取 CSV =====================
    df = pd.read_csv(csv_path).sort_values('window_center').reset_index(drop=True)
    time_hours = (df['window_center'] * 30) / 3600
    x_min = time_hours.min()
    x_max = time_hours.max()

    # 左移0.5小时
    SHIFT_HOURS = -0.5

    # ===================== 4. 创建画布：3行1列 =====================
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(22, 16), sharex=True)
    file_id = os.path.splitext(os.path.basename(xml_path))[0]

    # ===================== 上图：睡眠事件 =====================
    legend_events = []
    for name, info in EVENT_MAP.items():
        cnt = 0
        for ev in events:
            concept = ev.get("EventConcept", "")
            if concept in info['concepts']:
                s = float(ev["Start"]) / 3600
                e = (float(ev["Start"]) + float(ev["Duration"])) / 3600

                s_shifted = s + SHIFT_HOURS
                e_shifted = e + SHIFT_HOURS

                if e_shifted > x_min and s_shifted < x_max:
                    s_draw = max(s_shifted, x_min)
                    e_draw = min(e_shifted, x_max)
                    ax1.broken_barh([(s_draw, e_draw - s_draw)], (info['y_pos'] - 0.4, 0.8), color=info['color'])
                    cnt += 1
        if cnt > 0:
            legend_events.append(Patch(color=info['color'], label=f"{name} (n={cnt})"))

    ax1.set_title(f"Sleep Events - {file_id}", fontsize=16)
    ax1.set_yticks([0,1,2,3,4,5])
    ax1.set_yticklabels(['SpO2 Desat','Hypopnea','Obstructive','Central','Mixed','Arousal'])
    ax1.grid(axis='x', linestyle=':', alpha=0.5)
    ax1.legend(handles=legend_events, loc='upper right', bbox_to_anchor=(1.1, 1))

    # ===================== 中图：Wake & REM 概率 =====================
    scaler = MinMaxScaler()
    cols_to_norm = ['wake_depth_score', 'rem_prob_rem']
    df_norm = pd.DataFrame(scaler.fit_transform(df[cols_to_norm]), columns=cols_to_norm)

    ax2.fill_between(time_hours, df_norm['wake_depth_score'], color='red', alpha=0.2, label='Wake Probability')
    ax2.fill_between(time_hours, df_norm['rem_prob_rem'], color='green', alpha=0.2, label='REM Probability')

    ax2.set_title("Wake & REM Probabilities", fontsize=16)
    ax2.set_ylabel("Normalized Level (0 to 1)", fontsize=12)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.legend(loc='upper right', bbox_to_anchor=(1, 1), ncol=2)

    # ===================== 下图：睡眠阶段 —— 阶梯折线 =====================
    labels = df['label'].values
    # 映射：0=wake, 1/2=N1+N2, 3=N3, 4=REM
    stage_map = {
        0: 0,   # Wake
        1: 1,   # N1+N2
        2: 1,   # N1+N2
        3: 2,   # N3
        4: 3    # REM
    }
    y_stages = np.array([stage_map[lab] for lab in labels])

    # 阶梯折线
    ax3.step(time_hours, y_stages, where='post', linewidth=2, color='#2c3e50', label='Sleep Stage')

    ax3.set_title("Sleep Stage (Step Line)", fontsize=16)
    ax3.set_yticks([0, 1, 2, 3])
    ax3.set_yticklabels(['Wake', 'N1+N2', 'N3', 'REM'])
    ax3.set_ylim(-0.2, 3.2)
    ax3.set_xlabel("Time (Hours)", fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.4)
    ax3.legend(loc='upper right')

    # ===================== 保存 =====================
    plt.tight_layout()
    save_path = f"/home/zhaoqingshuo/SDI/ECG_sleepstage/transformer_W_NW_REM_NREM/output_arousal/{file_id}_3panel_step.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ 三图折线版已保存：{save_path}")

# ===================== 执行 =====================
if __name__ == "__main__":
    XML_FILE = "/data/0shared/NSRR/shhs/annotations-events-nsrr/shhs1-205218-nsrr.xml"
    CSV_FILE = "/data/0shared/zhaoqingshuo/SDI/model_W_NW_REM_NREM_all/test/csv_exports_shhs/csv/csv_exports/shhs1-205218_features_predictions.csv"
    plot_combined_sleep_analysis(XML_FILE, CSV_FILE)