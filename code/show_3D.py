import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from sklearn.preprocessing import MinMaxScaler

# ===================================
# 解决中文显示乱码（方块）问题
# ===================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ===================================
# 1. 读取数据
# ===================================
csv_path = r"C:\Users\SaberShuo\Desktop\case_study\shhs1-205218_features_predictions.csv"
df = pd.read_csv(csv_path)

# 只保留需要的列，并去掉缺失值
data = df[['window_center', 'wake_depth_score', 'rem_prob_rem', 'label']].dropna()

scaler = MinMaxScaler()
data[['wake_depth_norm', 'rem_prob_norm']] = scaler.fit_transform(
    data[['wake_depth_score', 'rem_prob_rem']]
)

time_sec = data['window_center'].values * 30.0
# 转换为相对时间（从 0 开始），使三轴在左下角共享同一个 0 点
time_rel = time_sec - time_sec.min()
body_depth  = data['wake_depth_norm'].values
brain_depth = data['rem_prob_norm'].values
stage = data['label'].values.astype(int)

# ===================================
# 3. 统计计算 (穿过平面、线条长度、四个象限比例)
# ===================================
# a) 四个象限点的比例 (基于 0.5 划分)
q1 = np.mean((body_depth >= 0.5) & (brain_depth >= 0.5))
q2 = np.mean((body_depth <  0.5) & (brain_depth >= 0.5))
q3 = np.mean((body_depth <  0.5) & (brain_depth <  0.5))
q4 = np.mean((body_depth >= 0.5) & (brain_depth <  0.5))

# b) 穿过十字平面的次数
cross_body  = np.sum((body_depth[:-1]  - 0.5) * (body_depth[1:]  - 0.5) < 0)
cross_brain = np.sum((brain_depth[:-1] - 0.5) * (brain_depth[1:] - 0.5) < 0)
total_crossings = cross_body + cross_brain

# c) 三维空间线条的总长度（归一化后）
time_norm = time_rel / time_rel.max()
dx = np.diff(time_norm)
dy = np.diff(body_depth)
dz = np.diff(brain_depth)
total_length_3d = np.sum(np.sqrt(dx**2 + dy**2 + dz**2))

# d) 按小时均值
duration_hours    = time_rel.max() / 3600.0
crossings_per_hour = total_crossings  / duration_hours
length_per_hour    = total_length_3d  / duration_hours

# ===================================
# 4. 象限颜色定义
# ===================================
quadrant_meta = {
    'Q1': ('#ff6b6b', 'Q1 (Body≥0.5, Brain≥0.5)'),   # 红
    'Q2': ('#4ecdc4', 'Q2 (Body<0.5,  Brain≥0.5)'),   # 青
    'Q3': ('#a8e6cf', 'Q3 (Body<0.5,  Brain<0.5) '),  # 浅绿
    'Q4': ('#f9ca24', 'Q4 (Body≥0.5,  Brain<0.5) '),  # 黄
}

def get_quadrant_color(bd, brd):
    if   bd >= 0.5 and brd >= 0.5: return quadrant_meta['Q1'][0]
    elif bd <  0.5 and brd >= 0.5: return quadrant_meta['Q2'][0]
    elif bd <  0.5 and brd <  0.5: return quadrant_meta['Q3'][0]
    else:                           return quadrant_meta['Q4'][0]

scatter_colors = [get_quadrant_color(bd, brd) for bd, brd in zip(body_depth, brain_depth)]

# ===================================
# 5. 绘图风格与轴比例设置
# ===================================
plt.style.use('seaborn-v0_8-darkgrid')
fig = plt.figure(figsize=(16, 8))
ax  = fig.add_subplot(111, projection='3d')

ax.set_facecolor('#1e1e1e')
fig.patch.set_facecolor('#1e1e1e')

ax.set_box_aspect((4, 1.2, 1.2))
ax.view_init(elev=20, azim=-50)

# ===================================
# 6. 绘制 0.5 十字相交平面
# ===================================
T_MAX = time_rel.max()

# Body=0.5 平面（X-Z 面）
xx, zz = np.meshgrid([0, T_MAX], [0, 1])
yy = np.full_like(xx, 0.5)
ax.plot_surface(xx, yy, zz, color='cyan', alpha=0.1, zorder=1)

# Brain=0.5 平面（X-Y 面）
xx2, yy2 = np.meshgrid([0, T_MAX], [0, 1])
zz2 = np.full_like(xx2, 0.5)
ax.plot_surface(xx2, yy2, zz2, color='magenta', alpha=0.1, zorder=1)

# 中心轴（两平面交线）
ax.plot([0, T_MAX], [0.5, 0.5], [0.5, 0.5],
        color='white', linewidth=1.5, linestyle='-.', zorder=2, alpha=0.6,
        label='Center Axis (0.5)')

# ===================================
# 7. 左侧面（x=0）象限投影 + 0.5 分界十字线
# ===================================
# 背景浅色左侧面，方便观察投影
xx_l = np.array([[0, 0], [0, 0]])
yy_l = np.array([[0, 1], [0, 1]])
zz_l = np.array([[0, 0], [1, 1]])
ax.plot_surface(xx_l, yy_l, zz_l, color='white', alpha=0.04, zorder=0)

# 象限分界线（左侧面 0.5 十字）
ax.plot([0, 0], [0.5, 0.5], [0, 1], color='white', linewidth=1.0, linestyle='--', alpha=0.55, zorder=2)
ax.plot([0, 0], [0, 1], [0.5, 0.5], color='white', linewidth=1.0, linestyle='--', alpha=0.55, zorder=2)

# 象限散点投影到左侧面
ax.scatter([0]*len(body_depth), body_depth, brain_depth,
           c=scatter_colors, s=4, alpha=0.55, zorder=3, depthshade=False)

# ===================================
# 8. 连续多色轨迹线绘制
# ===================================
stage_names = {
    0: "Wake",
    1: "N1 (Light)",
    2: "N2 (Light)",
    3: "N3 (Deep)",
    4: "REM"
}

stage_colors = {
    0: "#ff9800",   # 橙黄 Wake
    1: "#00bcd4",   # 浅蓝 N1
    2: "#03a9f4",   # 亮蓝 N2
    3: "#9c27b0",   # 紫色 N3
    4: "#4caf50"    # 翠绿 REM
}

points   = np.array([time_rel, body_depth, brain_depth]).T.reshape(-1, 1, 3)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
segment_colors = [stage_colors.get(s, "#ffffff") for s in stage[:-1]]

lc = Line3DCollection(segments, colors=segment_colors, linewidths=1.0, alpha=0.9, zorder=4)
ax.add_collection3d(lc)

for stg in sorted(np.unique(stage)):
    if stg in stage_colors:
        ax.plot([], [], [], color=stage_colors[stg], linewidth=2.0, label=stage_names[stg])

# ===================================
# 9. 坐标轴设置：X 轴保持默认，Y/Z 轴标签移至左侧手动标注
# ===================================
ax.set_xlabel("Time (s)", labelpad=15, color="white", fontsize=12)
ax.set_ylabel("")   # 清空默认 Y 轴标签
ax.set_zlabel("")   # 清空默认 Z 轴标签

ax.set_title("3D Sleep Depth Trajectory & Spatial Quadrant Analysis",
             pad=30, fontsize=18, color="white", fontweight="bold")

ax.set_xlim(0, T_MAX)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)

ax.tick_params(colors="lightgray", labelsize=10)

# ------------------------------------------------------------------
# Y / Z 轴刻度数字移到左侧面（x=0 面）手动绘制
# 隐藏原来右侧的刻度数字
# ------------------------------------------------------------------
ax.set_yticklabels([])
ax.set_zticklabels([])

tick_vals = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# Y 轴刻度：沿左侧面底边，紧贴 z 轴下方
for tv in tick_vals:
    ax.text(0, tv, -0.06,
            f'{tv:.1f}',
            color='lightgray', fontsize=9, ha='center', va='top',
            zorder=11)

# Z 轴刻度：沿左侧面左边，紧贴 y 轴左侧
for tv in tick_vals:
    ax.text(0, -0.06, tv,
            f'{tv:.1f}',
            color='lightgray', fontsize=9, ha='right', va='center',
            zorder=11)

# 网格线
ax.xaxis._axinfo["grid"]['color']     = (1, 1, 1, 0.15)
ax.yaxis._axinfo["grid"]['color']     = (1, 1, 1, 0.15)
ax.zaxis._axinfo["grid"]['color']     = (1, 1, 1, 0.15)
ax.xaxis._axinfo["grid"]['linestyle'] = '--'
ax.yaxis._axinfo["grid"]['linestyle'] = '--'
ax.zaxis._axinfo["grid"]['linestyle'] = '--'

# 面板底色
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.05))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.05))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.05))

# ------------------------------------------------------------------
# Y 轴标签（Body Depth）置于左侧面下方
#   x=0（左侧面），y=0.5（轴中心），z 略低于 0
# ------------------------------------------------------------------
ax.text(0, 0.5, -0.22,
        "Body Depth Score",
        color='white', fontsize=10, ha='center', va='top',
        zorder=10)

# ------------------------------------------------------------------
# Z 轴标签（Brain Depth）置于左侧面左侧
#   x=0（左侧面），y 略小于 0（面前方），z=0.5（轴中心）
# ------------------------------------------------------------------
ax.text(0, -0.22, 0.5,
        "Brain Activity Score",
        color='white', fontsize=10, ha='right', va='center',
        zorder=10)

# ===================================
# 10. 起点 / 终点高亮
# ===================================
ax.scatter(0,          body_depth[0],  brain_depth[0],  color="lime", s=100, marker='*', edgecolor="black", zorder=10, label="Start")
ax.scatter(time_rel[-1], body_depth[-1], brain_depth[-1], color="red",  s=100, marker='X', edgecolor="black", zorder=10, label="End")

# ===================================
# 11. 图例（睡眠阶段 + 象限颜色）
# ===================================
for qk, (qc, ql) in quadrant_meta.items():
    ax.plot([], [], 'o', color=qc, markersize=6, label=ql)

legend = ax.legend(loc='upper right', bbox_to_anchor=(1.18, 1.02),
                   frameon=True, facecolor="#333333", edgecolor="none",
                   fontsize=9)
plt.setp(legend.get_texts(), color="white")

# ===================================
# 12. 左侧统计信息文本框
# ===================================
stats_text = (
    f"--- Stats (Per Hour) ---\n"
    f"Duration: {duration_hours:.2f} h\n"
    f"Crossings: {crossings_per_hour:.1f} /h\n"
    f"3D Length: {length_per_hour:.2f} /h\n\n"
    f"--- Quadrant Distribution ---\n"
    f"Q1 (Body≥0.5, Brain>=0.5): {q1*100:.1f}%\n"
    f"Q2 (Body<0.5, Brain>=0.5): {q2*100:.1f}%\n"
    f"Q3 (Body<0.5, Brain<0.5) : {q3*100:.1f}%\n"
    f"Q4 (Body≥0.5, Brain<0.5) : {q4*100:.1f}%"
)

fig.text(0.02, 0.5, stats_text, color='white', fontsize=12,
         verticalalignment='center', family='monospace',
         bbox=dict(facecolor='#2a2a2a', edgecolor='#444444',
                   boxstyle='round,pad=1', alpha=0.9))

plt.tight_layout()
plt.subplots_adjust(left=0.25)
plt.show()
