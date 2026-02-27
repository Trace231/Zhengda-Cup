"""
generate_charts.py
==================
生成第五章描述性统计分析所需的全部图表，保存到 charts/ 目录。
"""

import sys, os
sys.path.insert(0, '.pip_pkgs')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib import rcParams

# ── 字体与样式 ────────────────────────────────────────────────────────────────
rcParams['font.family'] = ['Arial Unicode MS', 'STHeiti', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# 主色系（深蓝 / 琥珀 / 橙红 / 灰蓝 / 绿松石）
C_BLUE   = '#2E4057'
C_AMBER  = '#F4A261'
C_ORANGE = '#E76F51'
C_GRAY   = '#8D99AE'
C_TEAL   = '#048A81'
C_LIGHT  = '#A8DADC'
C_PURPLE = '#6B4E71'
C_GREEN  = '#4CAF50'
C_RED    = '#E53935'

PALETTE9 = [C_BLUE, C_TEAL, C_AMBER, C_ORANGE, C_GRAY, C_PURPLE,
            C_GREEN, C_RED, C_LIGHT]

os.makedirs('charts', exist_ok=True)

# ── 载入数据 ──────────────────────────────────────────────────────────────────
from data_pipeline import load_data, get_score_df, CONSTRUCTS, DEMO_COLS

df_raw, scale_data = load_data()
score_df = get_score_df(scale_data)

# 人口学列
gender_col  = df_raw.columns[DEMO_COLS['gender']]
age_col     = df_raw.columns[DEMO_COLS['age']]
occ_col     = df_raw.columns[DEMO_COLS['occupation']]
inc_col     = df_raw.columns[DEMO_COLS['income']]
exp_col     = df_raw.columns[DEMO_COLS['exp_concert']]
plan_col    = df_raw.columns[DEMO_COLS['plan_concert']]
idol_col    = df_raw.columns[DEMO_COLS['idol_count']]
seat_col    = df_raw.columns[DEMO_COLS['seat_pref']]
nontix_col  = df_raw.columns[DEMO_COLS['nontix_spend']]
merch_col   = df_raw.columns[DEMO_COLS['merch_monthly']]
channel_col = df_raw.columns[DEMO_COLS['info_channel']]
fanpur_col  = df_raw.columns[DEMO_COLS['idol_purchase']]

N = len(df_raw)

def save(fig, name, dpi=220):
    path = f'charts/{name}.png'
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  saved: {path}')


# ══════════════════════════════════════════════════════════════════════════════
# 图1  人口学特征概览（2×2）
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle('图1  样本人口学特征概览', fontsize=15, fontweight='bold', y=1.01)
fig.patch.set_facecolor('white')

# (a) 性别 - 饼图
ax = axes[0, 0]
gvals = df_raw[gender_col].value_counts()
wedge_props = dict(width=0.55, edgecolor='white', linewidth=2)
wedges, texts, autotexts = ax.pie(
    gvals.values, labels=gvals.index,
    autopct='%1.1f%%', colors=[C_BLUE, C_AMBER],
    startangle=90, wedgeprops=wedge_props,
    textprops={'fontsize': 11}
)
for at in autotexts:
    at.set_fontsize(11); at.set_fontweight('bold')
ax.set_title('(a) 性别分布', fontsize=12, fontweight='bold', pad=8)

# (b) 年龄 - 条形图
ax = axes[0, 1]
age_order = ['17-21岁（2005-2009年出生）', '22-26岁（2000-2004年出生）']
age_labels = ['17-21岁\n(Z世代末)', '22-26岁\n(Z世代主体)']
age_vals = [df_raw[age_col].value_counts().get(a, 0) for a in age_order]
bars = ax.bar(age_labels, age_vals, color=[C_TEAL, C_BLUE], width=0.5,
              edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, age_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f'{val}人\n({val/N*100:.1f}%)', ha='center', va='bottom',
            fontsize=10.5, fontweight='bold', color=C_BLUE)
ax.set_ylim(0, max(age_vals) * 1.25)
ax.set_title('(b) 年龄分布', fontsize=12, fontweight='bold')
ax.set_ylabel('人数', fontsize=10)
ax.spines[['top', 'right']].set_visible(False)
ax.yaxis.set_tick_params(labelsize=9)

# (c) 职业 - 水平条形图
ax = axes[1, 0]
occ_raw = df_raw[occ_col].value_counts()
occ_map = {
    '企业/单位在职人员（工作 1-3 年）': '在职1-3年',
    '本科生/专科生': '本科/专科生',
    '待业/备考': '待业/备考',
    '研究生': '研究生',
    '自由职业者': '自由职业',
    '高中生': '高中生',
    '企业/单位在职人员（工作 4-8 年）': '在职4-8年',
}
occ_labels = [occ_map.get(k, k) for k in occ_raw.index[:7]]
occ_vals = occ_raw.values[:7]
colors = [PALETTE9[i % len(PALETTE9)] for i in range(len(occ_vals))]
bars = ax.barh(occ_labels[::-1], occ_vals[::-1], color=colors[::-1],
               edgecolor='white', linewidth=1)
for bar, val in zip(bars, occ_vals[::-1]):
    ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
            f'{val}人 ({val/N*100:.0f}%)', va='center',
            fontsize=9.5, color='#333333')
ax.set_xlim(0, max(occ_vals) * 1.35)
ax.set_title('(c) 职业构成', fontsize=12, fontweight='bold')
ax.set_xlabel('人数', fontsize=10)
ax.spines[['top', 'right']].set_visible(False)
ax.xaxis.set_tick_params(labelsize=9)

# (d) 收入 - 条形图
ax = axes[1, 1]
inc_order = ['1000 元及以下', '1001-3000 元', '3001-6000 元',
             '6001-10000 元', '10001 元及以上', '不愿透露']
inc_labels = ['≤1000', '1001-3000', '3001-6000',
              '6001-10000', '≥10001', '未透露']
inc_vals = [df_raw[inc_col].value_counts().get(k, 0) for k in inc_order]
gradient_colors = ['#A8D8EA', '#7EC8E3', '#5BA4CF', '#2E86AB', C_BLUE, C_GRAY]
bars = ax.bar(inc_labels, inc_vals, color=gradient_colors,
              edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, inc_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{val}', ha='center', va='bottom', fontsize=9.5, fontweight='bold')
ax.set_title('(d) 月可支配收入', fontsize=12, fontweight='bold')
ax.set_ylabel('人数', fontsize=10)
ax.set_xlabel('收入区间（元）', fontsize=10)
ax.spines[['top', 'right']].set_visible(False)
plt.setp(ax.get_xticklabels(), rotation=20, ha='right', fontsize=9)

plt.tight_layout(pad=2.5)
save(fig, 'fig1_demographics')


# ══════════════════════════════════════════════════════════════════════════════
# 图2  粉丝特征与观演行为（1×3）
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle('图2  粉丝特征与观演行为分布', fontsize=14, fontweight='bold', y=1.03)
fig.patch.set_facecolor('white')

# (a) 偶像数量 - 甜甜圈
ax = axes[0]
idol_order = ['无', '1位', '2-3位', '4位及以上']
idol_vals = [df_raw[idol_col].value_counts().get(k, 0) for k in idol_order]
# exclude 跳过
idol_vals_clean = [v for v in idol_vals]
wedge_props = dict(width=0.5, edgecolor='white', linewidth=2)
wedges, texts, autotexts = ax.pie(
    idol_vals_clean, labels=idol_order,
    autopct='%1.1f%%', colors=[C_GRAY, C_TEAL, C_BLUE, C_ORANGE],
    startangle=90, wedgeprops=wedge_props,
    textprops={'fontsize': 9.5}
)
for at in autotexts: at.set_fontsize(9); at.set_fontweight('bold')
ax.text(0, 0, f'n={sum(idol_vals_clean)}', ha='center', va='center',
        fontsize=10, fontweight='bold', color='#333')
ax.set_title('(a) 长期支持偶像数量', fontsize=11, fontweight='bold', pad=5)

# (b) 观演经历 - 条形图
ax = axes[1]
exp_order = ['从未有过', '1-3次', '4-6次', '7次及以上']
exp_colors = ['#D3D3D3', C_LIGHT, C_TEAL, C_BLUE]
exp_vals = [df_raw[exp_col].value_counts().get(k, 0) for k in exp_order]
bars = ax.bar(exp_order, exp_vals, color=exp_colors, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, exp_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f'{val}人\n{val/N*100:.0f}%', ha='center', va='bottom',
            fontsize=9, fontweight='bold', color='#333')
ax.set_ylim(0, max(exp_vals) * 1.3)
ax.set_title('(b) 线下观演经历次数', fontsize=11, fontweight='bold')
ax.set_ylabel('人数', fontsize=10)
ax.spines[['top', 'right']].set_visible(False)

# (c) 未来计划 - 水平条形图
ax = axes[2]
plan_order = ['有', '暂时不确定', '否']
plan_colors = [C_TEAL, C_AMBER, C_GRAY]
plan_vals = [df_raw[plan_col].value_counts().get(k, 0) for k in plan_order]
total_plan = sum(plan_vals)
bars = ax.barh(plan_order, plan_vals, color=plan_colors, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, plan_vals):
    ax.text(val + 1, bar.get_y() + bar.get_height()/2,
            f'{val}人 ({val/total_plan*100:.0f}%)',
            va='center', fontsize=10, fontweight='bold', color='#333')
ax.set_xlim(0, max(plan_vals) * 1.4)
ax.set_title('(c) 未来观演计划意向', fontsize=11, fontweight='bold')
ax.set_xlabel('人数', fontsize=10)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout(pad=2.5)
save(fig, 'fig2_fan_features')


# ══════════════════════════════════════════════════════════════════════════════
# 图3  信息获取渠道（多选展开 + 排序）
# ══════════════════════════════════════════════════════════════════════════════
# 展开多选
channel_split = df_raw[channel_col].dropna().astype(str)
channel_split = channel_split[channel_split != '(跳过)']
channel_items = channel_split.str.split('┋').explode().str.strip()
# 缩短标签
label_map = {
    '微博/超话/粉丝群（私域流量）': '微博/粉丝群',
    '抖音/快手/小红书': '抖音/小红书',
    '/B站/视频号': 'B站/视频号',
    '大麦/猫眼/票星球等票务平台': '票务平台',
    '朋友圈/朋友推荐': '朋友推荐',
    '偶像或乐队官方账号/工作室': '官方账号',
    '演出主办方/场馆官方宣传': '主办方宣传',
}
channel_items = channel_items.map(lambda x: label_map.get(x, x))
ch_counts = channel_items.value_counts()
n_resp = channel_split.shape[0]

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor('white')
colors_ch = [PALETTE9[i % len(PALETTE9)] for i in range(len(ch_counts))]
bars = ax.barh(ch_counts.index[::-1], ch_counts.values[::-1],
               color=colors_ch[::-1], edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, ch_counts.values[::-1]):
    pct = val / n_resp * 100
    ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
            f'{val}  ({pct:.0f}%)', va='center', fontsize=10, color='#333')
ax.set_xlim(0, ch_counts.max() * 1.3)
ax.set_xlabel('提及次数（多选，n=208）', fontsize=11)
ax.set_title('图3  信息获取渠道分布（多选）', fontsize=13, fontweight='bold', pad=10)
ax.spines[['top', 'right']].set_visible(False)
ax.tick_params(axis='y', labelsize=10.5)
# 添加参考线
ax.axvline(n_resp * 0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
ax.text(n_resp * 0.5 + 1, 0.2, '50%渗透率', fontsize=8.5, color='gray')
plt.tight_layout()
save(fig, 'fig3_info_channels')


# ══════════════════════════════════════════════════════════════════════════════
# 图4  消费结构分布（非门票 + 周边月均，双列并排）
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle('图4  消费结构分布', fontsize=14, fontweight='bold', y=1.02)
fig.patch.set_facecolor('white')

# (a) 非门票消费
ax = axes[0]
nontix_order = ['200元及以下', '200-500元', '501-1000元',
                '1001-2000元', '2001元及以上']
nontix_vals_raw = df_raw[nontix_col].value_counts()
nontix_vals = [nontix_vals_raw.get(k, 0) for k in nontix_order]
n_nontix = sum(nontix_vals)
bars = ax.bar(range(len(nontix_order)), nontix_vals,
              color=[C_LIGHT, C_TEAL, C_BLUE, C_ORANGE, C_RED],
              edgecolor='white', linewidth=1.5, width=0.65)
for bar, val in zip(bars, nontix_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f'{val}\n({val/n_nontix*100:.0f}%)', ha='center', va='bottom',
            fontsize=9.5, fontweight='bold')
ax.set_xticks(range(len(nontix_order)))
ax.set_xticklabels(['≤200元', '200-500元', '501-1000元',
                    '1001-2000元', '≥2001元'], rotation=20, ha='right', fontsize=9.5)
ax.set_ylabel('人数', fontsize=11)
ax.set_title('(a) 每场非门票类消费（含交通/住宿/餐饮/周边）', fontsize=11, fontweight='bold')
ax.set_ylim(0, max(nontix_vals) * 1.3)
ax.spines[['top', 'right']].set_visible(False)
# 中位区间箭头标注
ax.annotate('', xy=(1.5, nontix_vals[1] + 5), xytext=(1.5, nontix_vals[1] + 18),
            arrowprops=dict(arrowstyle='->', color=C_ORANGE, lw=2))
ax.text(1.5, nontix_vals[1] + 21, '集中于200-500元\n(39%)', ha='center',
        fontsize=9, color=C_ORANGE, fontweight='bold')

# (b) 周边月均消费
ax = axes[1]
merch_order = ['50元及以下', '50-200元', '201-500元',
               '501-2000元', '2001元及以上']
merch_vals_raw = df_raw[merch_col].value_counts()
merch_vals = [merch_vals_raw.get(k, 0) for k in merch_order]
# exclude 跳过
n_merch = sum(merch_vals)
bars = ax.bar(range(len(merch_order)), merch_vals,
              color=[C_LIGHT, C_TEAL, C_BLUE, C_AMBER, C_ORANGE],
              edgecolor='white', linewidth=1.5, width=0.65)
for bar, val in zip(bars, merch_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val}\n({val/n_merch*100:.0f}%)', ha='center', va='bottom',
            fontsize=9.5, fontweight='bold')
ax.set_xticks(range(len(merch_order)))
ax.set_xticklabels(['≤50元', '50-200元', '201-500元',
                    '501-2000元', '≥2001元'], rotation=20, ha='right', fontsize=9.5)
ax.set_ylabel('人数', fontsize=11)
ax.set_title('(b) 周边月均消费（高消费粉丝群体）', fontsize=11, fontweight='bold')
ax.set_ylim(0, max(merch_vals) * 1.35)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout(pad=2.5)
save(fig, 'fig4_consumption')


# ══════════════════════════════════════════════════════════════════════════════
# 图5  九维度量表均值雷达图
# ══════════════════════════════════════════════════════════════════════════════
keys   = list(CONSTRUCTS.keys())
labels = [CONSTRUCTS[k]['label'] for k in keys]
means  = [score_df[k].mean() for k in keys]
N_axes = len(keys)

angles = np.linspace(0, 2 * np.pi, N_axes, endpoint=False).tolist()
angles += angles[:1]
means_plot = means + means[:1]

fig = plt.figure(figsize=(9, 8))
fig.patch.set_facecolor('white')
ax = fig.add_subplot(111, polar=True)

# 背景环
for r in [2, 3, 4, 5]:
    ax.plot(angles, [r] * (N_axes + 1), '--', color='#cccccc', linewidth=0.7, zorder=1)
    if r == 3:
        ax.text(0, r + 0.05, '3', ha='center', fontsize=8, color='#999')
    if r == 4:
        ax.text(0, r + 0.05, '4', ha='center', fontsize=8, color='#999')

# 填充区域
ax.fill(angles, means_plot, color=C_TEAL, alpha=0.25, zorder=2)
ax.plot(angles, means_plot, 'o-', color=C_TEAL, linewidth=2.5,
        markersize=8, markerfacecolor='white', markeredgewidth=2.5, zorder=3)

# 数值标注
for angle, val, label in zip(angles[:-1], means, labels):
    offset = 0.18
    ax.text(angle, val + offset, f'{val:.2f}', ha='center', va='center',
            fontsize=9.5, fontweight='bold', color=C_BLUE,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor='none', alpha=0.8))

# 标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=10, fontweight='bold', color='#333')
ax.tick_params(axis='x', pad=15)
ax.set_ylim(1, 5.3)
ax.set_yticks([])
ax.spines['polar'].set_visible(False)
ax.set_title('图5  九大构念量表均值剖面（雷达图）', fontsize=13, fontweight='bold',
             pad=25, color='#222')

# 基准线说明
ax.plot(angles, [3.5] * (N_axes + 1), '-', color=C_AMBER, linewidth=1.5,
        alpha=0.7, zorder=2, label='中性基准（3.5）')
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.12), fontsize=10)

plt.tight_layout()
save(fig, 'fig5_radar')


# ══════════════════════════════════════════════════════════════════════════════
# 图6  有/无偶像群体在关键维度的得分对比（分组条形图）
# ══════════════════════════════════════════════════════════════════════════════
idol_mask_yes = df_raw[idol_col].isin(['1位', '2-3位', '4位及以上'])
idol_mask_no  = df_raw[idol_col] == '无'

compare_keys = ['SMI', 'PSR', 'EEM', 'GBI', 'RSA', 'PVI', 'TWI']
compare_labels = [CONSTRUCTS[k]['label'] for k in compare_keys]

means_yes = [score_df.loc[idol_mask_yes, k].mean() for k in compare_keys]
means_no  = [score_df.loc[idol_mask_no,  k].mean() for k in compare_keys]
sems_yes  = [score_df.loc[idol_mask_yes, k].sem()  for k in compare_keys]
sems_no   = [score_df.loc[idol_mask_no,  k].sem()  for k in compare_keys]

x = np.arange(len(compare_keys))
width = 0.38

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor('white')

bars1 = ax.bar(x - width/2, means_yes, width, label=f'有偶像 (n={idol_mask_yes.sum()})',
               color=C_BLUE, edgecolor='white', linewidth=1.5,
               yerr=sems_yes, capsize=4, error_kw=dict(elinewidth=1.2, ecolor='#888'))
bars2 = ax.bar(x + width/2, means_no,  width, label=f'无偶像 (n={idol_mask_no.sum()})',
               color=C_AMBER, edgecolor='white', linewidth=1.5,
               yerr=sems_no,  capsize=4, error_kw=dict(elinewidth=1.2, ecolor='#888'))

# 标注均值
for bar, val in zip(bars1, means_yes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08 + 0.01,
            f'{val:.2f}', ha='center', fontsize=8.5, fontweight='bold', color=C_BLUE)
for bar, val in zip(bars2, means_no):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08 + 0.01,
            f'{val:.2f}', ha='center', fontsize=8.5, fontweight='bold', color='#A0522D')

# 显著性标注（简单差异标记）
for i, (m1, m2) in enumerate(zip(means_yes, means_no)):
    diff = abs(m1 - m2)
    if diff > 0.4:
        ymax = max(m1, m2) + 0.22
        ax.annotate('', xy=(x[i] + width/2, ymax), xytext=(x[i] - width/2, ymax),
                    arrowprops=dict(arrowstyle='-', color='#555', lw=1.2))
        ax.text(x[i], ymax + 0.05, f'Δ{diff:.2f}**', ha='center',
                fontsize=8.5, color=C_ORANGE, fontweight='bold')

ax.set_ylim(1, 5.5)
ax.set_xticks(x)
ax.set_xticklabels(compare_labels, fontsize=10.5, rotation=15, ha='right')
ax.set_ylabel('均值得分（1-5分）', fontsize=11)
ax.set_title('图6  有无偶像群体在各动机维度得分对比（含误差棒）', fontsize=13,
             fontweight='bold', pad=10)
ax.legend(fontsize=11, loc='upper right')
ax.spines[['top', 'right']].set_visible(False)
ax.axhline(3.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
ax.text(len(compare_keys) - 0.5, 3.55, '中性基准', fontsize=8.5, color='gray')

plt.tight_layout()
save(fig, 'fig6_idol_comparison')


# ══════════════════════════════════════════════════════════════════════════════
# 图7  收入层 × 座位偏好（100%堆叠条形图）
# ══════════════════════════════════════════════════════════════════════════════
# 过滤掉 '不愿透露' 和 '跳过'
inc_order_plot = ['1000 元及以下', '1001-3000 元', '3001-6000 元',
                  '6001-10000 元', '10001 元及以上']
inc_labels_plot = ['≤1000元', '1001-3000元', '3001-6000元',
                   '6001-10000元', '≥10001元']
seat_cats = ['基础档（蓝色方框区域）', '进阶档（黄色方框区域）', '高端档（红色方框区域）']
seat_labels = ['基础档', '进阶档', '高端档']
seat_colors = [C_LIGHT, C_TEAL, C_BLUE]

# 构建交叉表
cross = pd.crosstab(df_raw[inc_col], df_raw[seat_col])
cross = cross.reindex(inc_order_plot, fill_value=0)
cross = cross.reindex(columns=seat_cats, fill_value=0)
cross_pct = cross.div(cross.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor('white')

bottom = np.zeros(len(inc_order_plot))
for i, (cat, label, color) in enumerate(zip(seat_cats, seat_labels, seat_colors)):
    vals = cross_pct[cat].values
    bars = ax.bar(inc_labels_plot, vals, bottom=bottom, label=label,
                  color=color, edgecolor='white', linewidth=1.5)
    for j, (bar, val) in enumerate(zip(bars, vals)):
        if val > 6:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bottom[j] + val/2,
                    f'{val:.0f}%', ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white')
    bottom += vals

# 在条形顶部标注样本量
for j, inc in enumerate(inc_order_plot):
    n_inc = cross.loc[inc].sum()
    ax.text(j, 102, f'n={n_inc}', ha='center', va='bottom', fontsize=9.5, color='#555')

ax.set_ylim(0, 115)
ax.set_ylabel('比例（%）', fontsize=11)
ax.set_xlabel('月可支配收入区间', fontsize=11)
ax.set_title('图7  不同收入层级的座位档次偏好（100%堆叠图）', fontsize=13,
             fontweight='bold', pad=10)
ax.legend(fontsize=11, loc='upper left', framealpha=0.85)
ax.spines[['top', 'right']].set_visible(False)
plt.setp(ax.get_xticklabels(), fontsize=10.5)

plt.tight_layout()
save(fig, 'fig7_income_seat')


# ══════════════════════════════════════════════════════════════════════════════
# 图8  各构念得分分布（小提琴图 + 箱形图叠加）
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor('white')

plot_data = []
for k in keys:
    plot_data.append(score_df[k].dropna().values)

vp = ax.violinplot(plot_data, positions=range(len(keys)),
                   showmedians=False, showextrema=False, widths=0.75)

# 自定义小提琴颜色
for i, (body, color) in enumerate(zip(vp['bodies'], PALETTE9)):
    body.set_facecolor(color)
    body.set_edgecolor('white')
    body.set_alpha(0.7)
    body.set_linewidth(1.5)

# 叠加箱型图
bp = ax.boxplot(plot_data, positions=range(len(keys)), widths=0.15,
                patch_artist=True, notch=False,
                medianprops=dict(color='white', linewidth=2.5),
                boxprops=dict(facecolor='none', edgecolor='#333', linewidth=1.5),
                whiskerprops=dict(color='#555', linewidth=1.2),
                capprops=dict(color='#555', linewidth=1.5),
                flierprops=dict(marker='o', color='#999', markersize=3, alpha=0.6))

# 标注均值点
for i, k in enumerate(keys):
    m = score_df[k].mean()
    ax.plot(i, m, 'D', color='white', markersize=7, zorder=5,
            markeredgecolor='#333', markeredgewidth=1.5)
    ax.text(i, 1.05, f'{m:.2f}', ha='center', fontsize=9,
            fontweight='bold', color=PALETTE9[i])

ax.set_xticks(range(len(keys)))
ax.set_xticklabels([CONSTRUCTS[k]['label'] for k in keys],
                   rotation=22, ha='right', fontsize=10)
ax.set_ylabel('量表得分（1-5分）', fontsize=11)
ax.set_ylim(0.8, 5.5)
ax.set_yticks([1, 2, 3, 4, 5])
ax.axhline(3.5, color='#aaa', linestyle='--', linewidth=1, alpha=0.7)
ax.text(len(keys) - 0.4, 3.55, '中性基准3.5', fontsize=8.5, color='gray')
ax.set_title('图8  各构念量表得分分布（小提琴图 + 箱形图）', fontsize=13,
             fontweight='bold', pad=10)
ax.spines[['top', 'right']].set_visible(False)

# 图例说明
from matplotlib.lines import Line2D
legend_elems = [
    mpatches.Patch(facecolor=C_GRAY, alpha=0.7, label='分布密度（小提琴）'),
    Line2D([0], [0], marker='D', color='white', markeredgecolor='#333',
           markersize=7, label='均值（◆）'),
    Line2D([0], [0], color='white', linewidth=2.5,
           markeredgecolor='#333', label='中位数（箱内白线）'),
]
ax.legend(handles=legend_elems, loc='upper left', fontsize=9.5, framealpha=0.9)

plt.tight_layout()
save(fig, 'fig8_violin')

print('\n全部图表已生成到 charts/ 目录。')
