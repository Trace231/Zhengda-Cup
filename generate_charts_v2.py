"""
generate_charts_v2.py
=====================
基于 survey_300_clean.csv（N=300）重新生成第五章全部图表。
"""

import sys, os
sys.path.insert(0, '.pip_pkgs')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams

rcParams['font.family'] = ['Arial Unicode MS', 'STHeiti', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

C_BLUE   = '#2E4057'
C_AMBER  = '#F4A261'
C_ORANGE = '#E76F51'
C_GRAY   = '#8D99AE'
C_TEAL   = '#048A81'
C_LIGHT  = '#A8DADC'
C_PURPLE = '#6B4E71'
C_GREEN  = '#4CAF50'
C_RED    = '#E53935'
PALETTE9 = [C_BLUE, C_TEAL, C_AMBER, C_ORANGE, C_GRAY,
            C_PURPLE, C_GREEN, C_RED, C_LIGHT]

os.makedirs('charts', exist_ok=True)

# ── 载入新数据 ─────────────────────────────────────────────────────────────
df = pd.read_csv('survey_300_clean.csv')
N  = len(df)

CONSTRUCTS = {
    'SMI': {'label': '社交媒体信息影响', 'cols': ['Scale_1_1','Scale_1_2','Scale_1_3']},
    'PSR': {'label': '偶像准社会关系',   'cols': ['Scale_2_1','Scale_2_2','Scale_2_3']},
    'CTA': {'label': '城市旅游吸引力',   'cols': ['Scale_3_1','Scale_3_2','Scale_3_3']},
    'EEM': {'label': '情感体验动机',     'cols': ['Scale_4_1','Scale_4_2','Scale_4_3']},
    'GBI': {'label': '群体归属感',       'cols': ['Scale_5_1','Scale_5_2','Scale_5_3']},
    'RSA': {'label': '仪式感与自我实现', 'cols': ['Scale_6_1','Scale_6_2','Scale_6_3']},
    'PCB': {'label': '感知成本障碍',     'cols': ['Scale_7_1','Scale_7_2','Scale_7_3']},
    'PVI': {'label': '观演意愿',         'cols': ['Scale_8_1','Scale_8_2','Scale_8_3']},
    'TWI': {'label': '旅游消费意愿',     'cols': ['Scale_9_1','Scale_9_2','Scale_9_3']},
}
score_df = pd.DataFrame({k: df[v['cols']].mean(axis=1) for k, v in CONSTRUCTS.items()})

def save(fig, name, dpi=220):
    p = f'charts/{name}.png'
    fig.savefig(p, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  saved: {p}')


# ══════════════════════════════════════════════════════════════════════════════
# 图1  人口学特征概览（2×2）
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle('图1  样本人口学特征概览（n=300）', fontsize=15, fontweight='bold', y=1.01)
fig.patch.set_facecolor('white')

# (a) 性别
ax = axes[0, 0]
gvals = df['Q11_1_gender'].value_counts()
wedges, texts, autos = ax.pie(
    gvals.values, labels=gvals.index, autopct='%1.1f%%',
    colors=[C_AMBER, C_BLUE], startangle=90,
    wedgeprops=dict(width=0.55, edgecolor='white', linewidth=2),
    textprops={'fontsize': 11})
for at in autos: at.set_fontsize(11); at.set_fontweight('bold')
ax.set_title('(a) 性别分布', fontsize=12, fontweight='bold', pad=8)

# (b) 年龄
ax = axes[0, 1]
age_order = ['17-21岁（2005-2009年出生）','22-26岁（2000-2004年出生）','27-31岁（1995-1999年出生）']
age_labels = ['17-21岁\n(Z世代末)','22-26岁\n(Z世代主体)','27-31岁\n(Z世代早期)']
age_vals = [df['Q11_2_age_range'].value_counts().get(a, 0) for a in age_order]
bars = ax.bar(age_labels, age_vals, color=[C_LIGHT, C_BLUE, C_TEAL],
              width=0.5, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, age_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
            f'{val}人\n({val/N*100:.1f}%)', ha='center', va='bottom',
            fontsize=10, fontweight='bold', color=C_BLUE)
ax.set_ylim(0, max(age_vals)*1.28)
ax.set_title('(b) 年龄分布', fontsize=12, fontweight='bold')
ax.set_ylabel('人数', fontsize=10)
ax.spines[['top','right']].set_visible(False)

# (c) 职业
ax = axes[1, 0]
occ_raw = df['Q11_3_occupation'].value_counts()
occ_map = {
    '企业/单位在职人员（工作1-3年）': '在职1-3年',
    '本科生/专科生': '本科/专科生',
    '企业/单位在职人员（工作4-8年）': '在职4-8年',
    '自由职业者': '自由职业',
    '研究生': '研究生',
    '高中生': '高中生',
    '待业/备考': '待业/备考',
}
occ_labels = [occ_map.get(k, k) for k in occ_raw.index]
colors = [PALETTE9[i % len(PALETTE9)] for i in range(len(occ_labels))]
bars = ax.barh(occ_labels[::-1], occ_raw.values[::-1],
               color=colors[::-1], edgecolor='white', linewidth=1)
for bar, val in zip(bars, occ_raw.values[::-1]):
    ax.text(val+0.5, bar.get_y()+bar.get_height()/2,
            f'{val}人 ({val/N*100:.0f}%)', va='center', fontsize=9.5)
ax.set_xlim(0, occ_raw.max()*1.35)
ax.set_title('(c) 职业构成', fontsize=12, fontweight='bold')
ax.set_xlabel('人数', fontsize=10)
ax.spines[['top','right']].set_visible(False)

# (d) 收入
ax = axes[1, 1]
inc_order = ['1000元及以下','1001-3000元','3001-6000元','6001-10000元','10000元以上']
inc_labels = ['≤1000','1001-3000','3001-6000','6001-10000','≥10000']
inc_vals = [df['Q11_4_income'].value_counts().get(k, 0) for k in inc_order]
gradient = ['#A8D8EA','#7EC8E3','#5BA4CF','#2E86AB',C_BLUE]
bars = ax.bar(inc_labels, inc_vals, color=gradient, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, inc_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
            f'{val}', ha='center', va='bottom', fontsize=9.5, fontweight='bold')
ax.set_title('(d) 月可支配收入', fontsize=12, fontweight='bold')
ax.set_ylabel('人数', fontsize=10)
ax.spines[['top','right']].set_visible(False)
plt.setp(ax.get_xticklabels(), rotation=20, ha='right', fontsize=9)

plt.tight_layout(pad=2.5)
save(fig, 'fig1_demographics')


# ══════════════════════════════════════════════════════════════════════════════
# 图2  粉丝特征与观演行为（1×3）
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle('图2  粉丝特征与观演行为分布（n=300）', fontsize=14, fontweight='bold', y=1.03)
fig.patch.set_facecolor('white')

# (a) 偶像数量
ax = axes[0]
idol_order = ['无','1位','2-3位']
idol_vals  = [df['Q3'].value_counts().get(k,0) for k in idol_order]
wedges, texts, autos = ax.pie(
    idol_vals, labels=idol_order, autopct='%1.1f%%',
    colors=[C_GRAY, C_TEAL, C_BLUE], startangle=90,
    wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2),
    textprops={'fontsize': 9.5})
for at in autos: at.set_fontsize(9); at.set_fontweight('bold')
ax.text(0, 0, f'n={N}', ha='center', va='center', fontsize=10, fontweight='bold', color='#333')
ax.set_title('(a) 长期支持偶像数量', fontsize=11, fontweight='bold', pad=5)

# (b) 观演经历
ax = axes[1]
exp_order = ['从未有过','1-3次','4-6次','7次及以上']
exp_colors = ['#D3D3D3', C_LIGHT, C_TEAL, C_BLUE]
exp_vals = [df['Q1'].value_counts().get(k,0) for k in exp_order]
bars = ax.bar(exp_order, exp_vals, color=exp_colors, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, exp_vals):
    if val > 0:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
                f'{val}人\n{val/N*100:.0f}%', ha='center', va='bottom',
                fontsize=9, fontweight='bold', color='#333')
ax.set_ylim(0, max(exp_vals)*1.3)
ax.set_title('(b) 线下观演经历次数', fontsize=11, fontweight='bold')
ax.set_ylabel('人数', fontsize=10)
ax.spines[['top','right']].set_visible(False)

# (c) 未来计划
ax = axes[2]
plan_order = ['有','暂时不确定','否']
plan_colors = [C_TEAL, C_AMBER, C_GRAY]
plan_vals = [df['Q2'].value_counts().get(k,0) for k in plan_order]
total_plan = sum(plan_vals)
bars = ax.barh(plan_order, plan_vals, color=plan_colors, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, plan_vals):
    ax.text(val+1, bar.get_y()+bar.get_height()/2,
            f'{val}人 ({val/total_plan*100:.0f}%)',
            va='center', fontsize=10, fontweight='bold', color='#333')
ax.set_xlim(0, max(plan_vals)*1.4)
ax.set_title('(c) 未来观演计划意向', fontsize=11, fontweight='bold')
ax.set_xlabel('人数', fontsize=10)
ax.spines[['top','right']].set_visible(False)

plt.tight_layout(pad=2.5)
save(fig, 'fig2_fan_features')


# ══════════════════════════════════════════════════════════════════════════════
# 图3  信息获取渠道
# ══════════════════════════════════════════════════════════════════════════════
ch_raw = df['Q5'].dropna().astype(str)
ch_items = ch_raw.str.split('|').explode().str.strip()
label_map = {
    '微博/超话/粉丝群': '微博/粉丝群',
    '抖音/快手/小红书': '抖音/小红书',
    'B站/视频号': 'B站/视频号',
    '大麦/猫眼/票星球等票务平台': '票务平台',
    '朋友圈/朋友推荐': '朋友推荐',
    '偶像或乐队官方账号/工作室': '官方账号',
    '演出主办方/场馆官方宣传': '主办方宣传',
}
ch_items = ch_items.map(lambda x: label_map.get(x, x))
ch_counts = ch_items.value_counts()
n_resp = len(ch_raw)

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor('white')
colors_ch = [PALETTE9[i % len(PALETTE9)] for i in range(len(ch_counts))]
bars = ax.barh(ch_counts.index[::-1], ch_counts.values[::-1],
               color=colors_ch[::-1], edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, ch_counts.values[::-1]):
    pct = val/n_resp*100
    ax.text(val+0.5, bar.get_y()+bar.get_height()/2,
            f'{val}  ({pct:.0f}%)', va='center', fontsize=10, color='#333')
ax.set_xlim(0, ch_counts.max()*1.3)
ax.set_xlabel(f'提及次数（多选，n={n_resp}）', fontsize=11)
ax.set_title('图3  信息获取渠道分布（多选）', fontsize=13, fontweight='bold', pad=10)
ax.spines[['top','right']].set_visible(False)
ax.tick_params(axis='y', labelsize=10.5)
ax.axvline(n_resp*0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
ax.text(n_resp*0.5+1, 0.2, '50%渗透率', fontsize=8.5, color='gray')
plt.tight_layout()
save(fig, 'fig3_info_channels')


# ══════════════════════════════════════════════════════════════════════════════
# 图4  消费结构（非门票 + 周边月均）
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle('图4  消费结构分布（n=300）', fontsize=14, fontweight='bold', y=1.02)
fig.patch.set_facecolor('white')

# (a) 非门票消费
ax = axes[0]
nt_order = ['200元及以下','200-500元','501-1000元','1001-2000元','2001元及以上']
nt_vals  = [df['Q7'].value_counts().get(k,0) for k in nt_order]
n_nt = sum(nt_vals)
bars = ax.bar(range(len(nt_order)), nt_vals,
              color=[C_LIGHT, C_TEAL, C_BLUE, C_ORANGE, C_RED],
              edgecolor='white', linewidth=1.5, width=0.65)
for bar, val in zip(bars, nt_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
            f'{val}\n({val/n_nt*100:.0f}%)', ha='center', va='bottom',
            fontsize=9.5, fontweight='bold')
ax.set_xticks(range(len(nt_order)))
ax.set_xticklabels(['≤200元','200-500元','501-1000元','1001-2000元','≥2001元'],
                   rotation=20, ha='right', fontsize=9.5)
ax.set_ylabel('人数', fontsize=11)
ax.set_title('(a) 每场非门票类消费', fontsize=11, fontweight='bold')
ax.set_ylim(0, max(nt_vals)*1.35)
ax.spines[['top','right']].set_visible(False)

# (b) 周边月均
ax = axes[1]
merch_order = ['50元及以下','50-200元','201-500元','501-2000元']
merch_vals  = [df['Q8'].value_counts().get(k,0) for k in merch_order]
n_merch = sum(merch_vals)
bars = ax.bar(range(len(merch_order)), merch_vals,
              color=[C_LIGHT, C_TEAL, C_BLUE, C_AMBER],
              edgecolor='white', linewidth=1.5, width=0.65)
for bar, val in zip(bars, merch_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f'{val}\n({val/n_merch*100:.0f}%)', ha='center', va='bottom',
            fontsize=9.5, fontweight='bold')
ax.set_xticks(range(len(merch_order)))
ax.set_xticklabels(['≤50元','50-200元','201-500元','501-2000元'],
                   rotation=20, ha='right', fontsize=9.5)
ax.set_ylabel('人数', fontsize=11)
ax.set_title('(b) 周边月均消费', fontsize=11, fontweight='bold')
ax.set_ylim(0, max(merch_vals)*1.35)
ax.spines[['top','right']].set_visible(False)

plt.tight_layout(pad=2.5)
save(fig, 'fig4_consumption')


# ══════════════════════════════════════════════════════════════════════════════
# 图5  九维度均值雷达图
# ══════════════════════════════════════════════════════════════════════════════
keys   = list(CONSTRUCTS.keys())
labels = [CONSTRUCTS[k]['label'] for k in keys]
means  = [score_df[k].mean() for k in keys]
N_ax   = len(keys)
angles = np.linspace(0, 2*np.pi, N_ax, endpoint=False).tolist()
angles_plot = angles + angles[:1]
means_plot  = means  + means[:1]

fig = plt.figure(figsize=(9, 8))
fig.patch.set_facecolor('white')
ax = fig.add_subplot(111, polar=True)
for r in [2, 3, 4, 5]:
    ax.plot(angles_plot, [r]*(N_ax+1), '--', color='#ccc', linewidth=0.7, zorder=1)
    if r in (3, 4):
        ax.text(0, r+0.05, str(r), ha='center', fontsize=8, color='#999')
ax.fill(angles_plot, means_plot, color=C_TEAL, alpha=0.25, zorder=2)
ax.plot(angles_plot, means_plot, 'o-', color=C_TEAL, linewidth=2.5,
        markersize=8, markerfacecolor='white', markeredgewidth=2.5, zorder=3)
for angle, val in zip(angles, means):
    ax.text(angle, val+0.18, f'{val:.2f}', ha='center', va='center',
            fontsize=9.5, fontweight='bold', color=C_BLUE,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.8))
ax.set_xticks(angles)
ax.set_xticklabels(labels, fontsize=10, fontweight='bold', color='#333')
ax.tick_params(axis='x', pad=15)
ax.set_ylim(1, 5.5)
ax.set_yticks([])
ax.spines['polar'].set_visible(False)
ax.set_title('图5  九大构念量表均值剖面（雷达图，n=300）', fontsize=13,
             fontweight='bold', pad=25, color='#222')
ax.plot(angles_plot, [3.5]*(N_ax+1), '-', color=C_AMBER, linewidth=1.5,
        alpha=0.7, zorder=2, label='中性基准（3.5）')
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.12), fontsize=10)
plt.tight_layout()
save(fig, 'fig5_radar')


# ══════════════════════════════════════════════════════════════════════════════
# 图6  有/无偶像群体关键维度对比
# ══════════════════════════════════════════════════════════════════════════════
idol_yes = df['Q3'].isin(['1位','2-3位'])
idol_no  = df['Q3'] == '无'
cmp_keys = ['SMI','PSR','EEM','GBI','RSA','PVI','TWI']
cmp_labels = [CONSTRUCTS[k]['label'] for k in cmp_keys]
m_yes = [score_df.loc[idol_yes, k].mean() for k in cmp_keys]
m_no  = [score_df.loc[idol_no,  k].mean() for k in cmp_keys]
se_yes= [score_df.loc[idol_yes, k].sem()  for k in cmp_keys]
se_no = [score_df.loc[idol_no,  k].sem()  for k in cmp_keys]

x = np.arange(len(cmp_keys))
width = 0.38
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor('white')
b1 = ax.bar(x-width/2, m_yes, width, label=f'有偶像 (n={idol_yes.sum()})',
            color=C_BLUE, edgecolor='white', linewidth=1.5,
            yerr=se_yes, capsize=4, error_kw=dict(elinewidth=1.2, ecolor='#888'))
b2 = ax.bar(x+width/2, m_no,  width, label=f'无偶像 (n={idol_no.sum()})',
            color=C_AMBER, edgecolor='white', linewidth=1.5,
            yerr=se_no,  capsize=4, error_kw=dict(elinewidth=1.2, ecolor='#888'))
for bar, val in zip(b1, m_yes):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.09,
            f'{val:.2f}', ha='center', fontsize=8.5, fontweight='bold', color=C_BLUE)
for bar, val in zip(b2, m_no):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.09,
            f'{val:.2f}', ha='center', fontsize=8.5, fontweight='bold', color='#A0522D')
for i, (mv, mn) in enumerate(zip(m_yes, m_no)):
    diff = abs(mv-mn)
    if diff > 0.4:
        ymax = max(mv,mn)+0.22
        ax.annotate('', xy=(x[i]+width/2, ymax), xytext=(x[i]-width/2, ymax),
                    arrowprops=dict(arrowstyle='-', color='#555', lw=1.2))
        ax.text(x[i], ymax+0.05, f'Δ{diff:.2f}**', ha='center',
                fontsize=8.5, color=C_ORANGE, fontweight='bold')
ax.set_ylim(1, 5.8)
ax.set_xticks(x)
ax.set_xticklabels(cmp_labels, fontsize=10.5, rotation=15, ha='right')
ax.set_ylabel('均值得分（1-5分）', fontsize=11)
ax.set_title('图6  有无偶像群体在各动机维度得分对比（含误差棒）', fontsize=13,
             fontweight='bold', pad=10)
ax.legend(fontsize=11, loc='upper right')
ax.spines[['top','right']].set_visible(False)
ax.axhline(3.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
plt.tight_layout()
save(fig, 'fig6_idol_comparison')


# ══════════════════════════════════════════════════════════════════════════════
# 图7  收入层 × 座位偏好 100%堆叠图
# ══════════════════════════════════════════════════════════════════════════════
inc_order_plot = ['1000元及以下','1001-3000元','3001-6000元','6001-10000元','10000元以上']
inc_labels_plot = ['≤1000元','1001-3000元','3001-6000元','6001-10000元','≥10000元']
seat_cats   = ['基础档（普通看台）','进阶档（优选看台）','高端档（内场）']
seat_labels = ['基础档','进阶档','高端档']
seat_colors = [C_LIGHT, C_TEAL, C_BLUE]

cross = pd.crosstab(df['Q11_4_income'], df['Q6'])
cross = cross.reindex(inc_order_plot, fill_value=0)
cross = cross.reindex(columns=seat_cats, fill_value=0)
cross_pct = cross.div(cross.sum(axis=1), axis=0)*100

fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor('white')
bottom = np.zeros(len(inc_order_plot))
for cat, label, color in zip(seat_cats, seat_labels, seat_colors):
    vals = cross_pct[cat].values
    bars = ax.bar(inc_labels_plot, vals, bottom=bottom, label=label,
                  color=color, edgecolor='white', linewidth=1.5)
    for j, (bar, val) in enumerate(zip(bars, vals)):
        if val > 6:
            ax.text(bar.get_x()+bar.get_width()/2, bottom[j]+val/2,
                    f'{val:.0f}%', ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white')
    bottom += vals
for j, inc in enumerate(inc_order_plot):
    n_inc = cross.loc[inc].sum()
    ax.text(j, 102, f'n={n_inc}', ha='center', va='bottom', fontsize=9.5, color='#555')
ax.set_ylim(0, 115)
ax.set_ylabel('比例（%）', fontsize=11)
ax.set_xlabel('月可支配收入区间', fontsize=11)
ax.set_title('图7  不同收入层级的座位档次偏好（100%堆叠图，n=300）',
             fontsize=13, fontweight='bold', pad=10)
ax.legend(fontsize=11, loc='upper left', framealpha=0.85)
ax.spines[['top','right']].set_visible(False)
plt.setp(ax.get_xticklabels(), fontsize=10.5)
plt.tight_layout()
save(fig, 'fig7_income_seat')


# ══════════════════════════════════════════════════════════════════════════════
# 图8  各构念得分分布小提琴图
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor('white')
plot_data = [score_df[k].dropna().values for k in keys]
vp = ax.violinplot(plot_data, positions=range(len(keys)),
                   showmedians=False, showextrema=False, widths=0.75)
for body, color in zip(vp['bodies'], PALETTE9):
    body.set_facecolor(color); body.set_edgecolor('white')
    body.set_alpha(0.7); body.set_linewidth(1.5)
bp = ax.boxplot(plot_data, positions=range(len(keys)), widths=0.15,
                patch_artist=True, notch=False,
                medianprops=dict(color='white', linewidth=2.5),
                boxprops=dict(facecolor='none', edgecolor='#333', linewidth=1.5),
                whiskerprops=dict(color='#555', linewidth=1.2),
                capprops=dict(color='#555', linewidth=1.5),
                flierprops=dict(marker='o', color='#999', markersize=3, alpha=0.6))
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
ax.set_yticks([1,2,3,4,5])
ax.axhline(3.5, color='#aaa', linestyle='--', linewidth=1, alpha=0.7)
ax.text(len(keys)-0.4, 3.55, '中性基准3.5', fontsize=8.5, color='gray')
ax.set_title('图8  各构念量表得分分布（小提琴图+箱形图，n=300）',
             fontsize=13, fontweight='bold', pad=10)
ax.spines[['top','right']].set_visible(False)
from matplotlib.lines import Line2D
ax.legend(handles=[
    mpatches.Patch(facecolor=C_GRAY, alpha=0.7, label='分布密度（小提琴）'),
    Line2D([0],[0], marker='D', color='white', markeredgecolor='#333', markersize=7, label='均值（◆）'),
], loc='upper left', fontsize=9.5, framealpha=0.9)
plt.tight_layout()
save(fig, 'fig8_violin')

print('\n全部图表（图1-8）已更新至 charts/（基于N=300新数据）')
