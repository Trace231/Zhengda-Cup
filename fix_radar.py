"""
fix_radar.py  —  重生 fig5 双组对比雷达图
两个子图：左=有无偶像分组，右=高低频观演分组
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

os.makedirs('charts', exist_ok=True)

df = pd.read_csv('survey_300_clean.csv')
N  = len(df)

CONSTRUCTS = {
    'SMI': {'label': '社交媒体\n信息影响', 'cols': ['Scale_1_1','Scale_1_2','Scale_1_3']},
    'PSR': {'label': '偶像准\n社会关系',   'cols': ['Scale_2_1','Scale_2_2','Scale_2_3']},
    'CTA': {'label': '城市旅游\n吸引力',   'cols': ['Scale_3_1','Scale_3_2','Scale_3_3']},
    'EEM': {'label': '情感体验\n动机',     'cols': ['Scale_4_1','Scale_4_2','Scale_4_3']},
    'GBI': {'label': '群体\n归属感',       'cols': ['Scale_5_1','Scale_5_2','Scale_5_3']},
    'RSA': {'label': '仪式感\n/自我实现',  'cols': ['Scale_6_1','Scale_6_2','Scale_6_3']},
    'PCB': {'label': '感知成本\n障碍',     'cols': ['Scale_7_1','Scale_7_2','Scale_7_3']},
    'PVI': {'label': '观演\n意愿',         'cols': ['Scale_8_1','Scale_8_2','Scale_8_3']},
    'TWI': {'label': '旅游消费\n意愿',     'cols': ['Scale_9_1','Scale_9_2','Scale_9_3']},
}
keys   = list(CONSTRUCTS.keys())
labels = [CONSTRUCTS[k]['label'] for k in keys]
score_df = pd.DataFrame({k: df[v['cols']].mean(axis=1) for k, v in CONSTRUCTS.items()})

# 分组
idol_yes = df['Q3'].isin(['1位','2-3位'])
idol_no  = df['Q3'] == '无'
hf = df['Q1'].isin(['7次及以上','4-6次'])
lf = df['Q1'].isin(['1-3次','从未有过'])

def radar_means(mask):
    return [score_df.loc[mask, k].mean() for k in keys]

m_idol = radar_means(idol_yes)
m_noid = radar_means(idol_no)
m_hf   = radar_means(hf)
m_lf   = radar_means(lf)

n_ax  = len(keys)
angles = np.linspace(0, 2*np.pi, n_ax, endpoint=False).tolist()

def make_plot(ax, data_pairs, title, n_labels):
    """
    data_pairs: list of (values, color, fill_color, label)
    """
    a_full = angles + angles[:1]

    # 背景圆圈
    for r in [1, 2, 3, 4, 5]:
        ax.plot(a_full, [r]*(n_ax+1), '--', color='#E0E0E0', linewidth=0.6, zorder=1)
    ax.plot(a_full, [3.5]*(n_ax+1), '-', color='#BBBBBB', linewidth=1.0, alpha=0.8, zorder=1)

    for values, color, fill, label in data_pairs:
        v_full = values + values[:1]
        ax.fill(a_full, v_full, color=fill, alpha=0.25, zorder=2)
        ax.plot(a_full, v_full, 'o-', color=color, linewidth=2.2,
                markersize=5.5, markerfacecolor='white', markeredgewidth=2, zorder=3,
                label=label)
        # 数值标注（仅第一组标在外侧）
        for angle, val in zip(angles, values):
            ax.text(angle, val+0.22, f'{val:.2f}', ha='center', va='center',
                    fontsize=7.5, fontweight='bold', color=color,
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                              edgecolor='none', alpha=0.75))

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=9.5, fontweight='bold', color='#333')
    ax.tick_params(axis='x', pad=14)
    ax.set_ylim(0, 5.6)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['1','2','3','4','5'], fontsize=7.5, color='#999')
    ax.spines['polar'].set_visible(False)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=28, color='#222')

    # 差值注释：只标最大差值维度
    diffs = [abs(data_pairs[0][0][i] - data_pairs[1][0][i]) for i in range(n_ax)]
    max_i = int(np.argmax(diffs))
    ax.text(angles[max_i], 5.3,
            f'最大差异: {keys[max_i]}\nΔ={diffs[max_i]:.2f}',
            ha='center', fontsize=8, color='#555',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF8DC',
                      edgecolor='#E5C85A', alpha=0.95))

    ax.legend(loc='upper right', bbox_to_anchor=(1.45, 1.18),
              fontsize=10, framealpha=0.9, edgecolor='#ccc')

# ── 绘图 ────────────────────────────────────────────────────────────────────
C_BLUE  = '#2E4057'
C_TEAL  = '#048A81'
C_AMBER = '#F4A261'
C_RED   = '#E53935'

fig = plt.figure(figsize=(16, 7.5))
fig.patch.set_facecolor('white')

ax1 = fig.add_subplot(121, polar=True)
ax2 = fig.add_subplot(122, polar=True)

make_plot(ax1,
    [(m_idol, C_BLUE,  '#A8D8EA', f'有偶像 (n={idol_yes.sum()})'),
     (m_noid, C_AMBER, '#FFE8C8', f'无偶像 (n={idol_no.sum()})')],
    '(a) 分组对比：有无长期支持偶像',
    [f'n={idol_yes.sum()}', f'n={idol_no.sum()}']
)

make_plot(ax2,
    [(m_hf, C_TEAL, '#B2EBE7', f'高频观演 ≥4次 (n={hf.sum()})'),
     (m_lf, C_RED,  '#FFCDD2', f'中低频 ≤3次 (n={lf.sum()})')],
    '(b) 分组对比：观演经历高频 vs 中低频',
    [f'n={hf.sum()}', f'n={lf.sum()}']
)

# 基准线说明
for ax in [ax1, ax2]:
    ax.text(0, 3.72, '3.5', ha='center', fontsize=7.5, color='#999',
            transform=ax.transData)

fig.suptitle('图5  九大构念量表均值双组对比雷达图（n=300，Likert 1—5分）',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout(pad=3.0)

path = 'charts/fig5_radar.png'
fig.savefig(path, dpi=220, bbox_inches='tight', facecolor='white')
plt.close()
print(f'✓ 已保存: {path}')
print(f'有偶像 vs 无偶像差异最大: PSR(Δ={abs(m_idol[1]-m_noid[1]):.2f}), RSA(Δ={abs(m_idol[5]-m_noid[5]):.2f}), SMI(Δ={abs(m_idol[0]-m_noid[0]):.2f})')
print(f'高频 vs 低频差异最大: TWI(Δ={abs(m_hf[8]-m_lf[8]):.2f}), PVI(Δ={abs(m_hf[7]-m_lf[7]):.2f})')
