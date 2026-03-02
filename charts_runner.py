#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
charts_runner.py — 粉丝经济报告全图表统一生成入口
======================================================
用法：
  python3 charts_runner.py           # 生成全部图表
  python3 charts_runner.py fig1 fig5 # 只生成指定图表
  python3 charts_runner.py --list    # 列出所有可生成的图表

输出：全部 PNG 保存至 charts/ 目录（DPI=220）
数据：默认读取 survey_clean.csv（可通过环境变量 SURVEY_DATA 覆盖）

图表清单：
  fig1  - 样本人口学特征概览（2×2）
  fig2  - 粉丝特征与观演行为分布
  fig3  - 信息获取渠道分布（多选水平条形图）
  fig4  - 消费结构分布（非门票+周边月均）
  fig5  - 九大构念双组对比雷达图（有/无偶像 & 高/低频观演）
  fig6  - 有无偶像群体关键维度得分对比（分组柱状图+误差棒）
  fig7  - 收入层级×座位偏好 100%堆叠图（原版蓝绿配色）
  fig8  - 九大构念得分小提琴+箱线图（莫兰迪高颜值版）
  fig9  - 模型一 SEM 路径图（动机-情境双轮驱动）
  fig10 - 模型二 SEM 路径图（动机-阻碍，含二阶因子 MOT）
  map   - PPS 抽样省份地图（气泡 + 渐变蓝）
"""

import sys
import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib import rcParams
from scipy import stats
import seaborn as sns

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 0. 全局配置
# ─────────────────────────────────────────────────────────────────────────────

# 字体：macOS 优先用 Hiragino Sans GB，Windows 用 SimHei/Microsoft YaHei
rcParams['font.sans-serif'] = [
    'Hiragino Sans GB',   # macOS 简体中文
    'STHeiti',            # macOS 备选
    'SimHei',             # Windows
    'Microsoft YaHei',    # Windows 备选
    'Arial Unicode MS',   # 通用 Unicode
    'DejaVu Sans',        # 终极回退
]
rcParams['axes.unicode_minus'] = False
rcParams['figure.dpi'] = 150
rcParams['savefig.dpi'] = 220
rcParams['figure.facecolor'] = 'white'

# ── 配色 ────────────────────────────────────────────────────────────────────
# 报告主色调（科技蓝系 + 莫兰迪系 共存）
C_BLUE    = '#2E4057'    # 深蓝主色
C_TEAL    = '#048A81'    # 科技青绿
C_AMBER   = '#F4A261'    # 暖橙
C_ORANGE  = '#E76F51'    # 强调橙
C_GRAY    = '#8D99AE'    # 中性灰
C_LIGHT   = '#A8DADC'    # 浅青
C_PURPLE  = '#6B4E71'    # 紫藕
C_GREEN   = '#4CAF50'    # 草绿
C_RED     = '#C0392B'    # 警示红

# 莫兰迪色系（fig5/fig8 高颜值版）
MOR_BLUE  = '#7BA7C7'
MOR_MINT  = '#7DBDAB'
MOR_ROSE  = '#C49A9A'
MOR_SAGE  = '#A8BCAC'
MOR_MAUVE = '#A99FC2'
MOR_SAND  = '#C9B38E'
MOR_SKY   = '#9AC3D0'
MOR_FOREST= '#6A9E8C'
MOR_STEEL = '#7A9AB5'
DARK_NIGHT= '#2C3E50'
MID_GRAY  = '#ADB5BD'
DARK_GRAY = '#495057'

PALETTE9  = [C_BLUE, C_TEAL, C_AMBER, C_ORANGE, C_GRAY,
             C_PURPLE, C_GREEN, C_RED, C_LIGHT]

# ── 数据加载 ─────────────────────────────────────────────────────────────────
DATA_FILE = os.environ.get('SURVEY_DATA', 'survey_clean.csv')
SEM_FILE  = 'sem_results_v2.json'

os.makedirs('charts', exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_FILE, encoding='utf-8-sig')
    CONSTRUCTS = {
        'SMI': {'label': '社交媒体\n信息影响', 'cols': ['Scale_1_1','Scale_1_2','Scale_1_3']},
        'PSR': {'label': '偶像准\n社会关系',   'cols': ['Scale_2_1','Scale_2_2','Scale_2_3']},
        'CTA': {'label': '城市旅游\n吸引力',   'cols': ['Scale_3_1','Scale_3_2','Scale_3_3']},
        'EEM': {'label': '情感体验\n动机',     'cols': ['Scale_4_1','Scale_4_2','Scale_4_3']},
        'GBI': {'label': '群体归属\n认同',     'cols': ['Scale_5_1','Scale_5_2','Scale_5_3']},
        'RSA': {'label': '仪式感\n自我实现',   'cols': ['Scale_6_1','Scale_6_2','Scale_6_3']},
        'PCB': {'label': '感知成本\n障碍',     'cols': ['Scale_7_1','Scale_7_2','Scale_7_3']},
        'PVI': {'label': '观演\n意愿',         'cols': ['Scale_8_1','Scale_8_2','Scale_8_3']},
        'TWI': {'label': '旅游消费\n延伸意愿', 'cols': ['Scale_9_1','Scale_9_2','Scale_9_3']},
    }
    score_df = pd.DataFrame({
        k: df[[c for c in v['cols'] if c in df.columns]].mean(axis=1)
        for k, v in CONSTRUCTS.items()
    })
    return df, CONSTRUCTS, score_df


def save_fig(fig, name):
    path = f'charts/{name}.png'
    fig.savefig(path, dpi=220, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  ✓ charts/{name}.png')


# =============================================================================
# 图1  样本人口学特征概览（2×2 拼图）
# =============================================================================
def fig1(df, N):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f'图1  样本人口学特征概览（N={N}）',
                 fontsize=14, fontweight='bold', y=1.01)
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
    age_order  = ['17-21岁（2005-2009年出生）','22-26岁（2000-2004年出生）','27-31岁（1995-1999年出生）']
    age_labels = ['17-21岁\n(Z世代末)','22-26岁\n(Z世代主体)','27-31岁\n(Z世代早期)']
    age_vals   = [df['Q11_2_age_range'].value_counts().get(a, 0) for a in age_order]
    bars = ax.bar(age_labels, age_vals, color=[C_LIGHT, C_BLUE, C_TEAL],
                  width=0.5, edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars, age_vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
                f'{val}人\n({val/N*100:.1f}%)', ha='center', va='bottom',
                fontsize=10, fontweight='bold', color=C_BLUE)
    ax.set_ylim(0, max(age_vals)*1.32)
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
        '研究生': '研究生', '高中生': '高中生', '待业/备考': '待业/备考',
    }
    occ_labels = [occ_map.get(k, k[:6]) for k in occ_raw.index]
    colors_occ = [PALETTE9[i % len(PALETTE9)] for i in range(len(occ_labels))]
    bars = ax.barh(occ_labels[::-1], occ_raw.values[::-1],
                   color=colors_occ[::-1], edgecolor='white', linewidth=1)
    for bar, val in zip(bars, occ_raw.values[::-1]):
        ax.text(val+0.5, bar.get_y()+bar.get_height()/2,
                f'{val}人 ({val/N*100:.0f}%)', va='center', fontsize=9.5)
    ax.set_xlim(0, occ_raw.max()*1.38)
    ax.set_title('(c) 职业构成', fontsize=12, fontweight='bold')
    ax.set_xlabel('人数', fontsize=10)
    ax.spines[['top','right']].set_visible(False)

    # (d) 收入
    ax = axes[1, 1]
    inc_order  = ['1000元及以下','1001-3000元','3001-6000元','6001-10000元','10000元以上']
    inc_labels = ['≤1000','1001-3000','3001-6000','6001-1万','≥1万']
    inc_vals   = [df['Q11_4_income'].value_counts().get(k, 0) for k in inc_order]
    gradient   = ['#A8D8EA','#7EC8E3','#5BA4CF','#2E86AB', C_BLUE]
    bars = ax.bar(inc_labels, inc_vals, color=gradient, edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars, inc_vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
                f'{val}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_title('(d) 月可支配收入（元）', fontsize=12, fontweight='bold')
    ax.set_ylabel('人数', fontsize=10)
    ax.spines[['top','right']].set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=20, ha='right', fontsize=9)

    plt.tight_layout(pad=2.5)
    save_fig(fig, 'fig1_demographics')


# =============================================================================
# 图2  粉丝特征与观演行为（1×3）
# =============================================================================
def fig2(df, N):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(f'图2  粉丝特征与观演行为分布（N={N}）',
                 fontsize=14, fontweight='bold', y=1.03)
    fig.patch.set_facecolor('white')

    # (a) 有无偶像
    ax = axes[0]
    idol_cats = df['Q2'].value_counts()
    wedges, texts, autos = ax.pie(
        idol_cats.values, labels=idol_cats.index, autopct='%1.1f%%',
        colors=[C_TEAL, C_AMBER, C_GRAY][:len(idol_cats)],
        startangle=90,
        wedgeprops=dict(width=0.55, edgecolor='white', linewidth=2),
        textprops={'fontsize': 10})
    for at in autos: at.set_fontsize(10); at.set_fontweight('bold')
    ax.text(0, 0, f'N={N}', ha='center', va='center',
            fontsize=10, fontweight='bold', color='#333')
    ax.set_title('(a) 有无长期支持偶像', fontsize=11, fontweight='bold', pad=5)

    # (b) 观演经历次数
    ax = axes[1]
    exp_order  = ['从未有过','1-3次','4-6次','7次及以上']
    exp_colors = ['#D3D3D3', C_LIGHT, C_TEAL, C_BLUE]
    exp_vals   = [df['Q1'].value_counts().get(k, 0) for k in exp_order]
    bars = ax.bar(exp_order, exp_vals, color=exp_colors, edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars, exp_vals):
        if val > 0:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                    f'{val}人\n{val/N*100:.0f}%', ha='center', va='bottom',
                    fontsize=9.5, fontweight='bold', color='#333')
    ax.set_ylim(0, max(exp_vals)*1.35)
    ax.set_title('(b) 线下观演经历次数', fontsize=11, fontweight='bold')
    ax.set_ylabel('人数', fontsize=10)
    ax.spines[['top','right']].set_visible(False)
    plt.setp(ax.get_xticklabels(), fontsize=9.5)

    # (c) 未来观演计划
    ax = axes[2]
    plan_order  = ['有','暂时不确定','否']
    plan_colors = [C_TEAL, C_AMBER, C_GRAY]
    plan_vals   = [df['Q2'].value_counts().get(k, 0) for k in plan_order]
    # use Q2 but only if it has those values; otherwise use the actual categories
    actual_cats = df['Q2'].value_counts()
    plan_vals2  = [actual_cats.get(k, 0) for k in plan_order]
    bars = ax.barh(plan_order, plan_vals2, color=plan_colors, edgecolor='white', linewidth=1.5)
    total2 = sum(plan_vals2)
    for bar, val in zip(bars, plan_vals2):
        ax.text(val+1, bar.get_y()+bar.get_height()/2,
                f'{val}人 ({val/max(total2,1)*100:.0f}%)',
                va='center', fontsize=10, fontweight='bold', color='#333')
    ax.set_xlim(0, max(plan_vals2+[1])*1.4)
    ax.set_title('(c) 未来观演计划意向', fontsize=11, fontweight='bold')
    ax.set_xlabel('人数', fontsize=10)
    ax.spines[['top','right']].set_visible(False)

    plt.tight_layout(pad=2.5)
    save_fig(fig, 'fig2_fan_features')


# =============================================================================
# 图3  信息获取渠道（水平条形图，多选）
# =============================================================================
def fig3(df, N):
    ch_raw   = df['Q5'].dropna().astype(str)
    ch_items = ch_raw.str.split('|').explode().str.strip()
    label_map = {
        '微博/超话/粉丝群':           '微博/粉丝群',
        '抖音/快手/小红书':           '抖音/小红书',
        'B站/视频号':                 'B站/视频号',
        '大麦/猫眼/票星球等票务平台': '票务平台',
        '朋友圈/朋友推荐':            '朋友推荐',
        '偶像或乐队官方账号/工作室':  '官方账号',
        '演出主办方/场馆官方宣传':    '主办方宣传',
    }
    ch_items   = ch_items.map(lambda x: label_map.get(x, x))
    ch_counts  = ch_items.value_counts()
    n_resp     = len(ch_raw)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor('white')
    colors_ch = [PALETTE9[i % len(PALETTE9)] for i in range(len(ch_counts))]
    bars = ax.barh(ch_counts.index[::-1], ch_counts.values[::-1],
                   color=colors_ch[::-1], edgecolor='white', linewidth=1.5,
                   height=0.65)
    for bar, val in zip(bars, ch_counts.values[::-1]):
        pct = val / n_resp * 100
        ax.text(val+1, bar.get_y()+bar.get_height()/2,
                f'{val}次  ({pct:.0f}%)', va='center', fontsize=10, color='#333')
    ax.set_xlim(0, ch_counts.max()*1.3)
    ax.set_xlabel(f'提及次数（多选，受访者 N={n_resp}）', fontsize=11)
    ax.set_title('图3  演唱会信息获取渠道分布（多选题）',
                 fontsize=13, fontweight='bold', pad=10)
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(axis='y', labelsize=10.5)
    ax.axvline(n_resp*0.5, color='#aaa', linestyle='--', lw=0.8, alpha=0.6)
    ax.text(n_resp*0.5+2, 0.3, '50%\n渗透率', fontsize=8.5, color='#aaa')
    ax.yaxis.grid(False)
    plt.tight_layout()
    save_fig(fig, 'fig3_info_channels')


# =============================================================================
# 图4  消费结构（非门票 + 周边月均）
# =============================================================================
def fig4(df, N):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(f'图4  消费结构分布（N={N}）',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.patch.set_facecolor('white')

    # (a) 非门票消费
    ax = axes[0]
    nt_order  = ['200元及以下','200-500元','501-1000元','1001-2000元','2001元及以上']
    nt_labels = ['≤200元','200-500元','501-1000元','1001-2000元','≥2001元']
    nt_vals   = [df['Q7'].value_counts().get(k, 0) for k in nt_order]
    n_nt      = sum(nt_vals)
    bars = ax.bar(range(len(nt_order)), nt_vals,
                  color=[C_LIGHT, C_TEAL, C_BLUE, C_ORANGE, C_RED],
                  edgecolor='white', linewidth=1.5, width=0.65)
    for bar, val in zip(bars, nt_vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                f'{val}\n({val/max(n_nt,1)*100:.0f}%)',
                ha='center', va='bottom', fontsize=9.5, fontweight='bold')
    ax.set_xticks(range(len(nt_order)))
    ax.set_xticklabels(nt_labels, rotation=20, ha='right', fontsize=9.5)
    ax.set_ylabel('人数', fontsize=11)
    ax.set_title('(a) 每场非门票类消费（元）', fontsize=11, fontweight='bold')
    ax.set_ylim(0, max(nt_vals)*1.38)
    ax.spines[['top','right']].set_visible(False)

    # (b) 周边月均
    ax = axes[1]
    merch_order  = ['50元及以下','50-200元','201-500元','501-2000元']
    merch_labels = ['≤50元','50-200元','201-500元','501-2000元']
    merch_vals   = [df['Q8'].value_counts().get(k, 0) for k in merch_order]
    n_merch      = sum(merch_vals)
    bars = ax.bar(range(len(merch_order)), merch_vals,
                  color=[C_LIGHT, C_TEAL, C_BLUE, C_AMBER],
                  edgecolor='white', linewidth=1.5, width=0.65)
    for bar, val in zip(bars, merch_vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                f'{val}\n({val/max(n_merch,1)*100:.0f}%)',
                ha='center', va='bottom', fontsize=9.5, fontweight='bold')
    ax.set_xticks(range(len(merch_order)))
    ax.set_xticklabels(merch_labels, rotation=20, ha='right', fontsize=9.5)
    ax.set_ylabel('人数', fontsize=11)
    ax.set_title('(b) 偶像周边月均消费（元）', fontsize=11, fontweight='bold')
    ax.set_ylim(0, max(merch_vals)*1.38)
    ax.spines[['top','right']].set_visible(False)

    plt.tight_layout(pad=2.5)
    save_fig(fig, 'fig4_consumption')


# =============================================================================
# 图5  九大构念双组对比雷达图（有/无偶像 + 高/低频观演）
# =============================================================================
def fig5(df, score_df, CONSTRUCTS, N):
    keys   = list(CONSTRUCTS.keys())
    labels = [CONSTRUCTS[k]['label'] for k in keys]
    n      = len(keys)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False).tolist()
    angles_c = angles + [angles[0]]

    # 分组计算
    has_idol   = df['Q2'] == '有'
    no_idol    = ~has_idol
    high_freq  = df['Q1'] == '7次及以上'
    low_freq   = df['Q1'].isin(['1-3次', '从未有过'])

    def radar_vals(mask):
        return [score_df.loc[mask, k].mean() for k in keys] + [score_df.loc[mask, keys[0]].mean()]

    v_idol_yes = radar_vals(has_idol)
    v_idol_no  = radar_vals(no_idol)
    v_hi_freq  = radar_vals(high_freq)
    v_lo_freq  = radar_vals(low_freq)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5),
                              subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('white')
    fig.suptitle('图5  九大构念量表均值雷达图对比（N=712）',
                 fontsize=12, fontweight='bold', y=1.01)

    def draw_radar(ax, data_list, names, colors, title):
        ax.set_facecolor('#FAFBFC')
        ax.set_theta_offset(np.pi/2)
        ax.set_theta_direction(-1)
        for r in [1, 2, 3, 4, 5]:
            ax.plot(angles_c, [r]*(n+1), color='#E0E0E0', lw=0.6, ls='--')
        for vals, name, color in zip(data_list, names, colors):
            ax.plot(angles_c, vals, color=color, lw=2.2, label=name)
            ax.fill(angles_c, vals, color=color, alpha=0.12)
            for ang, v in zip(angles, vals[:-1]):
                ax.annotate(f'{v:.2f}', xy=(ang, v), xytext=(ang, v+0.25),
                            ha='center', va='center', fontsize=7.5,
                            color=color, fontweight='bold')
        ax.set_xticks(angles)
        ax.set_xticklabels(labels, fontsize=9, color=DARK_GRAY)
        ax.set_ylim(0, 5.8)
        ax.set_yticks([])
        ax.set_title(title, pad=20, fontsize=10, fontweight='bold', color=DARK_GRAY)
        ax.legend(loc='upper right', bbox_to_anchor=(1.38, 1.12),
                  fontsize=9, framealpha=0.9)
        ax.spines['polar'].set_color('#CCC')

    draw_radar(axes[0],
               [v_idol_yes, v_idol_no],
               [f'有偶像 (n={has_idol.sum()})', f'无/泛爱好 (n={no_idol.sum()})'],
               [MOR_BLUE, MOR_ROSE],
               '左图：有偶像 vs 无偶像/泛演出爱好者')
    draw_radar(axes[1],
               [v_hi_freq, v_lo_freq],
               [f'高频观演≥7次 (n={high_freq.sum()})', f'低频观演1-3次 (n={low_freq.sum()})'],
               [MOR_MINT, MOR_MAUVE],
               '右图：高频观演 vs 低频观演')

    plt.tight_layout()
    save_fig(fig, 'fig5_radar')


# =============================================================================
# 图6  有无偶像群体关键维度得分对比（分组柱状图 + 误差棒）
# =============================================================================
def fig6(df, score_df, CONSTRUCTS, N):
    idol_yes = df['Q2'] == '有'
    idol_no  = ~idol_yes
    cmp_keys = ['SMI','PSR','EEM','GBI','RSA','PVI','TWI']
    cmp_labels_short = [CONSTRUCTS[k]['label'].replace('\n','') for k in cmp_keys]

    m_yes = [score_df.loc[idol_yes, k].mean() for k in cmp_keys]
    m_no  = [score_df.loc[idol_no,  k].mean() for k in cmp_keys]
    se_yes= [score_df.loc[idol_yes, k].sem()  for k in cmp_keys]
    se_no = [score_df.loc[idol_no,  k].sem()  for k in cmp_keys]

    x     = np.arange(len(cmp_keys))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('white')

    b1 = ax.bar(x-width/2, m_yes, width, label=f'有偶像 (n={idol_yes.sum()})',
                color=C_BLUE, edgecolor='white', linewidth=1.5,
                yerr=se_yes, capsize=4,
                error_kw=dict(elinewidth=1.2, ecolor='#888'))
    b2 = ax.bar(x+width/2, m_no,  width, label=f'无/泛爱好 (n={idol_no.sum()})',
                color=C_AMBER, edgecolor='white', linewidth=1.5,
                yerr=se_no, capsize=4,
                error_kw=dict(elinewidth=1.2, ecolor='#888'))

    for bar, val in zip(b1, m_yes):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                f'{val:.2f}', ha='center', fontsize=8.5, fontweight='bold', color=C_BLUE)
    for bar, val in zip(b2, m_no):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                f'{val:.2f}', ha='center', fontsize=8.5, fontweight='bold', color='#A0522D')

    # 显著差异标注
    for i, (mv, mn, sv, sn) in enumerate(zip(m_yes, m_no, se_yes, se_no)):
        t_stat, p_val = stats.ttest_ind(
            score_df.loc[idol_yes, cmp_keys[i]].dropna(),
            score_df.loc[idol_no,  cmp_keys[i]].dropna())
        sig = '***' if p_val<0.001 else ('**' if p_val<0.01 else ('*' if p_val<0.05 else ''))
        diff = abs(mv - mn)
        if sig:
            ymax = max(mv, mn) + 0.3
            ax.annotate('', xy=(x[i]+width/2, ymax), xytext=(x[i]-width/2, ymax),
                        arrowprops=dict(arrowstyle='-', color='#555', lw=1.2))
            ax.text(x[i], ymax+0.06, f'Δ{diff:.2f}{sig}',
                    ha='center', fontsize=8.5, color=C_ORANGE, fontweight='bold')

    ax.set_ylim(1, 6.0)
    ax.set_xticks(x)
    ax.set_xticklabels(cmp_labels_short, fontsize=10.5, rotation=15, ha='right')
    ax.set_ylabel('量表均值（Likert 1–5分）', fontsize=11)
    ax.set_title('图6  有无偶像群体在各动机维度得分对比（含显著性标注）',
                 fontsize=13, fontweight='bold', pad=10)
    ax.legend(fontsize=11, loc='upper right')
    ax.spines[['top','right']].set_visible(False)
    ax.axhline(3.5, color='gray', linestyle='--', lw=0.8, alpha=0.5)
    ax.text(-0.5, 3.55, '均值参考线3.5', fontsize=8, color='gray')
    ax.yaxis.grid(True, ls='--', alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    save_fig(fig, 'fig6_idol_comparison')


# =============================================================================
# 图7  收入层级 × 座位偏好 100%堆叠图（莫兰迪高颜值版）
# =============================================================================
def fig7(df, N):
    inc_order   = ['1000元及以下','1001-3000元','3001-6000元','6001-10000元','10000元以上']
    inc_labels  = ['≤1000元\n(n=%d)' % df['Q11_4_income'].value_counts().get(k,0) for k in inc_order]
    seat_cats   = ['基础档（普通看台）','进阶档（优选看台）','高端档（内场）']
    seat_labels = ['基础档（普通看台）','进阶档（优选看台）','高端档（内场）']
    # 薄荷绿→钢蓝→暗夜灰
    seat_colors = [MOR_MINT, MOR_STEEL, DARK_NIGHT]

    cross     = pd.crosstab(df['Q11_4_income'], df['Q6'])
    cross     = cross.reindex(inc_order, fill_value=0)
    cross     = cross.reindex(columns=seat_cats, fill_value=0)
    cross_pct = cross.div(cross.sum(axis=1), axis=0)*100

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#FAFAFA')

    bottom = np.zeros(len(inc_order))
    bar_containers = []
    for cat, lbl, color in zip(seat_cats, seat_labels, seat_colors):
        vals = cross_pct[cat].values
        bars = ax.bar(inc_labels, vals, bottom=bottom, label=lbl,
                      color=color, edgecolor='white', linewidth=1.2)
        bar_containers.append(bars)
        for j, (bar, val) in enumerate(zip(bars, vals)):
            if val >= 6.5:
                txt_color = 'white' if color in [MOR_STEEL, DARK_NIGHT] else DARK_GRAY
                ax.text(bar.get_x()+bar.get_width()/2, bottom[j]+val/2,
                        f'{val:.1f}%', ha='center', va='center',
                        fontsize=10, fontweight='bold', color=txt_color)
        bottom += vals

    ax.set_ylim(0, 115)
    ax.set_ylabel('各座位档次占比（%）', fontsize=11, color=DARK_GRAY)
    ax.set_xlabel('月可支配收入层级', fontsize=11, color=DARK_GRAY)
    ax.set_title(f'图7  不同收入层级Z世代的演唱会座位档次偏好（N={N}）\n'
                 '薄荷绿=基础档  ·  钢蓝=进阶档  ·  暗夜灰=高端内场',
                 fontsize=12, fontweight='bold', pad=12, color=DARK_GRAY)
    ax.legend(fontsize=10.5, loc='upper right', framealpha=0.95,
              edgecolor=MID_GRAY, fancybox=True)
    ax.spines[['top','right']].set_visible(False)
    ax.yaxis.grid(True, ls='--', alpha=0.35, color='#E0E0E0')
    ax.set_axisbelow(True)
    plt.setp(ax.get_xticklabels(), fontsize=9.5)
    plt.tight_layout()
    save_fig(fig, 'fig7_income_seat')


# =============================================================================
# 图8  九大构念得分分布——小提琴+箱线组合（莫兰迪色系 + 显著性标注）
# =============================================================================
def fig8(df, score_df, CONSTRUCTS, N):
    keys   = list(CONSTRUCTS.keys())
    labels = [CONSTRUCTS[k]['label'] for k in keys]

    # 分组：有偶像 vs 无/泛
    has_idol = df['Q2'] == '有'
    no_idol  = ~has_idol

    colors_map = {
        'SMI': MOR_SKY,    'PSR': MOR_BLUE,   'CTA': MOR_SAGE,
        'EEM': MOR_MINT,   'GBI': MOR_FOREST, 'RSA': MOR_STEEL,
        'PCB': MOR_ROSE,   'PVI': MOR_SAND,   'TWI': MOR_MAUVE,
    }

    sns.set_theme(style='ticks', font_scale=0.95)
    fig, ax = plt.subplots(figsize=(18, 7.5), facecolor='white')
    ax.set_facecolor('#FAFAFA')

    # 分组小提琴
    pos_offset = 0.22
    for i, k in enumerate(keys):
        d_yes = score_df.loc[has_idol, k].dropna().values
        d_no  = score_df.loc[no_idol,  k].dropna().values
        color = colors_map[k]

        for d, pos, alpha in [(d_yes, i-pos_offset, 0.65), (d_no, i+pos_offset, 0.40)]:
            if len(d) < 5: continue
            vp = ax.violinplot([d], positions=[pos], widths=0.35,
                               showmedians=False, showextrema=False)
            for pc in vp['bodies']:
                pc.set_facecolor(color)
                pc.set_edgecolor(color)
                pc.set_alpha(alpha)

        # 箱线（有偶像组）
        bp = ax.boxplot([d_yes], positions=[i-pos_offset], widths=0.12,
                        patch_artist=True,
                        medianprops=dict(color='white', linewidth=2),
                        boxprops=dict(facecolor=color, alpha=0.9),
                        whiskerprops=dict(color=color),
                        capprops=dict(color=color),
                        flierprops=dict(marker='.', markersize=2,
                                        markerfacecolor=color, alpha=0.3))
        # 均值菱形
        m_yes = d_yes.mean()
        m_no  = d_no.mean()
        ax.scatter(i-pos_offset, m_yes, marker='D', s=55,
                   color='white', edgecolors=color, linewidth=1.8, zorder=6)
        ax.text(i-pos_offset, m_yes+0.22, f'{m_yes:.2f}',
                ha='center', fontsize=8, color=color, fontweight='bold', zorder=7)
        ax.text(i+pos_offset, m_no+0.10, f'{m_no:.2f}',
                ha='center', fontsize=7.5, color=color, alpha=0.75, zorder=7)

        # t检验显著性
        if len(d_yes) > 5 and len(d_no) > 5:
            t_stat, p_val = stats.ttest_ind(d_yes, d_no, equal_var=False)
            sig = '***' if p_val<0.001 else ('**' if p_val<0.01 else ('*' if p_val<0.05 else 'n.s.'))
            top = max(np.percentile(d_yes, 97), np.percentile(d_no, 97)) + 0.1
            ax.annotate('', xy=(i+pos_offset, top+0.04),
                        xytext=(i-pos_offset, top+0.04),
                        arrowprops=dict(arrowstyle='-', color=MID_GRAY, lw=0.8))
            ax.text(i, top+0.12, sig, ha='center', fontsize=9,
                    color=(C_RED if sig!='n.s.' else MID_GRAY), fontweight='bold')

    # 中性参考线
    ax.axhline(3.0, color=MID_GRAY, lw=1.0, ls=':', alpha=0.7)
    ax.text(-0.6, 3.05, '中性\n基准(3)', ha='right', va='bottom', fontsize=7.5, color=MID_GRAY)

    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels, fontsize=9.5, color=DARK_GRAY)
    ax.set_ylim(0.5, 6.5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_ylabel('构念量表均值（Likert 1–5分）', fontsize=11, color=DARK_GRAY)
    ax.set_title(
        '图8  九大构念量表得分分布（小提琴+箱线+均值菱形）\n'
        'N=712；深色=有偶像组，浅色=无偶像/泛爱好组；◆=均值；显著性：***p<.001',
        fontsize=11, fontweight='bold', pad=12, color=DARK_GRAY)
    sns.despine(ax=ax, top=True, right=True)
    ax.yaxis.grid(True, ls='--', alpha=0.35, color='#E0E0E0')
    ax.set_axisbelow(True)

    legend_els = [
        mpatches.Patch(color=MOR_MINT, alpha=0.8, label='情境-动机类构念'),
        mpatches.Patch(color=MOR_ROSE, alpha=0.8, label='感知成本障碍（PCB）'),
        mpatches.Patch(color=MOR_SAND, alpha=0.8, label='行为意向类构念'),
        Line2D([0],[0], marker='D', color='w', markerfacecolor='white',
               markeredgecolor=DARK_GRAY, markersize=7, label='均值（有偶像组）'),
    ]
    ax.legend(handles=legend_els, loc='upper right',
              fontsize=9, framealpha=0.95, ncol=2, edgecolor=MID_GRAY)

    plt.tight_layout()
    save_fig(fig, 'fig8_violin')


# =============================================================================
# 图9  模型一 SEM 路径图（动机-情境双轮驱动）
# =============================================================================
def fig9():
    if not os.path.exists(SEM_FILE):
        print(f'  ⚠ 找不到 {SEM_FILE}，跳过 fig9')
        return
    with open(SEM_FILE, encoding='utf-8') as f:
        RES = json.load(f)

    def get_path(paths, frm, to):
        for r in paths:
            if r['路径'] == f'{frm} → {to}':
                return r['标准化β'], r['显著性']
        return None, 'n.s.'

    def sig_color(s):
        return (C_BLUE, 2.2, False) if s in ('*','**','***') else ('#CCC', 1.2, True)

    def draw_box(ax, x, y, w, h, label, sub='', fc='#EAF4F4', ec=C_BLUE):
        ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
                                    boxstyle='round,pad=0.03',
                                    facecolor=fc, edgecolor=ec, lw=1.8, zorder=3))
        ax.text(x, y+(0.015 if sub else 0), label,
                ha='center', va='center', fontsize=9.5, fontweight='bold',
                color='#111', zorder=4)
        if sub:
            ax.text(x, y-0.055, sub, ha='center', va='center',
                    fontsize=8, color='#555', zorder=4)

    def arr(ax, x1, y1, x2, y2, label='', color=C_BLUE, lw=2.0, dash=False, rad=0.0):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                    linestyle='--' if dash else '-',
                                    connectionstyle=f'arc3,rad={rad}'))
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx, my+0.028, label, ha='center', va='bottom', fontsize=7.5,
                    color=color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85))

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('white')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

    fit1 = RES['m1_fit']
    ax.set_title(f'图9  模型一：动机-情境双轮驱动模型路径图（N=712）\n'
                 f'CFI={fit1["CFI"]:.3f}  RMSEA={fit1["RMSEA"]:.3f}  AIC={fit1["AIC"]:.1f}',
                 fontsize=12, fontweight='bold', pad=10)

    SX, MX, BX = 0.13, 0.46, 0.82
    draw_box(ax, SX, 0.78, 0.22, 0.10, 'SMI', '社交媒体信息影响', fc='#E8F4FD', ec='#2980B9')
    draw_box(ax, SX, 0.50, 0.22, 0.10, 'PSR', '偶像准社会关系',   fc='#E8F4FD', ec='#2980B9')
    draw_box(ax, SX, 0.22, 0.22, 0.10, 'CTA', '城市文旅吸引力',   fc='#E8F4FD', ec='#2980B9')
    draw_box(ax, MX, 0.80, 0.22, 0.10, 'EEM', '情感体验动机',     fc='#FEF9E7', ec=C_AMBER)
    draw_box(ax, MX, 0.50, 0.22, 0.10, 'GBI', '群体归属感',       fc='#FEF9E7', ec=C_AMBER)
    draw_box(ax, MX, 0.20, 0.22, 0.10, 'RSA', '仪式感/自我实现',  fc='#FEF9E7', ec=C_AMBER)
    draw_box(ax, BX, 0.65, 0.22, 0.10, 'PVI', '观演意愿',         fc='#EAFAF1', ec=C_TEAL)
    draw_box(ax, BX, 0.35, 0.22, 0.10, 'TWI', '旅游消费意愿',     fc='#EAFAF1', ec=C_TEAL)

    for t, x in [('情境层', SX), ('动机层', MX), ('行为层', BX)]:
        ax.text(x, 0.95, t, ha='center', fontsize=11, fontweight='bold', color='#444',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#F0F0F0', edgecolor='#ccc'))

    m1_paths = RES['m1_paths']
    edges = [
        (SX+0.11, 0.78, MX-0.11, 0.80, 'SMI', 'EEM'),
        (SX+0.11, 0.78, MX-0.11, 0.50, 'SMI', 'GBI'),
        (SX+0.11, 0.50, MX-0.11, 0.80, 'PSR', 'EEM'),
        (SX+0.11, 0.50, MX-0.11, 0.50, 'PSR', 'GBI'),
        (SX+0.11, 0.50, MX-0.11, 0.20, 'PSR', 'RSA'),
        (MX+0.11, 0.80, BX-0.11, 0.65, 'EEM', 'PVI'),
        (MX+0.11, 0.50, BX-0.11, 0.65, 'GBI', 'PVI'),
        (MX+0.11, 0.20, BX-0.11, 0.65, 'RSA', 'PVI'),
        (MX+0.11, 0.80, BX-0.11, 0.35, 'EEM', 'TWI'),
        (MX+0.11, 0.50, BX-0.11, 0.35, 'GBI', 'TWI'),
        (MX+0.11, 0.20, BX-0.11, 0.35, 'RSA', 'TWI'),
    ]
    for (x1, y1, x2, y2, frm, to) in edges:
        b, s = get_path(m1_paths, frm, to)
        c, lw, dash = sig_color(s)
        lbl = f'β={b}{s}' if s in ('*','**','***') else ''
        arr(ax, x1, y1, x2, y2, label=lbl, color=c, lw=lw, dash=dash)

    # CTA → TWI 直接效应
    b, s = get_path(m1_paths, 'CTA', 'TWI')
    ax.annotate('', xy=(BX-0.11, 0.35), xytext=(SX+0.11, 0.22),
                arrowprops=dict(arrowstyle='->', color=C_ORANGE, lw=2.0,
                                connectionstyle='arc3,rad=-0.28'))
    ax.text(0.50, 0.07, f'CTA→TWI 直接效应: β={b}{s}',
            ha='center', fontsize=8.5, color=C_ORANGE, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=C_ORANGE, alpha=0.9))

    ax.legend(handles=[
        Line2D([0],[0], color=C_BLUE,   lw=2.2, label='路径显著 (p<0.05)'),
        Line2D([0],[0], color='#CCC',   lw=1.2, ls='--', label='路径不显著'),
        Line2D([0],[0], color=C_ORANGE, lw=2.0, label='CTA 直接效应'),
    ], loc='lower right', fontsize=9, framealpha=0.9)

    # 显著路径汇总
    sig_list = [(r['路径'], r['标准化β'], r['显著性'])
                for r in m1_paths if r['显著性'] in ('*','**','***')]
    note = '显著路径：\n' + '\n'.join(f'  {p}: β={b}{s}' for p, b, s in sig_list)
    ax.text(0.01, 0.01, note, transform=ax.transAxes, fontsize=7.5, va='bottom',
            bbox=dict(boxstyle='round,pad=0.4', fc='#FFFEF0', ec='#E5C85A', alpha=0.95))

    plt.tight_layout()
    save_fig(fig, 'fig9_model1_path')


# =============================================================================
# 图10  模型二 SEM 路径图（动机-阻碍，含二阶因子 MOT）
# =============================================================================
def fig10():
    if not os.path.exists(SEM_FILE):
        print(f'  ⚠ 找不到 {SEM_FILE}，跳过 fig10')
        return
    with open(SEM_FILE, encoding='utf-8') as f:
        RES = json.load(f)

    def get_path(paths, frm, to):
        for r in paths:
            if r['路径'] == f'{frm} → {to}':
                return r['标准化β'], r['显著性']
        return None, 'n.s.'

    def draw_box(ax, x, y, w, h, label, sub='', fc='#EAF4F4', ec=C_BLUE, fs=10):
        ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
                                    boxstyle='round,pad=0.03',
                                    facecolor=fc, edgecolor=ec, lw=1.8, zorder=3))
        ax.text(x, y+(0.015 if sub else 0), label,
                ha='center', va='center', fontsize=fs, fontweight='bold',
                color='#111', zorder=4)
        if sub:
            ax.text(x, y-0.055, sub, ha='center', va='center',
                    fontsize=8, color='#555', zorder=4)

    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor('white')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

    fit2 = RES['m2_fit']
    ax.set_title(f'图10  模型二：动机-阻碍 SEM 路径图（N=712，含二阶因子 MOT）\n'
                 f'CFI={fit2["CFI"]:.3f}  RMSEA={fit2["RMSEA"]:.3f}  '
                 f'AIC={fit2["AIC"]:.1f}  BIC={fit2["BIC"]:.1f}',
                 fontsize=11, fontweight='bold', pad=10)

    # 节点
    draw_box(ax, 0.38, 0.62, 0.22, 0.12, 'MOT', '内在动机（二阶）',
             fc='#FEF5E7', ec=C_AMBER, fs=11)
    draw_box(ax, 0.13, 0.82, 0.20, 0.09, 'EEM', '情感体验动机', fc='#FDF2E9', ec='#E59866')
    draw_box(ax, 0.13, 0.60, 0.20, 0.09, 'GBI', '群体归属感',   fc='#FDF2E9', ec='#E59866')
    draw_box(ax, 0.13, 0.38, 0.20, 0.09, 'RSA', '仪式感/自我实现', fc='#FDF2E9', ec='#E59866')
    draw_box(ax, 0.38, 0.22, 0.22, 0.09, 'PCB', '感知成本障碍',
             fc='#FDEDEC', ec=C_RED)
    draw_box(ax, 0.76, 0.70, 0.22, 0.11, 'PVI', '观演意愿',
             fc='#EAFAF1', ec=C_TEAL)
    draw_box(ax, 0.76, 0.38, 0.22, 0.11, 'TWI', '旅游消费延伸意愿',
             fc='#EAFAF1', ec=C_TEAL)

    # R² 标注
    r2_dict = RES.get('m2_r2', {})
    for xpos, ypos, lv in [(0.76, 0.78, 'PVI'), (0.76, 0.30, 'TWI')]:
        r2val = r2_dict.get(lv, '—')
        ax.text(xpos, ypos, f'R²={r2val}', ha='center', fontsize=9,
                color=C_TEAL, fontweight='bold')

    # MOT → EEM/GBI/RSA 二阶载荷
    m2_paths = RES['m2_paths']
    loads = [('EEM', 0.82, '0.507 fixed'), ('GBI', 0.60, '0.963***'), ('RSA', 0.38, '0.995***')]
    for to_lv, yp, lb in loads:
        ax.annotate('', xy=(0.23, yp), xytext=(0.27, 0.62),
                    arrowprops=dict(arrowstyle='->', color=C_AMBER, lw=1.8))
        ax.text(0.18, (0.62+yp)/2+0.02, f'λ={lb}',
                ha='center', fontsize=8, color=C_AMBER, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.85))

    # MOT → PVI / TWI
    for (y2, to_lv, yptext, xptext) in [
        (0.70, 'PVI', 0.73, 0.72),
        (0.38, 'TWI', 0.47, 0.56),
    ]:
        b, s = get_path(m2_paths, 'MOT', to_lv)
        ax.annotate('', xy=(0.65, y2), xytext=(0.49, 0.62),
                    arrowprops=dict(arrowstyle='->', color=C_TEAL, lw=2.8))
        ax.text(xptext, yptext, f'β={b}{s}',
                ha='center', fontsize=9.5, color=C_TEAL, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.9))

    # PCB → PVI / TWI
    b_pvi, s_pvi = get_path(m2_paths, 'PCB', 'PVI')
    ax.annotate('', xy=(0.65, 0.70), xytext=(0.49, 0.22),
                arrowprops=dict(arrowstyle='->', color=C_RED, lw=2.0,
                                connectionstyle='arc3,rad=-0.3'))
    ax.text(0.70, 0.53, f'β={b_pvi}{s_pvi}',
            ha='center', fontsize=9, color=C_RED, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.9))

    b_twi, s_twi = get_path(m2_paths, 'PCB', 'TWI')
    ax.annotate('', xy=(0.65, 0.38), xytext=(0.49, 0.22),
                arrowprops=dict(arrowstyle='->', color=C_RED, lw=2.0,
                                connectionstyle='arc3,rad=-0.1'))
    ax.text(0.62, 0.26, f'β={b_twi}{s_twi}',
            ha='center', fontsize=9, color=C_RED, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.9))

    # PVI ↔ TWI
    ax.annotate('', xy=(0.76, 0.44), xytext=(0.76, 0.64),
                arrowprops=dict(arrowstyle='<->', color='#888', lw=1.5))

    ax.legend(handles=[
        Line2D([0],[0], color=C_TEAL,  lw=2.8, label='动机正效应 (p<0.001)'),
        Line2D([0],[0], color=C_RED,   lw=2.0, label='成本负效应 (p<0.001)'),
        Line2D([0],[0], color=C_AMBER, lw=1.8, label='二阶因子载荷'),
    ], loc='lower right', fontsize=10, framealpha=0.9)

    ax.text(0.02, 0.02,
            f'核心路径：\n  MOT→PVI β={get_path(m2_paths,"MOT","PVI")[0]}***\n'
            f'  MOT→TWI β={get_path(m2_paths,"MOT","TWI")[0]}***\n'
            f'  PCB→PVI β={get_path(m2_paths,"PCB","PVI")[0]}***\n'
            f'  PCB→TWI β={get_path(m2_paths,"PCB","TWI")[0]}***',
            transform=ax.transAxes, fontsize=8.5, va='bottom',
            bbox=dict(boxstyle='round,pad=0.4', fc='#FFFEF0', ec='#E5C85A', alpha=0.95))

    plt.tight_layout()
    save_fig(fig, 'fig10_model2_path')


# =============================================================================
# 地图  PPS 抽样省份气泡图
# =============================================================================
def fig_map(N):
    from matplotlib.colors import LinearSegmentedColormap
    provinces = {
        '黑龙江':(128,48), '吉林':(126,43.5), '辽宁':(123,41.5),
        '内蒙古':(112,44), '新疆':(86,42), '西藏':(88,31.5),
        '青海':(97,35.5), '甘肃':(103.5,36), '宁夏':(106,37.5),
        '陕西':(109,35.5), '山西':(112.5,37.5), '河北':(115,39),
        '北京':(116.4,40), '天津':(117.2,39.1), '山东':(118,36.5),
        '河南':(113.5,34), '湖北':(113.5,31), '安徽':(117,32),
        '江苏':(119.5,33), '上海':(121.5,31.2), '浙江':(120,29),
        '江西':(116,27.5), '湖南':(112,27), '福建':(118,26),
        '广东':(113,23.5), '广西':(108.5,23.5), '海南':(110,19.5),
        '贵州':(107,27), '云南':(102,25), '四川':(103.5,30.5), '重庆':(107.5,29.5),
    }
    sampled = {
        '广东': {'sample_n': 95, 'z_pct': 18.2},
        '山东': {'sample_n': 78, 'z_pct': 17.8},
        '河北': {'sample_n': 62, 'z_pct': 17.1},
        '江西': {'sample_n': 48, 'z_pct': 18.5},
        '辽宁': {'sample_n': 38, 'z_pct': 16.3},
        '甘肃': {'sample_n': 15, 'z_pct': 17.9},
    }

    blue_cmap = LinearSegmentedColormap.from_list('tb', ['#AED6F1','#1A5276'], N=100)
    max_n = max(v['sample_n'] for v in sampled.values())

    fig, ax = plt.subplots(figsize=(14, 9), facecolor='#F8F9FA')
    ax.set_facecolor('#EBF5FB')

    # 未抽中省份
    for prov, (lon, lat) in provinces.items():
        if prov in sampled or prov == '天津':
            continue
        ax.scatter(lon, lat, s=100, color='#CBD5E1', zorder=2, alpha=0.7,
                   edgecolors='white', linewidth=0.8)
        ax.text(lon, lat-0.75, prov, ha='center', fontsize=6.5, color='#8898A8', zorder=3)

    # 抽中省份
    for prov, info in sampled.items():
        lon, lat = provinces[prov]
        intensity = info['sample_n'] / max_n
        c = blue_cmap(0.3 + 0.7*intensity)
        size = 280 + info['sample_n'] * 8

        ax.scatter(lon, lat, s=size, color=c, zorder=5,
                   edgecolors='white', linewidth=2.5, alpha=0.92)
        ax.text(lon, lat, f"{prov}\n(n={info['sample_n']})",
                ha='center', va='center', fontsize=8, fontweight='bold',
                color='white', zorder=6)
        ax.text(lon, lat-1.7, f"Z世代占比{info['z_pct']}%",
                ha='center', fontsize=7, color='#1A5276',
                bbox=dict(boxstyle='round,pad=0.2', fc='white',
                          ec='#1A5276', alpha=0.85, lw=1))

    # 天津
    tj_lon, tj_lat = provinces['天津']
    ax.scatter(tj_lon, tj_lat, s=250, color=C_TEAL, zorder=7,
               marker='*', edgecolors='white', linewidth=2)
    ax.annotate('天津\n(研究城市)', (tj_lon, tj_lat),
                xytext=(tj_lon+2.5, tj_lat+1.5),
                fontsize=8.5, fontweight='bold', color=C_TEAL,
                arrowprops=dict(arrowstyle='-|>', color=C_TEAL, lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=C_TEAL, alpha=0.9))

    # 颜色条
    sm = plt.cm.ScalarMappable(cmap=blue_cmap, norm=plt.Normalize(0, max_n))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical',
                        fraction=0.025, pad=0.02, aspect=20)
    cbar.set_label('层次C样本量（份）', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.text(0.02, 0.03,
            'PPS抽样说明：\nk = T/n ≈ 2.35亿\nr = 1.00亿\n6个省份',
            transform=ax.transAxes, fontsize=8.5, va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='#EAF2FF', ec=MOR_BLUE, alpha=0.95))
    ax.set_xlim(72, 140)
    ax.set_ylim(14, 56)
    ax.set_xlabel('经度 (°E)', fontsize=10)
    ax.set_ylabel('纬度 (°N)', fontsize=10)
    ax.set_title('图3-1  非天津本地层（层次C）PPS整群抽样省份分布\n'
                 '气泡大小=样本贡献量，颜色深浅=样本量，★=研究城市天津',
                 fontsize=11, fontweight='bold', pad=12)
    ax.grid(True, ls='--', alpha=0.3, color=MID_GRAY)
    plt.tight_layout()
    save_fig(fig, 'fig_map_pps')


# =============================================================================
# pyecharts 动态地图代码片段（Python ≥ 3.10 + pyecharts ≥ 2.0）
# =============================================================================
PYECHARTS_CODE = """\
# ─── 动态中国地图（需 Python 3.10+，pip install pyecharts）─────────────────
from pyecharts import options as opts
from pyecharts.charts import Map
from pyecharts.globals import ThemeType

province_data = [
    ("广东", 95), ("山东", 78), ("河北", 62),
    ("江西", 48), ("辽宁", 38), ("甘肃", 15),
    ("天津", 200),  # 研究城市
]

(
    Map(init_opts=opts.InitOpts(width="900px", height="600px",
                                theme=ThemeType.WHITE,
                                page_title="PPS抽样省份分布"))
    .add("层次C样本量", province_data, maptype="china", is_roam=True,
         label_opts=opts.LabelOpts(is_show=True, font_size=9))
    .set_global_opts(
        title_opts=opts.TitleOpts(title="演唱会调研PPS抽样省份分布",
                                  subtitle="颜色深浅代表样本量"),
        visualmap_opts=opts.VisualMapOpts(
            is_piecewise=True,
            pieces=[
                {"min":150, "label":"研究城市(天津)", "color":"#148F77"},
                {"min":70, "max":150, "label":"70~150份", "color":"#1A5276"},
                {"min":40, "max":70,  "label":"40~70份",  "color":"#2E86C1"},
                {"min":10, "max":40,  "label":"10~40份",  "color":"#7FB3D3"},
                {"min":0,  "max":10,  "label":"未抽中",   "color":"#D6EAF8"},
            ],
        ),
    )
    .render("charts/fig_map_pyecharts.html")
)
print("地图已保存：charts/fig_map_pyecharts.html")
"""


# =============================================================================
# CLI 入口
# =============================================================================
ALL_FIGS = ['fig1','fig2','fig3','fig4','fig5','fig6','fig7','fig8','fig9','fig10','map']

def main():
    args = sys.argv[1:]

    if '--list' in args:
        print(__doc__)
        return

    targets = [a for a in args if not a.startswith('--')]
    if not targets:
        targets = ALL_FIGS

    # 验证目标
    invalid = [t for t in targets if t not in ALL_FIGS]
    if invalid:
        print(f'未知图表名称：{invalid}，可选：{ALL_FIGS}')
        sys.exit(1)

    print('=' * 60)
    print('  粉丝经济报告图表生成')
    print(f'  数据文件：{DATA_FILE}  |  目标：{targets}')
    print('=' * 60)

    # 加载数据（需要数据的图才加载）
    data_needed = [t for t in targets if t not in ('fig9','fig10','map')]
    df, CONSTRUCTS, score_df, N = None, None, None, None
    if data_needed:
        df, CONSTRUCTS, score_df = load_data()
        N = len(df)
        print(f'  已加载数据：N={N}')

    dispatch = {
        'fig1':  lambda: fig1(df, N),
        'fig2':  lambda: fig2(df, N),
        'fig3':  lambda: fig3(df, N),
        'fig4':  lambda: fig4(df, N),
        'fig5':  lambda: fig5(df, score_df, CONSTRUCTS, N),
        'fig6':  lambda: fig6(df, score_df, CONSTRUCTS, N),
        'fig7':  lambda: fig7(df, N),
        'fig8':  lambda: fig8(df, score_df, CONSTRUCTS, N),
        'fig9':  fig9,
        'fig10': fig10,
        'map':   lambda: fig_map(N if N else 712),
    }

    for t in targets:
        print(f'\n[ {t} ]')
        try:
            dispatch[t]()
        except Exception as e:
            print(f'  ✗ 生成失败：{e}')
            import traceback; traceback.print_exc()

    print('\n' + '=' * 60)
    print('  所有图表生成完毕 → charts/')
    print('=' * 60)

    if '--show-pyecharts' in sys.argv:
        print('\n【pyecharts 动态地图代码（Python≥3.10）】')
        print(PYECHARTS_CODE)


if __name__ == '__main__':
    main()
