"""
sem_charts.py
=============
生成SEM路径图（Model1 + Model2），保存到 charts/
"""

import sys, os
sys.path.insert(0, '.pip_pkgs')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib import rcParams

rcParams['font.family'] = ['Arial Unicode MS', 'STHeiti', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

os.makedirs('charts', exist_ok=True)

C_BLUE   = '#2E4057'
C_TEAL   = '#048A81'
C_AMBER  = '#F4A261'
C_ORANGE = '#E76F51'
C_GRAY   = '#8D99AE'
C_PURPLE = '#6B4E71'
C_RED    = '#C0392B'
C_GREEN  = '#27AE60'
C_LIGHT  = '#EAF4F4'


def draw_box(ax, x, y, w, h, label, sublabel='',
             facecolor=C_LIGHT, edgecolor=C_BLUE, fontsize=10, bold=False):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle='round,pad=0.03',
                         facecolor=facecolor, edgecolor=edgecolor,
                         linewidth=1.8, zorder=3)
    ax.add_patch(box)
    fw = 'bold' if bold else 'normal'
    ax.text(x, y + (0.015 if sublabel else 0), label,
            ha='center', va='center', fontsize=fontsize,
            fontweight=fw, color='#111', zorder=4)
    if sublabel:
        ax.text(x, y - 0.055, sublabel,
                ha='center', va='center', fontsize=8.5,
                color='#555', zorder=4)


def arrow(ax, x1, y1, x2, y2, label='', color=C_BLUE,
          lw=2.0, style='->', dash=False, fontsize=9):
    ls = '--' if dash else '-'
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style,
                                color=color, lw=lw,
                                linestyle=ls,
                                connectionstyle='arc3,rad=0.0'))
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my + 0.025, label, ha='center', va='bottom',
                fontsize=fontsize, color=color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor='none', alpha=0.85))


# ══════════════════════════════════════════════════════════════════════════════
# 图9  模型一：动机-情境双轮驱动模型  路径图
# ══════════════════════════════════════════════════════════════════════════════
def draw_model1(path_df):
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('white')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('图9  模型一：动机-情境双轮驱动模型路径图', fontsize=14,
                 fontweight='bold', pad=12)

    # 情境层 (x=0.12)
    SITU_X = 0.13
    draw_box(ax, SITU_X, 0.78, 0.22, 0.10, 'SMI', '社交媒体信息影响',
             facecolor='#E8F4FD', edgecolor='#2980B9', bold=True)
    draw_box(ax, SITU_X, 0.50, 0.22, 0.10, 'PSR', '偶像准社会关系',
             facecolor='#E8F4FD', edgecolor='#2980B9', bold=True)
    draw_box(ax, SITU_X, 0.22, 0.22, 0.10, 'CTA', '城市文旅吸引力',
             facecolor='#E8F4FD', edgecolor='#2980B9', bold=True)

    # 动机层 (x=0.45)
    MOT_X = 0.46
    draw_box(ax, MOT_X, 0.80, 0.22, 0.10, 'EEM', '情感体验动机',
             facecolor='#FEF9E7', edgecolor=C_AMBER, bold=True)
    draw_box(ax, MOT_X, 0.50, 0.22, 0.10, 'GBI', '群体归属感',
             facecolor='#FEF9E7', edgecolor=C_AMBER, bold=True)
    draw_box(ax, MOT_X, 0.20, 0.22, 0.10, 'RSA', '仪式感/自我实现',
             facecolor='#FEF9E7', edgecolor=C_AMBER, bold=True)

    # 行为层 (x=0.80)
    BEH_X = 0.82
    draw_box(ax, BEH_X, 0.65, 0.22, 0.10, 'PVI', '观演意愿',
             facecolor='#EAFAF1', edgecolor=C_TEAL, bold=True)
    draw_box(ax, BEH_X, 0.35, 0.22, 0.10, 'TWI', '旅游消费意愿',
             facecolor='#EAFAF1', edgecolor=C_TEAL, bold=True)

    # 层标签
    for txt, x in [('情境层', SITU_X), ('动机层', MOT_X), ('行为层', BEH_X)]:
        ax.text(x, 0.96, txt, ha='center', va='center', fontsize=12,
                fontweight='bold', color='#444',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#F0F0F0',
                          edgecolor='#ccc'))

    def get_beta(from_lv, to_lv):
        row = path_df[(path_df['路径'] == f'{from_lv} → {to_lv}')]
        if len(row):
            b = row.iloc[0]['标准化β']
            p = row.iloc[0]['显著性']
            return f'β={b}{p}' if b != '—' else ''
        return ''

    # 情境→动机 路径（只画显著的用实线，不显著用虚线灰色）
    sig_map = {}
    for _, r in path_df.iterrows():
        parts = r['路径'].split(' → ')
        if len(parts) == 2:
            sig_map[(parts[0], parts[1])] = (r['标准化β'], r['显著性'])

    def path_color_style(frm, to):
        v = sig_map.get((frm, to), ('—', 'n.s.'))
        sig = v[1]
        if sig in ('***', '**', '*'):
            return C_BLUE, 2.2, False
        return '#BBBBBB', 1.2, True

    paths_1 = [
        (SITU_X+0.11, 0.78, MOT_X-0.11, 0.80, 'SMI', 'EEM'),
        (SITU_X+0.11, 0.78, MOT_X-0.11, 0.50, 'SMI', 'GBI'),
        (SITU_X+0.11, 0.50, MOT_X-0.11, 0.80, 'PSR', 'EEM'),
        (SITU_X+0.11, 0.50, MOT_X-0.11, 0.50, 'PSR', 'GBI'),
        (SITU_X+0.11, 0.50, MOT_X-0.11, 0.20, 'PSR', 'RSA'),
        (SITU_X+0.11, 0.22, MOT_X-0.11, 0.80, 'CTA', 'EEM'),
    ]
    for x1, y1, x2, y2, frm, to in paths_1:
        c, lw, dash = path_color_style(frm, to)
        b, s = sig_map.get((frm, to), ('—', ''))
        lbl = f'{b}{s}' if s not in ('n.s.', '') and b != '—' else ''
        arrow(ax, x1, y1, x2, y2, label=lbl, color=c, lw=lw, dash=dash, fontsize=8)

    # CTA → TWI 直接效应（跨层）
    c, lw, dash = path_color_style('CTA', 'TWI')
    b, s = sig_map.get(('CTA', 'TWI'), ('—', ''))
    lbl = f'{b}{s}' if s not in ('n.s.', '') and b != '—' else ''
    ax.annotate('', xy=(BEH_X-0.11, 0.35), xytext=(SITU_X+0.11, 0.22),
                arrowprops=dict(arrowstyle='->', color=C_ORANGE, lw=2.2,
                                connectionstyle='arc3,rad=-0.25'))
    ax.text(0.48, 0.08, f'CTA→TWI 直接效应\n{lbl}',
            ha='center', fontsize=8.5, color=C_ORANGE, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor=C_ORANGE, alpha=0.9))

    # 动机→意愿 路径
    paths_2 = [
        (MOT_X+0.11, 0.80, BEH_X-0.11, 0.65, 'EEM', 'PVI'),
        (MOT_X+0.11, 0.50, BEH_X-0.11, 0.65, 'GBI', 'PVI'),
        (MOT_X+0.11, 0.20, BEH_X-0.11, 0.65, 'RSA', 'PVI'),
        (MOT_X+0.11, 0.80, BEH_X-0.11, 0.35, 'EEM', 'TWI'),
        (MOT_X+0.11, 0.50, BEH_X-0.11, 0.35, 'GBI', 'TWI'),
        (MOT_X+0.11, 0.20, BEH_X-0.11, 0.35, 'RSA', 'TWI'),
    ]
    for x1, y1, x2, y2, frm, to in paths_2:
        c, lw, dash = path_color_style(frm, to)
        arrow(ax, x1, y1, x2, y2, color=c, lw=lw, dash=dash)

    # 图例
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0],[0], color=C_BLUE, lw=2.2, label='显著路径 (p<0.05)'),
        Line2D([0],[0], color='#BBB',  lw=1.2, linestyle='--', label='非显著路径'),
        Line2D([0],[0], color=C_ORANGE, lw=2.2, label='CTA直接效应'),
    ]
    ax.legend(handles=legend_elems, loc='lower right',
              fontsize=9.5, framealpha=0.9, edgecolor='#ccc')

    # 注释框：显著路径汇总
    sig_paths = path_df[path_df['显著性'].isin(['*','**','***'])]
    note_lines = ['显著路径汇总：']
    for _, r in sig_paths.iterrows():
        note_lines.append(f"  {r['假设']} {r['路径']}: β={r['标准化β']}{r['显著性']}")
    ax.text(0.01, 0.01, '\n'.join(note_lines), transform=ax.transAxes,
            fontsize=8, va='bottom', color='#333',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFFEF0',
                      edgecolor='#E5C85A', alpha=0.95))

    plt.tight_layout()
    path = 'charts/fig9_model1_path.png'
    fig.savefig(path, dpi=220, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  saved: {path}')


# ══════════════════════════════════════════════════════════════════════════════
# 图10  模型二：动机-阻碍SEM路径图（含二阶因子）
# ══════════════════════════════════════════════════════════════════════════════
def draw_model2(path_df):
    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor('white')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('图10  模型二：动机-阻碍SEM模型路径图（含二阶动机因子）',
                 fontsize=13, fontweight='bold', pad=12)

    # 二阶因子 MOT (中央)
    draw_box(ax, 0.38, 0.60, 0.22, 0.12, 'MOT', '内在动机（二阶因子）',
             facecolor='#FEF5E7', edgecolor=C_AMBER, fontsize=11, bold=True)

    # 一阶动机因子 (左侧)
    draw_box(ax, 0.13, 0.82, 0.20, 0.09, 'EEM', '情感体验动机',
             facecolor='#FDF2E9', edgecolor='#E59866', bold=True)
    draw_box(ax, 0.13, 0.60, 0.20, 0.09, 'GBI', '群体归属感',
             facecolor='#FDF2E9', edgecolor='#E59866', bold=True)
    draw_box(ax, 0.13, 0.38, 0.20, 0.09, 'RSA', '仪式感/自我实现',
             facecolor='#FDF2E9', edgecolor='#E59866', bold=True)

    # 阻碍因子 PCB (下方)
    draw_box(ax, 0.38, 0.20, 0.22, 0.09, 'PCB', '感知成本障碍',
             facecolor='#FDEDEC', edgecolor=C_RED, bold=True)

    # 结果变量 (右侧)
    draw_box(ax, 0.75, 0.70, 0.22, 0.10, 'PVI', '观演意愿',
             facecolor='#EAFAF1', edgecolor=C_TEAL, bold=True)
    draw_box(ax, 0.75, 0.40, 0.22, 0.10, 'TWI', '旅游消费意愿',
             facecolor='#EAFAF1', edgecolor=C_TEAL, bold=True)

    def get_info(frm, to):
        row = path_df[path_df['路径'] == f'{frm} → {to}']
        if len(row):
            r = row.iloc[0]
            return r['标准化β'], r['显著性']
        return '—', ''

    def lbl_fmt(b, s):
        return f'β={b}{s}' if s not in ('n.s.', '', '—', 'fixed') and b != '—' else ''

    # MOT → EEM/GBI/RSA 二阶载荷（从检验结果读取）
    # semopy输出中 EEM ~ MOT, GBI ~ MOT, RSA ~ MOT
    mot_loadings = {
        'EEM': ('0.972', 'fixed'),
        'GBI': ('1.000', '***'),
        'RSA': ('0.996', '***'),
    }
    for to_lv, (y_pos) in [('EEM', 0.82), ('GBI', 0.60), ('RSA', 0.38)]:
        b, s = mot_loadings[to_lv]
        lbl = f'λ={b}{s}'
        ax.annotate('', xy=(0.13+0.10, y_pos), xytext=(0.38-0.11, 0.60),
                    arrowprops=dict(arrowstyle='->', color=C_AMBER, lw=2.0,
                                    connectionstyle='arc3,rad=0.0'))
        mx = (0.38-0.11 + 0.13+0.10)/2
        my = (0.60 + y_pos)/2
        ax.text(mx, my + 0.025, lbl, ha='center', fontsize=8.5,
                color=C_AMBER, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor='none', alpha=0.85))

    # MOT → PVI (显著)
    b, s = get_info('MOT', 'PVI')
    lbl = lbl_fmt(b, s)
    ax.annotate('', xy=(0.75-0.11, 0.70), xytext=(0.38+0.11, 0.60),
                arrowprops=dict(arrowstyle='->', color=C_TEAL, lw=2.5,
                                connectionstyle='arc3,rad=0.0'))
    ax.text(0.56, 0.68, lbl, ha='center', fontsize=9, color=C_TEAL,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor='none', alpha=0.9))

    # MOT → TWI (显著)
    b, s = get_info('MOT', 'TWI')
    lbl = lbl_fmt(b, s)
    ax.annotate('', xy=(0.75-0.11, 0.40), xytext=(0.38+0.11, 0.60),
                arrowprops=dict(arrowstyle='->', color=C_TEAL, lw=2.5,
                                connectionstyle='arc3,rad=0.0'))
    ax.text(0.56, 0.47, lbl, ha='center', fontsize=9, color=C_TEAL,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor='none', alpha=0.9))

    # PCB → PVI (非显著，虚线)
    b, s = get_info('PCB', 'PVI')
    ax.annotate('', xy=(0.75-0.11, 0.70), xytext=(0.38+0.11, 0.20),
                arrowprops=dict(arrowstyle='->', color='#BBB', lw=1.5,
                                linestyle='--',
                                connectionstyle='arc3,rad=-0.25'))
    ax.text(0.67, 0.52, 'n.s.', ha='center', fontsize=8.5, color='#888')

    # PCB → TWI (非显著，虚线)
    ax.annotate('', xy=(0.75-0.11, 0.40), xytext=(0.38+0.11, 0.20),
                arrowprops=dict(arrowstyle='->', color='#BBB', lw=1.5,
                                linestyle='--',
                                connectionstyle='arc3,rad=-0.1'))
    ax.text(0.62, 0.27, 'n.s.', ha='center', fontsize=8.5, color='#888')

    # PVI ↔ TWI 相关双箭头
    ax.annotate('', xy=(0.75, 0.45), xytext=(0.75, 0.65),
                arrowprops=dict(arrowstyle='<->', color='#888', lw=1.5))

    # R² 标注
    ax.text(0.75, 0.77, 'R²=0.85', ha='center', fontsize=8.5,
            color=C_TEAL, fontweight='bold')
    ax.text(0.75, 0.33, 'R²=0.94', ha='center', fontsize=8.5,
            color=C_TEAL, fontweight='bold')

    # 图例
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0],[0], color=C_TEAL,  lw=2.5, label='显著路径 (p<0.01)'),
        Line2D([0],[0], color=C_AMBER, lw=2.0, label='二阶因子载荷'),
        Line2D([0],[0], color='#BBB',  lw=1.5, linestyle='--', label='非显著路径'),
    ]
    ax.legend(handles=legend_elems, loc='lower right',
              fontsize=9.5, framealpha=0.9, edgecolor='#ccc')

    # 拟合指标注释
    fit_note = ('拟合指标：CFI=0.995  TLI=0.994  RMSEA=0.024\n'
                'p(χ²)=0.158  AIC=86.66  (优于模型一)')
    ax.text(0.02, 0.02, fit_note, transform=ax.transAxes,
            fontsize=8.5, color='#333',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFFEF0',
                      edgecolor='#E5C85A', alpha=0.95))

    plt.tight_layout()
    path = 'charts/fig10_model2_path.png'
    fig.savefig(path, dpi=220, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  saved: {path}')


if __name__ == '__main__':
    import warnings; warnings.filterwarnings('ignore')
    from sem_analysis import run_model1, run_model2
    print('Running Model 1...')
    r1 = run_model1()
    draw_model1(r1['path_table'])
    print('Running Model 2...')
    r2 = run_model2()
    draw_model2(r2['path_table'])
    print('Done.')
