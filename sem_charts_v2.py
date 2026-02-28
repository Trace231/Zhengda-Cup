"""sem_charts_v2.py — SEM路径图（基于N=300新数据结果）"""
import sys, os, json, warnings
sys.path.insert(0, '.pip_pkgs'); warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

rcParams['font.family'] = ['Arial Unicode MS', 'STHeiti', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

C_BLUE='#2E4057'; C_TEAL='#048A81'; C_AMBER='#F4A261'
C_ORANGE='#E76F51'; C_GRAY='#8D99AE'; C_RED='#C0392B'

os.makedirs('charts', exist_ok=True)

with open('sem_results_v2.json', encoding='utf-8') as f:
    RES = json.load(f)

def draw_box(ax, x, y, w, h, label, sub='', fc='#EAF4F4', ec=C_BLUE, fs=10, bold=False):
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle='round,pad=0.03',
                                facecolor=fc, edgecolor=ec, linewidth=1.8, zorder=3))
    ax.text(x, y+(0.015 if sub else 0), label, ha='center', va='center',
            fontsize=fs, fontweight='bold' if bold else 'normal', color='#111', zorder=4)
    if sub:
        ax.text(x, y-0.055, sub, ha='center', va='center', fontsize=8.5, color='#555', zorder=4)

def arr(ax, x1,y1,x2,y2, label='', color=C_BLUE, lw=2.0, dash=False, rad=0.0, fs=9):
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                linestyle='--' if dash else '-',
                                connectionstyle=f'arc3,rad={rad}'))
    if label:
        mx,my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my+0.025, label, ha='center', va='bottom', fontsize=fs,
                color=color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85))

def get_m1(frm, to):
    for r in RES['m1_paths']:
        if r['路径'] == f'{frm} → {to}': return r['标准化β'], r['显著性']
    return None, 'n.s.'

def get_m2(frm, to):
    for r in RES['m2_paths']:
        if r['路径'] == f'{frm} → {to}': return r['标准化β'], r['显著性']
    return None, 'n.s.'

def lbl(b, s): return f'β={b}{s}' if s in ('*','**','***') else ''
def sig_style(s): return (C_BLUE, 2.2, False) if s in ('*','**','***') else ('#BBBBBB', 1.2, True)


# ══════════════════════════════════════════════════════════════════════════════
# 图9  模型一路径图
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 8))
fig.patch.set_facecolor('white')
ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
fit1 = RES['m1_fit']
ax.set_title(f'图9  模型一：动机-情境双轮驱动模型路径图（N=300）\n'
             f'CFI={fit1["CFI"]:.3f}  TLI={fit1["TLI"]:.3f}  RMSEA={fit1["RMSEA"]:.3f}',
             fontsize=13, fontweight='bold', pad=10)

SX,MX,BX = 0.13, 0.46, 0.82
draw_box(ax, SX,0.78, 0.22,0.10, 'SMI','社交媒体信息影响', fc='#E8F4FD', ec='#2980B9', bold=True)
draw_box(ax, SX,0.50, 0.22,0.10, 'PSR','偶像准社会关系',   fc='#E8F4FD', ec='#2980B9', bold=True)
draw_box(ax, SX,0.22, 0.22,0.10, 'CTA','城市文旅吸引力',   fc='#E8F4FD', ec='#2980B9', bold=True)
draw_box(ax, MX,0.80, 0.22,0.10, 'EEM','情感体验动机',     fc='#FEF9E7', ec=C_AMBER, bold=True)
draw_box(ax, MX,0.50, 0.22,0.10, 'GBI','群体归属感',       fc='#FEF9E7', ec=C_AMBER, bold=True)
draw_box(ax, MX,0.20, 0.22,0.10, 'RSA','仪式感/自我实现',  fc='#FEF9E7', ec=C_AMBER, bold=True)
draw_box(ax, BX,0.65, 0.22,0.10, 'PVI','观演意愿',         fc='#EAFAF1', ec=C_TEAL, bold=True)
draw_box(ax, BX,0.35, 0.22,0.10, 'TWI','旅游消费意愿',     fc='#EAFAF1', ec=C_TEAL, bold=True)
for t, x in [('情境层',SX),('动机层',MX),('行为层',BX)]:
    ax.text(x, 0.95, t, ha='center', fontsize=12, fontweight='bold', color='#444',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F0F0F0', edgecolor='#ccc'))

# 情境→动机
for (x1,y1,x2,y2,frm,to) in [
    (SX+0.11,0.78, MX-0.11,0.80, 'SMI','EEM'),
    (SX+0.11,0.78, MX-0.11,0.50, 'SMI','GBI'),
    (SX+0.11,0.50, MX-0.11,0.80, 'PSR','EEM'),
    (SX+0.11,0.50, MX-0.11,0.50, 'PSR','GBI'),
    (SX+0.11,0.50, MX-0.11,0.20, 'PSR','RSA'),
    (SX+0.11,0.22, MX-0.11,0.80, 'CTA','EEM'),
]:
    b, s = get_m1(frm, to); c, lw, dash = sig_style(s)
    arr(ax, x1,y1,x2,y2, label=lbl(b,s), color=c, lw=lw, dash=dash, fs=7.5)

# CTA→TWI 直接效应
b,s = get_m1('CTA','TWI'); c,lw,dash = sig_style(s)
ax.annotate('', xy=(BX-0.11,0.35), xytext=(SX+0.11,0.22),
            arrowprops=dict(arrowstyle='->', color=C_ORANGE, lw=2.2,
                            connectionstyle='arc3,rad=-0.28'))
ax.text(0.50,0.08, f'CTA→TWI 直接: {lbl(b,s)}', ha='center', fontsize=8.5,
        color=C_ORANGE, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=C_ORANGE, alpha=0.9))

# 动机→意愿
for (x1,y1,x2,y2,frm,to) in [
    (MX+0.11,0.80, BX-0.11,0.65, 'EEM','PVI'),
    (MX+0.11,0.50, BX-0.11,0.65, 'GBI','PVI'),
    (MX+0.11,0.20, BX-0.11,0.65, 'RSA','PVI'),
    (MX+0.11,0.80, BX-0.11,0.35, 'EEM','TWI'),
    (MX+0.11,0.50, BX-0.11,0.35, 'GBI','TWI'),
    (MX+0.11,0.20, BX-0.11,0.35, 'RSA','TWI'),
]:
    b, s = get_m1(frm, to); c, lw, dash = sig_style(s)
    arr(ax, x1,y1,x2,y2, label=lbl(b,s), color=c, lw=lw, dash=dash, fs=7.5)

ax.legend(handles=[
    Line2D([0],[0], color=C_BLUE, lw=2.2, label='显著(p<0.05)'),
    Line2D([0],[0], color='#BBB', lw=1.2, linestyle='--', label='非显著'),
    Line2D([0],[0], color=C_ORANGE, lw=2.2, label='CTA直接效应'),
], loc='lower right', fontsize=9, framealpha=0.9)

sig_paths = [(r['路径'],r['标准化β'],r['显著性']) for r in RES['m1_paths'] if r['显著性'] in ('*','**','***')]
note = '显著路径：\n'+'\n'.join(f"  {p}: β={b}{s}" for p,b,s in sig_paths)
ax.text(0.01,0.01, note, transform=ax.transAxes, fontsize=7.8, va='bottom',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFFEF0', edgecolor='#E5C85A', alpha=0.95))

plt.tight_layout()
fig.savefig('charts/fig9_model1_path.png', dpi=220, bbox_inches='tight', facecolor='white')
plt.close(); print('saved: charts/fig9_model1_path.png')


# ══════════════════════════════════════════════════════════════════════════════
# 图10  模型二路径图（含二阶因子，所有结构路径均显著）
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(13, 7))
fig.patch.set_facecolor('white')
ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
fit2 = RES['m2_fit']
ax.set_title(f'图10  模型二：动机-阻碍SEM路径图（N=300，含二阶因子MOT）\n'
             f'CFI={fit2["CFI"]:.3f}  RMSEA={fit2["RMSEA"]:.3f}  AIC={fit2["AIC"]:.1f}',
             fontsize=12, fontweight='bold', pad=10)

draw_box(ax, 0.38,0.62, 0.22,0.12, 'MOT','内在动机（二阶）', fc='#FEF5E7', ec=C_AMBER, fs=11, bold=True)
draw_box(ax, 0.13,0.82, 0.20,0.09, 'EEM','情感体验动机',     fc='#FDF2E9', ec='#E59866', bold=True)
draw_box(ax, 0.13,0.60, 0.20,0.09, 'GBI','群体归属感',       fc='#FDF2E9', ec='#E59866', bold=True)
draw_box(ax, 0.13,0.38, 0.20,0.09, 'RSA','仪式感/自我实现',  fc='#FDF2E9', ec='#E59866', bold=True)
draw_box(ax, 0.38,0.22, 0.22,0.09, 'PCB','感知成本障碍',     fc='#FDEDEC', ec=C_RED, bold=True)
draw_box(ax, 0.76,0.70, 0.22,0.10, 'PVI','观演意愿',         fc='#EAFAF1', ec=C_TEAL, bold=True)
draw_box(ax, 0.76,0.40, 0.22,0.10, 'TWI','旅游消费意愿',     fc='#EAFAF1', ec=C_TEAL, bold=True)

# R² 标注
for x,y,lv,r2 in [(0.76,0.77,'PVI',RES['m2_r2'].get('PVI','—')),
                   (0.76,0.33,'TWI',RES['m2_r2'].get('TWI','—'))]:
    ax.text(x, y, f'R²={r2}', ha='center', fontsize=9, color=C_TEAL, fontweight='bold')

# MOT → EEM/GBI/RSA 二阶载荷
loads = [('EEM',0.82,'0.528 fixed'),('GBI',0.60,'0.976***'),('RSA',0.38,'0.966***')]
for to_lv, y_pos, lb in loads:
    ax.annotate('', xy=(0.23,y_pos), xytext=(0.27,0.62),
                arrowprops=dict(arrowstyle='->', color=C_AMBER, lw=1.8))
    ax.text(0.19, (0.62+y_pos)/2+0.02, f'λ={lb}', ha='center', fontsize=8,
            color=C_AMBER, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='none', alpha=0.85))

# MOT → PVI & TWI (显著，粗线)
b,s = get_m2('MOT','PVI')
ax.annotate('', xy=(0.65,0.70), xytext=(0.49,0.62),
            arrowprops=dict(arrowstyle='->', color=C_TEAL, lw=2.8))
ax.text(0.58, 0.70, f'β={b}{s}', ha='center', fontsize=9.5, color=C_TEAL, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.9))

b,s = get_m2('MOT','TWI')
ax.annotate('', xy=(0.65,0.40), xytext=(0.49,0.62),
            arrowprops=dict(arrowstyle='->', color=C_TEAL, lw=2.8))
ax.text(0.56, 0.47, f'β={b}{s}', ha='center', fontsize=9.5, color=C_TEAL, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.9))

# PCB → PVI & TWI (显著负效应，红色)
b,s = get_m2('PCB','PVI')
ax.annotate('', xy=(0.65,0.70), xytext=(0.49,0.22),
            arrowprops=dict(arrowstyle='->', color=C_RED, lw=2.0,
                            connectionstyle='arc3,rad=-0.3'))
ax.text(0.70,0.54, f'β={b}{s}', ha='center', fontsize=9, color=C_RED, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.9))

b,s = get_m2('PCB','TWI')
ax.annotate('', xy=(0.65,0.40), xytext=(0.49,0.22),
            arrowprops=dict(arrowstyle='->', color=C_RED, lw=2.0,
                            connectionstyle='arc3,rad=-0.1'))
ax.text(0.64,0.27, f'β={b}{s}', ha='center', fontsize=9, color=C_RED, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.9))

# PVI ↔ TWI
ax.annotate('', xy=(0.76,0.45), xytext=(0.76,0.65),
            arrowprops=dict(arrowstyle='<->', color='#888', lw=1.5))

ax.legend(handles=[
    Line2D([0],[0], color=C_TEAL, lw=2.8, label='动机正效应(p<0.001)'),
    Line2D([0],[0], color=C_RED,  lw=2.0, label='成本负效应(p<0.001)'),
    Line2D([0],[0], color=C_AMBER,lw=1.8, label='二阶因子载荷'),
], loc='lower right', fontsize=10, framealpha=0.9)

ax.text(0.02,0.02,
        '核心结论：\n  MOT→PVI β=0.693***  MOT→TWI β=0.379***\n  PCB→PVI β=-0.290***  PCB→TWI β=-0.629***',
        transform=ax.transAxes, fontsize=9, va='bottom',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFFEF0', edgecolor='#E5C85A', alpha=0.95))

plt.tight_layout()
fig.savefig('charts/fig10_model2_path.png', dpi=220, bbox_inches='tight', facecolor='white')
plt.close(); print('saved: charts/fig10_model2_path.png')
