# -*- coding: utf-8 -*-
"""
正大杯国奖级别可视化代码集
包含三张核心图表：
  1. 中国省份地图热力图（pyecharts版，需Python≥3.10；matplotlib备用版可直接运行）
  2. 高颜值小提琴+箱线组合图（seaborn + matplotlib）
  3. 100%堆叠柱状图：收入层级 × 座位偏好（matplotlib）

运行环境：Python 3.9+（图1 pyecharts版需 Python 3.10+ 及 pyecharts≥2.0）
依赖安装：pip install matplotlib seaborn pandas numpy scipy
         pip install pyecharts  # 可选，用于图1动态版
"""

# ─── 通用依赖 ─────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
from matplotlib import rcParams
import seaborn as sns
from scipy import stats
import os
import warnings
warnings.filterwarnings("ignore")

# ─── 中文字体与全局样式 ──────────────────────────────────────────────────────
plt.rcParams['font.sans-serif'] = ['SimHei', 'STHeiti', 'PingFang SC',
                                    'Microsoft YaHei', 'Arial Unicode MS',
                                    'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ── 调色板定义（莫兰迪 + 科技蓝绿）─────────────────────────────────────────
MORANDI_BLUE    = "#7BA7C7"   # 莫兰迪蓝
MORANDI_MINT    = "#7DBDAB"   # 薄荷绿
MORANDI_ROSE    = "#C49A9A"   # 莫兰迪玫瑰
MORANDI_SAGE    = "#A8BCAC"   # 鼠尾草绿
MORANDI_MAUVE   = "#A99FC2"   # 薄雾紫
MORANDI_SAND    = "#C9B38E"   # 沙漠金
MORANDI_SKY     = "#9AC3D0"   # 天际蓝
MORANDI_BLUSH   = "#D4ACAC"   # 晨曦粉
MORANDI_FOREST  = "#6A9E8C"   # 森林绿
MORANDI_STEEL   = "#7A9AB5"   # 钢蓝

TECH_BLUE       = "#1A5276"   # 科技深蓝（正大杯强调色）
TECH_TEAL       = "#148F77"   # 科技青绿
DARK_NIGHT      = "#2C3E50"   # 暗夜灰（高级感）
LIGHT_BG        = "#F8F9FA"
MID_GRAY        = "#ADB5BD"
DARK_GRAY       = "#495057"

# 三色渐变：薄荷绿→深湖蓝→暗夜灰（用于堆叠图）
STACK_COLORS    = [MORANDI_MINT, MORANDI_STEEL, DARK_NIGHT]

os.makedirs("charts", exist_ok=True)

# =============================================================================
# 图1A：中国省份PPS抽样地图（matplotlib版，可直接运行）
# =============================================================================
def plot_china_map_matplotlib():
    """
    用省份中心坐标气泡图模拟中国地图热力图。
    抽中省份用渐变蓝色高亮 + 气泡大小编码Z世代人口量。
    """
    # 省份中心坐标（经度，纬度）——近似值
    provinces = {
        "黑龙江": (128.0, 48.0), "吉林": (126.0, 43.5), "辽宁": (123.0, 41.5),
        "内蒙古": (112.0, 44.0), "新疆": (86.0, 42.0), "西藏": (88.0, 31.5),
        "青海": (97.0, 35.5), "甘肃": (103.5, 36.0), "宁夏": (106.0, 37.5),
        "陕西": (109.0, 35.5), "山西": (112.5, 37.5), "河北": (115.0, 39.0),
        "北京": (116.4, 40.0), "天津": (117.2, 39.1), "山东": (118.0, 36.5),
        "河南": (113.5, 34.0), "湖北": (113.5, 31.0), "安徽": (117.0, 32.0),
        "江苏": (119.5, 33.0), "上海": (121.5, 31.2), "浙江": (120.0, 29.0),
        "江西": (116.0, 27.5), "湖南": (112.0, 27.0), "福建": (118.0, 26.0),
        "广东": (113.0, 23.5), "广西": (108.5, 23.5), "海南": (110.0, 19.5),
        "贵州": (107.0, 27.0), "云南": (102.0, 25.0), "四川": (103.5, 30.5),
        "重庆": (107.5, 29.5),
    }

    # PPS抽中的6省及其相关特征（Z世代占比%，样本贡献量）
    sampled = {
        "广东": {"z_pop": 18.2, "sample_n": 95, "label": "广东\n(n=95)"},
        "山东": {"z_pop": 17.8, "sample_n": 78, "label": "山东\n(n=78)"},
        "河北": {"z_pop": 17.1, "sample_n": 62, "label": "河北\n(n=62)"},
        "江西": {"z_pop": 18.5, "sample_n": 48, "label": "江西\n(n=48)"},
        "辽宁": {"z_pop": 16.3, "sample_n": 38, "label": "辽宁\n(n=38)"},
        "甘肃": {"z_pop": 17.9, "sample_n": 15, "label": "甘肃\n(n=15)"},
    }

    fig, ax = plt.subplots(figsize=(14, 9), facecolor=LIGHT_BG)
    ax.set_facecolor("#EBF5FB")

    # 绘制所有省份底图（灰色气泡）
    for prov, (lon, lat) in provinces.items():
        if prov in sampled or prov == "天津":
            continue
        ax.scatter(lon, lat, s=120, color="#CBD5E1", zorder=2, alpha=0.7,
                   edgecolors="white", linewidth=0.8)
        ax.text(lon, lat - 0.7, prov, ha="center", fontsize=6.5,
                color="#8898A8", zorder=3)

    # 渐变蓝色：按样本量深浅
    from matplotlib.colors import LinearSegmentedColormap
    blue_cmap = LinearSegmentedColormap.from_list(
        "techblue", ["#AED6F1", "#1A5276"], N=100)

    max_n = max(v["sample_n"] for v in sampled.values())
    for prov, info in sampled.items():
        lon, lat = provinces[prov]
        color_intensity = info["sample_n"] / max_n
        c = blue_cmap(0.3 + 0.7 * color_intensity)
        size = 300 + info["sample_n"] * 8

        ax.scatter(lon, lat, s=size, color=c, zorder=5,
                   edgecolors="white", linewidth=2.5, alpha=0.92)
        ax.text(lon, lat, info["label"], ha="center", va="center",
                fontsize=8.0, fontweight="bold", color="white", zorder=6)

        # Z世代占比标签
        ax.text(lon, lat - 1.6,
                f"Z世代占比{info['z_pop']}%",
                ha="center", fontsize=7, color=TECH_BLUE,
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec=TECH_BLUE, alpha=0.85, lw=1))

    # 天津（研究城市，特殊标注）
    tj_lon, tj_lat = provinces["天津"]
    ax.scatter(tj_lon, tj_lat, s=280, color=TECH_TEAL, zorder=7,
               marker="*", edgecolors="white", linewidth=2)
    ax.annotate("天津\n(研究城市)", (tj_lon, tj_lat),
                xytext=(tj_lon + 2.5, tj_lat + 1.2),
                fontsize=8.5, fontweight="bold", color=TECH_TEAL,
                arrowprops=dict(arrowstyle="-|>", color=TECH_TEAL, lw=1.5),
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec=TECH_TEAL, alpha=0.9))

    # 颜色图例（样本量）
    sm = plt.cm.ScalarMappable(
        cmap=blue_cmap,
        norm=plt.Normalize(vmin=0, vmax=max_n))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical",
                        fraction=0.025, pad=0.02, aspect=20)
    cbar.set_label("层次C样本量", fontsize=9, color=DARK_GRAY)
    cbar.ax.tick_params(labelsize=8)

    # 信息注释框
    note = ("【PPS抽样说明】\n"
            "k = T/n ≈ 2.35亿，r = 1.00亿\n"
            "抽中序列：1.00→3.35→5.71\n"
            "        →8.06→10.41→12.77")
    ax.text(0.02, 0.04, note, transform=ax.transAxes,
            fontsize=8, va="bottom",
            bbox=dict(boxstyle="round,pad=0.5", fc="#EAF2FF",
                      ec=MORANDI_BLUE, alpha=0.95))

    ax.set_xlim(72, 140)
    ax.set_ylim(14, 56)
    ax.set_xlabel("经度 (°E)", fontsize=10, color=DARK_GRAY)
    ax.set_ylabel("纬度 (°N)", fontsize=10, color=DARK_GRAY)
    ax.set_title("图3-1  非天津本地层（层次C）PPS整群抽样省份分布\n"
                 "气泡大小 = 样本贡献量，颜色深浅 = 样本量，★ = 研究城市天津",
                 fontsize=11, fontweight="bold", pad=12, color=DARK_GRAY)
    ax.grid(True, linestyle="--", alpha=0.3, color=MID_GRAY)

    plt.tight_layout()
    plt.savefig("charts/fig_map_pps.png", dpi=200, bbox_inches="tight",
                facecolor=LIGHT_BG)
    plt.close()
    print("  ✓ 地图已保存：charts/fig_map_pps.png")


# =============================================================================
# 图1B：pyecharts 动态地图代码（Python ≥ 3.10 + pyecharts ≥ 2.0）
# 以下代码在 Python 3.9 环境中不运行，请复制到 Python 3.10+ 环境执行
# =============================================================================
PYECHARTS_MAP_CODE = '''
# ── pyecharts 动态中国地图（需 Python 3.10+，pip install pyecharts）─────────
from pyecharts import options as opts
from pyecharts.charts import Map
from pyecharts.globals import ThemeType

# 各省份样本量数据
province_data = [
    ("广东", 95), ("山东", 78), ("河北", 62),
    ("江西", 48), ("辽宁", 38), ("甘肃", 15),
    ("天津", 200),   # 研究城市单独标注
]

c = (
    Map(init_opts=opts.InitOpts(
        width="900px", height="600px",
        theme=ThemeType.WHITE,
        page_title="PPS抽样省份分布图"
    ))
    .add(
        series_name="层次C样本量",
        data_pair=province_data,
        maptype="china",
        is_roam=True,
        label_opts=opts.LabelOpts(is_show=True, font_size=9),
        itemstyle_opts=opts.ItemStyleOpts(
            border_color="#FFFFFF",
            border_width=0.8,
        ),
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title="粉丝经济演唱会调研——PPS整群抽样省份分布",
            subtitle="气泡颜色深浅代表层次C样本量，天津为研究城市",
            title_textstyle_opts=opts.TextStyleOpts(
                font_size=14, font_weight="bold", color="#1A5276"
            ),
        ),
        visualmap_opts=opts.VisualMapOpts(
            min_=0, max_=200,
            is_piecewise=True,
            pieces=[
                {"min": 150, "label": "研究城市(天津)", "color": "#148F77"},
                {"min": 70, "max": 150, "label": "70~150份",  "color": "#1A5276"},
                {"min": 40, "max": 70,  "label": "40~70份",   "color": "#2E86C1"},
                {"min": 10, "max": 40,  "label": "10~40份",   "color": "#7FB3D3"},
                {"min": 0,  "max": 10,  "label": "未抽中",    "color": "#D6EAF8"},
            ],
            textstyle_opts=opts.TextStyleOpts(color="#333333"),
        ),
        tooltip_opts=opts.TooltipOpts(
            trigger="item",
            formatter="{b}<br/>样本量：{c}份",
        ),
    )
    .render("charts/fig_map_pyecharts.html")
)
print("pyecharts地图已保存：charts/fig_map_pyecharts.html")
'''

# =============================================================================
# 图2：高颜值小提琴 + 箱线组合图（seaborn + matplotlib）
# =============================================================================
def plot_violin_box():
    """
    九大构念得分分布（小提琴 + 内嵌箱线 + 均值菱形 + 显著性标注）
    学术级配色：莫兰迪色调，去除多余边框，符合顶级期刊排版标准
    """
    # ── 读取数据 ─────────────────────────────────────────────────────────────
    SCALE_COLS = [f"Scale_{i}_{j}" for i in range(1, 10) for j in range(1, 4)]
    CONSTRUCTS = {
        "SMI": [0, 1, 2], "PSR": [3, 4, 5], "CTA": [6, 7, 8],
        "EEM": [9, 10, 11], "GBI": [12, 13, 14], "RSA": [15, 16, 17],
        "PCB": [18, 19, 20], "PVI": [21, 22, 23], "TWI": [24, 25, 26],
    }
    LABELS = {
        "SMI": "SMI\n社交媒体\n信息影响",
        "PSR": "PSR\n偶像准\n社会关系",
        "CTA": "CTA\n城市旅游\n吸引力",
        "EEM": "EEM\n情感体验\n动机",
        "GBI": "GBI\n群体归属\n认同",
        "RSA": "RSA\n仪式感\n自我实现",
        "PCB": "PCB\n感知成本\n障碍",
        "PVI": "PVI\n观演意愿",
        "TWI": "TWI\n旅游消费\n延伸意愿",
    }

    df = pd.read_csv("survey_clean.csv", encoding="utf-8-sig")
    scale_cols = [c for c in SCALE_COLS if c in df.columns]
    data = df[scale_cols].apply(pd.to_numeric, errors="coerce")

    score_df = pd.DataFrame()
    for k, idxs in CONSTRUCTS.items():
        cols = [data.columns[i] for i in idxs]
        score_df[k] = data[cols].mean(axis=1)

    # Long format for seaborn
    score_long = score_df.melt(var_name="Construct", value_name="Score")
    keys = list(CONSTRUCTS.keys())

    # 颜色：高动机构念用蓝绿系，阻碍类用玫红，意愿类用橙金
    colors = {
        "SMI": MORANDI_SKY,    "PSR": MORANDI_BLUE,   "CTA": MORANDI_SAGE,
        "EEM": MORANDI_MINT,   "GBI": MORANDI_FOREST, "RSA": MORANDI_STEEL,
        "PCB": MORANDI_ROSE,   "PVI": MORANDI_SAND,   "TWI": MORANDI_MAUVE,
    }
    palette = [colors[k] for k in keys]

    # ── 绘图 ─────────────────────────────────────────────────────────────────
    sns.set_theme(style="ticks", font_scale=0.95)
    fig, ax = plt.subplots(figsize=(18, 7.5), facecolor="white")
    ax.set_facecolor("#FAFAFA")

    # 小提琴图
    vp = sns.violinplot(
        data=score_long,
        x="Construct", y="Score",
        order=keys,
        palette=palette,
        inner=None,           # 不显示内部箱线，后面手绘
        cut=0,                # 不延伸超出数据范围
        width=0.75,
        linewidth=0.8,
        saturation=0.85,
        ax=ax,
    )
    # 降低透明度使色彩更雅致
    for violin in ax.collections:
        violin.set_alpha(0.55)

    # 手绘箱线（更细腻的控制）
    positions = range(len(keys))
    for i, k in enumerate(keys):
        d = score_df[k].dropna().values
        q1, med, q3 = np.percentile(d, [25, 50, 75])
        iqr = q3 - q1
        whislo = max(d.min(), q1 - 1.5 * iqr)
        whishi = min(d.max(), q3 + 1.5 * iqr)
        mean_val = d.mean()
        c = colors[k]

        # 须线
        ax.plot([i, i], [whislo, q1], color=c, lw=1.5, zorder=3)
        ax.plot([i, i], [q3, whishi], color=c, lw=1.5, zorder=3)
        # IQR box
        box = FancyBboxPatch((i - 0.12, q1), 0.24, iqr,
                              boxstyle="round,pad=0.01",
                              facecolor=c, edgecolor="white",
                              linewidth=1.5, alpha=0.88, zorder=4)
        ax.add_patch(box)
        # 中位数线
        ax.plot([i - 0.12, i + 0.12], [med, med],
                color="white", lw=2.0, zorder=5)
        # 均值菱形
        ax.scatter(i, mean_val, marker="D", s=55,
                   color="white", edgecolors=c,
                   linewidth=1.8, zorder=6)
        # 均值数字标注
        ax.text(i, mean_val + 0.22, f"{mean_val:.2f}",
                ha="center", fontsize=8, color=c, fontweight="bold", zorder=7)

    # 参考线：Likert量表中性点 3.0
    ax.axhline(3.0, color=MID_GRAY, lw=1.0, ls=":", alpha=0.8, zorder=1)
    ax.text(-0.55, 3.05, "中性\n基准线(3)", ha="right", va="bottom",
            fontsize=7.5, color=MID_GRAY)

    # 分组背景色带（动机类 vs 阻碍类 vs 意愿类）
    ax.axvspan(-0.5, 5.5, alpha=0.04, color=MORANDI_MINT)   # 动机类
    ax.axvspan(5.5, 6.5, alpha=0.06, color=MORANDI_ROSE)    # 阻碍类
    ax.axvspan(6.5, 8.5, alpha=0.04, color=MORANDI_SAND)    # 意愿类

    ax.text(2.5, 0.65, "► 情境-动机维度（SMI/PSR/CTA/EEM/GBI/RSA）",
            ha="center", fontsize=8, color=MORANDI_FOREST, style="italic")
    ax.text(6.0, 0.65, "►阻碍", ha="center", fontsize=8,
            color=MORANDI_ROSE, style="italic")
    ax.text(7.5, 0.65, "► 意愿维度（PVI/TWI）",
            ha="center", fontsize=8, color=MORANDI_SAND, style="italic")

    # 坐标轴
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([LABELS[k] for k in keys], fontsize=9, color=DARK_GRAY)
    ax.set_ylim(0.5, 6.0)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_ylabel("构念量表均值（Likert 1–5分）", fontsize=10.5, color=DARK_GRAY)
    ax.set_xlabel("")

    ax.set_title(
        "图6  九大构念量表得分分布形态（小提琴 + 箱线 + 均值菱形）\n"
        "N=712；◆=均值；白横线=中位数；误差条=1.5×IQR；"
        "色带：蓝绿=情境动机 · 玫红=感知阻碍 · 金=行为意愿",
        fontsize=11, fontweight="bold", pad=12, color=DARK_GRAY
    )

    sns.despine(ax=ax, top=True, right=True)
    ax.tick_params(axis="y", which="both", left=True, length=4)
    ax.yaxis.grid(True, ls="--", alpha=0.4, color="#E0E0E0")
    ax.set_axisbelow(True)

    # 图例
    legend_elements = [
        mpatches.Patch(color=MORANDI_MINT, alpha=0.7, label="情境-动机类构念"),
        mpatches.Patch(color=MORANDI_ROSE, alpha=0.7, label="感知成本阻碍（PCB）"),
        mpatches.Patch(color=MORANDI_SAND, alpha=0.7, label="行为意向类构念"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="white",
                   markeredgecolor=DARK_GRAY, markersize=7, label="均值 ◆"),
    ]
    ax.legend(handles=legend_elements, loc="upper right",
              fontsize=9, framealpha=0.95, ncol=2,
              edgecolor=MID_GRAY, fancybox=True)

    plt.tight_layout()
    plt.savefig("charts/fig6_violin_premium.png", dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("  ✓ 小提琴图已保存：charts/fig6_violin_premium.png")


# =============================================================================
# 图3：100%堆叠柱状图 —— 收入层级 × 座位偏好（薄荷绿→深湖蓝→暗夜灰）
# =============================================================================
def plot_stacked_bar_income_seat():
    """
    100%堆叠柱状图：展示不同收入层级的座位档次选择分布
    配色：薄荷绿（基础档）→ 科技深湖蓝（进阶档）→ 暗夜灰（高端档）
    """
    df = pd.read_csv("survey_clean.csv", encoding="utf-8-sig")

    income_order = ["1000元及以下", "1001-3000元", "3001-6000元",
                    "6001-10000元", "10000元以上"]
    seat_order   = ["基础档（普通看台）", "进阶档（优选看台）", "高端档（内场）"]
    seat_labels  = ["基础档\n（普通看台）", "进阶档\n（优选看台）", "高端档\n（内场）"]
    income_labels = ["≤1000元\n(n=59)", "1001~3000元\n(n=248)",
                     "3001~6000元\n(n=195)", "6001~1万元\n(n=128)", "≥1万元\n(n=82)"]

    ct = pd.crosstab(df["Q11_4_income"], df["Q6"])
    ct = ct.reindex(income_order)[seat_order]
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

    # 精确百分比数据
    vals = ct_pct.values  # shape (5, 3)

    # ── 绘图 ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")
    ax.set_facecolor("#FAFAFA")

    x = np.arange(len(income_order))
    bar_w = 0.62
    bar_colors = [MORANDI_MINT, MORANDI_STEEL, DARK_NIGHT]

    bottoms = np.zeros(len(income_order))
    bars = []
    for j, (seat_lbl, color) in enumerate(zip(seat_labels, bar_colors)):
        pct = vals[:, j]
        bar = ax.bar(x, pct, bottom=bottoms, width=bar_w,
                     color=color, edgecolor="white", linewidth=1.2,
                     label=seat_lbl, zorder=3)
        bars.append(bar)

        # 柱内百分比标签
        for i, (b, p) in enumerate(zip(bottoms, pct)):
            if p >= 6.0:   # 仅在足够宽时标注
                mid_y = b + p / 2
                txt_color = "white" if color in [MORANDI_STEEL, DARK_NIGHT] else DARK_GRAY
                ax.text(x[i], mid_y, f"{p:.1f}%",
                        ha="center", va="center", fontsize=9.5,
                        color=txt_color, fontweight="bold", zorder=5)
        bottoms += pct

    # ── 趋势标注：进阶档峰值与高端档梯度 ────────────────────────────────────
    # 绘制高端档趋势线
    high_end_pct = vals[:, 2]
    ax.plot(x, bottoms - vals[:, 2] + vals[:, 2] / 2,
            color=MORANDI_STEEL, lw=2, ls="--",
            marker="o", markersize=6, markerfacecolor="white",
            markeredgecolor=MORANDI_STEEL, markeredgewidth=2,
            zorder=6, label="高端档占比趋势")

    # 关键数据点注释
    annotations = [
        (0, vals[0, 2], f"仅{vals[0,2]:.1f}%\n高端档",  0.5, -4,  DARK_NIGHT),
        (1, vals[1, 1], f"进阶档\n峰值29.0%", 1,   45,  MORANDI_STEEL),
        (4, vals[4, 2], f"高端档\n{vals[4,2]:.1f}%",  3.5, -8, DARK_NIGHT),
    ]
    for xi, yi_rel, text, tx, ty, tc in annotations:
        ypos = (bottoms[xi] - vals[xi, 2] + vals[xi, 2] / 2
                if "高端" in text
                else sum(vals[xi, :2]) - vals[xi, 1] / 2)
        ax.annotate(text, (xi, ypos),
                    xytext=(tx, ty), textcoords="offset points",
                    fontsize=8.5, color=tc, fontweight="bold",
                    arrowprops=dict(arrowstyle="-|>", color=tc, lw=1.2),
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec=tc, alpha=0.9))

    # 坐标轴
    ax.set_xticks(x)
    ax.set_xticklabels(income_labels, fontsize=10, color=DARK_GRAY)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"],
                       fontsize=9, color=DARK_GRAY)
    ax.set_ylim(0, 110)
    ax.set_ylabel("各座位档次占比（%）", fontsize=10.5, color=DARK_GRAY)
    ax.set_xlabel("月可支配收入层级", fontsize=10.5, color=DARK_GRAY)
    ax.set_title(
        "图8  不同收入层级Z世代的演唱会座位档次偏好分布（N=712）\n"
        "薄荷绿=基础档 · 钢蓝=进阶档 · 暗夜灰=高端内场；虚线为高端档占比趋势",
        fontsize=11, fontweight="bold", pad=12, color=DARK_GRAY
    )

    sns.despine(ax=ax, top=True, right=True)
    ax.yaxis.grid(True, ls="--", alpha=0.35, color="#E0E0E0", zorder=0)
    ax.set_axisbelow(True)

    # 图例
    ax.legend(
        handles=[bars[0], bars[1], bars[2]],
        labels=seat_labels,
        loc="upper right",
        fontsize=10, framealpha=0.95,
        edgecolor=MID_GRAY, ncol=1
    )

    # 数据来源注释
    ax.text(0.99, 0.01,
            "数据来源：本研究问卷调查（N=712）",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.5, color=MID_GRAY)

    plt.tight_layout()
    plt.savefig("charts/fig8_stacked_income_seat.png", dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("  ✓ 100%堆叠柱状图已保存：charts/fig8_stacked_income_seat.png")


# =============================================================================
# 主入口
# =============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("  正大杯国奖级别可视化图表生成（莫兰迪 × 科技蓝绿配色）")
    print("=" * 65)

    print("\n[图1] 生成PPS抽样省份地图（matplotlib版）...")
    plot_china_map_matplotlib()

    print("\n[图2] 生成九大构念小提琴+箱线组合图...")
    plot_violin_box()

    print("\n[图3] 生成收入层级×座位偏好100%堆叠柱状图...")
    plot_stacked_bar_income_seat()

    print("\n" + "=" * 65)
    print("  所有图表已生成，保存至 charts/ 目录：")
    print("  · charts/fig_map_pps.png           (地图)")
    print("  · charts/fig6_violin_premium.png   (小提琴图)")
    print("  · charts/fig8_stacked_income_seat.png (堆叠图)")
    print("=" * 65)
    print()
    print("  【pyecharts动态地图说明】")
    print("  以下代码段需在 Python 3.10+ 环境中运行：")
    print()
    print(PYECHARTS_MAP_CODE)
