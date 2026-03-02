# -*- coding: utf-8 -*-
"""
高质量可视化代码集 - 正大杯国奖级别
包含：
  1. 全局莫兰迪色系 + 科技蓝样式配置
  2. 中国地图可视化（PPS抽样6省份高亮）
  3. 升级版雷达图（双组对比，带数据标签）
  4. 升级版小提琴图（带均值标注和显著性标记）

依赖：matplotlib, numpy, pandas, scipy
（不需要 pyecharts/geopandas，使用 matplotlib 内置方案）
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.gridspec import GridSpec
from matplotlib import rcParams
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────
# 1. 全局样式配置 —— 莫兰迪色系 + 学术排版
# ─────────────────────────────────────────────────────────────────────

# 莫兰迪色系（低饱和度优雅色板）
MORANDI = {
    "blue":    "#8BA7C7",   # 莫兰迪蓝
    "green":   "#91B5A0",   # 莫兰迪绿
    "rose":    "#C7A0A0",   # 莫兰迪玫瑰
    "sage":    "#B5C4B5",   # 莫兰迪鼠尾草
    "mauve":   "#B09FBF",   # 莫兰迪紫藕
    "sand":    "#C9B99A",   # 莫兰迪沙色
    "sky":     "#A8C8D5",   # 莫兰迪天蓝
    "terra":   "#C4A882",   # 莫兰迪陶土
    "mist":    "#B8BFCC",   # 莫兰迪薄雾
    "blush":   "#D4B5B5",   # 莫兰迪腮红
}

# 科技蓝强调色（正大杯常用）
TECH_BLUE = "#1A5276"
TECH_GREEN = "#1E8449"
ACCENT_RED = "#B03A2E"
LIGHT_GRAY = "#F2F4F7"
MID_GRAY   = "#ADB5BD"
DARK_GRAY  = "#343A40"

PALETTE = list(MORANDI.values())

def set_global_style():
    """设置全局 matplotlib 样式"""
    rcParams.update({
        "font.family":        ["SimHei", "Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "figure.facecolor":   "white",
        "axes.facecolor":     "#FAFAFA",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.spines.left":   True,
        "axes.spines.bottom": True,
        "axes.edgecolor":     MID_GRAY,
        "axes.linewidth":     0.8,
        "xtick.color":        DARK_GRAY,
        "ytick.color":        DARK_GRAY,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "axes.labelsize":     10,
        "axes.titlesize":     12,
        "legend.framealpha":  0.9,
        "legend.fontsize":    9,
        "grid.color":         "#E0E0E0",
        "grid.linewidth":     0.5,
    })

set_global_style()


# ─────────────────────────────────────────────────────────────────────
# 2. 中国地图可视化 —— PPS抽样6省份高亮
# ─────────────────────────────────────────────────────────────────────
# 使用简化的中国省份轮廓坐标（矩形近似+标注方式）
# 如需精准地图请安装 geopandas + 国家测绘局标准底图

def create_china_sampling_map():
    """
    绘制中国PPS抽样地图（简化版，用省份矩形+连线表示）
    如有 geopandas + shapefile，可替换为真实省界多边形
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 9))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#EBF5FB")

    # 中国各省份简化中心坐标 (经度, 纬度) —— 近似值
    province_coords = {
        "黑龙江": (128, 48), "吉林": (126, 44), "辽宁": (123, 41),
        "内蒙古": (112, 44), "新疆": (86, 42), "西藏": (88, 31),
        "青海": (97, 36), "甘肃": (103, 36), "宁夏": (106, 37),
        "陕西": (109, 35), "山西": (112, 37), "河北": (115, 39),
        "北京": (116, 40), "天津": (117, 39), "山东": (118, 36),
        "河南": (113, 34), "湖北": (113, 31), "安徽": (117, 32),
        "江苏": (120, 33), "上海": (121, 31), "浙江": (120, 29),
        "江西": (116, 27), "湖南": (112, 27), "福建": (118, 26),
        "广东": (113, 23), "广西": (108, 23), "海南": (110, 19),
        "贵州": (107, 27), "云南": (102, 25), "四川": (103, 30),
        "重庆": (107, 29),
    }

    # 抽中的6个省份
    sampled = {
        "广东": {"pop": 1.26, "z_ratio": 18.2, "note": "经济第一省\n粉丝经济活跃"},
        "山东": {"pop": 1.02, "z_ratio": 17.8, "note": "人口大省\n演唱会市场成熟"},
        "河北": {"pop": 0.75, "z_ratio": 17.1, "note": "毗邻天津\n来津观演便利"},
        "江西": {"pop": 0.45, "z_ratio": 18.5, "note": "Z世代占比高\n网红文化兴盛"},
        "辽宁": {"pop": 0.42, "z_ratio": 16.3, "note": "东北粉丝聚集\n追星热情高"},
        "甘肃": {"pop": 0.25, "z_ratio": 17.9, "note": "西部代表省\n长途追星意愿强"},
    }

    # 绘制所有省份（灰色圆点）
    for prov, (lon, lat) in province_coords.items():
        if prov in sampled:
            continue
        ax.scatter(lon, lat, s=80, color=MORANDI["mist"], zorder=3, alpha=0.7)
        ax.annotate(prov, (lon, lat), fontsize=6.5, ha="center", va="bottom",
                    color="#888888", xytext=(0, 5), textcoords="offset points")

    # 绘制天津（特殊标注）
    if "天津" in province_coords:
        lon, lat = province_coords["天津"]
        ax.scatter(lon, lat, s=200, color=TECH_BLUE, zorder=5, marker="*",
                   edgecolors="white", linewidth=1.5)
        ax.annotate("天津（研究地）", (lon, lat), fontsize=8.5, ha="left", va="bottom",
                    color=TECH_BLUE, fontweight="bold",
                    xytext=(5, 5), textcoords="offset points",
                    arrowprops=dict(arrowstyle="-", color=TECH_BLUE, lw=1))

    # 绘制抽中的6个省份（彩色高亮）
    sample_colors = [MORANDI["blue"], MORANDI["rose"], MORANDI["green"],
                     MORANDI["mauve"], MORANDI["sky"], MORANDI["sand"]]

    for idx, (prov, info) in enumerate(sampled.items()):
        lon, lat = province_coords[prov]
        color = sample_colors[idx]

        # 大圆点
        ax.scatter(lon, lat, s=350, color=color, zorder=6,
                   edgecolors="white", linewidth=2.5)
        # 序号标注
        ax.text(lon, lat, str(idx + 1), ha="center", va="center",
                fontsize=9, fontweight="bold", color="white", zorder=7)
        # 省份名称
        offset_x = 12 if lon < 115 else -12
        offset_y = 8 if lat < 30 else -10
        ax.annotate(
            f"{prov}\n(Pop:{info['pop']:.2f}亿)",
            (lon, lat),
            xytext=(offset_x, offset_y), textcoords="offset points",
            fontsize=8, fontweight="bold", color=color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=color, alpha=0.9, linewidth=1.5),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.2)
        )

    # 图例
    legend_elements = [
        mpatches.Patch(color=TECH_BLUE, label="天津（研究城市）"),
        mpatches.Patch(color=MORANDI["blue"], label="PPS抽中省份 (n=6)"),
        mpatches.Patch(color=MORANDI["mist"], label="未抽中省份"),
    ]
    ax.legend(handles=legend_elements, loc="lower left",
              fontsize=9, framealpha=0.95,
              edgecolor=MID_GRAY, fancybox=True)

    # 注释框：各省Z世代特征
    annot_text = (
        "【PPS抽样6省Z世代特征简析】\n"
        "① 广东 — 经济第一省，粉丝经济最活跃，演唱会消费力强；\n"
        "② 山东 — 人口大省，演唱会市场成熟，跨城追星比例高；\n"
        "③ 河北 — 毗邻天津，地理优势突出，本地化来津比例高；\n"
        "④ 江西 — Z世代比例高(18.5%)，网红经济与粉丝文化兴盛；\n"
        "⑤ 辽宁 — 东北粉丝集聚，追星消费占比高于全国均值；\n"
        "⑥ 甘肃 — 西部代表，长途追星意愿强，文旅结合期望高。"
    )
    ax.text(0.02, 0.02, annot_text, transform=ax.transAxes,
            fontsize=7.5, verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#EAF2FF",
                      edgecolor=MORANDI["blue"], alpha=0.95),
            family="SimHei")

    ax.set_xlim(73, 138)
    ax.set_ylim(15, 56)
    ax.set_xlabel("经度 (°E)", fontsize=10)
    ax.set_ylabel("纬度 (°N)", fontsize=10)
    ax.set_title(
        "图3-1  非天津本地层（层次C）PPS整群抽样省份分布图\n"
        "（抽样间距k≈2.35亿人，随机起点r=1.00亿，共抽取6个省份）",
        fontsize=11, fontweight="bold", pad=15, color=DARK_GRAY
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    # Add a simple "China outline" note
    ax.text(0.98, 0.98,
            "注：省份位置为近似中心坐标\n数据来源：第七次全国人口普查",
            transform=ax.transAxes, fontsize=7, ha="right", va="top",
            color=MID_GRAY)

    plt.tight_layout()
    plt.savefig("charts/fig_sampling_map.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print("  ✓ 抽样地图已保存至 charts/fig_sampling_map.png")


# ─────────────────────────────────────────────────────────────────────
# 3. 升级版雷达图 —— 双组对比（有/无偶像），莫兰迪配色+数据标签
# ─────────────────────────────────────────────────────────────────────

def create_upgraded_radar():
    """双组对比雷达图：有偶像 vs 无偶像，莫兰迪配色"""
    import pandas as pd

    SCALE_COLS = [f"Scale_{i}_{j}" for i in range(1, 10) for j in range(1, 4)]
    CONSTRUCTS = {
        "SMI": [0, 1, 2], "PSR": [3, 4, 5], "CTA": [6, 7, 8],
        "EEM": [9, 10, 11], "GBI": [12, 13, 14], "RSA": [15, 16, 17],
        "PCB": [18, 19, 20], "PVI": [21, 22, 23], "TWI": [24, 25, 26],
    }
    LABELS_CN = {
        "SMI": "社交媒体\n信息影响", "PSR": "偶像准\n社会关系",
        "CTA": "城市旅游\n吸引力", "EEM": "情感体验\n动机",
        "GBI": "群体归属\n认同", "RSA": "仪式感\n自我实现",
        "PCB": "感知成本\n障碍", "PVI": "观演意愿",
        "TWI": "旅游消费\n延伸意愿"
    }

    df = pd.read_csv("survey_clean.csv", encoding="utf-8-sig")
    scale_cols = [c for c in SCALE_COLS if c in df.columns]
    data = df[scale_cols].apply(pd.to_numeric, errors="coerce")

    # Compute construct means
    score_df = pd.DataFrame()
    for key, idxs in CONSTRUCTS.items():
        cols = [data.columns[i] for i in idxs]
        score_df[key] = data[cols].mean(axis=1)
    score_df["has_idol"] = df["Q2"].map(lambda x: "有偶像" if x == "有" else "无/泛爱好")

    keys = list(CONSTRUCTS.keys())
    n = len(keys)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles_closed = angles + [angles[0]]

    # Compute group means
    g1 = score_df[score_df["has_idol"] == "有偶像"]
    g2 = score_df[score_df["has_idol"] == "无/泛爱好"]
    vals1 = [g1[k].mean() for k in keys] + [g1[keys[0]].mean()]
    vals2 = [g2[k].mean() for k in keys] + [g2[keys[0]].mean()]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5),
                              subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "图5  九大构念量表均值雷达图对比（左：有/无偶像分组；右：高/低频观演分组）\n"
        "N=712，量表均值（1-5分制）",
        fontsize=11, fontweight="bold", y=1.01, color=DARK_GRAY
    )

    def draw_radar(ax, group_data_list, group_names, colors, title_text):
        ax.set_facecolor("#FAFBFC")
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        # Draw grid circles
        for r in [1, 2, 3, 4, 5]:
            ax.plot(angles_closed, [r] * (n + 1), color="#E0E0E0",
                    linewidth=0.6, linestyle="--", zorder=1)
            if r < 5:
                ax.text(angles[0], r + 0.05, str(r), ha="center", va="bottom",
                        fontsize=7, color=MID_GRAY)

        for gdata, gname, gcolor in zip(group_data_list, group_names, colors):
            ax.plot(angles_closed, gdata, color=gcolor, linewidth=2.2,
                    linestyle="-", zorder=3, label=gname)
            ax.fill(angles_closed, gdata, color=gcolor, alpha=0.12, zorder=2)
            # Data labels at each vertex
            for i, (angle, val) in enumerate(zip(angles, gdata[:-1])):
                ax.annotate(f"{val:.2f}",
                            xy=(angle, val),
                            xytext=(angle, val + 0.18),
                            ha="center", va="center",
                            fontsize=7.5, color=gcolor, fontweight="bold",
                            zorder=5)

        ax.set_xticks(angles)
        ax.set_xticklabels([LABELS_CN[k] for k in keys],
                           fontsize=8.5, color=DARK_GRAY)
        ax.set_ylim(0, 5.5)
        ax.set_yticks([])
        ax.set_title(title_text, pad=20, fontsize=10, fontweight="bold",
                     color=DARK_GRAY)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
                  fontsize=9, framealpha=0.9)
        ax.spines["polar"].set_color(MID_GRAY)

    # Left: 有/无偶像
    draw_radar(axes[0],
               [vals1, vals2],
               [f"有偶像 (n={len(g1)})", f"无偶像/泛爱好 (n={len(g2)})"],
               [MORANDI["blue"], MORANDI["rose"]],
               "分组一：有偶像 vs 无偶像/泛演出爱好者")

    # Right: 高/低频观演
    freq_map = {"1-3次": "低频", "4-6次": "中频", "7次及以上": "高频"}
    score_df["freq"] = df["Q1"].map(lambda x: "高频(7次+)" if "7次" in str(x) else "低频(1-3次)")
    gH = score_df[score_df["freq"] == "高频(7次+)"]
    gL = score_df[score_df["freq"] == "低频(1-3次)"]
    valsH = [gH[k].mean() for k in keys] + [gH[keys[0]].mean()]
    valsL = [gL[k].mean() for k in keys] + [gL[keys[0]].mean()]

    draw_radar(axes[1],
               [valsH, valsL],
               [f"高频观演(n={len(gH)})", f"低频观演(n={len(gL)})"],
               [MORANDI["green"], MORANDI["mauve"]],
               "分组二：高频观演(≥7次) vs 低频观演(1-3次)")

    plt.tight_layout()
    plt.savefig("charts/fig5_radar_upgraded.png", dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("  ✓ 升级版雷达图已保存至 charts/fig5_radar_upgraded.png")


# ─────────────────────────────────────────────────────────────────────
# 4. 升级版小提琴图 —— 九大构念得分分布（莫兰迪色 + 均值线 + 显著性）
# ─────────────────────────────────────────────────────────────────────

def create_upgraded_violin():
    """升级版小提琴图：莫兰迪配色，带均值标注，有/无偶像分组对比"""
    SCALE_COLS = [f"Scale_{i}_{j}" for i in range(1, 10) for j in range(1, 4)]
    CONSTRUCTS = {
        "SMI": [0, 1, 2], "PSR": [3, 4, 5], "CTA": [6, 7, 8],
        "EEM": [9, 10, 11], "GBI": [12, 13, 14], "RSA": [15, 16, 17],
        "PCB": [18, 19, 20], "PVI": [21, 22, 23], "TWI": [24, 25, 26],
    }
    LABELS_CN = ["社交媒体\n信息影响", "偶像准\n社会关系", "城市旅游\n吸引力",
                 "情感体验\n动机", "群体归属\n认同", "仪式感\n自我实现",
                 "感知成本\n障碍", "观演意愿", "旅游消费\n延伸意愿"]

    df = pd.read_csv("survey_clean.csv", encoding="utf-8-sig")
    scale_cols = [c for c in SCALE_COLS if c in df.columns]
    data = df[scale_cols].apply(pd.to_numeric, errors="coerce")

    score_df = pd.DataFrame()
    for key, idxs in CONSTRUCTS.items():
        cols = [data.columns[i] for i in idxs]
        score_df[key] = data[cols].mean(axis=1)
    score_df["has_idol"] = df["Q2"].map(lambda x: True if x == "有" else False)

    keys = list(CONSTRUCTS.keys())

    fig, ax = plt.subplots(figsize=(18, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFAFA")

    positions = np.arange(len(keys))
    width = 0.35

    for i, key in enumerate(keys):
        # Data for two groups
        d_idol = score_df[score_df["has_idol"] == True][key].dropna().values
        d_noido = score_df[score_df["has_idol"] == False][key].dropna().values

        # Violin for group 1 (有偶像)
        vp1 = ax.violinplot([d_idol], positions=[positions[i] - width / 2],
                             widths=0.32, showmedians=False, showextrema=False)
        for pc in vp1["bodies"]:
            pc.set_facecolor(MORANDI["blue"])
            pc.set_edgecolor(MORANDI["blue"])
            pc.set_alpha(0.55)

        # Violin for group 2 (无偶像)
        vp2 = ax.violinplot([d_noido], positions=[positions[i] + width / 2],
                             widths=0.32, showmedians=False, showextrema=False)
        for pc in vp2["bodies"]:
            pc.set_facecolor(MORANDI["rose"])
            pc.set_edgecolor(MORANDI["rose"])
            pc.set_alpha(0.55)

        # Box plots on top
        bp1 = ax.boxplot([d_idol], positions=[positions[i] - width / 2],
                          widths=0.12, patch_artist=True,
                          medianprops=dict(color="white", linewidth=2),
                          boxprops=dict(facecolor=MORANDI["blue"], alpha=0.8),
                          whiskerprops=dict(color=MORANDI["blue"]),
                          capprops=dict(color=MORANDI["blue"]),
                          flierprops=dict(marker=".", markersize=2,
                                          markerfacecolor=MORANDI["blue"], alpha=0.3))
        bp2 = ax.boxplot([d_noido], positions=[positions[i] + width / 2],
                          widths=0.12, patch_artist=True,
                          medianprops=dict(color="white", linewidth=2),
                          boxprops=dict(facecolor=MORANDI["rose"], alpha=0.8),
                          whiskerprops=dict(color=MORANDI["rose"]),
                          capprops=dict(color=MORANDI["rose"]),
                          flierprops=dict(marker=".", markersize=2,
                                          markerfacecolor=MORANDI["rose"], alpha=0.3))

        # Mean markers
        m1 = np.mean(d_idol)
        m2 = np.mean(d_noido)
        ax.scatter(positions[i] - width / 2, m1, s=50, color="white",
                   zorder=5, marker="D", edgecolors=MORANDI["blue"], linewidth=1.5)
        ax.scatter(positions[i] + width / 2, m2, s=50, color="white",
                   zorder=5, marker="D", edgecolors=MORANDI["rose"], linewidth=1.5)

        # Mean value labels
        ax.text(positions[i] - width / 2, m1 + 0.18, f"{m1:.2f}",
                ha="center", fontsize=7.5, color=MORANDI["blue"], fontweight="bold")
        ax.text(positions[i] + width / 2, m2 + 0.18, f"{m2:.2f}",
                ha="center", fontsize=7.5, color=MORANDI["rose"], fontweight="bold")

        # t-test significance stars
        t_stat, p_val = stats.ttest_ind(d_idol, d_noido, equal_var=False)
        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "n.s."))
        top = max(np.percentile(d_idol, 97), np.percentile(d_noido, 97)) + 0.1
        ax.annotate("", xy=(positions[i] + width / 2, top + 0.05),
                    xytext=(positions[i] - width / 2, top + 0.05),
                    arrowprops=dict(arrowstyle="-", color=MID_GRAY, lw=1))
        ax.text(positions[i], top + 0.12, sig, ha="center", fontsize=9,
                color=(ACCENT_RED if sig != "n.s." else MID_GRAY), fontweight="bold")

    ax.set_xticks(positions)
    ax.set_xticklabels(LABELS_CN, fontsize=9)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_ylim(0.5, 6.2)
    ax.set_ylabel("量表均值（Likert 1–5）", fontsize=10)
    ax.set_title(
        "图6  各构念量表得分分布形态（小提琴+箱形图）\n"
        "蓝：有偶像组(n=594)；玫红：无偶像/泛爱好组(n=118)；◆=均值；***p<.001",
        fontsize=11, fontweight="bold", pad=12, color=DARK_GRAY
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    # Reference line at y=3 (neutral point)
    ax.axhline(y=3, color=MID_GRAY, linestyle=":", linewidth=1, alpha=0.7)
    ax.text(-0.5, 3.05, "中性基准\n(=3)", fontsize=7.5, color=MID_GRAY, ha="right")

    # Legend
    legend_elements = [
        mpatches.Patch(color=MORANDI["blue"], alpha=0.7, label="有偶像组 (n=594)"),
        mpatches.Patch(color=MORANDI["rose"], alpha=0.7, label="无偶像/泛爱好组 (n=118)"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="white",
                   markeredgecolor=DARK_GRAY, markersize=7, label="均值 ◆"),
    ]
    ax.legend(handles=legend_elements, loc="upper right",
              fontsize=9, framealpha=0.95, ncol=3)

    plt.tight_layout()
    plt.savefig("charts/fig8_violin_upgraded.png", dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("  ✓ 升级版小提琴图已保存至 charts/fig8_violin_upgraded.png")


# ─────────────────────────────────────────────────────────────────────
# 5. SEM路径系数可视化（升级版）
# ─────────────────────────────────────────────────────────────────────

def create_sem_path_chart():
    """绘制模型二SEM路径图（莫兰迪色系，带标准化系数和显著性标注）"""
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def draw_box(ax, cx, cy, text, color, width=2.0, height=0.75, fs=9.5):
        box = FancyBboxPatch((cx - width / 2, cy - height / 2), width, height,
                              boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor="white",
                              linewidth=2, alpha=0.92, zorder=3)
        ax.add_patch(box)
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
                fontweight="bold", color="white", zorder=4)

    def draw_arrow(ax, x1, y1, x2, y2, coef, color, style="->"):
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2 + 0.1
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color,
                                    lw=2.0, connectionstyle="arc3,rad=0.0"))
        ax.text(mid_x, mid_y, coef, ha="center", va="bottom", fontsize=9,
                fontweight="bold", color=color,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor=color, alpha=0.9))

    # Node positions
    nodes = {
        "MOT": (2.0, 3.5),
        "EEM": (1.0, 5.5), "GBI": (2.0, 5.5), "RSA": (3.0, 5.5),
        "PCB": (2.0, 1.5),
        "PVI": (6.5, 5.0),
        "TWI": (6.5, 2.0),
    }

    # Draw MOT components
    draw_box(ax, 1.0, 5.5, "EEM\n情感体验动机", MORANDI["blue"], width=1.7)
    draw_box(ax, 2.0, 5.5, "GBI\n群体归属认同", MORANDI["green"], width=1.7)
    draw_box(ax, 3.0, 5.5, "RSA\n仪式感自我实现", MORANDI["mauve"], width=1.9)

    draw_box(ax, 2.0, 3.5, "MOT\n内在动机综合体", TECH_BLUE, width=2.2, height=0.85, fs=10)
    draw_box(ax, 2.0, 1.5, "PCB\n感知成本障碍", ACCENT_RED, width=2.2, height=0.85, fs=10)
    draw_box(ax, 6.5, 5.0, "PVI\n观演意愿\nR²=0.925", MORANDI["sky"], width=2.2, height=0.9)
    draw_box(ax, 6.5, 2.0, "TWI\n旅游消费延伸意愿\nR²=0.942", MORANDI["sage"], width=2.5, height=0.9)

    # Arrows: components to MOT
    for cx, cy in [(1.0, 5.2), (2.0, 5.2), (3.0, 5.2)]:
        ax.annotate("", xy=(2.0, 3.95), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle="-|>", color=MID_GRAY, lw=1.3))

    # Main structural paths
    # MOT -> PVI
    draw_arrow(ax, 3.1, 3.7, 5.3, 4.8, "β=0.743***", TECH_BLUE)
    # MOT -> TWI
    draw_arrow(ax, 3.1, 3.3, 5.3, 2.2, "β=0.506***", TECH_BLUE)
    # PCB -> PVI
    draw_arrow(ax, 3.1, 1.7, 5.3, 4.6, "β=−0.197***", ACCENT_RED)
    # PCB -> TWI
    draw_arrow(ax, 3.1, 1.5, 5.2, 1.7, "β=−0.459***", ACCENT_RED)

    # R2 annotations
    ax.text(8.3, 5.0, "R²=0.925", ha="center", va="center", fontsize=9,
            color=TECH_BLUE, fontstyle="italic")
    ax.text(8.3, 2.0, "R²=0.942", ha="center", va="center", fontsize=9,
            color=TECH_BLUE, fontstyle="italic")

    # Title and note
    ax.set_title(
        "图10  模型二：动机-阻碍SEM路径图（N=712）\n"
        "标准化路径系数，*** p<.001；CFI=0.859，RMSEA=0.158，AIC=81.3",
        fontsize=11, fontweight="bold", pad=15, color=DARK_GRAY,
        y=0.98
    )

    # Legend
    legend_elements = [
        mpatches.Patch(color=TECH_BLUE, label="正向路径（内在动机驱动）"),
        mpatches.Patch(color=ACCENT_RED, label="负向路径（感知成本阻碍）"),
    ]
    ax.legend(handles=legend_elements, loc="lower right",
              fontsize=9, framealpha=0.95)

    plt.tight_layout()
    plt.savefig("charts/fig10_sem_path_upgraded.png", dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("  ✓ 升级版SEM路径图已保存至 charts/fig10_sem_path_upgraded.png")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    os.makedirs("charts", exist_ok=True)
    set_global_style()

    print("生成可视化图表（正大杯国奖级别）...")
    print()

    print("1. 中国PPS抽样地图...")
    create_china_sampling_map()

    print("2. 升级版雷达图...")
    create_upgraded_radar()

    print("3. 升级版小提琴图...")
    create_upgraded_violin()

    print("4. 升级版SEM路径图...")
    create_sem_path_chart()

    print()
    print("=" * 60)
    print("所有图表已生成！保存位置：charts/")
    print("  - fig_sampling_map.png   （PPS抽样地图）")
    print("  - fig5_radar_upgraded.png（升级版雷达图）")
    print("  - fig8_violin_upgraded.png（升级版小提琴图）")
    print("  - fig10_sem_path_upgraded.png（SEM路径图）")
    print("=" * 60)
    print()
    print("【全局样式说明】")
    print("  色系：莫兰迪低饱和度色板 + 科技蓝强调色")
    print("  字体：SimHei（中文）+ DejaVu Sans（英文数字）")
    print("  分辨率：200 DPI（屏幕）/ 300 DPI可调（印刷）")
    print("  适用：Word文档插图、PPT报告、期刊投稿")
