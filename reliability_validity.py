"""
量表题信效度检验 —— 针对两个结构方程模型分别验证
- 信度：Cronbach's Alpha + CITC（校正后的项目-总体相关系数）
- 效度（收敛效度）：标准化因子载荷、AVE、CR         ← 两模型共用
- 效度（区分效度）：Fornell-Larcker 准则             ← 两模型分别跑

方案一（动机-情境双轮驱动，8个构念）：SMI PSR CTA EEM GBI RSA PVI TWI
方案二（动机-阻碍经典SEM，6个构念）：EEM GBI RSA PCB PVI TWI

数据加载、Likert 映射、构念定义由 survey_data_loader.py 管理（支持 synthetic_survey_data.csv）。
"""

from survey_data_loader import (
    load_data, get_score_df, get_rv_results,
    CONSTRUCTS, DEFAULT_FILE,
)
import numpy as np
import pandas as pd

# ── 1-3. 数据加载（委托给 data_pipeline）────────────────────────────────────
df, data = load_data()
score_df     = get_score_df(data)
result_dict  = get_rv_results(data)
ALL_CONSTRUCTS = CONSTRUCTS

# 两个模型各自包含的构念
MODEL1_KEYS = ['SMI', 'PSR', 'CTA', 'EEM', 'GBI', 'RSA', 'PVI', 'TWI']
MODEL2_KEYS = ['EEM', 'GBI', 'RSA', 'PCB', 'PVI', 'TWI']

# ── 4. 局部辅助函数（仅此文件使用）──────────────────────────────────────────

def print_discriminant(keys, result_dict, score_df, title):
    """打印 Fornell-Larcker 区分效度矩阵，并返回是否存在违反的列表"""
    corr_mat = score_df[keys].corr()
    col_width = 9
    print(f"\n{title}")
    print("-" * (6 + col_width * len(keys)))
    print(f"{'':>6}" + "".join(f"{k:>{col_width}}" for k in keys))

    violations = []
    for i, ki in enumerate(keys):
        ave_i = result_dict[ki]['AVE']
        sqrt_ave_i = np.sqrt(ave_i)
        row_str = f"{ki:>6}"
        for j, kj in enumerate(keys):
            if i == j:
                row_str += f"[{sqrt_ave_i:.3f}]".rjust(col_width)
            elif i > j:
                r = corr_mat.loc[ki, kj]
                row_str += f"{r:.3f}".rjust(col_width)
                # 检查是否违反 Fornell-Larcker（对角线值小于相关系数）
                sqrt_ave_j = np.sqrt(result_dict[kj]['AVE'])
                if r > sqrt_ave_i or r > sqrt_ave_j:
                    violations.append((ki, kj, r, sqrt_ave_i, sqrt_ave_j))
            else:
                row_str += " " * col_width
        print(row_str)
    print("-" * (6 + col_width * len(keys)))
    print("判断标准：对角线 √AVE 应大于同行/同列的所有相关系数（Fornell-Larcker准则）")
    return violations, corr_mat

def discriminant_to_df(keys, result_dict, corr_mat):
    rows = []
    for i, ki in enumerate(keys):
        row_d = {'构念': ki}
        ave_i = result_dict[ki]['AVE']
        for j, kj in enumerate(keys):
            if i == j:
                row_d[kj] = f"[{np.sqrt(ave_i):.3f}]"
            elif i > j:
                row_d[kj] = round(corr_mat.loc[ki, kj], 3)
            else:
                row_d[kj] = ''
        rows.append(row_d)
    return pd.DataFrame(rows)

# ── 5. 构建 result_rows 供打印（数据已由 data_pipeline 计算完毕）────────────
result_rows = []
from survey_data_loader import _citc, _pca_loadings_1f
for key, info in ALL_CONSTRUCTS.items():
    X      = data.iloc[:, info['indices']]
    citc_v = _citc(X)
    loads  = _pca_loadings_1f(X)
    rd     = result_dict[key]
    row    = {'构念': f"{key}（{rd['label']}）",
              'α': round(rd['alpha'], 3),
              'CR': round(rd['CR'], 3),
              'AVE': round(rd['AVE'], 3)}
    for i, (cv, lv) in enumerate(zip(citc_v.values, loads), 1):
        row[f'CITC_{i}'] = round(cv, 3)
        row[f'载荷_{i}'] = round(lv, 3)
    result_rows.append(row)

result_df = pd.DataFrame(result_rows)

# ── 6. 输出 ───────────────────────────────────────────────────────────────────
SEP90 = "=" * 90
SEP60 = "-" * 60

N_sample = len(data)
print(SEP90)
print(f"{'量表信效度检验报告':^88}")
print(("样本量 N = %d，量表题 27 题，9 个潜变量（每变量 3 题）" % N_sample).center(88))
print(SEP90)

# 【表1】收敛效度（两模型共用）
print("\n【表1】信度与收敛效度汇总（两个模型共用）")
print(SEP60)
print(f"{'构念':<22} {'α':>6} {'CR':>6} {'AVE':>6}  {'CITC1':>6} {'CITC2':>6} {'CITC3':>6}  {'载荷1':>6} {'载荷2':>6} {'载荷3':>6}")
print(SEP60)
for _, r in result_df.iterrows():
    print(f"{r['构念'][:22]:<22} {r['α']:>6.3f} {r['CR']:>6.3f} {r['AVE']:>6.3f}  "
          f"{r['CITC_1']:>6.3f} {r['CITC_2']:>6.3f} {r['CITC_3']:>6.3f}  "
          f"{r['载荷_1']:>6.3f} {r['载荷_2']:>6.3f} {r['载荷_3']:>6.3f}")
print(SEP60)
print("判断标准：α > 0.7  |  CITC > 0.4  |  CR > 0.7  |  AVE > 0.5  |  因子载荷 > 0.5")

# 【表2】方案一区分效度
v1, cm1 = print_discriminant(
    MODEL1_KEYS, result_dict, score_df,
    "【表2】方案一区分效度（SMI PSR CTA EEM GBI RSA PVI TWI，8个构念）"
)

# 【表3】方案二区分效度
v2, cm2 = print_discriminant(
    MODEL2_KEYS, result_dict, score_df,
    "【表3】方案二区分效度（EEM GBI RSA PCB PVI TWI，6个构念）"
)

# ── 7. 评估摘要 ───────────────────────────────────────────────────────────────
print(f"\n\n{'【评估摘要】':}")
print(SEP60)

issues = []
for _, r in result_df.iterrows():
    name = r['构念']
    if r['α'] < 0.7:
        issues.append(f"  ⚠ {name}: α={r['α']:.3f} < 0.7")
    if r['CR'] < 0.7:
        issues.append(f"  ⚠ {name}: CR={r['CR']:.3f} < 0.7")
    if r['AVE'] < 0.5:
        issues.append(f"  ⚠ {name}: AVE={r['AVE']:.3f} < 0.5")
    for ci in [1, 2, 3]:
        if r[f'CITC_{ci}'] < 0.4:
            issues.append(f"  ⚠ {name}: 题目{ci} CITC={r[f'CITC_{ci}']:.3f} < 0.4")
    for li in [1, 2, 3]:
        if r[f'载荷_{li}'] < 0.5:
            issues.append(f"  ⚠ {name}: 题目{li} 载荷={r[f'载荷_{li}']:.3f} < 0.5")

print("▶ 收敛效度：" + ("✓ 全部达标" if not issues else ""))
for issue in issues:
    print(issue)

print(f"\n▶ 方案一区分效度（Fornell-Larcker）：", end="")
if not v1:
    print("✓ 全部通过")
else:
    print(f"⚠ 存在 {len(v1)} 处违反")
    for ki, kj, r, sai, saj in v1:
        print(f"     {ki}(√AVE={sai:.3f}) — {kj}(√AVE={saj:.3f})  相关系数={r:.3f}")

print(f"\n▶ 方案二区分效度（Fornell-Larcker）：", end="")
if not v2:
    print("✓ 全部通过")
else:
    print(f"⚠ 存在 {len(v2)} 处违反")
    for ki, kj, r, sai, saj in v2:
        print(f"     {ki}(√AVE={sai:.3f}) — {kj}(√AVE={saj:.3f})  相关系数={r:.3f}")

print("\n" + SEP90)
print("分析完毕")
print(SEP90)

# ── 8. 导出结果 ─────────────────────────────────────────────────────────────
try:
    with pd.ExcelWriter('信效度检验结果.xlsx', engine='openpyxl') as writer:
        result_df.to_excel(writer, sheet_name='表1_信度与收敛效度', index=False)
        discriminant_to_df(MODEL1_KEYS, result_dict, cm1).to_excel(
            writer, sheet_name='表2_方案一区分效度', index=False)
        discriminant_to_df(MODEL2_KEYS, result_dict, cm2).to_excel(
            writer, sheet_name='表3_方案二区分效度', index=False)
    print("\n结果已导出至：信效度检验结果.xlsx（3个 sheet）")
except ImportError:
    result_df.to_csv('信效度检验_表1_信度与收敛效度.csv', index=False, encoding='utf-8-sig')
    discriminant_to_df(MODEL1_KEYS, result_dict, cm1).to_csv(
        '信效度检验_表2_方案一区分效度.csv', index=False, encoding='utf-8-sig')
    discriminant_to_df(MODEL2_KEYS, result_dict, cm2).to_csv(
        '信效度检验_表3_方案二区分效度.csv', index=False, encoding='utf-8-sig')
    print("\n未安装 openpyxl，结果已导出至 CSV：信效度检验_表1/2/3_*.csv")
