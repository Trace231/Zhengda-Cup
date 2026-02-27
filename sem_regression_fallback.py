"""
当 semopy 未安装时，用标准化 OLS 回归近似两个 SEM 的结构路径，
得到路径系数、显著性及 R²，用于结论汇总。
运行: python sem_regression_fallback.py
"""

import numpy as np
import pandas as pd
from survey_data_loader import load_data, get_score_df

def _standardize(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std() if s.std() > 0 else s - s.mean()

def _pvalue_from_t(t: float, df: int) -> float:
    """用标准正态近似 t 分布求双尾 p 值（df 较大时近似良好）。"""
    import math
    if np.isnan(t) or df < 1:
        return np.nan
    x = abs(t)
    # 标准正态 P(Z > x) = 0.5 * erfc(x/sqrt(2))
    p = math.erfc(x / math.sqrt(2))
    return min(1.0, 2 * p)


def ols_path(y_name: str, x_names: list, score: pd.DataFrame) -> dict:
    """单方程 OLS，因变量 y，自变量 x_names，返回标准化系数、p值、R²。"""
    y = score[y_name].dropna()
    X = score[x_names].reindex(y.index).dropna(how='any')
    valid = y.index.intersection(X.index)
    if len(valid) < 10:
        return {'beta': {}, 'p': {}, 'R2': np.nan}
    y = _standardize(y.loc[valid])
    X = X.loc[valid].apply(_standardize)
    X['_const'] = 1.0
    n, k = len(X), len(x_names)
    try:
        XtX = X.T @ X
        b = np.linalg.solve(XtX, X.T @ y)
    except np.linalg.LinAlgError:
        return {'beta': {x: np.nan for x in x_names}, 'p': {x: np.nan for x in x_names}, 'R2': np.nan}
    pred = X @ b
    resid = y - pred
    mse = (resid ** 2).sum() / (n - k - 1) if n > k + 1 else np.nan
    try:
        var_b = mse * np.linalg.inv(XtX)
        se = np.sqrt(np.diag(var_b))
    except Exception:
        se = np.full(k + 1, np.nan)
    t = b / np.where(se > 0, se, np.nan)
    df = n - k - 1
    p = np.array([_pvalue_from_t(float(t[i]), df) for i in range(len(x_names))])
    r2 = 1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum() if y.var() > 0 else np.nan
    betas = {x: b[i] for i, x in enumerate(x_names)}
    ps    = {x: p[i] for i, x in enumerate(x_names)}
    return {'beta': betas, 'p': ps, 'R2': r2}

def main():
    _, scale_data = load_data()
    score_df = get_score_df(scale_data)
    score_df = score_df.dropna(how='all')
    # 二阶因子：动机综合
    score_df['MOT'] = score_df[['EEM', 'GBI', 'RSA']].mean(axis=1)

    print('=' * 70)
    print('基于维度得分的路径回归（近似结构方程结构）')
    print('样本量 N =', len(score_df))
    print('=' * 70)

    # ---------- 模型一：情境 → 动机 → 意愿 ----------
    print('\n【模型一】情境 → 动机 → 意愿')
    print('-' * 50)
    paths_m1 = [
        ('EEM', ['SMI', 'PSR', 'CTA'], '情境→情感体验动机'),
        ('GBI', ['SMI', 'PSR'],          '情境→群体归属'),
        ('RSA', ['PSR', 'CTA'],          '情境→仪式感/自我实现'),
        ('PVI', ['EEM', 'GBI', 'RSA'],   '动机→观演意愿'),
        ('TWI', ['EEM', 'GBI', 'RSA', 'CTA', 'PVI'], '动机+情境+观演意愿→旅游消费意愿（参与影响消费）'),
    ]
    r2_m1 = {}
    for y, x_list, label in paths_m1:
        res = ols_path(y, x_list, score_df)
        r2_m1[y] = res['R2']
        print(f'\n  {label}  R² = {res["R2"]:.3f}')
        for x in x_list:
            beta, p = res['beta'].get(x, np.nan), res['p'].get(x, np.nan)
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.'))
            print(f'    {x} → {y}: β = {beta:.3f}, p = {p:.3f} {sig}')

    # ---------- 模型二：MOT + PCB → PVI；MOT + PCB + PVI → TWI（参与影响消费）----------
    print('\n' + '=' * 70)
    print('【模型二】动机(MOT) + 阻碍(PCB) → PVI；MOT + PCB + PVI → TWI（参与→消费）')
    print('-' * 50)
    paths_m2 = [
        ('PVI', ['MOT', 'PCB'], '动机+阻碍→观演意愿'),
        ('TWI', ['MOT', 'PCB', 'PVI'], '动机+阻碍+观演意愿→旅游消费意愿（参与影响消费）'),
    ]
    r2_m2 = {}
    for y, x_list, label in paths_m2:
        res = ols_path(y, x_list, score_df)
        r2_m2[y] = res['R2']
        print(f'\n  {label}  R² = {res["R2"]:.3f}')
        for x in x_list:
            beta, p = res['beta'].get(x, np.nan), res['p'].get(x, np.nan)
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.'))
            print(f'    {x} → {y}: β = {beta:.3f}, p = {p:.3f} {sig}')

    print('\n' + '=' * 70)
    return {'r2_m1': r2_m1, 'r2_m2': r2_m2, 'score_df': score_df}

if __name__ == '__main__':
    main()
