"""
问卷量表数据加载与信效度计算
用于 reliability_validity.py、generate_charts.py、sem_analysis.py 等。
支持 synthetic_survey_data.csv（合成数据）及同结构的 CSV。
"""

import numpy as np
import pandas as pd
from pathlib import Path

# 默认数据文件（合成问卷 CSV）
DEFAULT_FILE = "/Users/thomaswang/thomas/mks/survey_300.csv"

# 27 个量表题列名（与 data_pipeline 输出一致）
SCALE_COLS = [f"Scale_{i}_{j}" for i in range(1, 10) for j in range(1, 4)]

# 九大构念：题项对应 scale_data 的列下标（0~26）
CONSTRUCTS = {
    "SMI": {"label": "社交媒体信息影响", "indices": [0, 1, 2]},
    "PSR": {"label": "偶像准社会关系", "indices": [3, 4, 5]},
    "CTA": {"label": "城市旅游吸引力", "indices": [6, 7, 8]},
    "EEM": {"label": "情感体验动机", "indices": [9, 10, 11]},
    "GBI": {"label": "群体归属认同", "indices": [12, 13, 14]},
    "RSA": {"label": "仪式感与自我实现", "indices": [15, 16, 17]},
    "PCB": {"label": "感知成本障碍", "indices": [18, 19, 20]},
    "PVI": {"label": "观演意愿", "indices": [21, 22, 23]},
    "TWI": {"label": "旅游/消费延伸意愿", "indices": [24, 25, 26]},
}

# 人口学列在原始 df 中的列名（合成数据）
DEMO_COLS = {
    "gender": "Q11_1_gender",
    "age": "Q11_2_age_range",
    "occupation": "Q11_3_occupation",
    "income": "Q11_4_income",
    "residence": "Q11_5_residence",
}


def load_data(file_path: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    读取问卷 CSV，返回 (原始 df, 仅量表题 df)。
    量表题转为数值，非数值变为 NaN。
    """
    path = Path(file_path or DEFAULT_FILE)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    df = pd.read_csv(path, encoding="utf-8-sig")
    # 只保留存在的量表列
    scale_cols = [c for c in SCALE_COLS if c in df.columns]
    data = df[scale_cols].apply(pd.to_numeric, errors="coerce")
    return df, data


def get_score_df(data: pd.DataFrame) -> pd.DataFrame:
    """各构念维度得分（3 题均值），列名为 SMI, PSR, ..."""
    score = pd.DataFrame(index=data.index)
    for key, info in CONSTRUCTS.items():
        cols = [data.columns[i] for i in info["indices"]]
        score[key] = data[cols].mean(axis=1)
    return score


def _citc(X: pd.DataFrame) -> pd.Series:
    """校正后的项目-总体相关系数：每题与（其余题之和）的相关系数"""
    n = X.shape[1]
    out = []
    for i in range(n):
        item = X.iloc[:, i]
        rest_sum = X.drop(columns=X.columns[i]).sum(axis=1)
        out.append(item.corr(rest_sum))
    return pd.Series(out, index=X.columns)


def _pca_loadings_1f(X: pd.DataFrame) -> np.ndarray:
    """单因子 PCA 第一主成分载荷（与题项的相关系数，作为因子载荷近似）"""
    X_ = X.dropna(how="all")
    if X_.shape[0] < 3:
        return np.ones(X.shape[1]) * 0.5
    cov = X_.cov()
    eigvals, eigvecs = np.linalg.eigh(cov)
    pc = eigvecs[:, -1]
    scores = X_.values @ pc
    loadings = np.array([np.corrcoef(X_.iloc[:, j], scores)[0, 1] for j in range(X_.shape[1])])
    return np.abs(loadings)


def _cronbach_alpha(X: pd.DataFrame) -> float:
    """Cronbach's α"""
    X = X.dropna(how="all")
    k = X.shape[1]
    if k < 2 or X.shape[0] < 2:
        return 0.0
    var_items = X.var(axis=0, skipna=True)
    var_total = X.sum(axis=1).var(skipna=True)
    if var_total == 0:
        return 0.0
    return (k / (k - 1)) * (1 - var_items.sum() / var_total)


def get_rv_results(data: pd.DataFrame) -> dict:
    """
    各构念的信度与收敛效度：alpha, CR, AVE, label。
    CR = (Σλ)² / ((Σλ)² + Σ(1-λ²)), AVE = mean(λ²)。
    """
    result = {}
    for key, info in CONSTRUCTS.items():
        idx = info["indices"]
        X = data.iloc[:, idx].copy()
        X = X.astype(float, copy=False)
        alpha = _cronbach_alpha(X)
        loadings = _pca_loadings_1f(X)
        loadings = np.clip(loadings, 0.01, 0.99)
        ave = float(np.mean(loadings ** 2))
        s = float(np.sum(loadings))
        err = np.sum(1 - loadings ** 2)
        cr = (s ** 2) / (s ** 2 + err) if (s ** 2 + err) > 0 else 0.0
        result[key] = {
            "alpha": alpha,
            "CR": cr,
            "AVE": ave,
            "label": info["label"],
        }
    return result
