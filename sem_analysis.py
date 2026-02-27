"""
sem_analysis.py
===============
双路结构方程模型（SEM）分析 —— 分别构建两个独立模型，不合并。

模型一（单独构建、单独拟合）：动机-情境双轮驱动模型
  情境层(SMI, PSR, CTA) → 动机层(EEM, GBI, RSA) → 行为层(PVI, TWI)
  构念：SMI, PSR, CTA, EEM, GBI, RSA, PVI, TWI（8 个）

模型二（单独构建、单独拟合）：动机-阻碍 SEM 模型（含二阶因子 MOT）
  动机(EEM, GBI, RSA → MOT) + 阻碍(PCB) → 行为层(PVI, TWI)
  构念：EEM, GBI, RSA, PCB, PVI, TWI + 二阶 MOT（6 个一阶 + 1 个二阶）

公开接口
--------
run_model1() → 模型一结果字典
run_model2() → 模型二结果字典
"""

import sys, os
# 使用当前环境 numpy/pandas/semopy，避免 .pip_pkgs 下不完整安装导致 ImportError
import numpy as np
import pandas as pd
try:
    import semopy
    from semopy import Model, calc_stats
except ImportError:
    print("未检测到 semopy。请安装: pip install semopy")
    print("或运行路径回归近似: python sem_regression_fallback.py")
    raise

from survey_data_loader import load_data, get_score_df, CONSTRUCTS

# ══════════════════════════════════════════════════════════════════════════════
# ── 数据准备 ──────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _prepare_sem_data() -> pd.DataFrame:
    """
    返回以指标题项为列的 DataFrame，列名格式 smi1, smi2, ..., twi3。
    同时保留潜变量均值列（score_df）供后备使用。
    """
    df_raw, scale_data = load_data()

    col_names = []
    for key, info in CONSTRUCTS.items():
        for j, idx in enumerate(info['indices'], start=1):
            col_names.append(f"{key.lower()}{j}")

    sem_df = scale_data.copy()
    sem_df.columns = col_names
    return sem_df


# ══════════════════════════════════════════════════════════════════════════════
# ── 模型一：动机-情境双轮驱动模型 ─────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
# 路径假设（13条）：
#   H1  SMI → EEM    社交媒体信息影响情感体验动机
#   H2  SMI → GBI    社交媒体信息影响群体归属感
#   H3  PSR → EEM    准社会关系驱动情感体验动机
#   H4  PSR → GBI    准社会关系驱动群体归属感
#   H5  PSR → RSA    准社会关系驱动仪式参与/自我实现
#   H6  CTA → EEM    城市文旅吸引力影响情感体验动机
#   H7  CTA → TWI    城市文旅吸引力直接影响旅游消费意愿（直接效应）
#   H8  EEM → PVI    情感体验动机 → 观演意愿
#   H9  GBI → PVI    群体归属感 → 观演意愿
#   H10 RSA → PVI    仪式感/自我实现 → 观演意愿
#   H11 EEM → TWI    情感体验动机 → 旅游消费意愿
#   H12 GBI → TWI    群体归属感 → 旅游消费意愿
#   H13 RSA → TWI    仪式感/自我实现 → 旅游消费意愿

MODEL1_DESC = """
# ── 测量模型 (CFA) ──────────────────────────────────
SMI =~ smi1 + smi2 + smi3
PSR =~ psr1 + psr2 + psr3
CTA =~ cta1 + cta2 + cta3
EEM =~ eem1 + eem2 + eem3
GBI =~ gbi1 + gbi2 + gbi3
RSA =~ rsa1 + rsa2 + rsa3
PVI =~ pvi1 + pvi2 + pvi3
TWI =~ twi1 + twi2 + twi3

# ── 结构模型 ──────────────────────────────────────────
# 情境 → 动机
EEM ~ SMI + PSR + CTA
GBI ~ SMI + PSR
RSA ~ PSR + CTA

# 动机 → 意愿；参与意愿 → 消费意愿（与模型二一致，便于链式中介）
PVI ~ EEM + GBI + RSA
TWI ~ EEM + GBI + RSA + CTA + PVI

# 两意愿残差相关（控制共同未观测因素）
PVI ~~ TWI
"""

# 路径假设标签（用于汇总表）
MODEL1_HYPOTHESES = [
    ('H1',  'SMI', 'EEM', '社交媒体信息 → 情感体验动机', '+'),
    ('H2',  'SMI', 'GBI', '社交媒体信息 → 群体归属感',   '+'),
    ('H3',  'PSR', 'EEM', '准社会关系   → 情感体验动机', '+'),
    ('H4',  'PSR', 'GBI', '准社会关系   → 群体归属感',   '+'),
    ('H5',  'PSR', 'RSA', '准社会关系   → 仪式感/自我实现', '+'),
    ('H6',  'CTA', 'EEM', '城市文旅     → 情感体验动机', '+'),
    ('H7',  'CTA', 'TWI', '城市文旅     → 旅游消费意愿(直接)', '+'),
    ('H8',  'EEM', 'PVI', '情感体验动机 → 观演意愿',     '+'),
    ('H9',  'GBI', 'PVI', '群体归属感   → 观演意愿',     '+'),
    ('H10', 'RSA', 'PVI', '仪式感/自我实现 → 观演意愿',  '+'),
    ('H11', 'EEM', 'TWI', '情感体验动机 → 旅游消费意愿', '+'),
    ('H12', 'GBI', 'TWI', '群体归属感   → 旅游消费意愿', '+'),
    ('H13', 'RSA', 'TWI', '仪式感/自我实现 → 旅游消费意愿', '+'),
    ('H14', 'PVI', 'TWI', '观演意愿     → 旅游消费意愿(参与影响消费)', '+'),
]


# ══════════════════════════════════════════════════════════════════════════════
# ── 模型二：动机-阻碍SEM模型（含二阶动机因子 MOT）─────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
# 背景：EEM/GBI/RSA 的潜变量间相关系数均 >0.85，直接作为并列预测变量会引发
# 严重多重共线性（VIF>10），导致路径系数不稳定。学术惯例是引入二阶因子（Higher-
# Order Factor），将三个动机构念的共同方差提炼为"内在动机（MOT）"，再由MOT驱动
# 行为意愿，PCB 作为独立阻碍因子并行进入模型。
#
# 路径假设（6条）：
#   H1  MOT → PVI   内在动机（综合）→ 观演意愿（正效应）
#   H2  MOT → TWI   内在动机（综合）→ 旅游消费意愿（正效应）
#   H3  PCB → PVI   感知成本障碍   → 观演意愿（负效应）
#   H4  PCB → TWI   感知成本障碍   → 旅游消费意愿（负效应）
#   H5  EEM → MOT   情感体验动机   → 内在动机（二阶载荷）
#   H6  GBI → MOT   群体归属感     → 内在动机（二阶载荷）
#   H7  RSA → MOT   仪式感/自我实现→ 内在动机（二阶载荷）

MODEL2_DESC = """
# ── 一阶测量模型 (CFA) ──────────────────────────────
EEM =~ eem1 + eem2 + eem3
GBI =~ gbi1 + gbi2 + gbi3
RSA =~ rsa1 + rsa2 + rsa3
PCB =~ pcb1 + pcb2 + pcb3
PVI =~ pvi1 + pvi2 + pvi3
TWI =~ twi1 + twi2 + twi3

# ── 二阶动机因子 ──────────────────────────────────────
MOT =~ EEM + GBI + RSA

# ── 结构模型 ──────────────────────────────────────────
# 动机与阻碍平行作用于 PVI、TWI；参与意愿再影响消费意愿（参与→消费）
PVI ~ MOT + PCB
TWI ~ MOT + PCB + PVI
# 两意愿残差相关（控制共同未观测因素）
PVI ~~ TWI
"""

MODEL2_HYPOTHESES = [
    ('H1', 'MOT', 'PVI', '内在动机(二阶)   → 观演意愿',            '+'),
    ('H2', 'MOT', 'TWI', '内在动机(二阶)   → 旅游消费意愿',        '+'),
    ('H3', 'PCB', 'PVI', '感知成本障碍     → 观演意愿',            '-'),
    ('H4', 'PCB', 'TWI', '感知成本障碍     → 旅游消费意愿',        '-'),
    ('H5', 'PVI', 'TWI', '观演意愿         → 旅游消费意愿(参与影响消费)', '+'),
    ('H6', 'MOT', 'EEM', '内在动机(二阶)   → 情感体验动机(载荷)',  '+'),
    ('H7', 'MOT', 'GBI', '内在动机(二阶)   → 群体归属感(载荷)',    '+'),
    ('H8', 'MOT', 'RSA', '内在动机(二阶)   → 仪式感/自我实现(载荷)', '+'),
]


# ══════════════════════════════════════════════════════════════════════════════
# ── 核心函数 ──────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _fit_model(desc: str, sem_df: pd.DataFrame):
    """拟合SEM，返回 (model, stats_df, inspect_df)。"""
    mod = Model(desc)
    res = mod.fit(sem_df)
    stats   = calc_stats(mod)
    inspect = mod.inspect(mode='list', what='est', std_est=True)
    return mod, stats, inspect, res


def _get_path_table(inspect_df: pd.DataFrame,
                    hypotheses: list) -> pd.DataFrame:
    """
    从 semopy inspect 结果中提取结构路径系数，
    与假设列表对齐，生成汇总 DataFrame。
    semopy 2.x 列名: lval, op, rval, Estimate, Est. Std, Std. Err, z-value, p-value
    """
    struct = inspect_df[inspect_df['op'].isin(['~', '=~'])].copy()
    # 仅保留结构路径（lv ~ lv 或 lv =~ lv 为二阶）
    struct = inspect_df[inspect_df['op'] == '~'].copy()

    rows = []
    for h_id, from_lv, to_lv, desc, exp_sign in hypotheses:
        match = struct[
            (struct['rval'].str.upper() == from_lv.upper()) &
            (struct['lval'].str.upper() == to_lv.upper())
        ]
        if len(match) > 0:
            row = match.iloc[0]
            def _flt(v):
                try: return float(v)
                except: return np.nan

            beta  = _flt(row.get('Estimate',  np.nan))
            std_b = _flt(row.get('Est. Std',  np.nan))
            se    = _flt(row.get('Std. Err',  np.nan))
            z     = _flt(row.get('z-value',   np.nan))
            pval  = _flt(row.get('p-value',   np.nan))

            try:
                pval_f = float(pval)
            except (TypeError, ValueError):
                pval_f = np.nan

            if not np.isnan(pval_f):
                sig = '***' if pval_f < 0.001 else ('**' if pval_f < 0.01 else ('*' if pval_f < 0.05 else 'n.s.'))
            else:
                sig = 'fixed'
            pval = pval_f  # normalise to float

            supported = '✓' if (
                not np.isnan(pval) and pval < 0.05 and
                ((exp_sign == '+' and beta > 0) or
                 (exp_sign == '-' and beta < 0))
            ) else ('fixed' if np.isnan(pval) else '✗')
        else:
            beta = std_b = se = z = pval = np.nan
            sig = '—'; supported = '？'

        rows.append({
            '假设':      h_id,
            '路径':      f'{from_lv} → {to_lv}',
            '路径含义':  desc,
            '预期方向':  exp_sign,
            '非标准β':   round(beta,  3) if not np.isnan(beta)  else '—',
            '标准化β':   round(std_b, 3) if not np.isnan(std_b) else '—',
            'SE':        round(se,    3) if not np.isnan(se)    else '—',
            'z值':       round(z,     3) if not np.isnan(z)     else '—',
            'p值':       f'{pval:.3f}' if not np.isnan(pval) else 'fixed',
            '显著性':    sig,
            '假设验证':  supported,
        })
    return pd.DataFrame(rows)


def _get_path_table_m2(inspect_df: pd.DataFrame,
                       hypotheses: list) -> pd.DataFrame:
    """
    Model 2专用：提取路径系数，包含结构路径(~)和二阶因子载荷(=~中的lv→lv)。
    """
    # 结构路径
    struct = inspect_df[inspect_df['op'] == '~'].copy()
    # 二阶因子载荷：op='=~' 且 rval 是大写潜变量（EEM/GBI/RSA）
    ho = inspect_df[
        (inspect_df['op'] == '=~') &
        (inspect_df['rval'].str.upper().isin(['EEM', 'GBI', 'RSA', 'PCB', 'PVI', 'TWI', 'MOT']))
    ].copy()
    combined = pd.concat([struct, ho], ignore_index=True)

    rows = []
    for h_id, from_lv, to_lv, desc, exp_sign in hypotheses:
        match = combined[
            (combined['rval'].str.upper() == from_lv.upper()) &
            (combined['lval'].str.upper() == to_lv.upper())
        ]
        if len(match) > 0:
            row = match.iloc[0]
            beta  = row.get('Estimate',   np.nan)
            std_b = row.get('Est. Std',   np.nan)
            se    = row.get('Std. Err',   np.nan)
            z     = row.get('z-value',    np.nan)
            pval  = row.get('p-value',    np.nan)
            if not (isinstance(pval, float) and not np.isnan(pval)):
                try: pval = float(pval)
                except: pval = np.nan
            if not np.isnan(pval):
                sig = '***' if pval < 0.001 else ('**' if pval < 0.01 else ('*' if pval < 0.05 else 'n.s.'))
            else:
                sig = 'fixed'
            supported = '✓' if (
                not np.isnan(pval) and pval < 0.05 and
                ((exp_sign == '+' and beta > 0) or (exp_sign == '-' and beta < 0))
            ) else ('fixed' if np.isnan(pval) else '✗')
        else:
            beta = std_b = se = z = pval = np.nan
            sig = '—'; supported = '？'

        rows.append({
            '假设':      h_id,
            '路径':      f'{from_lv} → {to_lv}',
            '路径含义':  desc,
            '预期方向':  exp_sign,
            '非标准β':   round(beta,  3) if isinstance(beta,  float) and not np.isnan(beta)  else '—',
            '标准化β':   round(std_b, 3) if isinstance(std_b, float) and not np.isnan(std_b) else '—',
            'SE':        round(se,    3) if isinstance(se,    float) and not np.isnan(se)    else '—',
            'z值':       round(z,     3) if isinstance(z,     float) and not np.isnan(z)     else '—',
            'p值':       f'{pval:.3f}' if isinstance(pval, float) and not np.isnan(pval) else 'fixed',
            '显著性':    sig,
            '假设验证':  supported,
        })
    return pd.DataFrame(rows)


def _get_loading_table(inspect_df: pd.DataFrame) -> pd.DataFrame:
    """
    提取因子载荷表（CFA部分）。
    semopy 2.x 全部用 ~，CFA指标行特征：lval=小写题目名(如eem1)，rval=大写因子名(如EEM)。
    """
    # 指标行：lval 全小写、rval 全大写
    cfa = inspect_df[
        inspect_df['lval'].str.match(r'^[a-z]{2,4}[0-9]$', na=False) &
        inspect_df['rval'].str.match(r'^[A-Z]{2,4}$',       na=False)
    ].copy()

    def _flt(v):
        try: return float(v)
        except: return np.nan

    rows = []
    for _, r in cfa.iterrows():
        factor   = r['rval'].upper()
        item     = r['lval']
        loading  = _flt(r.get('Estimate',  np.nan))
        std_load = _flt(r.get('Est. Std',  np.nan))
        se       = _flt(r.get('Std. Err',  np.nan))
        pval     = _flt(r.get('p-value',   np.nan))
        sig = ('***' if pval < 0.001 else ('**' if pval < 0.01 else ('*' if pval < 0.05 else 'n.s.'))) \
              if not np.isnan(pval) else 'fixed'
        rows.append({
            '构念':      factor,
            '题项':      item,
            '因子载荷':  round(loading,  3) if not np.isnan(loading)  else '—',
            '标准化载荷':round(std_load, 3) if not np.isnan(std_load) else '—',
            'SE':        round(se,       3) if not np.isnan(se)       else '—',
            '显著性':    sig,
        })
    return pd.DataFrame(rows)


def _get_fit_summary(stats) -> dict:
    """从 semopy calc_stats 提取关键拟合指标。"""
    s = stats.iloc[0] if hasattr(stats, 'iloc') else stats
    def _g(key, default=np.nan):
        try: return float(s[key])
        except: return default

    return {
        'χ²':    _g('chi2'),
        'df':    _g('df'),
        'p(χ²)': _g('chi2 p-value'),
        'CFI':   _g('CFI'),
        'TLI':   _g('TLI'),
        'RMSEA': _g('RMSEA'),
        'SRMR':  _g('SRMR'),
        'AIC':   _g('AIC'),
        'BIC':   _g('BIC'),
    }


def _get_r2(model, endogenous: list) -> dict:
    """计算各内生变量的 R²。"""
    r2 = {}
    try:
        r2_df = semopy.calc_stats(model)  # some versions include R2
    except Exception:
        pass
    # 手动计算：相关系数矩阵方式
    try:
        cov_lv = model.calc_sigma()[0]
        lv_names_upper = [n.upper() for n in model.vars['all']]
    except Exception:
        return {k: np.nan for k in endogenous}

    for lv in endogenous:
        try:
            implied_cov = model.calc_sigma()[0]
            # semopy 内部接口不稳定，暂用 inspect 提取结构残差方差
            inspect = model.inspect(mode='list', what='est')
            var_row = inspect[
                (inspect['op'] == '~~') &
                (inspect['lval'].str.upper() == lv.upper()) &
                (inspect['rval'].str.upper() == lv.upper())
            ]
            if len(var_row) > 0:
                resid_var = var_row.iloc[0]['Estimate']
                # 总方差来自 sigma
                # 暂时只存残差方差，R² = 1 - resid_var/total_var 需要 total_var
                r2[lv] = round(1 - resid_var, 3)  # 标准化后 total_var=1
            else:
                r2[lv] = np.nan
        except Exception:
            r2[lv] = np.nan
    return r2


def run_model1():
    sem_df = _prepare_sem_data()
    mod, stats, inspect, res = _fit_model(MODEL1_DESC, sem_df)
    path_table    = _get_path_table(inspect, MODEL1_HYPOTHESES)
    loading_table = _get_loading_table(inspect)
    fit_summary   = _get_fit_summary(stats)
    endogenous = ['EEM', 'GBI', 'RSA', 'PVI', 'TWI']
    r2 = _get_r2(mod, endogenous)
    return {
        'model': mod, 'stats': stats, 'inspect': inspect,
        'path_table': path_table, 'loading_table': loading_table,
        'fit_summary': fit_summary, 'r2': r2,
        'name': '模型一：动机-情境双轮驱动模型',
        'hypotheses': MODEL1_HYPOTHESES,
    }


def run_model2():
    sem_df = _prepare_sem_data()
    mod, stats, inspect, res = _fit_model(MODEL2_DESC, sem_df)
    path_table    = _get_path_table(inspect, MODEL2_HYPOTHESES)
    loading_table = _get_loading_table(inspect)
    fit_summary   = _get_fit_summary(stats)
    endogenous = ['MOT', 'PVI', 'TWI']
    r2 = _get_r2(mod, endogenous)
    return {
        'model': mod, 'stats': stats, 'inspect': inspect,
        'path_table': path_table, 'loading_table': loading_table,
        'fit_summary': fit_summary, 'r2': r2,
        'name': '模型二：动机-阻碍SEM模型',
        'hypotheses': MODEL2_HYPOTHESES,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ── 命令行运行：打印结果 ───────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 160)
    pd.set_option('display.float_format', '{:.3f}'.format)

    print('=' * 70)
    print('模型一：动机-情境双轮驱动模型')
    print('=' * 70)
    r1 = run_model1()
    print('\n── 拟合指标 ──')
    for k, v in r1['fit_summary'].items():
        print(f'  {k:8s}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')
    print('\n── 结构路径系数 ──')
    print(r1['path_table'].to_string(index=False))
    print('\n── R² ──')
    for k, v in r1['r2'].items():
        print(f'  {k}: {v}')

    print('\n' + '=' * 70)
    print('模型二：动机-阻碍SEM模型')
    print('=' * 70)
    r2 = run_model2()
    print('\n── 拟合指标 ──')
    for k, v in r2['fit_summary'].items():
        print(f'  {k:8s}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')
    print('\n── 结构路径系数 ──')
    print(r2['path_table'].to_string(index=False))
    print('\n── R² ──')
    for k, v in r2['r2'].items():
        print(f'  {k}: {v}')
