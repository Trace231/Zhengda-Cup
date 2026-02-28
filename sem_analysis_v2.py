"""
sem_analysis_v2.py
==================
双路SEM（基于 survey_300_clean.csv，N=300）。
与 sem_analysis.py 逻辑相同，仅数据源改为新CSV。
"""

import sys, os, warnings
sys.path.insert(0, '.pip_pkgs')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import semopy
from semopy import Model, calc_stats

# ── 数据准备 ──────────────────────────────────────────────────────────────────
NEW_FILE = 'survey_300_clean.csv'

CONSTRUCTS = {
    'SMI': {'label': '社交媒体信息影响', 'cols': ['Scale_1_1','Scale_1_2','Scale_1_3']},
    'PSR': {'label': '偶像准社会关系',   'cols': ['Scale_2_1','Scale_2_2','Scale_2_3']},
    'CTA': {'label': '城市旅游吸引力',   'cols': ['Scale_3_1','Scale_3_2','Scale_3_3']},
    'EEM': {'label': '情感体验动机',     'cols': ['Scale_4_1','Scale_4_2','Scale_4_3']},
    'GBI': {'label': '群体归属感',       'cols': ['Scale_5_1','Scale_5_2','Scale_5_3']},
    'RSA': {'label': '仪式感与自我实现', 'cols': ['Scale_6_1','Scale_6_2','Scale_6_3']},
    'PCB': {'label': '感知成本障碍',     'cols': ['Scale_7_1','Scale_7_2','Scale_7_3']},
    'PVI': {'label': '观演意愿',         'cols': ['Scale_8_1','Scale_8_2','Scale_8_3']},
    'TWI': {'label': '旅游消费意愿',     'cols': ['Scale_9_1','Scale_9_2','Scale_9_3']},
}

def _prepare_sem_data():
    df = pd.read_csv(NEW_FILE)
    all_cols = []
    rename_map = {}
    for key, info in CONSTRUCTS.items():
        for j, col in enumerate(info['cols'], 1):
            new_name = f'{key.lower()}{j}'
            rename_map[col] = new_name
            all_cols.append(col)
    sem_df = df[all_cols].rename(columns=rename_map).apply(pd.to_numeric, errors='coerce')
    return sem_df

# ── 模型定义 ──────────────────────────────────────────────────────────────────
MODEL1_DESC = """
SMI =~ smi1 + smi2 + smi3
PSR =~ psr1 + psr2 + psr3
CTA =~ cta1 + cta2 + cta3
EEM =~ eem1 + eem2 + eem3
GBI =~ gbi1 + gbi2 + gbi3
RSA =~ rsa1 + rsa2 + rsa3
PVI =~ pvi1 + pvi2 + pvi3
TWI =~ twi1 + twi2 + twi3

EEM ~ SMI + PSR + CTA
GBI ~ SMI + PSR
RSA ~ PSR + CTA
PVI ~ EEM + GBI + RSA
TWI ~ EEM + GBI + RSA + CTA
"""

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
]

MODEL2_DESC = """
EEM =~ eem1 + eem2 + eem3
GBI =~ gbi1 + gbi2 + gbi3
RSA =~ rsa1 + rsa2 + rsa3
PCB =~ pcb1 + pcb2 + pcb3
PVI =~ pvi1 + pvi2 + pvi3
TWI =~ twi1 + twi2 + twi3

MOT =~ EEM + GBI + RSA

PVI ~ MOT + PCB
TWI ~ MOT + PCB
"""

MODEL2_HYPOTHESES = [
    ('H1', 'MOT', 'PVI', '内在动机(二阶) → 观演意愿',            '+'),
    ('H2', 'MOT', 'TWI', '内在动机(二阶) → 旅游消费意愿',        '+'),
    ('H3', 'PCB', 'PVI', '感知成本障碍   → 观演意愿',            '-'),
    ('H4', 'PCB', 'TWI', '感知成本障碍   → 旅游消费意愿',        '-'),
    ('H5', 'MOT', 'EEM', '内在动机(二阶) → 情感体验动机(载荷)',  '+'),
    ('H6', 'MOT', 'GBI', '内在动机(二阶) → 群体归属感(载荷)',    '+'),
    ('H7', 'MOT', 'RSA', '内在动机(二阶) → 仪式感/自我实现(载荷)', '+'),
]

# ── 辅助提取 ──────────────────────────────────────────────────────────────────
def _flt(v):
    try: return float(v)
    except: return np.nan

def _get_path_table(inspect_df, hypotheses):
    struct = inspect_df[inspect_df['op'] == '~'].copy()
    rows = []
    for h_id, frm, to, desc, exp_sign in hypotheses:
        match = struct[
            (struct['rval'].str.upper() == frm.upper()) &
            (struct['lval'].str.upper() == to.upper())
        ]
        if len(match):
            r = match.iloc[0]
            beta  = _flt(r.get('Estimate',  np.nan))
            std_b = _flt(r.get('Est. Std',  np.nan))
            se    = _flt(r.get('Std. Err',  np.nan))
            z     = _flt(r.get('z-value',   np.nan))
            pval  = _flt(r.get('p-value',   np.nan))
            sig = ('***' if pval < 0.001 else ('**' if pval < 0.01
                   else ('*' if pval < 0.05 else 'n.s.'))) if not np.isnan(pval) else 'fixed'
            sup = '✓' if (not np.isnan(pval) and pval < 0.05 and
                          ((exp_sign=='+' and beta>0) or (exp_sign=='-' and beta<0))
                         ) else ('fixed' if np.isnan(pval) else '✗')
        else:
            beta=std_b=se=z=pval=np.nan; sig='—'; sup='？'

        rows.append({'假设': h_id, '路径': f'{frm} → {to}', '路径含义': desc,
                     '预期方向': exp_sign,
                     '非标准β': round(beta,3)  if not np.isnan(beta)  else '—',
                     '标准化β': round(std_b,3) if not np.isnan(std_b) else '—',
                     'SE':      round(se,3)    if not np.isnan(se)    else '—',
                     'z值':     round(z,3)     if not np.isnan(z)     else '—',
                     'p值':     f'{pval:.3f}'  if not np.isnan(pval)  else 'fixed',
                     '显著性':  sig, '假设验证': sup})
    return pd.DataFrame(rows)

def _get_loading_table(inspect_df):
    cfa = inspect_df[
        inspect_df['lval'].str.match(r'^[a-z]{2,4}[0-9]$', na=False) &
        inspect_df['rval'].str.match(r'^[A-Z]{2,4}$',       na=False)
    ].copy()
    rows = []
    for _, r in cfa.iterrows():
        loading  = _flt(r.get('Estimate', np.nan))
        std_load = _flt(r.get('Est. Std', np.nan))
        se       = _flt(r.get('Std. Err', np.nan))
        pval     = _flt(r.get('p-value',  np.nan))
        sig = ('***' if pval<0.001 else ('**' if pval<0.01 else ('*' if pval<0.05 else 'n.s.'))) \
              if not np.isnan(pval) else 'fixed'
        rows.append({'构念': r['rval'].upper(), '题项': r['lval'],
                     '因子载荷': round(loading,3)  if not np.isnan(loading)  else '—',
                     '标准化载荷': round(std_load,3) if not np.isnan(std_load) else '—',
                     'SE': round(se,3) if not np.isnan(se) else '—',
                     '显著性': sig})
    return pd.DataFrame(rows)

def _get_fit(stats):
    s = stats.iloc[0]
    def g(k):
        try: return float(s[k])
        except: return np.nan
    return {'χ²': g('chi2'), 'df': g('df'), 'p(χ²)': g('chi2 p-value'),
            'CFI': g('CFI'), 'TLI': g('TLI'),
            'RMSEA': g('RMSEA'), 'SRMR': g('SRMR'),
            'AIC': g('AIC'), 'BIC': g('BIC')}

def _get_r2(model, endogenous):
    r2 = {}
    try:
        inspect = model.inspect(mode='list', what='est')
        for lv in endogenous:
            var_row = inspect[(inspect['op']=='~~') &
                              (inspect['lval'].str.upper()==lv.upper()) &
                              (inspect['rval'].str.upper()==lv.upper())]
            r2[lv] = round(1-float(var_row.iloc[0]['Estimate']),3) if len(var_row) else np.nan
    except Exception:
        for lv in endogenous: r2[lv] = np.nan
    return r2

# ── 主接口 ────────────────────────────────────────────────────────────────────
def run_model1():
    sem_df = _prepare_sem_data()
    mod = Model(MODEL1_DESC)
    mod.fit(sem_df)
    stats   = calc_stats(mod)
    inspect = mod.inspect(mode='list', what='est', std_est=True)
    return {
        'model': mod, 'inspect': inspect,
        'path_table':    _get_path_table(inspect, MODEL1_HYPOTHESES),
        'loading_table': _get_loading_table(inspect),
        'fit_summary':   _get_fit(stats),
        'r2':            _get_r2(mod, ['EEM','GBI','RSA','PVI','TWI']),
        'name': '模型一：动机-情境双轮驱动模型',
        'hypotheses': MODEL1_HYPOTHESES,
    }

def run_model2():
    sem_df = _prepare_sem_data()
    mod = Model(MODEL2_DESC)
    mod.fit(sem_df)
    stats   = calc_stats(mod)
    inspect = mod.inspect(mode='list', what='est', std_est=True)
    return {
        'model': mod, 'inspect': inspect,
        'path_table':    _get_path_table(inspect, MODEL2_HYPOTHESES),
        'loading_table': _get_loading_table(inspect),
        'fit_summary':   _get_fit(stats),
        'r2':            _get_r2(mod, ['MOT','PVI','TWI']),
        'name': '模型二：动机-阻碍SEM模型',
        'hypotheses': MODEL2_HYPOTHESES,
    }

if __name__ == '__main__':
    pd.set_option('display.max_columns',20); pd.set_option('display.width',160)

    print('='*70)
    print('模型一（N=300）')
    r1 = run_model1()
    for k,v in r1['fit_summary'].items():
        print(f'  {k}: {v:.4f}' if isinstance(v,float) else f'  {k}: {v}')
    print('\n路径系数:')
    print(r1['path_table'][['假设','路径','标准化β','p值','显著性','假设验证']].to_string(index=False))
    print('\nR²:', r1['r2'])

    print('\n'+'='*70)
    print('模型二（N=300）')
    r2 = run_model2()
    for k,v in r2['fit_summary'].items():
        print(f'  {k}: {v:.4f}' if isinstance(v,float) else f'  {k}: {v}')
    print('\n路径系数:')
    print(r2['path_table'][['假设','路径','标准化β','p值','显著性','假设验证']].to_string(index=False))
    print('\nR²:', r2['r2'])
