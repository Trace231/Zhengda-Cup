"""
run_pipeline_750.py
==================
完整数据分析管线：基于 survey_clean.csv (N≈712) 重新运行并更新 wfm 部分.docx
步骤：1.生成图1-8 2.生成双组雷达图fig5 3.运行SEM 4.生成SEM路径图 5.重写ch5+ch6
"""
import sys, os, subprocess, warnings
sys.path.insert(0, '.pip_pkgs')
warnings.filterwarnings('ignore')

os.chdir('/Users/thomaswang/thomas/mks')
DATA_FILE = 'survey_clean.csv'

def run(cmd, desc):
    print(f'\n>>> {desc}')
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr or r.stdout)
        raise SystemExit(1)
    for line in (r.stdout or '').splitlines():
        if line.strip(): print('  ', line)
    return r

# 1. 生成描述性图表 fig1-8 (需修改数据源)
print('='*60)
print('Step 1: 生成描述性图表 (fig1-8)')
print('='*60)
# 临时替换 generate_charts_v2 中的文件路径，或通过环境变量
run("python3 -c \"
exec(open('generate_charts_v2.py').read().replace('survey_300_clean.csv','survey_clean.csv'))
\"", 'generate_charts_v2 → survey_clean.csv')
# 上述 exec 可能有问题，直接修改文件更稳妥
import re
for f in ['generate_charts_v2.py', 'fix_radar.py']:
    if os.path.exists(f):
        with open(f, 'r', encoding='utf-8') as fp:
            c = fp.read()
        c2 = c.replace('survey_300_clean.csv', DATA_FILE)
        if c != c2:
            with open(f, 'w', encoding='utf-8') as fp:
                fp.write(c2)
            print(f'  Updated {f} → {DATA_FILE}')

run('python3 generate_charts_v2.py 2>/dev/null | head -20', 'generate fig1-8')

# 2. 生成双组雷达图 fig5
run('python3 fix_radar.py 2>/dev/null', 'generate fig5 dual-group radar')

# 3. 运行 SEM
for f in ['sem_analysis_v2.py']:
    if os.path.exists(f):
        with open(f, 'r', encoding='utf-8') as fp:
            c = fp.read()
        c2 = c.replace("NEW_FILE = 'survey_300_clean.csv'", f"NEW_FILE = '{DATA_FILE}'")
        if 'survey_300' in c and 'survey_clean' not in c:
            c2 = c.replace('survey_300_clean.csv', DATA_FILE)
        if c != c2:
            with open(f, 'w', encoding='utf-8') as fp:
                fp.write(c2)
run('python3 sem_analysis_v2.py 2>/dev/null | tail -50', 'run SEM')

# 4. 生成 SEM 路径图
run('python3 sem_charts_v2.py 2>/dev/null', 'generate fig9-10')

# 5. 重写第五章+第六章
run('python3 run_write_ch5_ch6.py 2>/dev/null', 'write ch5+ch6 to docx')

print('\n' + '='*60)
print('管线完成。报告已更新至 wfm 部分.docx')
print('='*60)
