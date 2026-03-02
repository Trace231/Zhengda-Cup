#!/usr/bin/env python3
"""
run_full_pipeline.py
====================
完整数据分析管线（基于 survey_clean.csv）。
依次执行：图1-8 → 图5双组雷达 → SEM → 图9-10 → 重写第五章+第六章到 docx
"""
import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get('SURVEY_DATA', 'survey_clean.csv')

steps = [
    ('generate_charts_v2.py', '图1-8 描述性图表'),
    ('fix_radar.py', '图5 双组雷达图'),
    ('sem_analysis_v2.py', '双路 SEM'),
    ('sem_charts_v2.py', '图9-10 SEM路径图'),
    ('run_write_ch5_ch6.py', '重写第五章+第六章到 docx'),
]

print('='*60)
print(f'数据分析管线 | 数据源: {DATA}')
print('='*60)

for script, desc in steps:
    print(f'\n>> {desc} ({script})')
    r = subprocess.run([sys.executable, script], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr or r.stdout)
        sys.exit(1)
    for line in (r.stdout or '').splitlines():
        if line.strip(): print(f'   {line}')

print('\n' + '='*60)
print('管线完成。报告已更新至 wfm 部分.docx')
print('='*60)
