"""
run_all.py — 一键更新数据 + 生成报告
用法：python run_all.py
可选参数：
  --no-report   只拉数据，不生成报告
  --report-only 只生成报告，不拉数据
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).parent
PY  = sys.executable

# ── ANSI 颜色 ──────────────────────────────────────────────────────────────
GREEN  = '\033[92m'
YELLOW = '\033[93m'
RED    = '\033[91m'
CYAN   = '\033[96m'
BOLD   = '\033[1m'
RESET  = '\033[0m'

def step(msg):  print(f'\n{CYAN}{BOLD}▶ {msg}{RESET}')
def ok(msg):    print(f'{GREEN}✅ {msg}{RESET}')
def warn(msg):  print(f'{YELLOW}⚠️  {msg}{RESET}')
def fail(msg):  print(f'{RED}❌ {msg}{RESET}')

def run(script: str, label: str) -> bool:
    step(label)
    t0 = time.time()
    result = subprocess.run([PY, script], cwd=BASE, capture_output=False)
    elapsed = time.time() - t0
    if result.returncode == 0:
        ok(f'{label} 完成（{elapsed:.1f}s）')
        return True
    else:
        fail(f'{label} 失败（exit {result.returncode}）')
        return False


def main():
    parser = argparse.ArgumentParser(description='一键更新数据 + 生成报告')
    parser.add_argument('--no-report',   action='store_true', help='只拉数据')
    parser.add_argument('--report-only', action='store_true', help='只生成报告')
    args = parser.parse_args()

    print(f'\n{BOLD}{"═"*48}')
    print('  特码统计分析系统 · 一键更新')
    print(f'{"═"*48}{RESET}')

    success = True

    # ── Step 1：拉取数据 ────────────────────────────────────────────────────
    if not args.report_only:
        ok_data = run('main.py', '拉取 / 更新数据')
        if not ok_data:
            warn('数据拉取失败，尝试用现有数据继续生成报告...')
            success = False

    # ── Step 2：生成报告 ────────────────────────────────────────────────────
    if not args.no_report:
        ok_report = run('report.py', '生成分析报告')
        if not ok_report:
            success = False

    # ── Step 3：发布博客 ────────────────────────────────────────────────────
    if not args.no_report:
        ok_pub = run('publish.py', '发布到博客')
        if not ok_pub:
            success = False

    # ── 输出报告路径 ────────────────────────────────────────────────────────
    if not args.no_report:
        reports_dir = BASE / 'output' / 'reports'
        print(f'\n{BOLD}报告输出：{RESET}')
        for fname in ['analysis_report.html', 'analysis_report.pdf', 'README.md']:
            p = reports_dir / fname
            if p.exists():
                size_kb = p.stat().st_size / 1024
                print(f'  {GREEN}●{RESET} {p}  ({size_kb:.0f} KB)')
            else:
                print(f'  {YELLOW}○{RESET} {p}  (未生成)')

    # ── 完成 ────────────────────────────────────────────────────────────────
    print(f'\n{BOLD}{"═"*48}{RESET}')
    if success:
        ok('全部完成')
    else:
        warn('部分步骤失败，请检查上方日志')
    print()


if __name__ == '__main__':
    main()
