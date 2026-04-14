"""
publish.py — 把最新报告发布到博客目录
用法：python publish.py
流程：
  1. 复制 output/reports/analysis_report.html → blog/posts/YYYY-MM-DD.html
  2. 重新生成 blog/index.html（含所有历史报告列表）
"""

import os
import shutil
import glob
import datetime
import subprocess
import sys
from pathlib import Path

BASE        = Path(__file__).parent
CF_TOKEN    = 'cfut_6Vfpc9ftmvb5SzfU7jXHWH3DDXQALnBrNqV21FTGaacb70fb'
WRANGLER    = str(Path.home() / '.npm-global' / 'bin' / 'wrangler')
REPORT_SRC  = BASE / 'output' / 'reports' / 'analysis_report.html'
BLOG_DIR    = BASE / 'blog'
POSTS_DIR   = BLOG_DIR / 'posts'

def publish():
    POSTS_DIR.mkdir(parents=True, exist_ok=True)

    if not REPORT_SRC.exists():
        print('❌ 未找到报告文件，请先运行 report.py')
        return False

    today = datetime.date.today().strftime('%Y-%m-%d')
    dest  = POSTS_DIR / f'{today}.html'
    shutil.copy2(REPORT_SRC, dest)
    print(f'✅ 报告已发布：{dest}')

    _build_index()
    return True


def _build_index():
    posts = sorted(
        [p.stem for p in POSTS_DIR.glob('????-??-??.html')],
        reverse=True
    )

    rows = ''
    for date_str in posts:
        try:
            d = datetime.date.fromisoformat(date_str)
            label = d.strftime('%Y年%m月%d日')
        except ValueError:
            label = date_str
        rows += f'''
        <tr>
          <td>{label}</td>
          <td><a href="posts/{date_str}.html" target="_blank">📊 查看报告</a></td>
        </tr>'''

    latest_href = f'posts/{posts[0]}.html' if posts else '#'
    latest_date = posts[0] if posts else '暂无'

    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>特码统计分析 · 历史报告</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, 'PingFang SC', 'Helvetica Neue', sans-serif;
      background: #F4F6F9;
      color: #2C3E50;
      min-height: 100vh;
    }}
    header {{
      background: linear-gradient(135deg, #2980B9, #1A5276);
      color: white;
      padding: 40px 24px 32px;
      text-align: center;
    }}
    header h1 {{ font-size: 28px; font-weight: 700; margin-bottom: 8px; }}
    header p  {{ font-size: 14px; opacity: 0.8; }}
    .warn {{
      background: #FEF9E7;
      border-left: 4px solid #F39C12;
      padding: 10px 20px;
      font-size: 13px;
      color: #7D6608;
      max-width: 800px;
      margin: 20px auto 0;
      border-radius: 0 6px 6px 0;
    }}
    .container {{
      max-width: 800px;
      margin: 32px auto;
      padding: 0 16px;
    }}
    .latest-card {{
      background: white;
      border-radius: 12px;
      padding: 24px;
      box-shadow: 0 2px 12px rgba(0,0,0,.08);
      margin-bottom: 28px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 12px;
    }}
    .latest-card h2 {{ font-size: 18px; color: #2980B9; }}
    .latest-card p  {{ font-size: 13px; color: #7F8C8D; margin-top: 4px; }}
    .btn {{
      display: inline-block;
      background: #2980B9;
      color: white;
      padding: 10px 22px;
      border-radius: 8px;
      text-decoration: none;
      font-size: 14px;
      font-weight: 600;
      transition: background .2s;
    }}
    .btn:hover {{ background: #1A5276; }}
    .history h3 {{
      font-size: 16px;
      font-weight: 600;
      margin-bottom: 12px;
      color: #566573;
    }}
    table {{
      width: 100%;
      background: white;
      border-radius: 10px;
      box-shadow: 0 2px 8px rgba(0,0,0,.06);
      border-collapse: collapse;
      overflow: hidden;
    }}
    th {{
      background: #EAF2F8;
      padding: 12px 16px;
      text-align: left;
      font-size: 13px;
      color: #566573;
      font-weight: 600;
    }}
    td {{
      padding: 11px 16px;
      font-size: 14px;
      border-top: 1px solid #F2F3F4;
    }}
    td a {{
      color: #2980B9;
      text-decoration: none;
      font-weight: 500;
    }}
    td a:hover {{ text-decoration: underline; }}
    tr:hover td {{ background: #FAFAFA; }}
    footer {{
      text-align: center;
      padding: 32px 16px;
      font-size: 12px;
      color: #AEB6BF;
    }}
  </style>
</head>
<body>

<header>
  <h1>📊 特码统计分析</h1>
  <p>基于历史数据的号码频率 · 遗漏 · 生肖多维度统计</p>
  <div class="warn">⚠️ 本站仅用于历史数据统计分析与学习研究，结果不具备任何预测能力，严禁用于赌博或选号。</div>
</header>

<div class="container">

  <div class="latest-card">
    <div>
      <h2>最新报告</h2>
      <p>更新日期：{latest_date}</p>
    </div>
    <a href="{latest_href}" class="btn" target="_blank">立即查看 →</a>
  </div>

  <div class="history">
    <h3>历史报告归档（共 {len(posts)} 份）</h3>
    <table>
      <thead>
        <tr><th>日期</th><th>报告</th></tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>
  </div>

</div>

<footer>
  特码统计分析系统 · 数据每日自动更新 · 严禁赌博选号
</footer>

</body>
</html>
'''

    index_path = BLOG_DIR / 'index.html'
    index_path.write_text(html, encoding='utf-8')
    print(f'✅ 首页已更新：{index_path}（共 {len(posts)} 篇报告）')


def deploy_to_pages():
    """推送 blog/ 到 Cloudflare Pages"""
    env = os.environ.copy()
    env['CLOUDFLARE_API_TOKEN'] = CF_TOKEN
    result = subprocess.run(
        [WRANGLER, 'pages', 'deploy', str(BLOG_DIR),
         '--project-name=macaujc-blog', '--branch=main'],
        cwd=BASE, env=env, capture_output=True, text=True
    )
    if result.returncode == 0:
        print('✅ 已部署到 Cloudflare Pages（https://lhq157.dpdns.org）')
    else:
        print(f'⚠️  Pages 部署失败：{result.stderr[-200:]}')


if __name__ == '__main__':
    publish()
    deploy_to_pages()
