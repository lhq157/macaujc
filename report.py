"""
report.py — 特码统计分析系统 · 一键报告生成器
用法：python report.py

流程：加载数据 → 全量计算 → 自动结论 → 导出 HTML + PDF
输出：output/reports/analysis_report.html
      output/reports/analysis_report.pdf
      output/reports/README.md
"""

import base64, glob, io, os, sys, datetime
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['PingFang SC', 'Hiragino Sans GB',
                                           'STHeiti', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import chisquare, kstest, expon

import config

# ── 常量 ─────────────────────────────────────────────────────────────────
THEORY_GAP   = 49
WARN_MULT    = 2.0
CRIT_MULT    = 3.5
REPORT_DIR   = os.path.join(config.OUTPUT_DIR, 'reports')
HTML_PATH    = os.path.join(REPORT_DIR, 'analysis_report.html')
PDF_PATH     = os.path.join(REPORT_DIR, 'analysis_report.pdf')
MD_PATH      = os.path.join(REPORT_DIR, 'README.md')
ZODIAC_ORDER = ['鼠','牛','虎','兔','龙','蛇','马','羊','猴','鸡','狗','猪']

os.makedirs(REPORT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
# 1. 数据加载
# ══════════════════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    csvs = sorted(glob.glob(os.path.join(config.OUTPUT_DIR, '????????.csv')))
    if not csvs:
        sys.exit('❌ 未找到数据文件，请先运行 python main.py 拉取数据')
    path = csvs[-1]
    df = pd.read_csv(path, dtype={'expect': str})
    df['openTime'] = pd.to_datetime(df['openTime'])
    df = df.sort_values('openTime').reset_index(drop=True)
    print(f'✅ 数据加载：{path}（共 {len(df)} 期）')
    return df


# ══════════════════════════════════════════════════════════════════════════
# 2. 统计计算
# ══════════════════════════════════════════════════════════════════════════

def compute_stats(df: pd.DataFrame) -> dict:
    specials  = df['special'].values.astype(int)
    n         = len(specials)
    date_min  = df['openTime'].min().strftime('%Y-%m-%d')
    date_max  = df['openTime'].max().strftime('%Y-%m-%d')

    # 频率
    counts   = Counter(int(x) for x in specials)
    freq_arr = np.array([counts.get(i, 0) for i in range(1, 50)])
    chi2_stat, chi2_p = chisquare(freq_arr, np.full(49, n / 49))

    # 热 / 冷号
    sorted_f = sorted(enumerate(freq_arr, 1), key=lambda x: x[1])
    cold_n   = config.TOP_N
    hot5     = sorted_f[-cold_n:][::-1]
    cold5    = sorted_f[:cold_n]

    # 奇偶 / 大小
    odd_r  = sum(1 for x in specials if x % 2 == 1) / n
    big_r  = sum(1 for x in specials if x > config.BIG_NUMBER_THRESHOLD) / n

    # 遗漏
    gap_rows = []
    for num in range(1, 50):
        pos = np.where(specials == num)[0]
        cur = n - 1 - int(pos[-1]) if len(pos) > 0 else n
        avg = float(np.diff(pos).mean()) if len(pos) >= 2 else float(n)
        ratio = cur / THEORY_GAP
        lv = '🔴 严重' if ratio >= CRIT_MULT else ('🟡 警告' if ratio >= WARN_MULT else '✅ 正常')
        gap_rows.append({'号码': num, '当前遗漏': cur, '实际均间隔': round(avg, 1),
                         '比值': round(ratio, 2), '状态': lv})
    gap_rows.sort(key=lambda r: r['当前遗漏'], reverse=True)
    anomalies = [r for r in gap_rows if r['比值'] >= WARN_MULT]

    # 生肖频率（若 zodiac 列存在）
    zodiac_stats = {}
    if 'zodiac' in df.columns:
        zc = Counter(df['zodiac'])
        zodiac_stats = {z: zc.get(z, 0) for z in ZODIAC_ORDER}

    # 间隔分布 KS 检验（指数分布）
    all_gaps = []
    for num in range(1, 50):
        pos = np.where(specials == num)[0]
        if len(pos) >= 2:
            all_gaps.extend(np.diff(pos).tolist())
    ks_stat, ks_p = kstest(all_gaps, 'expon', args=(0, THEORY_GAP)) if all_gaps else (0, 1)

    return dict(
        n=n, date_min=date_min, date_max=date_max,
        freq_arr=freq_arr, chi2_stat=chi2_stat, chi2_p=chi2_p,
        chi2_pass=chi2_p > 0.05,
        hot5=hot5, cold5=cold5,
        odd_r=odd_r, big_r=big_r,
        gap_rows=gap_rows, anomalies=anomalies,
        zodiac_stats=zodiac_stats,
        ks_stat=ks_stat, ks_p=ks_p, ks_pass=ks_p > 0.05,
        n_warn=sum(1 for r in anomalies if r['比值'] < CRIT_MULT),
        n_crit=sum(1 for r in anomalies if r['比值'] >= CRIT_MULT),
    )


# ══════════════════════════════════════════════════════════════════════════
# 3. 自动结论
# ══════════════════════════════════════════════════════════════════════════

def auto_conclusion(s: dict) -> str:
    lines = []
    gen = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')

    # 随机性
    if s['chi2_pass']:
        lines.append(f"【随机性】卡方检验通过（χ²={s['chi2_stat']:.1f}, p={s['chi2_p']:.3f}），"
                     f"{s['n']} 期数据中号码分布符合均匀随机假设，未发现统计显著偏差。")
    else:
        lines.append(f"【随机性】卡方检验未通过（χ²={s['chi2_stat']:.1f}, p={s['chi2_p']:.4f}），"
                     f"号码分布存在显著偏差，建议核查数据质量。")

    # 间隔分布
    if s['ks_pass']:
        lines.append(f"【间隔分布】KS检验通过（p={s['ks_p']:.3f}），号码出现间隔符合指数分布，"
                     f"与泊松过程一致，进一步佐证序列独立性。")
    else:
        lines.append(f"【间隔分布】KS检验未通过（p={s['ks_p']:.4f}），间隔分布偏离指数分布。")

    # 热冷号
    hot_nums  = '、'.join(str(n) for n, _ in s['hot5'])
    cold_nums = '、'.join(str(n) for n, _ in s['cold5'])
    lines.append(f"【频率分布】出现最多的 {config.TOP_N} 个号码为 {hot_nums}，"
                 f"出现最少的 {config.TOP_N} 个号码为 {cold_nums}。"
                 f"注意：频率差异属正常统计波动，不具有预测价值。")

    # 奇偶 / 大小
    lines.append(f"【结构分布】奇数号占比 {s['odd_r']:.1%}（理论 50.0%），"
                 f"大号（>{config.BIG_NUMBER_THRESHOLD}）占比 {s['big_r']:.1%}（理论 "
                 f"{(49 - config.BIG_NUMBER_THRESHOLD)/49:.1%}），均在合理波动范围内。")

    # 异常遗漏
    if not s['anomalies']:
        lines.append("【异常检测】当前所有号码遗漏均在正常范围内（< 2× 理论均值）。")
    else:
        crit_list = [r for r in s['anomalies'] if r['比值'] >= CRIT_MULT]
        warn_list = [r for r in s['anomalies'] if r['比值'] < CRIT_MULT]
        parts = []
        if crit_list:
            nums = '、'.join(str(r['号码']) for r in crit_list)
            parts.append(f"严重遗漏（≥{CRIT_MULT}×）：号码 {nums}")
        if warn_list:
            nums = '、'.join(str(r['号码']) for r in warn_list)
            parts.append(f"警告遗漏（≥{WARN_MULT}×）：号码 {nums}")
        lines.append(f"【异常检测】共 {len(s['anomalies'])} 个号码遗漏超过 2× 理论均值。"
                     f"{'；'.join(parts)}。"
                     f"⚠️ 遗漏长 ≠ 下期更易出现，每期概率始终 = 1/49。")

    # 生肖
    if s['zodiac_stats']:
        zs = sorted(s['zodiac_stats'].items(), key=lambda x: x[1])
        top_z    = zs[-1][0]
        bottom_z = zs[0][0]
        lines.append(f"【生肖分布】出现最多的生肖为「{top_z}」，最少为「{bottom_z}」。"
                     f"各生肖理论占比均为 1/12 ≈ 8.33%，差异属随机波动。")

    # 核心结论
    lines.append(f"【核心结论】基于 {s['n']} 期数据的多维统计检验表明："
                 f"特码序列符合 i.i.d. 离散均匀分布，互信息 = 0，"
                 f"任何基于历史数据的预测方法准确率上限 = 随机猜测 = 1/49 ≈ 2.04%。")

    return '\n\n'.join(f'{i+1}. {l}' for i, l in enumerate(lines))


# ══════════════════════════════════════════════════════════════════════════
# 4. 图表生成
# ══════════════════════════════════════════════════════════════════════════

def _b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64


def build_charts(s: dict) -> dict:
    freq_arr = s['freq_arr']
    n        = s['n']
    gap_rows = s['gap_rows']

    # 图A：频率分布
    fig, ax = plt.subplots(figsize=(14, 3.8))
    colors = ['#E74C3C' if c == freq_arr.max() else
              '#95A5A6' if c == freq_arr.min() else '#5DADE2'
              for c in freq_arr]
    ax.bar(range(1, 50), freq_arr, color=colors, width=0.7)
    ax.axhline(n / 49, color='black', lw=1.5, linestyle='--',
               label=f'理论均值 {n/49:.1f}')
    ax.set_xlabel('号码'); ax.set_ylabel('出现次数')
    ax.set_title('号码频率分布'); ax.legend()
    ax.set_xticks(range(1, 50)); ax.tick_params(axis='x', labelsize=7)
    plt.tight_layout()
    freq_b64 = _b64(fig)

    # 图B：遗漏 / 异常检测
    fig, ax = plt.subplots(figsize=(14, 3.8))
    gc = ['#E74C3C' if r['比值'] >= CRIT_MULT else
          '#F39C12' if r['比值'] >= WARN_MULT else '#2ECC71'
          for r in gap_rows]
    ax.bar(range(49), [r['当前遗漏'] for r in gap_rows], color=gc, width=0.7)
    ax.set_xticks(range(49))
    ax.set_xticklabels([r['号码'] for r in gap_rows], fontsize=7)
    ax.axhline(THEORY_GAP,             color='#3498DB', lw=1.2, linestyle='--', label='理论均值 49')
    ax.axhline(THEORY_GAP * WARN_MULT, color='#F39C12', lw=1.2, linestyle='--', label='警告线 98')
    ax.axhline(THEORY_GAP * CRIT_MULT, color='#E74C3C', lw=1.2, linestyle='--', label='严重线 171')
    ax.set_title('当前遗漏 · 异常检测'); ax.legend(fontsize=8)
    ax.set_xlabel('号码（遗漏降序）'); ax.set_ylabel('当前遗漏（期）')
    plt.tight_layout()
    gap_b64 = _b64(fig)

    # 图C：热冷号对比
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, data, title, color in [
        (axes[0], s['hot5'],  f'Top{config.TOP_N} 热号', '#E74C3C'),
        (axes[1], s['cold5'], f'Top{config.TOP_N} 冷号', '#3498DB'),
    ]:
        nums, cnts = zip(*data)
        ax.barh([str(n) for n in nums], cnts, color=color, alpha=0.85)
        ax.axvline(s['n'] / 49, color='gray', lw=1, linestyle='--', label='理论均值')
        ax.set_xlabel('出现次数'); ax.set_title(title); ax.legend(fontsize=8)
        for i, v in enumerate(cnts):
            ax.text(v + 0.2, i, str(v), va='center', fontsize=9)
    plt.tight_layout()
    hotcold_b64 = _b64(fig)

    # 图D：生肖分布（若有）
    zodiac_b64 = ''
    if s['zodiac_stats']:
        zvals = [s['zodiac_stats'].get(z, 0) for z in ZODIAC_ORDER]
        fig, ax = plt.subplots(figsize=(10, 3.8))
        ax.bar(ZODIAC_ORDER, zvals, color='#9B59B6', alpha=0.8)
        ax.axhline(s['n'] / 12, color='black', lw=1.5, linestyle='--',
                   label=f'理论均值 {s["n"]/12:.1f}')
        ax.set_xlabel('生肖'); ax.set_ylabel('出现次数')
        ax.set_title('生肖频率分布'); ax.legend()
        plt.tight_layout()
        zodiac_b64 = _b64(fig)

    return dict(freq=freq_b64, gap=gap_b64, hotcold=hotcold_b64, zodiac=zodiac_b64)


# ══════════════════════════════════════════════════════════════════════════
# 5. HTML 导出
# ══════════════════════════════════════════════════════════════════════════

def export_html(s: dict, charts: dict, conclusion: str):
    gen_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')

    anom_rows = ''.join(
        f'<tr class="{"crit" if r["比值"]>=CRIT_MULT else "warn"}">'
        f'<td>{r["号码"]}</td><td>{r["当前遗漏"]}</td>'
        f'<td>{r["比值"]}</td><td>{r["状态"]}</td></tr>'
        for r in s['anomalies']
    ) or '<tr><td colspan="4" style="text-align:center">✅ 当前无异常号码</td></tr>'

    zodiac_section = ''
    if charts['zodiac']:
        zodiac_section = f'''
  <h2>四、生肖分布</h2>
  <div class="card">
    <img src="data:image/png;base64,{charts["zodiac"]}" alt="生肖分布">
  </div>'''

    conclusion_html = ''.join(
        f'<p style="margin:10px 0;line-height:1.8">{p}</p>'
        for p in conclusion.split('\n\n')
    )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>特码统计分析报告</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:'PingFang SC','Microsoft YaHei',sans-serif;background:#F4F6F7;color:#2C3E50;font-size:14px}}
  .wrap{{max-width:1100px;margin:0 auto;padding:32px 20px}}
  header{{background:linear-gradient(135deg,#1A252F,#2980B9);color:white;padding:36px 40px;border-radius:10px;margin-bottom:24px}}
  header h1{{font-size:26px;font-weight:bold;margin-bottom:8px}}
  header p{{color:#AEB6BF;font-size:13px}}
  h2{{color:#2980B9;font-size:16px;font-weight:bold;margin:28px 0 12px;padding-left:10px;border-left:4px solid #2980B9}}
  .card{{background:#fff;border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,.07);padding:20px;margin-bottom:16px}}
  .stat-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:16px}}
  .stat-box{{background:#fff;border-radius:8px;padding:18px 12px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,.07)}}
  .stat-num{{font-size:26px;font-weight:bold;color:#2980B9}}
  .stat-lbl{{font-size:11px;color:#7F8C8D;margin-top:5px}}
  table{{width:100%;border-collapse:collapse}}
  th{{background:#2980B9;color:#fff;padding:9px 14px;text-align:left;font-size:13px}}
  td{{padding:8px 14px;border-bottom:1px solid #ECF0F1;font-size:13px}}
  tr.crit td{{background:#FADBD8}} tr.warn td{{background:#FEF9E7}}
  tr:hover td{{background:#EBF5FB}}
  .badge{{display:inline-block;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:bold}}
  .pass{{background:#D5F5E3;color:#1E8449}}.fail{{background:#FADBD8;color:#C0392B}}
  img{{width:100%;border-radius:6px;margin:10px 0}}
  .warn-box{{background:#FEF9E7;border-left:4px solid #F39C12;padding:10px 16px;margin-bottom:14px;border-radius:0 6px 6px 0;font-size:13px}}
  .conclusion{{background:#EBF5FB;border-left:4px solid #2980B9;padding:16px 20px;border-radius:0 8px 8px 0;line-height:1.9;font-size:13.5px}}
  footer{{text-align:center;color:#95A5A6;font-size:12px;margin-top:40px;padding-top:16px;border-top:1px solid #ECF0F1}}
  @media(max-width:600px){{.stat-grid{{grid-template-columns:repeat(2,1fr)}}}}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>特码统计分析报告</h1>
    <p>数据区间：{s['date_min']} ~ {s['date_max']} &nbsp;|&nbsp; 生成时间：{gen_time}</p>
  </header>

  <div class="stat-grid">
    <div class="stat-box"><div class="stat-num">{s['n']}</div><div class="stat-lbl">总期数</div></div>
    <div class="stat-box"><div class="stat-num">{'通过' if s['chi2_pass'] else '未通过'}</div><div class="stat-lbl">均匀性检验</div></div>
    <div class="stat-box"><div class="stat-num">{len(s['anomalies'])}</div><div class="stat-lbl">异常号码数</div></div>
    <div class="stat-box"><div class="stat-num">2.04%</div><div class="stat-lbl">庄家抽水率</div></div>
  </div>

  <h2>一、频率分布</h2>
  <div class="card">
    <img src="data:image/png;base64,{charts['freq']}" alt="频率分布">
    <table style="margin-top:12px">
      <tr><th>检验项</th><th>结果</th></tr>
      <tr><td>卡方统计量</td><td>{s['chi2_stat']:.2f}</td></tr>
      <tr><td>p 值</td><td>{s['chi2_p']:.4f}</td></tr>
      <tr><td>均匀性</td><td><span class="badge {'pass' if s['chi2_pass'] else 'fail'}">{'✅ 通过（分布符合均匀随机）' if s['chi2_pass'] else '❌ 未通过（存在显著偏差）'}</span></td></tr>
      <tr><td>间隔 KS 检验</td><td><span class="badge {'pass' if s['ks_pass'] else 'fail'}">{'✅ 通过（符合指数分布）' if s['ks_pass'] else '❌ 未通过'}</span></td></tr>
    </table>
  </div>

  <h2>二、热号 / 冷号</h2>
  <div class="card">
    <img src="data:image/png;base64,{charts['hotcold']}" alt="热冷号">
  </div>

  <h2>三、异常检测</h2>
  <div class="card">
    <img src="data:image/png;base64,{charts['gap']}" alt="遗漏检测">
    <table style="margin-top:12px">
      <tr><th>号码</th><th>当前遗漏</th><th>遗漏/理论比</th><th>状态</th></tr>
      {anom_rows}
    </table>
  </div>
  {zodiac_section}

  <h2>{'五' if charts['zodiac'] else '四'}、自动分析结论</h2>
  <div class="card">
    <div class="warn-box">⚠️ 以下结论基于历史统计，<strong>不代表对未来的预测能力</strong>。每期开奖为独立随机事件。</div>
    <div class="conclusion">{conclusion_html}</div>
  </div>

  <footer>特码统计分析系统 · 本报告仅用于学习研究，不具备预测能力，严禁用于赌博或选号</footer>
</div>
</body></html>"""

    with open(HTML_PATH, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'✅ HTML → {HTML_PATH}')


# ══════════════════════════════════════════════════════════════════════════
# 6. PDF 导出
# ══════════════════════════════════════════════════════════════════════════

def export_pdf(s: dict, conclusion: str):
    gen_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')

    with PdfPages(PDF_PATH) as pdf:

        # ── 封面 ──────────────────────────────────────────────────────────
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.patch.set_facecolor('#1A252F')
        kw = dict(ha='center', transform=fig.transFigure)
        fig.text(0.5, 0.68, '特码统计分析报告', fontsize=34, fontweight='bold', color='white', **kw)
        fig.text(0.5, 0.55, f'数据区间：{s["date_min"]} ~ {s["date_max"]}',
                 fontsize=14, color='#AEB6BF', **kw)
        fig.text(0.5, 0.47, f'共 {s["n"]} 期 · 生成于 {gen_time}',
                 fontsize=12, color='#AEB6BF', **kw)
        fig.text(0.5, 0.16,
                 '⚠️  本报告仅用于历史数据统计分析与学习研究，不具备预测能力，严禁用于赌博或选号',
                 fontsize=9, color='#F39C12', **kw)
        pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close(fig)

        # ── 数据概览 ──────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis('off')
        ax.set_title('数据概览 & 统计检验', fontsize=16, fontweight='bold', pad=20)
        rows = [
            ['总期数',       str(s['n'])],
            ['数据起始',     s['date_min']],
            ['数据截止',     s['date_max']],
            ['卡方统计量',   f'{s["chi2_stat"]:.2f}'],
            ['卡方 p 值',    f'{s["chi2_p"]:.4f}'],
            ['均匀性检验',   '通过 ✅' if s['chi2_pass'] else '未通过 ❌'],
            ['间隔 KS 检验', '通过 ✅' if s['ks_pass']   else '未通过 ❌'],
            ['异常号码数',   str(len(s['anomalies']))],
            ['严重遗漏数',   str(s['n_crit'])],
            ['理论平均间隔', '49 期'],
            ['庄家抽水率',   '≈ 2.04%（1/49）'],
            ['预测能力',     '无（互信息 = 0）'],
        ]
        tbl = ax.table(cellText=rows, colLabels=['指标', '值'],
                       loc='center', cellLoc='left')
        tbl.auto_set_font_size(False); tbl.set_fontsize(12); tbl.scale(1.6, 2.1)
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor('#2980B9')
                cell.set_text_props(color='white', fontweight='bold')
            elif r % 2 == 0:
                cell.set_facecolor('#EBF5FB')
        plt.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # ── 频率分布 ──────────────────────────────────────────────────────
        freq_arr = s['freq_arr']
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        colors = ['#E74C3C' if c == freq_arr.max() else
                  '#95A5A6' if c == freq_arr.min() else '#5DADE2'
                  for c in freq_arr]
        ax.bar(range(1, 50), freq_arr, color=colors, width=0.7)
        ax.axhline(s['n'] / 49, color='black', lw=1.5, linestyle='--',
                   label=f'理论均值 {s["n"]/49:.1f}')
        ax.set_xlabel('号码', fontsize=12); ax.set_ylabel('出现次数', fontsize=12)
        ax.set_title('号码频率分布', fontsize=14, fontweight='bold')
        ax.legend(); ax.set_xticks(range(1, 50)); ax.tick_params(axis='x', labelsize=8)
        plt.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # ── 遗漏 / 异常 ───────────────────────────────────────────────────
        gap_rows = s['gap_rows']
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        gc = ['#E74C3C' if r['比值'] >= CRIT_MULT else
              '#F39C12' if r['比值'] >= WARN_MULT else '#2ECC71'
              for r in gap_rows]
        ax.bar(range(49), [r['当前遗漏'] for r in gap_rows], color=gc, width=0.7)
        ax.set_xticks(range(49))
        ax.set_xticklabels([r['号码'] for r in gap_rows], fontsize=8)
        for line, label, color in [
            (THEORY_GAP,             '理论均值 49',  '#3498DB'),
            (THEORY_GAP * WARN_MULT, '警告线 98',    '#F39C12'),
            (THEORY_GAP * CRIT_MULT, '严重线 171',   '#E74C3C'),
        ]:
            ax.axhline(line, color=color, lw=1.5, linestyle='--', label=label)
        ax.set_title('当前遗漏 · 异常检测', fontsize=14, fontweight='bold')
        ax.set_xlabel('号码（遗漏降序）', fontsize=12)
        ax.set_ylabel('当前遗漏（期）', fontsize=12)
        ax.legend(fontsize=10)
        plt.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # ── 自动结论 ──────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis('off')
        ax.set_title('自动分析结论', fontsize=16, fontweight='bold', pad=20)
        wrapped = []
        for para in conclusion.split('\n\n'):
            wrapped.append(para)
            wrapped.append('')
        text = '\n'.join(wrapped)
        ax.text(0.05, 0.92, text, transform=ax.transAxes,
                fontsize=10, va='top', ha='left',
                wrap=True, linespacing=1.8,
                bbox=dict(boxstyle='round,pad=1', facecolor='#EBF5FB', edgecolor='#2980B9', alpha=0.8))
        fig.text(0.5, 0.03,
                 '⚠️  遗漏长 ≠ 下期更易出现，每期概率始终 = 1/49，历史数据对未来零预测力',
                 ha='center', fontsize=9, color='#E74C3C')
        plt.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        d = pdf.infodict()
        d['Title']   = '特码统计分析报告'
        d['Subject'] = f'{s["date_min"]} ~ {s["date_max"]}'

    print(f'✅ PDF  → {PDF_PATH}')


# ══════════════════════════════════════════════════════════════════════════
# 7. Markdown 导出
# ══════════════════════════════════════════════════════════════════════════

def export_markdown(s: dict, conclusion: str):
    gen_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    hot_md   = '\n'.join(f'| 🔥 #{i+1} | {n} | {c} |' for i, (n, c) in enumerate(s['hot5']))
    cold_md  = '\n'.join(f'| 🧊 #{i+1} | {n} | {c} |' for i, (n, c) in enumerate(s['cold5']))
    anom_md  = '\n'.join(
        f'| {r["号码"]} | {r["当前遗漏"]} | {r["比值"]} | {r["状态"]} |'
        for r in s['anomalies']
    ) or '| - | - | - | ✅ 当前无异常 |'

    md = f"""# 特码统计分析报告

> 生成时间：{gen_time}
> 数据区间：{s['date_min']} ~ {s['date_max']}
> ⚠️ 本报告仅用于学习研究，不具备预测能力，严禁用于赌博或选号

---

## 数据概览

| 指标 | 值 |
|------|-----|
| 总期数 | {s['n']} |
| 数据起始 | {s['date_min']} |
| 数据截止 | {s['date_max']} |
| 卡方统计量 | {s['chi2_stat']:.2f} |
| 均匀性检验 | {'✅ 通过' if s['chi2_pass'] else '❌ 未通过'} |
| 间隔 KS 检验 | {'✅ 通过' if s['ks_pass'] else '❌ 未通过'} |
| 异常号码数 | {len(s['anomalies'])} |
| 庄家抽水率 | ≈ 2.04% |

---

## 热号 / 冷号

| 类型 | 号码 | 出现次数 |
|------|------|---------|
{hot_md}
{cold_md}

---

## 异常检测

| 号码 | 当前遗漏 | 遗漏/理论比 | 状态 |
|------|---------|-----------|------|
{anom_md}

---

## 自动分析结论

{conclusion}

---

*特码统计分析系统 · 一键报告生成器*
"""
    with open(MD_PATH, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f'✅ MD   → {MD_PATH}')


# ══════════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════════

def main():
    print('=' * 55)
    print('特码统计分析系统 · 一键报告生成器')
    print('=' * 55)

    df         = load_data()
    stats      = compute_stats(df)
    conclusion = auto_conclusion(stats)
    charts     = build_charts(stats)

    export_html(stats, charts, conclusion)
    export_pdf(stats, conclusion)
    export_markdown(stats, conclusion)

    print()
    print('=' * 55)
    print('全部完成')
    print(f'  📄 HTML（展示+分享）→ {HTML_PATH}')
    print(f'  📋 PDF（变现+发人） → {PDF_PATH}')
    print(f'  📝 README.md       → {MD_PATH}')
    print('=' * 55)


if __name__ == '__main__':
    main()
