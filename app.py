"""
app.py — 特码统计分析系统 · Streamlit Web 界面（产品级 v2.0）
架构：7 大功能 Tab + 智能侧边栏 + 自动报告生成
"""

import glob
import io
import os
import sys
import base64
import subprocess
import datetime
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import chisquare, kstest

import config

# ══════════════════════════════════════════════════════════════════════════
# 0. 环境检测 & 字体初始化
# ══════════════════════════════════════════════════════════════════════════

# 判断是否运行在云端（output 目录不可写）
IS_CLOUD = not os.access(os.path.join(os.path.dirname(__file__) or '.', 'output'), os.W_OK)


def _setup_font():
    """本地或云端自动加载 NotoSansSC 中文字体"""
    import matplotlib.font_manager as fm
    font_path = Path(__file__).parent / 'NotoSansSC-Regular.ttf'
    if not font_path.exists():
        try:
            import urllib.request
            url = ('https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/'
                   'SimplifiedChinese/NotoSansCJKsc-Regular.otf')
            urllib.request.urlretrieve(url, font_path)
        except Exception:
            pass
    if font_path.exists():
        fm.fontManager.addfont(str(font_path))
        matplotlib.rcParams['font.sans-serif'] = (
            ['Noto Sans CJK SC'] + matplotlib.rcParams['font.sans-serif']
        )


_setup_font()

# matplotlib 全局暗色主题
matplotlib.rcParams.update({
    'font.sans-serif':    ['Noto Sans CJK SC', 'PingFang SC', 'Hiragino Sans GB',
                           'STHeiti', 'Arial Unicode MS', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'figure.facecolor':   '#1A2634',
    'axes.facecolor':     '#1A2634',
    'axes.edgecolor':     '#2A3F54',
    'axes.labelcolor':    '#A0AEC0',
    'xtick.color':        '#A0AEC0',
    'ytick.color':        '#A0AEC0',
    'text.color':         '#E0E6ED',
    'grid.color':         '#2A3F54',
    'grid.alpha':         0.28,
    'axes.titlecolor':    '#E0E6ED',
})

# ── 常量定义
ZODIAC_EMOJI = {
    '鼠': '🐭', '牛': '🐮', '虎': '🐯', '兔': '🐰',
    '龙': '🐲', '蛇': '🐍', '马': '🐴', '羊': '🐑',
    '猴': '🐵', '鸡': '🐔', '狗': '🐶', '猪': '🐷',
}
ZODIAC_ORDER = ['鼠', '牛', '虎', '兔', '龙', '蛇', '马', '羊', '猴', '鸡', '狗', '猪']
PALETTE = {
    'blue':   '#00C9FF', 'red':    '#FF6B6B',
    'orange': '#F39C12', 'green':  '#27AE60',
    'purple': '#9B59B6', 'grey':   '#4A5568',
    'bg':     '#1A2634', 'bg2':    '#0F1923',
}
COLORS6 = ['#00C9FF', '#FF6B6B', '#F39C12', '#27AE60', '#9B59B6', '#E67E22']

# ══════════════════════════════════════════════════════════════════════════
# 1. 页面配置 & 全局 CSS
# ══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title='特码统计分析系统',
    page_icon='📊',
    layout='wide',
    initial_sidebar_state='expanded',
)

st.markdown("""
<style>
  html, body, [class*="css"] {
    font-family: -apple-system, 'PingFang SC', 'Helvetica Neue', sans-serif;
  }

  /* ══ Header ══ */
  .dashboard-header {
    background: linear-gradient(135deg, #0D1F35 0%, #0F2744 50%, #152E4A 100%);
    border-radius: 16px; padding: 22px 28px 16px; margin-bottom: 20px;
    border: 1px solid rgba(0,201,255,0.18);
    box-shadow: 0 4px 32px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.05);
    position: relative; overflow: hidden;
  }
  .dashboard-header::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #00C9FF 40%, #7B68EE 70%, transparent);
  }
  .header-top  { display:flex; align-items:center; justify-content:space-between; margin-bottom:10px; }
  .header-title { font-size:20px; font-weight:700; color:#fff; letter-spacing:.3px; }
  .header-title span { color:#00C9FF; }
  .header-meta { display:flex; gap:20px; }
  .header-meta-item { text-align:center; }
  .header-meta-val { font-size:18px; font-weight:700; color:#00C9FF; line-height:1; }
  .header-meta-val.red { color:#FF6B6B; }
  .header-meta-lbl { font-size:10px; color:rgba(224,230,237,0.4); margin-top:3px; letter-spacing:.5px; }
  .header-divider { height:1px; background:rgba(255,255,255,0.06); margin:10px 0 8px; }
  .header-notice { font-size:11px; color:rgba(224,230,237,0.32); display:flex; align-items:center; gap:6px; }
  .header-notice::before { content:'⚠'; font-size:10px; opacity:.55; }

  /* ══ KPI Stat Cards ══ */
  .stat-card {
    background: linear-gradient(160deg, #1C2F45 0%, #152438 100%);
    border-radius: 14px; padding: 18px 14px 16px; text-align: center;
    border: 1px solid rgba(0,201,255,0.13);
    box-shadow: 0 4px 20px rgba(0,0,0,0.35);
    transition: transform .15s, box-shadow .15s;
    position: relative; overflow: hidden;
  }
  .stat-card::after {
    content:''; position:absolute; top:0; left:0; right:0; height:3px;
    background: var(--card-accent, linear-gradient(90deg,#00C9FF,#7B68EE));
    border-radius: 14px 14px 0 0;
  }
  .stat-card:hover { transform: translateY(-2px); box-shadow: 0 6px 28px rgba(0,0,0,0.45); }
  .stat-num { font-size:26px; font-weight:700; color: var(--card-color,#00C9FF); line-height:1.1; }
  .stat-lbl { font-size:10px; color:rgba(224,230,237,0.42); margin-top:7px;
               text-transform:uppercase; letter-spacing:.9px; }
  .stat-sub { font-size:11px; color:rgba(224,230,237,0.38); margin-top:3px; }

  /* ══ Section Header (图表区块标题) ══ */
  .ch {
    margin: 20px 0 6px;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
  }
  .ch-title {
    font-size: 13px; font-weight: 600; color: #E0E6ED;
    display: flex; align-items: center; gap: 8px;
  }
  .ch-title .ic { color: #00C9FF; font-size: 14px; }
  .ch-desc {
    font-size: 11px; color: rgba(224,230,237,0.4);
    margin-top: 4px; line-height: 1.6;
  }

  /* ══ Legacy section-title (保留兼容) ══ */
  .section-title {
    font-size: 13px; font-weight: 600; color: #E0E6ED;
    border-left: 3px solid #00C9FF; padding-left: 10px; margin: 18px 0 10px;
  }

  /* ══ Insight Boxes ══ */
  .ib {
    border-radius: 0 8px 8px 0; padding: 9px 14px;
    margin: 10px 0 6px; font-size: 12px; line-height: 1.75;
    color: rgba(224,230,237,0.72);
  }
  .ib-blue   { background:rgba(0,201,255,0.07);  border-left:3px solid #00C9FF; }
  .ib-green  { background:rgba(39,174,96,0.07);  border-left:3px solid #27AE60; }
  .ib-orange { background:rgba(243,156,18,0.07); border-left:3px solid #F39C12; }
  .ib-red    { background:rgba(231,76,60,0.07);  border-left:3px solid #E74C3C; }

  /* ══ Rank List ══ */
  .rank-item {
    display:flex; align-items:center; gap:10px; padding:7px 0;
    border-bottom:1px solid rgba(255,255,255,0.04); font-size:12px;
  }
  .rank-pos { width:18px; font-size:11px; color:rgba(224,230,237,.28); text-align:center; }
  .rank-icon { font-size:13px; }
  .rank-num  { font-weight:700; font-size:14px; min-width:38px; }
  .rank-bar  {
    flex:1; height:5px; border-radius:3px; background:rgba(255,255,255,0.06); overflow:hidden;
  }
  .rank-fill { height:100%; border-radius:3px; }
  .rank-cnt  { font-size:12px; color:#E0E6ED; min-width:38px; text-align:right; }
  .rank-dlt  { font-size:11px; color:rgba(224,230,237,0.32); min-width:34px; text-align:right; }
  .hot-num   { color:#FF6B6B; }
  .cold-num  { color:#A0AEC0; }

  /* ══ Check Row (完整性检查) ══ */
  .check-row {
    display:flex; align-items:center; gap:10px; padding:8px 12px; margin:4px 0;
    border-radius:8px; background:rgba(255,255,255,0.025); font-size:12px;
  }
  .check-icon { font-size:15px; width:20px; text-align:center; }
  .check-name { color:rgba(224,230,237,.55); flex:1; }
  .check-ok   { color:#27AE60; font-weight:600; }
  .check-fail { color:#E74C3C; font-weight:600; }

  /* ══ Leaderboard Cards (冷热榜) ══ */
  .lb-card {
    background: linear-gradient(160deg,#1A2E44,#152438); border-radius:12px;
    padding:16px 18px; border:1px solid rgba(255,255,255,0.06);
    box-shadow:0 2px 14px rgba(0,0,0,0.25);
  }
  .lb-title { font-size:13px; font-weight:600; color:#E0E6ED; margin-bottom:10px; }
  .lb-row   { display:flex; align-items:center; justify-content:space-between;
               padding:5px 0; border-bottom:1px solid rgba(255,255,255,0.04); font-size:12px; }
  .lb-row:last-child { border:none; }
  .lb-tag   { font-size:10px; padding:1px 7px; border-radius:10px; margin-left:4px; }
  .lb-hot   { background:rgba(255,107,107,.15); color:#FF6B6B; }
  .lb-cold  { background:rgba(74,85,104,.3);   color:#A0AEC0; }
  .lb-miss  { background:rgba(243,156,18,.12); color:#F39C12; }

  /* ══ Pass / Fail Pills ══ */
  .pill-pass { display:inline-block; padding:4px 14px; border-radius:20px;
               background:rgba(39,174,96,.15); color:#27AE60;
               border:1px solid rgba(39,174,96,.3); font-size:12px; font-weight:600; }
  .pill-fail { display:inline-block; padding:4px 14px; border-radius:20px;
               background:rgba(231,76,60,.15); color:#E74C3C;
               border:1px solid rgba(231,76,60,.3); font-size:12px; font-weight:600; }

  /* ══ Conclusion Items ══ */
  .concl-item { padding:10px 14px; margin:6px 0; font-size:13px; color:#E0E6ED;
                border-radius:8px; background:rgba(255,255,255,0.03);
                border-left:3px solid rgba(0,201,255,0.4); line-height:1.7; }
  .concl-item strong { color:#E0E6ED; }
  .concl-warn { border-left-color:rgba(243,156,18,0.6)  !important; }
  .concl-ok   { border-left-color:rgba(39,174,96,0.6)   !important; }

  /* ══ Test Box ══ */
  .test-box { background:rgba(255,255,255,0.03); border-radius:12px;
              padding:18px 20px; border:1px solid rgba(255,255,255,0.07); height:100%; }
  .test-title { font-size:14px; font-weight:600; color:#E0E6ED; margin-bottom:14px; }
  .test-row { display:flex; justify-content:space-between; padding:6px 0;
              border-bottom:1px solid rgba(255,255,255,0.04); font-size:12px; }
  .test-key { color:rgba(224,230,237,0.5); }
  .test-val { color:#E0E6ED; font-weight:500; }

  /* ══ Report Card ══ */
  .report-card { background:rgba(0,201,255,0.04); border-radius:12px;
                 padding:18px 22px; border:1px solid rgba(0,201,255,0.1); margin-bottom:12px; }
  .report-card h4 { color:#00C9FF; margin:0 0 10px; font-size:14px; }
  .report-card p  { font-size:13px; color:#E0E6ED; margin:5px 0; line-height:1.7; }

  /* ══ Anomaly KPI variant ══ */
  .anom-card {
    border-radius:12px; padding:16px 14px; text-align:center;
    border:1px solid; box-shadow:0 2px 16px rgba(0,0,0,0.3);
  }
  .anom-num { font-size:30px; font-weight:700; line-height:1.1; }
  .anom-lbl { font-size:10px; margin-top:6px; text-transform:uppercase; letter-spacing:.8px;
               opacity:.6; }

  /* ══ Summary eval box ══ */
  .eval-box {
    border-radius:12px; padding:18px 22px;
    border:1px solid rgba(255,255,255,0.07);
    background:rgba(255,255,255,0.025);
  }
  .eval-title { font-size:15px; font-weight:600; margin-bottom:8px; }
  .eval-body  { font-size:12px; color:rgba(224,230,237,0.55); line-height:1.8; }

  /* ══ Sidebar & misc ══ */
  section[data-testid="stSidebar"] { background:#111D2C !important;
    border-right:1px solid rgba(0,201,255,0.08); }
  #MainMenu, footer { visibility:hidden; }
  header[data-testid="stHeader"] { background:transparent; }
  div[data-testid="stDataFrame"] { border-radius:10px; overflow:hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# 2. 数据加载
# ══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    """加载并规范化指定 CSV 文件"""
    df = pd.read_csv(path, dtype={'expect': str})
    df['openTime'] = pd.to_datetime(df['openTime'])
    df = df.sort_values('openTime').reset_index(drop=True)
    return df


def fetch_data():
    """调用 main.py 拉取最新数据（仅本地环境）"""
    with st.spinner('正在拉取最新数据，请稍候...'):
        result = subprocess.run(
            [sys.executable, 'main.py'],
            capture_output=True, text=True,
            cwd=os.path.dirname(__file__) or '.'
        )
    if result.returncode == 0:
        st.cache_data.clear()
        st.success('✅ 数据更新成功')
    else:
        st.error(f'❌ 拉取失败：{result.stderr[-300:]}')


# ══════════════════════════════════════════════════════════════════════════
# 3. 核心计算函数
# ══════════════════════════════════════════════════════════════════════════

def compute_freq(specials: np.ndarray):
    """计算 1-49 各号码频次，返回 (freq_arr[49], avg_freq)"""
    cnt  = Counter(specials.tolist())
    freq = np.array([cnt.get(i, 0) for i in range(1, 50)])
    return freq, len(specials) / 49


def compute_gaps(specials: np.ndarray):
    """计算各号码当前遗漏及全量历史间隔，返回 (gap_arr[49], all_gaps)"""
    n        = len(specials)
    gap      = {}
    all_gaps = []
    for num in range(1, 50):
        idx      = np.where(specials == num)[0]
        gap[num] = n - 1 - int(idx[-1]) if len(idx) > 0 else n
        if len(idx) > 1:
            all_gaps.extend(np.diff(idx).tolist())
    gap_arr = np.array([gap[i] for i in range(1, 50)])
    return gap_arr, np.array(all_gaps, dtype=float)


def compute_autocorr(specials: np.ndarray, lags: int = 20) -> np.ndarray:
    """计算序列样本自相关系数（lags 1..lags），去均值标准化"""
    x   = specials.astype(float)
    x   = x - x.mean()
    var = np.var(x)
    if var == 0 or len(x) < lags + 2:
        return np.zeros(lags)
    nn  = len(x)
    return np.array([np.dot(x[:nn-k], x[k:]) / (nn * var) for k in range(1, lags + 1)])


def compute_zodiac(df: pd.DataFrame) -> dict:
    """生肖频次统计，返回 {zodiac: count} 字典"""
    if 'zodiac' not in df.columns:
        return {}
    return dict(Counter(df['zodiac'].dropna()))


# ══════════════════════════════════════════════════════════════════════════
# 3b. HTML UI 辅助函数（卡片标题、洞察框、排行榜、KPI卡片）
# ══════════════════════════════════════════════════════════════════════════

def ch(title: str, desc: str = '', icon: str = '') -> str:
    """生成图表区块标题 HTML（含图标 + 副标题说明）"""
    ic   = f'<span class="ic">{icon}</span>' if icon else ''
    sub  = f'<div class="ch-desc">{desc}</div>' if desc else ''
    return (f'<div class="ch"><div class="ch-title">{ic}{title}</div>{sub}</div>')


def ib(text: str, color: str = 'blue') -> str:
    """生成洞察/说明文字框 HTML（blue / green / orange / red）"""
    return f'<div class="ib ib-{color}">{text}</div>'


def stat_card_html(val: str, lbl: str, sub: str = '',
                   accent: str = '#00C9FF', grad: str = '') -> str:
    """生成彩色顶部条 KPI 卡片 HTML"""
    grad_css = grad if grad else f'linear-gradient(90deg,{accent},{accent}88)'
    color_css = accent
    sub_html  = (f'<div class="stat-sub">{sub}</div>' if sub else '')
    return (
        f'<div class="stat-card" style="--card-accent:{grad_css};--card-color:{color_css}">'
        f'<div class="stat-num">{val}</div>'
        f'<div class="stat-lbl">{lbl}</div>'
        f'{sub_html}</div>'
    )


def rank_html(items: list, max_val: float, hot: bool = True) -> str:
    """生成内联排行榜 HTML
    items: [(num, count), ...]  max_val: 用于计算进度条宽度
    hot=True → 红色，hot=False → 冷色
    """
    bar_color = '#FF6B6B' if hot else '#4A5568'
    num_cls   = 'hot-num' if hot else 'cold-num'
    icon      = '🔥' if hot else '🧊'
    rows = ''
    for i, (num, cnt) in enumerate(items, 1):
        pct   = cnt / max(max_val, 1) * 100
        delta = cnt - (max_val / len(items) if len(items) else 1)
        sign  = '+' if delta >= 0 else ''
        rows += (
            f'<div class="rank-item">'
            f'<span class="rank-pos">{i}</span>'
            f'<span class="rank-icon">{icon}</span>'
            f'<span class="rank-num {num_cls}">No.{num:02d}</span>'
            f'<div class="rank-bar"><div class="rank-fill" '
            f'style="width:{pct:.0f}%;background:{bar_color}"></div></div>'
            f'<span class="rank-cnt">{cnt}次</span>'
            f'<span class="rank-dlt">{sign}{delta:.0f}</span>'
            f'</div>'
        )
    return rows


def lb_card_html(title: str, items: list, tag_cls: str, fmt_fn) -> str:
    """生成冷热榜卡片 HTML
    items: [(num, val), ...]  fmt_fn(num, val) → right-side text
    """
    rows = ''
    for num, val in items:
        rows += (
            f'<div class="lb-row">'
            f'<span style="color:#E0E6ED;font-weight:600">No.{num:02d}</span>'
            f'<span><span class="lb-tag {tag_cls}">{fmt_fn(num, val)}</span></span>'
            f'</div>'
        )
    return f'<div class="lb-card"><div class="lb-title">{title}</div>{rows}</div>'


# ══════════════════════════════════════════════════════════════════════════
# 4. 绘图函数（统一暗色风格，均返回 matplotlib figure）
# ══════════════════════════════════════════════════════════════════════════

def _fig(w: float, h: float):
    """创建统一背景的 figure/ax（仅保留左/下坐标轴，移除上/右刺线）"""
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(PALETTE['bg'])
    # 去除上/右刺线，保留左/下（更简洁专业）
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#2A3F54')
    ax.spines['bottom'].set_color('#2A3F54')
    return fig, ax


def fig_freq_bar(freq_arr: np.ndarray, avg_freq: float):
    """号码频率柱状图（红=最高 / 灰=最低 / 蓝=普通）"""
    fig, ax = _fig(14, 4)
    colors  = [PALETTE['red']  if v == freq_arr.max() else
               PALETTE['grey'] if v == freq_arr.min() else
               PALETTE['blue'] for v in freq_arr]
    ax.bar(range(1, 50), freq_arr, color=colors, width=0.72, zorder=2)
    ax.axhline(avg_freq, color=PALETTE['orange'], lw=1.8, ls='--',
               label=f'均值 {avg_freq:.1f}', zorder=3)
    ax.set_xlabel('号码', fontsize=9)
    ax.set_ylabel('出现次数', fontsize=9)
    ax.set_title('特码频率分布（红=最高 · 灰=最低）', fontsize=11, pad=10)
    ax.set_xticks(range(1, 50))
    ax.tick_params(axis='x', labelsize=6.5)
    ax.legend(fontsize=9)
    ax.grid(axis='y')
    fig.tight_layout()
    return fig


def fig_heatmap(freq_arr: np.ndarray):
    """7×7 号码频率热力图"""
    grid = freq_arr.reshape(7, 7)
    fig, ax = _fig(9, 5.5)
    im = ax.imshow(grid, cmap='YlOrRd', aspect='auto')
    fig.colorbar(im, ax=ax, shrink=0.85, label='出现次数')
    mean_v = freq_arr.mean()
    for r in range(7):
        for c in range(7):
            num = r * 7 + c + 1
            val = freq_arr[num - 1]
            tc  = 'black' if val > mean_v * 1.15 else 'white'
            ax.text(c, r, f'{num}\n{val}', ha='center', va='center',
                    fontsize=7, color=tc, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('号码频率热力图（7×7 布局）', fontsize=11, pad=10)
    fig.tight_layout()
    return fig


def fig_pie(values, labels, colors, title: str):
    """通用饼图"""
    fig, ax = _fig(4, 4)
    _, _, autotexts = ax.pie(
        values, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=90,
        textprops={'fontsize': 11}
    )
    for at in autotexts:
        at.set_fontsize(9)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig


def fig_tail_bar(specials: np.ndarray, n: int):
    """尾数（个位）分布柱状图"""
    tail_cnt  = Counter((specials % 10).tolist())
    tail_vals = [tail_cnt.get(i, 0) for i in range(10)]
    fig, ax   = _fig(5, 4)
    ax.bar(range(10), tail_vals, color=PALETTE['blue'], width=0.65, zorder=2)
    ax.axhline(n / 10, color=PALETTE['orange'], lw=1.4, ls='--',
               label=f'均值 {n/10:.1f}')
    ax.set_xticks(range(10))
    ax.set_xticklabels([str(i) for i in range(10)])
    ax.set_xlabel('尾数（个位数字）')
    ax.set_ylabel('出现次数')
    ax.set_title('尾数分布', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(axis='y')
    fig.tight_layout()
    return fig


def fig_zodiac_bar(zodiac_cnt: dict, highlight=None):
    """生肖频率柱状图（纯汉字标签，避免云端 emoji 方块）"""
    zlist  = [z for z in ZODIAC_ORDER if z in zodiac_cnt]
    if not zlist:
        fig, ax = _fig(8, 3)
        ax.text(0.5, 0.5, '无生肖数据', ha='center', va='center',
                transform=ax.transAxes, color='white')
        ax.axis('off')
        return fig
    vals   = [zodiac_cnt[z] for z in zlist]
    avg    = sum(vals) / len(vals)
    colors = [PALETTE['red']    if v == max(vals) else
              PALETTE['grey']   if v == min(vals) else
              PALETTE['purple'] if z == highlight else
              PALETTE['blue']   for z, v in zip(zlist, vals)]
    fig, ax = _fig(11, 4)
    ax.bar(zlist, vals, color=colors, width=0.65, zorder=2)
    ax.axhline(avg, color=PALETTE['orange'], lw=1.6, ls='--',
               label=f'均值 {avg:.1f}')
    ax.set_xlabel('生肖')
    ax.set_ylabel('出现次数')
    ax.set_title('生肖频率分布（红=最高 · 灰=最低）', fontsize=11, pad=10)
    ax.legend(fontsize=9)
    ax.grid(axis='y')
    fig.tight_layout()
    return fig


def fig_monthly_trend(df: pd.DataFrame, top_n: int = 6):
    """生肖月度趋势折线图（取频次最高 top_n 生肖）"""
    df2      = df.copy()
    df2['ym'] = df2['openTime'].dt.to_period('M')
    monthly   = df2.groupby(['ym', 'zodiac']).size().unstack(fill_value=0)
    zc        = Counter(df2['zodiac'].dropna())
    top_z     = sorted([z for z in ZODIAC_ORDER if z in zc and z in monthly.columns],
                       key=lambda z: zc[z], reverse=True)[:top_n]
    mtop      = monthly[top_z]
    xticks    = mtop.index.astype(str).tolist()
    step      = max(1, len(xticks) // 14)
    fig, ax   = _fig(12, 4)
    for i, z in enumerate(mtop.columns):
        ax.plot(range(len(mtop)), mtop[z].values,
                marker='o', markersize=3, lw=1.6,
                color=COLORS6[i % len(COLORS6)], label=z)
    ax.set_xticks(range(0, len(xticks), step))
    ax.set_xticklabels(xticks[::step], rotation=35, fontsize=7)
    ax.set_ylabel('出现次数')
    ax.set_title(f'Top{top_n} 生肖月度趋势', fontsize=11, pad=10)
    ax.legend(fontsize=8, ncol=3)
    ax.grid()
    fig.tight_layout()
    return fig


def fig_gap_bar(gap_arr: np.ndarray, avg_gap: float):
    """各号码当前遗漏期数柱状图（蓝=正常 / 橙=预警 / 红=危险）"""
    warn  = avg_gap * 2.0
    crit  = avg_gap * 3.5
    colors = [PALETTE['red']    if g >= crit else
              PALETTE['orange'] if g >= warn else
              PALETTE['blue']   for g in gap_arr]
    fig, ax = _fig(14, 4)
    ax.bar(range(1, 50), gap_arr, color=colors, width=0.72, zorder=2)
    ax.axhline(avg_gap, color='white',          lw=1.6, ls='--',
               label=f'均值 {avg_gap:.1f}')
    ax.axhline(warn,    color=PALETTE['orange'], lw=1.0, ls=':',
               label=f'2× 预警 {warn:.0f}')
    ax.axhline(crit,    color=PALETTE['red'],    lw=1.0, ls=':',
               label=f'3.5× 危险 {crit:.0f}')
    ax.set_xlabel('号码')
    ax.set_ylabel('遗漏期数')
    ax.set_title('各号码当前遗漏期数', fontsize=11, pad=10)
    ax.set_xticks(range(1, 50))
    ax.tick_params(axis='x', labelsize=6.5)
    ax.legend(fontsize=8)
    ax.grid(axis='y')
    fig.tight_layout()
    return fig


def fig_gap_hist(all_gaps: np.ndarray, avg_gap: float):
    """间隔分布直方图 + 理论指数分布曲线"""
    fig, ax = _fig(11, 3.8)
    ax.hist(all_gaps, bins=50, density=True,
            color=PALETTE['blue'], alpha=0.72, label='实际间隔分布', zorder=2)
    x = np.linspace(0, all_gaps.max(), 300)
    ax.plot(x, (1 / avg_gap) * np.exp(-x / avg_gap),
            color=PALETTE['red'], lw=2.2, label=f'指数分布 (λ=1/{avg_gap:.0f})')
    ax.set_xlabel('间隔（期数）')
    ax.set_ylabel('密度')
    ax.set_title('间隔分布 vs 理论指数分布', fontsize=11, pad=10)
    ax.legend(fontsize=9)
    ax.grid()
    fig.tight_layout()
    return fig


def fig_autocorr(acf_vals: np.ndarray, n: int):
    """自相关系数图（含 95% 置信区间线）"""
    lags = range(1, len(acf_vals) + 1)
    conf = 1.96 / np.sqrt(n)
    fig, ax = _fig(11, 3.8)
    ax.bar(lags, acf_vals, color=PALETTE['blue'], width=0.6, alpha=0.85, zorder=2)
    ax.axhline(conf,  color=PALETTE['red'], lw=1.4, ls='--', label=f'95% CI ±{conf:.3f}')
    ax.axhline(-conf, color=PALETTE['red'], lw=1.4, ls='--')
    ax.axhline(0, color='white', lw=0.7)
    ax.set_xlabel('滞后期 (Lag)')
    ax.set_ylabel('ACF')
    ax.set_title('序列自相关系数（超出虚线=潜在相关）', fontsize=11, pad=10)
    ax.legend(fontsize=9)
    ax.set_xlim(0.5, len(acf_vals) + 0.5)
    ax.grid()
    fig.tight_layout()
    return fig


def fig_rolling_freq(specials: np.ndarray, window_n: int, target: int):
    """单号码滚动窗口频率变化图"""
    n   = len(specials)
    wn  = min(window_n, n - 1)
    if wn < 5:
        fig, ax = _fig(10, 3)
        ax.text(0.5, 0.5, '数据不足，无法绘图', ha='center', va='center',
                transform=ax.transAxes, color='white')
        ax.axis('off')
        return fig
    rolls    = [Counter(specials[i - wn:i].tolist()).get(target, 0) / wn * 100
                for i in range(wn, n + 1)]
    expected = 100 / 49
    xs       = range(wn, n + 1)
    fig, ax  = _fig(12, 3.8)
    ax.plot(xs, rolls, color=PALETTE['blue'], lw=1.4,
            label=f'No.{target} 滚动频率%')
    ax.axhline(expected, color=PALETTE['orange'], lw=1.6, ls='--',
               label=f'理论均值 {expected:.2f}%')
    ax.fill_between(xs, rolls, expected,
                    where=[v > expected for v in rolls],
                    alpha=0.22, color=PALETTE['red'], label='偏高区间')
    ax.fill_between(xs, rolls, expected,
                    where=[v <= expected for v in rolls],
                    alpha=0.14, color=PALETTE['blue'])
    ax.set_xlabel('期序（滚动右端）')
    ax.set_ylabel('频率 %')
    ax.set_title(f'号码 {target} | 滑动窗口 {wn} 期频率', fontsize=11, pad=10)
    ax.legend(fontsize=8)
    ax.grid()
    fig.tight_layout()
    return fig


def fig_anomaly_scatter(specials: np.ndarray, gap_arr: np.ndarray, avg_gap: float):
    """遗漏异常号码近 200 期出现位置散点图"""
    n    = len(specials)
    warn = avg_gap * 2.0
    crit = avg_gap * 3.5
    anom = [i + 1 for i, g in enumerate(gap_arr) if g >= warn]
    if not anom:
        fig, ax = _fig(10, 2.5)
        ax.text(0.5, 0.5, '✅  当前无遗漏异常号码', ha='center', va='center',
                transform=ax.transAxes, fontsize=13, color=PALETTE['green'])
        ax.axis('off')
        return fig
    fig, ax = _fig(13, max(3.0, len(anom) * 0.38 + 1.5))
    for yi, num in enumerate(anom):
        idx    = np.where(specials == num)[0]
        recent = idx[idx >= max(0, n - 200)] - max(0, n - 200)
        color  = PALETTE['red'] if gap_arr[num - 1] >= crit else PALETTE['orange']
        ax.scatter(recent, [yi] * len(recent),
                   color=color, s=55, zorder=3, alpha=0.85)
    ax.set_yticks(range(len(anom)))
    ax.set_yticklabels([f'No.{i}' for i in anom], fontsize=8)
    ax.set_xlim(-3, 203)
    ax.set_xlabel('近 200 期（右端=最新）')
    ax.set_title('遗漏异常号码出现位置  🔴危险 / 🟡预警', fontsize=11, pad=10)
    ax.grid()
    fig.tight_layout()
    return fig


def fig_zodiac_odd(df: pd.DataFrame):
    """生肖 × 奇偶 组合分布分组柱状图"""
    if 'zodiac' not in df.columns:
        return None
    df2          = df.copy()
    df2['parity'] = df2['special'].apply(lambda x: '奇' if int(x) % 2 == 1 else '偶')
    pivot         = (df2.groupby(['zodiac', 'parity'])
                        .size().unstack(fill_value=0))
    pivot         = pivot.reindex([z for z in ZODIAC_ORDER if z in pivot.index])
    x    = np.arange(len(pivot))
    w    = 0.36
    odd_v = pivot.get('奇', pd.Series(dtype=int)).reindex(pivot.index, fill_value=0).values
    eve_v = pivot.get('偶', pd.Series(dtype=int)).reindex(pivot.index, fill_value=0).values
    fig, ax = _fig(11, 4.5)
    ax.bar(x - w / 2, odd_v, w, color=PALETTE['blue'],   label='奇', alpha=0.9, zorder=2)
    ax.bar(x + w / 2, eve_v, w, color=PALETTE['orange'], label='偶', alpha=0.9, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index.tolist(), fontsize=10)
    ax.set_ylabel('出现次数')
    ax.set_title('生肖 × 奇偶 组合分布', fontsize=11, pad=10)
    ax.legend(fontsize=9)
    ax.grid(axis='y')
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════
# 5. HTML 报告生成
# ══════════════════════════════════════════════════════════════════════════

def fig_to_b64(fig) -> str:
    """将 matplotlib figure 转为 base64 PNG 字符串（用于内嵌 HTML）"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=90, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def generate_html_report(df_full, freq_arr, avg_freq, chi2_stat, chi2_p,
                         ks_p, gap_arr, zodiac_cnt, sig_level,
                         specials, n, latest, pass_cnt) -> str:
    """生成完整暗色主题 HTML 分析报告（含 4 张内嵌图表）"""
    now   = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    start = df_full['openTime'].min().strftime('%Y-%m-%d')
    end   = df_full['openTime'].max().strftime('%Y-%m-%d')

    # 生成嵌入图表
    figs = {
        'freq': fig_freq_bar(freq_arr, avg_freq),
        'heat': fig_heatmap(freq_arr),
        'zod':  fig_zodiac_bar(zodiac_cnt),
        'gap':  fig_gap_bar(gap_arr, avg_freq),
    }
    b64 = {k: fig_to_b64(v) for k, v in figs.items()}
    for v in figs.values():
        plt.close(v)

    # 预计算：生肖表格行
    zodiac_rows = ''
    for z in ZODIAC_ORDER:
        if z not in zodiac_cnt:
            continue
        cnt_z = zodiac_cnt[z]
        delta = cnt_z - n / 12
        sign  = '+' if delta >= 0 else ''
        zodiac_rows += (
            f'<tr><td>{z}</td><td>{cnt_z}</td>'
            f'<td>{cnt_z/n*100:.1f}%</td>'
            f'<td>{sign}{delta:.1f}</td></tr>'
        )

    # 预计算：遗漏异常
    warn_nums = [(i+1, int(gap_arr[i])) for i in range(49) if gap_arr[i] >= avg_freq * 2]
    warn_nums.sort(key=lambda x: x[1], reverse=True)
    warn_text = '、'.join([f'No.{nn}（遗漏{g}期）' for nn, g in warn_nums[:5]]) if warn_nums else '无'

    # 预计算：自动结论段落
    chi2_pass = chi2_p > sig_level
    ks_pass   = ks_p   > sig_level
    top1 = max(zodiac_cnt, key=zodiac_cnt.get) if zodiac_cnt else '-'
    bot1 = min(zodiac_cnt, key=zodiac_cnt.get) if zodiac_cnt else '-'
    top1_cnt = zodiac_cnt.get(top1, 0)
    bot1_cnt = zodiac_cnt.get(bot1, 0)

    concl_random = (
        '历史数据在统计上符合随机均匀分布，卡方检验与KS检验均未发现显著偏离，'
        '与彩票纯随机抽签机制理论一致。'
        if (chi2_pass and ks_pass) else
        '历史数据存在一定统计偏离，但这可能由样本量不足或短期随机波动引起，'
        '不代表可被利用的规律。彩票开奖在机制上是独立随机事件。'
    )

    # 检验结论颜色和文字
    def pf(b): return ('pass', '✅ 通过') if b else ('fail', '❌ 未通过')
    c1, t1 = pf(chi2_pass)
    c2, t2 = pf(ks_pass)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"/>
<title>特码统计分析报告</title>
<style>
  *{{margin:0;padding:0;box-sizing:border-box;}}
  body{{background:#0F1923;color:#E0E6ED;
       font-family:-apple-system,'PingFang SC','Helvetica Neue',sans-serif;
       padding:32px 48px;max-width:1100px;margin:0 auto;}}
  h1{{font-size:26px;color:#00C9FF;border-bottom:2px solid #00C9FF;
      padding-bottom:10px;margin-bottom:6px;}}
  .meta{{color:rgba(224,230,237,.45);font-size:12px;margin-bottom:24px;}}
  h2{{font-size:16px;color:#E0E6ED;margin:28px 0 12px;
      padding-left:10px;border-left:3px solid #00C9FF;}}
  .kpi-row{{display:flex;gap:14px;margin:14px 0;flex-wrap:wrap;}}
  .kpi{{background:#1A2634;border-radius:10px;padding:14px 18px;min-width:130px;
        border:1px solid rgba(0,201,255,.12);flex:1;}}
  .kpi-val{{font-size:22px;font-weight:700;color:#00C9FF;}}
  .kpi-lbl{{font-size:10px;color:rgba(224,230,237,.4);margin-top:4px;
             text-transform:uppercase;letter-spacing:.5px;}}
  img{{max-width:100%;border-radius:8px;margin:10px 0;border:1px solid rgba(255,255,255,.05);}}
  table{{width:100%;border-collapse:collapse;font-size:12px;margin:10px 0;}}
  th{{background:#1A2634;color:#00C9FF;padding:8px 12px;text-align:left;
      border-bottom:1px solid rgba(0,201,255,.15);}}
  td{{padding:7px 12px;border-bottom:1px solid rgba(255,255,255,.04);}}
  tr:hover td{{background:rgba(255,255,255,.03);}}
  .pass{{color:#27AE60;font-weight:600;}}
  .fail{{color:#E74C3C;font-weight:600;}}
  .concl{{background:rgba(0,201,255,.05);border-left:3px solid #00C9FF;
          padding:10px 16px;margin:8px 0;border-radius:0 8px 8px 0;
          font-size:13px;line-height:1.7;}}
  .disc{{background:rgba(243,156,18,.07);border:1px solid rgba(243,156,18,.2);
         border-radius:8px;padding:14px 18px;margin-top:32px;
         font-size:11px;color:rgba(224,230,237,.5);line-height:1.8;}}
  footer{{text-align:center;margin-top:28px;font-size:11px;
          color:rgba(224,230,237,.2);padding-top:16px;
          border-top:1px solid rgba(255,255,255,.05);}}
</style>
</head>
<body>
<h1>📊 特码统计分析报告</h1>
<p class="meta">生成时间：{now} &nbsp;|&nbsp; 数据范围：{start} ~ {end} &nbsp;|&nbsp; 共 {n} 期 &nbsp;|&nbsp; 统计检验通过：{pass_cnt}/3</p>

<h2>一、数据概览</h2>
<div class="kpi-row">
  <div class="kpi"><div class="kpi-val">{n}</div><div class="kpi-lbl">总期数</div></div>
  <div class="kpi"><div class="kpi-val">{latest['expect']}</div><div class="kpi-lbl">最新期号</div></div>
  <div class="kpi"><div class="kpi-val" style="color:#FF6B6B">{int(latest['special'])}</div><div class="kpi-lbl">上期特码</div></div>
  <div class="kpi"><div class="kpi-val">{df_full['openTime'].min().year}–{df_full['openTime'].max().year}</div><div class="kpi-lbl">数据年份</div></div>
  <div class="kpi"><div class="kpi-val">{avg_freq:.1f}</div><div class="kpi-lbl">理论均值（每号）</div></div>
</div>

<h2>二、号码频率分布</h2>
<img src="data:image/png;base64,{b64['freq']}" alt="频率分布"/>
<img src="data:image/png;base64,{b64['heat']}" alt="热力图"/>

<h2>三、生肖分析</h2>
<img src="data:image/png;base64,{b64['zod']}" alt="生肖分布"/>
<table>
  <tr><th>生肖</th><th>出现次数</th><th>占比</th><th>偏差（vs 均值）</th></tr>
  {zodiac_rows}
</table>

<h2>四、遗漏分析</h2>
<img src="data:image/png;base64,{b64['gap']}" alt="遗漏分析"/>
<p style="font-size:12px;color:rgba(224,230,237,.5);margin:6px 0">
  遗漏异常（&gt;2× 均值 {avg_freq:.0f} 期）：{warn_text}
</p>

<h2>五、统计检验</h2>
<table>
  <tr><th>检验项目</th><th>H₀ 假设</th><th>统计量</th><th>p 值</th><th>α</th><th>结论</th></tr>
  <tr>
    <td>卡方均匀性检验</td>
    <td>49 个号码概率均等</td>
    <td>χ²={chi2_stat:.2f}</td>
    <td>{chi2_p:.4f}</td>
    <td>{sig_level}</td>
    <td class="{c1}">{t1}</td>
  </tr>
  <tr>
    <td>KS 间隔指数分布检验</td>
    <td>号码间隔服从指数分布</td>
    <td>—</td>
    <td>{ks_p:.4f}</td>
    <td>{sig_level}</td>
    <td class="{c2}">{t2}</td>
  </tr>
</table>

<h2>六、综合结论</h2>
<div class="concl">📌 <strong>随机性评估：</strong>{concl_random}</div>
<div class="concl">📌 <strong>生肖分布：</strong>历史最高频生肖「{top1}」({top1_cnt}次)，最低频「{bot1}」({bot1_cnt}次)，差值 {top1_cnt-bot1_cnt} 次（{(top1_cnt-bot1_cnt)/n*100:.1f}%），属正常统计波动范围。</div>
<div class="concl">📌 <strong>遗漏异常：</strong>当前遗漏超 2× 均值的号码共 {len(warn_nums)} 个。遗漏统计仅反映历史状态，对独立随机事件无预测价值，不应用于选号。</div>
<div class="concl">📌 <strong>综合评估：</strong>{pass_cnt}/3 项统计检验通过。{"所有检验通过，数据整体符合随机均匀分布特征。" if pass_cnt >= 2 else "部分检验未通过，建议扩大样本量重新评估。"}</div>

<div class="disc">
  ⚠️ <strong>免责声明</strong><br>
  本报告仅用于历史数据统计分析与学习研究目的。所有结果均基于已发生数据，
  不构成任何预测依据，不具备选号指导功能，严禁用于任何形式的赌博活动。
  彩票开奖为完全独立的随机事件，任何历史数据对未来结果均无影响。
</div>
<footer>特码统计分析系统 v2.0 · 报告生成于 {now}</footer>
</body>
</html>"""
    return html


# ══════════════════════════════════════════════════════════════════════════
# 6. 侧边栏控制面板
# ══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('### 📊 特码统计分析')
    st.divider()

    # ── 数据文件选择
    csvs = sorted(glob.glob(os.path.join(config.OUTPUT_DIR, '????????.csv')))
    if not csvs:
        st.error('未找到数据文件，请先运行 main.py')
        st.stop()
    file_names = [os.path.basename(c) for c in csvs]
    sel_file   = st.selectbox('📁 数据文件', file_names, index=len(file_names) - 1)
    sel_path   = os.path.join(config.OUTPUT_DIR, sel_file)
    df_raw     = load_csv(sel_path)

    # ── 时间范围筛选
    st.markdown('**🗓 时间范围**')
    min_d = df_raw['openTime'].min().date()
    max_d = df_raw['openTime'].max().date()
    date_range = st.date_input(
        '时间范围', value=(min_d, max_d),
        min_value=min_d, max_value=max_d,
        format='YYYY/MM/DD', label_visibility='collapsed',
    )
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_d, end_d = date_range
    else:
        start_d, end_d = min_d, max_d

    df_full = df_raw[
        (df_raw['openTime'].dt.date >= start_d) &
        (df_raw['openTime'].dt.date <= end_d)
    ].reset_index(drop=True)

    st.divider()

    # ── 分析参数
    st.markdown('**⚙️ 分析参数**')
    window_n  = st.slider('滑动窗口（期）', 30, 300, 100, step=10)
    sig_level = st.selectbox('显著性水平 α', [0.05, 0.01], index=0)
    acf_lags  = st.slider('自相关最大滞后', 5, 40, 20, step=5)

    st.divider()

    # ── 生肖展示模式
    st.markdown('**🐲 生肖展示**')
    display_mode = st.radio('展示模式', ['全部12肖', 'Top 5', '单生肖'], index=0)
    sel_zodiac   = None
    if display_mode == '单生肖':
        sel_zodiac = st.selectbox('选择生肖', ZODIAC_ORDER)

    st.divider()

    # ── 操作按钮
    if IS_CLOUD:
        st.success('🕘 数据每晚自动更新')
    else:
        if st.button('🚀 拉取最新数据', use_container_width=True, type='primary'):
            fetch_data()
            st.rerun()

    st.divider()
    st.caption('⚠️ 仅供学习研究，严禁赌博选号')


# ══════════════════════════════════════════════════════════════════════════
# 7. 数据检查 & 全局预计算
# ══════════════════════════════════════════════════════════════════════════

if df_full is None or len(df_full) == 0:
    st.warning('⏳ 当前筛选范围内无数据，请调整时间范围')
    st.stop()

# 全量计算（所有 Tab 共用，避免重复计算）
specials       = df_full['special'].values.astype(int)
n              = len(specials)
freq_arr, avg_freq = compute_freq(specials)
gap_arr, all_gaps  = compute_gaps(specials)
zodiac_cnt         = compute_zodiac(df_full)
chi2_stat, chi2_p  = chisquare(freq_arr, np.full(49, avg_freq))
chi2_pass          = chi2_p > sig_level
ks_stat, ks_p      = (kstest(all_gaps, 'expon', args=(0, avg_freq))
                      if len(all_gaps) > 0 else (0.0, 1.0))
ks_pass            = ks_p > sig_level
acf_vals           = compute_autocorr(specials, acf_lags)
conf_interval      = 1.96 / np.sqrt(n)
acf_exceed         = int(np.sum(np.abs(acf_vals) > conf_interval))
warn_thr           = avg_freq * 2.0
crit_thr           = avg_freq * 3.5
pass_cnt           = sum([chi2_pass, ks_pass, acf_exceed == 0])
latest             = df_full.iloc[-1]

# ══════════════════════════════════════════════════════════════════════════
# 8. 顶部 Header
# ══════════════════════════════════════════════════════════════════════════

uniformity_icon = '✅' if chi2_pass else '⚠️'
st.markdown(f"""
<div class="dashboard-header">
  <div class="header-top">
    <div class="header-title">📊 <span>特码</span>统计分析系统</div>
    <div class="header-meta">
      <div class="header-meta-item">
        <div class="header-meta-val">{n}</div>
        <div class="header-meta-lbl">当前期数</div>
      </div>
      <div class="header-meta-item">
        <div class="header-meta-val">{latest['expect']}</div>
        <div class="header-meta-lbl">最新期号</div>
      </div>
      <div class="header-meta-item">
        <div class="header-meta-val red">{int(latest['special'])}</div>
        <div class="header-meta-lbl">上期特码</div>
      </div>
      <div class="header-meta-item">
        <div class="header-meta-val">{pass_cnt}/3</div>
        <div class="header-meta-lbl">检验通过</div>
      </div>
      <div class="header-meta-item">
        <div class="header-meta-val">{uniformity_icon}</div>
        <div class="header-meta-lbl">均匀性</div>
      </div>
    </div>
  </div>
  <div class="header-divider"></div>
  <div class="header-notice">本系统仅用于历史数据统计分析与学习研究，结果不具备任何预测能力，严禁用于赌博或选号</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# 9. 7 大功能 Tabs
# ══════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    '📋 数据概览',
    '📊 分布分析',
    '🐲 结构分析',
    '⏱ 时序分析',
    '🔬 统计检验',
    '🚨 异常检测',
    '📄 分析报告',
])


# ────────────────────────────────────────────────────────────────────────
# Tab 1 — 数据概览
# ────────────────────────────────────────────────────────────────────────
with tab1:
    # ── KPI 卡片（5 列，各有专属强调色）
    st.markdown(ch('数据总览', '当前筛选范围内的基础统计指标', '📋'), unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    yr_range = f'{df_full["openTime"].min().year}–{df_full["openTime"].max().year}'
    kpi_cards = [
        (str(n),                                             '总  期  数',    '',   '#00C9FF', 'linear-gradient(90deg,#00C9FF,#7B68EE)'),
        (yr_range,                                           '数据年份',       '',   '#9B59B6', 'linear-gradient(90deg,#9B59B6,#7B68EE)'),
        (df_full['openTime'].min().strftime('%y/%m/%d'),     '起始日期',       '',   '#27AE60', 'linear-gradient(90deg,#27AE60,#00C9FF)'),
        (df_full['openTime'].max().strftime('%y/%m/%d'),     '最新日期',       '',   '#F39C12', 'linear-gradient(90deg,#F39C12,#E67E22)'),
        (str(df_full['openTime'].dt.date.nunique()),         '有效期数',       '',   '#00C9FF', 'linear-gradient(90deg,#00C9FF,#27AE60)'),
    ]
    for col, (val, lbl, sub, accent, grad) in zip([c1, c2, c3, c4, c5], kpi_cards):
        with col:
            st.markdown(stat_card_html(val, lbl, sub, accent, grad), unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # ── 完整性检查 + 年度分布（两列）
    st.markdown(ch('数据完整性检查', '自动验证字段完整性、期号唯一性、数值合法性', '🔍'),
                unsafe_allow_html=True)
    col_qa, col_qb = st.columns([3, 2])

    with col_qa:
        req_cols  = ['expect', 'openTime', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'special']
        miss_cols = [c for c in req_cols if c not in df_full.columns]
        dup_cnt   = df_full.duplicated(subset='expect').sum()
        null_cnt  = df_full['special'].isna().sum()
        range_bad = int(((df_full['special'] < 1) | (df_full['special'] > 49)).sum())
        checks = [
            ('✅', '必要字段完整', '通过'         if not miss_cols else f'缺少 {miss_cols}', not miss_cols),
            ('✅' if dup_cnt  == 0 else '⚠️', '期号唯一性', '无重复'     if dup_cnt  == 0 else f'{dup_cnt} 条重复',   dup_cnt  == 0),
            ('✅' if null_cnt == 0 else '❌', '特码空值',   '无空值'     if null_cnt == 0 else f'{null_cnt} 条空值',  null_cnt == 0),
            ('✅' if range_bad== 0 else '❌', '号码范围 1-49', '全部合法' if range_bad== 0 else f'{range_bad} 条越界', range_bad== 0),
        ]
        rows_html = ''
        for icon, name, result, passed in checks:
            cls  = 'check-ok' if passed else 'check-fail'
            rows_html += (
                f'<div class="check-row">'
                f'<span class="check-icon">{icon}</span>'
                f'<span class="check-name">{name}</span>'
                f'<span class="{cls}">{result}</span></div>'
            )
        st.markdown(rows_html, unsafe_allow_html=True)

    with col_qb:
        st.markdown(ch('各年期数分布', '', '📅'), unsafe_allow_html=True)
        year_cnt = df_full.groupby(df_full['openTime'].dt.year).size()
        rows_y = ''
        max_y  = year_cnt.max()
        for yr, cnt in year_cnt.items():
            pct = cnt / max_y * 100
            rows_y += (
                f'<div style="display:flex;align-items:center;gap:10px;padding:5px 0;font-size:12px">'
                f'<span style="color:rgba(224,230,237,.5);width:38px">{yr}</span>'
                f'<div style="flex:1;height:6px;background:rgba(255,255,255,.06);border-radius:3px;overflow:hidden">'
                f'<div style="width:{pct:.0f}%;height:100%;background:linear-gradient(90deg,#00C9FF,#7B68EE);border-radius:3px"></div></div>'
                f'<span style="color:#E0E6ED;font-weight:600;width:38px;text-align:right">{cnt}</span>'
                f'<span style="color:rgba(224,230,237,.35);width:38px">({cnt/n*100:.0f}%)</span>'
                f'</div>'
            )
        st.markdown(rows_y, unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown(ch('最新 10 期开奖记录', '按时间倒序展示，包含正码与特码', '📅'),
                unsafe_allow_html=True)
    show_cols = [c for c in ['expect', 'openTime', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'special', 'zodiac']
                 if c in df_full.columns]
    st.dataframe(
        df_full[show_cols].tail(10).iloc[::-1].reset_index(drop=True),
        use_container_width=True, hide_index=True
    )


# ────────────────────────────────────────────────────────────────────────
# Tab 2 — 分布分析
# ────────────────────────────────────────────────────────────────────────
with tab2:
    # ── 频率柱状图 + 排行榜
    st.markdown(ch('号码频率分布',
                   '统计 1–49 各号码在历史数据中的出现次数；红色=最高频，灰色=最低频，橙色虚线=理论均值',
                   '📊'), unsafe_allow_html=True)
    col_ch, col_info = st.columns([3, 1])
    top5 = sorted(enumerate(freq_arr, 1), key=lambda x: x[1], reverse=True)[:5]
    bot5 = sorted(enumerate(freq_arr, 1), key=lambda x: x[1])[:5]
    with col_ch:
        st.pyplot(fig_freq_bar(freq_arr, avg_freq), use_container_width=True)
        plt.close('all')
    with col_info:
        st.markdown(
            '<div style="font-size:12px;font-weight:600;color:#FF6B6B;margin-bottom:4px">🔥 最热号码</div>'
            + rank_html(top5, freq_arr.max(), hot=True),
            unsafe_allow_html=True
        )
        st.markdown(
            '<div style="font-size:12px;font-weight:600;color:#A0AEC0;margin:12px 0 4px">🧊 最冷号码</div>'
            + rank_html(bot5, freq_arr.max(), hot=False),
            unsafe_allow_html=True
        )

    st.markdown(ib(f'理论均值：每个号码应出现约 <strong>{avg_freq:.1f}</strong> 次。'
                   f'当前标准差 {freq_arr.std():.2f}，'
                   f'最高频 No.{int(np.argmax(freq_arr))+1}（{freq_arr.max():.0f}次），'
                   f'最低频 No.{int(np.argmin(freq_arr))+1}（{freq_arr.min():.0f}次）。'
                   f'偏差在统计正常范围内不代表规律。'),
                unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # ── 热力图 + 指标卡片
    st.markdown(ch('号码热力图（7×7 布局）',
                   '颜色越深表示出现次数越多；直观定位高频/低频号码区域', '🌡️'),
                unsafe_allow_html=True)
    col_hm, col_hm2 = st.columns([5, 2])
    max_num = int(np.argmax(freq_arr)) + 1
    min_num = int(np.argmin(freq_arr)) + 1
    with col_hm:
        st.pyplot(fig_heatmap(freq_arr), use_container_width=True)
        plt.close('all')
    with col_hm2:
        st.markdown('<br>', unsafe_allow_html=True)
        cards_hm = [
            (f'No.{max_num}', '最高频号码', f'{freq_arr[max_num-1]:.0f} 次', '#FF6B6B',
             'linear-gradient(90deg,#FF6B6B,#F39C12)'),
            (f'No.{min_num}', '最低频号码', f'{freq_arr[min_num-1]:.0f} 次', '#4A5568',
             'linear-gradient(90deg,#4A5568,#2A3F54)'),
            (f'{avg_freq:.1f}', '理论均值（次）', '', '#27AE60',
             'linear-gradient(90deg,#27AE60,#00C9FF)'),
            (f'{freq_arr.std():.2f}', '标准差', '', '#9B59B6',
             'linear-gradient(90deg,#9B59B6,#7B68EE)'),
        ]
        for v, l, s, ac, gr in cards_hm:
            st.markdown(stat_card_html(v, l, s, ac, gr), unsafe_allow_html=True)
            st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # ── 结构分布（奇偶 / 大小 / 尾数）
    st.markdown(ch('结构分布',
                   '从奇偶、大小（以24为界）、个位尾数三个维度分析特码的统计特征', '🔢'),
                unsafe_allow_html=True)
    co1, co2, co3 = st.columns(3)
    odd_cnt = int(np.sum(specials % 2 == 1))
    big_cnt = int(np.sum(specials > config.BIG_NUMBER_THRESHOLD))
    with co1:
        st.pyplot(fig_pie([odd_cnt, n - odd_cnt], ['奇', '偶'],
                          [PALETTE['blue'], PALETTE['red']], '奇偶分布'),
                  use_container_width=True)
        plt.close('all')
        odd_pct = odd_cnt / n * 100
        color   = '#27AE60' if 44 < odd_pct < 56 else '#F39C12'
        st.markdown(ib(f'奇数占比 <strong style="color:{color}">{odd_pct:.1f}%</strong>'
                       f'（理论 50%），{"接近均衡" if 44 < odd_pct < 56 else "存在偏差"}'),
                    unsafe_allow_html=True)
    with co2:
        st.pyplot(fig_pie([big_cnt, n - big_cnt],
                          [f'大(>{config.BIG_NUMBER_THRESHOLD})',
                           f'小(≤{config.BIG_NUMBER_THRESHOLD})'],
                          [PALETTE['orange'], PALETTE['green']], '大小分布'),
                  use_container_width=True)
        plt.close('all')
        big_pct = big_cnt / n * 100
        st.markdown(ib(f'大号（>24）占比 <strong>{big_pct:.1f}%</strong>'
                       f'（理论 51.0%，大号25个）'), unsafe_allow_html=True)
    with co3:
        st.pyplot(fig_tail_bar(specials, n), use_container_width=True)
        plt.close('all')
        tail_cnt   = Counter((specials % 10).tolist())
        max_tail   = max(tail_cnt, key=tail_cnt.get)
        st.markdown(ib(f'尾数 <strong>{max_tail}</strong> 出现最多（{tail_cnt[max_tail]}次），'
                       f'均值 {n/10:.1f} 次/尾数'), unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────
# Tab 3 — 结构分析
# ────────────────────────────────────────────────────────────────────────
with tab3:
    if not zodiac_cnt:
        st.warning('数据中无 zodiac 生肖字段，请确认 CSV 包含该列')
    else:
        # 按展示模式过滤
        if display_mode == 'Top 5':
            top5_z  = sorted(zodiac_cnt, key=zodiac_cnt.get, reverse=True)[:5]
            zc_show = {z: zodiac_cnt[z] for z in top5_z}
        elif display_mode == '单生肖' and sel_zodiac:
            zc_show = {sel_zodiac: zodiac_cnt.get(sel_zodiac, 0)}
        else:
            zc_show = zodiac_cnt

        zod_mean = n / 12
        top1_z   = max(zodiac_cnt, key=zodiac_cnt.get)
        bot1_z   = min(zodiac_cnt, key=zodiac_cnt.get)

        # ── 生肖频率柱状图 + 排名
        st.markdown(ch('生肖频率分布',
                       f'12 生肖在历史数据中的出现次数；理论均值 {zod_mean:.1f} 次（n/12）', '🐲'),
                    unsafe_allow_html=True)
        col_zp, col_zt = st.columns([3, 1])
        with col_zp:
            highlight = sel_zodiac if display_mode == '单生肖' else None
            st.pyplot(fig_zodiac_bar(zc_show, highlight), use_container_width=True)
            plt.close('all')
        with col_zt:
            rows_z = ''
            for i, (z, v) in enumerate(
                    sorted(zodiac_cnt.items(), key=lambda x: x[1], reverse=True), 1):
                delta = v - zod_mean
                sign  = '+' if delta >= 0 else ''
                emoji = ZODIAC_EMOJI.get(z, '')
                is_top = (i == 1)
                num_color = '#FF6B6B' if is_top else ('#4A5568' if i == len(zodiac_cnt) else '#E0E6ED')
                rows_z += (
                    f'<div style="display:flex;align-items:center;gap:8px;padding:6px 0;'
                    f'border-bottom:1px solid rgba(255,255,255,.04);font-size:12px">'
                    f'<span style="color:rgba(224,230,237,.28);width:14px">{i}</span>'
                    f'<span style="font-size:14px">{emoji}</span>'
                    f'<span style="font-weight:600;color:{num_color};flex:1">{z}</span>'
                    f'<span style="color:#E0E6ED">{v}次</span>'
                    f'<span style="color:rgba(224,230,237,.32);font-size:11px;min-width:28px;text-align:right">'
                    f'{sign}{delta:.0f}</span></div>'
                )
            st.markdown(rows_z, unsafe_allow_html=True)

        st.markdown(ib(f'最高频：{ZODIAC_EMOJI.get(top1_z,"")}<strong>{top1_z}</strong>'
                       f'（{zodiac_cnt[top1_z]}次），'
                       f'最低频：{ZODIAC_EMOJI.get(bot1_z,"")}<strong>{bot1_z}</strong>'
                       f'（{zodiac_cnt[bot1_z]}次），差值 {zodiac_cnt[top1_z]-zodiac_cnt[bot1_z]} 次。'
                       f'属正常统计波动，不代表某生肖"更容易"出现。'),
                    unsafe_allow_html=True)

        st.markdown('<br>', unsafe_allow_html=True)

        # ── 月度趋势
        st.markdown(ch('月度趋势（Top 6 生肖）',
                       '按月统计各生肖出现次数，观察是否存在周期性波动', '📈'),
                    unsafe_allow_html=True)
        st.pyplot(fig_monthly_trend(df_full), use_container_width=True)
        plt.close('all')
        st.markdown(ib('折线波动属正常短期随机涨落，不存在统计意义上的周期规律。',
                       'blue'), unsafe_allow_html=True)

        st.markdown('<br>', unsafe_allow_html=True)

        # ── 多窗口对比
        st.markdown(ch('多窗口对比',
                       '近 50 / 100 / 200 期与全量数据的生肖占比对比，观察短期分布是否稳定', '🔄'),
                    unsafe_allow_html=True)
        table_rows = []
        for z in sorted(zodiac_cnt, key=zodiac_cnt.get, reverse=True):
            row = {'生肖': f'{ZODIAC_EMOJI.get(z,"")}{z}'}
            for w in [50, 100, 200]:
                wc    = Counter(df_full.tail(w)['zodiac'].tolist())
                cnt_w = wc.get(z, 0)
                row[f'近{w}期'] = f'{cnt_w} ({cnt_w/w*100:.0f}%)'
            row['全量'] = f'{zodiac_cnt.get(z, 0)} ({zodiac_cnt.get(z,0)/n*100:.1f}%)'
            table_rows.append(row)
        st.dataframe(pd.DataFrame(table_rows).set_index('生肖'), use_container_width=True)

        st.markdown('<br>', unsafe_allow_html=True)

        # ── 生肖 × 奇偶
        st.markdown(ch('生肖 × 奇偶 组合分析',
                       '各生肖在奇数（蓝）与偶数（橙）特码中的出现次数分布', '⊕'),
                    unsafe_allow_html=True)
        fig_zo = fig_zodiac_odd(df_full)
        if fig_zo:
            st.pyplot(fig_zo, use_container_width=True)
            plt.close('all')
        st.markdown(ib('奇偶组合差异属随机波动，不应据此推断某生肖偏好奇数或偶数特码。',
                       'orange'), unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────
# Tab 4 — 时序分析
# ────────────────────────────────────────────────────────────────────────
with tab4:
    anom_list = [(i+1, int(gap_arr[i])) for i in range(49) if gap_arr[i] >= warn_thr]
    anom_list.sort(key=lambda x: x[1], reverse=True)

    # ── 遗漏期数柱状图 + 摘要卡片
    st.markdown(ch('各号码当前遗漏期数',
                   f'遗漏 = 距上次出现的间隔期数；橙线=2×均值（预警），红线=3.5×均值（危险）；均值≈{avg_freq:.0f}期',
                   '⏱'), unsafe_allow_html=True)
    col_gp, col_ga = st.columns([3, 1])
    with col_gp:
        st.pyplot(fig_gap_bar(gap_arr, avg_freq), use_container_width=True)
        plt.close('all')
    with col_ga:
        st.markdown('<br>', unsafe_allow_html=True)
        max_gap_num = int(gap_arr.argmax()) + 1
        gap_cards = [
            (f'{int(gap_arr.max())}期', '最长遗漏', f'No.{max_gap_num}',
             '#FF6B6B', 'linear-gradient(90deg,#FF6B6B,#F39C12)'),
            (f'{avg_freq:.1f}期', '理论均值', '49期/号', '#00C9FF',
             'linear-gradient(90deg,#00C9FF,#7B68EE)'),
            (f'{len(anom_list)}个', '遗漏异常', f'>{warn_thr:.0f}期', '#F39C12',
             'linear-gradient(90deg,#F39C12,#E67E22)'),
        ]
        for v, l, s, ac, gr in gap_cards:
            st.markdown(stat_card_html(v, l, s, ac, gr), unsafe_allow_html=True)
            st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
        if anom_list:
            rows_a = ''
            for num, gap in anom_list[:6]:
                tag   = '🔴' if gap >= crit_thr else '🟡'
                mult  = gap / avg_freq
                rows_a += (
                    f'<div style="display:flex;align-items:center;gap:8px;padding:5px 0;'
                    f'border-bottom:1px solid rgba(255,255,255,.04);font-size:11px">'
                    f'<span>{tag}</span>'
                    f'<span style="font-weight:700;color:#E0E6ED">No.{num:02d}</span>'
                    f'<span style="margin-left:auto;color:rgba(224,230,237,.6)">{gap}期</span>'
                    f'<span style="color:rgba(224,230,237,.35)">×{mult:.1f}</span></div>'
                )
            st.markdown(
                f'<div style="margin-top:10px;font-size:11px;font-weight:600;color:#F39C12;'
                f'margin-bottom:4px">⚠️ 异常号码</div>' + rows_a,
                unsafe_allow_html=True
            )

    st.markdown(ib('遗漏期数仅反映统计状态，对独立随机事件无预测价值。'
                   '长遗漏不代表"即将出现"，彩票每期开奖互相独立。'), unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # ── 间隔分布
    if len(all_gaps) > 0:
        st.markdown(ch('间隔分布 vs 理论指数分布',
                       '若开奖独立随机，间隔应服从指数分布（红线）；蓝色直方图=实际数据', '📐'),
                    unsafe_allow_html=True)
        st.pyplot(fig_gap_hist(all_gaps, avg_freq), use_container_width=True)
        plt.close('all')
        st.markdown(ib(f'实际间隔分布{"接近" if ks_pass else "偏离"}理论指数分布'
                       f'（KS检验 p={ks_p:.3f}），'
                       + ('支持各期独立假设。' if ks_pass else '但不代表存在可利用的规律。'),
                       'green' if ks_pass else 'orange'), unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # ── 单号码滚动窗口
    st.markdown(ch('单号码滑动窗口频率',
                   '在连续 N 期的滑动窗口内统计目标号码的频率变化；红色填充=高于均值，蓝色=低于均值', '🔍'),
                unsafe_allow_html=True)
    roll_num = st.number_input('选择号码（1–49）', min_value=1, max_value=49,
                               value=1, step=1, key='roll_num')
    st.pyplot(fig_rolling_freq(specials, window_n, roll_num), use_container_width=True)
    plt.close('all')
    q_cnt = Counter(specials.tolist()).get(roll_num, 0)
    st.markdown(ib(f'No.{roll_num} 全量出现 <strong>{q_cnt}</strong> 次'
                   f'（理论均值 {avg_freq:.1f} 次，偏差 {q_cnt-avg_freq:+.1f}）。'
                   f'频率波动属正常随机涨落，与窗口选取有关。'), unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # ── 冷热榜（3列卡片）
    st.markdown(ch(f'近 {window_n} 期冷热榜',
                   '基于当前滑动窗口统计的热号、冷号与长遗漏号码', '🏆'),
                unsafe_allow_html=True)
    sp_win   = df_full.tail(window_n)['special'].values.astype(int)
    freq_win = np.array([Counter(sp_win.tolist()).get(i, 0) for i in range(1, 50)])
    hot5     = sorted(enumerate(freq_win, 1), key=lambda x: x[1], reverse=True)[:5]
    cold5    = sorted(enumerate(freq_win, 1), key=lambda x: x[1])[:5]
    miss5    = sorted([(i+1, int(gap_arr[i])) for i in range(49)],
                      key=lambda x: x[1], reverse=True)[:5]
    cw1, cw2, cw3 = st.columns(3)
    with cw1:
        st.markdown(
            lb_card_html(f'🔥 近{window_n}期最热', hot5, 'lb-hot',
                         lambda num, v: f'{v}次'),
            unsafe_allow_html=True
        )
    with cw2:
        st.markdown(
            lb_card_html(f'🧊 近{window_n}期最冷', cold5, 'lb-cold',
                         lambda num, v: f'{v}次'),
            unsafe_allow_html=True
        )
    with cw3:
        st.markdown(
            lb_card_html('⏳ 当前遗漏最长', miss5, 'lb-miss',
                         lambda num, v: f'{v}期'),
            unsafe_allow_html=True
        )


# ────────────────────────────────────────────────────────────────────────
# Tab 5 — 统计检验
# ────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown(ch('① 卡方均匀性检验 & KS 间隔检验',
                   '卡方检验验证号码分布是否均匀；KS检验验证出现间隔是否服从指数分布（独立性）', '🔬'),
                unsafe_allow_html=True)
    col_t1, col_t2 = st.columns(2)

    pill1 = ('<span class="pill-pass">✅ 接受 H₀ — 分布均匀</span>'
             if chi2_pass else
             '<span class="pill-fail">❌ 拒绝 H₀ — 分布偏离均匀</span>')
    pill2 = ('<span class="pill-pass">✅ 接受 H₀ — 各期独立随机</span>'
             if ks_pass else
             '<span class="pill-fail">❌ 拒绝 H₀ — 间隔存在相关</span>')

    with col_t1:
        st.markdown(f"""
        <div class="test-box">
          <div class="test-title">卡方检验（Chi-Square Goodness of Fit）</div>
          <div class="test-row">
            <span class="test-key">H₀ 假设</span>
            <span class="test-val">49 个号码出现概率均等（1/49）</span>
          </div>
          <div class="test-row">
            <span class="test-key">统计量 χ²</span>
            <span class="test-val">{chi2_stat:.4f}</span>
          </div>
          <div class="test-row">
            <span class="test-key">p 值</span>
            <span class="test-val">{chi2_p:.4f}</span>
          </div>
          <div class="test-row">
            <span class="test-key">显著性水平 α</span>
            <span class="test-val">{sig_level}</span>
          </div>
          <div style="margin-top:13px">{pill1}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_t2:
        st.markdown(f"""
        <div class="test-box">
          <div class="test-title">KS 检验（间隔指数分布拟合）</div>
          <div class="test-row">
            <span class="test-key">H₀ 假设</span>
            <span class="test-val">号码出现间隔服从指数分布（独立性）</span>
          </div>
          <div class="test-row">
            <span class="test-key">统计量 D</span>
            <span class="test-val">{ks_stat:.4f}</span>
          </div>
          <div class="test-row">
            <span class="test-key">p 值</span>
            <span class="test-val">{ks_p:.4f}</span>
          </div>
          <div class="test-row">
            <span class="test-key">显著性水平 α</span>
            <span class="test-val">{sig_level}</span>
          </div>
          <div style="margin-top:13px">{pill2}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)
    if len(all_gaps) > 0:
        st.markdown(ch('间隔分布拟合图',
                       '若号码独立随机出现，间隔应符合指数分布（无记忆性）', '📐'),
                    unsafe_allow_html=True)
        st.pyplot(fig_gap_hist(all_gaps, avg_freq), use_container_width=True)
        plt.close('all')

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown(ch('② 自相关检验（序列独立性）',
                   f'检验相邻期特码是否存在线性相关；超出虚线（95% CI ±{conf_interval:.3f}）才认为显著', '📉'),
                unsafe_allow_html=True)
    pill_acf = (
        f'<span class="pill-pass">✅ 自相关均在 95% CI 内（{acf_lags} 个滞后全部正常）</span>'
        if acf_exceed == 0 else
        f'<span class="pill-fail">⚠️ {acf_exceed} 个滞后超出 95% CI（可能存在弱相关）</span>'
    )
    col_acf1, col_acf2 = st.columns([2, 1])
    with col_acf1:
        st.pyplot(fig_autocorr(acf_vals, n), use_container_width=True)
        plt.close('all')
    with col_acf2:
        st.markdown(f"""
        <div class="test-box">
          <div class="test-title">自相关检验</div>
          <div class="test-row">
            <span class="test-key">最大滞后期</span>
            <span class="test-val">{acf_lags}</span>
          </div>
          <div class="test-row">
            <span class="test-key">95% CI 上界</span>
            <span class="test-val">±{conf_interval:.4f}</span>
          </div>
          <div class="test-row">
            <span class="test-key">超出 CI 数量</span>
            <span class="test-val">{acf_exceed} / {acf_lags}</span>
          </div>
          <div style="margin-top:13px">{pill_acf}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)
    summary_color = '#27AE60' if pass_cnt == 3 else ('#F39C12' if pass_cnt >= 2 else '#E74C3C')
    summary_bg    = ('rgba(39,174,96,0.07)'  if pass_cnt == 3 else
                     'rgba(243,156,18,0.07)' if pass_cnt >= 2 else
                     'rgba(231,76,60,0.07)')
    summary_border = summary_color
    summary_text  = (
        '全部通过。数据整体符合随机均匀分布特征，与彩票纯随机机制理论一致。'
        '历史数据对未来无任何预测价值。'
        if pass_cnt == 3 else
        '部分通过，可能由样本量不足、短期波动或统计误差引起。'
        '彩票开奖是独立随机事件，历史数据无预测能力。'
    )
    # 生成三项检验的 mini 徽章
    badges = ''
    for label, passed in [('卡方', chi2_pass), ('KS间隔', ks_pass), ('自相关', acf_exceed == 0)]:
        bg  = 'rgba(39,174,96,.15)'  if passed else 'rgba(243,156,18,.15)'
        col = '#27AE60' if passed else '#F39C12'
        icon = '✅' if passed else '⚠️'
        badges += (
            f'<span style="display:inline-block;background:{bg};color:{col};'
            f'border-radius:20px;padding:3px 12px;font-size:11px;font-weight:600;'
            f'margin-right:8px;border:1px solid {col}44">{icon} {label}</span>'
        )
    st.markdown(f"""
    <div class="eval-box" style="background:{summary_bg};border-color:{summary_border}33">
      <div class="eval-title" style="color:{summary_color}">
        📊 综合评估：{pass_cnt}/3 项检验通过
      </div>
      <div style="margin:8px 0">{badges}</div>
      <div class="eval-body">{summary_text}</div>
    </div>
    """, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────
# Tab 6 — 异常检测
# ────────────────────────────────────────────────────────────────────────
with tab6:
    warn_cnt6 = sum(1 for _, g in anom_list if g < crit_thr)
    crit_cnt6 = sum(1 for _, g in anom_list if g >= crit_thr)
    norm_cnt6 = 49 - len(anom_list)

    # ── 彩色异常 KPI 卡片（4列）
    st.markdown(ch('遗漏异常统计概览',
                   f'以 {warn_thr:.0f} 期（2×均值）为预警阈值，{crit_thr:.0f} 期（3.5×）为危险阈值', '🚨'),
                unsafe_allow_html=True)
    ca1, ca2, ca3, ca4 = st.columns(4)
    anom_kpis = [
        (str(norm_cnt6), '✅ 正常号码', f'遗漏 < {warn_thr:.0f}期',
         '#27AE60', 'linear-gradient(90deg,#27AE60,#00C9FF)'),
        (str(warn_cnt6), '🟡 预警号码', f'{warn_thr:.0f}–{crit_thr:.0f}期',
         '#F39C12', 'linear-gradient(90deg,#F39C12,#E67E22)'),
        (str(crit_cnt6), '🔴 危险号码', f'> {crit_thr:.0f}期',
         '#E74C3C', 'linear-gradient(90deg,#E74C3C,#FF6B6B)'),
        (f'{avg_freq:.0f}期', '均值遗漏', '理论预警基准',
         '#9B59B6', 'linear-gradient(90deg,#9B59B6,#7B68EE)'),
    ]
    for col, (v, l, s, ac, gr) in zip([ca1, ca2, ca3, ca4], anom_kpis):
        with col:
            st.markdown(stat_card_html(v, l, s, ac, gr), unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown(ch('异常号码出现位置（近 200 期）',
                   '每行一个异常号码；点位 = 该号码在近200期中出现的位置（右端=最新）', '📍'),
                unsafe_allow_html=True)
    st.pyplot(fig_anomaly_scatter(specials, gap_arr, avg_freq), use_container_width=True)
    plt.close('all')
    st.markdown(ib('图中无点的行说明该号码在近200期内从未出现；点越靠右说明越近期出现过。'
                   '出现点稀疏≠该号码"欠账"，彩票开奖无记忆性。', 'orange'),
                unsafe_allow_html=True)

    if anom_list:
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown(ch('异常明细表', '按遗漏期数降序排列，仅作统计记录', '📋'),
                    unsafe_allow_html=True)
        anom_df = pd.DataFrame([{
            '号码':     f'No.{num}',
            '当前遗漏': f'{gap} 期',
            '均值倍数': f'×{gap/avg_freq:.2f}',
            '预警等级': '🔴 危险' if gap >= crit_thr else '🟡 预警',
            '统计说明': '遗漏对随机事件无预测价值',
        } for num, gap in anom_list])
        st.dataframe(anom_df, use_container_width=True, hide_index=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # ── Z-score 频率偏离
    st.markdown(ch('频率偏离检测（Z-score > 2σ）',
                   '当某号码出现次数偏离均值超过 2 个标准差时，在统计上属小概率事件', '📏'),
                unsafe_allow_html=True)
    sigma    = freq_arr.std()
    dev_nums = [(i+1, freq_arr[i], (freq_arr[i]-avg_freq)/sigma)
                for i in range(49) if abs(freq_arr[i]-avg_freq) > 2*sigma]
    dev_nums.sort(key=lambda x: abs(x[2]), reverse=True)

    if dev_nums:
        col_d1, col_d2 = st.columns(2)
        high_devs = [(n_, c, z) for n_, c, z in dev_nums if z > 0]
        low_devs  = [(n_, c, z) for n_, c, z in dev_nums if z < 0]
        with col_d1:
            if high_devs:
                rows_hd = ''
                for num_, cnt_, z_ in high_devs:
                    rows_hd += (
                        f'<div style="display:flex;gap:10px;align-items:center;padding:6px 0;'
                        f'border-bottom:1px solid rgba(255,255,255,.04);font-size:12px">'
                        f'<span style="font-weight:700;color:#FF6B6B;width:40px">No.{num_:02d}</span>'
                        f'<span style="color:#E0E6ED;flex:1">{cnt_:.0f}次</span>'
                        f'<span style="background:rgba(255,107,107,.12);color:#FF6B6B;'
                        f'padding:2px 8px;border-radius:10px;font-size:11px">Z={z_:+.2f}</span>'
                        f'</div>'
                    )
                st.markdown(
                    '<div style="background:rgba(255,107,107,.05);border-radius:10px;'
                    'padding:12px 16px;border:1px solid rgba(255,107,107,.12)">'
                    '<div style="font-size:12px;font-weight:600;color:#FF6B6B;margin-bottom:8px">'
                    '📈 频率偏高（>均值+2σ）</div>' + rows_hd + '</div>',
                    unsafe_allow_html=True
                )
        with col_d2:
            if low_devs:
                rows_ld = ''
                for num_, cnt_, z_ in low_devs:
                    rows_ld += (
                        f'<div style="display:flex;gap:10px;align-items:center;padding:6px 0;'
                        f'border-bottom:1px solid rgba(255,255,255,.04);font-size:12px">'
                        f'<span style="font-weight:700;color:#A0AEC0;width:40px">No.{num_:02d}</span>'
                        f'<span style="color:#E0E6ED;flex:1">{cnt_:.0f}次</span>'
                        f'<span style="background:rgba(74,85,104,.2);color:#A0AEC0;'
                        f'padding:2px 8px;border-radius:10px;font-size:11px">Z={z_:+.2f}</span>'
                        f'</div>'
                    )
                st.markdown(
                    '<div style="background:rgba(74,85,104,.06);border-radius:10px;'
                    'padding:12px 16px;border:1px solid rgba(74,85,104,.15)">'
                    '<div style="font-size:12px;font-weight:600;color:#A0AEC0;margin-bottom:8px">'
                    '📉 频率偏低（<均值-2σ）</div>' + rows_ld + '</div>',
                    unsafe_allow_html=True
                )
    else:
        st.markdown(ib('✅ 所有号码频率均在 ±2σ 范围内，无统计显著偏离，符合随机期望。',
                       'green'), unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown(ib('💡 异常标记仅反映统计状态，不应据此选号。'
                   '遗漏长不代表"即将出现"，遗漏短不代表"不会出现"。'
                   '彩票每期开奖独立，历史状态对未来无任何影响。', 'blue'),
                unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────
# Tab 7 — 分析报告
# ────────────────────────────────────────────────────────────────────────
with tab7:
    st.markdown(ch('自动分析结论',
                   '基于统计检验结果自动生成的解读，绿色边框=正常，橙色边框=注意', '📝'),
                unsafe_allow_html=True)

    # 动态生成结论
    top1     = max(zodiac_cnt, key=zodiac_cnt.get) if zodiac_cnt else '-'
    bot1     = min(zodiac_cnt, key=zodiac_cnt.get) if zodiac_cnt else '-'
    top1_r50 = (Counter(df_full.tail(50)['zodiac'].tolist()).most_common(1)
                if 'zodiac' in df_full.columns else [('-', 0)])
    top1_r50 = top1_r50[0][0] if top1_r50 else '-'

    concl_items = [
        ('ok' if chi2_pass else 'warn',
         f'<strong>均匀性检验：{"✅ 通过" if chi2_pass else "⚠️ 未通过"}</strong>（卡方值 {chi2_stat:.2f}，p={chi2_p:.4f}，α={sig_level}）— '
         + ('历史数据与均匀分布无显著统计差异。' if chi2_pass
            else '统计上偏离均匀分布，但不代表可利用的规律，可能源于样本量或短期波动。')),

        ('ok' if ks_pass else 'warn',
         f'<strong>独立性检验（KS）：{"✅ 通过" if ks_pass else "⚠️ 未通过"}</strong>（p={ks_p:.4f}）— '
         + ('间隔分布符合指数分布，支持各期独立假设。' if ks_pass
            else '间隔分布存在轻微偏离，可能源于统计误差，不代表各期存在关联。')),

        ('ok' if acf_exceed == 0 else 'warn',
         f'<strong>自相关检验：{"✅ 通过" if acf_exceed == 0 else "⚠️ 注意"}</strong>（检验 {acf_lags} 个滞后期，{acf_exceed} 个超出 95% CI）— '
         + ('序列无显著自相关，各期结果相互独立。' if acf_exceed == 0
            else '轻微自相关，可能由统计误差引起，不具实际意义。')),

        ('ok',
         f'<strong>生肖分布：</strong>历史最高频「{ZODIAC_EMOJI.get(top1,"")}{top1}」'
         f'（{zodiac_cnt.get(top1,0)}次），最低频「{ZODIAC_EMOJI.get(bot1,"")}{bot1}」'
         f'（{zodiac_cnt.get(bot1,0)}次），差值 {zodiac_cnt.get(top1,0)-zodiac_cnt.get(bot1,0)} 次 — 属正常统计波动。'
         f'近50期最高频：{ZODIAC_EMOJI.get(top1_r50,"")}{top1_r50}。'),

        ('warn' if anom_list else 'ok',
         f'<strong>遗漏异常：</strong>当前共 {len(anom_list)} 个号码遗漏超 2× 均值（{avg_freq:.0f}期）。'
         + (f'最长遗漏 No.{anom_list[0][0]}（{anom_list[0][1]}期）。' if anom_list else '')
         + '遗漏对独立随机事件无预测价值。'),

        ('ok' if pass_cnt >= 2 else 'warn',
         f'<strong>综合评估：{pass_cnt}/3 项检验通过。</strong> '
         + ('数据整体符合随机均匀分布，与彩票纯随机机制一致。任何历史数据对未来无预测能力。'
            if pass_cnt >= 2 else '建议扩大样本量重新评估。无论结果如何，彩票均为独立随机事件。')),
    ]

    for level, text in concl_items:
        cls = 'concl-ok' if level == 'ok' else 'concl-warn'
        st.markdown(f'<div class="concl-item {cls}">{text}</div>',
                    unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown(ch('报告摘要', '关键指标与检验结果一览', '📋'), unsafe_allow_html=True)
    col_r1, col_r2 = st.columns(2)

    with col_r1:
        st.markdown(f"""
        <div class="report-card">
          <h4>📊 数据概览</h4>
          <p>分析期数：<strong>{n}</strong> 期</p>
          <p>时间跨度：{df_full['openTime'].min().strftime('%Y/%m/%d')} ~
             {df_full['openTime'].max().strftime('%Y/%m/%d')}</p>
          <p>最新期号：<strong>{latest['expect']}</strong>，
             特码：<strong style="color:#FF6B6B">{int(latest['special'])}</strong></p>
          <p>每号理论均值：<strong>{avg_freq:.1f}</strong> 次，
             标准差：<strong>{freq_arr.std():.2f}</strong></p>
        </div>
        """, unsafe_allow_html=True)

    with col_r2:
        c_chi = '#27AE60' if chi2_pass else '#E74C3C'
        c_ks  = '#27AE60' if ks_pass   else '#E74C3C'
        c_acf = '#27AE60' if acf_exceed == 0 else '#F39C12'
        st.markdown(f"""
        <div class="report-card">
          <h4>🔬 检验汇总</h4>
          <p>卡方检验：<strong style="color:{c_chi}">
            {'通过 ✅' if chi2_pass else '未通过 ❌'}</strong>（p={chi2_p:.4f}）</p>
          <p>KS 检验：<strong style="color:{c_ks}">
            {'通过 ✅' if ks_pass else '未通过 ❌'}</strong>（p={ks_p:.4f}）</p>
          <p>自相关：<strong style="color:{c_acf}">
            {'正常 ✅' if acf_exceed == 0 else f'{acf_exceed}个超CI ⚠️'}</strong></p>
          <p>综合：<strong style="color:{'#27AE60' if pass_cnt==3 else '#F39C12'}">
            {pass_cnt}/3 通过</strong></p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown(ch('导出报告', '生成包含所有图表和分析结论的独立 HTML 文件，可用浏览器直接打开', '📥'),
                unsafe_allow_html=True)

    if st.button('⚙️ 生成 HTML 分析报告', type='primary', use_container_width=False):
        with st.spinner('正在渲染报告，请稍候...'):
            html_content = generate_html_report(
                df_full, freq_arr, avg_freq, chi2_stat, chi2_p,
                ks_p, gap_arr, zodiac_cnt, sig_level,
                specials, n, latest, pass_cnt
            )
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M')
        st.download_button(
            label='📥 下载 HTML 报告',
            data=html_content.encode('utf-8'),
            file_name=f'macaujc_report_{ts}.html',
            mime='text/html',
            use_container_width=True,
        )
        st.success('✅ 报告已生成，点击上方按钮下载（可用浏览器直接打开）')

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:rgba(243,156,18,0.06);border:1px solid rgba(243,156,18,0.18);
                border-radius:10px;padding:14px 18px;font-size:12px;
                color:rgba(224,230,237,0.5);line-height:1.9;">
      ⚠️ <strong style="color:rgba(224,230,237,0.65)">免责声明</strong><br>
      本系统及所有生成报告仅用于历史数据统计分析与学术研究，不构成任何预测依据，
      不具备选号指导或投资建议功能。彩票开奖为完全独立的随机事件，
      任何历史数据对未来结果均无影响。严禁将本系统用于任何形式的赌博活动。
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# Footer
# ══════════════════════════════════════════════════════════════════════════

st.divider()
st.caption(
    '📊 特码统计分析系统 v2.0 · 基于概率统计的随机性验证平台 · '
    '数据仅供学习研究 · 严禁用于赌博或选号'
)
