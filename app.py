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
from scipy.stats import chisquare, kstest, chi2 as chi2_dist

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
    initial_sidebar_state='collapsed',
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
  section[data-testid="stSidebar"] {
    background:#0D1821 !important;
    border-right:1px solid rgba(0,201,255,0.08);
  }
  section[data-testid="stSidebar"] .block-container { padding-top:0 !important; }

  /* 侧边栏 section 标签 */
  .sb-sec {
    font-size:10px; font-weight:700; color:rgba(0,201,255,0.55);
    text-transform:uppercase; letter-spacing:1.2px;
    padding:2px 0 6px; margin-top:4px;
    display:flex; align-items:center; gap:6px;
  }
  .sb-sec::after {
    content:''; flex:1; height:1px;
    background:rgba(0,201,255,0.1);
  }

  /* 侧边栏 mini-stat 行 */
  .sb-stat-row { display:flex; gap:6px; margin:6px 0 10px; }
  .sb-stat {
    flex:1; background:rgba(0,201,255,0.07); border-radius:8px;
    padding:7px 6px; text-align:center;
    border:1px solid rgba(0,201,255,0.1);
  }
  .sb-stat-v { font-size:14px; font-weight:700; color:#00C9FF; line-height:1.2; }
  .sb-stat-l { font-size:9px; color:rgba(224,230,237,0.32);
               text-transform:uppercase; letter-spacing:.6px; margin-top:2px; }

  /* 侧边栏 radio 水平样式 */
  section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] {
    gap:6px; flex-wrap:wrap;
  }
  section[data-testid="stSidebar"] .stRadio label {
    background:rgba(255,255,255,0.05);
    border:1px solid rgba(255,255,255,0.1);
    border-radius:20px; padding:4px 12px;
    font-size:12px; cursor:pointer;
    transition:all .15s;
  }
  section[data-testid="stSidebar"] .stRadio label:has(input:checked) {
    background:rgba(0,201,255,0.15);
    border-color:rgba(0,201,255,0.5);
    color:#00C9FF;
  }

  /* 侧边栏按钮美化 */
  section[data-testid="stSidebar"] .stButton > button {
    border-radius:8px !important;
    font-size:12px !important;
    padding:6px 14px !important;
    height:auto !important;
  }
  section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background:linear-gradient(135deg,#0077AA,#00C9FF) !important;
    border:none !important;
  }

  /* 完全隐藏侧边栏及展开按钮 */
  section[data-testid="stSidebar"],
  button[data-testid="collapsedControl"],
  button[data-testid="baseButton-headerNoPadding"] { display:none !important; }

  /* ── 顶部控制栏 ────────────────────────────────── */
  .ctrl-label {
    font-size: 9px; font-weight: 700; color: rgba(0,201,255,.42);
    text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 4px;
    margin-top: 0; padding: 0;
  }
  .ctrl-stat {
    background: rgba(255,255,255,.04);
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 9px; padding: 6px 9px; text-align: center;
    transition: border-color .25s, background .25s;
  }
  .ctrl-stat:hover {
    background: rgba(0,201,255,.08);
    border-color: rgba(0,201,255,.22);
  }
  .ctrl-stat-v {
    font-size: 15px; font-weight: 800; color: #00C9FF;
    line-height: 1.2; letter-spacing: -.3px;
  }
  .ctrl-stat-l {
    font-size: 8px; color: rgba(224,230,237,.25);
    text-transform: uppercase; letter-spacing: .6px; margin-top: 2px;
  }
  /* expander 美化 */
  div[data-testid="stExpander"] details {
    background: rgba(255,255,255,.025) !important;
    border: 1px solid rgba(255,255,255,.07) !important;
    border-radius: 8px !important;
  }
  div[data-testid="stExpander"] summary {
    font-size: 11px !important;
    color: rgba(224,230,237,.45) !important;
    padding: 5px 12px !important;
  }

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

@st.cache_data(show_spinner=False)
def compute_freq(specials: np.ndarray):
    """计算 1-49 各号码频次，返回 (freq_arr[49], avg_freq)"""
    cnt  = Counter(specials.tolist())
    freq = np.array([cnt.get(i, 0) for i in range(1, 50)])
    return freq, len(specials) / 49


@st.cache_data(show_spinner=False)
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


@st.cache_data(show_spinner=False)
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

def fig_window_delta_bar(specials_all: np.ndarray, w_start: int, w_size: int,
                         label: str = '') -> plt.Figure:
    """窗口内各号码频率与期望值的偏差柱状图（红=超期望，蓝=低于期望）"""
    win    = specials_all[w_start:w_start + w_size]
    freq_w = np.array([Counter(win.tolist()).get(i, 0) for i in range(1, 50)])
    exp_w  = max(len(win) / 49.0, 0.001)
    delta  = freq_w - exp_w
    colors = [PALETTE['red'] if d > 0 else PALETTE['blue'] for d in delta]
    fig, ax = _fig(13, 3.8)
    ax.bar(range(1, 50), delta, color=colors, alpha=0.82, zorder=2)
    ax.axhline(0, color='#2A3F54', linewidth=1.2, zorder=3)
    ax.set_xlabel('号码', color='#A0AEC0', fontsize=9)
    ax.set_ylabel('偏差（实际 − 期望）', color='#A0AEC0', fontsize=9)
    ax.set_title(f'窗口频率偏差  {label}  · 期望值 {exp_w:.2f} 次/号',
                 fontsize=10, pad=8, color='#E0E6ED')
    ax.tick_params(colors='#A0AEC0', labelsize=8)
    ax.grid(axis='y', alpha=0.18, zorder=1)
    fig.tight_layout()
    return fig


def fig_multi_rolling(specials_all: np.ndarray, window_n: int,
                      nums: list) -> plt.Figure:
    """多号码滚动窗口频率折线对比"""
    colors6 = ['#FF6B6B', '#00C9FF', '#FFEAA7', '#A29BFE']
    exp_rate = window_n / 49.0
    fig, ax = _fig(13, 4.2)
    for idx, num in enumerate(nums[:4]):
        mask  = (specials_all == num).astype(int)
        roll  = np.convolve(mask, np.ones(window_n, dtype=int), mode='valid')
        color = colors6[idx % 4]
        ax.plot(np.arange(len(roll)), roll, color=color, linewidth=1.6,
                label=f'No.{num}', alpha=0.9)
    ax.axhline(exp_rate, color='rgba(255,255,255,0.25)', linewidth=1.2,
               linestyle='--', label=f'期望 {exp_rate:.1f}')
    ax.fill_between(np.arange(len(roll)), exp_rate, alpha=0.04, color='white')
    ax.set_xlabel(f'窗口末端期序（窗口长度={window_n}期）',
                  color='#A0AEC0', fontsize=9)
    ax.set_ylabel(f'近{window_n}期出现次数', color='#A0AEC0', fontsize=9)
    ax.set_title(f'{window_n}期滑动窗口 · 多号码频率对比',
                 fontsize=10, pad=8, color='#E0E6ED')
    ax.tick_params(colors='#A0AEC0', labelsize=8)
    ax.legend(fontsize=9, framealpha=0.12,
              labelcolor='#E0E6ED', facecolor='#1A2634')
    ax.grid(axis='y', alpha=0.18)
    fig.tight_layout()
    return fig


def fig_chunk_heatmap(specials_all: np.ndarray, chunk_size: int = 50) -> plt.Figure:
    """时期分块频率热力图：行=号码1-49，列=时间块，颜色=偏差程度"""
    n_total  = len(specials_all)
    n_chunks = max(2, n_total // chunk_size)
    eff_chunk = n_total // n_chunks
    mat = np.zeros((49, n_chunks))
    for ci in range(n_chunks):
        chunk = specials_all[ci * eff_chunk:(ci + 1) * eff_chunk]
        for num in chunk:
            if 1 <= int(num) <= 49:
                mat[int(num) - 1, ci] += 1
    expected = eff_chunk / 49.0
    mat_norm = (mat - expected) / max(expected, 0.001)
    fig_h = max(5.5, min(9, 49 * 0.17))
    fig_w = max(8,   min(16, n_chunks * 0.55))
    fig, ax = _fig(fig_w, fig_h)
    im = ax.imshow(mat_norm, aspect='auto', cmap='RdBu_r',
                   vmin=-1.5, vmax=1.5, interpolation='nearest')
    ax.set_yticks(range(0, 49, 6))
    ax.set_yticklabels([str(i + 1) for i in range(0, 49, 6)],
                       fontsize=8, color='#A0AEC0')
    x_step = max(1, n_chunks // 20)
    ax.set_xticks(range(0, n_chunks, x_step))
    ax.set_xticklabels(
        [f'{ci * eff_chunk + 1}' for ci in range(0, n_chunks, x_step)],
        rotation=45, ha='right', fontsize=7, color='#A0AEC0')
    ax.set_ylabel('号码', color='#A0AEC0', fontsize=9)
    ax.set_xlabel('起始期序', color='#A0AEC0', fontsize=9)
    ax.set_title(
        f'时期分块频率热力图  每块≈{eff_chunk}期 · 共{n_chunks}块\n'
        '红=高于期望  蓝=低于期望  白=接近期望',
        fontsize=9, pad=8, color='#E0E6ED')
    cb = plt.colorbar(im, ax=ax, label='偏差/期望', shrink=0.75, pad=0.02)
    cb.ax.tick_params(colors='#A0AEC0', labelsize=7)
    cb.set_label('偏差/期望', color='#A0AEC0', fontsize=8)
    fig.tight_layout()
    return fig


def fig_chi2_vs_window(window_sizes: list, chi2_vals: list,
                       p_vals: list, sig_level: float) -> plt.Figure:
    """卡方统计量与 p 值随窗口大小的变化折线图"""
    bg = PALETTE['bg']
    spine_c = '#2A3F54'
    tc = '#A0AEC0'
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    fig.patch.set_facecolor(bg)
    for ax in (ax1, ax2):
        ax.set_facecolor(bg)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(spine_c)
        ax.spines['bottom'].set_color(spine_c)
        ax.tick_params(colors=tc, labelsize=8)
    ax1.plot(window_sizes, chi2_vals, color='#00C9FF', linewidth=1.6, zorder=2)
    ax1.fill_between(window_sizes, chi2_vals, alpha=0.08, color='#00C9FF')
    ax1.set_ylabel('χ² 统计量', color=tc, fontsize=9)
    ax1.yaxis.label.set_color(tc)
    ax1.grid(axis='y', alpha=0.15)
    ax2.plot(window_sizes, p_vals, color='#FFEAA7', linewidth=1.6, zorder=2)
    ax2.axhline(sig_level, color='#E74C3C', linewidth=1.2,
                linestyle='--', label=f'α = {sig_level}', zorder=3)
    ax2.fill_between(window_sizes, p_vals, sig_level,
                     where=[p <= sig_level for p in p_vals],
                     color='#E74C3C', alpha=0.12, label='拒绝H₀区域')
    ax2.set_ylabel('p 值', color=tc, fontsize=9)
    ax2.set_xlabel('窗口大小（期）', color=tc, fontsize=9)
    ax2.xaxis.label.set_color(tc)
    ax2.yaxis.label.set_color(tc)
    ax2.legend(fontsize=8, framealpha=0.12,
               labelcolor='#E0E6ED', facecolor=bg)
    ax2.grid(axis='y', alpha=0.15)
    fig.suptitle('卡方检验统计量随窗口大小的变化（使用最近 N 期数据）',
                 fontsize=10, color='#E0E6ED', y=1.01)
    fig.tight_layout()
    return fig


def fig_number_timeline(specials: np.ndarray, num: int,
                        lookback: int = 300) -> plt.Figure:
    """号码历史出现时间线：近 lookback 期内每次出现标一个点"""
    recent   = specials[-lookback:] if len(specials) > lookback else specials
    total    = len(recent)
    appeared = np.where(recent == num)[0]
    gaps_num = np.diff(appeared).tolist() if len(appeared) > 1 else []

    fig, (ax_tl, ax_gap) = plt.subplots(2, 1, figsize=(13, 4),
                                         gridspec_kw={'height_ratios': [1, 2]})
    fig.patch.set_facecolor(PALETTE['bg'])
    for ax in (ax_tl, ax_gap):
        ax.set_facecolor(PALETTE['bg'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#2A3F54')
        ax.spines['bottom'].set_color('#2A3F54')
        ax.tick_params(colors='#A0AEC0', labelsize=8)

    # ── 上图：时间线条带
    ax_tl.fill_between([0, total], [-0.3, -0.3], [0.3, 0.3],
                        color='#2A3F54', alpha=0.4, zorder=1)
    if len(appeared) > 0:
        ax_tl.vlines(appeared, -0.28, 0.28, color=PALETTE['blue'],
                     linewidth=1.6, alpha=0.75, zorder=2)
        ax_tl.scatter([appeared[-1]], [0], s=100, color=PALETTE['red'],
                      zorder=4, label=f'最近：第{total - appeared[-1]}期前')
    ax_tl.set_xlim(-2, total + 2)
    ax_tl.set_ylim(-0.5, 0.5)
    ax_tl.set_yticks([])
    ax_tl.set_xlabel('')
    ax_tl.set_title(
        f'No.{num:02d} 出现时间线（近{total}期，共出现 {len(appeared)} 次）',
        fontsize=10, pad=6, color='#E0E6ED')
    if len(appeared) > 0:
        ax_tl.legend(fontsize=8, loc='upper left', framealpha=0.1,
                     labelcolor='#E0E6ED', facecolor=PALETTE['bg'])

    # ── 下图：间隔分布直方图
    if gaps_num:
        ax_gap.hist(gaps_num, bins=min(30, len(set(gaps_num))),
                    color=PALETTE['blue'], alpha=0.75, zorder=2)
        mean_g = float(np.mean(gaps_num))
        ax_gap.axvline(mean_g, color=PALETTE['orange'], linewidth=1.4,
                       linestyle='--', label=f'均值 {mean_g:.1f}')
        ax_gap.axvline(avg_freq, color='#A29BFE', linewidth=1.2,
                       linestyle=':', label=f'理论 {avg_freq:.1f}')
        ax_gap.legend(fontsize=8, framealpha=0.1,
                      labelcolor='#E0E6ED', facecolor=PALETTE['bg'])
    else:
        ax_gap.text(0.5, 0.5, '间隔数据不足', transform=ax_gap.transAxes,
                    ha='center', va='center', color='#A0AEC0')
    ax_gap.set_xlabel('出现间隔（期）', fontsize=9)
    ax_gap.set_ylabel('频数', fontsize=9)
    ax_gap.set_title(f'No.{num:02d} 历史出现间隔分布', fontsize=9, pad=4, color='#E0E6ED')
    ax_gap.grid(axis='y', alpha=0.15)

    fig.tight_layout(pad=1.2)
    return fig


def fig_dual_window_compare(specials: np.ndarray,
                             a_start: int, a_size: int,
                             b_start: int, b_size: int) -> plt.Figure:
    """双时段每期发生率对比柱状图（归一化到每期发生率）"""
    win_a  = specials[a_start:a_start + a_size]
    win_b  = specials[b_start:b_start + b_size]
    rate_a = np.array([Counter(win_a.tolist()).get(i, 0) / max(a_size, 1)
                       for i in range(1, 50)])
    rate_b = np.array([Counter(win_b.tolist()).get(i, 0) / max(b_size, 1)
                       for i in range(1, 50)])
    exp_r  = 1.0 / 49.0
    x = np.arange(1, 50)
    w = 0.38
    fig, ax = _fig(14, 4.5)
    ax.bar(x - w / 2, rate_a, w, color=PALETTE['blue'],
           alpha=0.82, label='时段 A', zorder=2)
    ax.bar(x + w / 2, rate_b, w, color=PALETTE['orange'],
           alpha=0.82, label='时段 B', zorder=2)
    ax.axhline(exp_r, color='rgba(255,255,255,0.28)', linewidth=1.2,
               linestyle='--', label=f'期望 {exp_r:.4f}', zorder=3)
    ax.set_xlabel('号码', fontsize=9)
    ax.set_ylabel('每期发生率', fontsize=9)
    ax.set_title('双时段频率对比（蓝=时段 A，橙=时段 B，虚线=期望值）',
                 fontsize=10, pad=8)
    ax.set_xticks(range(1, 50))
    ax.tick_params(axis='x', labelsize=6)
    ax.legend(fontsize=9, framealpha=0.1,
              labelcolor='#E0E6ED', facecolor=PALETTE['bg'])
    ax.grid(axis='y', alpha=0.18)
    fig.tight_layout()
    return fig


def fig_to_b64(fig) -> str:
    """将 matplotlib figure 转为 base64 PNG 字符串（用于内嵌 HTML）"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=90, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def generate_html_report(df_full, freq_arr, avg_freq, chi2_stat, chi2_p,
                         ks_p, gap_arr, zodiac_cnt, sig_level,
                         specials, n, latest, pass_cnt, **kwargs) -> str:
    """生成专业统计分析 HTML 报告（暗色主题，含描述性统计、假设检验、效应量）"""
    now   = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    start = df_full['openTime'].min().strftime('%Y-%m-%d')
    end   = df_full['openTime'].max().strftime('%Y-%m-%d')

    # 额外统计量（从 kwargs 提取，回退到内部计算）
    cramers_v     = kwargs.get('cramers_v',    np.sqrt(chi2_stat / (n * 48)))
    cv_pct        = kwargs.get('cv_pct',       freq_arr.std() / freq_arr.mean() * 100)
    freq_skew     = kwargs.get('freq_skew',    0.0)
    freq_kurt     = kwargs.get('freq_kurt',    0.0)
    gap_std       = kwargs.get('gap_std',      0.0)
    gap_p25       = kwargs.get('gap_p25',      0.0)
    gap_p75       = kwargs.get('gap_p75',      0.0)
    ks_stat_val   = kwargs.get('ks_stat',      0.0)
    chi2_crit     = kwargs.get('chi2_crit',    65.17)
    zod_chi2_stat = kwargs.get('zod_chi2_stat', 0.0)
    zod_chi2_p    = kwargs.get('zod_chi2_p',    1.0)
    zod_chi2_pass = kwargs.get('zod_chi2_pass', True)
    zod_chi2_crit = kwargs.get('zod_chi2_crit', 19.68)
    zod_obs_list  = kwargs.get('zod_obs',       [])
    acf_exceed    = kwargs.get('acf_exceed',    0)
    conf_interval = kwargs.get('conf_interval', 1.96 / np.sqrt(n))
    acf_lags_val  = kwargs.get('acf_lags',      20)

    chi2_pass = chi2_p > sig_level
    ks_pass   = ks_p   > sig_level
    acf_pass  = acf_exceed == 0
    overall_pass = pass_cnt >= 2

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

    # 生肖表格行
    top1 = max(zodiac_cnt, key=zodiac_cnt.get) if zodiac_cnt else '-'
    bot1 = min(zodiac_cnt, key=zodiac_cnt.get) if zodiac_cnt else '-'
    zodiac_rows = ''
    for z in ZODIAC_ORDER:
        if z not in zodiac_cnt:
            continue
        cnt_z = zodiac_cnt[z]
        delta = cnt_z - n / 12
        sign  = '+' if delta >= 0 else ''
        dev_p = delta / (n / 12) * 100
        sign2 = '+' if dev_p >= 0 else ''
        zodiac_rows += (
            f'<tr><td>{ZODIAC_EMOJI.get(z,"")}&nbsp;{z}</td>'
            f'<td style="text-align:right;font-weight:600">{cnt_z}</td>'
            f'<td style="text-align:right">{cnt_z/n*100:.1f}%</td>'
            f'<td style="text-align:right">{sign}{delta:.1f}</td>'
            f'<td style="text-align:right;color:{"#27AE60" if abs(dev_p)<8 else "#F39C12"}">'
            f'{sign2}{dev_p:.1f}%</td></tr>'
        )

    # 遗漏异常
    warn_nums = [(i+1, int(gap_arr[i])) for i in range(49) if gap_arr[i] >= avg_freq * 2]
    warn_nums.sort(key=lambda x: x[1], reverse=True)
    warn_text = '、'.join([f'No.{nn}（遗漏{g}期）' for nn, g in warn_nums[:5]]) if warn_nums else '无'

    # Cramér's V interpretation
    if cramers_v < 0.1:
        cv_level, cv_color = '微弱（Negligible）', '#27AE60'
    elif cramers_v < 0.3:
        cv_level, cv_color = '小效应（Small）', '#F39C12'
    else:
        cv_level, cv_color = '中等效应（Medium）', '#E74C3C'

    # p-value formatter
    def pfmt(p): return f'{p:.4f}' if p >= 0.0001 else '&lt; 0.0001'
    def dec(passed):
        return ('<span class="pass">不拒绝 H₀</span>' if passed
                else '<span class="fail">拒绝 H₀</span>')

    # 正式结论段落
    concl_chi2 = (
        f'在 α={sig_level} 显著性水平下，卡方统计量 χ²(48) = {chi2_stat:.4f}（p = {pfmt(chi2_p)}），'
        + ('p &gt; α，<em>未能拒绝零假设</em>，49个号码频率与均匀分布无统计显著差异。'
           if chi2_pass else
           'p ≤ α，<em>拒绝零假设</em>，频率存在偏离，但效应量微弱，实际意义有限。')
        + f' Cramér\'s V = {cramers_v:.4f}（{cv_level}）。'
    )
    concl_ks = (
        f'KS 统计量 D = {ks_stat_val:.4f}（p = {pfmt(ks_p)}），'
        + ('p &gt; α，<em>未能拒绝零假设</em>，号码间隔服从指数分布，支持各期独立随机假设。'
           if ks_pass else
           'p ≤ α，<em>拒绝零假设</em>，间隔分布存在偏离，可能由离散化误差或边界效应引起。')
    )
    concl_acf = (
        f'检验 {acf_lags_val} 个滞后期，{acf_exceed} 个 ACF 系数超出 95% CI（±{conf_interval:.4f}）。'
        + ('<em>未能拒绝无自相关零假设</em>，序列无线性时序依赖。'
           if acf_pass else
           '超出比例约 5%，与随机期望一致，不具实际显著意义。')
    )
    concl_zod = (
        f'生肖卡方 χ²({len(zod_obs_list)-1 if len(zod_obs_list)>=2 else "—"}) = {zod_chi2_stat:.4f}'
        f'（p = {pfmt(zod_chi2_p)}），'
        + ('<em>未能拒绝零假设</em>，12生肖分布符合均匀性。'
           if zod_chi2_pass else
           '<em>拒绝零假设</em>，生肖分布存在偏离，但无实际预测价值。')
    )
    concl_overall = (
        f'{pass_cnt}/3 项核心假设检验通过。'
        + ('数据整体符合独立随机均匀分布特征，与彩票纯随机抽签机制一致。'
           if overall_pass else
           '部分检验存在偏离，建议扩大样本量后重新评估；彩票仍为独立随机事件。')
    )

    verd_c  = '#27AE60' if overall_pass else '#E74C3C'
    verd_bg = 'rgba(39,174,96,0.08)' if overall_pass else 'rgba(231,76,60,0.08)'
    verd_bd = 'rgba(39,174,96,0.3)'  if overall_pass else 'rgba(231,76,60,0.3)'
    verd_ic = '✅' if overall_pass else '⚠️'
    verd_tx = '随机均匀分布假设成立' if overall_pass else '数据存在统计显著偏离'

    # 描述性统计表行
    freq_iqr  = float(np.percentile(freq_arr, 75) - np.percentile(freq_arr, 25))
    freq_ci95 = 1.96 * freq_arr.std() / np.sqrt(49)
    skew_interp = '近似对称' if abs(freq_skew) < 0.5 else ('正偏' if freq_skew > 0 else '负偏')
    kurt_interp = '近似正态' if abs(freq_kurt) < 0.5 else ('厚尾' if freq_kurt > 0 else '薄尾')

    desc_rows_html = ''
    desc_data = [
        ('样本量（期数 n）',     f'{n}',                               '开奖记录总数'),
        ('号码均值 μ',           f'{avg_freq:.3f}',                     '期望出现次数 = n/49'),
        ('标准差 σ',             f'{freq_arr.std():.3f}',               '频次均方根偏差'),
        ('变异系数 CV',          f'{cv_pct:.2f}%',                      'σ/μ×100%'),
        ('中位数',               f'{float(np.median(freq_arr)):.1f}',   '稳健中心估计'),
        ('四分位距 IQR',         f'{freq_iqr:.1f}',                     'Q₃ − Q₁'),
        ('均值 95% CI 半宽',     f'±{freq_ci95:.3f}',                   '1.96 × σ / √49'),
        ('偏度 γ₁',              f'{freq_skew:.4f}',                    skew_interp),
        ('超额峰度 γ₂',          f'{freq_kurt:.4f}',                    kurt_interp),
        ('极差 Range',           f'{int(freq_arr.max() - freq_arr.min())}', 'max − min'),
        ('间隔均值 E[G]',        f'{avg_freq:.2f}',                     '≈ n/49'),
        ('间隔标准差 σ[G]',      f'{gap_std:.2f}',                      '遗漏离散程度'),
        ('间隔四分位 Q₁/Q₃',    f'{gap_p25:.1f} / {gap_p75:.1f}',      ''),
    ]
    for nm, val, note in desc_data:
        desc_rows_html += (
            f'<tr><td>{nm}</td>'
            f'<td style="text-align:right;font-weight:600;color:#E0E6ED">{val}</td>'
            f'<td style="color:rgba(224,230,237,.45);font-size:11px">{note}</td></tr>'
        )

    # 假设检验表行
    n_zod_df = len(zod_obs_list) - 1 if len(zod_obs_list) >= 2 else 0
    test_rows_html = (
        f'<tr><td>卡方拟合优度检验</td><td>H₀：49号等概率</td>'
        f'<td>χ²(48) = {chi2_stat:.4f}</td><td>{pfmt(chi2_p)}</td>'
        f'<td>{chi2_crit:.3f}</td><td>{dec(chi2_pass)}</td>'
        f'<td>V = {cramers_v:.4f}<br><small style="color:{cv_color}">{cv_level}</small></td></tr>'

        f'<tr><td>KS 指数分布拟合检验</td><td>H₀：间隔 ~ Exp(λ)</td>'
        f'<td>D = {ks_stat_val:.4f}</td><td>{pfmt(ks_p)}</td>'
        f'<td>—</td><td>{dec(ks_pass)}</td><td>—</td></tr>'

        f'<tr><td>自相关检验（ACF）</td>'
        f'<td>H₀：序列无自相关</td>'
        f'<td>{acf_exceed}/{acf_lags_val} 超 CI</td><td>—</td>'
        f'<td>±{conf_interval:.4f}</td><td>{dec(acf_pass)}</td><td>—</td></tr>'

        f'<tr><td>生肖均匀性卡方</td>'
        f'<td>H₀：12生肖等频</td>'
        f'<td>χ²({n_zod_df}) = {zod_chi2_stat:.4f}</td>'
        f'<td>{pfmt(zod_chi2_p)}</td>'
        f'<td>{zod_chi2_crit:.3f}</td>'
        f'<td>{dec(zod_chi2_pass)}</td><td>—</td></tr>'
    )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>特码统计分析报告 — {start} ~ {end}</title>
<style>
  *{{margin:0;padding:0;box-sizing:border-box;}}
  body{{background:#0F1923;color:#E0E6ED;
       font-family:-apple-system,'PingFang SC','Helvetica Neue',sans-serif;
       padding:32px 48px;max-width:1120px;margin:0 auto;line-height:1.6;}}
  h1{{font-size:24px;color:#00C9FF;border-bottom:2px solid rgba(0,201,255,.3);
      padding-bottom:12px;margin-bottom:8px;}}
  .meta{{color:rgba(224,230,237,.4);font-size:12px;margin-bottom:28px;}}
  h2{{font-size:15px;color:#E0E6ED;margin:32px 0 14px;
      padding-left:12px;border-left:3px solid #00C9FF;letter-spacing:.3px;}}
  .verdict{{background:{verd_bg};border:1px solid {verd_bd};border-radius:12px;
            padding:20px 24px;margin-bottom:24px;display:flex;gap:16px;align-items:flex-start;}}
  .verdict-icon{{font-size:28px;line-height:1;}}
  .verdict-title{{font-size:17px;font-weight:700;color:{verd_c};margin-bottom:6px;}}
  .verdict-body{{font-size:13px;color:rgba(224,230,237,.75);}}
  .kpi-row{{display:flex;gap:12px;margin:16px 0;flex-wrap:wrap;}}
  .kpi{{background:#1A2634;border-radius:10px;padding:14px 18px;min-width:120px;
        border:1px solid rgba(0,201,255,.12);flex:1;text-align:center;}}
  .kpi-val{{font-size:20px;font-weight:700;color:#00C9FF;}}
  .kpi-lbl{{font-size:10px;color:rgba(224,230,237,.4);margin-top:4px;
             text-transform:uppercase;letter-spacing:.5px;}}
  img{{max-width:100%;border-radius:8px;margin:10px 0;border:1px solid rgba(255,255,255,.05);}}
  table{{width:100%;border-collapse:collapse;font-size:12px;margin:12px 0;}}
  th{{background:#1A2634;color:#00C9FF;padding:9px 13px;text-align:left;
      border-bottom:1px solid rgba(0,201,255,.18);font-size:11px;white-space:nowrap;}}
  td{{padding:8px 13px;border-bottom:1px solid rgba(255,255,255,.04);font-size:12px;}}
  tr:hover td{{background:rgba(255,255,255,.025);}}
  .pass{{color:#27AE60;font-weight:700;}}
  .fail{{color:#E74C3C;font-weight:700;}}
  .concl{{border-left:3px solid #00C9FF;background:rgba(0,201,255,.04);
          padding:11px 16px;margin:9px 0;border-radius:0 8px 8px 0;
          font-size:13px;line-height:1.75;}}
  .concl strong{{color:#E0E6ED;}}
  .lim-item{{display:flex;gap:12px;padding:10px 0;
             border-bottom:1px solid rgba(255,255,255,.04);}}
  .lim-num{{min-width:22px;height:22px;background:rgba(0,201,255,.1);border-radius:50%;
            display:flex;align-items:center;justify-content:center;
            font-size:11px;color:#00C9FF;font-weight:700;margin-top:2px;flex-shrink:0;}}
  .lim-text{{font-size:12px;color:rgba(224,230,237,.6);line-height:1.7;}}
  .eff-box{{background:#1A2634;border:1px solid rgba(0,201,255,.1);border-radius:10px;
            padding:16px 20px;display:inline-block;width:48%;margin:6px 1%;
            vertical-align:top;}}
  .eff-lbl{{font-size:10px;color:rgba(224,230,237,.4);text-transform:uppercase;
            letter-spacing:.6px;margin-bottom:6px;}}
  .eff-val{{font-size:18px;font-weight:700;margin-bottom:6px;}}
  .eff-desc{{font-size:12px;color:rgba(224,230,237,.6);line-height:1.65;}}
  .disc{{background:rgba(243,156,18,.06);border:1px solid rgba(243,156,18,.18);
         border-radius:10px;padding:16px 20px;margin-top:36px;
         font-size:11px;color:rgba(224,230,237,.45);line-height:1.85;}}
  .disc strong{{color:rgba(224,230,237,.65);}}
  footer{{text-align:center;margin-top:24px;font-size:11px;
          color:rgba(224,230,237,.18);padding-top:14px;
          border-top:1px solid rgba(255,255,255,.04);}}
</style>
</head>
<body>
<h1>📊 特码统计分析报告</h1>
<p class="meta">
  生成时间：{now} &nbsp;·&nbsp; 数据范围：{start} ~ {end}
  &nbsp;·&nbsp; 共 <strong style="color:#E0E6ED">{n}</strong> 期
  &nbsp;·&nbsp; 显著性水平 α = {sig_level}
</p>

<!-- 执行摘要 -->
<div class="verdict">
  <div class="verdict-icon">{verd_ic}</div>
  <div>
    <div class="verdict-title">执行摘要 — {verd_tx}</div>
    <div class="verdict-body">
      在 α={sig_level} 显著性水平下，{pass_cnt}/3 项核心假设检验
      {"未能拒绝零假设，数据整体符合独立随机均匀分布特征，与彩票纯随机抽签机制理论一致。" if overall_pass else "拒绝零假设，数据存在统计偏离，建议扩大样本量评估；彩票仍为独立随机事件。"}
    </div>
    <div class="kpi-row" style="margin-top:14px">
      <div class="kpi"><div class="kpi-val" style="color:{verd_c}">{pass_cnt}/3</div><div class="kpi-lbl">检验通过</div></div>
      <div class="kpi"><div class="kpi-val">{n}</div><div class="kpi-lbl">分析期数</div></div>
      <div class="kpi"><div class="kpi-val">{sig_level}</div><div class="kpi-lbl">显著性水平 α</div></div>
      <div class="kpi"><div class="kpi-val" style="color:{cv_color}">{cramers_v:.4f}</div><div class="kpi-lbl">Cramér's V</div></div>
      <div class="kpi"><div class="kpi-val">{cv_pct:.1f}%</div><div class="kpi-lbl">变异系数 CV</div></div>
    </div>
  </div>
</div>

<!-- 一、描述性统计 -->
<h2>一、描述性统计</h2>
<p style="font-size:12px;color:rgba(224,230,237,.45);margin-bottom:10px">
  特码频率分布基本统计特征（k = 49 个号码，n = {n} 期）
</p>
<table>
  <tr><th>统计量</th><th style="text-align:right">数值</th><th>说明</th></tr>
  {desc_rows_html}
</table>

<!-- 二、号码频率分布 -->
<h2>二、号码频率分布</h2>
<img src="data:image/png;base64,{b64['freq']}" alt="频率分布"/>
<img src="data:image/png;base64,{b64['heat']}" alt="热力图"/>

<!-- 三、假设检验汇总 -->
<h2>三、假设检验汇总</h2>
<p style="font-size:12px;color:rgba(224,230,237,.45);margin-bottom:10px">
  所有检验基于双侧或右尾原则，显著性水平 α = {sig_level}
</p>
<table>
  <tr><th>检验项目</th><th>H₀ 假设</th><th>统计量值</th>
      <th>p 值</th><th>临界值</th><th>决策</th><th>效应量</th></tr>
  {test_rows_html}
</table>

<!-- 四、效应量解读 -->
<h2>四、效应量解读</h2>
<div class="eff-box">
  <div class="eff-lbl">Cramér's V（卡方效应量）</div>
  <div class="eff-val" style="color:{cv_color}">{cramers_v:.4f} — {cv_level}</div>
  <div class="eff-desc">
    {"V &lt; 0.1，号码频率偏差极小，实际意义可忽略不计。统计显著性与实际意义相独立。" if cramers_v < 0.1 else ("V = 0.1~0.3，存在小幅偏离，实际意义仍然有限。" if cramers_v < 0.3 else "V ≥ 0.3，中等效应，建议检查数据质量。")}
    <br><small style="color:rgba(224,230,237,.3)">V = √(χ² / (n × (k−1)))，k=49，n={n}</small>
  </div>
</div>
<div class="eff-box">
  <div class="eff-lbl">变异系数 CV（频率相对离散度）</div>
  <div class="eff-val" style="color:#00C9FF">{cv_pct:.2f}%</div>
  <div class="eff-desc">
    {"CV &lt; 15%，各号频率高度均匀，接近理论等概率值。" if cv_pct < 15 else "CV 较高，频率分布存在不均匀性，但属正常统计波动范围。"}
    偏度 γ₁ = {freq_skew:.4f}（{skew_interp}），超额峰度 γ₂ = {freq_kurt:.4f}（{kurt_interp}）。
  </div>
</div>

<!-- 五、生肖分布分析 -->
<h2>五、生肖分布分析</h2>
<img src="data:image/png;base64,{b64['zod']}" alt="生肖分布"/>
<table>
  <tr><th>生肖</th><th style="text-align:right">频次</th>
      <th style="text-align:right">占比</th>
      <th style="text-align:right">偏差（次）</th>
      <th style="text-align:right">偏差%</th></tr>
  {zodiac_rows}
</table>
<p style="font-size:11px;color:rgba(224,230,237,.35);margin-top:6px">
  理论均值 {n/12:.1f} 次/生肖 &nbsp;·&nbsp;
  最高：{ZODIAC_EMOJI.get(top1,'')}{top1}（{zodiac_cnt.get(top1,0)}次）&nbsp;·&nbsp;
  最低：{ZODIAC_EMOJI.get(bot1,'')}{bot1}（{zodiac_cnt.get(bot1,0)}次）&nbsp;·&nbsp;
  生肖卡方 p = {pfmt(zod_chi2_p)} — {"✅ 均匀" if zod_chi2_pass else "⚠️ 偏离"}
</p>

<!-- 六、遗漏分析 -->
<h2>六、遗漏分析</h2>
<img src="data:image/png;base64,{b64['gap']}" alt="遗漏分析"/>
<p style="font-size:12px;color:rgba(224,230,237,.45);margin:6px 0">
  遗漏异常（&gt; 2× 均值 {avg_freq:.0f} 期）：{warn_text}
</p>

<!-- 七、研究结论 -->
<h2>七、研究结论</h2>
<div class="concl">
  <strong>① 均匀性检验：</strong>{concl_chi2}
</div>
<div class="concl">
  <strong>② 独立性检验（KS）：</strong>{concl_ks}
</div>
<div class="concl">
  <strong>③ 自相关分析：</strong>{concl_acf}
</div>
<div class="concl">
  <strong>④ 生肖分布检验：</strong>{concl_zod}
</div>
<div class="concl" style="border-left-color:#A29BFE;background:rgba(162,155,254,.04)">
  <strong>⑤ 综合评估：</strong>{concl_overall}
</div>

<!-- 八、研究局限性 -->
<h2>八、研究局限性</h2>
<div style="background:#1A2634;border:1px solid rgba(255,255,255,.05);
            border-radius:10px;padding:14px 18px;">
  <div class="lim-item">
    <div class="lim-num">1</div>
    <div class="lim-text"><strong style="color:#E0E6ED">样本量约束：</strong>
    基于 {n} 期历史数据，较小样本可能导致统计功效不足；较大样本则因统计过敏感而拒绝零假设（即便效应量微弱）。</div>
  </div>
  <div class="lim-item">
    <div class="lim-num">2</div>
    <div class="lim-text"><strong style="color:#E0E6ED">多重检验问题：</strong>
    本报告同时执行 4 项假设检验，未进行 Bonferroni/FDR 校正，存在 I 类错误累积风险，建议参考效应量而非单纯依赖 p 值。</div>
  </div>
  <div class="lim-item">
    <div class="lim-num">3</div>
    <div class="lim-text"><strong style="color:#E0E6ED">模型假设局限：</strong>
    KS 检验假设连续型指数分布，而间隔数据为离散整数，存在检验偏差。ACF 仅检验线性依赖，无法探测非线性时序结构。</div>
  </div>
  <div class="lim-item">
    <div class="lim-num">4</div>
    <div class="lim-text"><strong style="color:#E0E6ED">随机性本质：</strong>
    即使统计检验显示偏离，彩票开奖在物理与制度层面保证每期独立性。统计偏离不等同于可预测性或可操作性。</div>
  </div>
  <div class="lim-item" style="border-bottom:none">
    <div class="lim-num">5</div>
    <div class="lim-text"><strong style="color:#E0E6ED">研究性质声明：</strong>
    本分析为纯描述性统计，无任何预测或选号功能，全部结论仅供学术与教育目的。</div>
  </div>
</div>

<div class="disc">
  ⚠️ <strong>免责声明</strong><br>
  本报告仅用于历史数据统计分析与学术研究，不构成任何预测依据，不具备选号指导功能，
  严禁用于任何形式的赌博或商业活动。彩票开奖为完全独立的随机事件，
  任何历史数据对未来结果均无影响。
</div>
<footer>特码统计分析系统 v2.0 &nbsp;·&nbsp; 报告生成于 {now} &nbsp;·&nbsp; 数据来源：history.macaumarksix.com</footer>
</body>
</html>"""
    return html


# ══════════════════════════════════════════════════════════════════════════
# 6. 顶部控制栏（无侧边栏设计）
# ══════════════════════════════════════════════════════════════════════════

# ── Row 1：数据源 + 时间范围 + 指标卡片 + 操作 ──────────────────────────────
_cb1, _cb2, _cb3, _cb4 = st.columns([1.8, 3.2, 2.4, 1.6])

with _cb1:
    st.markdown('<p class="ctrl-label">📁 数据文件</p>', unsafe_allow_html=True)
    csvs = sorted(glob.glob(os.path.join(config.OUTPUT_DIR, '????????.csv')))
    if not csvs:
        st.error('未找到数据文件，请先运行 main.py')
        st.stop()
    file_names = [os.path.basename(c) for c in csvs]
    sel_file   = st.selectbox('数据文件', file_names, index=len(file_names) - 1,
                               label_visibility='collapsed')
    sel_path   = os.path.join(config.OUTPUT_DIR, sel_file)
    df_raw     = load_csv(sel_path)

with _cb2:
    st.markdown('<p class="ctrl-label">🗓 时间范围</p>', unsafe_allow_html=True)
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

with _cb3:
    _n_filt  = len(df_full)
    _yr_span = (f'{start_d.year}' if start_d.year == end_d.year
                else f'{start_d.year}–{end_d.year}')
    # 4 格 2×2 指标 grid
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:5px">
      <div class="ctrl-stat">
        <div class="ctrl-stat-v">{_n_filt}</div>
        <div class="ctrl-stat-l">总期数</div>
      </div>
      <div class="ctrl-stat">
        <div class="ctrl-stat-v" style="font-size:13px;letter-spacing:0">{_yr_span}</div>
        <div class="ctrl-stat-l">年份跨度</div>
      </div>
      <div class="ctrl-stat">
        <div class="ctrl-stat-v" style="font-size:12px;color:#27AE60;letter-spacing:0">{start_d.strftime('%y/%m/%d')}</div>
        <div class="ctrl-stat-l">起始日期</div>
      </div>
      <div class="ctrl-stat">
        <div class="ctrl-stat-v" style="font-size:12px;color:#F39C12;letter-spacing:0">{end_d.strftime('%y/%m/%d')}</div>
        <div class="ctrl-stat-l">最新日期</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with _cb4:
    if IS_CLOUD:
        _lf = file_names[-1].replace('.csv', '')
        _ld = f'{_lf[:4]}/{_lf[4:6]}/{_lf[6:]}' if len(_lf) == 8 else _lf
        st.markdown(f"""
        <div style="background:rgba(39,174,96,.08);border:1px solid rgba(39,174,96,.2);
             border-radius:10px;padding:10px 12px;text-align:center">
          <div style="font-size:10px;font-weight:700;color:#27AE60;letter-spacing:.5px">
            🟢 &nbsp;云端运行</div>
          <div style="font-size:9px;color:rgba(224,230,237,.28);margin-top:4px;letter-spacing:.3px">{_ld}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        if st.button('🚀 拉取最新', use_container_width=True, type='primary'):
            fetch_data()
            st.rerun()
        if st.button('🔄 刷新页面', use_container_width=True):
            st.rerun()

st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)

# ── Row 2：分析参数（可折叠，默认收起）────────────────────────────────────
with st.expander('⚙️ 分析参数', expanded=False):
    _ap1, _ap2, _ap3 = st.columns([3, 2, 3])
    with _ap1:
        window_n = st.slider('🔁 滑动窗口（期）', 30, 300, 100, step=10,
                              help='影响 Tab 4 冷热榜、多号码对比图')
    with _ap2:
        _sig_sel  = st.radio('显著性水平 α', ['0.05', '0.01'], horizontal=True)
        sig_level = float(_sig_sel)
    with _ap3:
        acf_lags = st.slider('📉 自相关最大滞后', 5, 40, 20, step=5,
                              help='影响 Tab 5 自相关检验图')

st.divider()


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

# ── 专业报告附加统计量 ────────────────────────────────────────────────────
cramers_v  = np.sqrt(chi2_stat / (n * 48))           # Cramér's V（k=49，df=48）
cv_pct     = freq_arr.std() / freq_arr.mean() * 100   # 变异系数 %
_z_tmp     = (freq_arr - freq_arr.mean()) / freq_arr.std()
freq_skew  = float(np.mean(_z_tmp ** 3))              # 偏度
freq_kurt  = float(np.mean(_z_tmp ** 4) - 3)          # 超额峰度（正态=0）
gap_std    = float(all_gaps.std())  if len(all_gaps) > 0 else 0.0
gap_p25    = float(np.percentile(all_gaps, 25)) if len(all_gaps) > 0 else 0.0
gap_p75    = float(np.percentile(all_gaps, 75)) if len(all_gaps) > 0 else 0.0
chi2_crit  = chi2_dist.isf(sig_level, df=48)          # 卡方临界值（右尾）
zod_obs    = [zodiac_cnt.get(z, 0) for z in ZODIAC_ORDER if z in zodiac_cnt]
if len(zod_obs) >= 2:
    zod_chi2_stat, zod_chi2_p = chisquare(zod_obs, [n / len(zod_obs)] * len(zod_obs))
    zod_chi2_crit = chi2_dist.isf(sig_level, df=len(zod_obs) - 1)
else:
    zod_chi2_stat, zod_chi2_p, zod_chi2_crit = 0.0, 1.0, 0.0
zod_chi2_pass = zod_chi2_p > sig_level

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

    # ── 最近10期走势快览
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown(ch('最近 10 期走势快览', '号码球 · 近30期趋势脉冲 · 奇偶大小快速统计', '🎯'),
                unsafe_allow_html=True)

    # ─ 号码球卡片（最近10期，最新在前）
    recent10_df  = df_full.tail(10).iloc[::-1].reset_index(drop=True)
    r10_nums     = recent10_df['special'].astype(int).tolist()
    r10_periods  = recent10_df['expect'].astype(str).tolist()

    def _ball_style(num: int):
        """按号码段返回 (球色, 辉光色)"""
        if 1  <= num <= 12: return '#FF6B6B', '#FF6B6B44'
        if 13 <= num <= 24: return '#F39C12', '#F39C1244'
        if 25 <= num <= 36: return '#00C9FF', '#00C9FF44'
        return '#27AE60', '#27AE6044'

    balls_html = '<div style="display:flex;gap:10px;flex-wrap:wrap;margin:10px 0 16px">'
    for period, num in zip(r10_periods, r10_nums):
        bg, glow = _ball_style(num)
        is_odd   = num % 2 == 1
        is_big   = num > 24
        tag_o    = '奇' if is_odd else '偶'
        tag_s    = '大' if is_big else '小'
        c_o      = '#00C9FF' if is_odd else '#FF6B6B'
        c_s      = '#F39C12' if is_big else '#9B59B6'
        short_p  = period[-3:] if len(period) >= 3 else period
        balls_html += f'''
        <div style="display:flex;flex-direction:column;align-items:center;gap:4px;min-width:52px">
          <div style="font-size:9px;color:rgba(224,230,237,.35);letter-spacing:.4px">{short_p}</div>
          <div style="width:46px;height:46px;border-radius:50%;background:{bg};
               display:flex;align-items:center;justify-content:center;
               font-size:15px;font-weight:800;color:#fff;
               box-shadow:0 0 12px {glow}">{num:02d}</div>
          <div style="display:flex;gap:3px">
            <span style="font-size:8px;color:{c_o};background:{c_o}22;
                  border-radius:3px;padding:1px 4px;letter-spacing:.3px">{tag_o}</span>
            <span style="font-size:8px;color:{c_s};background:{c_s}22;
                  border-radius:3px;padding:1px 4px;letter-spacing:.3px">{tag_s}</span>
          </div>
        </div>'''
    balls_html += '</div>'
    st.markdown(balls_html, unsafe_allow_html=True)

    # ─ 色块图例说明
    legend_html = (
        '<div style="display:flex;gap:16px;margin-bottom:14px;flex-wrap:wrap">'
        + ''.join(
            f'<span style="font-size:10px;color:{c};background:{c}18;'
            f'border-radius:4px;padding:2px 8px">{lab}</span>'
            for c, lab in [
                ('#FF6B6B','1–12'), ('#F39C12','13–24'),
                ('#00C9FF','25–36'), ('#27AE60','37–49'),
            ]
        )
        + '</div>'
    )
    st.markdown(legend_html, unsafe_allow_html=True)

    # ─ 走势图 + 快速统计（2列）
    _tw1, _tw2 = st.columns([3, 1])

    with _tw1:
        lookback = min(30, len(specials))
        recent_vals = specials[-lookback:]
        xs = list(range(lookback))

        fig_t, ax_t = _fig(9, 2.4)
        # 底部彩色分区（号码段背景）
        for ylo, yhi, c in [(0,12,'#FF6B6B'),(12,24,'#F39C12'),(24,36,'#00C9FF'),(36,49,'#27AE60')]:
            ax_t.axhspan(ylo, yhi, color=c, alpha=0.04)
        # 高亮最近10期区间
        ax_t.axvspan(lookback - 10 - 0.5, lookback - 0.5, color=PALETTE['blue'], alpha=0.07)
        # 折线
        ax_t.plot(xs, recent_vals, color=PALETTE['blue'], lw=1.8, alpha=0.85, zorder=3)
        ax_t.scatter(xs[:-10], recent_vals[:-10],
                     s=14, color=PALETTE['grey'], zorder=4, alpha=0.55)
        ax_t.scatter(xs[-10:], recent_vals[-10:],
                     s=24, color=PALETTE['blue'], zorder=5, alpha=0.9)
        # 大小分界线
        ax_t.axhline(24.5, color=PALETTE['grey'], lw=0.9, ls='--', alpha=0.45)
        ax_t.text(0.5, 24.8, '大/小分界', transform=ax_t.get_yaxis_transform(),
                  fontsize=7, color=PALETTE['grey'], alpha=0.6)
        # 标注最近10期数值
        for i, v in enumerate(recent_vals[-10:]):
            xi = lookback - 10 + i
            offset = 6 if i % 2 == 0 else -12
            ax_t.annotate(f'{int(v)}', (xi, v),
                          textcoords='offset points', xytext=(0, offset),
                          fontsize=7.5, color='#E0E6ED', ha='center', alpha=0.92,
                          fontweight='bold')
        ax_t.set_xlim(-0.5, lookback - 0.5)
        ax_t.set_ylim(0, 51)
        ax_t.set_yticks([1, 12, 24, 36, 49])
        ax_t.set_xticks([])
        ax_t.set_title(f'近 {lookback} 期特码走势  ▏蓝色阴影 = 最近 10 期', fontsize=9, pad=6)
        ax_t.grid(axis='y', alpha=0.18)
        fig_t.tight_layout(pad=0.4)
        st.pyplot(fig_t, use_container_width=True)

    with _tw2:
        r10_odd  = sum(1 for x in r10_nums if x % 2 == 1)
        r10_even = 10 - r10_odd
        r10_big  = sum(1 for x in r10_nums if x > 24)
        r10_sml  = 10 - r10_big
        r10_mean = float(np.mean(r10_nums))
        r10_sum  = sum(r10_nums)
        r10_max  = max(r10_nums)
        r10_min  = min(r10_nums)

        mini_items = [
            ('奇偶比', f'{r10_odd} : {r10_even}', '#00C9FF',
             f'奇 {r10_odd*10}% / 偶 {r10_even*10}%'),
            ('大小比', f'{r10_big} : {r10_sml}',  '#F39C12',
             f'大(>24) {r10_big*10}% / 小 {r10_sml*10}%'),
            ('区间均值', f'{r10_mean:.1f}',         '#27AE60',
             f'全局均值 25.0'),
            ('极差',   f'{r10_max}–{r10_min}',    '#9B59B6',
             f'范围跨度 {r10_max - r10_min}'),
        ]
        for label, value, color, sub in mini_items:
            st.markdown(f'''
            <div style="background:rgba(255,255,255,.04);border-left:3px solid {color};
                 border-radius:6px;padding:8px 12px;margin-bottom:8px">
              <div style="font-size:9px;color:rgba(224,230,237,.4);text-transform:uppercase;
                   letter-spacing:.7px;margin-bottom:2px">{label}</div>
              <div style="font-size:17px;font-weight:700;color:{color};line-height:1.1">{value}</div>
              <div style="font-size:9px;color:rgba(224,230,237,.28);margin-top:3px">{sub}</div>
            </div>''', unsafe_allow_html=True)


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

    st.markdown('<br>', unsafe_allow_html=True)

    # ── ⚡ 动态窗口频率偏差 ──────────────────────────────────────────────
    st.markdown(ch('⚡ 动态窗口频率偏差',
                   '拖动两个滑块选择任意历史时间窗口，实时刷新号码相对期望的频率偏差；红柱=超期望，蓝柱=低于期望',
                   '🎛️'), unsafe_allow_html=True)
    col_wctrl, col_wchart = st.columns([1, 3])
    with col_wctrl:
        w2_max   = min(n, 500)
        w2_size  = st.slider('窗口大小（期）', 20, w2_max, min(100, w2_max),
                              step=10, key='w2_size')
        w2_end   = st.slider('窗口末尾位置', w2_size, n, n, step=1, key='w2_end',
                              help='向左拖动查看历史任意窗口，右端=最新数据')
        w2_start = w2_end - w2_size
        win2_sp  = specials[w2_start:w2_end]
        freq_w2  = np.array([Counter(win2_sp.tolist()).get(i, 0) for i in range(1, 50)])
        exp_w2   = len(win2_sp) / 49.0
        hot3_w   = sorted(enumerate(freq_w2, 1), key=lambda x: x[1], reverse=True)[:3]
        cold3_w  = sorted(enumerate(freq_w2, 1), key=lambda x: x[1])[:3]
        hot_html  = ''.join(
            f'<div style="margin:3px 0"><strong style="color:#E0E6ED">No.{nm}</strong>'
            f' <span style="color:rgba(224,230,237,0.5);font-size:11px">'
            f'{v:.0f}次 (<span style="color:#FF6B6B">+{v-exp_w2:.1f}</span>)</span></div>'
            for nm, v in hot3_w
        )
        cold_html = ''.join(
            f'<div style="margin:3px 0"><strong style="color:#E0E6ED">No.{nm}</strong>'
            f' <span style="color:rgba(224,230,237,0.5);font-size:11px">'
            f'{v:.0f}次 (<span style="color:#74B9FF">{v-exp_w2:.1f}</span>)</span></div>'
            for nm, v in cold3_w
        )
        st.markdown(f"""
        <div style="background:#1A2634;border:1px solid rgba(0,201,255,0.12);
                    border-radius:10px;padding:14px 16px;margin-top:10px;font-size:12px">
          <div style="color:rgba(224,230,237,0.4);font-size:10px;text-transform:uppercase;
                      letter-spacing:.5px;margin-bottom:8px">窗口摘要</div>
          <div style="margin:3px 0">第 <strong>{w2_start+1}</strong>–<strong>{w2_end}</strong> 期</div>
          <div style="margin:3px 0;color:rgba(224,230,237,0.5)">期望值 {exp_w2:.2f} 次/号</div>
          <div style="margin:10px 0 4px;color:#FF6B6B;font-size:11px;font-weight:600">
            🔥 窗口热号（超期望）</div>
          {hot_html}
          <div style="margin:10px 0 4px;color:#74B9FF;font-size:11px;font-weight:600">
            🧊 窗口冷号（低于期望）</div>
          {cold_html}
        </div>
        """, unsafe_allow_html=True)
    with col_wchart:
        st.pyplot(
            fig_window_delta_bar(specials, w2_start, w2_size,
                                 f'第{w2_start+1}–{w2_end}期'),
            use_container_width=True
        )
        plt.close('all')
        st.markdown(
            ib('窗口越小，随机波动越大，偏差越剧烈。'
               '任意窗口内出现的热号/冷号均为正常统计波动，不代表该号码具有选取价值。',
               'blue'),
            unsafe_allow_html=True
        )

    st.markdown('<br>', unsafe_allow_html=True)

    # ── 🔍 单号码深度档案 ─────────────────────────────────────────────────
    st.markdown(ch('🔍 单号码深度档案',
                   '选择任意号码（1–49），一键查看其完整历史统计数据与出现时间线',
                   '🗂️'), unsafe_allow_html=True)

    col_np, col_nd = st.columns([1, 3])
    with col_np:
        probe_num = st.number_input(
            '查询号码', min_value=1, max_value=49, value=1, step=1,
            key='probe_num',
            help='输入 1–49 任意号码，右侧实时显示完整档案',
        )
        pn = int(probe_num)
        pn_freq  = int(freq_arr[pn - 1])
        pn_gap   = int(gap_arr[pn - 1])
        pn_exp   = avg_freq
        pn_delta = pn_freq - pn_exp
        pn_rank  = int(np.sum(freq_arr > pn_freq)) + 1   # 频次排名（1=最高）
        pn_rank_cold = int(np.sum(gap_arr > pn_gap)) + 1  # 遗漏排名（1=遗漏最长）

        # 生肖/奇偶/大小
        pn_odd  = '奇' if pn % 2 == 1 else '偶'
        pn_size = f'大（>{config.BIG_NUMBER_THRESHOLD}）' if pn > config.BIG_NUMBER_THRESHOLD else f'小（≤{config.BIG_NUMBER_THRESHOLD}）'
        pn_tail = pn % 10
        pn_zodiac = ''
        if 'zodiac' in df_full.columns:
            rows_z = df_full[df_full['special'] == pn]['zodiac'].dropna()
            pn_zodiac = rows_z.iloc[-1] if len(rows_z) > 0 else '—'

        # 历史间隔统计
        pn_idx   = np.where(specials == pn)[0]
        pn_gaps_h = np.diff(pn_idx).tolist() if len(pn_idx) > 1 else []
        pn_max_gap  = int(max(pn_gaps_h)) if pn_gaps_h else 0
        pn_avg_gap  = float(np.mean(pn_gaps_h)) if pn_gaps_h else 0.0
        pn_last_date = (df_full[df_full['special'] == pn]['openTime'].max()
                        .strftime('%Y/%m/%d') if len(pn_idx) > 0 else '—')

        delta_color = '#FF6B6B' if pn_delta > 0 else ('#74B9FF' if pn_delta < 0 else '#A0AEC0')
        gap_color   = '#E74C3C' if pn_gap >= crit_thr else ('#F39C12' if pn_gap >= warn_thr else '#27AE60')

        st.markdown(f"""
        <div style="background:#1A2634;border:1px solid rgba(0,201,255,0.15);
                    border-radius:12px;padding:16px 18px;font-size:13px">
          <div style="font-size:28px;font-weight:700;color:#00C9FF;
                      border-bottom:1px solid rgba(0,201,255,0.15);
                      padding-bottom:10px;margin-bottom:12px">
            No.{pn:02d}
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
            <div>
              <div style="color:rgba(224,230,237,0.42);font-size:10px;letter-spacing:.5px">总出现次数</div>
              <div style="font-weight:700;font-size:16px;margin-top:2px">{pn_freq}次</div>
            </div>
            <div>
              <div style="color:rgba(224,230,237,0.42);font-size:10px;letter-spacing:.5px">vs 理论均值</div>
              <div style="font-weight:700;font-size:16px;margin-top:2px;color:{delta_color}">{pn_delta:+.1f}</div>
            </div>
            <div>
              <div style="color:rgba(224,230,237,0.42);font-size:10px;letter-spacing:.5px">当前遗漏</div>
              <div style="font-weight:700;font-size:16px;margin-top:2px;color:{gap_color}">{pn_gap}期</div>
            </div>
            <div>
              <div style="color:rgba(224,230,237,0.42);font-size:10px;letter-spacing:.5px">最长历史遗漏</div>
              <div style="font-weight:700;font-size:16px;margin-top:2px">{pn_max_gap}期</div>
            </div>
            <div>
              <div style="color:rgba(224,230,237,0.42);font-size:10px;letter-spacing:.5px">频次排名</div>
              <div style="font-weight:700;font-size:16px;margin-top:2px">#{pn_rank}/49</div>
            </div>
            <div>
              <div style="color:rgba(224,230,237,0.42);font-size:10px;letter-spacing:.5px">平均间隔</div>
              <div style="font-weight:700;font-size:16px;margin-top:2px">{pn_avg_gap:.1f}期</div>
            </div>
          </div>
          <div style="margin-top:12px;border-top:1px solid rgba(255,255,255,0.05);
                      padding-top:10px;font-size:12px;color:rgba(224,230,237,0.6);
                      line-height:1.9">
            <span style="color:rgba(224,230,237,0.38)">生肖</span> {ZODIAC_EMOJI.get(pn_zodiac,'')}{pn_zodiac} &nbsp;
            <span style="color:rgba(224,230,237,0.38)">奇偶</span> {pn_odd} &nbsp;
            <span style="color:rgba(224,230,237,0.38)">大小</span> {pn_size}<br>
            <span style="color:rgba(224,230,237,0.38)">尾数</span> {pn_tail} &nbsp;
            <span style="color:rgba(224,230,237,0.38)">末次出现</span> {pn_last_date}
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_nd:
        st.pyplot(fig_number_timeline(specials, pn, lookback=300),
                  use_container_width=True)
        plt.close('all')
        st.markdown(
            ib(f'No.{pn:02d} 在全量 {n} 期中共出现 <strong>{pn_freq}</strong> 次，'
               f'理论均值 {pn_exp:.1f} 次，偏差 <strong style="color:{delta_color}">{pn_delta:+.1f}</strong>。'
               f'当前遗漏 <strong>{pn_gap}</strong> 期'
               + (f'（预警 {warn_thr:.0f}）' if pn_gap >= warn_thr else '') + '。'
               '遗漏长短对独立随机事件无预测价值，不应据此选号。',
               'blue' if pn_gap < warn_thr else 'orange'),
            unsafe_allow_html=True
        )


# ────────────────────────────────────────────────────────────────────────
# Tab 3 — 结构分析
# ────────────────────────────────────────────────────────────────────────
with tab3:
    if not zodiac_cnt:
        st.warning('数据中无 zodiac 生肖字段，请确认 CSV 包含该列')
    else:
        # ── 生肖筛选控件（内联）─────────────────────────────────────────
        _zt1, _zt2 = st.columns([2, 2])
        with _zt1:
            display_mode = st.radio('🐲 展示模式', ['全部12肖', 'Top 5', '单生肖'],
                                     horizontal=True, label_visibility='visible')
        with _zt2:
            sel_zodiac = None
            if display_mode == '单生肖':
                sel_zodiac = st.selectbox('选择生肖', ZODIAC_ORDER,
                                           label_visibility='collapsed')
        st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)

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

    st.markdown('<br>', unsafe_allow_html=True)

    # ── 多号码滚动频率对比 ───────────────────────────────────────────────
    st.markdown(ch('📊 多号码滚动频率对比',
                   f'选择最多 4 个号码，同时查看它们在 {window_n} 期滑动窗口内的频率变化趋势',
                   '📈'), unsafe_allow_html=True)
    _default_nums = list(dict.fromkeys([
        int(np.argmax(freq_arr)) + 1,
        int(np.argmin(freq_arr)) + 1, 7, 42
    ]))[:4]
    nums_sel = st.multiselect(
        '选择追踪的号码（最多 4 个）',
        options=list(range(1, 50)),
        default=_default_nums,
        key='multi_nums',
    )
    if len(nums_sel) > 4:
        st.warning('最多选择 4 个号码，已截取前 4 个')
        nums_sel = nums_sel[:4]
    if nums_sel and len(specials) >= window_n:
        st.pyplot(fig_multi_rolling(specials, window_n, nums_sel),
                  use_container_width=True)
        plt.close('all')
        st.markdown(
            ib('各号码频率线在期望值附近随机波动，窗口越小波动越剧烈。'
               '频率线的升降不代表趋势，彩票每期开奖互相独立。', 'blue'),
            unsafe_allow_html=True
        )
    else:
        st.info(f'数据不足（需 ≥ {window_n} 期），请在侧边栏扩大时间范围或缩小滑动窗口。')

    st.markdown('<br>', unsafe_allow_html=True)

    # ── 时期分块热力图 ───────────────────────────────────────────────────
    st.markdown(ch('🗓 时期分块频率热力图',
                   '将历史数据按固定期数分块，以颜色深浅直观观察各号码出现频率随时间的演变',
                   '🌡️'), unsafe_allow_html=True)
    _chunk_opts = [sz for sz in [20, 30, 50, 100, 150, 200] if n // sz >= 3]
    if not _chunk_opts:
        _chunk_opts = [max(10, n // 4)]
    _chunk_def = _chunk_opts[min(2, len(_chunk_opts) - 1)]
    chunk_size_sel = st.select_slider(
        '分块大小（期/块）',
        options=_chunk_opts,
        value=_chunk_def,
        key='chunk_size_sel',
        help='块越小，时间分辨率越高；块越大，统计越稳定',
    )
    _n_chunks_preview = n // chunk_size_sel
    st.caption(f'当前设置：共 {n} 期 ÷ {chunk_size_sel} 期/块 ≈ {_n_chunks_preview} 块')
    st.pyplot(fig_chunk_heatmap(specials, chunk_size_sel), use_container_width=True)
    plt.close('all')
    st.markdown(
        ib('热力图中任何局部红/蓝区块均为随机波动，不存在统计意义上的"号码周期"或"冷热轮换"规律。',
           'orange'),
        unsafe_allow_html=True
    )

    st.markdown('<br>', unsafe_allow_html=True)

    # ── ⚖️ 双时段频率对比 ────────────────────────────────────────────────
    st.markdown(ch('⚖️ 双时段频率对比',
                   '独立设置两个时间窗口（A / B），对比各号码在两个时段内的每期发生率差异',
                   '🆚'), unsafe_allow_html=True)

    st.markdown("""
    <div style="display:flex;gap:8px;margin-bottom:4px">
      <div style="width:14px;height:14px;background:#00C9FF;border-radius:3px;margin-top:2px"></div>
      <span style="font-size:12px;color:rgba(224,230,237,0.6)">时段 A（蓝色）</span>
      <div style="width:14px;height:14px;background:#F39C12;border-radius:3px;margin-top:2px;margin-left:12px"></div>
      <span style="font-size:12px;color:rgba(224,230,237,0.6)">时段 B（橙色）</span>
    </div>
    """, unsafe_allow_html=True)

    col_dwa, col_dwb = st.columns(2)
    _dw_max = min(n, 500)
    with col_dwa:
        st.markdown('<div style="color:#00C9FF;font-size:12px;font-weight:600;'
                    'margin-bottom:4px">🔵 时段 A</div>', unsafe_allow_html=True)
        dw_a_size = st.slider('A 窗口大小', 20, _dw_max, min(100, _dw_max),
                               step=10, key='dw_a_size')
        dw_a_end  = st.slider('A 末尾位置', dw_a_size, n, n,
                               step=1, key='dw_a_end',
                               help='右端=最新，向左拖动选取历史窗口')
    with col_dwb:
        st.markdown('<div style="color:#F39C12;font-size:12px;font-weight:600;'
                    'margin-bottom:4px">🟠 时段 B</div>', unsafe_allow_html=True)
        dw_b_size = st.slider('B 窗口大小', 20, _dw_max, min(100, _dw_max),
                               step=10, key='dw_b_size')
        dw_b_end  = st.slider('B 末尾位置', dw_b_size, n, max(dw_b_size, n - min(100, n // 2)),
                               step=1, key='dw_b_end',
                               help='默认偏早，可拖到任意位置与 A 对比')

    dw_a_start = dw_a_end - dw_a_size
    dw_b_start = dw_b_end - dw_b_size

    # 摘要信息行
    _wa_sp = specials[dw_a_start:dw_a_end]
    _wb_sp = specials[dw_b_start:dw_b_end]
    _rate_a = np.array([Counter(_wa_sp.tolist()).get(i, 0) / max(dw_a_size, 1) for i in range(1, 50)])
    _rate_b = np.array([Counter(_wb_sp.tolist()).get(i, 0) / max(dw_b_size, 1) for i in range(1, 50)])
    _top_a  = int(np.argmax(_rate_a)) + 1
    _top_b  = int(np.argmax(_rate_b)) + 1
    _diff   = _rate_a - _rate_b
    _most_diff = int(np.argmax(np.abs(_diff))) + 1

    col_di1, col_di2, col_di3 = st.columns(3)
    def _mini_kpi(col, val, lbl, color='#00C9FF'):
        with col:
            st.markdown(f"""
            <div style="background:#1A2634;border:1px solid rgba(255,255,255,0.07);
                        border-radius:8px;padding:10px 14px;text-align:center">
              <div style="font-size:16px;font-weight:700;color:{color}">{val}</div>
              <div style="font-size:10px;color:rgba(224,230,237,0.4);
                          margin-top:3px;text-transform:uppercase;letter-spacing:.4px">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)
    _mini_kpi(col_di1, f'第{dw_a_start+1}–{dw_a_end}期 | No.{_top_a}最热',
              f'时段 A（{dw_a_size}期）', '#00C9FF')
    _mini_kpi(col_di2, f'第{dw_b_start+1}–{dw_b_end}期 | No.{_top_b}最热',
              f'时段 B（{dw_b_size}期）', '#F39C12')
    _mini_kpi(col_di3, f'No.{_most_diff}差异最大',
              '两时段发生率差异最大号码', '#A29BFE')

    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
    st.pyplot(
        fig_dual_window_compare(specials, dw_a_start, dw_a_size, dw_b_start, dw_b_size),
        use_container_width=True
    )
    plt.close('all')
    st.markdown(
        ib(f'时段 A（第{dw_a_start+1}–{dw_a_end}期）vs 时段 B（第{dw_b_start+1}–{dw_b_end}期）。'
           '两时段差异源于随机波动，不代表号码分布发生了真实变化。'
           '如需统计检验，可使用 Tab 5 的卡方窗口敏感性分析。',
           'orange'),
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

    st.markdown('<br>', unsafe_allow_html=True)

    # ── 卡方检验窗口敏感性分析 ───────────────────────────────────────────
    st.markdown(ch('③ 卡方检验窗口敏感性分析',
                   '观察卡方统计量与 p 值如何随窗口大小（样本量）变化，揭示统计显著性对样本量的依赖性',
                   '📈'), unsafe_allow_html=True)
    col_ws1, col_ws2 = st.columns(2)
    with col_ws1:
        w5_min  = st.slider('起始窗口（期）', 30, max(31, n - 1), min(50, n // 4),
                             step=10, key='w5_min')
    with col_ws2:
        w5_step = st.slider('步长（期）', 5, 50, 10, step=5, key='w5_step')
    _win_sizes = list(range(w5_min, n + 1, w5_step))
    if len(_win_sizes) > 120:
        _win_sizes = _win_sizes[::max(1, len(_win_sizes) // 120)]
    if len(_win_sizes) >= 3:
        _chi2_ws, _p_ws = [], []
        for _ws in _win_sizes:
            _sp_ws   = specials[-_ws:]
            _freq_ws = np.array([Counter(_sp_ws.tolist()).get(i, 0) for i in range(1, 50)])
            _c2, _p2 = chisquare(_freq_ws, np.full(49, _freq_ws.mean()))
            _chi2_ws.append(_c2)
            _p_ws.append(_p2)
        st.pyplot(
            fig_chi2_vs_window(_win_sizes, _chi2_ws, _p_ws, sig_level),
            use_container_width=True
        )
        plt.close('all')
        _cross_alpha = sum(1 for p in _p_ws if p <= sig_level)
        st.markdown(
            ib(f'在 {len(_win_sizes)} 个窗口中，'
               f'<strong>{_cross_alpha}</strong> 个（{_cross_alpha/len(_win_sizes)*100:.0f}%）的 p 值低于 α={sig_level}。'
               '统计显著性高度依赖窗口大小选取；当 p 值在显著与不显著之间游走时，'
               '结论本身就不可靠——这是随机性的有力证据。',
               'blue' if _cross_alpha / len(_win_sizes) < 0.5 else 'orange'),
            unsafe_allow_html=True
        )
    else:
        st.info('数据量不足以进行窗口敏感性分析，请扩大时间范围。')


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
# Tab 7 — 分析报告（专业统计报告格式）
# ────────────────────────────────────────────────────────────────────────
with tab7:

    # ── 执行摘要 ─────────────────────────────────────────────────────────
    overall_pass = pass_cnt >= 2
    verd_c   = '#27AE60' if overall_pass else '#E74C3C'
    verd_bg  = 'rgba(39,174,96,0.08)'  if overall_pass else 'rgba(231,76,60,0.08)'
    verd_bd  = 'rgba(39,174,96,0.28)'  if overall_pass else 'rgba(231,76,60,0.28)'
    verd_ic  = '✅' if overall_pass else '⚠️'
    verd_tx  = '随机均匀分布假设成立' if overall_pass else '数据存在统计显著偏离'
    verd_sub = (
        f'在 α={sig_level} 显著性水平下，{pass_cnt}/3 项核心假设检验未能拒绝零假设，'
        '数据整体符合独立随机均匀分布特征，与彩票纯随机抽签机制理论一致。'
        if overall_pass else
        f'在 α={sig_level} 显著性水平下，{3-pass_cnt}/3 项假设检验拒绝零假设，'
        '建议扩大样本量后重新评估；当前偏离可能源于随机波动而非系统性规律。'
    )
    if cramers_v < 0.1:
        cv_level, cv_color = '微弱效应 (Negligible)', '#27AE60'
    elif cramers_v < 0.3:
        cv_level, cv_color = '小效应 (Small)', '#F39C12'
    else:
        cv_level, cv_color = '中等效应 (Medium)', '#E74C3C'

    st.markdown(f"""
    <div style="background:{verd_bg};border:1px solid {verd_bd};
                border-radius:14px;padding:22px 26px;margin-bottom:6px">
      <div style="display:flex;align-items:flex-start;gap:16px">
        <div style="font-size:30px;line-height:1.1">{verd_ic}</div>
        <div style="flex:1">
          <div style="font-size:17px;font-weight:700;color:{verd_c};margin-bottom:7px">
            执行摘要 — {verd_tx}</div>
          <div style="font-size:13px;color:rgba(224,230,237,0.76);line-height:1.75">
            {verd_sub}</div>
          <div style="display:flex;gap:20px;margin-top:16px;flex-wrap:wrap">
            <div style="text-align:center;min-width:60px">
              <div style="font-size:22px;font-weight:700;color:{verd_c}">{pass_cnt}/3</div>
              <div style="font-size:10px;color:rgba(224,230,237,0.38);text-transform:uppercase;
                          letter-spacing:.5px;margin-top:3px">检验通过</div>
            </div>
            <div style="text-align:center;min-width:60px">
              <div style="font-size:22px;font-weight:700;color:#00C9FF">{n}</div>
              <div style="font-size:10px;color:rgba(224,230,237,0.38);text-transform:uppercase;
                          letter-spacing:.5px;margin-top:3px">分析期数</div>
            </div>
            <div style="text-align:center;min-width:60px">
              <div style="font-size:22px;font-weight:700;color:#E0E6ED">{sig_level}</div>
              <div style="font-size:10px;color:rgba(224,230,237,0.38);text-transform:uppercase;
                          letter-spacing:.5px;margin-top:3px">显著性 α</div>
            </div>
            <div style="text-align:center;min-width:80px">
              <div style="font-size:22px;font-weight:700;color:{cv_color}">{cramers_v:.4f}</div>
              <div style="font-size:10px;color:rgba(224,230,237,0.38);text-transform:uppercase;
                          letter-spacing:.5px;margin-top:3px">Cramér's V</div>
            </div>
            <div style="text-align:center;min-width:80px">
              <div style="font-size:22px;font-weight:700;color:#A29BFE">{cv_pct:.1f}%</div>
              <div style="font-size:10px;color:rgba(224,230,237,0.38);text-transform:uppercase;
                          letter-spacing:.5px;margin-top:3px">变异系数 CV</div>
            </div>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Section I: 描述性统计 ──────────────────────────────────────────────
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown(ch('I. 描述性统计',
                   f'特码频率分布基本统计特征（k = 49 个号码，n = {n} 期）', '📐'),
                unsafe_allow_html=True)

    freq_median  = float(np.median(freq_arr))
    freq_iqr     = float(np.percentile(freq_arr, 75) - np.percentile(freq_arr, 25))
    freq_ci95_hw = 1.96 * freq_arr.std() / np.sqrt(49)
    skew_interp  = '近似对称' if abs(freq_skew) < 0.5 else ('正偏' if freq_skew > 0 else '负偏')
    kurt_interp  = '近似正态' if abs(freq_kurt) < 0.5 else ('厚尾' if freq_kurt > 0 else '薄尾')
    cv_interp    = '低离散' if cv_pct < 15 else ('中等离散' if cv_pct < 30 else '高离散')

    def _th7(v):
        return (f'<th style="background:#1A2634;color:#00C9FF;padding:9px 14px;'
                f'text-align:left;border-bottom:1px solid rgba(0,201,255,0.18);'
                f'font-size:11px;white-space:nowrap">{v}</th>')
    def _td7(v):
        return (f'<td style="padding:7px 14px;border-bottom:1px solid '
                f'rgba(255,255,255,0.04);font-size:13px">{v}</td>')
    def _tdv7(v):
        return (f'<td style="padding:7px 14px;border-bottom:1px solid '
                f'rgba(255,255,255,0.04);font-weight:600;color:#E0E6ED;'
                f'text-align:right;font-size:13px">{v}</td>')
    def _tds7(v):
        return (f'<td style="padding:7px 14px;border-bottom:1px solid '
                f'rgba(255,255,255,0.04);color:rgba(224,230,237,0.42);'
                f'font-size:11px">{v}</td>')

    desc_data = [
        ('样本量（期数 n）',    f'{n}',                                 '开奖记录总数'),
        ('号码均值 μ',          f'{avg_freq:.3f}',                       '期望出现次数 = n / 49'),
        ('标准差 σ',            f'{freq_arr.std():.3f}',                 '频次的均方根偏差'),
        ('变异系数 CV',         f'{cv_pct:.2f}%',                        f'σ/μ×100% → {cv_interp}'),
        ('中位数',              f'{freq_median:.1f}',                    '分布中心的稳健估计'),
        ('四分位距 IQR',        f'{freq_iqr:.1f}',                       'Q₃ − Q₁，中间50%散布范围'),
        ('均值 95% CI 半宽',    f'±{freq_ci95_hw:.3f}',                  '1.96 × σ / √49'),
        ('偏度 γ₁',             f'{freq_skew:.4f}',                      f'→ {skew_interp}（|γ₁| < 0.5 为近似对称）'),
        ('超额峰度 γ₂',         f'{freq_kurt:.4f}',                      f'→ {kurt_interp}（正态分布 = 0）'),
        ('最大值 / 最小值',     f'{int(freq_arr.max())} / {int(freq_arr.min())}', '号码出现次数极值'),
        ('极差 Range',          f'{int(freq_arr.max() - freq_arr.min())}', 'max − min'),
        ('间隔均值 E[G]',       f'{avg_freq:.2f}',                       '各号平均遗漏期数 ≈ n/49'),
        ('间隔标准差 σ[G]',     f'{gap_std:.2f}',                        '遗漏期数的离散程度'),
        ('间隔四分位 Q₁/Q₃',   f'{gap_p25:.1f} / {gap_p75:.1f}',        '遗漏期数四分位数'),
    ]

    desc_html = ('<table style="width:100%;border-collapse:collapse;margin:12px 0">'
                 + '<tr>' + _th7('统计量') + _th7('数值') + _th7('说明') + '</tr>')
    for nm, val, note in desc_data:
        desc_html += f'<tr>{_td7(nm)}{_tdv7(val)}{_tds7(note)}</tr>'
    desc_html += '</table>'
    st.markdown(desc_html, unsafe_allow_html=True)

    # ── Section II: 假设检验汇总表 ───────────────────────────────────────
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown(ch('II. 假设检验汇总',
                   f'显著性水平 α = {sig_level}，共 4 项检验；建议结合效应量解读 p 值', '🔬'),
                unsafe_allow_html=True)

    def _pfmt7(p):
        return f'{p:.4f}' if p >= 0.0001 else '< 0.0001'

    def _dec7(passed):
        c = '#27AE60' if passed else '#E74C3C'
        txt = '不拒绝 H₀' if passed else '拒绝 H₀'
        return f'<span style="color:{c};font-weight:700">{txt}</span>'

    n_zod_df7 = len(zod_obs) - 1 if len(zod_obs) >= 2 else 0
    acf_pass7 = acf_exceed == 0

    th_css = ('background:#1A2634;color:#00C9FF;padding:9px 11px;text-align:left;'
              'border-bottom:1px solid rgba(0,201,255,0.18);font-size:11px;white-space:nowrap')
    td_css = 'padding:8px 11px;border-bottom:1px solid rgba(255,255,255,0.04);font-size:12px'

    test_html = f'<table style="width:100%;border-collapse:collapse;margin:12px 0">'
    test_html += (f'<tr>'
                  f'<th style="{th_css}">检验项目</th>'
                  f'<th style="{th_css}">H₀ 假设</th>'
                  f'<th style="{th_css}">统计量值</th>'
                  f'<th style="{th_css}">p 值</th>'
                  f'<th style="{th_css}">临界值</th>'
                  f'<th style="{th_css}">决策</th>'
                  f'<th style="{th_css}">效应量</th>'
                  f'</tr>')
    test_rows = [
        ('卡方拟合优度检验',
         'H₀：49号出现概率均等（1/49）',
         f'χ²(48) = {chi2_stat:.4f}',
         _pfmt7(chi2_p),
         f'{chi2_crit:.3f}',
         _dec7(chi2_pass),
         f"V = {cramers_v:.4f}<br><small style='color:{cv_color}'>{cv_level}</small>"),
        ('KS 指数分布拟合检验',
         'H₀：号码间隔服从指数分布',
         f'D = {ks_stat:.4f}',
         _pfmt7(ks_p),
         '—',
         _dec7(ks_pass),
         '—'),
        (f'自相关检验（ACF，{acf_lags} 滞后期）',
         'H₀：序列无自相关',
         f'{acf_exceed}/{acf_lags} 滞后超 95% CI',
         '—',
         f'±{conf_interval:.4f}',
         _dec7(acf_pass7),
         '—'),
        ('生肖均匀性卡方检验',
         'H₀：12生肖出现概率均等（1/12）',
         f'χ²({n_zod_df7}) = {zod_chi2_stat:.4f}',
         _pfmt7(zod_chi2_p),
         f'{zod_chi2_crit:.3f}',
         _dec7(zod_chi2_pass),
         '—'),
    ]
    for row in test_rows:
        test_html += '<tr>' + ''.join(f'<td style="{td_css}">{cell}</td>' for cell in row) + '</tr>'
    test_html += '</table>'
    st.markdown(test_html, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:11px;color:rgba(224,230,237,0.35);margin-top:4px;line-height:1.6">
      ⚠️ 注：同时执行多项检验存在 I 类错误累积风险（未进行 Bonferroni 校正），
      建议优先参考效应量而非单纯依赖 p 值。
    </div>""", unsafe_allow_html=True)

    # ── Section III: 效应量解读 ───────────────────────────────────────────
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown(ch('III. 效应量解读',
                   '效应量衡量偏离的实际显著性，独立于样本量，是解读统计结论的关键补充', '📏'),
                unsafe_allow_html=True)

    cv_desc_txt = (
        f"V = {cramers_v:.4f} < 0.1，号码频率偏差极小，在实际意义上可忽略不计。"
        "即使 p 值达到统计显著，效应量表明偏离程度在现实中无任何可操作价值。"
        if cramers_v < 0.1 else
        f"V = {cramers_v:.4f}（0.1–0.3），存在小幅偏离，实际意义仍然有限，"
        "不足以支持任何选号策略或预测行为。"
        if cramers_v < 0.3 else
        f"V = {cramers_v:.4f} ≥ 0.3，偏离程度较为明显，建议检查数据完整性与质量。"
    )
    cv_str_txt = (
        f"CV = {cv_pct:.2f}%，"
        + ("各号频率高度均匀，接近理论等概率值。" if cv_pct < 15
           else "频率分布存在一定不均匀性，但属于正常统计波动范围。")
        + f" 偏度 γ₁ = {freq_skew:.4f}（{skew_interp}），"
        f"超额峰度 γ₂ = {freq_kurt:.4f}（{kurt_interp}）。"
    )

    col_eff1, col_eff2 = st.columns(2)
    with col_eff1:
        st.markdown(f"""
        <div style="background:#1A2634;border:1px solid rgba(0,201,255,0.1);
                    border-radius:10px;padding:18px 20px;height:100%">
          <div style="font-size:10px;color:rgba(224,230,237,0.38);text-transform:uppercase;
                      letter-spacing:.6px;margin-bottom:8px">Cramér's V（卡方效应量）</div>
          <div style="font-size:20px;font-weight:700;color:{cv_color};margin-bottom:8px">
            {cramers_v:.4f} — {cv_level}</div>
          <div style="font-size:12px;color:rgba(224,230,237,0.65);line-height:1.7">
            {cv_desc_txt}</div>
          <div style="margin-top:10px;font-size:11px;color:rgba(224,230,237,0.3)">
            公式：V = √( χ² / (n × (k−1)) )，k=49，n={n}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_eff2:
        st.markdown(f"""
        <div style="background:#1A2634;border:1px solid rgba(0,201,255,0.1);
                    border-radius:10px;padding:18px 20px;height:100%">
          <div style="font-size:10px;color:rgba(224,230,237,0.38);text-transform:uppercase;
                      letter-spacing:.6px;margin-bottom:8px">变异系数 CV + 分布形态</div>
          <div style="font-size:20px;font-weight:700;color:#A29BFE;margin-bottom:8px">
            {cv_pct:.2f}%</div>
          <div style="font-size:12px;color:rgba(224,230,237,0.65);line-height:1.7">
            {cv_str_txt}</div>
          <div style="margin-top:10px;font-size:11px;color:rgba(224,230,237,0.3)">
            公式：CV = σ/μ × 100%，μ = {avg_freq:.2f}，σ = {freq_arr.std():.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Section IV: 生肖分布统计检验 ─────────────────────────────────────
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown(ch('IV. 生肖分布统计检验',
                   '对12生肖出现频率进行卡方均匀性检验，验证生肖标注的随机性', '🐉'),
                unsafe_allow_html=True)

    r7_top1 = max(zodiac_cnt, key=zodiac_cnt.get) if zodiac_cnt else '-'
    r7_bot1 = min(zodiac_cnt, key=zodiac_cnt.get) if zodiac_cnt else '-'
    zod_exp = n / 12

    col_z1, col_z2 = st.columns([1, 2])
    with col_z1:
        zod_c7 = '#27AE60' if zod_chi2_pass else '#E74C3C'
        st.markdown(f"""
        <div style="background:#1A2634;border:1px solid rgba(0,201,255,0.1);
                    border-radius:10px;padding:18px 20px">
          <div style="font-size:10px;color:rgba(224,230,237,0.38);letter-spacing:.5px;
                      text-transform:uppercase;margin-bottom:10px">生肖卡方检验</div>
          <div style="font-size:12px;color:rgba(224,230,237,0.6);margin-bottom:8px">
            H₀：12生肖出现概率均等（1/12）</div>
          <div style="margin:5px 0;font-size:13px">
            <span style="color:rgba(224,230,237,0.45)">χ²({n_zod_df7}) = </span>
            <strong>{zod_chi2_stat:.4f}</strong></div>
          <div style="margin:5px 0;font-size:13px">
            <span style="color:rgba(224,230,237,0.45)">p 值 = </span>
            <strong>{_pfmt7(zod_chi2_p)}</strong></div>
          <div style="margin:5px 0;font-size:13px">
            <span style="color:rgba(224,230,237,0.45)">α = {sig_level}</span>
            &nbsp;|&nbsp;
            <span style="color:rgba(224,230,237,0.45)">临界值 = </span>
            <strong>{zod_chi2_crit:.3f}</strong></div>
          <div style="margin-top:14px;font-size:13px;font-weight:700;color:{zod_c7}">
            {'✅ 不拒绝 H₀ — 生肖分布均匀' if zod_chi2_pass else '❌ 拒绝 H₀ — 生肖分布偏离均匀'}</div>
          <div style="margin-top:6px;font-size:11px;color:rgba(224,230,237,0.35)">
            理论均值：{zod_exp:.1f} 次/生肖<br>
            最高：{ZODIAC_EMOJI.get(r7_top1,'')}{r7_top1}（{zodiac_cnt.get(r7_top1,0)}次）<br>
            最低：{ZODIAC_EMOJI.get(r7_bot1,'')}{r7_bot1}（{zodiac_cnt.get(r7_bot1,0)}次）
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_z2:
        zod_table_html = (
            '<table style="width:100%;border-collapse:collapse;font-size:12px">'
            f'<tr>'
            f'<th style="{th_css}">生肖</th>'
            f'<th style="{th_css};text-align:right">频次</th>'
            f'<th style="{th_css};text-align:right">占比</th>'
            f'<th style="{th_css};text-align:right">偏差%</th>'
            f'<th style="{th_css}">相对理论均值</th>'
            f'</tr>'
        )
        for z in ZODIAC_ORDER:
            if z not in zodiac_cnt:
                continue
            c_z   = zodiac_cnt[z]
            dev_p = (c_z - zod_exp) / zod_exp * 100
            sign_z = '+' if dev_p >= 0 else ''
            bar_w  = min(100, int(c_z / max(zodiac_cnt.values()) * 100))
            bar_c  = '#27AE60' if abs(dev_p) < 8 else ('#F39C12' if abs(dev_p) < 15 else '#E74C3C')
            icon_z = ZODIAC_EMOJI.get(z, '')
            zod_table_html += (
                f'<tr>'
                f'<td style="{td_css}">{icon_z} {z}</td>'
                f'<td style="{td_css};text-align:right;font-weight:600">{c_z}</td>'
                f'<td style="{td_css};text-align:right">{c_z/n*100:.1f}%</td>'
                f'<td style="{td_css};text-align:right;color:{"#27AE60" if dev_p>=0 else "#E74C3C"}">'
                f'{sign_z}{dev_p:.1f}%</td>'
                f'<td style="{td_css}">'
                f'<div style="background:rgba(255,255,255,0.07);border-radius:3px;height:7px">'
                f'<div style="background:{bar_c};width:{bar_w}%;height:100%;border-radius:3px">'
                f'</div></div></td>'
                f'</tr>'
            )
        zod_table_html += '</table>'
        st.markdown(zod_table_html, unsafe_allow_html=True)

    # ── Section V: 研究结论（正式统计语言）──────────────────────────────
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown(ch('V. 研究结论',
                   '基于假设检验框架的形式化推断，采用规范统计学表述', '📝'),
                unsafe_allow_html=True)

    concl_pro = []
    # ① Chi-square
    if chi2_pass:
        concl_pro.append(('ok',
            f'<strong>① 均匀性检验</strong>：在 α={sig_level} 显著性水平下，'
            f'卡方统计量 χ²(48) = {chi2_stat:.4f}（p = {_pfmt7(chi2_p)}，临界值 {chi2_crit:.3f}），'
            f'p > α，<em>未能拒绝零假设</em>。49个号码出现频率与理论均匀分布无统计显著差异。'
            f' 效应量 Cramér\'s V = {cramers_v:.4f}（{cv_level}），实际偏离可忽略不计。'))
    else:
        concl_pro.append(('warn',
            f'<strong>① 均匀性检验</strong>：在 α={sig_level} 显著性水平下，'
            f'卡方统计量 χ²(48) = {chi2_stat:.4f}（p = {_pfmt7(chi2_p)}），'
            f'p ≤ α，<em>拒绝零假设</em>。然而效应量 Cramér\'s V = {cramers_v:.4f}（{cv_level}），'
            '偏离程度在实际意义上可能可忽略，可能源于大样本的统计敏感性。'))
    # ② KS
    if ks_pass:
        concl_pro.append(('ok',
            f'<strong>② 独立性检验（KS）</strong>：KS 统计量 D = {ks_stat:.4f}（p = {_pfmt7(ks_p)}），'
            f'p > α，<em>未能拒绝零假设</em>。号码出现间隔符合指数分布，'
            '支持各期开奖相互独立的随机过程假设（即历史状态对未来无影响）。'))
    else:
        concl_pro.append(('warn',
            f'<strong>② 独立性检验（KS）</strong>：KS 统计量 D = {ks_stat:.4f}（p = {_pfmt7(ks_p)}），'
            f'p ≤ α，<em>拒绝零假设</em>。间隔分布存在统计偏离，'
            '可能由样本量、离散化误差或边界效应引起，建议扩大样本量重新评估。'))
    # ③ ACF
    if acf_exceed == 0:
        concl_pro.append(('ok',
            f'<strong>③ 自相关分析</strong>：检验 {acf_lags} 个滞后期，'
            f'0 个 ACF 系数超出 95% 置信区间（±{conf_interval:.4f}），'
            '<em>未能拒绝无自相关零假设</em>。序列各期无统计显著的线性依赖关系。'))
    else:
        concl_pro.append(('warn',
            f'<strong>③ 自相关分析</strong>：检验 {acf_lags} 个滞后期，'
            f'{acf_exceed} 个 ACF 系数（{acf_exceed/acf_lags*100:.0f}%）超出 95% CI（±{conf_interval:.4f}）。'
            '在随机序列中，理论上约有 5% 的滞后期偶然超出 CI，当前比例与此一致，不具实际显著意义。'))
    # ④ Zodiac
    if zod_chi2_pass:
        concl_pro.append(('ok',
            f'<strong>④ 生肖分布检验</strong>：χ²({n_zod_df7}) = {zod_chi2_stat:.4f}'
            f'（p = {_pfmt7(zod_chi2_p)}，临界值 {zod_chi2_crit:.3f}），'
            '<em>未能拒绝零假设</em>。12生肖出现频率符合均匀分布，无系统性偏向。'))
    else:
        concl_pro.append(('warn',
            f'<strong>④ 生肖分布检验</strong>：χ²({n_zod_df7}) = {zod_chi2_stat:.4f}'
            f'（p = {_pfmt7(zod_chi2_p)}），<em>拒绝零假设</em>。'
            '生肖分布存在统计偏离，但偏离幅度有限，不具可操作预测价值。'))
    # ⑤ Overall
    concl_pro.append(('ok' if overall_pass else 'warn',
        f'<strong>⑤ 综合评估</strong>：{pass_cnt}/3 项核心假设检验通过。'
        + ('在当前分析框架下，历史数据整体符合独立随机均匀分布特征，与彩票纯随机机制理论一致。'
           '所有统计结论仅描述历史数据特征，彩票开奖为独立随机事件，任何历史规律对未来无影响。'
           if overall_pass else
           '部分检验结果存在偏离，但结合效应量分析，实际偏离程度有限。'
           '彩票开奖在机制上为独立随机事件，任何历史数据均无预测价值。')))

    for level, text in concl_pro:
        cls = 'concl-ok' if level == 'ok' else 'concl-warn'
        st.markdown(f'<div class="concl-item {cls}">{text}</div>',
                    unsafe_allow_html=True)

    # ── Section VI: 研究局限性 ────────────────────────────────────────────
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown(ch('VI. 研究局限性',
                   '本分析的方法论约束与注意事项，是科学解读结论的必要背景', '⚠️'),
                unsafe_allow_html=True)

    limitations = [
        (f'<strong>样本量约束</strong>：本次分析基于 {n} 期历史数据。'
         '较小样本可能导致统计功效不足，无法检出微小偏离；'
         '较大样本则因统计过敏感而拒绝零假设（即便效应量极微弱）。'
         '增加期数可提升检验稳定性。'),
        ('<strong>多重检验问题</strong>：本报告同时执行 4 项假设检验，'
         '未进行 Bonferroni 或 FDR 校正，存在 I 类错误（误判显著）的累积风险。'
         '建议优先参考效应量（Cramér\'s V、CV）而非单纯依赖 p 值。'),
        ('<strong>模型假设局限</strong>：KS 检验假设连续型指数分布，'
         '而实际间隔数据为离散整数，存在系统性检验偏差。'
         'ACF 仅能检验线性依赖关系，无法探测非线性时序结构或高阶相关性。'),
        ('<strong>随机性本质</strong>：即使所有检验均显示统计偏离，'
         '彩票开奖在物理与制度层面保证每期独立性。'
         '统计偏离 ≠ 可预测性 ≠ 可操作性。任何"热号/冷号"策略均无统计依据。'),
        ('<strong>研究性质声明</strong>：本分析为纯描述性统计，'
         '无任何预测、选号或投资建议功能。全部结论仅供学术与教育目的。'),
    ]
    lim_html = '<div style="background:#1A2634;border:1px solid rgba(255,255,255,0.06);border-radius:10px;padding:14px 18px">'
    for i, lim in enumerate(limitations, 1):
        border = '' if i == len(limitations) else 'border-bottom:1px solid rgba(255,255,255,0.04);'
        lim_html += (
            f'<div style="display:flex;gap:12px;padding:10px 0;{border}">'
            f'<div style="min-width:22px;height:22px;background:rgba(0,201,255,0.1);'
            f'border-radius:50%;display:flex;align-items:center;justify-content:center;'
            f'font-size:11px;color:#00C9FF;font-weight:700;margin-top:2px;flex-shrink:0">{i}</div>'
            f'<div style="font-size:13px;color:rgba(224,230,237,0.62);line-height:1.75">{lim}</div>'
            f'</div>'
        )
    lim_html += '</div>'
    st.markdown(lim_html, unsafe_allow_html=True)

    # ── 导出报告 ─────────────────────────────────────────────────────────
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown(ch('导出报告',
                   '生成包含描述性统计、假设检验汇总、效应量及正式结论的完整 HTML 报告', '📥'),
                unsafe_allow_html=True)

    if st.button('⚙️ 生成专业 HTML 分析报告', type='primary', use_container_width=False):
        with st.spinner('正在渲染报告，请稍候...'):
            html_content = generate_html_report(
                df_full, freq_arr, avg_freq, chi2_stat, chi2_p,
                ks_p, gap_arr, zodiac_cnt, sig_level,
                specials, n, latest, pass_cnt,
                cramers_v=cramers_v, cv_pct=cv_pct,
                freq_skew=freq_skew, freq_kurt=freq_kurt,
                gap_std=gap_std, gap_p25=gap_p25, gap_p75=gap_p75,
                ks_stat=ks_stat, chi2_crit=chi2_crit,
                zod_chi2_stat=zod_chi2_stat, zod_chi2_p=zod_chi2_p,
                zod_chi2_pass=zod_chi2_pass, zod_chi2_crit=zod_chi2_crit,
                zod_obs=zod_obs, acf_exceed=acf_exceed, conf_interval=conf_interval,
                acf_lags=acf_lags,
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
