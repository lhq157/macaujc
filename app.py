"""
app.py — 特码统计分析系统 · Streamlit Web 界面
用法：streamlit run app.py
"""

import glob
import os
import sys
import subprocess
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import streamlit as st

# 中文字体 & 深色主题图表
matplotlib.rcParams['font.sans-serif'] = ['PingFang SC', 'Hiragino Sans GB',
                                           'STHeiti', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['figure.facecolor']  = '#1A2634'
matplotlib.rcParams['axes.facecolor']    = '#1A2634'
matplotlib.rcParams['axes.edgecolor']    = '#2A3F54'
matplotlib.rcParams['axes.labelcolor']   = '#A0AEC0'
matplotlib.rcParams['xtick.color']       = '#A0AEC0'
matplotlib.rcParams['ytick.color']       = '#A0AEC0'
matplotlib.rcParams['text.color']        = '#E0E6ED'
matplotlib.rcParams['grid.color']        = '#2A3F54'
matplotlib.rcParams['axes.titlecolor']   = '#E0E6ED'
from scipy.stats import chisquare

import config

# ── 环境检测（本地 or 云端）──────────────────────────────────────────────
IS_CLOUD = not os.access(os.path.join(os.path.dirname(__file__), 'output'), os.W_OK)

# ── 页面配置（必须第一行）────────────────────────────────────────────────
st.set_page_config(
    page_title='特码统计分析系统',
    page_icon='📊',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ── 全局样式 ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── 字体 & 基础 ── */
  html, body, [class*="css"] {
    font-family: -apple-system, 'PingFang SC', 'Helvetica Neue', sans-serif;
  }

  /* ── 顶部标题区 ── */
  .dashboard-header {
    background: linear-gradient(135deg, #0F2744 0%, #1A3A5C 100%);
    border-radius: 14px;
    padding: 28px 32px 22px;
    margin-bottom: 24px;
    border: 1px solid rgba(0,201,255,0.15);
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
  }
  .dashboard-header h1 {
    font-size: 26px;
    font-weight: 700;
    color: #fff;
    margin: 0 0 6px;
    letter-spacing: .5px;
  }
  .dashboard-header p {
    font-size: 13px;
    color: rgba(255,255,255,0.5);
    margin: 0;
  }

  /* ── 统计卡片 ── */
  .stat-card {
    background: linear-gradient(145deg, #1A2E44, #152438);
    border-radius: 12px;
    padding: 20px 16px;
    text-align: center;
    border: 1px solid rgba(0,201,255,0.12);
    box-shadow: 0 2px 16px rgba(0,0,0,0.3);
    transition: transform .15s;
  }
  .stat-card:hover { transform: translateY(-2px); }
  .stat-num {
    font-size: 30px;
    font-weight: 700;
    color: #00C9FF;
    letter-spacing: -.5px;
  }
  .stat-lbl {
    font-size: 11px;
    color: rgba(224,230,237,0.5);
    margin-top: 6px;
    text-transform: uppercase;
    letter-spacing: .8px;
  }

  /* ── 警告条 ── */
  .warn-box {
    background: rgba(243,156,18,0.1);
    border-left: 3px solid #F39C12;
    padding: 9px 14px;
    border-radius: 0 6px 6px 0;
    font-size: 12px;
    color: rgba(224,230,237,0.7);
    margin-bottom: 16px;
  }

  /* ── 分节标题 ── */
  .section-title {
    font-size: 15px;
    font-weight: 600;
    color: #00C9FF;
    letter-spacing: .5px;
    padding: 6px 0 2px;
    border-bottom: 1px solid rgba(0,201,255,0.15);
    margin-bottom: 14px;
  }

  /* ── 侧边栏美化 ── */
  section[data-testid="stSidebar"] {
    background: #111D2C !important;
    border-right: 1px solid rgba(0,201,255,0.08);
  }
  section[data-testid="stSidebar"] .stMarkdown p {
    color: rgba(224,230,237,0.7);
    font-size: 13px;
  }

  /* ── 图表背景透明 ── */
  .stPlotlyChart, .stImage { background: transparent !important; }

  /* ── 隐藏 Streamlit 品牌 ── */
  #MainMenu, footer { visibility: hidden; }
  header[data-testid="stHeader"] { background: transparent; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# 数据加载
# ══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_data():
    csvs = sorted(glob.glob(os.path.join(config.OUTPUT_DIR, '????????.csv')))
    if not csvs:
        return None, None
    path = csvs[-1]
    df = pd.read_csv(path, dtype={'expect': str})
    df['openTime'] = pd.to_datetime(df['openTime'])
    df = df.sort_values('openTime').reset_index(drop=True)
    return df, path


def fetch_data():
    """调用 main.py 拉取最新数据"""
    with st.spinner('正在拉取最新数据...'):
        result = subprocess.run(
            [sys.executable, 'main.py'],
            capture_output=True, text=True, cwd=os.path.dirname(__file__)
        )
    if result.returncode == 0:
        st.cache_data.clear()
        st.success('✅ 数据更新成功')
    else:
        st.error(f'❌ 拉取失败：{result.stderr[-300:]}')


# ══════════════════════════════════════════════════════════════════════════
# 侧边栏
# ══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title('📊 特码统计分析系统')
    st.divider()

    if IS_CLOUD:
        st.success('🕘 数据每晚自动更新')
        st.caption('由 GitHub Actions 定时推送')
    else:
        if st.button('🔄 拉取最新数据', use_container_width=True, type='primary'):
            fetch_data()
            st.rerun()

    st.divider()

    df, csv_path = load_data()

    if df is not None:
        st.markdown('**数据文件**')
        st.code(os.path.basename(csv_path), language=None)

        st.markdown('**数据概览**')
        st.markdown(f'- 共 **{len(df)}** 期')
        st.markdown(f'- 起始：`{df["openTime"].min().strftime("%Y-%m-%d")}`')
        st.markdown(f'- 截止：`{df["openTime"].max().strftime("%Y-%m-%d")}`')

        latest = df.iloc[-1]
        st.markdown('**最新一期**')
        st.markdown(f'- 期号：`{latest["expect"]}`')
        st.markdown(f'- 时间：`{latest["openTime"].strftime("%Y-%m-%d")}`')
        st.markdown(f'- 特码：**:red[{int(latest["special"])}]**')
        if 'zodiac' in df.columns:
            st.markdown(f'- 生肖：**{latest["zodiac"]}**')
    else:
        st.warning('未找到数据，请点击上方按钮拉取。')

    # ── 时间筛选 ──────────────────────────────────────────────────────────
    if df is not None:
        st.divider()
        st.markdown('**时间筛选**')
        years = sorted(df['openTime'].dt.year.unique(), reverse=True)
        year_opts = ['全部'] + [str(y) for y in years]
        sel_year = st.selectbox('选择年份', year_opts, index=0)

    # ── 报告下载 ──────────────────────────────────────────────────────────
    report_path = Path(__file__).parent / 'output' / 'reports' / 'analysis_report.html'
    if report_path.exists():
        st.divider()
        st.markdown('**报告下载**')
        with open(report_path, 'rb') as f:
            st.download_button(
                label='📥 下载分析报告',
                data=f,
                file_name='analysis_report.html',
                mime='text/html',
                use_container_width=True,
            )

    st.divider()
    st.caption('⚠️ 本系统仅用于历史数据统计分析，不具备预测能力，严禁用于赌博或选号。')


# ══════════════════════════════════════════════════════════════════════════
# 主区域
# ══════════════════════════════════════════════════════════════════════════

if df is None:
    if IS_CLOUD:
        st.warning('⏳ 数据尚未就绪，请稍后刷新页面。')
    else:
        st.info('👈 点击左侧「拉取最新数据」开始使用')
    st.stop()

# ── 应用时间筛选 ──────────────────────────────────────────────────────────
df_full = df.copy()
if sel_year != '全部':
    df = df[df['openTime'].dt.year == int(sel_year)].reset_index(drop=True)
    st.info(f'📅 当前显示 {sel_year} 年数据（共 {len(df)} 期）')

# ── 顶部 Header（在 df_full 定义之后）────────────────────────────────────
latest_period = df_full.iloc[-1]
st.markdown(f"""
<div class="dashboard-header">
  <h1>📊 特码统计分析系统</h1>
  <p>最新期号 {latest_period['expect']} &nbsp;·&nbsp;
     特码 <strong style="color:#00C9FF">{int(latest_period['special'])}</strong> &nbsp;·&nbsp;
     截至 {latest_period['openTime'].strftime('%Y-%m-%d')} &nbsp;·&nbsp;
     共 {len(df_full)} 期历史数据</p>
</div>
<div class="warn-box">⚠️ 本系统仅用于历史数据统计分析与学习研究，结果不具备任何预测能力，严禁用于赌博或选号。</div>
""", unsafe_allow_html=True)

# ── 统计卡片 ─────────────────────────────────────────────────────────────
specials  = df['special'].values.astype(int)
n         = len(specials)
freq_arr  = np.array([Counter(specials).get(i, 0) for i in range(1, 50)])
chi2_stat, chi2_p = chisquare(freq_arr, np.full(49, n / 49))
chi2_pass = chi2_p > 0.05

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="stat-card"><div class="stat-num">{n}</div>'
                f'<div class="stat-lbl">总期数</div></div>', unsafe_allow_html=True)
with c2:
    yr_range = f'{df["openTime"].min().year}–{df["openTime"].max().year}'
    st.markdown(f'<div class="stat-card"><div class="stat-num">{yr_range}</div>'
                f'<div class="stat-lbl">数据年份</div></div>', unsafe_allow_html=True)
with c3:
    label = '✅ 通过' if chi2_pass else '❌ 未通过'
    color = '#27AE60' if chi2_pass else '#E74C3C'
    st.markdown(f'<div class="stat-card">'
                f'<div class="stat-num" style="color:{color};font-size:22px">{label}</div>'
                f'<div class="stat-lbl">均匀性卡方检验</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="stat-card"><div class="stat-num">2.04%</div>'
                f'<div class="stat-lbl">庄家抽水率（1/49）</div></div>', unsafe_allow_html=True)

st.markdown('<br>', unsafe_allow_html=True)

# ── 频率分布图 ────────────────────────────────────────────────────────────
st.subheader('号码频率分布')

col_chart, col_info = st.columns([3, 1])

with col_chart:
    fig, ax = plt.subplots(figsize=(14, 4))
    bar_colors = ['#FF6B6B' if c == freq_arr.max() else
                  '#4A5568' if c == freq_arr.min() else '#00C9FF'
                  for c in freq_arr]
    ax.bar(range(1, 50), freq_arr, color=bar_colors, width=0.7)
    ax.axhline(n / 49, color='black', lw=1.5, linestyle='--',
               label=f'理论均值 {n/49:.1f}')
    ax.set_xlabel('号码'); ax.set_ylabel('出现次数')
    ax.set_xticks(range(1, 50)); ax.tick_params(axis='x', labelsize=7)
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    fig.patch.set_facecolor('none')
    st.pyplot(fig, width='stretch')
    plt.close(fig)

with col_info:
    sorted_f = sorted(enumerate(freq_arr, 1), key=lambda x: x[1])
    hot = sorted_f[-config.TOP_N:][::-1]
    cold = sorted_f[:config.TOP_N]

    st.markdown('**🔥 出现最多**')
    for num, cnt in hot:
        st.markdown(f'`{num:2d}` — {cnt} 次')

    st.markdown('**🧊 出现最少**')
    for num, cnt in cold:
        st.markdown(f'`{num:2d}` — {cnt} 次')

    st.markdown('---')
    st.metric('最大值', f'{freq_arr.max()} 次', f'+{freq_arr.max() - n/49:.1f}')
    st.metric('最小值', f'{freq_arr.min()} 次', f'{freq_arr.min() - n/49:.1f}')

st.markdown('<br>', unsafe_allow_html=True)

# ── 遗漏分析 ──────────────────────────────────────────────────────────────
st.subheader('当前遗漏期数（距上次出现）')

# 计算每个号码的当前遗漏
current_gap = {}
for num in range(1, 50):
    idx = np.where(specials == num)[0]
    current_gap[num] = n - 1 - int(idx[-1]) if len(idx) > 0 else n

gap_arr  = np.array([current_gap[i] for i in range(1, 50)])
avg_gap  = n / 49
warn_thr = avg_gap * 2.0
crit_thr = avg_gap * 3.5

def gap_color(g):
    if g >= crit_thr: return '#E74C3C'
    if g >= warn_thr: return '#F39C12'
    return '#5DADE2'

gap_colors = [gap_color(g) for g in gap_arr]

col_gap, col_anom = st.columns([3, 1])

with col_gap:
    fig2, ax2 = plt.subplots(figsize=(14, 4))
    ax2.bar(range(1, 50), gap_arr, color=gap_colors, width=0.7)
    ax2.axhline(avg_gap,  color='black',  lw=1.5, linestyle='--', label=f'理论均值 {avg_gap:.1f}')
    ax2.axhline(warn_thr, color='#F39C12', lw=1,   linestyle=':',  label=f'2× 警告 {warn_thr:.0f}')
    ax2.axhline(crit_thr, color='#E74C3C', lw=1,   linestyle=':',  label=f'3.5× 严重 {crit_thr:.0f}')
    ax2.set_xlabel('号码'); ax2.set_ylabel('遗漏期数')
    ax2.set_xticks(range(1, 50)); ax2.tick_params(axis='x', labelsize=7)
    ax2.legend(fontsize=8); ax2.grid(axis='y', alpha=0.3)
    fig2.patch.set_facecolor('none')
    st.pyplot(fig2, width='stretch')
    plt.close(fig2)

with col_anom:
    # 异常号码表（≥ 2× 均值）
    anom = [(i+1, int(gap_arr[i])) for i in range(49) if gap_arr[i] >= warn_thr]
    anom.sort(key=lambda x: x[1], reverse=True)

    if anom:
        st.markdown('**⚠️ 遗漏异常号码**')
        for num, gap in anom:
            ratio = gap / avg_gap
            if gap >= crit_thr:
                tag = '🔴 严重'
            else:
                tag = '🟡 警告'
            st.markdown(f'`{num:2d}` 遗漏 **{gap}** 期 ×{ratio:.1f} {tag}')
    else:
        st.markdown('✅ 暂无遗漏异常号码')

    st.markdown('---')
    st.caption(f'均值 {avg_gap:.1f} / 警告 ≥{warn_thr:.0f} / 严重 ≥{crit_thr:.0f}')
    most_gap_num = int(gap_arr.argmax()) + 1
    st.metric('最长遗漏', f'{int(gap_arr.max())} 期', f'号码 {most_gap_num}')

st.markdown('<br>', unsafe_allow_html=True)

# ── 奇偶 / 大小 分布 ──────────────────────────────────────────────────────
st.subheader('结构分布')

col_oe, col_bs, col_tail = st.columns(3)

odd_cnt  = int(np.sum(specials % 2 == 1))
even_cnt = n - odd_cnt
big_cnt  = int(np.sum(specials > config.BIG_NUMBER_THRESHOLD))
small_cnt = n - big_cnt

with col_oe:
    fig3, ax3 = plt.subplots(figsize=(4, 4))
    ax3.pie([odd_cnt, even_cnt], labels=['奇数', '偶数'],
            colors=['#5DADE2', '#F1948A'], autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 11})
    ax3.set_title('奇偶分布', fontsize=12)
    fig3.patch.set_facecolor('none')
    st.pyplot(fig3)
    plt.close(fig3)

with col_bs:
    fig4, ax4 = plt.subplots(figsize=(4, 4))
    ax4.pie([big_cnt, small_cnt], labels=[f'大(>{config.BIG_NUMBER_THRESHOLD})', f'小(≤{config.BIG_NUMBER_THRESHOLD})'],
            colors=['#E59866', '#82E0AA'], autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 11})
    ax4.set_title('大小分布', fontsize=12)
    fig4.patch.set_facecolor('none')
    st.pyplot(fig4)
    plt.close(fig4)

with col_tail:
    # 尾数（个位数）分布
    tail_cnt = Counter(specials % 10)
    tail_labels = [str(i) for i in range(10)]
    tail_vals   = [tail_cnt.get(i, 0) for i in range(10)]
    fig5, ax5 = plt.subplots(figsize=(4, 4))
    ax5.bar(tail_labels, tail_vals, color='#A9CCE3', width=0.6)
    ax5.axhline(n / 10, color='black', lw=1.2, linestyle='--',
                label=f'均值 {n/10:.1f}')
    ax5.set_xlabel('尾数（个位）'); ax5.set_ylabel('次数')
    ax5.set_title('尾数分布', fontsize=12)
    ax5.legend(fontsize=8); ax5.grid(axis='y', alpha=0.3)
    fig5.patch.set_facecolor('none')
    st.pyplot(fig5)
    plt.close(fig5)

st.markdown('<br>', unsafe_allow_html=True)

# ── 近 N 期走势 ───────────────────────────────────────────────────────────
st.subheader('近期走势')

recent_n = st.slider('显示最近期数', min_value=30, max_value=min(300, n),
                     value=min(100, n), step=10)

recent_df = df.tail(recent_n).reset_index(drop=True)

fig6, ax6 = plt.subplots(figsize=(14, 3.5))
ax6.scatter(range(recent_n), recent_df['special'].values,
            c=recent_df['special'].values, cmap='RdYlGn_r',
            s=40, zorder=3, edgecolors='none')
ax6.plot(range(recent_n), recent_df['special'].values,
         color='#AEB6BF', lw=0.8, zorder=2)
ax6.set_xlim(-1, recent_n)
ax6.set_ylim(0, 50)
ax6.set_xlabel('期序（最新在右）'); ax6.set_ylabel('特码')
ax6.set_yticks(range(1, 50, 7))
ax6.grid(alpha=0.25)
fig6.patch.set_facecolor('none')
st.pyplot(fig6, width='stretch')
plt.close(fig6)

st.markdown('<br>', unsafe_allow_html=True)

# ── 生肖分析 ──────────────────────────────────────────────────────────────
if 'zodiac' in df.columns:
    st.subheader('生肖分布')

    zodiac_cnt = Counter(df['zodiac'].dropna())
    zodiacs    = list(zodiac_cnt.keys())
    zod_vals   = list(zodiac_cnt.values())
    zod_mean   = n / 12

    col_zp, col_zt = st.columns([2, 1])

    with col_zp:
        fig7, ax7 = plt.subplots(figsize=(10, 4))
        zod_colors = ['#E74C3C' if v == max(zod_vals) else
                      '#95A5A6' if v == min(zod_vals) else '#5DADE2'
                      for v in zod_vals]
        ax7.bar(zodiacs, zod_vals, color=zod_colors, width=0.6)
        ax7.axhline(zod_mean, color='black', lw=1.5, linestyle='--',
                    label=f'理论均值 {zod_mean:.1f}')
        ax7.set_xlabel('生肖'); ax7.set_ylabel('出现次数')
        ax7.legend(); ax7.grid(axis='y', alpha=0.3)
        fig7.patch.set_facecolor('none')
        st.pyplot(fig7, width='stretch')
        plt.close(fig7)

    with col_zt:
        st.markdown('**生肖频次排名**')
        for z, v in sorted(zodiac_cnt.items(), key=lambda x: x[1], reverse=True):
            delta = v - zod_mean
            sign = '+' if delta >= 0 else ''
            st.markdown(f'**{z}** — {v} 次 ({sign}{delta:.1f})')

st.markdown('<br>', unsafe_allow_html=True)

# ── 号码查询 ──────────────────────────────────────────────────────────────
st.subheader('🔍 号码查询')

query_num = st.number_input('输入号码（1–49）', min_value=1, max_value=49,
                            value=1, step=1)

# 使用全量数据做查询（不受年份筛选影响）
q_df    = df_full.copy()
q_specs = q_df['special'].values.astype(int)
q_n     = len(q_specs)

# 基础统计
q_count   = int(np.sum(q_specs == query_num))
q_freq    = q_count / q_n * 100
q_theory  = q_n / 49
q_idx     = np.where(q_specs == query_num)[0]
q_gap_now = q_n - 1 - int(q_idx[-1]) if len(q_idx) > 0 else q_n
q_last    = q_df.iloc[q_idx[-1]]['openTime'].strftime('%Y-%m-%d') if len(q_idx) > 0 else '从未出现'

# 历史间隔
gaps = np.diff(q_idx).tolist() if len(q_idx) > 1 else []
avg_gap_q = np.mean(gaps) if gaps else 0

col_qi, col_qc, col_qt = st.columns(3)
with col_qi:
    st.metric('出现次数',   f'{q_count} 次',  f'{q_freq:.1f}%')
    st.metric('理论期望',   f'{q_theory:.1f} 次')
    st.metric('最近一次',   q_last)
    st.metric('当前遗漏',   f'{q_gap_now} 期')
    if gaps:
        st.metric('平均间隔', f'{avg_gap_q:.1f} 期')

with col_qc:
    # 近 200 期出现位置散点图
    recent_mask = q_idx[q_idx >= max(0, q_n - 200)]
    fig_q1, ax_q1 = plt.subplots(figsize=(7, 3))
    ax_q1.scatter(recent_mask - max(0, q_n - 200),
                  [query_num] * len(recent_mask),
                  color='#E74C3C', s=60, zorder=3)
    ax_q1.set_xlim(0, 200)
    ax_q1.set_ylim(0, 50)
    ax_q1.set_xlabel('近 200 期（最新在右）')
    ax_q1.set_title(f'号码 {query_num} 出现位置', fontsize=11)
    ax_q1.grid(alpha=0.3)
    fig_q1.patch.set_facecolor('none')
    st.pyplot(fig_q1)
    plt.close(fig_q1)

with col_qt:
    # 间隔分布直方图
    if len(gaps) >= 3:
        fig_q2, ax_q2 = plt.subplots(figsize=(7, 3))
        ax_q2.hist(gaps, bins=20, color='#5DADE2', edgecolor='white')
        ax_q2.axvline(avg_gap_q, color='#E74C3C', lw=1.5,
                      linestyle='--', label=f'均值 {avg_gap_q:.1f}')
        ax_q2.axvline(49, color='black', lw=1, linestyle=':',
                      label='理论 49')
        ax_q2.set_xlabel('出现间隔（期）')
        ax_q2.set_title('历史间隔分布', fontsize=11)
        ax_q2.legend(fontsize=8)
        ax_q2.grid(alpha=0.3)
        fig_q2.patch.set_facecolor('none')
        st.pyplot(fig_q2)
        plt.close(fig_q2)
    else:
        st.info('数据不足，无法绘制间隔分布')

st.divider()
st.caption('📊 特码统计分析系统 · 数据仅供学习研究 · 严禁用于赌博或选号')
