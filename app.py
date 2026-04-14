"""
app.py — 特码统计分析系统 · Streamlit Web 界面
"""

import glob
import os
import sys
import subprocess
from pathlib import Path
from collections import Counter
import datetime

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import chisquare, kstest, expon

import config

# ── 环境检测 ──────────────────────────────────────────────────────────────
IS_CLOUD = not os.access(os.path.join(os.path.dirname(__file__), 'output'), os.W_OK)

# ── 中文字体 & 深色图表 ───────────────────────────────────────────────────
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

# ── 生肖 emoji ────────────────────────────────────────────────────────────
ZODIAC_EMOJI = {
    '鼠': '🐭', '牛': '🐮', '虎': '🐯', '兔': '🐰',
    '龙': '🐲', '蛇': '🐍', '马': '🐴', '羊': '🐑',
    '猴': '🐵', '鸡': '🐔', '狗': '🐶', '猪': '🐷',
}

# ── 页面配置 ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='特码统计分析系统',
    page_icon='📊',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ── 全局样式 ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  html, body, [class*="css"] {
    font-family: -apple-system, 'PingFang SC', 'Helvetica Neue', sans-serif;
  }
  .dashboard-header {
    background: linear-gradient(135deg, #0F2744 0%, #1A3A5C 100%);
    border-radius: 14px;
    padding: 24px 32px 18px;
    margin-bottom: 20px;
    border: 1px solid rgba(0,201,255,0.15);
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
  }
  .dashboard-header h1 { font-size: 24px; font-weight: 700; color: #fff; margin: 0 0 4px; }
  .dashboard-header p  { font-size: 12px; color: rgba(255,255,255,0.5); margin: 0; }
  .stat-card {
    background: linear-gradient(145deg, #1A2E44, #152438);
    border-radius: 12px; padding: 18px 12px; text-align: center;
    border: 1px solid rgba(0,201,255,0.12);
    box-shadow: 0 2px 16px rgba(0,0,0,0.3);
  }
  .stat-num { font-size: 28px; font-weight: 700; color: #00C9FF; }
  .stat-lbl { font-size: 11px; color: rgba(224,230,237,0.5); margin-top: 5px;
               text-transform: uppercase; letter-spacing: .8px; }
  .warn-box {
    background: rgba(243,156,18,0.08); border-left: 3px solid #F39C12;
    padding: 8px 14px; border-radius: 0 6px 6px 0;
    font-size: 12px; color: rgba(224,230,237,0.65); margin-bottom: 12px;
  }
  .concl-item { padding: 6px 0; font-size: 13px; color: #E0E6ED; }
  .concl-item span { color: #00C9FF; font-weight: 600; margin-right: 6px; }
  section[data-testid="stSidebar"] { background: #111D2C !important;
    border-right: 1px solid rgba(0,201,255,0.08); }
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

df_raw, csv_path = load_data()

with st.sidebar:
    st.markdown('### 📊 特码统计分析')
    st.divider()

    # ── 数据文件 ──────────────────────────────────────────────────────────
    st.markdown('**数据筛选**')
    if df_raw is not None:
        csvs = sorted(glob.glob(os.path.join(config.OUTPUT_DIR, '????????.csv')))
        file_names = [os.path.basename(c) for c in csvs]
        sel_file = st.selectbox('选择数据文件', file_names,
                                index=len(file_names)-1, label_visibility='collapsed')
        # 加载选中文件
        sel_path = os.path.join(config.OUTPUT_DIR, sel_file)
        df_sel = pd.read_csv(sel_path, dtype={'expect': str})
        df_sel['openTime'] = pd.to_datetime(df_sel['openTime'])
        df_sel = df_sel.sort_values('openTime').reset_index(drop=True)

        # ── 时间范围 ──────────────────────────────────────────────────────
        min_d = df_sel['openTime'].min().date()
        max_d = df_sel['openTime'].max().date()
        c1, c2 = st.columns(2)
        with c1:
            start_d = st.date_input('开始', min_d, min_value=min_d, max_value=max_d)
        with c2:
            end_d   = st.date_input('结束', max_d, min_value=min_d, max_value=max_d)

        df_full = df_sel[
            (df_sel['openTime'].dt.date >= start_d) &
            (df_sel['openTime'].dt.date <= end_d)
        ].reset_index(drop=True)
    else:
        df_full = None

    st.divider()

    # ── 分析参数 ──────────────────────────────────────────────────────────
    st.markdown('**分析权重**')
    window_n  = st.slider('滚动窗口', 50, 500, 100, step=10, format='%d期')
    sig_level = st.number_input('显著性水平', 0.01, 0.10, 0.05, step=0.01, format='%.2f')

    st.divider()

    # ── 显示模式 ──────────────────────────────────────────────────────────
    st.markdown('**显示模式**')
    display_mode = st.radio('', ['各生肖', 'Top 5', '全部12肖'],
                            index=2, label_visibility='collapsed')

    st.divider()

    # ── 操作按钮 ──────────────────────────────────────────────────────────
    if IS_CLOUD:
        st.button('🚀 开始分析', use_container_width=True, type='primary', disabled=True)
        st.success('🕘 数据每晚自动更新')
    else:
        if st.button('🚀 开始分析', use_container_width=True, type='primary'):
            fetch_data()
            st.rerun()

    # ── 报告下载 ──────────────────────────────────────────────────────────
    report_path = Path(__file__).parent / 'output' / 'reports' / 'analysis_report.html'
    if report_path.exists():
        st.divider()
        with open(report_path, 'rb') as f:
            st.download_button('📥 下载分析报告', f,
                               file_name='analysis_report.html',
                               mime='text/html',
                               use_container_width=True)

    st.divider()
    st.caption('⚠️ 仅供学习研究，不具备预测能力，严禁赌博选号。')


# ══════════════════════════════════════════════════════════════════════════
# 数据检查
# ══════════════════════════════════════════════════════════════════════════

if df_full is None or len(df_full) == 0:
    if IS_CLOUD:
        st.warning('⏳ 数据尚未就绪，请稍后刷新页面。')
    else:
        st.info('👈 点击左侧「开始分析」加载数据')
    st.stop()

# ── 基础计算 ─────────────────────────────────────────────────────────────
specials  = df_full['special'].values.astype(int)
n         = len(specials)
freq_arr  = np.array([Counter(specials).get(i, 0) for i in range(1, 50)])
avg_freq  = n / 49
chi2_stat, chi2_p = chisquare(freq_arr, np.full(49, avg_freq))
chi2_pass = chi2_p > sig_level
avg_gap   = n / 49

latest = df_full.iloc[-1]

# ── Header ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="dashboard-header">
  <h1>📊 特码统计分析系统</h1>
  <p>基于概率统计的随机抽样分析平台 &nbsp;·&nbsp;
     最新期号 <strong style="color:#00C9FF">{latest['expect']}</strong> &nbsp;·&nbsp;
     特码 <strong style="color:#FF6B6B">{int(latest['special'])}</strong> &nbsp;·&nbsp;
     当前筛选 <strong style="color:#00C9FF">{n}</strong> 期数据</p>
</div>
<div class="warn-box">⚠️ 本系统仅用于历史数据统计分析与学习研究，结果不具备任何预测能力，严禁用于赌博或选号。</div>
""", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ['📈 基础统计', '🐲 生肖分析', '⏱ 间隔分析', '🔄 滚动窗口', '🔬 假设检验', '🔍 号码查询']
)


# ══════════════════════════════════════════════════════════════════════════
# Tab 1：基础统计
# ══════════════════════════════════════════════════════════════════════════

with tab1:
    # 统计卡片
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="stat-card"><div class="stat-num">{n}</div>'
                    f'<div class="stat-lbl">总期数</div></div>', unsafe_allow_html=True)
    with c2:
        yr = f'{df_full["openTime"].min().year}–{df_full["openTime"].max().year}'
        st.markdown(f'<div class="stat-card"><div class="stat-num">{yr}</div>'
                    f'<div class="stat-lbl">数据年份</div></div>', unsafe_allow_html=True)
    with c3:
        lbl   = '✅ 通过' if chi2_pass else '❌ 未通过'
        color = '#27AE60' if chi2_pass else '#E74C3C'
        st.markdown(f'<div class="stat-card">'
                    f'<div class="stat-num" style="color:{color};font-size:20px">{lbl}</div>'
                    f'<div class="stat-lbl">均匀性检验 (p={chi2_p:.3f})</div></div>',
                    unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="stat-card"><div class="stat-num">2.04%</div>'
                    '<div class="stat-lbl">庄家抽水率（1/49）</div></div>',
                    unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # 频率分布
    st.subheader('号码频率分布')
    col_ch, col_info = st.columns([3, 1])
    with col_ch:
        fig, ax = plt.subplots(figsize=(14, 4))
        bar_colors = ['#FF6B6B' if c == freq_arr.max() else
                      '#4A5568' if c == freq_arr.min() else '#00C9FF'
                      for c in freq_arr]
        ax.bar(range(1, 50), freq_arr, color=bar_colors, width=0.7)
        ax.axhline(avg_freq, color='#F39C12', lw=1.5, linestyle='--',
                   label=f'理论均值 {avg_freq:.1f}')
        ax.set_xlabel('号码'); ax.set_ylabel('出现次数')
        ax.set_xticks(range(1, 50)); ax.tick_params(axis='x', labelsize=7)
        ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
        fig.patch.set_facecolor('#1A2634')
        st.pyplot(fig, width='stretch'); plt.close(fig)
    with col_info:
        top5  = sorted(enumerate(freq_arr, 1), key=lambda x: x[1], reverse=True)[:5]
        bot5  = sorted(enumerate(freq_arr, 1), key=lambda x: x[1])[:5]
        st.markdown('**🔥 最热号码**')
        for num, cnt in top5:
            st.markdown(f'`{num:2d}` — **{cnt}** 次')
        st.markdown('**🧊 最冷号码**')
        for num, cnt in bot5:
            st.markdown(f'`{num:2d}` — {cnt} 次')

    st.markdown('<br>', unsafe_allow_html=True)

    # 结构分布
    st.subheader('结构分布')
    co1, co2, co3 = st.columns(3)
    odd_cnt   = int(np.sum(specials % 2 == 1))
    big_cnt   = int(np.sum(specials > config.BIG_NUMBER_THRESHOLD))
    tail_cnt  = Counter(specials % 10)
    tail_vals = [tail_cnt.get(i, 0) for i in range(10)]

    with co1:
        fig2, ax2 = plt.subplots(figsize=(4, 4))
        ax2.pie([odd_cnt, n - odd_cnt], labels=['奇', '偶'],
                colors=['#00C9FF', '#FF6B6B'], autopct='%1.1f%%',
                startangle=90, textprops={'fontsize': 12})
        ax2.set_title('奇偶分布'); fig2.patch.set_facecolor('#1A2634')
        st.pyplot(fig2); plt.close(fig2)

    with co2:
        fig3, ax3 = plt.subplots(figsize=(4, 4))
        ax3.pie([big_cnt, n - big_cnt],
                labels=[f'大(>{config.BIG_NUMBER_THRESHOLD})',
                        f'小(≤{config.BIG_NUMBER_THRESHOLD})'],
                colors=['#F39C12', '#27AE60'], autopct='%1.1f%%',
                startangle=90, textprops={'fontsize': 12})
        ax3.set_title('大小分布'); fig3.patch.set_facecolor('#1A2634')
        st.pyplot(fig3); plt.close(fig3)

    with co3:
        fig4, ax4 = plt.subplots(figsize=(4, 4))
        ax4.bar(range(10), tail_vals, color='#5DADE2', width=0.6)
        ax4.axhline(n / 10, color='#F39C12', lw=1.2, linestyle='--')
        ax4.set_xticks(range(10)); ax4.set_title('尾数分布')
        ax4.grid(axis='y', alpha=0.3)
        fig4.patch.set_facecolor('#1A2634')
        st.pyplot(fig4); plt.close(fig4)


# ══════════════════════════════════════════════════════════════════════════
# Tab 2：生肖分析
# ══════════════════════════════════════════════════════════════════════════

with tab2:
    if 'zodiac' not in df_full.columns:
        st.warning('数据中无生肖字段')
    else:
        zodiac_cnt  = Counter(df_full['zodiac'].dropna())
        zodiac_list = [z for z in ZODIAC_EMOJI if z in zodiac_cnt]
        zod_mean    = n / 12

        # 显示模式过滤
        if display_mode == 'Top 5':
            zodiac_list = sorted(zodiac_cnt, key=zodiac_cnt.get, reverse=True)[:5]
        elif display_mode == '各生肖':
            zodiac_list = list(zodiac_cnt.keys())

        zod_vals   = [zodiac_cnt.get(z, 0) for z in zodiac_list]
        zod_labels = [f'{ZODIAC_EMOJI.get(z,"")}{z}' for z in zodiac_list]

        col_zp, col_zt = st.columns([2, 1])
        with col_zp:
            st.subheader('生肖统计')
            fig5, ax5 = plt.subplots(figsize=(10, 4))
            zcolors = ['#FF6B6B' if v == max(zod_vals) else
                       '#4A5568' if v == min(zod_vals) else '#00C9FF'
                       for v in zod_vals]
            ax5.bar(zod_labels, zod_vals, color=zcolors, width=0.6)
            ax5.axhline(zod_mean, color='#F39C12', lw=1.5, linestyle='--',
                        label=f'理论均值 {zod_mean:.1f}')
            ax5.set_ylabel('出现次数'); ax5.legend(fontsize=9)
            ax5.grid(axis='y', alpha=0.3)
            fig5.patch.set_facecolor('#1A2634')
            st.pyplot(fig5, width='stretch'); plt.close(fig5)

        with col_zt:
            st.subheader('频次排名')
            for z, v in sorted(zodiac_cnt.items(), key=lambda x: x[1], reverse=True):
                delta = v - zod_mean
                sign  = '+' if delta >= 0 else ''
                emoji = ZODIAC_EMOJI.get(z, '')
                st.markdown(f'{emoji} **{z}** — {v} 次 `{sign}{delta:.0f}`')

        st.markdown('<br>', unsafe_allow_html=True)

        # 月度走势
        st.subheader('月度走势')
        df_full['ym'] = df_full['openTime'].dt.to_period('M')
        monthly = (df_full.groupby(['ym', 'zodiac'])
                   .size().unstack(fill_value=0))
        top_z = sorted(zodiac_cnt, key=zodiac_cnt.get, reverse=True)[:6]
        monthly_top = monthly[[z for z in top_z if z in monthly.columns]]

        fig6, ax6 = plt.subplots(figsize=(12, 4))
        colors6 = ['#00C9FF','#FF6B6B','#F39C12','#27AE60','#9B59B6','#E67E22']
        for i, z in enumerate(monthly_top.columns):
            ax6.plot(range(len(monthly_top)), monthly_top[z].values,
                     marker='o', markersize=4, lw=1.5,
                     color=colors6[i % len(colors6)],
                     label=f'{ZODIAC_EMOJI.get(z,"")}{z}')
        xticks = monthly_top.index.astype(str).tolist()
        step   = max(1, len(xticks) // 12)
        ax6.set_xticks(range(0, len(xticks), step))
        ax6.set_xticklabels(xticks[::step], rotation=30, fontsize=8)
        ax6.set_ylabel('出现次数'); ax6.legend(fontsize=8, ncol=3)
        ax6.grid(alpha=0.25); fig6.patch.set_facecolor('#1A2634')
        st.pyplot(fig6, width='stretch'); plt.close(fig6)

        st.markdown('<br>', unsafe_allow_html=True)

        # 多窗口对比
        st.subheader('多窗口对比')
        windows = [50, 100, 200]
        table_rows = []
        for z in sorted(zodiac_cnt, key=zodiac_cnt.get, reverse=True)[:8]:
            row = {'生肖': f'{ZODIAC_EMOJI.get(z,"")}{z}'}
            for w in windows:
                df_w = df_full.tail(w)
                wc   = Counter(df_w['zodiac'])
                row[f'近{w}期'] = f"{wc.get(z,0)/w*100:.0f}%"
            row['全部'] = f"{zodiac_cnt.get(z,0)/n*100:.1f}%"
            table_rows.append(row)
        st.dataframe(pd.DataFrame(table_rows).set_index('生肖'),
                     use_container_width=True)

        st.markdown('<br>', unsafe_allow_html=True)

        # 分析结论
        st.subheader('分析结论')
        top1    = max(zodiac_cnt, key=zodiac_cnt.get)
        bot1    = min(zodiac_cnt, key=zodiac_cnt.get)
        top1_50 = Counter(df_full.tail(50)['zodiac']).most_common(1)[0][0]
        conclusions = [
            f"近期（近50期）频率最高生肖为 <span>{ZODIAC_EMOJI.get(top1_50,'')}{top1_50}</span>",
            f"全量数据中 <span>{ZODIAC_EMOJI.get(top1,'')}{top1}</span> 出现最多（{zodiac_cnt[top1]}次），"
            f"<span>{ZODIAC_EMOJI.get(bot1,'')}{bot1}</span> 出现最少（{zodiac_cnt[bot1]}次）",
            f"整体分布{'符合' if chi2_pass else '偏离'}随机均匀分布（卡方检验 p={chi2_p:.3f}）",
        ]
        for c in conclusions:
            st.markdown(f'<div class="concl-item">● {c}</div>',
                        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# Tab 3：间隔分析
# ══════════════════════════════════════════════════════════════════════════

with tab3:
    current_gap = {}
    for num in range(1, 50):
        idx = np.where(specials == num)[0]
        current_gap[num] = n - 1 - int(idx[-1]) if len(idx) > 0 else n

    gap_arr   = np.array([current_gap[i] for i in range(1, 50)])
    warn_thr  = avg_gap * 2.0
    crit_thr  = avg_gap * 3.5

    def gap_color(g):
        if g >= crit_thr: return '#E74C3C'
        if g >= warn_thr: return '#F39C12'
        return '#00C9FF'

    st.subheader('当前遗漏期数')
    col_gp, col_ga = st.columns([3, 1])
    with col_gp:
        fig7, ax7 = plt.subplots(figsize=(14, 4))
        ax7.bar(range(1, 50), gap_arr,
                color=[gap_color(g) for g in gap_arr], width=0.7)
        ax7.axhline(avg_gap,  color='white',  lw=1.5, linestyle='--',
                    label=f'均值 {avg_gap:.1f}')
        ax7.axhline(warn_thr, color='#F39C12', lw=1, linestyle=':',
                    label=f'2× 警告 {warn_thr:.0f}')
        ax7.axhline(crit_thr, color='#E74C3C', lw=1, linestyle=':',
                    label=f'3.5× 严重 {crit_thr:.0f}')
        ax7.set_xlabel('号码'); ax7.set_ylabel('遗漏期数')
        ax7.set_xticks(range(1, 50)); ax7.tick_params(axis='x', labelsize=7)
        ax7.legend(fontsize=8); ax7.grid(axis='y', alpha=0.3)
        fig7.patch.set_facecolor('#1A2634')
        st.pyplot(fig7, width='stretch'); plt.close(fig7)

    with col_ga:
        anom = [(i+1, int(gap_arr[i])) for i in range(49) if gap_arr[i] >= warn_thr]
        anom.sort(key=lambda x: x[1], reverse=True)
        if anom:
            st.markdown('**⚠️ 遗漏异常**')
            for num, gap in anom:
                tag = '🔴' if gap >= crit_thr else '🟡'
                st.markdown(f'{tag} `{num:2d}` — **{gap}** 期 ×{gap/avg_gap:.1f}')
        else:
            st.success('✅ 暂无遗漏异常')
        st.divider()
        st.metric('最长遗漏', f'{int(gap_arr.max())} 期',
                  f'号码 {int(gap_arr.argmax())+1}')
        st.metric('均值', f'{avg_gap:.1f} 期')


# ══════════════════════════════════════════════════════════════════════════
# Tab 4：滚动窗口
# ══════════════════════════════════════════════════════════════════════════

with tab4:
    df_win   = df_full.tail(window_n).reset_index(drop=True)
    sp_win   = df_win['special'].values.astype(int)
    freq_win = np.array([Counter(sp_win).get(i, 0) for i in range(1, 50)])

    st.subheader(f'近 {window_n} 期分析')

    col_w1, col_w2 = st.columns(2)
    with col_w1:
        fig8, ax8 = plt.subplots(figsize=(7, 3.5))
        wcolors = ['#FF6B6B' if c == freq_win.max() else
                   '#4A5568' if c == 0 else '#00C9FF'
                   for c in freq_win]
        ax8.bar(range(1, 50), freq_win, color=wcolors, width=0.7)
        ax8.axhline(window_n/49, color='#F39C12', lw=1.5, linestyle='--')
        ax8.set_title(f'近{window_n}期频率分布')
        ax8.set_xticks(range(1, 50, 3)); ax8.tick_params(axis='x', labelsize=7)
        ax8.grid(axis='y', alpha=0.3)
        fig8.patch.set_facecolor('#1A2634')
        st.pyplot(fig8); plt.close(fig8)

    with col_w2:
        fig9, ax9 = plt.subplots(figsize=(7, 3.5))
        ax9.scatter(range(window_n), sp_win,
                    c=sp_win, cmap='RdYlGn_r', s=30, zorder=3)
        ax9.plot(range(window_n), sp_win, color='#2A3F54', lw=0.6, zorder=2)
        ax9.set_ylim(0, 50); ax9.set_xlabel('期序（最新在右）')
        ax9.set_title(f'近{window_n}期走势')
        ax9.grid(alpha=0.2)
        fig9.patch.set_facecolor('#1A2634')
        st.pyplot(fig9); plt.close(fig9)

    # 冷热榜
    st.markdown('<br>', unsafe_allow_html=True)
    cw1, cw2, cw3 = st.columns(3)
    hot5  = sorted(enumerate(freq_win, 1), key=lambda x: x[1], reverse=True)[:5]
    cold5 = sorted(enumerate(freq_win, 1), key=lambda x: x[1])[:5]
    miss5 = sorted([(i+1, int(gap_arr[i])) for i in range(49)],
                   key=lambda x: x[1], reverse=True)[:5]
    with cw1:
        st.markdown(f'**🔥 近{window_n}期最热**')
        for num, cnt in hot5:
            st.markdown(f'`{num:2d}` — **{cnt}** 次')
    with cw2:
        st.markdown(f'**🧊 近{window_n}期最冷**')
        for num, cnt in cold5:
            st.markdown(f'`{num:2d}` — {cnt} 次')
    with cw3:
        st.markdown('**⏳ 当前遗漏最长**')
        for num, gap in miss5:
            st.markdown(f'`{num:2d}` — 已遗漏 **{gap}** 期')


# ══════════════════════════════════════════════════════════════════════════
# Tab 5：假设检验
# ══════════════════════════════════════════════════════════════════════════

with tab5:
    st.subheader('统计假设检验')

    # 间隔序列（用于 KS 检验）
    all_idx  = [np.where(specials == i)[0] for i in range(1, 50)]
    all_gaps = []
    for idx in all_idx:
        if len(idx) > 1:
            all_gaps.extend(np.diff(idx).tolist())
    all_gaps = np.array(all_gaps)

    ks_stat, ks_p = kstest(all_gaps, 'expon',
                            args=(0, avg_gap)) if len(all_gaps) > 0 else (0, 1)

    c_t1, c_t2 = st.columns(2)
    with c_t1:
        color1 = '#27AE60' if chi2_pass else '#E74C3C'
        st.markdown(f"""
        **① 卡方均匀性检验**
        - H₀：49 个号码出现概率相等（均为 1/49）
        - 统计量：χ² = {chi2_stat:.2f}
        - p 值：{chi2_p:.4f}
        - 显著性水平：{sig_level}
        - 结论：<span style="color:{color1}">{"✅ 接受 H₀，分布均匀" if chi2_pass else "❌ 拒绝 H₀，分布不均匀"}</span>
        """, unsafe_allow_html=True)

    with c_t2:
        ks_pass  = ks_p > sig_level
        color2   = '#27AE60' if ks_pass else '#E74C3C'
        st.markdown(f"""
        **② KS 间隔指数分布检验**
        - H₀：号码出现间隔服从指数分布（独立性检验）
        - 统计量：D = {ks_stat:.4f}
        - p 值：{ks_p:.4f}
        - 显著性水平：{sig_level}
        - 结论：<span style="color:{color2}">{"✅ 接受 H₀，各期独立" if ks_pass else "❌ 拒绝 H₀，存在相关性"}</span>
        """, unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # 间隔分布图
    if len(all_gaps) > 0:
        fig10, ax10 = plt.subplots(figsize=(10, 3.5))
        ax10.hist(all_gaps, bins=50, density=True,
                  color='#00C9FF', alpha=0.7, label='实际间隔分布')
        x = np.linspace(0, all_gaps.max(), 200)
        ax10.plot(x, (1/avg_gap) * np.exp(-x/avg_gap),
                  color='#FF6B6B', lw=2, label=f'理论指数分布（λ=1/{avg_gap:.0f}）')
        ax10.set_xlabel('间隔期数'); ax10.set_ylabel('密度')
        ax10.set_title('号码出现间隔分布 vs 理论指数分布')
        ax10.legend(fontsize=9); ax10.grid(alpha=0.3)
        fig10.patch.set_facecolor('#1A2634')
        st.pyplot(fig10, width='stretch'); plt.close(fig10)

    st.markdown('<br>', unsafe_allow_html=True)
    st.info('💡 两项检验均通过说明：历史数据与纯随机抽样在统计上无显著差异，历史数据对未来无预测能力。')


# ══════════════════════════════════════════════════════════════════════════
# Tab 6：号码查询
# ══════════════════════════════════════════════════════════════════════════

with tab6:
    st.subheader('🔍 单号码深度查询')
    query_num = st.number_input('输入号码（1–49）', min_value=1, max_value=49,
                                value=1, step=1)

    q_idx     = np.where(specials == query_num)[0]
    q_count   = len(q_idx)
    q_freq    = q_count / n * 100
    q_gap_now = n - 1 - int(q_idx[-1]) if q_count > 0 else n
    q_last    = df_full.iloc[q_idx[-1]]['openTime'].strftime('%Y-%m-%d') if q_count > 0 else '从未'
    gaps_q    = np.diff(q_idx).tolist() if q_count > 1 else []
    avg_gap_q = np.mean(gaps_q) if gaps_q else 0

    c_qi, c_qc, c_qt = st.columns(3)
    with c_qi:
        st.metric('出现次数',  f'{q_count} 次',  f'{q_freq:.1f}%')
        st.metric('理论期望',  f'{n/49:.1f} 次')
        st.metric('最近一次',  q_last)
        st.metric('当前遗漏',  f'{q_gap_now} 期')
        if gaps_q:
            st.metric('平均间隔', f'{avg_gap_q:.1f} 期')

    with c_qc:
        recent_idx = q_idx[q_idx >= max(0, n - 200)]
        fig11, ax11 = plt.subplots(figsize=(6, 3))
        ax11.scatter(recent_idx - max(0, n-200), [query_num]*len(recent_idx),
                     color='#FF6B6B', s=60, zorder=3)
        ax11.set_xlim(0, 200); ax11.set_ylim(0, 50)
        ax11.set_xlabel('近 200 期（最新在右）')
        ax11.set_title(f'号码 {query_num} 出现位置')
        ax11.grid(alpha=0.3); fig11.patch.set_facecolor('#1A2634')
        st.pyplot(fig11); plt.close(fig11)

    with c_qt:
        if len(gaps_q) >= 3:
            fig12, ax12 = plt.subplots(figsize=(6, 3))
            ax12.hist(gaps_q, bins=20, color='#00C9FF', edgecolor='#1A2634')
            ax12.axvline(avg_gap_q, color='#FF6B6B', lw=1.5,
                         linestyle='--', label=f'均值 {avg_gap_q:.1f}')
            ax12.axvline(49, color='#F39C12', lw=1, linestyle=':',
                         label='理论 49')
            ax12.set_xlabel('出现间隔（期）'); ax12.set_title('历史间隔分布')
            ax12.legend(fontsize=8); ax12.grid(alpha=0.3)
            fig12.patch.set_facecolor('#1A2634')
            st.pyplot(fig12); plt.close(fig12)
        else:
            st.info('数据不足，无法绘制间隔分布')

st.divider()
st.caption('📊 特码统计分析系统 · 数据仅供学习研究 · 严禁用于赌博或选号')
