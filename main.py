"""
main.py — 特码统计分析系统 · 数据管道
职责：拉取最新数据 → 解析 → 导出 CSV
可视化与统计分析请在 analysis.ipynb 中进行
"""

import glob
import os
import requests
import pandas as pd
from datetime import datetime

import config


# ── 生肖映射 ─────────────────────────────────────────────────────────────

_ZODIAC = ['鼠','牛','虎','兔','龙','蛇','马','羊','猴','鸡','狗','猪']

_CNY_DATES = {
    2017:(1,28), 2018:(2,16), 2019:(2, 5), 2020:(1,25),
    2021:(2,12), 2022:(2, 1), 2023:(1,22), 2024:(2,10),
    2025:(1,29), 2026:(2,17), 2027:(1,26), 2028:(2,12),
    2029:(1,31), 2030:(2,19), 2031:(2, 8), 2032:(1,27),
    2033:(2,15), 2034:(2, 4), 2035:(1,24), 2036:(2,12),
    2037:(1,31), 2038:(2,19), 2039:(2, 8), 2040:(1,27),
    2041:(2,14), 2042:(2, 3), 2043:(1,23), 2044:(2,11),
    2045:(1,31), 2046:(2,18), 2047:(2, 7), 2048:(1,27),
    2049:(2,14), 2050:(2, 3), 2051:(1,24), 2052:(2,11),
    2053:(1,31), 2054:(2,19), 2055:(2, 8), 2056:(1,28),
    2057:(2,15), 2058:(2, 4), 2059:(1,24), 2060:(2,12),
    2061:(2, 1), 2062:(1,21), 2063:(2, 9), 2064:(1,29),
    2065:(2,16), 2066:(2, 5), 2067:(1,25), 2068:(2,13),
    2069:(2, 2), 2070:(1,22), 2071:(2,10), 2072:(1,30),
    2073:(2,17), 2074:(2, 6), 2075:(1,26), 2076:(2,14),
    2077:(2, 2), 2078:(1,22), 2079:(2,10), 2080:(1,30),
    2081:(2,17), 2082:(2, 5), 2083:(1,25), 2084:(2,13),
    2085:(2, 2), 2086:(1,22), 2087:(2, 9), 2088:(1,29),
    2089:(2,16), 2090:(2, 5), 2091:(1,25), 2092:(2,13),
    2093:(2, 1), 2094:(1,21), 2095:(2, 9), 2096:(1,29),
    2097:(2,16), 2098:(2, 5), 2099:(1,25),
}
_BASE_YEAR = 2020  # 鼠年，索引 0


def _zodiac_year(date) -> int:
    """根据开奖日期返回所属农历生肖年"""
    y, m, d = date.year, date.month, date.day
    cny_m, cny_d = _CNY_DATES.get(y, (2, 1))
    return y if (m, d) >= (cny_m, cny_d) else y - 1


def number_to_zodiac(number: int, date) -> str:
    """将特别号（1-49）转换为对应生肖"""
    zy      = _zodiac_year(date)
    yr_idx  = (zy - _BASE_YEAR) % 12
    num_idx = (yr_idx - (number - 1)) % 12
    return _ZODIAC[num_idx]


# ── 1. 拉取数据 ──────────────────────────────────────────────────────────

def fetch_year(year: int) -> list:
    url = f"{config.API_BASE_URL}/{year}"
    try:
        r = requests.get(url, timeout=config.API_TIMEOUT)
        data = r.json()
        if data.get("result"):
            return data["data"]
    except Exception as e:
        print(f"  {year} 年数据获取失败：{e}")
    return []


def fetch_all(start_year: int, end_year: int) -> list:
    records = []
    for year in range(start_year, end_year + 1):
        rows = fetch_year(year)
        records.extend(rows)
        print(f"  {year} 年：{len(rows)} 期")
    print(f"共获取：{len(records)} 条原始记录")
    return records


# ── 2. 解析数据 ──────────────────────────────────────────────────────────

def parse(records: list) -> pd.DataFrame:
    rows = []
    for r in records:
        codes   = r.get("openCode", "")
        numbers = [int(x.strip()) for x in codes.split(",") if x.strip().isdigit()]
        if len(numbers) == 7:
            rows.append({
                "expect":   r["expect"],
                "openTime": r["openTime"],
                "n1": numbers[0], "n2": numbers[1], "n3": numbers[2],
                "n4": numbers[3], "n5": numbers[4], "n6": numbers[5],
                "special": numbers[6],
            })

    df = pd.DataFrame(rows)
    df["openTime"] = pd.to_datetime(df["openTime"])

    before = len(df)
    df = (df.drop_duplicates(subset="expect")
            .sort_values("openTime")
            .reset_index(drop=True))

    if len(df) < before:
        print(f"去重：移除 {before - len(df)} 条重复期数")

    df["zodiac"] = df.apply(
        lambda r: number_to_zodiac(r["special"], r["openTime"]), axis=1
    )

    print(f"有效数据：{len(df)} 期  "
          f"({df['openTime'].min().date()} ~ {df['openTime'].max().date()})")
    return df


# ── 3. 增量合并并导出 CSV ─────────────────────────────────────────────────
# 文件名：YYYYMMDD.csv（每次运行以当天日期命名）
# 策略：与本地最新 CSV 合并，旧数据永久保留，API 删除历史数据也不受影响

def export_csv(df_new: pd.DataFrame, date_str: str) -> str:
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(config.OUTPUT_DIR, f"{date_str}.csv")

    # 找本地最新的 CSV（按文件名排序，YYYYMMDD 格式天然可排序）
    existing = sorted(glob.glob(os.path.join(config.OUTPUT_DIR, "????????.csv")))
    latest   = existing[-1] if existing else None

    if latest:
        df_old             = pd.read_csv(latest, dtype={"expect": str})
        df_old["openTime"] = pd.to_datetime(df_old["openTime"])  # 统一类型，避免 str vs Timestamp 排序报错
        df_merged = pd.concat([df_old, df_new], ignore_index=True)
        df_merged = (df_merged
                     .drop_duplicates(subset="expect")
                     .sort_values("openTime")
                     .reset_index(drop=True))
        added = len(df_merged) - len(df_old)
        print(f"本地已有 {len(df_old)} 期（{os.path.basename(latest)}），"
              f"本次新增 {added} 期，合并后共 {len(df_merged)} 期")
        df_merged.to_csv(csv_path, index=False, encoding="utf-8-sig")
        # 新文件写入成功后删除旧文件（新文件已包含全量历史，旧文件无保留价值）
        if latest != csv_path:
            os.remove(latest)
            print(f"已删除旧文件：{os.path.basename(latest)}")
    else:
        df_new.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"首次写入 {len(df_new)} 期数据")

    print(f"CSV 已保存：{csv_path}")
    return csv_path


# ── 入口 ─────────────────────────────────────────────────────────────────

def main():
    now          = datetime.now()
    current_year = now.year
    date_str     = now.strftime("%Y%m%d")

    print(f"=== 特码统计分析系统 · 数据管道  {now.strftime('%Y-%m-%d %H:%M')} ===\n")
    print(f"拉取范围：{config.START_YEAR} ~ {current_year}")
    records = fetch_all(config.START_YEAR, current_year)
    df      = parse(records)
    path    = export_csv(df, date_str)

    print(f"\n完成。数据路径：{path}")
    print("在 analysis.ipynb 中运行 load_data() 即可加载。")


if __name__ == "__main__":
    main()
