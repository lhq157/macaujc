# ========== 数据获取 ==========
API_BASE_URL = "https://history.macaumarksix.com/history/macaujc2/y"
START_YEAR   = 2020
API_TIMEOUT  = 15          # 请求超时秒数

# ========== 分析参数 ==========
SLIDING_WINDOW       = 100  # 滑动窗口期数
TOP_N                = 5    # 热号/冷号显示数量
BIG_NUMBER_THRESHOLD = 24   # 大小号分界（>24为大号）
AUTOCORR_LAGS        = 20   # 自相关检验最大滞后期数

# ========== 输出 ==========
OUTPUT_DIR = "output"
# CSV 文件名格式：YYYYMMDD.csv（每次运行以当天日期命名，增量合并保留全量历史）
