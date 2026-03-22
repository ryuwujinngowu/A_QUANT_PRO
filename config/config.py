# ===================== a_quant v1.3 全局配置 =====================
# 版本号
VERSION = "v1.3"

# ===================== 回测基础配置 =====================
DEFAULT_INIT_CAPITAL = 1000000  # 默认初始本金
COMMISSION_RATE = 0.0001       # 交易手续费（万1，双向）
STAMP_DUTY_RATE = 0.001        # 印花税（千1，仅卖出）
SLIPPAGE_RATE = 0.001          # 滑点比例
T_PLUS_1 = True                 # T+1交易规则
MIN_TRADE_VOLUME = 100          # 最小交易单位（1手）
ANNUAL_TRADE_DAYS = 252         # 年化交易日数
RISK_FREE_RATE = 0.03           # 无风险年化收益率

# ===================== 分仓策略配置 =====================
MAX_POSITION_COUNT = 5  # 最大持仓数，总仓位均分5份
# 股票类型过滤配置
FILTER_BSE_STOCK = True    # 过滤北交所股票
FILTER_STAR_BOARD = False
FILTER_MAIN_BOARD = False
FILTER_688_BOARD = False
# ===================== A股板块涨停阈值配置（带0.2%容错） =====================
# 主板（60/00开头）
MAIN_BOARD_LIMIT_UP_RATE = 0.098
# 创业板/科创板（300/688开头）
STAR_BOARD_LIMIT_UP_RATE = 0.198
# ST股（备用）
# ST_BOARD_LIMIT_UP_RATE = 0.048
#BJ所涨停幅度
BJ_BOARD_LIMIT_UP_RATE = 0.298

# ===================== 北交所过滤配置 =====================
BSE_STOCK_PREFIX = ('83', '87', '88')
BSE_EXCHANGE_SUFFIX = 'BJ'
