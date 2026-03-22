"""
agent_stats 模块独立配置
不和主项目config耦合，可独立调整
"""
from datetime import datetime

# ==================== 核心运行配置 ====================
# 统计起始日期（第一次运行时从该日期开始补全历史数据）
START_DATE = "2024-11-01"

# ==================== 数据库配置 ====================
# 统计表名（和建表SQL完全一致）
STATS_TABLE_NAME = "agent_daily_profit_stats"

# ==================== 运行配置 ====================
# 单只股票/单个智能体异常最大重试次数
MAX_RETRY_TIMES = 3
# 数据缺失重试间隔（秒）
RETRY_INTERVAL = 1800
# 最大运行超时时间（秒）
MAX_RUN_TIMEOUT = 3600

# ==================== 日志配置 ====================
LOGGER_NAME = "agent_stats"