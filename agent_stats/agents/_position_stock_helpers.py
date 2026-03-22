"""
高位股 / 中位股策略共享逻辑
============================
供 high_position_stock / mid_position_stock 复用。
文件名以 _ 开头，引擎自动发现时会跳过。
"""
from typing import Dict, List, Tuple

import pandas as pd

from utils.common_tools import calc_limit_up_price, get_kline_day_range, get_qfq_kline_range
from utils.db_utils import db
from utils.log_utils import logger

# ── 共享策略参数 ─────────────────────────────────────────────────────────────
BASE_PCT      = 0.01    # 全市场前 1% 为基础池
HIGH_PCT      = 0.10    # 基础池中前 10% 为高位股
MIN_HIGH      = 3       # 高位股最少 3 只
LOOKBACK_DAYS = 21      # 回看交易日数（计算 20 日涨幅）
LIMIT_UP_TOL  = 0.01    # 涨停判断容差


def get_list_date_map() -> Dict[str, str]:
    """从 stock_basic 获取全市场 {ts_code: list_date}（YYYYMMDD 格式）"""
    sql = "SELECT ts_code, list_date FROM stock_basic WHERE list_date IS NOT NULL"
    try:
        df = db.query(sql, return_df=True)
        if df is not None and not df.empty:
            return dict(zip(df["ts_code"], df["list_date"].astype(str)))
    except Exception as e:
        logger.warning(f"[position_stock_helpers] stock_basic list_date 查询失败：{e}")
    return {}


def is_yizi_limit_up(row, limit_up_price: float) -> bool:
    """
    判断是否一字涨停无开板（买不进去）：
    open ~ close ~ high，且收盘价贴近涨停价。
    """
    o, c, h = float(row["open"]), float(row["close"]), float(row["high"])
    if limit_up_price <= 0:
        return False
    return (
        c >= limit_up_price - LIMIT_UP_TOL
        and abs(o - c) < LIMIT_UP_TOL
        and abs(h - c) < LIMIT_UP_TOL
    )


def calc_buy_price(row, pre_close: float) -> float:
    """
    计算买入价：
    - 一字板开过板（收盘涨停但盘中曾开板）：涨停价排队买入
    - 其他：开盘价买入
    返回 0 表示无法确定，调用方应跳过该股。
    """
    limit_up = calc_limit_up_price(row["ts_code"], pre_close) if pre_close > 0 else 0.0
    open_p  = float(row["open"])
    close_p = float(row["close"])
    high_p  = float(row["high"])

    if limit_up > 0 and close_p >= limit_up - LIMIT_UP_TOL and high_p > open_p + LIMIT_UP_TOL:
        return limit_up   # 一字板开过板 -> 涨停价
    return open_p         # 普通情况 -> 开盘价


def build_gain_list(
    daily_data: pd.DataFrame,
    st_set: set,
    trade_dates: List[str],
    trade_date: str,
) -> Tuple[List[tuple], Dict[str, float]]:
    """
    构建全市场 20 日涨幅排序列表和前收价映射。

    涨幅计算使用前复权（QFQ）数据，避免除权除息导致的价格跳空失真。
    若 QFQ 数据不可用（表未建立等），自动降级为不复权数据。

    :return: (gain_list, pre_close_map)
        gain_list:     [(ts_code, name, gain_20d, row), ...] 降序排列
        pre_close_map: {ts_code: pre_close}
    """
    idx = trade_dates.index(trade_date)
    t21_date   = trade_dates[idx - LOOKBACK_DAYS]
    ipo_cutoff = t21_date.replace("-", "")

    list_date_map = get_list_date_map()

    all_codes = daily_data["ts_code"].tolist()

    # 优先用 QFQ（前复权）计算涨幅：消除除权除息导致的价格跳空
    # T-21 和 T 两天的 QFQ 收盘价同口径比较，得到真实连续收益率
    t21_qfq = get_qfq_kline_range(all_codes, t21_date, t21_date)
    today_qfq = get_qfq_kline_range(all_codes, trade_date, trade_date)

    use_qfq = not t21_qfq.empty and not today_qfq.empty

    if use_qfq:
        t21_close_map = dict(zip(t21_qfq["ts_code"], t21_qfq["close"].astype(float)))
        today_close_map = dict(zip(today_qfq["ts_code"], today_qfq["close"].astype(float)))
    else:
        # 降级：QFQ 不可用时使用不复权数据（与原逻辑一致）
        logger.warning(
            f"[build_gain_list][{trade_date}] QFQ 数据不可用，降级使用不复权价格计算涨幅"
        )
        t21_df = get_kline_day_range(all_codes, t21_date, t21_date)
        t21_close_map = dict(zip(t21_df["ts_code"], t21_df["close"].astype(float))) if not t21_df.empty else {}
        today_close_map = {}  # 使用 daily_data 中的 close

    gain_list = []
    for _, row in daily_data.iterrows():
        ts_code = row["ts_code"]
        if ts_code in st_set:
            continue
        if ts_code.endswith(".BJ") or ts_code.split(".")[0].startswith(("83", "87", "88")):
            continue
        ld = list_date_map.get(ts_code, "")
        if ld and ld >= ipo_cutoff:
            continue
        close_t21 = t21_close_map.get(ts_code, 0.0)
        if close_t21 <= 0:
            continue

        if use_qfq:
            close_today = today_close_map.get(ts_code, 0.0)
            if close_today <= 0:
                continue
        else:
            close_today = float(row["close"])

        gain_20d = close_today / close_t21 - 1
        name = str(row.get("name", "")) if "name" in row.index else ""
        gain_list.append((ts_code, name, gain_20d, row))

    gain_list.sort(key=lambda x: x[2], reverse=True)
    return gain_list, t21_close_map


def build_pre_close_map(daily_data: pd.DataFrame, context: Dict) -> Dict[str, float]:
    """构建前收价映射（从 context.pre_close_data + daily_data.pre_close 合并）"""
    pre_close_map: Dict[str, float] = {}
    pre_data = context.get("pre_close_data", pd.DataFrame())
    if not pre_data.empty and "ts_code" in pre_data.columns and "close" in pre_data.columns:
        pre_close_map = dict(zip(pre_data["ts_code"], pre_data["close"]))
    if "pre_close" in daily_data.columns:
        for _, row in daily_data.iterrows():
            if row["ts_code"] not in pre_close_map:
                pre_close_map[row["ts_code"]] = float(row["pre_close"])
    return pre_close_map
