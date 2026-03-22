"""
引擎共享工具
============
短线引擎（AgentStatsEngine）和长线引擎（AgentLongStatsEngine）共用的
辅助方法，避免两个引擎之间的逻辑重复。
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.common_tools import get_daily_kline_data, get_st_stock_codes
from utils.log_utils import logger


def get_next_trade_date(all_trade_dates: List[str], trade_date: str) -> Optional[str]:
    """返回 trade_date 的下一个交易日，不存在时返回 None"""
    try:
        idx = all_trade_dates.index(trade_date)
        if idx + 1 < len(all_trade_dates):
            return all_trade_dates[idx + 1]
    except ValueError:
        pass
    return None


def build_trade_date_context(
    trade_date: str,
    context_trade_dates: List[str],
    extra: Optional[Dict] = None,
) -> Dict:
    """
    构建当日选股上下文（ST 列表 / 历史交易日 / T-1 收盘）。
    extra: 调用方可传入额外上下文字段（如 ex_div_stocks）。
    """
    context = {
        "trade_date":    trade_date,
        "st_stock_list": get_st_stock_codes(trade_date),
        "trade_dates":   context_trade_dates,
    }
    try:
        idx = context_trade_dates.index(trade_date)
        pre_date = context_trade_dates[idx - 1] if idx > 0 else None
        context["pre_close_data"] = get_daily_kline_data(pre_date) if pre_date else pd.DataFrame()
    except Exception as e:
        logger.warning(f"[{trade_date}] pre_close_data 获取失败：{e}")
        context["pre_close_data"] = pd.DataFrame()
    if extra:
        context.update(extra)
    return context


def calc_intraday_stats(
    stock_list: List[Dict],
    trade_date: str,
) -> Tuple[float, List[Dict]]:
    """
    计算 T 日收盘后相对买入价的日内收益。
    由短线引擎和长线引擎共用。
    """
    if not stock_list:
        return 0.0, []

    ts_code_list  = [s["ts_code"]   for s in stock_list]
    buy_price_map = {s["ts_code"]: s["buy_price"]    for s in stock_list}
    name_map      = {s["ts_code"]: s.get("stock_name", "")  for s in stock_list}

    daily_df = get_daily_kline_data(trade_date, ts_code_list=ts_code_list)
    if daily_df.empty:
        logger.warning(f"[{trade_date}] 日线数据空，日内收益无法计算")
        return 0.0, stock_list

    detail, returns = [], []
    for _, row in daily_df.iterrows():
        bp = buy_price_map.get(row["ts_code"], 0)
        if bp <= 0:
            continue
        close_p = float(row["close"])
        ret = (close_p - bp) / bp * 100
        returns.append(ret)
        detail.append({
            "ts_code": row["ts_code"],
            "stock_name": name_map.get(row["ts_code"], ""),
            "buy_price": bp,
            "intraday_close_price": close_p,
            "intraday_return": round(ret, 4),
        })
    return round(float(np.mean(returns)) if returns else 0.0, 4), detail
