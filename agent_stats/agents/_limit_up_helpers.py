"""
涨停板策略共享逻辑
==================
供 _morning_limit_up / _afternoon_limit_up 复用。
文件名以 _ 开头，引擎自动发现时会跳过（pkgutil 不扫描私有模块）。
"""
from typing import Optional

import pandas as pd

from data.data_cleaner import data_cleaner
from utils.log_utils import logger

TOL = 0.999


def get_min_df(ts_code: str, trade_date: str) -> pd.DataFrame:
    """获取分钟线（自动入库），返回空 DF 表示无数据"""
    result = data_cleaner.get_kline_min_by_stock_date(ts_code, trade_date)
    if result is None or result.empty:
        return pd.DataFrame()
    return result


def get_first_limit_time(min_df: pd.DataFrame, limit_price: float) -> Optional[pd.Timestamp]:
    """返回分钟线中首次 high >= limit_price * TOL 的时间，无则 None"""
    if not pd.api.types.is_datetime64_any_dtype(min_df["trade_time"]):
        min_df = min_df.copy()
        min_df["trade_time"] = pd.to_datetime(min_df["trade_time"])
    hit = min_df[min_df["high"] >= limit_price * TOL]
    if hit.empty:
        return None
    return hit.sort_values("trade_time")["trade_time"].iloc[0]


def check_reopen(min_df: pd.DataFrame, limit_price: float) -> Optional[pd.Timestamp]:
    """
    一字板开板+回封校验：
      - 若全天 min_low >= threshold -> 未曾开板 -> None
      - 找到第一个 low < threshold 的分钟，检查本分钟或下一分钟 close >= threshold -> 回封时间
      - 未回封 -> None
    """
    threshold = limit_price * TOL
    if not pd.api.types.is_datetime64_any_dtype(min_df["trade_time"]):
        min_df = min_df.copy()
        min_df["trade_time"] = pd.to_datetime(min_df["trade_time"])
    df = min_df.sort_values("trade_time").reset_index(drop=True)
    if df["low"].min() >= threshold:
        return None   # 全天未开板
    for i, row in df.iterrows():
        if row["low"] < threshold:
            if row["close"] >= threshold:
                return row["trade_time"]
            if i + 1 < len(df):
                nxt = df.iloc[i + 1]
                if nxt["close"] >= threshold:
                    return nxt["trade_time"]
    return None   # 开板后未回封
