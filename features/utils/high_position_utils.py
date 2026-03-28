"""
高位股统一计算口径（无 IO，纯 DataFrame 操作）
===============================================
统一高位股参数常量，供特征层和 agent 层共同参考。

高位股定义（与 agent_stats/agents/_position_stock_helpers.py 对应）：
  1. 全市场按 20 日涨幅降序排列（特征层用不复权，agent 层用前复权）
  2. 取前 HP_BASE_PCT（1%）为基础池（约50只）
  3. 基础池中取前 HP_HIGH_PCT（10%），最少 HP_MIN_HIGH 只，为高位股

过滤规则：
  - 剔除 ST / *ST 股票
  - 剔除北交所（.BJ）股票
  - 剔除近 HP_IPO_LOOKBACK 个交易日内上市的新股
"""
from typing import Set, Dict, Tuple
import pandas as pd

# ── 高位股规范参数（特征层与 agent 层的共同参照）────────────────────────────
HP_BASE_PCT      = 0.01   # 全市场前 1% 为基础池
HP_HIGH_PCT      = 0.10   # 基础池中前 10% 为高位股
HP_MIN_HIGH      = 3      # 高位股最少只数
HP_IPO_LOOKBACK  = 21     # 新股过滤：近 21 个交易日内上市的剔除

# 内部列名常量
_EMPTY_DF = pd.DataFrame(columns=["ts_code", "close", "gain_20d", "amount"])


def _build_filtered_ranking(
    d0_df: pd.DataFrame,
    d21_df: pd.DataFrame,
    st_set: Set[str],
    list_date_map: Dict[str, str],
    d21_date_str: str,
) -> pd.DataFrame:
    """
    内部函数：过滤 + 计算 20 日涨幅 + 降序排列。

    :param d0_df:        D 日全市场日线，需含 ts_code / close / amount 列
    :param d21_df:       D-21 日全市场日线，需含 ts_code / close 列
    :param st_set:       当日 ST/ST* 股票代码集合
    :param list_date_map:{ts_code: list_date} 上市日期映射（YYYYMMDD 格式）
    :param d21_date_str: D-21 日期（YYYYMMDD），用于新股过滤（上市日期早于此日期才保留）
    :return: 过滤后按 gain_20d 降序的 DataFrame，含 ts_code / close / gain_20d / amount
    """
    if d0_df.empty or d21_df.empty:
        return _EMPTY_DF.copy()

    # D0 过滤：ST / BJ / close <= 0
    d0 = d0_df[["ts_code", "close", "amount"]].copy()
    d0["ts_code"] = d0["ts_code"].astype(str)
    d0["close"]  = pd.to_numeric(d0["close"],  errors="coerce").fillna(0.0)
    d0["amount"] = pd.to_numeric(d0["amount"], errors="coerce").fillna(0.0)
    mask = (
        ~d0["ts_code"].isin(st_set)
        & ~d0["ts_code"].str.endswith(".BJ")
        & (d0["close"] > 0)
    )
    d0 = d0[mask].copy()

    # 新股过滤：list_date >= d21_date_str 的剔除
    if list_date_map:
        ld_series = d0["ts_code"].map(list_date_map).fillna("")
        d0 = d0[~((ld_series != "") & (ld_series >= d21_date_str))].copy()

    if d0.empty:
        return _EMPTY_DF.copy()

    # 合并 D-21 收盘价
    d21 = d21_df[["ts_code", "close"]].copy()
    d21["ts_code"] = d21["ts_code"].astype(str)
    d21["close"]   = pd.to_numeric(d21["close"], errors="coerce").fillna(0.0)
    d21 = d21[d21["close"] > 0].rename(columns={"close": "close_21"})

    merged = d0.merge(d21, on="ts_code", how="inner")
    if merged.empty:
        return _EMPTY_DF.copy()

    merged["gain_20d"] = merged["close"] / merged["close_21"] - 1
    result = (
        merged[["ts_code", "close", "gain_20d", "amount"]]
        .sort_values("gain_20d", ascending=False)
        .reset_index(drop=True)
    )
    return result


def compute_high_pos_selection(
    d0_df: pd.DataFrame,
    d21_df: pd.DataFrame,
    st_set: Set[str],
    list_date_map: Dict[str, str],
    d21_date_str: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    计算高位股基础池和高位股（两段选取）。

    :return: (base_pool_df, high_pos_df)
        base_pool_df: 全市场前 HP_BASE_PCT（~1%），约50只，按 gain_20d 降序
        high_pos_df:  基础池前 HP_HIGH_PCT（~10%），即全市场前 0.1%，最少 HP_MIN_HIGH 只
    """
    all_df = _build_filtered_ranking(d0_df, d21_df, st_set, list_date_map, d21_date_str)

    if all_df.empty:
        return _EMPTY_DF.copy(), _EMPTY_DF.copy()

    # 基础池：前 1%
    n_base = max(HP_MIN_HIGH, int(len(all_df) * HP_BASE_PCT))
    base_df = all_df.head(n_base).copy()

    # 高位股：基础池前 10%
    n_high = max(HP_MIN_HIGH, int(len(base_df) * HP_HIGH_PCT))
    high_df = base_df.head(n_high).copy()

    return base_df, high_df
