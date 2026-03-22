"""
板块高开策略共享逻辑
====================
供 sector_top_high_open / sector_rank2_high_open / sector_rank3_high_open 复用。
文件名以 _ 开头，引擎自动发现时会跳过（pkgutil 不扫描私有模块）。
"""
from typing import Dict, List

import pandas as pd

from utils.common_tools import get_limit_cpt_list, get_stocks_in_sector
from utils.log_utils import logger


def get_sector_high_open_signals(
    agent_id: str,
    sector_rank: int,
    trade_date: str,
    daily_data: pd.DataFrame,
    context: Dict,
) -> List[Dict]:
    """
    取 T-1 日连板热度排名 sector_rank 的板块，筛选 T 日高开股票。

    :param agent_id:    调用方 agent_id（用于日志）
    :param sector_rank: 目标板块排名（1=最强, 2=次强, 3=第三）
    :param trade_date:  当日交易日 YYYY-MM-DD
    :param daily_data:  全市场日线 DataFrame
    :param context:     引擎上下文（trade_dates / st_stock_list / pre_close_data）
    :return:            命中股票列表 [{ts_code, stock_name, buy_price}, ...]
    """
    st_set = set(context.get("st_stock_list", []))
    trade_dates: List[str] = context.get("trade_dates", [])

    # ── 获取 T-1 交易日 ────────────────────────────────────────────────
    if trade_date not in trade_dates:
        logger.warning(f"[{agent_id}][{trade_date}] trade_date 不在 trade_dates 中")
        return []
    idx = trade_dates.index(trade_date)
    if idx == 0:
        logger.info(f"[{agent_id}][{trade_date}] 无前一交易日，无信号")
        return []
    prev_date = trade_dates[idx - 1]

    # ── 查 T-1 对应排名的板块 ─────────────────────────────────────────
    cpt_df = get_limit_cpt_list(prev_date)
    if cpt_df is None or cpt_df.empty:
        logger.info(f"[{agent_id}][{trade_date}] T-1={prev_date} 无 limit_cpt_list 数据，无信号")
        return []
    if "rank" not in cpt_df.columns or "name" not in cpt_df.columns:
        logger.warning(f"[{agent_id}][{trade_date}] limit_cpt_list 缺少 rank/name 列")
        return []

    cpt_df = cpt_df.copy()
    cpt_df["rank"] = pd.to_numeric(cpt_df["rank"], errors="coerce")
    sorted_df = cpt_df.sort_values("rank").reset_index(drop=True)

    if len(sorted_df) < sector_rank:
        logger.info(
            f"[{agent_id}][{trade_date}] T-1={prev_date} 板块数量({len(sorted_df)}) "
            f"< 目标排名({sector_rank})，无信号"
        )
        return []

    target_row = sorted_df.iloc[sector_rank - 1]   # 0-indexed
    target_sector = str(target_row["name"]).strip()
    logger.info(
        f"[{agent_id}][{trade_date}] T-1 排名{sector_rank}板块：{target_sector}"
        f"（rank={target_row['rank']}）"
    )

    # ── 获取该板块股票集合 ─────────────────────────────────────────────
    sector_stocks_raw = get_stocks_in_sector(target_sector)
    if not sector_stocks_raw:
        logger.info(f"[{agent_id}][{trade_date}] 板块[{target_sector}]无对应股票，无信号")
        return []
    sector_ts_set = {item["ts_code"] for item in sector_stocks_raw}

    # ── 构建前收价映射 ────────────────────────────────────────────────
    pre_close_map: Dict[str, float] = {}
    pre_data = context.get("pre_close_data", pd.DataFrame())
    if not pre_data.empty and "ts_code" in pre_data.columns and "close" in pre_data.columns:
        pre_close_map = dict(zip(pre_data["ts_code"], pre_data["close"]))
    if "pre_close" in daily_data.columns:
        for _, row in daily_data.iterrows():
            if row["ts_code"] not in pre_close_map:
                pre_close_map[row["ts_code"]] = row["pre_close"]

    # ── 筛选当日高开股票 ──────────────────────────────────────────────
    result: Dict[str, Dict] = {}
    for _, row in daily_data.iterrows():
        ts_code = row["ts_code"]
        if ts_code not in sector_ts_set:
            continue
        if ts_code in st_set:
            continue
        code_prefix = ts_code.split(".")[0]
        if code_prefix.startswith(("83", "87", "88")) or ts_code.endswith(".BJ"):
            continue
        pre_close = pre_close_map.get(ts_code, 0.0)
        if pre_close <= 0:
            continue
        open_price = float(row.get("open", 0.0) if hasattr(row, "get") else 0.0)
        if open_price <= pre_close:
            continue

        if ts_code not in result:
            result[ts_code] = {
                "ts_code":    ts_code,
                "stock_name": str(row.get("name", "") if hasattr(row, "get") else ""),
                "buy_price":  open_price,
            }

    final = list(result.values())
    logger.info(f"[{agent_id}][{trade_date}] 板块[{target_sector}]高开命中 {len(final)} 只")
    return final
