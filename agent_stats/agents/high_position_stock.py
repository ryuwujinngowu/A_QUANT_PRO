"""
高位股跟踪策略（HighPositionStockAgent）
=========================================
策略逻辑
--------
全市场股票按 D 日 20 日涨幅排序，取前 1% 为基础池（约 50 只强势股），
再从中取前 10% 作为高位股（约 5 只龙头，最少 3 只）。

选股步骤：
  1. 20 日涨幅 = D 日收盘价 / D-21 收盘价 - 1
  2. 剔除新股（上市日期在近 21 个交易日内）
  3. 剔除 ST / *ST 股票
  4. 降序排序，取前 1% = 基础池
  5. 基础池中前 10%（不足 3 只则取前 3 只）= 高位股候选
  6. 剔除一字涨停无开板（open == close == high == 涨停价，买不进去）

买入价：
  - 非一字板：开盘价买入
  - 一字板开过板（high > open，收盘仍封板）：涨停价买入（模拟排队）

注意事项
--------
- 基础池大小随行情动态变化（约等于 1% × 有效股票总数），龙头更精准。
- 最少 3 只保证行情极端时仍有信号，避免空仓偏差。
"""
from typing import List, Dict

import pandas as pd

from agent_stats.agent_base import BaseAgent
from agent_stats.agents._position_stock_helpers import (
    BASE_PCT, HIGH_PCT, MIN_HIGH, LOOKBACK_DAYS,
    build_gain_list, is_yizi_limit_up,
)
from utils.common_tools import calc_limit_up_price, get_daily_kline_data
from utils.log_utils import logger


class HighPositionStockAgent(BaseAgent):
    agent_id   = "high_position_stock"
    agent_name = "高位股跟踪选手"
    agent_desc = (
        "高位股跟踪：全市场按20日涨幅排序取前1%为基础池，"
        "再取基础池前10%（最少3只）为高位股；"
        "剔除新股/ST/一字板无开板；"
        "非一字板开盘买入，一字板开过板涨停价排队买入。"
    )

    def get_signal_stock_pool(
        self,
        trade_date: str,
        daily_data: pd.DataFrame,
        context: Dict,
    ) -> List[Dict]:
        st_set      = set(context.get("st_stock_list", []))
        trade_dates = context.get("trade_dates", [])

        if trade_date not in trade_dates:
            logger.warning(f"[{self.agent_id}][{trade_date}] trade_date 不在 trade_dates 中")
            return []
        idx = trade_dates.index(trade_date)
        # 需要 D-1 日，再加上 D-1 之前的 LOOKBACK_DAYS 历史
        if idx < LOOKBACK_DAYS + 1:
            logger.info(f"[{self.agent_id}][{trade_date}] 历史数据不足，跳过")
            return []

        # ── 用 D-1 日数据选股（无未来函数）─────────────────────────────────
        prev_date = trade_dates[idx - 1]  # D-1 日
        prev_daily = get_daily_kline_data(prev_date)
        if prev_daily is None or prev_daily.empty:
            logger.warning(f"[{self.agent_id}][{trade_date}] D-1({prev_date}) 日线为空，跳过")
            return []

        # D-1 日的前收价（= D-2 日收盘，来自 D-1 的 pre_close 列）
        prev_pre_close_map: Dict[str, float] = {}
        if "pre_close" in prev_daily.columns:
            for _, row in prev_daily.iterrows():
                v = float(row.get("pre_close", 0) or 0)
                if v > 0:
                    prev_pre_close_map[row["ts_code"]] = v

        gain_list, _ = build_gain_list(prev_daily, st_set, trade_dates, prev_date)

        if not gain_list:
            logger.info(f"[{self.agent_id}][{trade_date}] 无有效股票可排序")
            return []

        # 前 1% 基础池
        base_n = max(MIN_HIGH, int(len(gain_list) * BASE_PCT))
        base_pool = gain_list[:base_n]

        # 基础池中前 10%，最少 3 只
        high_n = max(MIN_HIGH, int(len(base_pool) * HIGH_PCT))
        high_pool = base_pool[:high_n]

        # D 日 open map（买入价来源，无未来函数）
        d_open_map: Dict[str, float] = {
            row["ts_code"]: float(row.get("open", 0) or 0)
            for _, row in daily_data.iterrows()
        }

        # 剔除 D-1 一字板无开板（D-1 数据已知，无未来函数）+ 确定 D 日买入价
        result = []
        for ts_code, name, gain_20d, prev_row in high_pool:
            pre_close = prev_pre_close_map.get(ts_code, 0.0)  # D-2 收盘 = D-1 前收
            limit_up_d1 = calc_limit_up_price(ts_code, pre_close) if pre_close > 0 else 0.0

            # 若 D-1 一字板无开板，今日可能继续封板，跳过
            if is_yizi_limit_up(prev_row, limit_up_d1):
                continue

            # 买入价 = D 日开盘价（已知数据）
            buy_price = d_open_map.get(ts_code, 0.0)
            if buy_price <= 0:
                continue

            result.append({
                "ts_code":    ts_code,
                "stock_name": name,
                "buy_price":  round(buy_price, 2),
            })

        logger.info(
            f"[{self.agent_id}][{trade_date}] 命中 {len(result)} 只"
            f"（D-1有效股={len(gain_list)}，基础池={base_n}，高位={high_n}，"
            f"剔除一字板后={len(result)}）"
            f" | D-1高位股最大20日涨幅 {base_pool[0][2]:.1%}"
        )
        return result
