"""
中位股跟踪策略（MidPositionStockAgent）
=========================================
策略逻辑
--------
在高位股基础池（全市场前 1%，约 50 只）中，取前 50% 并剔除高位股部分，
得到中位股候选（约 10~20 只）。

选股步骤：
  1. 20 日涨幅 = D 日收盘价 / D-21 收盘价 - 1
  2. 剔除新股（上市日期在近 21 个交易日内）
  3. 剔除 ST / *ST 股票
  4. 降序排序，取前 1% = 基础池
  5. 高位股 = 基础池前 10%（最少 3 只）
  6. 中位股候选 = 基础池前 50%，剔除高位股
  7. 区间爆发过滤：将 D-21 ~ D 平分为两段（各 ~10 交易日），
     至少一段涨幅 > 高位股第一名 20 日涨幅 * 50%（动态门槛，排除小碎阳慢涨）
  8. 剔除一字涨停无开板

买入价：
  - 非一字板：开盘价买入
  - 一字板开过板（high > open，收盘仍封板）：涨停价买入（模拟排队）

注意事项
--------
- 动态爆发门槛随行情强度自适应：强市门槛高，弱市门槛低，比固定 60% 更合理。
- 该策略与 high_position_stock 互斥，同一只股票不会同时出现在两个池中。
"""
from typing import List, Dict

import pandas as pd

from agent_stats.agent_base import BaseAgent
from agent_stats.agents._position_stock_helpers import (
    BASE_PCT, HIGH_PCT, MIN_HIGH, LOOKBACK_DAYS,
    build_gain_list, is_yizi_limit_up,
)
from utils.common_tools import calc_limit_up_price, get_daily_kline_data, get_kline_day_range, get_qfq_kline_range
from utils.log_utils import logger

# ── 中位股专用参数 ───────────────────────────────────────────────────────────
MID_PCT_OF_BASE   = 0.50    # 基础池中前 50% 为中位股候选范围（含高位股部分）
BURST_RATIO       = 0.50    # 动态爆发门槛 = 高位股第一名 20 日涨幅 * 50%


def _check_burst(
    close_by_date: Dict[str, float],
    key_dates_fmt: List[str],
    burst_threshold: float,
) -> bool:
    """
    区间爆发检查：将 key_dates_fmt = [D-21, D-10, D] 分为两段，
    任一段涨幅 > burst_threshold 即通过。
    """
    if len(key_dates_fmt) < 3:
        return False
    segments = [
        (key_dates_fmt[0], key_dates_fmt[1]),
        (key_dates_fmt[1], key_dates_fmt[2]),
    ]
    for seg_start, seg_end in segments:
        c_start = close_by_date.get(seg_start, 0.0)
        c_end   = close_by_date.get(seg_end, 0.0)
        if c_start <= 0 or c_end <= 0:
            continue
        if c_end / c_start - 1 > burst_threshold:
            return True
    return False


class MidPositionStockAgent(BaseAgent):
    agent_id   = "mid_position_stock"
    agent_name = "中位股跟踪选手"
    agent_desc = (
        "中位股跟踪：全市场按20日涨幅排序取前1%为基础池，"
        "基础池前50%剔除高位股为中位候选；"
        "21日窗口分两段，至少一段涨幅>高位股最大涨幅*50%（动态门槛，排除小碎阳）；"
        "剔除新股/ST/一字板无开板；非一字板开盘买入，一字板开过板涨停价排队买入。"
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

        # lookback_dates 基于 D-1（含 D-1，不含 D）
        prev_idx = idx - 1
        lookback_dates = trade_dates[prev_idx - LOOKBACK_DAYS: prev_idx + 1]

        gain_list, _ = build_gain_list(prev_daily, st_set, trade_dates, prev_date)

        if not gain_list:
            logger.info(f"[{self.agent_id}][{trade_date}] 无有效股票可排序")
            return []

        # 前 1% 基础池
        base_n = max(MIN_HIGH, int(len(gain_list) * BASE_PCT))
        base_pool = gain_list[:base_n]

        # 高位股集合 + 动态爆发门槛
        high_n = max(MIN_HIGH, int(len(base_pool) * HIGH_PCT))
        high_set = set(g[0] for g in base_pool[:high_n])
        top_gain        = base_pool[0][2]
        burst_threshold = top_gain * BURST_RATIO

        # 中位股候选 = 基础池前 50% 排除高位股
        mid_n    = max(MIN_HIGH, int(len(base_pool) * MID_PCT_OF_BASE))
        mid_pool = [g for g in base_pool[:mid_n] if g[0] not in high_set]

        if not mid_pool:
            logger.info(f"[{self.agent_id}][{trade_date}] 中位股候选池为空")
            return []

        # 区间爆发过滤（动态门槛，基于 D-1 回看窗口，无未来函数）
        mid_date_idx  = len(lookback_dates) // 2
        key_dates     = [lookback_dates[0], lookback_dates[mid_date_idx], lookback_dates[-1]]
        key_dates_fmt = [d.replace("-", "") for d in key_dates]

        mid_codes  = [g[0] for g in mid_pool]
        # 优先 QFQ 计算区间爆发（消除除权除息导致的价格跳空失真），降级用不复权
        key_kline = get_qfq_kline_range(mid_codes, lookback_dates[0], lookback_dates[-1])
        if key_kline.empty:
            key_kline = get_kline_day_range(mid_codes, lookback_dates[0], lookback_dates[-1])

        stock_date_close: Dict[str, Dict[str, float]] = {}
        if not key_kline.empty:
            key_date_set = set(key_dates_fmt)
            for _, kr in key_kline.iterrows():
                ts = kr["ts_code"]
                td = str(kr["trade_date"]).replace("-", "")
                if td in key_date_set:
                    stock_date_close.setdefault(ts, {})[td] = float(kr["close"])

        burst_passed = [
            g for g in mid_pool
            if _check_burst(stock_date_close.get(g[0], {}), key_dates_fmt, burst_threshold)
        ]

        if not burst_passed:
            logger.info(
                f"[{self.agent_id}][{trade_date}] 中位候选 {len(mid_pool)} 只，"
                f"区间爆发过滤（门槛={burst_threshold:.1%}）后全部淘汰"
            )
            return []

        # D 日 open map（买入价来源，无未来函数）
        d_open_map: Dict[str, float] = {
            row["ts_code"]: float(row.get("open", 0) or 0)
            for _, row in daily_data.iterrows()
        }

        # 剔除 D-1 一字板无开板 + 确定 D 日买入价
        result = []
        for ts_code, name, gain_20d, prev_row in burst_passed:
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
            f"（D-1有效股={len(gain_list)}，基础池={base_n}，中位候选={len(mid_pool)}，"
            f"爆发门槛={burst_threshold:.1%}，爆发通过={len(burst_passed)}，"
            f"剔除一字板后={len(result)}）"
        )
        return result
