"""
模型信号拉涨买入（ModelSurgeBuyAgent）
=======================================
策略逻辑
--------
跟踪 SectorHeatStrategy 模型输出的买入信号，D+1 日根据分钟线判断拉涨信号后买入：

1. D 日：调用模型完整选股流程，获取信号股列表
2. D+1 日 09:30-10:00（前 30 分钟），以 3 分钟为时间切片遍历分钟线：
   - 先决条件：在拉涨出现之前，股价不能跌破 D+1 开盘价的 -1%（即 open × 0.99）
     若某切片 low < open × 0.99 → 放弃该股
   - 拉涨信号：某切片内 (high - low) / open > 4% 且 close > open × 1.02
     → 以该切片的 close 价格买入

buy_price = 拉涨切片的 close 价格

设计意图
--------
核心假设：模型选出的强势股在 D+1 开盘后未破位且出现放量拉升时，
追涨买入可以捕捉到日内趋势性收益。通过「未破开盘-1%」条件过滤
弱势股和低开补跌票。
"""
from typing import List, Dict

import pandas as pd

from agent_stats.agent_base import BaseAgent
from agent_stats.agents._model_signal_helper import get_model_signal_stocks
from data.data_cleaner import data_cleaner, TushareRateLimitAbort
from utils.common_tools import get_daily_kline_data
from utils.log_utils import logger

# ── 策略参数 ──────────────────────────────────────────────────────────────────
WINDOW_START     = "09:30"  # 监测窗口开始
WINDOW_END       = "10:00"  # 监测窗口结束（前 30 分钟）
SLICE_MINUTES    = 3        # 时间切片长度（分钟）
BREAK_PCT        = 0.01     # 跌破阈值：open × (1 - BREAK_PCT)
SURGE_AMPLITUDE  = 0.04     # 拉涨振幅阈值：(high - low) / open > 4%
SURGE_CLOSE_PCT  = 0.02     # 拉涨收盘阈值：close > open × (1 + SURGE_CLOSE_PCT)


class ModelSurgeBuyAgent(BaseAgent):
    agent_id   = "model_surge_buy"
    agent_name = "模型信号拉涨买入"
    agent_desc = (
        "跟踪 SectorHeatStrategy 模型信号，D+1 前 30 分钟以 3 分钟切片监测："
        "未跌破开盘-1% 且出现振幅>4%/收盘>开盘+2% 的拉涨时买入。"
    )

    def get_signal_stock_pool(
        self,
        trade_date: str,
        daily_data: pd.DataFrame,
        context: Dict,
    ) -> List[Dict]:
        # ── 日期格式 ─────────────────────────────────────────────────────────
        if len(trade_date) == 8 and trade_date.isdigit():
            trade_date_dash = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}"
        else:
            trade_date_dash = trade_date

        # ── 获取 D 日模型信号 ────────────────────────────────────────────────
        signals = get_model_signal_stocks(trade_date_dash, daily_data, caller_agent_id=self.agent_id)
        if not signals:
            return []

        # ── 获取 D+1 交易日 ──────────────────────────────────────────────────
        trade_dates = context.get("trade_dates", [])
        next_date = _get_next_trade_date(trade_dates, trade_date_dash)
        if not next_date:
            logger.warning(f"[{self.agent_id}][{trade_date}] 无法获取 D+1 交易日，跳过")
            return []
        next_date_8 = next_date.replace("-", "")

        # ── 获取 D+1 日线（取开盘价）─────────────────────────────────────────
        ts_codes = [s["ts_code"] for s in signals]
        next_daily = get_daily_kline_data(next_date, ts_code_list=ts_codes)
        if next_daily.empty:
            logger.warning(f"[{self.agent_id}][{trade_date}] D+1({next_date}) 日线数据为空")
            return []

        open_map = {row["ts_code"]: float(row["open"]) for _, row in next_daily.iterrows()}
        name_map = {s["ts_code"]: s["stock_name"] for s in signals}

        # ── 逐股扫描 D+1 分钟线 ──────────────────────────────────────────────
        result = []
        for sig in signals:
            ts = sig["ts_code"]
            open_p = open_map.get(ts, 0)
            if open_p <= 0:
                continue

            # 拉取 D+1 分钟线
            try:
                min_df = data_cleaner.get_kline_min_by_stock_date(ts, next_date_8)
            except TushareRateLimitAbort:
                raise
            except Exception as e:
                logger.warning(f"[{self.agent_id}][{trade_date}][{ts}] D+1 分钟线获取失败: {e}")
                self._minute_fetch_failures.append(ts)
                continue

            if min_df is None or min_df.empty:
                continue

            # 检测拉涨信号
            buy_price = _detect_surge(min_df, open_p, ts, trade_date, self.agent_id)
            if buy_price is not None:
                result.append({
                    "ts_code":    ts,
                    "stock_name": name_map.get(ts, ""),
                    "buy_price":  buy_price,
                })

        logger.info(
            f"[{self.agent_id}][{trade_date}] D+1({next_date}) 拉涨买入 {len(result)} 只 "
            f"（信号={len(signals)} 只）: "
            + " | ".join(f"{s['ts_code']}(buy={s['buy_price']:.2f})" for s in result)
        )
        return result


def _detect_surge(
    min_df: pd.DataFrame,
    open_price: float,
    ts_code: str,
    trade_date: str,
    agent_id: str,
) -> float:
    """
    检测 D+1 前 30 分钟内的拉涨信号。

    以 3 分钟为切片遍历 09:30-10:00 的分钟线：
      1. 先决条件：在拉涨出现前，所有切片 low 不能跌破 open × 0.99
      2. 拉涨信号：某切片 (high - low) / open > 4% 且 close > open × 1.02
      3. 满足则返回该切片的 close 作为买入价，否则返回 None

    :return: 买入价（float），或 None（未触发）
    """
    min_df = min_df.copy()
    try:
        min_df["_hm"] = pd.to_datetime(min_df["trade_time"]).dt.strftime("%H:%M")
    except Exception:
        return None

    # 截取 09:30-10:00 窗口
    window = min_df[(min_df["_hm"] >= WINDOW_START) & (min_df["_hm"] <= WINDOW_END)].copy()
    if window.empty:
        return None

    # 按时间排序
    window = window.sort_values("_hm").reset_index(drop=True)

    # 将 1 分钟线聚合为 3 分钟切片
    break_price = open_price * (1 - BREAK_PCT)
    surge_close_price = open_price * (1 + SURGE_CLOSE_PCT)

    n = len(window)
    i = 0
    while i < n:
        end_i = min(i + SLICE_MINUTES, n)
        slice_df = window.iloc[i:end_i]

        slice_low  = float(slice_df["low"].min())
        slice_high = float(slice_df["high"].max())
        slice_close = float(slice_df.iloc[-1]["close"])

        # 先决条件：未跌破 open - 1%
        if slice_low < break_price:
            logger.debug(
                f"[{agent_id}][{trade_date}][{ts_code}] "
                f"跌破 open-1%: low={slice_low:.2f} < {break_price:.2f}，放弃"
            )
            return None

        # 拉涨检测：振幅 > 4% 且 close > open + 2%
        amplitude = (slice_high - slice_low) / open_price if open_price > 0 else 0
        if amplitude > SURGE_AMPLITUDE and slice_close > surge_close_price:
            logger.debug(
                f"[{agent_id}][{trade_date}][{ts_code}] "
                f"拉涨触发: amp={amplitude:.2%} close={slice_close:.2f} > {surge_close_price:.2f}"
            )
            return round(slice_close, 2)

        i = end_i

    return None


def _get_next_trade_date(trade_dates: List[str], trade_date: str) -> str:
    """从交易日列表中找到 trade_date 的下一个交易日"""
    try:
        idx = trade_dates.index(trade_date)
        if idx + 1 < len(trade_dates):
            return trade_dates[idx + 1]
    except ValueError:
        pass
    return ""
