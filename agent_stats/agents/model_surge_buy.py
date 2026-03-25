"""
模型信号拉涨买入（ModelSurgeBuyAgent）
=======================================
策略逻辑（时序：D-1 日选股，D 日买入）
--------
跟踪 SectorHeatStrategy 模型输出的买入信号，D 日根据分钟线判断拉涨信号后买入：

1. D-1 日（trade_date 前一个交易日）：调用模型完整选股流程，获取信号股列表
2. D 日（trade_date）09:30-10:30，以 5 分钟为时间切片遍历分钟线：
   - 拉涨信号：某切片内 (high - low) / open > 4%
     → 以该切片的 close 价格买入

buy_price = 拉涨切片的 close 价格（无未来函数）

设计意图
--------
核心假设：模型选出的强势股在 D 日开盘后未破位且出现放量拉升时，
追涨买入可以捕捉到日内趋势性收益。通过「未破开盘-1%」条件过滤
弱势股和低开补跌票。与其他 agent 时序对齐：D-1 日生成候选池，D 日完成买入。
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
WINDOW_END       = "10:30"  # 监测窗口结束
SLICE_MINUTES    = 5        # 时间切片长度（分钟）
SURGE_AMPLITUDE  = 0.04     # 拉涨振幅阈值：(high - low) / open > 4%


class ModelSurgeBuyAgent(BaseAgent):
    agent_id   = "model_surge_buy"
    agent_name = "模型信号拉涨买入"
    agent_desc = (
        "跟踪 SectorHeatStrategy 模型信号，D-1 日生成信号，D 日 09:30-10:30 以 5 分钟切片监测："
        "出现振幅(high-low)/open > 4% 的切片时以切片 close 买入。"
        "时序与其他 agent 对齐：D-1 日生成候选池，D 日完成买入。"
    )

    def get_signal_stock_pool(
        self,
        trade_date: str,
        daily_data: pd.DataFrame,
        context: Dict,
    ) -> List[Dict]:
        # ── 日期格式（trade_date = D 日，即买入日）──────────────────────────
        if len(trade_date) == 8 and trade_date.isdigit():
            trade_date_dash = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}"
        else:
            trade_date_dash = trade_date
        trade_date_8 = trade_date_dash.replace("-", "")

        # ── 获取 D-1 日（信号生成日）────────────────────────────────────────
        trade_dates = context.get("trade_dates", [])
        if trade_date_dash not in trade_dates:
            return []
        idx = trade_dates.index(trade_date_dash)
        if idx == 0:
            logger.info(f"[{self.agent_id}][{trade_date}] 无 D-1 交易日，跳过")
            return []
        prev_date = trade_dates[idx - 1]  # D-1 日（YYYY-MM-DD）

        # ── 获取 D-1 日日线并生成模型信号 ───────────────────────────────────
        prev_daily = get_daily_kline_data(prev_date)
        if prev_daily is None or prev_daily.empty:
            logger.warning(f"[{self.agent_id}][{trade_date}] D-1({prev_date}) 日线为空，跳过")
            return []

        signals = get_model_signal_stocks(prev_date, prev_daily, caller_agent_id=self.agent_id)
        if not signals:
            return []

        # ── 从 D 日日线（daily_data）取开盘价 ───────────────────────────────
        ts_codes = [s["ts_code"] for s in signals]
        name_map = {s["ts_code"]: s["stock_name"] for s in signals}
        d_sub = daily_data[daily_data["ts_code"].isin(ts_codes)]
        open_map: Dict[str, float] = {}
        for _, row in d_sub.iterrows():
            open_p = float(row.get("open", 0) or 0)
            if open_p > 0:
                open_map[row["ts_code"]] = open_p

        # ── 逐股扫描 D 日（trade_date）分钟线 ───────────────────────────────
        result = []
        for sig in signals:
            ts = sig["ts_code"]
            open_p = open_map.get(ts, 0)
            if open_p <= 0:
                continue

            # 拉取 D 日分钟线
            try:
                min_df = data_cleaner.get_kline_min_by_stock_date(ts, trade_date_8)
            except TushareRateLimitAbort:
                raise
            except Exception as e:
                logger.warning(f"[{self.agent_id}][{trade_date}][{ts}] D 日分钟线获取失败: {e}")
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
            f"[{self.agent_id}][{trade_date}] D日拉涨买入 {len(result)} 只 "
            f"（D-1({prev_date})信号={len(signals)} 只）: "
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

    以 5 分钟为切片遍历 09:30-10:30 的分钟线：
      - 拉涨信号：某切片 (high - low) / open > 4%
      - 满足则返回该切片的 close 作为买入价，否则返回 None

    :return: 买入价（float），或 None（未触发）
    """
    min_df = min_df.copy()
    try:
        min_df["_hm"] = pd.to_datetime(min_df["trade_time"]).dt.strftime("%H:%M")
    except Exception:
        return None

    # 截取 09:30-10:30 窗口
    window = min_df[(min_df["_hm"] >= WINDOW_START) & (min_df["_hm"] <= WINDOW_END)].copy()
    if window.empty:
        return None

    # 按时间排序，聚合为 5 分钟切片
    window = window.sort_values("_hm").reset_index(drop=True)

    n = len(window)
    i = 0
    while i < n:
        end_i = min(i + SLICE_MINUTES, n)
        slice_df = window.iloc[i:end_i]

        slice_low   = float(slice_df["low"].min())
        slice_high  = float(slice_df["high"].max())
        slice_close = float(slice_df.iloc[-1]["close"])

        # 振幅 > 4%（基准为开盘价）
        amplitude = (slice_high - slice_low) / open_price if open_price > 0 else 0
        if amplitude > SURGE_AMPLITUDE:
            logger.debug(
                f"[{agent_id}][{trade_date}][{ts_code}] "
                f"拉涨触发: amp={amplitude:.2%} slice_close={slice_close:.2f}"
            )
            return round(slice_close, 2)

        i = end_i

    return None


