"""
模型信号平铺开盘买入（ModelOpenBuyAgent）
==========================================
策略逻辑
--------
跟踪 SectorHeatStrategy 模型输出的买入信号：

1. D 日：调用模型完整选股流程，获取信号股列表
2. D+1 日：以开盘价平铺买入所有信号股（模型输出几个就平铺几个）

buy_price = D+1 开盘价，引擎 T+1 跟踪计算的 close_return 即为
    (D+1 收盘 - D+1 开盘) / D+1 开盘 = D+1 日内收益率

设计意图
--------
最简单的信号跟踪方式：次日开盘无条件买入。
用于衡量模型信号在"次日开盘买"场景下的盈亏表现。
"""
from typing import List, Dict

import pandas as pd

from agent_stats.agent_base import BaseAgent
from agent_stats.agents._model_signal_helper import get_model_signal_stocks
from utils.common_tools import get_daily_kline_data
from utils.log_utils import logger


class ModelOpenBuyAgent(BaseAgent):
    agent_id   = "model_open_buy"
    agent_name = "模型信号平铺开盘买入"
    agent_desc = (
        "跟踪 SectorHeatStrategy 模型信号，D+1 日以开盘价平铺买入所有信号股。"
        "buy_price=D+1 开盘价，close_return 即日内收益率。"
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
            logger.info(f"[{self.agent_id}][{trade_date}] 模型无信号，跳过")
            return []

        # ── 获取 D+1 交易日 ──────────────────────────────────────────────────
        trade_dates = context.get("trade_dates", [])
        next_date = _get_next_trade_date(trade_dates, trade_date_dash)
        if not next_date:
            logger.warning(f"[{self.agent_id}][{trade_date}] 无法获取 D+1 交易日，跳过")
            return []

        # ── 获取 D+1 日线（取开盘价）─────────────────────────────────────────
        ts_codes = [s["ts_code"] for s in signals]
        next_daily = get_daily_kline_data(next_date, ts_code_list=ts_codes)
        if next_daily.empty:
            logger.warning(f"[{self.agent_id}][{trade_date}] D+1({next_date}) 日线数据为空，跳过")
            return []

        open_map = {row["ts_code"]: float(row["open"]) for _, row in next_daily.iterrows()}
        name_map = {s["ts_code"]: s["stock_name"] for s in signals}

        # ── 构建结果：buy_price = D+1 开盘价 ────────────────────────────────
        result = []
        for sig in signals:
            ts = sig["ts_code"]
            open_p = open_map.get(ts, 0)
            if open_p <= 0:
                continue
            result.append({
                "ts_code":    ts,
                "stock_name": name_map.get(ts, ""),
                "buy_price":  open_p,
            })

        logger.info(
            f"[{self.agent_id}][{trade_date}] D+1({next_date}) 开盘买入 {len(result)} 只: "
            + " | ".join(f"{s['ts_code']}(open={s['buy_price']:.2f})" for s in result)
        )
        return result


def _get_next_trade_date(trade_dates: List[str], trade_date: str) -> str:
    """从交易日列表中找到 trade_date 的下一个交易日"""
    try:
        idx = trade_dates.index(trade_date)
        if idx + 1 < len(trade_dates):
            return trade_dates[idx + 1]
    except ValueError:
        pass
    return ""
