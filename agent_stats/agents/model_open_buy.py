"""
模型信号平铺开盘买入（ModelOpenBuyAgent）
==========================================
策略逻辑（时序：D-1 日选股，D 日买入）
--------
跟踪 SectorHeatStrategy 模型输出的买入信号：

1. D-1 日（trade_date 前一个交易日）：调用模型完整选股流程，获取信号股列表
2. D 日（trade_date）：以开盘价平铺买入所有信号股

buy_price = D 日开盘价（已知数据，无未来函数）

设计意图
--------
最简单的信号跟踪方式：用 D-1 日模型信号，次日（D 日）开盘无条件买入。
与 high_position_stock、mid_position_stock 等 agent 时序对齐。
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
        "跟踪 SectorHeatStrategy 模型信号，D-1 日生成信号，D 日以开盘价平铺买入所有信号股。"
        "buy_price=D 日开盘价，无未来函数，与其他 agent 时序对齐。"
    )

    def get_signal_stock_pool(
        self,
        trade_date: str,
        daily_data: pd.DataFrame,
        context: Dict,
    ) -> List[Dict]:
        # ── 日期格式（trade_date = D 日）────────────────────────────────────
        if len(trade_date) == 8 and trade_date.isdigit():
            trade_date_dash = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}"
        else:
            trade_date_dash = trade_date

        # ── 获取 D-1 日（信号生成日）────────────────────────────────────────
        trade_dates = context.get("trade_dates", [])
        if trade_date_dash not in trade_dates:
            return []
        idx = trade_dates.index(trade_date_dash)
        if idx == 0:
            logger.info(f"[{self.agent_id}][{trade_date}] 无 D-1 交易日，跳过")
            return []
        prev_date = trade_dates[idx - 1]  # D-1 日（YYYY-MM-DD）

        # ── 获取 D-1 日日线 ──────────────────────────────────────────────────
        prev_daily = get_daily_kline_data(prev_date)
        if prev_daily is None or prev_daily.empty:
            logger.warning(f"[{self.agent_id}][{trade_date}] D-1({prev_date}) 日线为空，跳过")
            return []

        # ── 用 D-1 日数据获取模型信号（无未来函数）──────────────────────────
        signals = get_model_signal_stocks(prev_date, prev_daily, caller_agent_id=self.agent_id)
        if not signals:
            logger.info(f"[{self.agent_id}][{trade_date}] D-1({prev_date}) 模型无信号，跳过")
            return []

        # ── 买入价 = D 日开盘价（daily_data 即 D 日数据）────────────────────
        ts_codes = [s["ts_code"] for s in signals]
        name_map = {s["ts_code"]: s["stock_name"] for s in signals}
        d_sub = daily_data[daily_data["ts_code"].isin(ts_codes)]
        open_map: Dict[str, float] = {}
        for _, row in d_sub.iterrows():
            open_p = float(row.get("open", 0) or 0)
            if open_p > 0:
                open_map[row["ts_code"]] = open_p

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
            f"[{self.agent_id}][{trade_date}] D-1({prev_date})信号→D({trade_date})开盘买 {len(result)} 只: "
            + " | ".join(f"{s['ts_code']}(open={s['buy_price']:.2f})" for s in result)
        )
        return result
