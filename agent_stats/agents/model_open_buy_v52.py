"""
模型v5.2信号平铺开盘买入（ModelOpenBuyV52Agent）
=================================================
策略逻辑（时序：D-1 日选股，D 日买入）
--------
使用 sector_heat_xgb_v5.2_auc_first.pkl，独立于策略层：

1. D-1 日（trade_date 前一个交易日）：
   - 选 Top3 热点板块（基于 D-1 日数据，无未来函数）
   - 构建候选池（涨停基因过滤 + 流动性过滤 + 封板过滤）
   - 计算特征 → sector_heat_xgb_v5.2_auc_first.pkl 预测（prob 阈值 0.50）
   - 生成买入信号列表

2. D 日（trade_date）：以开盘价平铺买入所有信号股

buy_price = D 日开盘价（已知数据，无未来函数）

区别于 model_open_buy
----------------------
model_open_buy 使用老版模型（sector_heat_xgb_model.pkl，prob 阈值 0.60）。
本 agent 使用 v5.2 AUC-first 模型，prob 阈值 0.50，可横向对比两版模型表现。
"""
from typing import List, Dict

import pandas as pd

from agent_stats.agent_base import BaseAgent
from agent_stats.agents._model_signal_helper_v52 import get_model_signal_stocks
from utils.common_tools import get_daily_kline_data
from utils.log_utils import logger


class ModelOpenBuyV52Agent(BaseAgent):
    agent_id   = "model_open_buy_v52"
    agent_name = "模型v5.2信号开盘平铺"
    agent_desc = (
        "使用 sector_heat_xgb_v5.2_auc_first 模型（prob≥0.50），D-1 日生成信号，"
        "D 日以开盘价平铺买入所有信号股。与 model_open_buy（老模型prob≥0.60）可做横向比较。"
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
            f"[{self.agent_id}][{trade_date}] D-1({prev_date})v5.2信号→D({trade_date})开盘买 {len(result)} 只: "
            + " | ".join(f"{s['ts_code']}(open={s['buy_price']:.2f})" for s in result)
        )
        return result
