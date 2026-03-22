"""
昨日最强板块今日高开买入（SectorTopHighOpenAgent）
======================================================
策略逻辑
--------
取前一交易日（T-1）连板热度排名第一的板块，
当日（T 日）该板块内所有高开股票（open > pre_close）以开盘价买入。

命中条件：
  1. T-1 日 limit_cpt_list 中 rank 最小（= 1）的板块为"最强板块"
  2. T 日 daily_data 中该板块所有股票：
     - open > pre_close（高开）
     - 非 ST / *ST 股票
     - 排除北交所
  3. 买入价 = 当日开盘价（open），可在集合竞价结束后（9:25）获知

核心逻辑理由
------------
连板热度排名第一的板块往往是当日市场情绪最强主线，
主线次日惯性效应显著：开盘高开说明资金持续流入，
择机以开盘价买入即可参与惯性上涨。

注意事项
--------
- 若 T-1 无 limit_cpt_list 数据，则当日无信号（不降级）。
- 同一股票只计入一次（dict 去重）。
- 买入价为 open（非固定涨停价），日内收益由 high/low/close 计算。
"""
from typing import List, Dict

import pandas as pd

from agent_stats.agent_base import BaseAgent
from utils.common_tools import get_limit_cpt_list, get_stocks_in_sector
from utils.log_utils import logger


class SectorTopHighOpenAgent(BaseAgent):
    agent_id   = "sector_top_high_open"
    agent_name = "昨日top1板块高开无脑跟选手"
    agent_desc = (
        "昨日最强板块今日高开买入策略：取T-1日连板热度排名第1的板块，"
        "筛选该板块内T日高开（open > pre_close）的非ST/非北交所股票，以开盘价买入。"
    )

    def get_signal_stock_pool(
        self,
        trade_date: str,
        daily_data: pd.DataFrame,
        context: Dict,
    ) -> List[Dict]:
        st_set = set(context.get("st_stock_list", []))
        trade_dates: List[str] = context.get("trade_dates", [])

        # ── 获取 T-1 交易日 ────────────────────────────────────────────────
        if trade_date not in trade_dates:
            logger.warning(f"[{self.agent_id}][{trade_date}] trade_date 不在 trade_dates 中")
            return []
        idx = trade_dates.index(trade_date)
        if idx == 0:
            logger.info(f"[{self.agent_id}][{trade_date}] 无前一交易日，无信号")
            return []
        prev_date = trade_dates[idx - 1]

        # ── 查 T-1 最强板块（rank 最小） ──────────────────────────────────
        cpt_df = get_limit_cpt_list(prev_date)
        if cpt_df is None or cpt_df.empty:
            logger.info(f"[{self.agent_id}][{trade_date}] T-1={prev_date} 无 limit_cpt_list 数据，无信号")
            return []

        if "rank" not in cpt_df.columns or "name" not in cpt_df.columns:
            logger.warning(f"[{self.agent_id}][{trade_date}] limit_cpt_list 缺少 rank/name 列")
            return []

        cpt_df["rank"] = pd.to_numeric(cpt_df["rank"], errors="coerce")
        top_row = cpt_df.sort_values("rank").iloc[0]
        top_sector = str(top_row["name"]).strip()
        logger.info(f"[{self.agent_id}][{trade_date}] T-1 最强板块：{top_sector}（rank={top_row['rank']}）")

        # ── 获取该板块股票集合 ─────────────────────────────────────────────
        sector_stocks_raw = get_stocks_in_sector(top_sector)
        if not sector_stocks_raw:
            logger.info(f"[{self.agent_id}][{trade_date}] 板块 [{top_sector}] 无对应股票，无信号")
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
                continue   # 未高开

            if ts_code not in result:
                result[ts_code] = {
                    "ts_code":    ts_code,
                    "stock_name": str(row.get("name", "") if hasattr(row, "get") else ""),
                    "buy_price":  open_price,
                }

        final = list(result.values())
        logger.info(
            f"[{self.agent_id}][{trade_date}] 板块[{top_sector}]高开命中 {len(final)} 只"
        )
        return final
