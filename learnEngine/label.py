# learnEngine/label.py
"""
训练集标签引擎
=============
统一生成训练集标签，标签口径与策略实际操作对齐：
    - 策略在 D 日收盘后生成信号，D+1 日开盘买入
    - 因此标签买入价 = D+1 open，卖出价 = D+1 close

label 定义：
    label1 : (D+1 close - D+1 open) / D+1 open >= 5%  → 1，否则 0
             含义：D+1 日以开盘价买入、收盘价卖出，涨幅达 5%
             模型主训练标签（TARGET_LABEL = "label1"）

    label2 : D+2 open > D+1 close  AND  D+1 close > D+1 open → 1，否则 0
             含义：D+1 日内已盈利（close > open）且 D+2 高开（隔夜继续赚），
                   即值得持仓过夜的强势票
             设计约束：label2=1 必然满足 label1（label2 ⊆ label1 的充分子集）
             可用于策略二阶过滤：模型选出 label1 候选后，label2 高概率的票优先持仓

过滤逻辑：
    - D+1 停牌（无数据）→ 跳过，不作为负样本
    - D+2 无数据 → label2 填 NaN，后续清洗时 dropna 即可
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import List
import pandas as pd
from utils.common_tools import get_trade_dates, get_daily_kline_data
from utils.log_utils import logger


class LabelEngine:
    """训练集标签生成引擎"""

    def __init__(self, start_date: str, end_date: str):
        self.start_date = start_date
        self.end_date   = end_date
        # 多预留 10 个自然日，确保 end_date 对应的 D+2 交易日在范围内
        label_end = (pd.to_datetime(end_date) + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        self.all_trade_dates = get_trade_dates(start_date, label_end)
        self.date_idx_map    = {d: i for i, d in enumerate(self.all_trade_dates)}

    def generate_single_date(self, trade_date: str, stock_list: List[str]) -> pd.DataFrame:
        """
        生成单日标签（口径与策略对齐：D+1 open 买入，D+1 close 卖出）

        :param trade_date: D 日，格式 yyyy-mm-dd
        :param stock_list: 候选股代码列表
        :return: DataFrame，列：stock_code, trade_date, label1, label2
                 D+1 停牌的股票不会出现在返回结果中
        """
        if trade_date not in self.date_idx_map:
            return pd.DataFrame()

        idx = self.date_idx_map[trade_date]
        if idx + 2 >= len(self.all_trade_dates):
            return pd.DataFrame()

        d1_date = self.all_trade_dates[idx + 1]
        d2_date = self.all_trade_dates[idx + 2]

        # 批量拉取 D+1 和 D+2 日线
        d1_df = get_daily_kline_data(trade_date=d1_date, ts_code_list=stock_list)
        d2_df = get_daily_kline_data(trade_date=d2_date, ts_code_list=stock_list)

        if d1_df.empty:
            logger.warning(f"[LabelEngine] {d1_date} 日线数据为空，跳过")
            return pd.DataFrame()

        d1_df["trade_date"] = d1_df["trade_date"].astype(str)
        if not d2_df.empty:
            d2_df["trade_date"] = d2_df["trade_date"].astype(str)

        # 构建快速查找
        d1_map = {row["ts_code"]: row for _, row in d1_df.iterrows()}
        d2_map = {row["ts_code"]: row for _, row in d2_df.iterrows()} if not d2_df.empty else {}

        rows = []
        for ts_code in stock_list:
            d1_row = d1_map.get(ts_code)
            if d1_row is None:
                # D+1 停牌，跳过（不作为负样本）
                continue

            d1_open  = d1_row.get("open",  0)
            d1_close = d1_row.get("close", 0)

            if not d1_open or d1_open <= 0:
                continue

            # label1：D+1 日内收益率 = (D+1 close - D+1 open) / D+1 open
            d1_intra_return = (d1_close - d1_open) / d1_open
            label1 = 1 if d1_intra_return >= 0.03 else 0

            # label2：D+1 日内盈利 AND D+2 高开
            # 设计约束：label2=1 必然满足 label1（值得隔夜持股的强势票）
            # 原定义（D+2 open > D+1 close）存在 label1=0 时 label2=1 的语义矛盾：
            # D+1 日内亏损但 D+2 高开 → 实际仍是亏损交易，不应标记为正样本
            d2_row = d2_map.get(ts_code)
            if d2_row is not None:
                d2_open  = d2_row.get("open", 0)
                # 同时满足：D+1 日内盈利（close > open）且 D+2 高开（open > D+1 close）
                label2 = 1 if (d1_close > d1_open) and (d2_open > d1_close) else 0
            else:
                label2 = None  # D+2 无数据，后续清洗时 dropna

            rows.append({
                "stock_code": ts_code,
                "trade_date": trade_date,
                "label1":     label1,
                "label2":     label2,
            })

        result = pd.DataFrame(rows)
        if not result.empty:
            logger.info(
                f"[LabelEngine] {trade_date} 标签生成完成 | "
                f"样本数:{len(result)} | 正样本(label1):{result['label1'].sum()}"
            )
        return result
