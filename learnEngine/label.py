# learnEngine/label.py
"""
训练集标签引擎
=============
统一生成训练集标签，标签口径与策略实际操作对齐：
    - 策略在 D 日收盘后生成信号，D+1 日开盘买入
    - 因此标签买入价 = D+1 open，卖出价 = D+1 close

label 定义（完整版）：

二分类标签（1/0，用于分类模型训练）：
    label1              : D+1 日内收益率 >= 5%（主训练标签）
    label2              : D+1 日内盈利 AND D+2 高开（隔夜强势票）
    label1_3pct         : D+1 日内收益率 >= 3%（低门槛，更多正样本）
    label1_8pct         : D+1 日内收益率 >= 8%（高门槛，强势票过滤）
    label_d2_limit_down : D+2 日跌停（pct_chg <= -9.5%）→ 1，用于黑名单模型

浮点标签（用于回归模型或分级惩罚）：
    label_raw_return    : D+1 日内实际收益率（分级惩罚权重基础）
    label_open_gap      : (D+1 open - D close) / D close，开盘溢价率
    label_d1_high       : (D+1 high - D+1 open) / D+1 open，日内最大浮盈
    label_d1_low        : (D+1 low  - D+1 open) / D+1 open，日内最大回撤（负数）
    label_d1_pct_chg    : D+1 pct_chg（收盘相对前收涨跌幅 %，与实盘统计口径对齐）
    label_d2_return     : (D+2 close - D+1 open) / D+1 open，持有2日总收益

过滤逻辑：
    - D+1 停牌（无数据）→ 跳过，不作为负样本
    - D+2 无数据 → 涉及 D+2 的标签填 None（dataset.py 不 dropna 这些列）
    - D close 无法获取 → label_open_gap 填 None
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
        生成单日全量标签（口径与策略对齐：D+1 open 买入，D+1 close 卖出）

        :param trade_date: D 日，格式 yyyy-mm-dd
        :param stock_list: 候选股代码列表
        :return: DataFrame，列见模块头部 label 定义
                 D+1 停牌的股票不会出现在返回结果中
        """
        if trade_date not in self.date_idx_map:
            return pd.DataFrame()

        idx = self.date_idx_map[trade_date]
        if idx + 2 >= len(self.all_trade_dates):
            return pd.DataFrame()

        d1_date = self.all_trade_dates[idx + 1]
        d2_date = self.all_trade_dates[idx + 2]

        # 批量拉取 D / D+1 / D+2 日线
        d0_df = get_daily_kline_data(trade_date=trade_date, ts_code_list=stock_list)
        d1_df = get_daily_kline_data(trade_date=d1_date,   ts_code_list=stock_list)
        d2_df = get_daily_kline_data(trade_date=d2_date,   ts_code_list=stock_list)

        if d1_df.empty:
            logger.warning(f"[LabelEngine] {d1_date} 日线数据为空，跳过")
            return pd.DataFrame()

        d1_df["trade_date"] = d1_df["trade_date"].astype(str)
        if not d0_df.empty:
            d0_df["trade_date"] = d0_df["trade_date"].astype(str)
        if not d2_df.empty:
            d2_df["trade_date"] = d2_df["trade_date"].astype(str)

        # 构建快速查找
        d0_map = {row["ts_code"]: row for _, row in d0_df.iterrows()} if not d0_df.empty else {}
        d1_map = {row["ts_code"]: row for _, row in d1_df.iterrows()}
        d2_map = {row["ts_code"]: row for _, row in d2_df.iterrows()} if not d2_df.empty else {}

        rows = []
        for ts_code in stock_list:
            d1_row = d1_map.get(ts_code)
            if d1_row is None:
                # D+1 停牌，跳过（不作为负样本）
                continue

            d1_open  = float(d1_row.get("open",  0) or 0)
            d1_close = float(d1_row.get("close", 0) or 0)
            d1_high  = float(d1_row.get("high",  0) or 0)
            d1_low   = float(d1_row.get("low",   0) or 0)

            if d1_open <= 0:
                continue

            # ── D+1 日内收益率（核心基础量）──────────────────────────────────
            d1_intra_return = (d1_close - d1_open) / d1_open

            # ── 二分类标签 ────────────────────────────────────────────────────
            label1      = 1 if d1_intra_return >= 0.05 else 0
            label1_3pct = 1 if d1_intra_return >= 0.03 else 0
            label1_8pct = 1 if d1_intra_return >= 0.08 else 0

            # ── D+2 相关标签 ──────────────────────────────────────────────────
            d2_row = d2_map.get(ts_code)
            if d2_row is not None:
                d2_open    = float(d2_row.get("open",     0) or 0)
                d2_close   = float(d2_row.get("close",    0) or 0)
                d2_pct_chg = float(d2_row.get("pct_chg",  0) or 0)

                # label2：D+1 日内盈利 AND D+2 高开（值得隔夜持股的强势票）
                label2 = 1 if (d1_close > d1_open) and (d2_open > d1_close) else 0

                # label_d2_return：持有至 D+2 收盘的总收益
                label_d2_return = round((d2_close - d1_open) / d1_open, 6) if d2_close > 0 else None

                # label_d2_limit_down：D+2 跌停（主板 -10%，阈值 -9.5% 以包含精度误差）
                label_d2_limit_down = 1 if d2_pct_chg <= -9.5 else 0
            else:
                # D+2 无数据，涉及 D+2 的标签填 None
                label2              = None
                label_d2_return     = None
                label_d2_limit_down = None

            # ── 浮点标签（日内结构）─────────────────────────────────────────
            label_raw_return = round(d1_intra_return, 6)
            label_d1_high    = round((d1_high - d1_open) / d1_open, 6) if d1_high > 0 else None
            label_d1_low     = round((d1_low  - d1_open) / d1_open, 6) if d1_low  > 0 else None
            label_d1_pct_chg = float(d1_row.get("pct_chg", None) or None) if d1_row.get("pct_chg") is not None else None

            # ── 开盘溢价（需要 D close）──────────────────────────────────────
            d0_row   = d0_map.get(ts_code)
            d0_close = float(d0_row.get("close", 0) or 0) if d0_row else 0
            label_open_gap = round((d1_open - d0_close) / d0_close, 6) if d0_close > 0 else None

            rows.append({
                "stock_code":           ts_code,
                "trade_date":           trade_date,
                # 二分类
                "label1":               label1,
                "label2":               label2,
                "label1_3pct":          label1_3pct,
                "label1_8pct":          label1_8pct,
                "label_d2_limit_down":  label_d2_limit_down,
                # 浮点
                "label_raw_return":     label_raw_return,
                "label_open_gap":       label_open_gap,
                "label_d1_high":        label_d1_high,
                "label_d1_low":         label_d1_low,
                "label_d1_pct_chg":     label_d1_pct_chg,
                "label_d2_return":      label_d2_return,
            })

        result = pd.DataFrame(rows)
        if not result.empty:
            logger.info(
                f"[LabelEngine] {trade_date} 标签生成完成 | "
                f"样本数:{len(result)} | "
                f"label1正样本:{result['label1'].sum()} | "
                f"label1_3pct:{result['label1_3pct'].sum()} | "
                f"label1_8pct:{result['label1_8pct'].sum()}"
            )
        return result
