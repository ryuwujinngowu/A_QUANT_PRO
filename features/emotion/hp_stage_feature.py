"""
高位股阶段因子
==============
输出列（全部为 D 日截面，全局因子，无 stock_code）：

  hp_stage_vwap_bias   : 高位股成交额加权 MA5 乖离率（%）
                          MA5 = 近5个交易日（D-4~D0）均价
                          乖离率 = (close_D0 - MA5) / MA5 × 100
                          成交额加权：成交额越大的高位股权重越大
                          > 0 = 股价整体偏离均线上方（亢奋）
                          < 0 = 股价整体跌破均线（回调）
                          绝对值越大 = 乖离越极端（天量暴涨 or 深度调整）

  hp_stage_pct_chg     : 高位股成交额加权 D0 涨跌幅（%）
                          = weighted_avg(pct_chg_i × amount_i) / sum(amount_i)
                          正值 = 高位股今日整体上涨（情绪延续）
                          负值 = 高位股今日整体回调（情绪降温）

  hp_stage_amount_ratio: 高位股成交额加权的量能倍数
                          = D0 成交额 / 近4日（D-4~D-1）均成交额
                          > 1 = 放量（资金涌入）
                          < 1 = 缩量（资金撤退）
                          天量连板时该值极大（如 3~10）

高位股定义：
  全市场不复权20日涨幅前1%（基础池）→ 基础池前10%（高位股），最少3只
  过滤：ST / BJ / 近21交易日新股

注意：本因子只用 D 日及之前数据，无未来函数。
"""
from typing import List
import numpy as np
import pandas as pd

from features.base_feature import BaseFeature
from features.feature_registry import feature_registry
from utils.log_utils import logger

_NEUTRAL = {
    "hp_stage_vwap_bias":    0.0,
    "hp_stage_pct_chg":      0.0,
    "hp_stage_amount_ratio": 1.0,
}


@feature_registry.register("hp_stage")
class HPStageFeature(BaseFeature):
    """高位股阶段因子（成交额加权 MA5 乖离率 + 涨跌幅 + 量能倍数）"""

    feature_name = "hp_stage"

    factor_columns = list(_NEUTRAL.keys())

    def calculate(self, data_bundle) -> tuple:
        trade_date = data_bundle.trade_date
        hp_ext     = getattr(data_bundle, "hp_ext_cache", {})

        row = {"trade_date": trade_date, **_NEUTRAL}

        if not hp_ext:
            return pd.DataFrame([row]), {}

        high_pos_df = hp_ext.get("hp_high_pos", pd.DataFrame())
        recent5d_df = hp_ext.get("hp_base_pool_recent5d", pd.DataFrame())
        kd          = hp_ext.get("key_dates", {})

        if high_pos_df.empty or recent5d_df.empty:
            logger.debug(f"[hp_stage] {trade_date} 无高位股或近5日数据，返回中性值")
            return pd.DataFrame([row]), {}

        # ── 过滤：仅保留高位股，基础池近5日数据中取对应行 ───────────────────
        hp_codes    = set(high_pos_df["ts_code"].tolist())
        d0_date_str = kd.get("d0", "").replace("-", "")   # YYYYMMDD，用于显式日期对齐

        # 统一 trade_date 格式为 YYYYMMDD，便于与 d0_date_str 直接比较
        recent5d_df = recent5d_df.copy()
        recent5d_df["trade_date"] = recent5d_df["trade_date"].astype(str).str.replace("-", "")

        # 按股票分组，计算各高位股的 MA5 乖离率 + 涨跌幅 + 量能比
        biases:       List[float] = []
        pct_chgs:     List[float] = []
        amount_ratios:List[float] = []
        weights:      List[float] = []

        for code in hp_codes:
            stock_rows = recent5d_df[recent5d_df["ts_code"] == code].sort_values("trade_date")

            # 显式按日期定位 D0 行，避免停牌缺行时 closes[-1] 取到错误日期
            d0_row = stock_rows[stock_rows["trade_date"] == d0_date_str]
            if d0_row.empty:
                continue   # D0 无数据（可能停牌），跳过

            c_d0   = float(d0_row["close"].iloc[0])
            amt_d0 = float(d0_row["amount"].iloc[0]) if "amount" in d0_row.columns else 0.0

            # D0 之前的行（D-4 ~ D-1，停牌时可能少于4行）
            prior_rows = stock_rows[stock_rows["trade_date"] < d0_date_str]

            # MA5 乖离率：全部可用收盘价（含D0）的简单均值
            all_closes = prior_rows["close"].astype(float).tolist() + [c_d0]
            ma5  = float(np.mean(all_closes))
            bias = float(np.clip((c_d0 - ma5) / ma5 * 100, -30.0, 30.0)) if ma5 > 0 else 0.0

            # D0 涨跌幅 = (D0_close / 最近一个前收日 close - 1) × 100
            # 显式取 prior_rows 最后一行（日期最近的非D0日），避免停牌跳行
            if not prior_rows.empty:
                c_d1    = float(prior_rows["close"].iloc[-1])
                pct_chg = (c_d0 / c_d1 - 1) * 100 if c_d1 > 0 else 0.0
            else:
                pct_chg = 0.0

            # 量能倍数 = D0 成交额 / 近4日（D-4~D-1）均成交额
            hist_amt = prior_rows["amount"].astype(float).tolist() if "amount" in prior_rows.columns else []
            avg_hist = float(np.mean(hist_amt)) if hist_amt else 0.0
            amt_r    = amt_d0 / avg_hist if avg_hist > 0 else 1.0

            # 权重 = D0 成交额（成交额越大，权重越大）
            w = amt_d0 if amt_d0 > 0 else 1.0

            biases.append(bias)
            pct_chgs.append(pct_chg)
            amount_ratios.append(float(np.clip(amt_r, 0.01, 20.0)))
            weights.append(w)

        total_w = sum(weights)
        if total_w > 0 and biases:
            row["hp_stage_vwap_bias"]    = round(float(np.dot(biases,        weights) / total_w), 3)
            row["hp_stage_pct_chg"]      = round(float(np.dot(pct_chgs,      weights) / total_w), 3)
            row["hp_stage_amount_ratio"] = round(float(np.dot(amount_ratios, weights) / total_w), 3)

        logger.debug(
            f"[hp_stage] {trade_date} 高位股:{len(hp_codes)}只 "
            f"vwap_bias:{row['hp_stage_vwap_bias']:.2f}% "
            f"pct_chg:{row['hp_stage_pct_chg']:.2f}% "
            f"amount_ratio:{row['hp_stage_amount_ratio']:.2f}"
        )
        return pd.DataFrame([row]), {}
