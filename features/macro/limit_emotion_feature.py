"""
涨停板情绪因子
==============
输出列（全部为 D 日截面，全局级因子，无 stock_code 列）：

  market_zhaban_count           : D0 炸板数（最高价触及涨停但收盘未封板，绝对量）
  market_zhaban_ratio           : D0 炸板数 / 近4日(D1~D4)均值，clip[0.1, 10.0]
                                  放量>1 说明炸板增多（多空分歧扩大），缩量<1 封板能力增强
  market_limit_touch_amount     : D0 涨停+炸板股首次触板分钟K线成交额之和（万亿元）
                                  衡量市场在涨停板博弈投入的真实资金量（绝对量信号）
  market_limit_touch_amount_ratio : D0 触板成交额 / 近4日(D1~D4)均值，clip[0.1, 10.0]
                                  趋势信号，分子分母口径完全一致（均为分钟线精确值）

三维度联动解读（配合 market_limit_up_5d_trend 使用）：
    · 涨停多 + 炸板少 + 触板成交高 → 资金积极封板，做多合力强
    · 涨停多 + 炸板多 + 触板成交高 → 多空分歧大，博弈激烈但封板意愿不足
    · 涨停少 + 炸板多 + 触板成交低 → 情绪收敛，空方主导

设计说明：
    - 全局级因子（无 stock_code），由 FeatureEngine 通过 left join 广播到所有个股行
    - 数据来源：macro_cache（由 FeatureDataBundle 预加载，本模块零 IO）
    - 分子分母均使用分钟线首次触板成交额，口径统一，无系统性偏差
    - 中性值：count=0 时 ratio=1.0（无数据不引入方向性偏差）
"""
import numpy as np
import pandas as pd

from features.base_feature import BaseFeature
from features.feature_registry import feature_registry
from utils.log_utils import logger


@feature_registry.register("limit_emotion")
class LimitEmotionFeature(BaseFeature):
    """涨停板情绪因子（炸板 + 触板成交额，全局因子）"""

    feature_name = "limit_emotion"

    factor_columns = [
        "market_zhaban_count",
        "market_zhaban_ratio",
        "market_limit_touch_amount",
        "market_limit_touch_amount_ratio",
    ]

    def calculate(self, data_bundle) -> tuple:
        """
        从 data_bundle.macro_cache 读取预加载数据，计算涨停板情绪因子。

        :return: (feature_df, {})
                 feature_df：单行 DataFrame，列 = trade_date + factor_columns
        """
        trade_date  = data_bundle.trade_date
        macro_cache = getattr(data_bundle, "macro_cache", {})
        lookback_5d = getattr(data_bundle, "lookback_dates_5d", [])

        row = {"trade_date": trade_date}

        # D1~D4（升序，最后一个=D0），用于历史均值分母
        hist_dates = lookback_5d[:-1] if len(lookback_5d) > 1 else []

        # ========== 炸板因子 ==========
        zhaban_counts_5d = macro_cache.get("zhaban_counts_5d", {})
        d0_zhaban = int(zhaban_counts_5d.get(trade_date, 0) or 0)
        row["market_zhaban_count"] = d0_zhaban

        if hist_dates and zhaban_counts_5d:
            hist_zhaban = [zhaban_counts_5d.get(d, 0) for d in hist_dates]
            avg_hist_zhaban = float(np.mean(hist_zhaban)) if hist_zhaban else 0.0
            if avg_hist_zhaban >= 1:
                row["market_zhaban_ratio"] = round(
                    float(np.clip(d0_zhaban / avg_hist_zhaban, 0.1, 10.0)), 3
                )
            else:
                row["market_zhaban_ratio"] = 1.0
        else:
            row["market_zhaban_ratio"] = 1.0

        # ========== 触板分钟成交额（分子分母口径一致：均为分钟线精确值）==========
        # 单位换算：千元 → 万亿元 = ÷ 1e9（与 market_amount_d0 保持一致）
        limit_touch_5d = macro_cache.get("limit_touch_amount_5d", {})
        d0_touch_raw = float(limit_touch_5d.get(trade_date, 0.0) or 0.0)
        row["market_limit_touch_amount"] = round(d0_touch_raw / 1e9, 6)

        if hist_dates and limit_touch_5d:
            hist_touch = [float(limit_touch_5d.get(d, 0.0) or 0.0) for d in hist_dates]
            avg_hist_touch = float(np.mean(hist_touch)) if hist_touch else 0.0
            if avg_hist_touch > 0:
                row["market_limit_touch_amount_ratio"] = round(
                    float(np.clip(d0_touch_raw / avg_hist_touch, 0.1, 10.0)), 3
                )
            else:
                row["market_limit_touch_amount_ratio"] = 1.0
        else:
            row["market_limit_touch_amount_ratio"] = 1.0

        feature_df = pd.DataFrame([row])
        logger.debug(
            f"[涨停板情绪] {trade_date} "
            f"炸板:{d0_zhaban}只 炸板比:{row['market_zhaban_ratio']:.3f} "
            f"触板成交:{row['market_limit_touch_amount']:.4f}万亿 "
            f"触板量比:{row['market_limit_touch_amount_ratio']:.3f}"
        )
        return feature_df, {}
