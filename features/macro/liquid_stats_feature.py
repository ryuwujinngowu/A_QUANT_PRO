"""
液态股广度因子（60日突破 + 高开低走）
======================================
输出列（D 日截面，全局因子，无 stock_code）：

  liquid_60d_breakout_ratio : 液态股（5日均成交额≥20亿）当日破60日新高或新低的占比
                               60日历史区间：[D-60, D-1]（不含 D0，避免自引用）
                               "破60日新高" = close_D0 ≥ max(close) in [D-60, D-1]
                               "破60日新低" = close_D0 ≤ min(close) in [D-60, D-1]
                               0.0 = 无液态股在历史边界（市场平静）
                               > 0.2 = 市场分化明显（强势股创新高 + 弱势股创新低）
                               归一化 [0, 1]

  liquid_holf_ratio          : 液态股当日高开低走的占比
                               "高开低走" = open > pre_close（高开）AND close < open（低走）
                               0.0 = 无高开低走（多头情绪正常）
                               > 0.3 = 大量冲高回落（做多情绪疲软，主力出货信号）
                               归一化 [0, 1]

定义：
  液态股 = 过去5个交易日（D-4 ~ D0）日均成交额 ≥ 20亿元（= 2,000,000 千元）
  pre_close = D0 的前收盘价（= D-1 收盘价），来自 kline_day.pre_close 列

数据来源：
  hp_ext_cache["liquid_stats"]：由 data_bundle._load_hp_ext_cache() 通过 SQL 聚合计算完毕

无未来函数：
  - D0 的 open/close/pre_close 均为 D0 盘后已知数据
  - 60日历史区间明确排除 D0（使用 [D-60, D-1]），无自引用
"""
import pandas as pd

from features.base_feature import BaseFeature
from features.feature_registry import feature_registry
from utils.log_utils import logger

_NEUTRAL = {
    "liquid_60d_breakout_ratio": 0.0,
    "liquid_holf_ratio":         0.0,
}


@feature_registry.register("liquid_stats")
class LiquidStatsFeature(BaseFeature):
    """液态股广度因子（60日突破 + 高开低走）"""

    feature_name = "liquid_stats"

    factor_columns = list(_NEUTRAL.keys())

    def calculate(self, data_bundle) -> tuple:
        trade_date = data_bundle.trade_date
        hp_ext     = getattr(data_bundle, "hp_ext_cache", {})

        row = {"trade_date": trade_date, **_NEUTRAL}

        if not hp_ext:
            return pd.DataFrame([row]), {}

        liquid = hp_ext.get("liquid_stats", {})
        if not liquid:
            logger.debug(f"[liquid_stats] {trade_date} 无液态股统计数据，返回中性值")
            return pd.DataFrame([row]), {}

        row["liquid_60d_breakout_ratio"] = float(
            liquid.get("liquid_60d_breakout_ratio", 0.0)
        )
        row["liquid_holf_ratio"] = float(
            liquid.get("liquid_holf_ratio", 0.0)
        )

        logger.debug(
            f"[liquid_stats] {trade_date} "
            f"liquid_total:{liquid.get('liquid_total', 0)} "
            f"breakout:{row['liquid_60d_breakout_ratio']:.3f} "
            f"holf:{row['liquid_holf_ratio']:.3f}"
        )
        return pd.DataFrame([row]), {}
