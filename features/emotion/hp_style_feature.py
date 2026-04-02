"""
市场行情高/宽风格因子 + 市场高度历史水分位
==========================================
输出列（D 日截面，全局因子，无 stock_code）：

  hp_style_breadth_ratio : 市场行情宽度比
                           = count(中位股候选) / max(1, count(高位股))
                           其中：
                             高位股 = 全市场20日涨幅前1%基础池中的前10%（最少3只）
                             中位股候选 = 基础池前50%剔除高位股后，
                                        且10日涨幅 > max(10%, 高位股10日平均涨幅×40%)
                           值越高 = 从极少数龙头向更大范围扩散，偏“宽”风格
                           值越低 = 仅少数龙头独强，偏“高”风格

  hp_style_height_pct    : 市场高度历史水分位
                           = min(count(高位股) / 20.0, 1.0)
                           历史经验基准：20只 = 市场处于历史极热状态（满分 1.0）
                           但由于高位股本身定义来自基础池前10%，该值更多表达
                           “当前高位龙头层是否达到足够厚度”

过滤条件：
  - 剔除 ST / *ST 股票
  - 剔除北交所（.BJ）
  - 剔除近 21 个交易日内上市的新股（与高位股定义对齐）

数据来源：
  - hp_ext_cache["market_all_d0"]   : D0 全市场日线（close / amount）
  - hp_ext_cache["market_all_d10"]  : D-10 全市场日线（close）
  - hp_ext_cache["market_all_d21"]  : D-21 全市场日线（close，用于高位股基础池）
  - hp_ext_cache["st_set"]          : ST 集合
  - hp_ext_cache["list_date_map"]   : 上市日期映射

无未来函数：10日涨幅与20日涨幅均只用 D 日及之前已知收盘价。
"""
import pandas as pd

from features.base_feature import BaseFeature
from features.feature_registry import feature_registry
from features.utils.high_position_utils import compute_high_pos_selection
from utils.log_utils import logger

# 市场高度历史经验基准：20只高位龙头 = 市场极热
_HEIGHT_BENCHMARK = 20
_MID_PCT_OF_BASE = 0.50
_BURST_RATIO = 0.40
_MIN_MID_GAIN_10D = 0.10

_NEUTRAL = {
    "hp_style_breadth_ratio": 0.0,
    "hp_style_height_pct":    0.0,
}


@feature_registry.register("hp_style")
class HPStyleFeature(BaseFeature):
    """市场行情高/宽风格 + 历史高度水分位"""

    feature_name = "hp_style"

    factor_columns = list(_NEUTRAL.keys())

    def calculate(self, data_bundle) -> tuple:
        trade_date = data_bundle.trade_date
        hp_ext     = getattr(data_bundle, "hp_ext_cache", {})

        row = {"trade_date": trade_date, **_NEUTRAL}

        if not hp_ext:
            return pd.DataFrame([row]), {}

        d0_df   = hp_ext.get("market_all_d0",  pd.DataFrame())
        d10_df  = hp_ext.get("market_all_d10", pd.DataFrame())
        d21_df  = hp_ext.get("market_all_d21", pd.DataFrame())
        st_set  = hp_ext.get("st_set",         set())
        ldmap   = hp_ext.get("list_date_map",  {})
        kd      = hp_ext.get("key_dates",      {})

        if d0_df.empty or d10_df.empty or d21_df.empty:
            logger.debug(f"[hp_style] {trade_date} 数据不足，返回中性值")
            return pd.DataFrame([row]), {}

        d21_date_str = kd.get("d21", "").replace("-", "")
        base_pool_df, high_pos_df = compute_high_pos_selection(
            d0_df=d0_df,
            d21_df=d21_df,
            st_set=st_set,
            list_date_map=ldmap,
            d21_date_str=d21_date_str,
        )
        if base_pool_df.empty or high_pos_df.empty:
            logger.debug(f"[hp_style] {trade_date} 高位股基础池为空，返回中性值")
            return pd.DataFrame([row]), {}

        d10_close = dict(zip(d10_df["ts_code"], pd.to_numeric(d10_df["close"], errors="coerce").fillna(0.0)))
        d0_close = dict(zip(d0_df["ts_code"], pd.to_numeric(d0_df["close"], errors="coerce").fillna(0.0)))

        high_codes = set(high_pos_df["ts_code"].astype(str).tolist())
        n_high = len(high_codes)

        high_gains_10d = []
        for code in high_codes:
            c0 = float(d0_close.get(code, 0.0) or 0.0)
            c10 = float(d10_close.get(code, 0.0) or 0.0)
            if c0 > 0 and c10 > 0:
                high_gains_10d.append(c0 / c10 - 1)
        high_avg_gain_10d = float(sum(high_gains_10d) / len(high_gains_10d)) if high_gains_10d else 0.0
        burst_threshold = max(_MIN_MID_GAIN_10D, high_avg_gain_10d * _BURST_RATIO)

        mid_range_n = max(n_high, int(len(base_pool_df) * _MID_PCT_OF_BASE))
        mid_candidates_df = base_pool_df.head(mid_range_n).copy()
        mid_candidates_df = mid_candidates_df[~mid_candidates_df["ts_code"].astype(str).isin(high_codes)].copy()

        n_mid = 0
        for code in mid_candidates_df["ts_code"].astype(str).tolist():
            c0 = float(d0_close.get(code, 0.0) or 0.0)
            c10 = float(d10_close.get(code, 0.0) or 0.0)
            if c0 <= 0 or c10 <= 0:
                continue
            gain_10d = c0 / c10 - 1
            if gain_10d > burst_threshold:
                n_mid += 1

        row["hp_style_breadth_ratio"] = round(n_mid / max(1, n_high), 4)
        row["hp_style_height_pct"]    = round(min(n_high / _HEIGHT_BENCHMARK, 1.0), 4)

        logger.debug(
            f"[hp_style] {trade_date} high:{n_high} mid:{n_mid} "
            f"high_avg_10d:{high_avg_gain_10d:.3f} threshold:{burst_threshold:.3f} "
            f"breadth:{row['hp_style_breadth_ratio']:.3f} "
            f"height_pct:{row['hp_style_height_pct']:.3f}"
        )
        return pd.DataFrame([row]), {}
