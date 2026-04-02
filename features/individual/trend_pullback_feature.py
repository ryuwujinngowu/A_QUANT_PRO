"""
趋势股回调位置因子（trend_pullback）
=====================================
输出列（个股级因子，含 stock_code + trade_date）：

  pullback_days_from_high60  : D0 距60日内最高点（最高价 high）的交易日数；0=今日是高点，60=高点在最早一天
  pullback_pct_from_high60   : D0 收盘相对60日最高点的涨跌幅（%），≤0，clip[-50, 0]
  days_since_max_yang15      : D0 距15个交易日内最大阳线（pct_chg最高的正收益日）的交易日数；0=今日是最大阳线
  max_yang15_pct             : 15个交易日内最大阳线的pct_chg（%），≥0，clip[0, 30]；无阳线=0
  pullback_time_ratio        : pullback_days_from_high60 / 60，∈[0,1]，归一化调整时长（跨股可比）
  pullback_depth_ratio       : |pullback_pct| / max(0.01, max_yang15_pct)，调整幅度/最大阳线幅度；
                               衡量已吐回最大阳线涨幅的比例，clip[0, 10]

设计说明（XGBoost学习调整时间/幅度规律）：
  - 4个原始因子：让模型学习绝对量（调整几天、跌了多少、阳线多强、阳线多久前）
  - 2个派生比率：归一化跨股比较 + 调整深度与发动力量的相对关系
  - 组合使用：model可以学到"调整3-8天 + 跌幅5-10% + 大阳线在10天前 + 吐回50%"之类的pattern

数据来源：data_bundle.daily_grouped（不复权日线，60日窗口，零IO）
注意：60日窗口需 data_bundle.lookback_dates_60d 支持（FeatureDataBundle已扩展）
"""
import pandas as pd
import numpy as np

from features.base_feature import BaseFeature
from features.feature_registry import feature_registry
from utils.log_utils import logger


@feature_registry.register("trend_pullback")
class TrendPullbackFeature(BaseFeature):
    """趋势股回调位置因子（60日高点调整 + 15日最大阳线，个股级）"""

    feature_name = "trend_pullback"

    factor_columns = [
        "pullback_days_from_high60",
        "pullback_pct_from_high60",
        "days_since_max_yang15",
        "max_yang15_pct",
        "pullback_time_ratio",
        "pullback_depth_ratio",
    ]

    # 中性值：数据不足时填充，不引入方向性偏差
    _NEUTRAL = {
        "pullback_days_from_high60": 0,
        "pullback_pct_from_high60":  0.0,
        "days_since_max_yang15":     0,
        "max_yang15_pct":            0.0,
        "pullback_time_ratio":       0.0,
        "pullback_depth_ratio":      0.0,
    }

    def calculate(self, data_bundle) -> tuple:
        trade_date    = data_bundle.trade_date
        daily_grouped = data_bundle.daily_grouped
        lookback_60d  = getattr(data_bundle, "lookback_dates_60d", [])
        target_codes  = getattr(data_bundle, "target_ts_codes", [])

        if not lookback_60d or not target_codes:
            return pd.DataFrame(), {}

        rows = []
        for ts_code in target_codes:
            row = {"stock_code": ts_code, "trade_date": trade_date}

            # ── 按时间升序取60日日线（最后一个=D0）──────────────────────────
            daily_data = []
            for date in lookback_60d:
                d = daily_grouped.get((ts_code, date))
                if d:
                    daily_data.append({
                        "high":    float(d.get("high",    0) or 0),
                        "close":   float(d.get("close",   0) or 0),
                        "pct_chg": float(d.get("pct_chg", 0) or 0),
                    })

            if len(daily_data) < 2:
                row.update(self._NEUTRAL)
                rows.append(row)
                continue

            n       = len(daily_data)
            d0_close = daily_data[-1]["close"]

            # ── Factor 1&2：60日最高价高点 ──────────────────────────────────
            max_high     = 0.0
            max_high_pos = 0    # 在 daily_data 中的索引（0=最旧）
            for i, d in enumerate(daily_data):
                if d["high"] > max_high:
                    max_high     = d["high"]
                    max_high_pos = i

            if max_high > 0 and d0_close > 0:
                # pullback_days：高点到D0的交易日数（高点在D0则为0）
                pullback_days = n - 1 - max_high_pos
                pullback_pct  = float(np.clip(
                    (d0_close - max_high) / max_high * 100, -50.0, 0.0
                ))
            else:
                pullback_days = 0
                pullback_pct  = 0.0

            row["pullback_days_from_high60"] = pullback_days
            row["pullback_pct_from_high60"]  = round(pullback_pct, 4)

            # ── Factor 3&4：15日内最大阳线 ──────────────────────────────────
            # 取最近15个有效日（D-14~D0）；pct_chg>0 为阳线
            window_15 = daily_data[-15:]
            max_yang_pct   = 0.0
            max_yang_pos   = -1   # 在 window_15 中的索引
            for i, d in enumerate(window_15):
                if d["pct_chg"] > max_yang_pct:
                    max_yang_pct = d["pct_chg"]
                    max_yang_pos = i

            if max_yang_pos >= 0:
                days_since_yang = len(window_15) - 1 - max_yang_pos
                max_yang_pct    = float(np.clip(max_yang_pct, 0.0, 30.0))
            else:
                # 15日内无阳线：中性值
                days_since_yang = 0
                max_yang_pct    = 0.0

            row["days_since_max_yang15"] = days_since_yang
            row["max_yang15_pct"]        = round(max_yang_pct, 4)

            # ── 派生因子 ──────────────────────────────────────────────────────
            row["pullback_time_ratio"] = round(pullback_days / 60.0, 4)

            if max_yang_pct > 0:
                depth = abs(pullback_pct) / max(0.01, max_yang_pct)
                row["pullback_depth_ratio"] = round(float(np.clip(depth, 0.0, 10.0)), 4)
            else:
                row["pullback_depth_ratio"] = 0.0

            rows.append(row)

        if not rows:
            return pd.DataFrame(), {}

        feature_df = pd.DataFrame(rows)
        logger.debug(
            f"[趋势回调] {trade_date} 计算完成 | 股票数:{len(rows)}"
        )
        return feature_df, {}
