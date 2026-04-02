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

    # 🛑【修复1】无偏中性值：数据不足时填中位数，不误导XGBoost
    _NEUTRAL = {
        "pullback_days_from_high60": 30,    # 60日中位数，非0（非假高点）
        "pullback_pct_from_high60":  0.0,
        "days_since_max_yang15":     7,     # 15日中位数，非0
        "max_yang15_pct":            0.0,
        "pullback_time_ratio":       0.5,   # 归一化中位数
        "pullback_depth_ratio":      0.0,
    }
    # 固定窗口常量
    MAX_LOOKBACK_HIGH = 60
    MAX_LOOKBACK_YANG = 15
    YANG_THRESHOLD = 0.01  # 🛑【修复2】阳线阈值：过滤平盘/微小涨幅

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

            # 🛑【修复3】固定60交易日窗口，补全缺失数据，保证天数计算准确
            daily_data = []
            for date in lookback_60d:
                d = daily_grouped.get((ts_code, date))
                if d:
                    daily_data.append({
                        "high":    float(d.get("high", 0) or 0),
                        "close":   float(d.get("close", 0) or 0),
                        "pct_chg": float(d.get("pct_chg", 0) or 0),
                    })
                else:
                    # 缺失数据填充空值，保留窗口长度
                    daily_data.append({"high": 0, "close": 0, "pct_chg": 0})

            # 数据不足直接填充中性值
            if len([x for x in daily_data if x["high"] > 0]) < 2:
                row.update(self._NEUTRAL)
                rows.append(row)
                continue

            d0_close = daily_data[-1]["close"]

            # ── Factor 1&2：60日最高价高点（固定60日窗口）──────────────────
            max_high = 0.0
            max_high_pos = 0
            for i, d in enumerate(daily_data):
                if d["high"] > max_high and d["high"] > 0:
                    max_high = d["high"]
                    max_high_pos = i

            if max_high > 0 and d0_close > 0:
                # 🛑【修复4】固定60日计算天数，100%准确
                pullback_days = self.MAX_LOOKBACK_HIGH - 1 - max_high_pos
                pullback_days = np.clip(pullback_days, 0, self.MAX_LOOKBACK_HIGH)
                pullback_pct = float(np.clip((d0_close - max_high) / max_high * 100, -50.0, 0.0))
            else:
                pullback_days = self._NEUTRAL["pullback_days_from_high60"]
                pullback_pct = self._NEUTRAL["pullback_pct_from_high60"]

            row["pullback_days_from_high60"] = pullback_days
            row["pullback_pct_from_high60"]  = round(pullback_pct, 4)

            # ── Factor 3&4：15日内最大阳线（严格筛选阳线）──────────────────
            window_15 = daily_data[-self.MAX_LOOKBACK_YANG:]
            max_yang_pct = 0.0
            max_yang_pos = -1

            for i, d in enumerate(window_15):
                # 🛑【修复5】仅统计真正阳线（涨幅>0.01），排除平盘
                if d["pct_chg"] > self.YANG_THRESHOLD and d["pct_chg"] > max_yang_pct:
                    max_yang_pct = d["pct_chg"]
                    max_yang_pos = i

            if max_yang_pos >= 0:
                days_since_yang = self.MAX_LOOKBACK_YANG - 1 - max_yang_pos
                max_yang_pct = float(np.clip(max_yang_pct, 0.0, 30.0))
            else:
                # 无阳线：中性值
                days_since_yang = self._NEUTRAL["days_since_max_yang15"]
                max_yang_pct = self._NEUTRAL["max_yang15_pct"]

            row["days_since_max_yang15"] = days_since_yang
            row["max_yang15_pct"]        = round(max_yang_pct, 4)

            # ── 派生因子（口径统一）──────────────────────────────────────
            row["pullback_time_ratio"] = round(pullback_days / self.MAX_LOOKBACK_HIGH, 4)

            if max_yang_pct > 0:
                depth = abs(pullback_pct) / max(0.01, max_yang_pct)
                row["pullback_depth_ratio"] = round(float(np.clip(depth, 0.0, 10.0)), 4)
            else:
                row["pullback_depth_ratio"] = self._NEUTRAL["pullback_depth_ratio"]

            rows.append(row)

        feature_df = pd.DataFrame(rows) if rows else pd.DataFrame()
        logger.debug(f"[趋势回调] {trade_date} 计算完成 | 股票数:{len(rows)}")
        return feature_df, {}