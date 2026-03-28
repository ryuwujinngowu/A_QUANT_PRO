"""
120日高位股涨幅周期因子
========================
输出列（D 日截面，全局因子，无 stock_code）：

  hp_cycle_height_pct   : 当前市场情绪强度（相对历史120日）
                           = 当前切片高位股均涨幅 / 120日内最高切片均涨幅
                           clip [0, 2.0]（> 1.0 表示当前超越近120日历史最高，市场亢奋）
                           0.0 = 高位股均无正收益（熊市）
                           0.5 = 当前处于历史中位
                           1.0 = 持平历史最高点
                           > 1.0 = 超越近期高点（历史新高或亢奋突破）

  hp_cycle_peak_dist_pct: 情绪峰值距今归一化距离
                           = 峰值切片索引 / max(1, 总切片数 - 1)
                           切片索引 0 = D0（当前），11 = 最早（D-110）
                           0.0 = 当前就是峰值（情绪正在高点）
                           1.0 = 峰值在120日前（情绪峰值已远，当前压抑）
                           与 hp_cycle_height_pct 联合解读：
                             height高 + dist低 → 市场亢奋上升（正在或刚过峰值）
                             height高 + dist高 → 与历史高点等高但已过峰（可能筑顶）
                             height低 + dist高 → 情绪压抑，高峰已在120天前

切片定义：
  以10个交易日为一个切片，从 D0 往前共12个切片（120个交易日）。
  切片0 = D0~D-9（当前），切片11 = D-110~D-119（最远）。
  每个切片取末日的"全市场前1%个股的平均20日涨幅"作为该切片的高度指标。

数据来源：
  hp_ext_cache["hp_cycle_slices"]：已在 data_bundle._load_hp_ext_cache() 通过 SQL 聚合计算完毕
  索引 0 = D0 切片的高度，索引 11 = D-110 切片的高度

无未来函数：
  每个切片的数据均为历史数据，切片0的 D0 收盘价为当日盘后已知数据。
"""
import numpy as np
import pandas as pd

from features.base_feature import BaseFeature
from features.feature_registry import feature_registry
from utils.log_utils import logger

_NEUTRAL = {
    "hp_cycle_height_pct":    0.5,   # 中性：处于历史中位
    "hp_cycle_peak_dist_pct": 0.5,   # 中性：峰值在历史中段
}


@feature_registry.register("hp_cycle")
class HPCycleFeature(BaseFeature):
    """120日高位股涨幅周期因子"""

    feature_name = "hp_cycle"

    factor_columns = list(_NEUTRAL.keys())

    def calculate(self, data_bundle) -> tuple:
        trade_date = data_bundle.trade_date
        hp_ext     = getattr(data_bundle, "hp_ext_cache", {})

        row = {"trade_date": trade_date, **_NEUTRAL}

        if not hp_ext:
            return pd.DataFrame([row]), {}

        slices = hp_ext.get("hp_cycle_slices", [])

        # 至少需要2个正值切片才能计算相对位置。
        # 全零（熊市）或仅1个正值时返回中性值(0.5, 0.5)，而非(0.0, 0.0)。
        valid_slices = [g for g in slices if isinstance(g, (int, float)) and g > 0]
        if len(valid_slices) < 2:
            logger.debug(f"[hp_cycle] {trade_date} 切片数据不足（有效={len(valid_slices)}），返回中性值")
            return pd.DataFrame([row]), {}

        # 当前切片（索引0 = D0）
        current_gain = slices[0] if slices and slices[0] is not None else 0.0

        # 120日内最高切片
        gains_arr = np.array([g if g is not None else 0.0 for g in slices])
        max_gain  = float(gains_arr.max())
        peak_idx  = int(np.argmax(gains_arr))

        # hp_cycle_height_pct：当前 / 最高，clip [0, 2.0]
        if max_gain > 0:
            height_pct = float(np.clip(current_gain / max_gain, 0.0, 2.0))
        else:
            height_pct = 0.0

        # hp_cycle_peak_dist_pct：峰值索引 / (总切片数 - 1)，0=当前是峰值，1=最远切片是峰值
        n_slices = len(gains_arr)
        peak_dist_pct = round(peak_idx / max(1, n_slices - 1), 4)

        row["hp_cycle_height_pct"]    = round(height_pct, 4)
        row["hp_cycle_peak_dist_pct"] = peak_dist_pct

        logger.debug(
            f"[hp_cycle] {trade_date} "
            f"current_gain:{current_gain:.1f}% max_gain:{max_gain:.1f}% "
            f"height_pct:{height_pct:.3f} peak_idx:{peak_idx} "
            f"peak_dist:{peak_dist_pct:.3f}"
        )
        return pd.DataFrame([row]), {}
