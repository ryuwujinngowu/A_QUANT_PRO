"""
市场行情高/宽风格因子 + 市场高度历史水分位
==========================================
输出列（D 日截面，全局因子，无 stock_code）：

  hp_style_breadth_ratio : 市场行情宽度比
                            = count(10日涨幅 > 100%) / max(1, count(10日涨幅 > 50%))
                            值域 [0, 1]
                            接近 1 = 高度集中（少数超强股，其余普通），"高"风格
                            接近 0 = 宽度结构（普涨但无超高倍股），"宽"风格
                            0 = 市场无 50% 以上涨幅股票（极度低迷）

  hp_style_height_pct    : 市场高度历史水分位
                            = min(count(10日涨幅 > 100%) / 20.0, 1.0)
                            历史经验基准：20只 = 市场处于历史极热状态（满分 1.0）
                            0   = 无超高倍股（冷淡市）
                            0.5 = 约10只（情绪中等偏热）
                            1.0 = ≥20只（市场亢奋）

过滤条件：
  - 剔除 ST / *ST 股票
  - 剔除北交所（.BJ）
  - 剔除近 10 个交易日内上市的新股（list_date ≥ D-10 日期）

数据来源：
  - hp_ext_cache["market_all_d0"]   : D0 全市场日线（close）
  - hp_ext_cache["market_all_d10"]  : D-10 全市场日线（close，10日涨幅基准）
  - hp_ext_cache["st_set"]          : ST 集合
  - hp_ext_cache["list_date_map"]   : 上市日期映射

无未来函数：10日涨幅用 D-10 → D0 收盘价，均为 D0 盘后已知数据。
"""
import numpy as np
import pandas as pd

from features.base_feature import BaseFeature
from features.feature_registry import feature_registry
from utils.log_utils import logger

# 市场高度历史经验基准：20只10日涨幅>100%的股票 = 市场极热
_HEIGHT_BENCHMARK = 20

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
        st_set  = hp_ext.get("st_set",         set())
        ldmap   = hp_ext.get("list_date_map",  {})
        kd      = hp_ext.get("key_dates",      {})

        if d0_df.empty or d10_df.empty:
            logger.debug(f"[hp_style] {trade_date} 数据不足，返回中性值")
            return pd.DataFrame([row]), {}

        # D-10 日期的 YYYYMMDD 格式（用于新股 IPO 过滤：列入d10之后上市的新股剔除）
        d10_date_str = kd.get("d10", "").replace("-", "")

        d10_close: dict = dict(
            zip(d10_df["ts_code"], d10_df["close"].astype(float))
        )

        cnt_100 = 0    # 10日涨幅 > 100%
        cnt_50  = 0    # 10日涨幅 > 50%

        for _, r in d0_df.iterrows():
            code = str(r["ts_code"])

            # ── 过滤 ──────────────────────────────────────────────────────
            if code in st_set or code.endswith(".BJ"):
                continue
            # 近10个交易日内上市的新股剔除（list_date >= D-10 = ipo_cutoff）
            ld = ldmap.get(code, "")
            if ld and d10_date_str and ld >= d10_date_str:
                continue

            c0  = float(r.get("close", 0) or 0)
            c10 = d10_close.get(code, 0.0)
            if c0 <= 0 or c10 <= 0:
                continue

            gain_10d = c0 / c10 - 1

            if gain_10d > 0.5:    # > 50%
                cnt_50 += 1
            if gain_10d > 1.0:    # > 100%
                cnt_100 += 1

        # 市场宽度比
        row["hp_style_breadth_ratio"] = round(cnt_100 / max(1, cnt_50), 4)
        # 市场高度水分位（硬基准 20 只 = 满分）
        row["hp_style_height_pct"]    = round(min(cnt_100 / _HEIGHT_BENCHMARK, 1.0), 4)

        logger.debug(
            f"[hp_style] {trade_date} "
            f"cnt_100:{cnt_100} cnt_50:{cnt_50} "
            f"breadth:{row['hp_style_breadth_ratio']:.3f} "
            f"height_pct:{row['hp_style_height_pct']:.3f}"
        )
        return pd.DataFrame([row]), {}
