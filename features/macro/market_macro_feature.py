"""
当日市场宏观因子
================
输出列（全部为 D 日截面，全局级因子，无 stock_code 列）：

【涨跌停维度】
  market_limit_up_count     : D 日涨停池股票数量
  market_limit_down_count   : D 日跌停池股票数量

【连板维度】
  market_max_consec_num     : D 日最高连板数（市场高度）
  market_consec_2plus_count : D 日 2 板及以上股票数量（连板梯队宽度）

【最强板块维度】
  market_top_cpt_up_nums    : D 日最强板块涨停家数（榜首）
  market_top_cpt_cons_nums  : D 日最强板块连板家数（榜首）

【指数维度】
  index_sh_pct_chg          : D 日上证指数涨跌幅
  index_sz_pct_chg          : D 日深证成指涨跌幅
  index_cyb_pct_chg         : D 日创业板指涨跌幅

【全市场成交量比率（窗口内归一化）】
  market_vol_ratio_d{0-4}   : vol_di / mean(vol_d0..d4)。放量>1，缩量<1，无数据=1.0
                              5日窗口内归一化，消除绝对额的跨日期差异

【派生趋势因子】
  market_limit_up_rate      : D 日涨停数 / 全市场总股数（约5200）≈ 涨停参与率
  market_limit_up_5d_trend  : D 日涨停数 / 近4日均值（有效数据≥1时计算，else=1.0），
                              clip [0.1, 10.0] 防止历史数据缺失导致极值
  market_consec_5d_trend    : D 日最高连板数 / 近4日均值（同上），clip [0.1, 10.0]

设计说明：
    - 本模块输出全局级（无 stock_code），由 FeatureEngine 通过 left join 广播到所有个股行
    - 数据来源：limit_list_ths / limit_step / limit_cpt_list / index_daily 四张表
    - 依赖 data_bundle.macro_cache（由 FeatureDataBundle 在初始化时预加载）
    - 后续可在此文件中继续新增其他宏观维度因子（如融资融券余额、北向资金等）
"""
from typing import Dict
import numpy as np
import pandas as pd

from features.base_feature import BaseFeature
from features.feature_registry import feature_registry
from utils.log_utils import logger


# A 股上市股票总数近似值（用于涨停参与率计算）
TOTAL_LISTED_APPROX = 5200

# 关注的核心指数代码
INDEX_CODES = {
    "000001.SH": "index_sh_pct_chg",    # 上证指数
    "399001.SZ": "index_sz_pct_chg",    # 深证成指
    "399006.SZ": "index_cyb_pct_chg",   # 创业板指
}


@feature_registry.register("market_macro")
class MarketMacroFeature(BaseFeature):
    """当日市场宏观因子"""

    feature_name = "market_macro"

    factor_columns = [
        # 涨跌停
        "market_limit_up_count", "market_limit_down_count",
        # 连板
        "market_max_consec_num", "market_consec_2plus_count",
        # 最强板块
        "market_top_cpt_up_nums", "market_top_cpt_cons_nums",
        # 指数
        "index_sh_pct_chg", "index_sz_pct_chg", "index_cyb_pct_chg",
        # 全市场成交量比率（窗口内归一化，d0=当日，d1~d4=近4个交易日）
        "market_vol_ratio_d0", "market_vol_ratio_d1", "market_vol_ratio_d2",
        "market_vol_ratio_d3", "market_vol_ratio_d4",
        # 派生趋势因子
        "market_limit_up_rate", "market_limit_up_5d_trend", "market_consec_5d_trend",
    ]

    def calculate(self, data_bundle) -> tuple:
        """
        从 data_bundle.macro_cache 读取预加载数据，计算宏观因子

        :return: (feature_df, factor_dict)
                 feature_df：单行 DataFrame，列 = trade_date + factor_columns
                 factor_dict：兼容接口，空 dict
        """
        trade_date  = data_bundle.trade_date
        macro_cache = getattr(data_bundle, "macro_cache", {})

        row = {"trade_date": trade_date}

        # ========== 涨跌停维度 ==========
        limit_up_df   = macro_cache.get("limit_up_df",   pd.DataFrame())
        limit_down_df = macro_cache.get("limit_down_df", pd.DataFrame())
        row["market_limit_up_count"]   = len(limit_up_df)
        row["market_limit_down_count"] = len(limit_down_df)

        # ========== 连板维度 ==========
        limit_step_df = macro_cache.get("limit_step_df", pd.DataFrame())
        if not limit_step_df.empty and "nums" in limit_step_df.columns:
            nums_series = pd.to_numeric(limit_step_df["nums"], errors="coerce").dropna()
            row["market_max_consec_num"]     = int(nums_series.max()) if len(nums_series) > 0 else 0
            row["market_consec_2plus_count"] = int((nums_series >= 2).sum())
        else:
            row["market_max_consec_num"]     = 0
            row["market_consec_2plus_count"] = 0

        # ========== 最强板块维度 ==========
        limit_cpt_df = macro_cache.get("limit_cpt_df", pd.DataFrame())
        if not limit_cpt_df.empty:
            # 取排名第一的板块
            top_row = limit_cpt_df.iloc[0]
            row["market_top_cpt_up_nums"]   = int(top_row.get("up_nums",   0) or 0)
            row["market_top_cpt_cons_nums"] = int(top_row.get("cons_nums", 0) or 0)
        else:
            row["market_top_cpt_up_nums"]   = 0
            row["market_top_cpt_cons_nums"] = 0

        # ========== 指数维度 ==========
        index_df = macro_cache.get("index_df", pd.DataFrame())
        if not index_df.empty and "ts_code" in index_df.columns:
            idx_map = {r["ts_code"]: r.get("pct_chg", 0) for _, r in index_df.iterrows()}
        else:
            idx_map = {}

        for ts_code, col_name in INDEX_CODES.items():
            row[col_name] = float(idx_map.get(ts_code, 0) or 0)

        # ========== 全市场成交量（窗口内归一化比率）==========
        # market_vol_ratio_d{i} = vol_di / mean(vol_d0..d4)
        # 与 stock_amount_5d_ratio 设计对称，消除绝对额跨日期差异
        market_vol_df = macro_cache.get("market_vol_df", pd.DataFrame())
        lookback_5d   = getattr(data_bundle, "lookback_dates_5d", [])
        if not market_vol_df.empty and "trade_date" in market_vol_df.columns and lookback_5d:
            vol_map = {
                str(r["trade_date"]).replace("-", ""): float(r.get("market_total_vol", 0) or 0)
                for _, r in market_vol_df.iterrows()
            }
            # 计算5日均值（含d0）
            all_vols = [vol_map.get(d.replace("-", ""), 0) for d in lookback_5d]
            avg_vol  = float(np.mean(all_vols)) if any(v > 0 for v in all_vols) else 0.0
            # lookback_5d 升序，最后一个=d0，往前=d1..d4
            for di in range(5):
                idx = len(lookback_5d) - 1 - di
                if 0 <= idx < len(lookback_5d):
                    vol_val = vol_map.get(lookback_5d[idx].replace("-", ""), 0)
                    row[f"market_vol_ratio_d{di}"] = round(vol_val / (avg_vol + 1e-6), 3) if avg_vol > 0 else 1.0
                else:
                    row[f"market_vol_ratio_d{di}"] = 1.0
        else:
            for di in range(5):
                row[f"market_vol_ratio_d{di}"] = 1.0

        # ========== 派生趋势因子 ==========
        limit_up_counts_5d = macro_cache.get("limit_up_counts_5d", {})
        consec_max_5d      = macro_cache.get("consec_max_5d", {})
        lookback_5d        = getattr(data_bundle, "lookback_dates_5d", [])
        hist_dates         = lookback_5d[:-1] if len(lookback_5d) > 1 else []  # d1~d4

        row["market_limit_up_rate"] = round(row["market_limit_up_count"] / TOTAL_LISTED_APPROX, 4)

        if hist_dates and limit_up_counts_5d:
            hist_up_counts = [limit_up_counts_5d.get(d, 0) for d in hist_dates]
            avg_hist_up    = float(np.mean(hist_up_counts)) if hist_up_counts else 0.0
            # avg_hist_up < 1 说明历史数据缺失（未入库），退化为中性 1.0 避免除以零
            if avg_hist_up >= 1:
                raw = row["market_limit_up_count"] / avg_hist_up
                row["market_limit_up_5d_trend"] = round(float(np.clip(raw, 0.1, 10.0)), 3)
            else:
                row["market_limit_up_5d_trend"] = 1.0
        else:
            row["market_limit_up_5d_trend"] = 1.0

        if hist_dates and consec_max_5d:
            hist_consec     = [consec_max_5d.get(d, 0) for d in hist_dates]
            avg_hist_consec = float(np.mean(hist_consec)) if hist_consec else 0.0
            if avg_hist_consec >= 1:
                raw = row["market_max_consec_num"] / avg_hist_consec
                row["market_consec_5d_trend"] = round(float(np.clip(raw, 0.1, 10.0)), 3)
            else:
                row["market_consec_5d_trend"] = 1.0
        else:
            row["market_consec_5d_trend"] = 1.0

        feature_df = pd.DataFrame([row])
        logger.debug(
            f"[市场宏观] {trade_date} 涨停:{row['market_limit_up_count']} "
            f"跌停:{row['market_limit_down_count']} "
            f"最高板:{row['market_max_consec_num']} "
            f"上证:{row['index_sh_pct_chg']:.2f}% "
            f"全市场成交量比率(d0):{row['market_vol_ratio_d0']:.3f}"
        )
        return feature_df, {}
