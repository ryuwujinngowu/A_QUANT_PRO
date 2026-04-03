"""
板块内个股特征计算
==================
输出因子全览（每个交易日 d0~d4 各输出一组）：

【原始行情】
  stock_open/high/low/close/pct_chg_{d}

【量能比因子】
  stock_amount_5d_ratio_{d} : 成交额量级比 = 当日成交额 / 近5日均额
                              放量>1，缩量<1，停牌=0（消除跨股绝对额差异）

【情绪合成分（保留，与原子因子并存）】
  stock_profit_{d}      : 上涨日 SEI；下跌/平盘日 = 0
  stock_loss_{d}        : 下跌日 100-SEI；上涨/平盘日 = 0
  stock_hdi_{d}         : HDI 持股难度指数（0-100）

【方向/价格原子因子 ★ 新增独立列】
  stock_gap_return_{d}  : 开盘缺口率 = (open-pre_close)/pre_close
                          正=高开（主力溢价拉升），负=低开（出货压力）
  stock_candle_{d}      : K 线结构 {2=真阳,1=假阳,-1=假阴,-2=真阴}
                          假阴(低开高走)是次日大涨强预测子
  stock_cpr_{d}         : 收盘位置比 = (close-low)/(high-low)
                          0=收于最低，1=收于最高（封板股≈1）

【波动/持仓结构原子因子 ★ 新增独立列】
  stock_max_dd_{d}      : 日内最大回撤（涨停板次日走势的强预测子）
  stock_upper_shadow_{d}: 上影线比率，反映上方抛压
  stock_lower_shadow_{d}: 下影线比率，反映下方支撑
  stock_trend_r2_{d}    : 分钟线趋势 R²（0=震荡，1=单边趋势）
  stock_vwap_dev_{d}    : VWAP 偏离度（筹码散乱程度）

【涨跌停行为原子因子 ★ 新增独立列】
  stock_seal_times_{d}  : 封板次数
  stock_break_times_{d} : 开板次数（开板越多情绪越弱）
  stock_lift_times_{d}  : 跌停翘板次数

【量能原子因子 ★ 全新维度】
  stock_vol_ratio_{d}   : 量比 = 当日量 / 近5日均量
                          放量>1，缩量<1，停牌=0

【时间持续类因子 ★ 新增维度】
  stock_red_time_ratio_{d}          : 红盘持续时间 = 收盘>昨收的分钟数 / 全天总分钟数
                                      0=全天绿盘，1=全天红盘，0.5=均衡（无分钟线时默认）
  stock_float_profit_time_ratio_{d} : 浮盈持续时间 = 收盘>昨日均价的分钟数 / 全天总分钟数
                                      昨日均价 = amount / (vol × 100)
                                      相比红盘时间更反映"大资金成本"的盈亏状态
  stock_red_session_pm_ratio_{d}    : 红盘时段午盘占比
                                      = 午盘红盘分钟数 / (早盘+午盘红盘分钟数)
                                      0=红盘全在早盘（可能日内衰减），1=全在午盘（持续性强）
                                      无红盘时填 0.5（中性），早盘: 9:30-11:30，午盘: 13:00-15:00
  stock_float_session_pm_ratio_{d}  : 浮盈时段午盘占比（同 red_session_pm_ratio 逻辑）

【超预期强度因子 ★ 新增维度】
  stock_expect_surprise_{d}          : 超预期强度 = 开盘预期 + 日内兑现 + 午后承接 的合成
                                      用于区分“高开高走真强 / 低开修复超预期 / 高开低走不及预期”
  sector_avg_profit_{d}, sector_avg_loss_{d}

【个股排名】
  stock_sector_20d_rank

设计说明：
  sei_cache 存储全量 factors dict，避免双重计算
  无分钟线时从日线 OHLC 回退（calc_daily_atomic），保证缓存完整
  停牌 vs 无分钟线 使用不同中性值，语义区分明确
"""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import numpy as np
import pandas as pd
from utils.log_utils import logger
from features.base_feature import BaseFeature
from features.feature_registry import feature_registry
from features.emotion.sei_feature import SEIFeature
from utils.common_tools import (
    sort_by_recent_gain,
    calc_limit_up_price,
    calc_limit_down_price,
)

# SEI/HDI 并行计算线程数（numpy 释放 GIL，线程并发有效）
_SEI_WORKERS = 6


# ============================================================
# 无分钟线时的回退中性值（语义：数据不完整，不是停牌）
# ============================================================
ATOMIC_NEUTRAL = {
    "gap_return":             0.0,    # 无法判断，填平开
    "candle_type":            0,      # 0=无结构信息
    "cpr":                    0.5,    # 中性位置
    "max_dd_intra":           0.10,   # 保守中位值
    "upper_shadow":           0.0,
    "lower_shadow":           0.0,
    "trend_r2":               0.5,    # 中性
    "vwap_deviation":         0.01,   # 保守小值
    "seal_times":             0,
    "break_times":            0,
    "lift_times":             0,
    "vol_ratio":              1.0,    # 正常量
    "hdi":                    50.0,
    # 时间持续类因子（无数据时中性 0.5）
    "red_time_ratio":         0.5,
    "float_profit_time_ratio":0.5,
    "red_session_pm_ratio":   0.5,
    "float_session_pm_ratio": 0.5,
}

# 停牌时的填充值（语义：该股无行情，成交量 = 0）
SUSPENDED_FILL = dict(ATOMIC_NEUTRAL)
SUSPENDED_FILL["vol_ratio"]              = 0.0    # 停牌量为 0，量比=0 有明确语义
# 停牌 = 确定无红盘/浮盈 → session 偏向设为 -1（区别于 ATOMIC_NEUTRAL 的 0.5 未知值）
SUSPENDED_FILL["red_session_pm_ratio"]   = -1.0
SUSPENDED_FILL["float_session_pm_ratio"] = -1.0


@feature_registry.register("sector_stock")
class SectorStockFeature(BaseFeature):
    """板块内个股特征类（全量原子因子版）"""

    feature_name = "sector_stock"
    _day_tags    = [f"d{i}" for i in range(5)]

    factor_columns = [
        "sector_id", "sector_name", "stock_sector_20d_rank",
        *[f"stock_open_{t}"          for t in _day_tags],
        *[f"stock_high_{t}"          for t in _day_tags],
        *[f"stock_low_{t}"           for t in _day_tags],
        *[f"stock_close_{t}"         for t in _day_tags],
        *[f"stock_pct_chg_{t}"            for t in _day_tags],
        *[f"stock_amount_5d_ratio_{t}"   for t in _day_tags],
        *[f"stock_profit_{t}"        for t in _day_tags],
        *[f"stock_loss_{t}"          for t in _day_tags],
        *[f"stock_hdi_{t}"           for t in _day_tags],
        *[f"stock_gap_return_{t}"    for t in _day_tags],
        *[f"stock_candle_{t}"        for t in _day_tags],
        *[f"stock_cpr_{t}"           for t in _day_tags],
        *[f"stock_max_dd_{t}"        for t in _day_tags],
        *[f"stock_upper_shadow_{t}"  for t in _day_tags],
        *[f"stock_lower_shadow_{t}"  for t in _day_tags],
        *[f"stock_trend_r2_{t}"      for t in _day_tags],
        *[f"stock_vwap_dev_{t}"      for t in _day_tags],
        *[f"stock_seal_times_{t}"    for t in _day_tags],
        *[f"stock_break_times_{t}"   for t in _day_tags],
        *[f"stock_lift_times_{t}"    for t in _day_tags],
        *[f"stock_vol_ratio_{t}"              for t in _day_tags],
        *[f"stock_red_time_ratio_{t}"         for t in _day_tags],
        *[f"stock_float_profit_time_ratio_{t}" for t in _day_tags],
        *[f"stock_red_session_pm_ratio_{t}"   for t in _day_tags],
        *[f"stock_float_session_pm_ratio_{t}" for t in _day_tags],
        *[f"stock_expect_surprise_{t}"       for t in _day_tags],
        *[f"sector_avg_profit_{t}"            for t in _day_tags],
        *[f"sector_avg_loss_{t}"              for t in _day_tags],
    ]

    def __init__(self):
        super().__init__()
        self.sei_calculator = SEIFeature()

    # ------------------------------------------------------------------ #
    # 工具方法
    # ------------------------------------------------------------------ #

    @staticmethod
    def _calc_vol_ratio(ts_code: str, target_date: str,
                        all_dates_5d: List[str], daily_grouped: dict) -> float:
        """量比 = 当日成交量 / 近5日（含当日）均量
        注意：kline_day 表中成交量列名为 volume（data_cleaner 入库时已从 vol 重命名）"""
        today_row = daily_grouped.get((ts_code, target_date))
        if not today_row:
            return 0.0
        today_vol = float(today_row.get("volume", 0) or 0)  # ← 列名 volume，非 vol
        if today_vol == 0:
            return 0.0
        vols = [
            float(daily_grouped[(ts_code, d)].get("volume", 0) or 0)
            for d in all_dates_5d
            if (ts_code, d) in daily_grouped
        ]
        avg_vol = np.mean(vols) if vols else 0.0
        return round(today_vol / (avg_vol + 1e-6), 3)

    @staticmethod
    def _calc_amount_ratio(ts_code: str, target_date: str,
                           all_dates_5d: List[str], daily_grouped: dict) -> float:
        """成交额量级比 = 当日成交额 / 近5日（含当日）平均成交额（消除跨股绝对额差异）"""
        today_row = daily_grouped.get((ts_code, target_date))
        if not today_row:
            return 0.0
        today_amt = float(today_row.get("amount", 0) or 0)
        if today_amt == 0:
            return 0.0
        amts = [
            float(daily_grouped[(ts_code, d)].get("amount", 0) or 0)
            for d in all_dates_5d
            if (ts_code, d) in daily_grouped
        ]
        avg_amt = np.mean(amts) if amts else 0.0
        return round(today_amt / (avg_amt + 1e-6), 3)

    @staticmethod
    def _calc_expect_surprise(f: dict, pct_chg: float) -> float:
        """
        超预期强度：衡量开盘预期是否被日内走势兑现/超额兑现。
        核心思想：
          - 高开后还能全天红盘、午后更强、收盘靠近高点 → 高分
          - 低开但持续修复、浮盈时间长、假阴转强 → 也给高分
          - 高开低走、冲高回落、午后走弱 → 低分
        返回 0-100，50 为中性。
        """
        gap = float(f.get("gap_return", 0.0) or 0.0)
        cpr = float(f.get("cpr", 0.5) or 0.5)
        red_time = float(f.get("red_time_ratio", 0.5) or 0.5)
        float_time = float(f.get("float_profit_time_ratio", 0.5) or 0.5)
        red_pm = float(f.get("red_session_pm_ratio", 0.5) or 0.5)
        if red_pm < 0:
            red_pm = 0.0
        max_dd = float(f.get("max_dd_intra", 0.10) or 0.10)
        candle = int(f.get("candle_type", 0) or 0)

        score = 50.0
        score += max(min(gap * 120.0, 15.0), -15.0)
        score += (cpr - 0.5) * 24.0
        score += (red_time - 0.5) * 26.0
        score += (float_time - 0.5) * 20.0
        score += (red_pm - 0.5) * 14.0
        score -= min(max_dd * 120.0, 18.0)
        score += {2: 8.0, -1: 6.0, 1: -6.0, -2: -10.0}.get(candle, 0.0)
        score += max(min(pct_chg * 0.8, 8.0), -8.0)
        return round(float(np.clip(score, 0.0, 100.0)), 2)

    # ------------------------------------------------------------------ #
    # 主计算入口
    # ------------------------------------------------------------------ #

    def calculate(self, data_bundle) -> tuple:
        result_rows   = []
        factor_dict   = {}
        trade_date     = data_bundle.trade_date
        top3_sectors   = data_bundle.top3_sectors
        sector_map     = data_bundle.sector_candidate_map
        daily_grouped  = data_bundle.daily_grouped
        minute_cache   = data_bundle.minute_cache
        all_dates_5d   = data_bundle.lookback_dates_5d    # 升序，[-1]=trade_date
        all_dates_20d  = sorted(data_bundle.lookback_dates_20d)  # 升序，用于 prev_date 查找

        # d0=trade_date，d4=4天前
        day_tag_map: Dict[str, str] = {
            f"d{4 - i}": date for i, date in enumerate(all_dates_5d)
        }

        # ================================================================
        # SEI/HDI 计算闭包（移到循环外，供 sector 股票和非 sector 股票共用）
        # ================================================================
        sei_calc = self.sei_calculator

        def _compute_sei(task):
            daily_key, daily_row, pre_close = task
            ts_code, target_date = daily_key
            up_limit   = calc_limit_up_price(ts_code, pre_close)
            down_limit = calc_limit_down_price(ts_code, pre_close)
            minute_df  = minute_cache.get(daily_key, pd.DataFrame())

            # 昨日 VWAP（元/股）= amount(千元) × 1000 / (volume(手) × 100股/手)
            #                  = amount × 10 / volume
            # 注意：kline_day 列名为 volume（不是 vol），data_cleaner 入库时已重命名
            # 用于浮盈持续时间计算；找不到昨日数据则传 0（sei_feature 内部降级为昨收）
            vwap_prev = 0.0
            try:
                idx = all_dates_20d.index(target_date)
                if idx > 0:
                    prev_row = daily_grouped.get((ts_code, all_dates_20d[idx - 1]), {})
                    prev_vol = float(prev_row.get("volume", 0) or 0)  # ← volume，非 vol
                    prev_amt = float(prev_row.get("amount", 0) or 0)
                    if prev_vol > 0:
                        vwap_prev = prev_amt * 10 / prev_vol   # 千元×1000/(手×100) = 元/股
            except (ValueError, IndexError):
                pass

            hdi, factors = sei_calc._calculate_minute_hdi(
                minute_df, pre_close, up_limit, down_limit, vwap_prev=vwap_prev
            )
            if factors:
                sei = sei_calc._factors_to_sei(factors, up_limit)
            else:
                sei = hdi = 50.0
                factors = SEIFeature.calc_daily_atomic(
                    open_price  = daily_row.get("open",  pre_close),
                    high_price  = daily_row.get("high",  pre_close),
                    low_price   = daily_row.get("low",   pre_close),
                    close_price = daily_row.get("close", pre_close),
                    pre_close   = pre_close,
                )
            return daily_key, {"sei": float(sei), "hdi": float(hdi), "factors": factors}

        # ================================================================
        # 阶段 1：板块级预计算
        # ================================================================
        # 记录所有板块已处理的股票，供后续非 sector_heat 股票跳过重复计算
        sector_processed_codes: set = set()

        for sector_idx, sector_name in enumerate(top3_sectors, 1):
            if not sector_name or sector_name not in sector_map:
                continue
            sector_d0_df = sector_map[sector_name]
            if sector_d0_df.empty:
                continue

            sector_ts_codes: List[str] = sector_d0_df["ts_code"].unique().tolist()
            sector_processed_codes.update(sector_ts_codes)

            # ---- 20 日涨幅排名 ----
            rank_map, rank_median = {}, 0
            try:
                sorted_df = sort_by_recent_gain(sector_d0_df, trade_date, day_count=20)
                if not sorted_df.empty:
                    sorted_df = sorted_df.copy()
                    sorted_df["rank"] = range(1, len(sorted_df) + 1)
                    rank_map    = dict(zip(sorted_df["ts_code"], sorted_df["rank"]))
                    rank_median = sorted_df["rank"].median()
            except Exception as e:
                logger.warning(f"[板块个股] {sector_name} 排名失败: {e}")

            # ================================================================
            # SEI/HDI/全量原子因子统一缓存
            # value = {"sei": float, "hdi": float, "factors": dict}
            # 无分钟线时用 calc_daily_atomic 从日线回退，保证缓存始终有值
            # ================================================================
            sei_cache: Dict[tuple, dict] = {}

            # 收集需要计算 SEI/HDI 的任务
            sei_tasks = []
            seen_keys = set()
            for ts_code in sector_ts_codes:
                for _, target_date in day_tag_map.items():
                    daily_key = (ts_code, target_date)
                    if daily_key in seen_keys or daily_key not in daily_grouped:
                        continue
                    seen_keys.add(daily_key)
                    daily_row = daily_grouped[daily_key]
                    pre_close = daily_row.get("pre_close", 0)
                    if not pre_close or pre_close <= 0:
                        continue
                    sei_tasks.append((daily_key, daily_row, pre_close))

            # 多线程并行计算 SEI/HDI
            with ThreadPoolExecutor(max_workers=_SEI_WORKERS) as pool:
                for key, result in pool.map(_compute_sei, sei_tasks):
                    sei_cache[key] = result

            # ---- 板块每日赚钱/亏钱效应 ----
            sector_day_factors: Dict[str, dict] = {}
            day_sei_mean = defaultdict(lambda: {"up": [], "down": []})

            for day_tag, target_date in day_tag_map.items():
                p_list, l_list = [], []
                for ts_code in sector_ts_codes:
                    dk = (ts_code, target_date)
                    if dk not in daily_grouped or dk not in sei_cache:
                        continue
                    pct = daily_grouped[dk].get("pct_chg", 0)
                    sei = sei_cache[dk]["sei"]
                    if pct > 1e-6:
                        p_list.append(sei);  day_sei_mean[day_tag]["up"].append(sei)
                    elif pct < -1e-6:
                        l_list.append(sei);  day_sei_mean[day_tag]["down"].append(sei)

                sector_day_factors[day_tag] = {
                    "profit": round(np.mean(p_list), 2) if p_list else 50.0,
                    "loss":   round(100 - np.mean(l_list), 2) if l_list else 50.0,
                }
                day_sei_mean[day_tag]["up"]   = float(np.mean(day_sei_mean[day_tag]["up"]))   if day_sei_mean[day_tag]["up"]   else 50.0
                day_sei_mean[day_tag]["down"] = float(np.mean(day_sei_mean[day_tag]["down"])) if day_sei_mean[day_tag]["down"] else 50.0

            # ================================================================
            # 阶段 2：个股级特征组装
            # ================================================================
            for _, d0_row in sector_d0_df.iterrows():
                ts_code = d0_row["ts_code"]
                d0_key  = (ts_code, trade_date)
                if d0_key not in daily_grouped:
                    logger.warning(f"[板块个股] {ts_code} {trade_date} 无日线，跳过")
                    continue

                d0_data = daily_grouped[d0_key]
                row = {
                    "stock_code":            ts_code,
                    "trade_date":            trade_date,
                    "sector_id":             sector_idx,
                    "sector_name":           sector_name,
                    "stock_sector_20d_rank": rank_map.get(ts_code, rank_median),
                }

                for day_tag, target_date in day_tag_map.items():
                    daily_key  = (ts_code, target_date)
                    daily_data = daily_grouped.get(daily_key)
                    cache      = sei_cache.get(daily_key)

                    # ---- 原始行情 ----
                    if daily_data:
                        row[f"stock_open_{day_tag}"]             = daily_data.get("open",    d0_data.get("open",    0))
                        row[f"stock_high_{day_tag}"]             = daily_data.get("high",    d0_data.get("high",    0))
                        row[f"stock_low_{day_tag}"]              = daily_data.get("low",     d0_data.get("low",     0))
                        row[f"stock_close_{day_tag}"]            = daily_data.get("close",   d0_data.get("close",   0))
                        row[f"stock_pct_chg_{day_tag}"]          = daily_data.get("pct_chg", 0.0)
                        row[f"stock_amount_5d_ratio_{day_tag}"]  = self._calc_amount_ratio(
                            ts_code, target_date, all_dates_5d, daily_grouped
                        )
                    else:
                        row[f"stock_open_{day_tag}"]             = d0_data.get("open",  0)
                        row[f"stock_high_{day_tag}"]             = d0_data.get("high",  0)
                        row[f"stock_low_{day_tag}"]              = d0_data.get("low",   0)
                        row[f"stock_close_{day_tag}"]            = d0_data.get("close", 0)
                        row[f"stock_pct_chg_{day_tag}"]          = 0.0
                        row[f"stock_amount_5d_ratio_{day_tag}"]  = 0.0

                    # ---- 情绪合成分 + 全量原子因子 ----
                    if daily_data and cache:
                        pct_chg = daily_data.get("pct_chg", 0)
                        sei     = cache["sei"]
                        hdi     = cache["hdi"]
                        f       = cache["factors"]

                        # 分钟线缺失时 SEI 用板块均值替代（candle_type 已由日线回退）
                        if minute_cache.get(daily_key, pd.DataFrame()).empty:
                            sei = day_sei_mean[day_tag]["up"]   if pct_chg > 1e-6 else \
                                  day_sei_mean[day_tag]["down"] if pct_chg < -1e-6 else sei

                        if pct_chg > 1e-6:
                            row[f"stock_profit_{day_tag}"] = round(sei, 2)
                            row[f"stock_loss_{day_tag}"]   = 0.0
                        elif pct_chg < -1e-6:
                            row[f"stock_profit_{day_tag}"] = 0.0
                            row[f"stock_loss_{day_tag}"]   = round(100 - sei, 2)
                        else:
                            row[f"stock_profit_{day_tag}"] = 0.0
                            row[f"stock_loss_{day_tag}"]   = 0.0

                        row[f"stock_hdi_{day_tag}"]          = hdi
                        row[f"stock_gap_return_{day_tag}"]   = f.get("gap_return",     ATOMIC_NEUTRAL["gap_return"])
                        row[f"stock_candle_{day_tag}"]        = f.get("candle_type",    ATOMIC_NEUTRAL["candle_type"])
                        row[f"stock_cpr_{day_tag}"]           = f.get("cpr",            ATOMIC_NEUTRAL["cpr"])
                        row[f"stock_max_dd_{day_tag}"]        = f.get("max_dd_intra",   ATOMIC_NEUTRAL["max_dd_intra"])
                        row[f"stock_upper_shadow_{day_tag}"]  = f.get("upper_shadow",   ATOMIC_NEUTRAL["upper_shadow"])
                        row[f"stock_lower_shadow_{day_tag}"]  = f.get("lower_shadow",   ATOMIC_NEUTRAL["lower_shadow"])
                        row[f"stock_trend_r2_{day_tag}"]      = f.get("trend_r2",       ATOMIC_NEUTRAL["trend_r2"])
                        row[f"stock_vwap_dev_{day_tag}"]      = f.get("vwap_deviation", ATOMIC_NEUTRAL["vwap_deviation"])
                        row[f"stock_seal_times_{day_tag}"]               = f.get("seal_times",             0)
                        row[f"stock_break_times_{day_tag}"]              = f.get("break_times",            0)
                        row[f"stock_lift_times_{day_tag}"]               = f.get("lift_times",             0)
                        row[f"stock_vol_ratio_{day_tag}"]                = self._calc_vol_ratio(
                            ts_code, target_date, all_dates_5d, daily_grouped
                        )
                        row[f"stock_red_time_ratio_{day_tag}"]           = f.get("red_time_ratio",          ATOMIC_NEUTRAL["red_time_ratio"])
                        row[f"stock_float_profit_time_ratio_{day_tag}"]  = f.get("float_profit_time_ratio", ATOMIC_NEUTRAL["float_profit_time_ratio"])
                        row[f"stock_red_session_pm_ratio_{day_tag}"]     = f.get("red_session_pm_ratio",    ATOMIC_NEUTRAL["red_session_pm_ratio"])
                        row[f"stock_float_session_pm_ratio_{day_tag}"]   = f.get("float_session_pm_ratio",  ATOMIC_NEUTRAL["float_session_pm_ratio"])
                        row[f"stock_expect_surprise_{day_tag}"]          = self._calc_expect_surprise(f, pct_chg)

                    else:
                        # 停牌或完全无数据
                        row[f"stock_profit_{day_tag}"]                   = 0.0
                        row[f"stock_loss_{day_tag}"]                     = 0.0
                        row[f"stock_hdi_{day_tag}"]                      = SUSPENDED_FILL["hdi"]
                        row[f"stock_gap_return_{day_tag}"]               = SUSPENDED_FILL["gap_return"]
                        row[f"stock_candle_{day_tag}"]                   = SUSPENDED_FILL["candle_type"]
                        row[f"stock_cpr_{day_tag}"]                      = SUSPENDED_FILL["cpr"]
                        row[f"stock_max_dd_{day_tag}"]                   = SUSPENDED_FILL["max_dd_intra"]
                        row[f"stock_upper_shadow_{day_tag}"]             = SUSPENDED_FILL["upper_shadow"]
                        row[f"stock_lower_shadow_{day_tag}"]             = SUSPENDED_FILL["lower_shadow"]
                        row[f"stock_trend_r2_{day_tag}"]                 = SUSPENDED_FILL["trend_r2"]
                        row[f"stock_vwap_dev_{day_tag}"]                 = SUSPENDED_FILL["vwap_deviation"]
                        row[f"stock_seal_times_{day_tag}"]               = 0
                        row[f"stock_break_times_{day_tag}"]              = 0
                        row[f"stock_lift_times_{day_tag}"]               = 0
                        row[f"stock_vol_ratio_{day_tag}"]                = SUSPENDED_FILL["vol_ratio"]
                        row[f"stock_red_time_ratio_{day_tag}"]           = SUSPENDED_FILL["red_time_ratio"]
                        row[f"stock_float_profit_time_ratio_{day_tag}"]  = SUSPENDED_FILL["float_profit_time_ratio"]
                        row[f"stock_red_session_pm_ratio_{day_tag}"]     = SUSPENDED_FILL["red_session_pm_ratio"]
                        row[f"stock_float_session_pm_ratio_{day_tag}"]   = SUSPENDED_FILL["float_session_pm_ratio"]
                        row[f"stock_expect_surprise_{day_tag}"]          = 50.0

                    # ---- 板块平均（无条件填充）----
                    row[f"sector_avg_profit_{day_tag}"] = sector_day_factors[day_tag]["profit"]
                    row[f"sector_avg_loss_{day_tag}"]   = sector_day_factors[day_tag]["loss"]

                result_rows.append(row)

        # ================================================================
        # 阶段 3：非 sector_heat 候选股特征计算（BUG-12 修复）
        # ================================================================
        # 对于未被任何板块覆盖的候选股，同样计算完整的 per-stock 因子。
        # sector 级别字段填中性值（sector_id=0, sector_name="", rank=0, avg_profit/loss=50.0）
        # 这些股票的日线/分钟线数据已包含在 data_bundle 中（target_ts_codes 扩展后已预加载）。
        all_target_codes = list(getattr(data_bundle, "target_ts_codes", None) or [])
        extra_codes = [c for c in all_target_codes if c not in sector_processed_codes]

        if extra_codes:
            # 收集额外股票的 SEI/HDI 任务
            extra_sei_tasks = []
            extra_seen_keys: set = set()
            for ts_code in extra_codes:
                for _, target_date in day_tag_map.items():
                    daily_key = (ts_code, target_date)
                    if daily_key in extra_seen_keys or daily_key not in daily_grouped:
                        continue
                    extra_seen_keys.add(daily_key)
                    daily_row = daily_grouped[daily_key]
                    pre_close = daily_row.get("pre_close", 0)
                    if not pre_close or pre_close <= 0:
                        continue
                    extra_sei_tasks.append((daily_key, daily_row, pre_close))

            extra_sei_cache: Dict[tuple, dict] = {}
            if extra_sei_tasks:
                with ThreadPoolExecutor(max_workers=_SEI_WORKERS) as pool:
                    for key, result in pool.map(_compute_sei, extra_sei_tasks):
                        extra_sei_cache[key] = result

            # 组装每只非 sector 股票的特征行（无板块均值，填中性 50.0）
            for ts_code in extra_codes:
                d0_key = (ts_code, trade_date)
                if d0_key not in daily_grouped:
                    continue

                d0_data = daily_grouped[d0_key]
                row = {
                    "stock_code":            ts_code,
                    "trade_date":            trade_date,
                    "sector_id":             0,      # 无板块
                    "sector_name":           "",
                    "stock_sector_20d_rank": 0,      # 无排名
                }

                for day_tag, target_date in day_tag_map.items():
                    daily_key  = (ts_code, target_date)
                    daily_data = daily_grouped.get(daily_key)
                    cache      = extra_sei_cache.get(daily_key)

                    # ---- 原始行情 ----
                    if daily_data:
                        row[f"stock_open_{day_tag}"]             = daily_data.get("open",    d0_data.get("open",    0))
                        row[f"stock_high_{day_tag}"]             = daily_data.get("high",    d0_data.get("high",    0))
                        row[f"stock_low_{day_tag}"]              = daily_data.get("low",     d0_data.get("low",     0))
                        row[f"stock_close_{day_tag}"]            = daily_data.get("close",   d0_data.get("close",   0))
                        row[f"stock_pct_chg_{day_tag}"]          = daily_data.get("pct_chg", 0.0)
                        row[f"stock_amount_5d_ratio_{day_tag}"]  = self._calc_amount_ratio(
                            ts_code, target_date, all_dates_5d, daily_grouped
                        )
                    else:
                        row[f"stock_open_{day_tag}"]             = d0_data.get("open",  0)
                        row[f"stock_high_{day_tag}"]             = d0_data.get("high",  0)
                        row[f"stock_low_{day_tag}"]              = d0_data.get("low",   0)
                        row[f"stock_close_{day_tag}"]            = d0_data.get("close", 0)
                        row[f"stock_pct_chg_{day_tag}"]          = 0.0
                        row[f"stock_amount_5d_ratio_{day_tag}"]  = 0.0

                    # ---- 情绪合成分 + 全量原子因子 ----
                    if daily_data and cache:
                        pct_chg = daily_data.get("pct_chg", 0)
                        sei     = cache["sei"]
                        hdi     = cache["hdi"]
                        f       = cache["factors"]

                        if pct_chg > 1e-6:
                            row[f"stock_profit_{day_tag}"] = round(sei, 2)
                            row[f"stock_loss_{day_tag}"]   = 0.0
                        elif pct_chg < -1e-6:
                            row[f"stock_profit_{day_tag}"] = 0.0
                            row[f"stock_loss_{day_tag}"]   = round(100 - sei, 2)
                        else:
                            row[f"stock_profit_{day_tag}"] = 0.0
                            row[f"stock_loss_{day_tag}"]   = 0.0

                        row[f"stock_hdi_{day_tag}"]          = hdi
                        row[f"stock_gap_return_{day_tag}"]   = f.get("gap_return",     ATOMIC_NEUTRAL["gap_return"])
                        row[f"stock_candle_{day_tag}"]        = f.get("candle_type",    ATOMIC_NEUTRAL["candle_type"])
                        row[f"stock_cpr_{day_tag}"]           = f.get("cpr",            ATOMIC_NEUTRAL["cpr"])
                        row[f"stock_max_dd_{day_tag}"]        = f.get("max_dd_intra",   ATOMIC_NEUTRAL["max_dd_intra"])
                        row[f"stock_upper_shadow_{day_tag}"]  = f.get("upper_shadow",   ATOMIC_NEUTRAL["upper_shadow"])
                        row[f"stock_lower_shadow_{day_tag}"]  = f.get("lower_shadow",   ATOMIC_NEUTRAL["lower_shadow"])
                        row[f"stock_trend_r2_{day_tag}"]      = f.get("trend_r2",       ATOMIC_NEUTRAL["trend_r2"])
                        row[f"stock_vwap_dev_{day_tag}"]      = f.get("vwap_deviation", ATOMIC_NEUTRAL["vwap_deviation"])
                        row[f"stock_seal_times_{day_tag}"]               = f.get("seal_times",             0)
                        row[f"stock_break_times_{day_tag}"]              = f.get("break_times",            0)
                        row[f"stock_lift_times_{day_tag}"]               = f.get("lift_times",             0)
                        row[f"stock_vol_ratio_{day_tag}"]                = self._calc_vol_ratio(
                            ts_code, target_date, all_dates_5d, daily_grouped
                        )
                        row[f"stock_red_time_ratio_{day_tag}"]           = f.get("red_time_ratio",          ATOMIC_NEUTRAL["red_time_ratio"])
                        row[f"stock_float_profit_time_ratio_{day_tag}"]  = f.get("float_profit_time_ratio", ATOMIC_NEUTRAL["float_profit_time_ratio"])
                        row[f"stock_red_session_pm_ratio_{day_tag}"]     = f.get("red_session_pm_ratio",    ATOMIC_NEUTRAL["red_session_pm_ratio"])
                        row[f"stock_float_session_pm_ratio_{day_tag}"]   = f.get("float_session_pm_ratio",  ATOMIC_NEUTRAL["float_session_pm_ratio"])
                        row[f"stock_expect_surprise_{day_tag}"]          = self._calc_expect_surprise(f, pct_chg)

                    else:
                        # 停牌或完全无数据
                        row[f"stock_profit_{day_tag}"]                   = 0.0
                        row[f"stock_loss_{day_tag}"]                     = 0.0
                        row[f"stock_hdi_{day_tag}"]                      = SUSPENDED_FILL["hdi"]
                        row[f"stock_gap_return_{day_tag}"]               = SUSPENDED_FILL["gap_return"]
                        row[f"stock_candle_{day_tag}"]                   = SUSPENDED_FILL["candle_type"]
                        row[f"stock_cpr_{day_tag}"]                      = SUSPENDED_FILL["cpr"]
                        row[f"stock_max_dd_{day_tag}"]                   = SUSPENDED_FILL["max_dd_intra"]
                        row[f"stock_upper_shadow_{day_tag}"]             = SUSPENDED_FILL["upper_shadow"]
                        row[f"stock_lower_shadow_{day_tag}"]             = SUSPENDED_FILL["lower_shadow"]
                        row[f"stock_trend_r2_{day_tag}"]                 = SUSPENDED_FILL["trend_r2"]
                        row[f"stock_vwap_dev_{day_tag}"]                 = SUSPENDED_FILL["vwap_deviation"]
                        row[f"stock_seal_times_{day_tag}"]               = 0
                        row[f"stock_break_times_{day_tag}"]              = 0
                        row[f"stock_lift_times_{day_tag}"]               = 0
                        row[f"stock_vol_ratio_{day_tag}"]                = SUSPENDED_FILL["vol_ratio"]
                        row[f"stock_red_time_ratio_{day_tag}"]           = SUSPENDED_FILL["red_time_ratio"]
                        row[f"stock_float_profit_time_ratio_{day_tag}"]  = SUSPENDED_FILL["float_profit_time_ratio"]
                        row[f"stock_red_session_pm_ratio_{day_tag}"]     = SUSPENDED_FILL["red_session_pm_ratio"]
                        row[f"stock_float_session_pm_ratio_{day_tag}"]   = SUSPENDED_FILL["float_session_pm_ratio"]
                        row[f"stock_expect_surprise_{day_tag}"]          = 50.0

                    # ---- 板块平均（无板块上下文，填中性值）----
                    row[f"sector_avg_profit_{day_tag}"] = 50.0
                    row[f"sector_avg_loss_{day_tag}"]   = 50.0

                result_rows.append(row)

            logger.info(
                f"[板块个股特征] {trade_date} 非sector_heat候选股补充计算"
                f" | extra_codes: {len(extra_codes)} | SEI任务: {len(extra_sei_tasks)}"
            )

        feature_df = pd.DataFrame(result_rows)
        logger.info(
            f"[板块个股特征] {trade_date} 完成"
            f" | 样本: {len(feature_df)} | 列数: {len(feature_df.columns) if not feature_df.empty else 0}"
        )
        return feature_df, factor_dict