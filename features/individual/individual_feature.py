"""
个股精细因子集 (features/individual/individual_feature.py)
============================================================
新增因子列表：

【振幅类 - 日线单值计算，d0~d4】
  stock_intra_amp_{d}     : 日内振幅 = (high - low) / pre_close
  stock_body_amp_{d}      : 实体振幅 = |close - open| / pre_close
  stock_body_ratio_{d}    : 实体占比 = |close - open| / (high - low)
                            一字板/停牌时填 0.5（中性）

【位置类 - 日线/分钟线，d0~d4】
  stock_open_pos_{d}      : 开盘位置比 = (open - low) / (high - low)
                            一字板/停牌时填 0.5（中性）
  stock_vwap_pos_{d}      : VWAP均价位置比 = (vwap - low) / (high - low)
                            无分钟线时填 0.5（中性）

【波动结构类 - 单次遍历5分钟K线，d0~d4】
  stock_up_vol_ratio_{d}  : 上行波动占比 = 阳线实体涨幅之和 / 日内总振幅
  stock_dn_vol_ratio_{d}  : 下行波动占比 = 阴线实体跌幅之和 / 日内总振幅
  stock_vol_conc_{d}      : 波动集中度 = 最大单根K线实体 / 日内总振幅

【高点维持类 - 5分钟K线，d0~d4】
  stock_high_hold_ratio_{d}: 高点维持时间占比
                              = K线收盘 ≥ 当日最高×0.9 的条数 / 总条数

【量价配合类 - 5分钟K线（成交金额），d0~d4】
  stock_buy_vol_ratio_{d} : 放量上涨比例 = 阳线平均成交金额 / 阴线平均成交金额
                            分母为0时填 1（中性）
  stock_pv_corr_{d}       : 量价同步性 = 收盘价序列与成交金额序列的皮尔逊相关系数
                            方差为0时填 0（中性）
  stock_tail_surge_ratio_{d}: 尾盘量能突袭比 = 尾盘(14:30-15:00)均额 / 非尾盘均额

【风险/压力类 - 5分钟K线，d0~d4】
  stock_max_rebound_{d}   : 日内最大冲高回落幅度（正值越大=冲高回落越严重）
  stock_max_bounce_{d}    : 日内最大反弹幅度（正值越大=低点反弹越强）

【强弱对称类 - 5分钟K线，d0~d4】
  stock_bull_extreme_{d}  : 阳线极值占比 = 阳线最大实体 / 日内总振幅
  stock_bear_extreme_{d}  : 阴线极值占比 = 阴线最大实体 / 日内总振幅
  stock_bull_bear_ratio_{d}: 多空极值比 = max(阳极值,阴极值) / min(阳极值,阴极值)
                             一方为0时填 1（中性）

【跟风指数 - d0 only，板块内主动性得分 [0,1]】
  stock_follower_score_d0 : 1=完全主导，0=完全跟风
                            时间维度+振幅维度各50%，以同板块股票成交金额加权排名

【追高成功率 - d0 only，近20日 [0,1]】
  stock_chase_success_d0  : 近20日中，昨日涨幅>3%且次日收盘>昨日均价的天数占比

【低吸成功率 - d0 only，近20日 [0,1]】
  stock_dip_success_d0    : 近20日中，昨日跌幅>3%且次日收盘>昨日均价的天数占比

【个股强势因子 - d0 only，近20日 [0,1]】
  stock_strength_d0       : 近20日阳线(close>open)占比，指数下跌时阳线权重×2

设计规则：
  - 价格：全部不复权
  - 成交量 → 成交金额（amount），禁止使用换手率
  - 连续型因子全部执行 1%/99% 分位缩尾处理
  - 分母为0时填中性值0；一字板/停牌时位置类填0.5；相关系数方差为0填0
  - 无分钟线数据时，分钟依赖因子填对应中性值
  - 无未来函数：所有因子仅使用 T 日及之前数据
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from features.base_feature import BaseFeature
from features.data_bundle import FeatureDataBundle
from features.feature_registry import feature_registry
from utils.common_tools import ensure_ths_daily_data
from utils.db_utils import db
from utils.log_utils import logger

# ── 缩尾分位数 ──────────────────────────────────────────────
_WINSOR_LOW  = 0.01
_WINSOR_HIGH = 0.99

# ── 高点维持阈值（收盘 ≥ 当日最高 × 0.90）───────────────────
_HIGH_HOLD_THRESHOLD = 0.90

# ── 尾盘时段定义（14:30:00 <= time <= 15:00:00）─────────────
_TAIL_START_HHMM = (14, 30)
_TAIL_END_HHMM   = (15,  0)


# ============================================================
# 辅助函数
# ============================================================

def _winsorize_series(s: pd.Series) -> pd.Series:
    """1%/99% 分位缩尾处理，返回原 Series（就地）"""
    if s.empty or s.isna().all():
        return s
    lo = s.quantile(_WINSOR_LOW)
    hi = s.quantile(_WINSOR_HIGH)
    return s.clip(lo, hi)


def _safe_ratio(num: float, denom: float, neutral: float = 0.0) -> float:
    """除法安全兜底"""
    return float(num / denom) if abs(denom) > 1e-9 else neutral


def _compute_daily_factors(daily: dict) -> Dict[str, float]:
    """
    仅依赖日线 OHLC 计算的原子因子（单股单日）

    输入 daily 必须包含 open/high/low/close/pre_close。
    返回 dict，key = 因子名（不含 _d? 后缀）。
    """
    eps = 1e-9
    o  = float(daily.get("open", 0) or 0)
    h  = float(daily.get("high", 0) or 0)
    lo = float(daily.get("low", 0) or 0)
    c  = float(daily.get("close", 0) or 0)
    pc = float(daily.get("pre_close", 0) or 0)

    if pc <= 0:
        return {
            "intra_amp":   0.0,
            "body_amp":    0.0,
            "body_ratio":  0.5,
            "open_pos":    0.5,
        }

    rng = h - lo          # 日内总振幅（价格空间）
    body = abs(c - o)     # 实体绝对值

    intra_amp  = _safe_ratio(rng,  pc, 0.0)
    body_amp   = _safe_ratio(body, pc, 0.0)
    body_ratio = _safe_ratio(body, rng, 0.5) if rng > eps else 0.5
    open_pos   = _safe_ratio(o - lo, rng, 0.5) if rng > eps else 0.5

    return {
        "intra_amp":  float(intra_amp),
        "body_amp":   float(body_amp),
        "body_ratio": float(body_ratio),
        "open_pos":   float(open_pos),
    }


def _compute_minute_factors(
        minute_df: pd.DataFrame,
        day_high: float,
        day_low: float,
        pre_close: float,
) -> Dict[str, float]:
    """
    单次遍历5分钟K线，计算所有分钟依赖因子（单股单日）。

    :param minute_df: 当日5分钟K线 DataFrame
                      必须含列: trade_time, open, high, low, close, amount
    :param day_high:  当日日线最高价
    :param day_low:   当日日线最低价
    :param pre_close: 前收盘价
    :return: dict of factor values
    """
    # ── 无分钟线：返回中性值 ───────────────────────────────────
    neutral = {
        "vwap_pos":         0.5,
        "up_vol_ratio":     0.0,
        "dn_vol_ratio":     0.0,
        "vol_conc":         0.0,
        "high_hold_ratio":  0.5,
        "buy_vol_ratio":    1.0,
        "pv_corr":          0.0,
        "tail_surge_ratio": 1.0,
        "max_rebound":      0.0,
        "max_bounce":       0.0,
        "bull_extreme":     0.0,
        "bear_extreme":     0.0,
        "bull_bear_ratio":  1.0,
    }
    if minute_df is None or minute_df.empty:
        return neutral

    # ── 排序 + 提取数组 ──────────────────────────────────────────
    df = minute_df.sort_values("trade_time").reset_index(drop=True)

    # 必须列检查
    for col in ["open", "close", "high", "low", "amount"]:
        if col not in df.columns:
            return neutral

    open_arr   = df["open"].values.astype(float)
    close_arr  = df["close"].values.astype(float)
    high_arr   = df["high"].values.astype(float)
    low_arr    = df["low"].values.astype(float)
    amount_arr = df["amount"].values.astype(float)

    eps   = 1e-9
    n     = len(close_arr)
    rng_d = day_high - day_low   # 日内总振幅（日线口径）

    # ── 1. VWAP 位置比 ─────────────────────────────────────────
    # 用分钟成交金额/分钟收盘价得到分钟成交股数（量），再累计
    # vwap = Σ(amount) / Σ(amount / close)
    minute_vol = np.where(close_arr > eps, amount_arr / close_arr, 0.0)
    total_amt  = float(amount_arr.sum())
    total_vol  = float(minute_vol.sum())
    vwap       = _safe_ratio(total_amt, total_vol, (day_high + day_low) / 2.0) if total_vol > eps else (day_high + day_low) / 2.0
    vwap_pos   = _safe_ratio(vwap - day_low, rng_d, 0.5) if rng_d > eps else 0.5

    # ── 2/3/4. 波动结构类（单次遍历 open/close）─────────────────
    body_arr    = close_arr - open_arr            # >0=阳线实体, <0=阴线实体
    body_abs    = np.abs(body_arr)

    is_yang     = body_arr > 0
    is_yin      = body_arr < 0

    # 上行波动占比：阳线实体之和 / 日内总振幅
    up_sum      = float(body_arr[is_yang].sum()) if is_yang.any() else 0.0
    up_vol_ratio = _safe_ratio(up_sum, rng_d, 0.0) if rng_d > eps else 0.0

    # 下行波动占比：阴线实体绝对值之和 / 日内总振幅
    dn_sum       = float((-body_arr[is_yin]).sum()) if is_yin.any() else 0.0
    dn_vol_ratio = _safe_ratio(dn_sum, rng_d, 0.0) if rng_d > eps else 0.0

    # 波动集中度：最大单根K线实体 / 日内总振幅
    max_body     = float(body_abs.max()) if n > 0 else 0.0
    vol_conc     = _safe_ratio(max_body, rng_d, 0.0) if rng_d > eps else 0.0

    # ── 5. 高点维持时间占比 ─────────────────────────────────────
    high_threshold = day_high * _HIGH_HOLD_THRESHOLD
    high_hold_cnt  = int((close_arr >= high_threshold).sum())
    high_hold_ratio = _safe_ratio(high_hold_cnt, n, 0.5) if n > 0 else 0.5

    # ── 6. 放量上涨比例 ─────────────────────────────────────────
    # 全阳（无阴线）→ 分母为0：返回阳线成交金额 / 全日均额 作为相对放量倍数
    # 全阴（无阳线）→ 分子为0：返回0（无买盘能量）
    yang_amt  = amount_arr[is_yang]
    yin_amt   = amount_arr[is_yin]
    yang_mean = float(yang_amt.mean()) if len(yang_amt) > 0 else 0.0
    yin_mean  = float(yin_amt.mean())  if len(yin_amt)  > 0 else 0.0
    if yin_mean > eps:
        buy_vol_ratio = yang_mean / yin_mean
    elif yang_mean > eps:
        # 全阳无阴：用阳线均额 / 全日均额，反映"多头完全主导"程度
        all_mean = float(amount_arr.mean()) if n > 0 else 1.0
        buy_vol_ratio = yang_mean / max(all_mean, eps)
    else:
        buy_vol_ratio = 1.0   # 全停牌或数据缺失，中性

    # ── 7. 量价同步性（皮尔逊相关）──────────────────────────────
    pv_corr = 0.0
    if n >= 2:
        std_c = float(np.std(close_arr))
        std_a = float(np.std(amount_arr))
        if std_c > eps and std_a > eps:
            pv_corr = float(np.corrcoef(close_arr, amount_arr)[0, 1])
            if np.isnan(pv_corr):
                pv_corr = 0.0
        # else: 方差为0，保持 0.0（中性）

    # ── 8. 尾盘量能突袭比 ─────────────────────────────────────
    tail_surge_ratio = 1.0
    if "trade_time" in df.columns:
        times_dt = pd.to_datetime(df["trade_time"], errors="coerce")
        hour_min = times_dt.dt.hour * 60 + times_dt.dt.minute
        tail_start = _TAIL_START_HHMM[0] * 60 + _TAIL_START_HHMM[1]   # 870
        tail_end   = _TAIL_END_HHMM[0]   * 60 + _TAIL_END_HHMM[1]     # 900
        tail_mask  = (hour_min >= tail_start) & (hour_min <= tail_end)
        non_tail_mask = ~tail_mask

        tail_mean     = float(amount_arr[tail_mask].mean())     if tail_mask.sum() > 0 else 0.0
        non_tail_mean = float(amount_arr[non_tail_mask].mean()) if non_tail_mask.sum() > 0 else 0.0
        tail_surge_ratio = _safe_ratio(tail_mean, non_tail_mean, 1.0)

    # ── 9. 日内最大冲高回落幅度 ─────────────────────────────────
    # 按规格："至今累计最高价 - 该时间点之后的全量最低价"
    # suffix_min[i] = min(low_arr[i+1:]) 严格取当前点之后，不含自身
    # 最后一根K线之后无数据 → suffix_min[-1]=+inf → rebound=负→clip为0
    max_rebound = 0.0
    if n > 1 and pre_close > eps:
        cum_high = np.maximum.accumulate(high_arr)
        # 构造严格后缀最小值（i+1 起）
        suffix_tail = np.minimum.accumulate(low_arr[1:][::-1])[::-1]    # len n-1
        suffix_min_strict = np.concatenate([suffix_tail, [np.inf]])      # len n
        rebound_arr = np.maximum(0.0, (cum_high - suffix_min_strict) / pre_close)
        max_rebound = float(rebound_arr.max())

    # ── 10. 日内最大反弹幅度 ─────────────────────────────────────
    # 按规格："该时间点之后的全量最高价 - 至今累计最低价"
    # suffix_max[i] = max(high_arr[i+1:]) 严格取当前点之后，不含自身
    # 最后一根K线之后无数据 → suffix_max[-1]=-inf → bounce=负→clip为0
    max_bounce = 0.0
    if n > 1 and pre_close > eps:
        cum_low = np.minimum.accumulate(low_arr)
        # 构造严格后缀最大值（i+1 起）
        suffix_head = np.maximum.accumulate(high_arr[1:][::-1])[::-1]    # len n-1
        suffix_max_strict = np.concatenate([suffix_head, [-np.inf]])      # len n
        bounce_arr = np.maximum(0.0, (suffix_max_strict - cum_low) / pre_close)
        max_bounce = float(bounce_arr.max())

    # ── 11/12/13. 强弱对称类 ─────────────────────────────────────
    bull_max   = float(body_arr[is_yang].max())  if is_yang.any() else 0.0
    bear_max   = float((-body_arr[is_yin]).max()) if is_yin.any()  else 0.0

    bull_extreme  = _safe_ratio(bull_max, rng_d, 0.0) if rng_d > eps else 0.0
    bear_extreme  = _safe_ratio(bear_max, rng_d, 0.0) if rng_d > eps else 0.0

    # 多空极值比：max/min，分母为0填1
    b_min = min(bull_max, bear_max)
    b_max = max(bull_max, bear_max)
    bull_bear_ratio = _safe_ratio(b_max, b_min, 1.0) if b_min > eps else 1.0

    return {
        "vwap_pos":         float(np.clip(vwap_pos,         0.0, 1.0)),
        "up_vol_ratio":     float(np.clip(up_vol_ratio,     0.0, 1.0)),
        "dn_vol_ratio":     float(np.clip(dn_vol_ratio,     0.0, 1.0)),
        "vol_conc":         float(np.clip(vol_conc,         0.0, 1.0)),
        "high_hold_ratio":  float(np.clip(high_hold_ratio,  0.0, 1.0)),
        "buy_vol_ratio":    float(max(0.0, buy_vol_ratio)),
        "pv_corr":          float(np.clip(pv_corr,         -1.0, 1.0)),
        "tail_surge_ratio": float(max(0.0, tail_surge_ratio)),
        "max_rebound":      float(max(0.0, max_rebound)),
        "max_bounce":       float(max(0.0, max_bounce)),
        "bull_extreme":     float(np.clip(bull_extreme,     0.0, 1.0)),
        "bear_extreme":     float(np.clip(bear_extreme,     0.0, 1.0)),
        "bull_bear_ratio":  float(max(0.0, bull_bear_ratio)),
    }


def _get_minute_peak(min_df: pd.DataFrame) -> Tuple[int, float, float]:
    """
    从5分钟K线提取 (peak_bar_idx, peak_amplitude, total_amount)。
    peak_amplitude = (累计最高 - 首根开盘) / 首根开盘

    此函数提升到模块级，供 calculate() 预计算全量 peak_cache，
    避免 _compute_follower_score 在每只股票内重复解析相同分钟线。
    """
    if min_df is None or min_df.empty:
        return -1, 0.0, 0.0
    df = min_df.sort_values("trade_time").reset_index(drop=True)
    for col in ["open", "high", "amount"]:
        if col not in df.columns:
            return -1, 0.0, 0.0
    open0 = float(df["open"].iloc[0])
    if open0 <= 1e-9:
        return -1, 0.0, 0.0
    high_arr  = df["high"].values.astype(float)
    amt_arr   = df["amount"].values.astype(float)
    cum_high  = np.maximum.accumulate(high_arr)
    amp_arr   = (cum_high - open0) / open0
    peak_idx  = int(np.argmax(amp_arr))
    peak_amp  = float(amp_arr[peak_idx])
    total_amt = float(amt_arr.sum())
    return peak_idx, peak_amp, total_amt


def _compute_follower_score(
        ts_code: str,
        sector_candidate_map: Dict[str, pd.DataFrame],
        peak_cache: Dict[str, Tuple[int, float, float]],
) -> float:
    """
    个股跟风指数（d0 only）。

    评分逻辑：在同板块内，以其他股票的 d0 成交金额为权重，
    计算目标股在「时间领先性」和「振幅领先性」两个维度上的加权排名百分位。

    最终得分 = (时间维度加权得分 + 振幅维度加权得分) / 2，范围 [0, 1]。
    1 = 板块内最主动领先，0 = 完全跟风末尾。

    :param peak_cache: 由 calculate() 预计算，key=ts_code，value=(peak_idx, peak_amp, total_amount)
                       避免对每只股票重复解析相同分钟线（性能关键）
    """
    # ── 找目标股所属板块 ─────────────────────────────────────────
    target_sectors = [
        name for name, df in sector_candidate_map.items()
        if not df.empty and ts_code in df["ts_code"].values
    ]
    if not target_sectors:
        return 0.5   # 不在任何候选板块内，中性值

    t_idx, t_amp, _ = peak_cache.get(ts_code, (-1, 0.0, 0.0))
    if t_idx < 0:
        return 0.5   # 目标股无分钟线，中性值

    scores: List[float] = []

    for sector_name in target_sectors:
        sector_df  = sector_candidate_map[sector_name]
        peer_codes = [c for c in sector_df["ts_code"].unique() if c != ts_code]
        if not peer_codes:
            continue

        # 直接从预计算缓存取 peer 数据（O(1) per peer，无分钟线解析开销）
        peer_data: List[Tuple[int, float, float]] = []
        for peer in peer_codes:
            entry = peak_cache.get(peer)
            if entry is None or entry[0] < 0 or entry[2] <= 0:
                continue
            peer_data.append(entry)

        if not peer_data:
            continue

        # ── 成交金额加权排名 ────────────────────────────────────
        total_peer_amt = sum(p[2] for p in peer_data)
        if total_peer_amt <= 0:
            continue

        timing_wt_lead = sum(p[2] for p in peer_data if p[0] > t_idx)
        amp_wt_lead    = sum(p[2] for p in peer_data if p[1] < t_amp)

        timing_score   = timing_wt_lead / total_peer_amt
        amp_score      = amp_wt_lead    / total_peer_amt
        scores.append((timing_score + amp_score) / 2.0)

    if not scores:
        return 0.5

    return float(np.mean(scores))


def _compute_behavioral_factors(
        ts_code: str,
        trade_date: str,
        lookback_dates_20d: List[str],
        daily_grouped: Dict[tuple, dict],
        hist_sh_pct_chg: Dict[str, float],
        concept_pct_map: Optional[Dict[Tuple[str, str], float]] = None,
        stock_concepts: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    d0 only 的 20 日回看行为因子：追高成功率、低吸成功率、个股强势因子。

    - 追高成功率：近20日中，「昨日涨幅>3%」的次日收盘>昨日均价的占比
    - 低吸成功率：近20日中，「昨日跌幅>3%」的次日收盘>昨日均价的占比
    - 个股强势因子：近20日阳线（close>pre_close）加权占比
        权重规则：大盘跌超0.1% → 1.5；个股涨幅>最高板块均涨幅×2 → 2（取最大值）

    「昨日均价」= prev_day.amount / (prev_day.volume * 100)，单位元/股
    均价计算失败时降级用 prev_day.close 作为成本参考。
    """
    eps = 1e-9
    dates_asc = sorted(lookback_dates_20d)  # 升序排列

    chase_hits   = 0.0   # 昨日>3%，次日胜
    chase_total  = 0.0   # 昨日>3%的次数
    dip_hits     = 0.0   # 昨日<-3%，次日胜
    dip_total    = 0.0   # 昨日<-3%的次数
    strength_num = 0.0   # 加权阳线天数
    strength_den = 0.0   # 加权总天数

    for i in range(1, len(dates_asc)):
        prev_date = dates_asc[i - 1]
        curr_date = dates_asc[i]

        prev_day = daily_grouped.get((ts_code, prev_date))
        curr_day = daily_grouped.get((ts_code, curr_date))

        if not prev_day or not curr_day:
            continue

        prev_pct  = float(prev_day.get("pct_chg", 0) or 0)
        curr_o    = float(curr_day.get("open",  0) or 0)
        curr_c    = float(curr_day.get("close", 0) or 0)

        # ── 昨日均价（用作次日浮盈判断基准）────────────────────
        prev_amt = float(prev_day.get("amount", 0) or 0)
        prev_vol = float(prev_day.get("volume", 0) or 0)
        if prev_vol > eps:
            prev_vwap = prev_amt / (prev_vol * 100.0)
        else:
            prev_vwap = float(prev_day.get("close", curr_c) or curr_c)

        next_positive = curr_c > prev_vwap   # 次日收盘 > 昨日均价

        # ── 追高成功率 ─────────────────────────────────────────
        if prev_pct > 3.0:
            chase_total += 1.0
            if next_positive:
                chase_hits += 1.0

        # ── 低吸成功率 ─────────────────────────────────────────
        if prev_pct < -3.0:
            dip_total += 1.0
            if next_positive:
                dip_hits += 1.0

        # ── 个股强势因子：阳线（close>pre_close）加权占比 ─────────
        prev_close_val = float(prev_day.get("close", 0) or 0)
        is_yang  = curr_c > prev_close_val if prev_close_val > 0 else False
        sh_pct   = hist_sh_pct_chg.get(curr_date, 0.0)
        curr_pct = float(curr_day.get("pct_chg", 0.0) or 0.0)

        # 基础权重：大盘下跌超 0.1% → 1.5，否则 1.0
        weight = 1.5 if sh_pct < -0.1 else 1.0

        # 板块维度：个股涨幅 > 对应最强板块平均涨幅 × 2 → 权重升至 2.0
        if concept_pct_map and stock_concepts:
            max_concept_pct = max(
                (concept_pct_map.get((c, curr_date), 0.0) for c in stock_concepts),
                default=0.0,
            )
            if max_concept_pct > 0 and curr_pct > max_concept_pct * 2:
                weight = 2.0

        strength_den += weight
        if is_yang:
            strength_num += weight

    chase_success  = _safe_ratio(chase_hits,   chase_total,  0.5)
    dip_success    = _safe_ratio(dip_hits,     dip_total,    0.5)
    strength       = _safe_ratio(strength_num, strength_den, 0.5)

    return {
        "chase_success": float(np.clip(chase_success, 0.0, 1.0)),
        "dip_success":   float(np.clip(dip_success,   0.0, 1.0)),
        "strength":      float(np.clip(strength,       0.0, 1.0)),
    }


# ============================================================
# 主特征类
# ============================================================

@feature_registry.register("individual")
class IndividualFeature(BaseFeature):
    """
    个股精细因子集（振幅 / 位置 / 波动结构 / 量价配合 / 风险压力 / 强弱对称 / 行为因子）

    注册名: individual
    输出: 含 stock_code + trade_date 的宽表，每行一个股票（D 日截面）
    """

    feature_name = "individual"

    _day_tags = [f"d{i}" for i in range(5)]

    # 声明所有输出列（与实际 calculate 中构建的列名必须一致）
    factor_columns = [
        # ── 振幅 (d0-d4) ───────────────────────────────────────
        *[f"stock_intra_amp_{t}"     for t in _day_tags],
        *[f"stock_body_amp_{t}"      for t in _day_tags],
        *[f"stock_body_ratio_{t}"    for t in _day_tags],
        # ── 位置 (d0-d4) ───────────────────────────────────────
        *[f"stock_open_pos_{t}"      for t in _day_tags],
        *[f"stock_vwap_pos_{t}"      for t in _day_tags],
        # ── 波动结构 (d0-d4) ───────────────────────────────────
        *[f"stock_up_vol_ratio_{t}"  for t in _day_tags],
        *[f"stock_dn_vol_ratio_{t}"  for t in _day_tags],
        *[f"stock_vol_conc_{t}"      for t in _day_tags],
        # ── 高点维持 (d0-d4) ───────────────────────────────────
        *[f"stock_high_hold_ratio_{t}" for t in _day_tags],
        # ── 量价配合 (d0-d4) ───────────────────────────────────
        *[f"stock_buy_vol_ratio_{t}" for t in _day_tags],
        *[f"stock_pv_corr_{t}"       for t in _day_tags],
        *[f"stock_tail_surge_ratio_{t}" for t in _day_tags],
        # ── 风险压力 (d0-d4) ───────────────────────────────────
        *[f"stock_max_rebound_{t}"   for t in _day_tags],
        *[f"stock_max_bounce_{t}"    for t in _day_tags],
        # ── 强弱对称 (d0-d4) ───────────────────────────────────
        *[f"stock_bull_extreme_{t}"  for t in _day_tags],
        *[f"stock_bear_extreme_{t}"  for t in _day_tags],
        *[f"stock_bull_bear_ratio_{t}" for t in _day_tags],
        # ── D0-only 行为因子 ────────────────────────────────────
        "stock_follower_score_d0",
        "stock_chase_success_d0",
        "stock_dip_success_d0",
        "stock_strength_d0",
    ]

    def calculate(self, data_bundle: FeatureDataBundle) -> tuple:
        """
        单日全量个股精细因子计算。

        :param data_bundle: 预加载的数据容器
        :return: (feature_df, factor_dict)
                 feature_df: stock_code + trade_date + factor columns
                 factor_dict: 空字典（保持接口一致）
        """
        trade_date     = data_bundle.trade_date
        target_codes   = data_bundle.target_ts_codes
        daily_grouped  = data_bundle.daily_grouped
        minute_cache   = data_bundle.minute_cache
        lookback_5d    = data_bundle.lookback_dates_5d   # 升序，[-1]=trade_date
        lookback_20d   = data_bundle.lookback_dates_20d  # 升序，[-1]=trade_date
        sector_map     = data_bundle.sector_candidate_map
        macro_cache    = data_bundle.macro_cache

        # d0=trade_date, d1=前1日, ..., d4=前4日
        # lookback_5d 升序，所以 d0=lookback_5d[-1], d4=lookback_5d[0]
        day_tag_dates: Dict[str, str] = {
            f"d{4 - i}": date for i, date in enumerate(lookback_5d)
        }

        # 20d 历史指数涨跌幅（factor 11 用）
        hist_sh_pct_chg: Dict[str, float] = macro_cache.get("hist_sh_pct_chg", {})

        # ── 板块涨幅映射（ths_daily + ths_member，个股强势因子板块维度用）────────────────────────────
        # 来源：THS 同花顺板块指数日行情（pct_change），比 concept_tags 更准确鲜活
        # 链路：ensure_ths_daily_data → DB 查 ths_member → DB 查 ths_daily → 构建映射
        concept_pct_map: Dict[Tuple[str, str], float] = {}
        stock_concepts_map: Dict[str, List[str]] = {}
        try:
            # Step 1: 确保 lookback_20d 每日的板块行情数据已入库
            for _d in lookback_20d:
                ensure_ths_daily_data(_d)

            # Step 2: 批量查 ths_member，获取 target_codes 所属板块代码
            _ph_c = ", ".join(["%s"] * len(target_codes))
            _member_rows = db.query(
                f"SELECT ts_code, con_code FROM ths_member "
                f"WHERE con_code IN ({_ph_c}) AND is_new = 'Y'",
                params=tuple(target_codes),
            ) or []
            _concept_codes_needed: set = set()
            for _r in _member_rows:
                stock_concepts_map.setdefault(_r["con_code"], []).append(_r["ts_code"])
                _concept_codes_needed.add(_r["ts_code"])

            # Step 3: 批量查 ths_daily，获取板块各日涨幅
            if _concept_codes_needed:
                _date_params = [
                    _d if "-" in _d else f"{_d[:4]}-{_d[4:6]}-{_d[6:]}"
                    for _d in lookback_20d
                ]
                _ph_d  = ", ".join(["%s"] * len(_date_params))
                _ph_cc = ", ".join(["%s"] * len(_concept_codes_needed))
                _daily_rows = db.query(
                    f"SELECT ts_code, trade_date, pct_change FROM ths_daily "
                    f"WHERE trade_date IN ({_ph_d}) AND ts_code IN ({_ph_cc})",
                    params=tuple(_date_params) + tuple(_concept_codes_needed),
                ) or []
                for _r in _daily_rows:
                    _td = str(_r.get("trade_date", ""))
                    if "-" not in _td:
                        _td = f"{_td[:4]}-{_td[4:6]}-{_td[6:]}"
                    concept_pct_map[(_r["ts_code"], _td)] = float(_r.get("pct_change", 0.0) or 0.0)
        except Exception as _e:
            logger.warning(f"[Individual] ths_daily 板块涨幅映射构建失败（非致命）：{_e}")

        # ── 预计算 D0 分钟线 peak 缓存（性能关键）────────────────
        # _compute_follower_score 需要所有股票的 peak 数据做跨股比较。
        # 若在每只股票的循环内按需解析 peer 的分钟线，代价为 O(N²×T)。
        # 预计算后代价降为 O(N×T) + O(N²)（纯内存比较）。
        d0_date = lookback_5d[-1]  # D0 = 最近交易日
        peak_cache: Dict[str, Tuple[int, float, float]] = {}
        for code in target_codes:
            min_df_d0 = minute_cache.get((code, d0_date), pd.DataFrame())
            peak_cache[code] = _get_minute_peak(min_df_d0)

        result_rows: List[dict] = []

        for ts_code in target_codes:
            row: Dict[str, object] = {
                "stock_code": ts_code,
                "trade_date": trade_date,
            }

            # ── 阶段 1: d0~d4 日线 + 分钟线因子 ──────────────────
            for tag, date in day_tag_dates.items():
                daily  = daily_grouped.get((ts_code, date))
                min_df = minute_cache.get((ts_code, date), pd.DataFrame())

                # 日线缺失 → 全部填中性值
                if not daily:
                    for col in [
                        f"stock_intra_amp_{tag}",     f"stock_body_amp_{tag}",
                        f"stock_body_ratio_{tag}",    f"stock_open_pos_{tag}",
                        f"stock_vwap_pos_{tag}",      f"stock_up_vol_ratio_{tag}",
                        f"stock_dn_vol_ratio_{tag}",  f"stock_vol_conc_{tag}",
                        f"stock_high_hold_ratio_{tag}", f"stock_buy_vol_ratio_{tag}",
                        f"stock_pv_corr_{tag}",       f"stock_tail_surge_ratio_{tag}",
                        f"stock_max_rebound_{tag}",   f"stock_max_bounce_{tag}",
                        f"stock_bull_extreme_{tag}",  f"stock_bear_extreme_{tag}",
                        f"stock_bull_bear_ratio_{tag}",
                    ]:
                        row[col] = 0.0 if "ratio" not in col and "pos" not in col else 0.5
                    continue

                # 日线因子
                d_facs = _compute_daily_factors(daily)
                row[f"stock_intra_amp_{tag}"]   = d_facs["intra_amp"]
                row[f"stock_body_amp_{tag}"]    = d_facs["body_amp"]
                row[f"stock_body_ratio_{tag}"]  = d_facs["body_ratio"]
                row[f"stock_open_pos_{tag}"]    = d_facs["open_pos"]

                # 分钟线因子
                dh  = float(daily.get("high",      0) or 0)
                dl  = float(daily.get("low",       0) or 0)
                pc  = float(daily.get("pre_close", 0) or 0)
                m_facs = _compute_minute_factors(min_df, dh, dl, pc)

                row[f"stock_vwap_pos_{tag}"]        = m_facs["vwap_pos"]
                row[f"stock_up_vol_ratio_{tag}"]    = m_facs["up_vol_ratio"]
                row[f"stock_dn_vol_ratio_{tag}"]    = m_facs["dn_vol_ratio"]
                row[f"stock_vol_conc_{tag}"]        = m_facs["vol_conc"]
                row[f"stock_high_hold_ratio_{tag}"] = m_facs["high_hold_ratio"]
                row[f"stock_buy_vol_ratio_{tag}"]   = m_facs["buy_vol_ratio"]
                row[f"stock_pv_corr_{tag}"]         = m_facs["pv_corr"]
                row[f"stock_tail_surge_ratio_{tag}"] = m_facs["tail_surge_ratio"]
                row[f"stock_max_rebound_{tag}"]     = m_facs["max_rebound"]
                row[f"stock_max_bounce_{tag}"]      = m_facs["max_bounce"]
                row[f"stock_bull_extreme_{tag}"]    = m_facs["bull_extreme"]
                row[f"stock_bear_extreme_{tag}"]    = m_facs["bear_extreme"]
                row[f"stock_bull_bear_ratio_{tag}"] = m_facs["bull_bear_ratio"]

            # ── 阶段 2: D0-only 行为因子 ──────────────────────────

            # 因子 8：跟风指数（使用预计算的 peak_cache，避免 O(N²) 重复解析）
            try:
                row["stock_follower_score_d0"] = _compute_follower_score(
                    ts_code, sector_map, peak_cache
                )
            except Exception as e:
                logger.debug(f"[Individual] {ts_code} 跟风指数计算失败: {e}")
                row["stock_follower_score_d0"] = 0.5

            # 因子 9/10/11：20 日行为因子
            try:
                beh = _compute_behavioral_factors(
                    ts_code, trade_date, lookback_20d, daily_grouped, hist_sh_pct_chg,
                    concept_pct_map=concept_pct_map,
                    stock_concepts=stock_concepts_map.get(ts_code),
                )
                row["stock_chase_success_d0"] = beh["chase_success"]
                row["stock_dip_success_d0"]   = beh["dip_success"]
                row["stock_strength_d0"]      = beh["strength"]
            except Exception as e:
                logger.debug(f"[Individual] {ts_code} 行为因子计算失败: {e}")
                row["stock_chase_success_d0"] = 0.5
                row["stock_dip_success_d0"]   = 0.5
                row["stock_strength_d0"]      = 0.5

            result_rows.append(row)

        if not result_rows:
            logger.warning(f"[Individual] {trade_date} 无结果行")
            return pd.DataFrame(), {}

        feature_df = pd.DataFrame(result_rows)

        # ── 阶段 3: 缩尾处理（以 factor_columns 为准，避免意外列）──────────
        winsorize_cols = [c for c in self.factor_columns if c in feature_df.columns]
        for col in winsorize_cols:
            feature_df[col] = _winsorize_series(feature_df[col].astype(float))

        logger.info(
            f"[Individual] {trade_date} 计算完成 | "
            f"股票数:{len(feature_df)} | 因子列:{len(winsorize_cols)}"
        )
        return feature_df, {}
