"""
SEI / HDI 情绪因子
==================
HDI (Holding Difficulty Index，持股难度指数)
    衡量日内持仓体验的合成分，取值 0-100，越高越煎熬

SEI (Sentiment Emotion Index，情绪强度指数)
    多头情绪强度合成分，取值 0-100，越高多头越强

_calculate_minute_hdi 返回的 factors dict（全部原子因子）：
    ret_intra      : 涨跌幅，相对前收
    gap_return     : 开盘缺口率（高开>0，低开<0）★ 独立输出
    cpr            : 收盘位置比 = (close-low)/(high-low)  ★ 独立输出
    max_dd_intra   : 日内最大回撤  ★ 独立输出
    upper_shadow   : 上影线比率    ★ 独立输出
    lower_shadow   : 下影线比率    ★ 独立输出
    trend_r2       : 分钟线趋势 R²  ★ 独立输出
    vwap_deviation : VWAP 偏离度    ★ 独立输出
    candle_type    : K 线结构 {2=真阳,1=假阳,-1=假阴,-2=真阴}  ★ 独立输出
    break_times    : 涨停开板次数   ★ 独立输出
    seal_times     : 涨停封板次数   ★ 独立输出
    lift_times     : 跌停翘板次数   ★ 独立输出
    day_close      : 当日收盘价（内部用于涨停判断，不输出到 CSV）

candle_type 四分类定义：
    真阳( 2): close > open  且  close > pre_close  → 阳线+实涨，最强多头信号
    假阳( 1): close > open  且  close < pre_close  → 阳线但实跌（高开分配），弱势
    假阴(-1): close < open  且  close > pre_close  → 阴线但实涨（低开高走），强势
    真阴(-2): close < open  且  close < pre_close  → 阴线+实跌，最强空头信号

为什么以上原子因子要单独输出（不只嵌入 SEI/HDI）：
    XGBoost 对每个特征有独立分裂阈值，嵌入合成分会让模型失去对单维度的感知。
    例如：假阴（低开高走）是次日大涨的强预测子，但嵌入 SEI 后会被涨跌幅稀释。
    gap_return 高开/低开是主力意图的独立信号，与 pct_chg 不是同一个维度。

无分钟线时的回退策略（calc_daily_atomic）：
    从日线 OHLC 计算 gap_return/candle_type/cpr/上下影线等可得因子，
    无法计算的维度（max_dd/trend_r2/vwap_dev）填充保守中性值，
    保证 sei_cache 在任何情况下都有完整 factors，不出现 KeyError。

本模块不注册到 FeatureRegistry，仅作 SectorStockFeature 内部工具。
"""

import math
import numpy as np
import pandas as pd

from features.base_feature import BaseFeature
from utils.common_tools import calc_limit_up_price, calc_limit_down_price


# ============================================================
# HDI 权重配置
# ============================================================
# 调整说明：
#   各权重无需加总为 1（最终 hdi_raw × 100 映射到 0-100）
#   增大权重 → 该维度对 HDI 影响更显著
# 校准参考：
#   涨停板次日高开股票 HDI 目标范围：20-40（持仓舒适）
#   大幅冲高回落股票 HDI 目标范围：60-80（持仓煎熬）
# ============================================================
HDI_WEIGHTS = {
    "amp":           0.20,  # 日内振幅 = (最高-最低)/前收
    "max_dd_intra":  0.30,  # 日内最大回撤（最重要，权重最大）
    "pullback_abs":  0.20,  # 收盘离日高距离（冲高回落程度）
    "ret_abs":       0.10,  # 涨跌幅绝对值（大涨大跌都增加换手冲动）
    "cross_times":   0.10,  # VWAP 穿越次数（震荡程度）
    "divergence":    0.10,  # 量价背离率（涨价缩量/跌价放量的分钟数占比）
}

# K 线结构对 HDI 的修正（叠加到 hdi_raw，然后再 ×100）
# 建议每项幅度控制在 -0.05 ~ +0.05，避免主导整体评分
CANDLE_HDI_ADJUST = {
    2:  -0.04,   # 真阳：持仓最舒适，略降 HDI
    1:   0.03,   # 假阳：高开实跌，持仓比外观更煎熬
    -1: -0.03,   # 假阴：低开实涨，持仓比外观更舒适
    -2:  0.05,   # 真阴：持仓最煎熬，略升 HDI
}

# ============================================================
# SEI 参数配置
# ============================================================
# 基础分：base = 50 + 50 × tanh(tanh_k × ret)
# tanh_k 越大斜率越陡，中等涨幅更快达到满/低分
# tanh_k=10 参考值：+10%→~91，+5%→~73，±0%→50，-5%→~27
SEI_PARAMS = {
    "tanh_k":      10,    # 推荐范围：5（平滑）~ 15（陡峭）

    # 调整项系数（单位：分/单位偏离）
    "gap_coeff":   20,    # 缺口贡献：gap × 20（约 ±0.5~2 分）
    "trend_coeff": 15,    # 趋势贡献：(R²-0.5) × 15（约 ±7.5 分）
    "vwap_coeff":  20,    # VWAP 减分：-偏离 × 20（约 -0.2~3 分）

    # K 线结构固定加减分（按实际涨跌方向，不按 K 线颜色）
    "candle_adjust": {
        2:   8,    # 真阳：最强多头
        -1:  3,    # 假阴：低开高走，强势
        1:  -3,    # 假阳：高开跌破前收，弱势
        -2: -8,    # 真阴：最强空头
    },

    # 涨跌停调整（以收盘是否在涨停位区分封板加分/开板减分）
    "seal_bonus_per":    2,    # 每次封板加分
    "break_penalty_per": 3,    # 每次开板减分
    "seal_max_bonus":    6,    # 封板加分上限
    "break_max_penalty": 10,   # 开板减分上限
}


class SEIFeature(BaseFeature):
    """
    SEI / HDI 情绪因子计算工具类
    ⚠️ 不注册到 FeatureRegistry，仅作 SectorStockFeature 内部工具
    """

    feature_name = "sei_emotion"

    # ------------------------------------------------------------------ #
    # 静态工具：K 线结构分类
    # ------------------------------------------------------------------ #

    @staticmethod
    def classify_candle(open_price: float, close_price: float, pre_close: float) -> int:
        """
        K 线结构分类（仅需日线 OHLC，不依赖分钟线）
        :return: 2=真阳, 1=假阳, 0=平盘/十字星, -1=假阴, -2=真阴

        平盘/十字星单独归 0（中性，不引入方向性偏差）：
          - close ≈ pre_close（平盘）：价格无变化，无信号
          - close ≈ open（十字星）：多空拮抗，无明确方向
        """
        eps = 1e-6
        # 平盘：收盘价与前收接近 → 无方向信号，返回 0（中性）
        if abs(close_price - pre_close) < eps:
            return 0
        # 十字星：收盘价与开盘接近（多空均衡）→ 返回 0（中性）
        if abs(close_price - open_price) < eps:
            return 0
        is_yang = close_price > open_price    # 收 > 开 = 阳线
        is_up   = close_price > pre_close     # 收 > 前收 = 实涨
        if is_yang and is_up:    return  2    # 真阳：阳线+实涨
        elif is_yang:            return  1    # 假阳：阳线但实跌
        elif is_up:              return -1    # 假阴：阴线但实涨
        else:                    return -2    # 真阴：阴线+实跌

    # ------------------------------------------------------------------ #
    # 无分钟线时的日线回退计算
    # ------------------------------------------------------------------ #

    @staticmethod
    def calc_daily_atomic(
            open_price:  float,
            high_price:  float,
            low_price:   float,
            close_price: float,
            pre_close:   float,
    ) -> dict:
        """
        仅用日线 OHLC 计算可得的原子因子
        无法从日线获取的维度（max_dd/trend_r2/vwap_dev）填充保守中性值

        :return: 与 _calculate_minute_hdi 的 factors 格式一致的 dict
                 保证 sei_cache 在无分钟线时也有完整字段，不出现 KeyError
        """
        eps = 1e-6
        rng = high_price - low_price + eps

        gap_return   = (open_price  / (pre_close + eps)) - 1
        candle_type  = SEIFeature.classify_candle(open_price, close_price, pre_close)
        body_top     = max(open_price, close_price)
        body_bottom  = min(open_price, close_price)
        upper_shadow = (high_price   - body_top)    / (pre_close + eps)
        lower_shadow = (body_bottom  - low_price)   / (pre_close + eps)
        cpr          = (close_price  - low_price)   / rng
        ret          = (close_price  / (pre_close + eps)) - 1

        return {
            # 日线可计算
            "ret_intra":               round(ret,          4),
            "gap_return":              round(gap_return,   4),
            "candle_type":             candle_type,
            "upper_shadow":            round(upper_shadow, 4),
            "lower_shadow":            round(lower_shadow, 4),
            "cpr":                     round(cpr,          4),
            # 需要分钟线，填保守中性值
            "max_dd_intra":            0.10,   # 保守中位值
            "trend_r2":                0.50,   # 中性
            "vwap_deviation":          0.01,   # 保守小值
            "break_times":             0,
            "seal_times":              0,
            "lift_times":              0,
            # 时间持续类因子（无分钟线时无法判断，填中性未知值 0.5）
            # 注意：0.5 ≠ 无红盘（无红盘时 session_pm_ratio = -1），此处是真正"不可知"
            "red_time_ratio":          0.5,
            "float_profit_time_ratio": 0.5,
            "red_session_pm_ratio":    0.5,   # 无分钟线 → 未知，0.5 中性
            "float_session_pm_ratio":  0.5,
            "day_close":               close_price,
        }

    # ------------------------------------------------------------------ #
    # 核心：HDI 计算（依赖分钟线）
    # ------------------------------------------------------------------ #

    def _calculate_minute_hdi(
            self,
            minute_df:  pd.DataFrame,
            pre_close:  float,
            up_limit:   float,
            down_limit: float,
            vwap_prev:  float = 0.0,
    ) -> tuple:
        """
        基于分钟线计算 HDI 及全量原子因子

        :param vwap_prev: 昨日 VWAP（元/股），用于计算浮盈持续时间
                          = 昨日 amount / (昨日 vol × 100)
                          为 0 时浮盈持续时间降级为红盘持续时间

        :return: (hdi_score [0-100], factors dict)
                 无分钟线时返回 (50.0, {})
                 调用方应检测 factors 为空，使用 calc_daily_atomic 回退
        """
        if minute_df is None or minute_df.empty:
            return 50.0, {}

        df = minute_df.sort_values("trade_time").reset_index(drop=True)
        close_arr  = df["close"].values.astype(float)
        open_arr   = df["open"].values.astype(float)
        high_arr   = df["high"].values.astype(float)
        low_arr    = df["low"].values.astype(float)
        volume_arr = df["volume"].values.astype(float)

        day_open  = float(open_arr[0])
        day_close = float(close_arr[-1])
        day_high  = float(high_arr.max())
        day_low   = float(low_arr.min())
        eps = 1e-6
        if day_high == day_low:
            day_high += eps

        # [1] 日内振幅
        amp = (day_high - day_low) / (pre_close + eps)

        # [2] 日内最大回撤（从累计高点到当时价的最大跌幅）
        #     涨停板次日是否高开的强预测子
        cummax       = np.maximum.accumulate(close_arr)
        drawdown     = (cummax - close_arr) / (cummax + eps)
        max_dd_intra = float(drawdown.max())

        # [3] 收盘离日高距离（冲高回落程度）
        pullback_abs = abs((day_close - day_high) / (day_high + eps))

        # [4] 涨跌幅
        ret     = (day_close / (pre_close + eps)) - 1
        ret_abs = abs(ret)

        # [5] VWAP 偏离度（筹码散乱程度）
        cum_vol        = np.cumsum(volume_arr)
        cum_amt        = np.cumsum(close_arr * volume_arr)
        vwap_arr       = cum_amt / (cum_vol + eps)
        vwap_deviation = float(np.mean(np.abs(close_arr - vwap_arr) / (vwap_arr + eps)))

        # [6] VWAP 穿越次数（震荡程度），归一化到 0-1
        cross_sign       = np.sign(close_arr - vwap_arr)
        cross_times      = float(np.sum(np.abs(np.diff(cross_sign))) / 2)
        cross_times_norm = min(cross_times / 20.0, 1.0)

        # [7] 量价背离率
        if len(close_arr) > 1:
            ret_min = np.diff(close_arr) / (close_arr[:-1] + eps)
            vol_chg = np.diff(volume_arr) / (volume_arr[:-1] + eps)
            div_cnt = int(np.sum(
                ((ret_min > 0) & (vol_chg < 0)) | ((ret_min < 0) & (vol_chg > 0))
            ))
            diverge_ratio = div_cnt / len(ret_min)
        else:
            diverge_ratio = 0.0

        # [8] 开盘缺口（独立输出：正=高开，负=低开）
        gap_return = (day_open / (pre_close + eps)) - 1

        # [9] 趋势 R²（分钟线线性拟合，越接近 1 趋势越稳定）
        x        = np.arange(len(close_arr), dtype=float)
        coef     = np.polyfit(x, close_arr, 1)
        fitted   = coef[0] * x + coef[1]
        ss_res   = float(np.sum((close_arr - fitted) ** 2))
        ss_tot   = float(np.sum((close_arr - close_arr.mean()) ** 2))
        trend_r2 = max(0.0, 1.0 - ss_res / (ss_tot + eps))

        # [10] CPR（收盘位置比）
        cpr = (day_close - day_low) / (day_high - day_low)

        # [11] 影线比率
        body_top     = max(day_open, day_close)
        body_bottom  = min(day_open, day_close)
        upper_shadow = (day_high   - body_top)    / (pre_close + eps)
        lower_shadow = (body_bottom - day_low)    / (pre_close + eps)

        # [12] K 线结构分类
        candle_type = self.classify_candle(day_open, day_close, pre_close)

        # [13] 涨跌停行为
        touch_up    = high_arr >= (up_limit   - 0.01)
        break_times = int(np.sum(np.diff(touch_up.astype(int)) == -1))
        seal_times  = int(np.sum(np.diff(touch_up.astype(int)) ==  1))
        touch_dn    = low_arr  <= (down_limit + 0.01)
        lift_times  = int(np.sum(np.diff(touch_dn.astype(int)) == -1))

        # [14] 红盘持续时间 & 浮盈持续时间 & 早/午盘偏向
        #   早盘: 9:30~11:30（120分钟），午盘: 13:00~15:00（120分钟）
        #   red_session_pm_ratio / float_session_pm_ratio ∈ [0,1]
        #   0=全在早盘, 0.5=均衡/无红盘, 1=全在午盘
        total_min = len(close_arr)
        if total_min > 0:
            times_dt    = pd.to_datetime(df["trade_time"])
            hour_frac   = times_dt.dt.hour + times_dt.dt.minute / 60.0
            am_mask     = ((hour_frac >= 9.5)  & (hour_frac <= 11.5)).values
            pm_mask     = ((hour_frac >= 13.0) & (hour_frac <= 15.0)).values

            # 红盘（高于昨收）
            red_mask            = close_arr > pre_close
            red_time_ratio      = float(red_mask.sum()) / total_min
            am_red = int((red_mask & am_mask).sum())
            pm_red = int((red_mask & pm_mask).sum())
            _red_tot            = am_red + pm_red
            # 无红盘 → -1（语义最弱，区别于"早盘主导 0"和"均衡 0.5"）
            # 有红盘 → pm比例 ∈ [0,1]：0=全早盘，0.5=均衡，1=全午盘
            red_session_pm_ratio = pm_red / (_red_tot + 1e-9) if _red_tot > 0 else -1.0

            # 浮盈（高于昨日 VWAP）；vwap_prev=0 时降级用昨收
            _vwap_ref           = vwap_prev if vwap_prev and vwap_prev > 0 else pre_close
            float_mask          = close_arr > _vwap_ref
            float_profit_time_ratio = float(float_mask.sum()) / total_min
            am_flt = int((float_mask & am_mask).sum())
            pm_flt = int((float_mask & pm_mask).sum())
            _flt_tot               = am_flt + pm_flt
            float_session_pm_ratio = pm_flt / (_flt_tot + 1e-9) if _flt_tot > 0 else -1.0
        else:
            red_time_ratio         = 0.5
            float_profit_time_ratio = 0.5
            red_session_pm_ratio   = 0.5   # 无分钟线 = 未知，保持中性
            float_session_pm_ratio = 0.5

        # ---- HDI 合成 ----
        w = HDI_WEIGHTS
        hdi_raw = (
            w["amp"]          * amp
            + w["max_dd_intra"] * max_dd_intra
            + w["pullback_abs"] * pullback_abs
            + w["ret_abs"]      * ret_abs
            + w["cross_times"]  * cross_times_norm
            + w["divergence"]   * diverge_ratio
            + CANDLE_HDI_ADJUST[candle_type]
        )
        hdi_score = round(float(np.clip(hdi_raw * 100, 0, 100)), 2)

        factors = {
            "ret_intra":               round(ret,                     4),
            "gap_return":              round(gap_return,              4),
            "cpr":                     round(cpr,                     4),
            "max_dd_intra":            round(max_dd_intra,            4),
            "upper_shadow":            round(upper_shadow,            4),
            "lower_shadow":            round(lower_shadow,            4),
            "trend_r2":                round(trend_r2,                4),
            "vwap_deviation":          round(vwap_deviation,          4),
            "candle_type":             candle_type,
            "break_times":             break_times,
            "seal_times":              seal_times,
            "lift_times":              lift_times,
            # ── 新增：时间持续类因子 ────────────────────────────────────
            "red_time_ratio":          round(red_time_ratio,          4),
            "float_profit_time_ratio": round(float_profit_time_ratio, 4),
            "red_session_pm_ratio":    round(red_session_pm_ratio,    4),
            "float_session_pm_ratio":  round(float_session_pm_ratio,  4),
            # ────────────────────────────────────────────────────────────
            "day_close":               day_close,   # 内部使用，不输出到 CSV
        }
        return hdi_score, factors

    # ------------------------------------------------------------------ #
    # 核心：SEI 计算
    # ------------------------------------------------------------------ #

    def _factors_to_sei(self, factors: dict, up_limit: float) -> float:
        """
        由完整 factors dict 计算 SEI（避免重复调用 _calculate_minute_hdi）
        供 sector_stock_feature 在已有缓存时直接调用
        """
        p    = SEI_PARAMS
        base = 50.0 + 50.0 * math.tanh(p["tanh_k"] * factors["ret_intra"])

        gap_adj    = factors.get("gap_return",     0.0)  * p["gap_coeff"]
        trend_adj  = (factors.get("trend_r2",      0.5) - 0.5) * p["trend_coeff"]
        vwap_adj   = -factors.get("vwap_deviation", 0.0) * p["vwap_coeff"]
        candle_adj = p["candle_adjust"].get(factors.get("candle_type", 0), 0)

        limit_adj = 0.0
        break_n   = factors.get("break_times", 0)
        if break_n > 0:
            at_limit = abs(factors.get("day_close", 0) - up_limit) < 0.02
            if at_limit:
                limit_adj = min(break_n * p["seal_bonus_per"],    p["seal_max_bonus"])
            else:
                limit_adj = -min(break_n * p["break_penalty_per"], p["break_max_penalty"])

        sei = base + gap_adj + trend_adj + vwap_adj + candle_adj + limit_adj
        return round(float(np.clip(sei, 0.0, 100.0)), 2)

    def _calculate_minute_sei(
            self,
            minute_df:  pd.DataFrame,
            pre_close:  float,
            up_limit:   float,
            down_limit: float,
    ) -> float:
        """直接从分钟线计算 SEI（内部调用 HDI，适合单独使用场景）"""
        _, factors = self._calculate_minute_hdi(minute_df, pre_close, up_limit, down_limit)
        if not factors:
            return 50.0
        return self._factors_to_sei(factors, up_limit)

    # ------------------------------------------------------------------ #
    # BaseFeature 抽象接口（单独作为因子时的入口，正常流程不走这里）
    # ------------------------------------------------------------------ #

    def calculate(self, data_bundle) -> tuple:
        rows = []
        for ts_code in data_bundle.target_ts_codes:
            key = (ts_code, data_bundle.trade_date)
            if key not in data_bundle.daily_grouped:
                continue
            daily_row = data_bundle.daily_grouped[key]
            pre_close = daily_row.get("pre_close", 0)
            if not pre_close or pre_close <= 0:
                continue
            minute_df  = data_bundle.minute_cache.get(key, pd.DataFrame())
            up_limit   = calc_limit_up_price(ts_code, pre_close)
            down_limit = calc_limit_down_price(ts_code, pre_close)
            hdi, facts = self._calculate_minute_hdi(minute_df, pre_close, up_limit, down_limit)
            if not facts:
                facts = self.calc_daily_atomic(
                    daily_row.get("open",  pre_close),
                    daily_row.get("high",  pre_close),
                    daily_row.get("low",   pre_close),
                    daily_row.get("close", pre_close),
                    pre_close,
                )
                hdi = 50.0
            sei = self._factors_to_sei(facts, up_limit)
            rows.append({
                "stock_code": ts_code,
                "trade_date": data_bundle.trade_date,
                "hdi_score":  hdi,
                "sei_score":  sei,
                **{k: v for k, v in facts.items() if k != "day_close"},
            })
        return pd.DataFrame(rows), {}