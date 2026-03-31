"""
MA 均线 + 个股位置因子
======================
本模块在 data_bundle 已有的 20 日日线数据上直接计算，
不发起额外 IO（复用计算公式逻辑，数据来源改为 daily_grouped）。

输出列（全部为 D 日截面，无 d0-d4 后缀）：

均线值（保留在 CSV，训练时由 EXCLUDE_PATTERNS 过滤）：
    ma5 / ma10 / ma13           : D 日收盘后的简单移动均线（前 N 日收盘价均值）
                                  绝对价格跨股无可比性，train.py 默认排除；
                                  bias*/pos_20d/ma_align 已携带归一化后的均线信息

乖离率（BIAS）：
    bias5 / bias10 / bias13     : (close_D - MAn) / MAn × 100
                                  正=股价在均线上方，负=在下方
                                  与 candle_type 交叉可识别"低位假阴/高位假阳"

均线斜率：
    ma5_slope                   : (MA5_D - MA5_{D-1}) / MA5_{D-1}
                                  正=短期动能上行，负=动能减弱

均线排列：
    ma_align                    : 多头/空头排列评分
                                   2: MA5 > MA10 > MA13（完美多头排列）
                                   1: MA5 > MA10，MA10 < MA13（弱多头）
                                   0: MA5 = MA10 = MA13 或无法判断
                                  -1: MA5 < MA10，MA10 > MA13（弱空头）
                                  -2: MA5 < MA10 < MA13（完美空头排列）
                                  XGBoost 对此整数列有独立分裂能力

价格位置（量化"当前位置"核心）：
    pos_20d                     : (close_D - low_20d) / (high_20d - low_20d)
                                  0=20日最低价，1=20日最高价
                                  结合 candle_type 可区分"低位假阴强信号"vs"高位假阴陷阱"

    pos_5d                      : (close_D - low_5d) / (high_5d - low_5d)
                                  短期位置，反映近一周是否已超买

    from_high_20d               : (high_20d - close_D) / high_20d
                                  距 20 日最高点的跌幅，量化套牢盘压力
                                  from_high_20d 大 + pos_20d 低 → 深回调后的低位
                                  from_high_20d 小 + pos_20d 高 → 创新高附近的强势位

设计说明：
    - 均线用不复权（bfq）数据：与 kline_day / 分钟线 / stk_factor_pro(_bfq) 保持统一口径。
      振荡类指标（RSI/KDJ/MACD 等）bfq=qfq=hfq，BIAS 计算对短线策略受除权影响极小。
      统一口径消除 d0-d4 特征跨日混用不同复权来源的训练数据污染。
    - data_bundle 的 lookback_dates_20d 已有 20 日，MA13 需要 13 日，数据足够。
    - 无足够历史数据时（停牌/新股）输出中性值并记录 warning，不抛出异常。
"""

from typing import List, Dict

import numpy as np
import pandas as pd

from features.base_feature import BaseFeature
from features.feature_registry import feature_registry
from utils.log_utils import logger


@feature_registry.register("ma_position")
class MAPositionFeature(BaseFeature):
    """
    MA 均线 + 个股位置因子

    注册名：ma_position
    输出类型：个股级（含 stock_code 列）→ FeatureEngine inner join
    """

    feature_name = "ma_position"

    # 本模块关注的均线周期
    MA_PERIODS = [5, 10, 13]

    # ------------------------------------------------------------------ #
    # 均线计算（复用 ma_indicator.py 的 rolling mean 逻辑，数据来源改为内存）
    # ------------------------------------------------------------------ #

    @staticmethod
    def _calc_ma_series(close_series: List[float], period: int) -> float:
        """
        计算 D 日的 MA(period)，即最后 period 个收盘价的均值。
        数据不足 period 时取 min_periods=1（与 ma_indicator.py 行为一致）。
        """
        if not close_series:
            return float("nan")
        window = close_series[-period:] if len(close_series) >= period else close_series
        valid = [v for v in window if v and not np.isnan(v)]
        return round(float(np.mean(valid)), 4) if valid else float("nan")

    @staticmethod
    def _calc_bias(close: float, ma: float) -> float:
        """BIAS = (close - MA) / MA × 100，MA 为 0 或 nan 时返回 0"""
        if not ma or np.isnan(ma) or ma == 0:
            return 0.0
        return round((close - ma) / ma * 100, 4)

    @staticmethod
    def _calc_ma_align(ma5: float, ma10: float, ma13: float) -> int:
        """
        均线排列评分
          2: MA5 > MA10 > MA13（完美多头排列）
          1: MA5 > MA10，MA10 <= MA13（弱多头，短线转强但中线未确认）
          0: 其他 / 无法判断（含 nan）
         -1: MA5 < MA10，MA10 >= MA13（弱空头）
         -2: MA5 < MA10 < MA13（完美空头排列）
        """
        if any(np.isnan(v) for v in [ma5, ma10, ma13]):
            return 0
        if ma5 > ma10 > ma13:
            return 2
        elif ma5 > ma10 and ma10 <= ma13:
            return 1
        elif ma5 < ma10 < ma13:
            return -2
        elif ma5 < ma10 and ma10 >= ma13:
            return -1
        return 0

    # ------------------------------------------------------------------ #
    # 主计算入口
    # ------------------------------------------------------------------ #

    def calculate(self, data_bundle) -> tuple:
        """
        :param data_bundle: FeatureDataBundle，需含 daily_grouped / lookback_dates_20d / target_ts_codes
        :return: (feature_df, {})

        均线原始值/排列/位置类因子继续本地计算；
        bias5/bias10 优先复用 stk_factor_pro API 已入库的 ma_bias_5/10，
        保证与技术面因子统一口径；若 API 字段缺失，再回退到本地计算结果。
        """
        trade_date      = data_bundle.trade_date
        dates_20d       = sorted(data_bundle.lookback_dates_20d)   # 升序，最后一个 = trade_date
        daily_grouped   = data_bundle.daily_grouped                 # 不复权
        target_ts_codes = data_bundle.target_ts_codes

        stk_factor_map = {}
        try:
            from utils.common_tools import ensure_stk_factor_pro_data
            from utils.db_utils import db
            ensure_stk_factor_pro_data(trade_date)
            sql = "SELECT ts_code, ma_bfq_5, ma_bfq_10 FROM stk_factor_pro WHERE trade_date = %s"
            raw = db.query(sql, (trade_date.replace("-", ""),), return_df=True)
            if raw is not None and not raw.empty:
                stk_factor_map = raw.set_index("ts_code").to_dict(orient="index")
        except Exception as e:
            logger.warning(f"[MAPosition] 读取 stk_factor_pro MA 参考值失败，回退本地 bias：{e}")

        if not dates_20d:
            logger.error("[MAPosition] lookback_dates_20d 为空，跳过计算")
            return pd.DataFrame(), {}

        rows = []
        missing = 0

        for ts_code in target_ts_codes:
            # ── 提取 20 日收盘价序列（升序，index -1 = D 日）──
            close_series = []
            for d in dates_20d:
                row = daily_grouped.get((ts_code, d), {})
                v   = row.get("close", None)
                close_series.append(float(v) if v else float("nan"))

            d_close = close_series[-1]  # D 日收盘价

            # D 日无收盘价（停牌），输出中性值
            if not d_close or np.isnan(d_close):
                missing += 1
                rows.append(self._neutral_row(ts_code, trade_date))
                continue

            # ── 均线计算（MA5 / MA10 / MA13）──
            ma5  = self._calc_ma_series(close_series, 5)
            ma10 = self._calc_ma_series(close_series, 10)
            ma13 = self._calc_ma_series(close_series, 13)

            # ── BIAS（乖离率）──
            api_ma = stk_factor_map.get(ts_code, {})
            api_ma5 = pd.to_numeric(api_ma.get("ma_bfq_5"), errors="coerce") if api_ma else np.nan
            api_ma10 = pd.to_numeric(api_ma.get("ma_bfq_10"), errors="coerce") if api_ma else np.nan
            bias5  = self._calc_bias(d_close, float(api_ma5)) if not np.isnan(api_ma5) else self._calc_bias(d_close, ma5)
            bias10 = self._calc_bias(d_close, float(api_ma10)) if not np.isnan(api_ma10) else self._calc_bias(d_close, ma10)
            bias13 = self._calc_bias(d_close, ma13)

            # ── MA5 斜率（今日 MA5 vs 昨日 MA5）──
            ma5_prev  = self._calc_ma_series(close_series[:-1], 5)   # 去掉 D 日后算 D-1 的 MA5
            ma5_slope = 0.0
            if ma5_prev and not np.isnan(ma5_prev) and ma5_prev != 0:
                ma5_slope = round((ma5 - ma5_prev) / ma5_prev, 6)

            # ── 均线排列 ──
            ma_align = self._calc_ma_align(ma5, ma10, ma13)

            # ── 价格位置：20 日区间 ──
            valid_20 = [v for v in close_series if not np.isnan(v)]
            high_20d  = max(valid_20) if valid_20 else d_close
            low_20d   = min(valid_20) if valid_20 else d_close
            rng_20d   = high_20d - low_20d

            pos_20d      = round((d_close - low_20d) / rng_20d, 4) if rng_20d > 0 else 0.5
            from_high_20d = round((high_20d - d_close) / high_20d, 4) if high_20d > 0 else 0.0

            # ── 价格位置：5 日区间 ──
            close_5d  = [v for v in close_series[-5:] if not np.isnan(v)]
            high_5d   = max(close_5d) if close_5d else d_close
            low_5d    = min(close_5d) if close_5d else d_close
            rng_5d    = high_5d - low_5d
            pos_5d    = round((d_close - low_5d) / rng_5d, 4) if rng_5d > 0 else 0.5

            rows.append({
                "stock_code":    ts_code,
                "trade_date":    trade_date,
                # 均线原始价格（保留在 CSV；train.py EXCLUDE_PATTERNS 默认过滤，
                # 若需跨股比较请改用 bias；绝对价格对 XGBoost 跨股无意义）
                "ma5":           ma5,
                "ma10":          ma10,
                "ma13":          ma13,
                # 乖离率
                "bias5":         bias5,
                "bias10":        bias10,
                "bias13":        bias13,
                # 均线动能
                "ma5_slope":     ma5_slope,
                "ma_align":      ma_align,
                # 价格位置
                "pos_20d":       pos_20d,
                "pos_5d":        pos_5d,
                "from_high_20d": from_high_20d,
            })

        feature_df = pd.DataFrame(rows)
        logger.info(
            f"[MAPosition] {trade_date} 计算完成 | 有效:{len(feature_df) - missing} "
            f"| 停牌/无数据填充中性:{missing} | 列数:{len(feature_df.columns)}"
        )
        return feature_df, {}

    @staticmethod
    def _neutral_row(ts_code: str, trade_date: str) -> dict:
        """
        停牌或无数据时的中性填充
        注意：停牌股通常不在候选池，此处为防御性处理
        中性值的语义：
            ma5/10/13 = 0     → 告知模型"无均线数据"（区别于 0 乖离率）
            bias* = 0         → 无偏离
            pos* = 0.5        → 区间中位（中性）
            ma_align = 0      → 无排列信号
            from_high_20d = 0 → 无距离信息
        """
        return {
            "stock_code":    ts_code,
            "trade_date":    trade_date,
            "ma5":           0.0,
            "ma10":          0.0,
            "ma13":          0.0,
            "bias5":         0.0,
            "bias10":        0.0,
            "bias13":        0.0,
            "ma5_slope":     0.0,
            "ma_align":      0,
            "pos_20d":       0.5,
            "pos_5d":        0.5,
            "from_high_20d": 0.0,
        }