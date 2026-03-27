"""
股票技术面因子特征（来源：Tushare stk_factor_pro）
===================================================
数据来源：stk_factor_pro 接口，每个交易日一次拉全市场所有股票。
建表 SQL：sql/create_stk_factor_pro.sql
更新策略：按需更新（ensure_stk_factor_pro_data 确保当日数据存在）

输出列（所有列无绝对价格，跨股可比）：

MACD 系列：
    macd_dif            : DIF 快线（EMA12 - EMA26），正=短期强势
    macd_dea            : DEA 慢线（信号线），正=趋势向上
    macd_bar            : MACD 柱（DIF - DEA），正=金叉趋势
    macd_golden_cross   : 1=DIF 上穿 DEA（金叉），0=未穿越，-1=死叉

KDJ 系列：
    kdj_k               : K 值（快速随机指标）
    kdj_d               : D 值（K 的移动均值，慢速）
    kdj_j               : J 值（可超 0~100，超买超卖敏感指标）
    kdj_overbought      : 1=超买（K>80 且 D>80），-1=超卖（K<20 且 D<20），0=中性

RSI 系列：
    rsi_6               : RSI 6 日（短周期，快速反应）
    rsi_12              : RSI 12 日（中周期）
    rsi_24              : RSI 24 日（长周期）
    rsi_diverge         : rsi_6 - rsi_24，正=短强长弱（超短线强势），负=短弱长强

Bollinger Bands 系列：
    boll_pct            : (close - 下轨) / (上轨 - 下轨)，0=下轨 1=上轨，跨股可比
    boll_width          : (上轨 - 下轨) / 中轨，带宽相对中轨，量化波动扩张/收缩

其他技术指标（直接使用原始值，量纲已天然可比）：
    cci                 : CCI 顺势指标（±100 为超买超卖分界）
    wr                  : W&R 威廉指标（-100~0，靠近 0=超买，靠近 -100=超卖）

已移除列（避免与现有模块重复）：
    bias1/2/3（6/12/24日乖离率）→ 由 ma_position 模块提供 bias5/10/13，语义相同且本地计算更快

设计说明：
    - boll_pct / boll_width 需要 close 价格，从 data_bundle.daily_grouped 取。
    - 所有字段缺失（停牌/首日上市/数据库尚无数据）时输出中性值，不抛异常。
    - macd_golden_cross / kdj_overbought 为离散信号，XGBoost 可直接用。
    - ATR / OBV / VR / MFI 量纲依赖个股绝对价格/成交量，跨股无法直接比较，
      故不纳入本特征输出（如需可扩展）。
"""

from typing import List, Dict

import numpy as np
import pandas as pd

from features.base_feature import BaseFeature
from features.feature_registry import feature_registry
from utils.common_tools import ensure_stk_factor_pro_data
from utils.db_utils import db
from utils.log_utils import logger

# 从 DB 读取的原始列（与 Tushare 实际返回列名一致，_bfq = 不复权口径）
# 振荡类指标（RSI/KDJ/BIAS/CCI/WR/MACD）bfq=hfq=qfq，取 bfq 即可。
# 价格类（Bollinger）用 bfq 与不复权 close 保持同口径。
#
# bias1/2/3（6/12/24日乖离率）已从输出中移除：
#   ma_position 模块已通过本地计算提供 bias5/10/13（5/10/13日），
#   语义相同（乖离率），本地计算快，无需重复引入 API 数据。
#   如需更多乖离率周期，建议在 ma_position 中扩展（MA_PERIODS 配置），
#   而非依赖本模块的 API 数据。
_DB_COLS = [
    "ts_code", "trade_date",
    "macd_dif_bfq", "macd_dea_bfq", "macd_bfq",
    "kdj_k_bfq", "kdj_d_bfq", "kdj_bfq",
    "rsi_bfq_6", "rsi_bfq_12", "rsi_bfq_24",
    "boll_upper_bfq", "boll_mid_bfq", "boll_lower_bfq",
    # "bias1_bfq", "bias2_bfq", "bias3_bfq",  # 已由 ma_position 覆盖，避免重复
    "cci_bfq", "wr_bfq",
]


@feature_registry.register("stk_factor_pro")
class StkFactorProFeature(BaseFeature):
    """
    股票技术面因子特征（Tushare stk_factor_pro）

    注册名：stk_factor_pro
    输出类型：个股级（含 stock_code 列）→ FeatureEngine inner join
    """

    feature_name = "stk_factor_pro"

    # ------------------------------------------------------------------ #
    # 主计算入口
    # ------------------------------------------------------------------ #

    def calculate(self, data_bundle) -> tuple:
        """
        :param data_bundle: FeatureDataBundle，需含 trade_date / target_ts_codes /
                            daily_grouped（取 close，用于 Bollinger 位置归一化）
        :return: (feature_df, {})
        """
        trade_date = data_bundle.trade_date          # yyyy-mm-dd
        target_ts_codes = data_bundle.target_ts_codes
        daily_grouped = data_bundle.daily_grouped    # (ts_code, trade_date) → row dict

        trade_date_fmt = trade_date.replace("-", "")  # YYYYMMDD，用于 API / DB 查询

        # ── 1. 确保当日数据在 DB 中 ──
        try:
            ensure_stk_factor_pro_data(trade_date_fmt)
        except Exception as e:
            logger.warning(f"[StkFactorPro] ensure 失败，尝试用已有 DB 数据继续：{e}")

        # ── 2. 从 DB 批量拉取当日全市场技术因子 ──
        cols_str = ", ".join(_DB_COLS)
        sql = f"SELECT {cols_str} FROM stk_factor_pro WHERE trade_date = %s"
        try:
            raw = db.query(sql, (trade_date_fmt,), return_df=True)
        except Exception as e:
            logger.error(f"[StkFactorPro] DB 查询失败，{trade_date}：{e}")
            raw = pd.DataFrame()

        if raw is None or raw.empty:
            logger.warning(f"[StkFactorPro] {trade_date} DB 无数据，全部填充中性值")
            raw = pd.DataFrame(columns=_DB_COLS)

        # ts_code → row dict，O(1) 查找
        factor_map: Dict[str, dict] = {}
        if not raw.empty:
            for _, row in raw.iterrows():
                factor_map[row["ts_code"]] = row.to_dict()

        # ── 3. 逐股计算特征 ──
        rows = []
        missing = 0

        for ts_code in target_ts_codes:
            frow = factor_map.get(ts_code)
            if frow is None:
                missing += 1
                rows.append(self._neutral_row(ts_code, trade_date))
                continue

            # 取收盘价（用于 Bollinger 位置）
            close_raw = daily_grouped.get((ts_code, trade_date), {}).get("close", None)
            close = float(close_raw) if close_raw and not _is_nan(close_raw) else None

            rows.append(self._compute_row(ts_code, trade_date, frow, close))

        feature_df = pd.DataFrame(rows)
        logger.info(
            f"[StkFactorPro] {trade_date} 计算完成 | 有效:{len(feature_df) - missing} "
            f"| 无数据填充中性:{missing} | 列数:{len(feature_df.columns)}"
        )
        return feature_df, {}

    # ------------------------------------------------------------------ #
    # 单股特征计算
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_row(ts_code: str, trade_date: str, frow: dict, close: float) -> dict:
        """从 DB 行 + 收盘价计算全部特征列。"""

        def _f(key: str) -> float:
            """安全取 float，NaN/None → float('nan')"""
            v = frow.get(key)
            return float(v) if v is not None and not _is_nan(v) else float("nan")

        macd_dif = _f("macd_dif_bfq")
        macd_dea = _f("macd_dea_bfq")
        macd_bar = _f("macd_bfq")        # J 值在 Tushare 中字段名为 kdj_bfq

        kdj_k = _f("kdj_k_bfq")
        kdj_d = _f("kdj_d_bfq")
        kdj_j = _f("kdj_bfq")           # J 值字段名为 kdj_bfq（不带 _j）

        rsi_6  = _f("rsi_bfq_6")
        rsi_12 = _f("rsi_bfq_12")
        rsi_24 = _f("rsi_bfq_24")

        boll_upper = _f("boll_upper_bfq")
        boll_mid   = _f("boll_mid_bfq")
        boll_lower = _f("boll_lower_bfq")

        # bias1/2/3 已移除（由 ma_position 模块覆盖，避免重复）
        cci   = _f("cci_bfq")
        wr    = _f("wr_bfq")

        # ── MACD 金叉/死叉信号 ──
        # 用 macd_bar 符号近似（DIF > DEA → bar > 0 → 金叉区间）
        if np.isnan(macd_bar):
            macd_golden_cross = 0
        elif macd_bar > 0:
            macd_golden_cross = 1
        elif macd_bar < 0:
            macd_golden_cross = -1
        else:
            macd_golden_cross = 0

        # ── KDJ 超买超卖 ──
        if np.isnan(kdj_k) or np.isnan(kdj_d):
            kdj_overbought = 0
        elif kdj_k > 80 and kdj_d > 80:
            kdj_overbought = 1
        elif kdj_k < 20 and kdj_d < 20:
            kdj_overbought = -1
        else:
            kdj_overbought = 0

        # ── RSI 短长发散 ──
        rsi_diverge = round(rsi_6 - rsi_24, 4) if not (np.isnan(rsi_6) or np.isnan(rsi_24)) else 0.0

        # ── Bollinger 位置 (需要 close) ──
        boll_pct   = 0.5   # 中性：区间中点
        boll_width = 0.0
        if close is not None and not np.isnan(boll_upper) and not np.isnan(boll_lower):
            band = boll_upper - boll_lower
            if band > 0:
                boll_pct = round((close - boll_lower) / band, 4)
                boll_pct = max(0.0, min(1.0, boll_pct))  # clip 到 [0,1]
            if not np.isnan(boll_mid) and boll_mid > 0:
                boll_width = round(band / boll_mid, 4)

        return {
            "stock_code":        ts_code,
            "trade_date":        trade_date,
            # MACD
            "macd_dif":          _safe(macd_dif),
            "macd_dea":          _safe(macd_dea),
            "macd_bar":          _safe(macd_bar),
            "macd_golden_cross": macd_golden_cross,
            # KDJ
            "kdj_k":             _safe(kdj_k),
            "kdj_d":             _safe(kdj_d),
            "kdj_j":             _safe(kdj_j),
            "kdj_overbought":    kdj_overbought,
            # RSI
            "rsi_6":             _safe(rsi_6),
            "rsi_12":            _safe(rsi_12),
            "rsi_24":            _safe(rsi_24),
            "rsi_diverge":       rsi_diverge,
            # Bollinger
            "boll_pct":          boll_pct,
            "boll_width":        boll_width,
            # CCI / WR（bias1/2/3 已移除，由 ma_position 覆盖）
            "cci":               _safe(cci),
            "wr":                _safe(wr),
        }

    @staticmethod
    def _neutral_row(ts_code: str, trade_date: str) -> dict:
        """
        无数据时的中性填充（不对模型引入方向性偏差）。
        中性值语义：
            macd_dif/dea/bar = 0    → 无趋势信号
            kdj_k/d/j = 50          → 中性区域
            rsi_* = 50              → 无超买超卖
            boll_pct = 0.5          → 区间中点
            boll_width = 0          → 无波动信息
            bias* / cci / wr = 0    → 无偏离信号
            *_cross / *_overbought  → 0 = 无信号
        """
        return {
            "stock_code":        ts_code,
            "trade_date":        trade_date,
            "macd_dif":          0.0,
            "macd_dea":          0.0,
            "macd_bar":          0.0,
            "macd_golden_cross": 0,
            "kdj_k":             50.0,
            "kdj_d":             50.0,
            "kdj_j":             50.0,
            "kdj_overbought":    0,
            "rsi_6":             50.0,
            "rsi_12":            50.0,
            "rsi_24":            50.0,
            "rsi_diverge":       0.0,
            "boll_pct":          0.5,
            "boll_width":        0.0,
            # bias1/2/3 已移除（由 ma_position 覆盖）
            "cci":               0.0,
            "wr":                0.0,
        }


# ------------------------------------------------------------------ #
# 模块级工具函数
# ------------------------------------------------------------------ #

def _is_nan(v) -> bool:
    """兼容 None / float NaN / pandas NA 的 NaN 判断。"""
    try:
        return v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v)
    except (TypeError, ValueError):
        return False


def _safe(v: float, default: float = 0.0) -> float:
    """NaN → default（默认 0），正常值 round 到 4 位。"""
    if _is_nan(v):
        return default
    return round(float(v), 4)
