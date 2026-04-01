"""
股票技术面因子特征（来源：Tushare stk_factor_pro）
===================================================
数据来源：stk_factor_pro 接口，每个交易日一次拉全市场所有股票。
建表 SQL：sql/create_stk_factor_pro.sql
扩列 SQL：sql/alter_stk_factor_pro_v2.sql
更新策略：按需更新（ensure_stk_factor_pro_data 确保当日数据存在）

口径规则（统一不复权 bfq）：
    - 所有列使用 _bfq 后缀，与 kline_day / 分钟线 / ma_position 统一口径
    - 振荡类指标（RSI/KDJ/MACD/CCI/WR/PSY/ROC/TRIX 等）：bfq=hfq=qfq，直接透传
    - 价格类指标（MA/EMA/Bollinger/Keltner/Donchian/XSII）：转化为 BIAS 或通道位置后输出
    - ATR/MTM（绝对量）：除以 close 转化为 % 再输出

输出列（全部归一化/无量纲，跨股可比）：

MACD 系列：
    macd_dif / macd_dea / macd_bar / macd_golden_cross
KDJ 系列：
    kdj_k / kdj_d / kdj_j / kdj_overbought
RSI 系列：
    rsi_6 / rsi_12 / rsi_24 / rsi_diverge
Bollinger：
    boll_pct / boll_width
BIAS 乖离率（6/12/24日，区别于 ma_position 的 5/10/13日）：
    bias1 / bias2 / bias3
CCI / WR / WR1：
    cci / wr / wr1
ATR 波动率：
    atr_pct = ATR / close * 100
MFI / VR 量能：
    mfi / vr
BBI 多空：
    bbi_bias = (close - BBI) / BBI * 100
BRAR 情绪：
    brar_ar / brar_br
CR 动量：
    cr
DFMA 差：
    dfma_dif / dfma_difma
DMI 趋势强度：
    dmi_adx / dmi_adxr / dmi_mdi / dmi_pdi
连涨跌/新高低周期：
    downdays / updays / lowdays / topdays
DPO 震荡：
    dpo / madpo
MA BIAS（5/10/20/30/60/90/250日）：
    ma_bias_5 / ma_bias_10 / ma_bias_20 / ma_bias_30 / ma_bias_60 / ma_bias_90 / ma_bias_250
EMA BIAS（5/10/20/30/60/90/250日）：
    ema_bias_5 / ema_bias_10 / ema_bias_20 / ema_bias_30 / ema_bias_60 / ema_bias_90 / ema_bias_250
EXPMA BIAS（12/50日）：
    expma_bias_12 / expma_bias_50
EMV 波动：
    emv / maemv
Keltner 通道位置：
    ktn_pct = (close - ktn_lower) / (ktn_upper - ktn_lower)
MASS 梅斯：
    mass / ma_mass
MTM % 动量：
    mtm_pct / mtmma_pct
PSY 心理线：
    psy / psyma
ROC 变动率：
    roc / maroc
Donchian 唐安奇通道位置：
    taq_pct = (close - taq_lower) / (taq_upper - taq_lower)
TRIX 三重平滑：
    trix / trma
ASI 振动：
    asi / asit
XSII 通道位置：
    xsii_pct = (close - xsii_td1) / (xsii_td4 - xsii_td1)

设计说明：
    - boll_pct / MA_BIAS / channel_pct 都需要 close，从 data_bundle.daily_grouped 取（bfq）。
    - 所有字段缺失（停牌/首日上市/DB 尚无数据）时输出中性值，不抛异常。
    - XGBoost 对离散信号列（macd_golden_cross / kdj_overbought）有独立分裂能力。
    - OBV（累计量，跨股/时间无法比较）不纳入输出。
"""

from typing import Dict

import numpy as np
import pandas as pd

from features.base_feature import BaseFeature
from features.feature_registry import feature_registry
from utils.common_tools import ensure_stk_factor_pro_data
from utils.db_utils import db
from utils.log_utils import logger

# ── DB 查询列（_bfq 口径，与 create/alter_stk_factor_pro.sql 保持一致）──────────
_DB_COLS = [
    "ts_code", "trade_date",
    # MACD
    "macd_dif_bfq", "macd_dea_bfq", "macd_bfq",
    # KDJ
    "kdj_k_bfq", "kdj_d_bfq", "kdj_bfq",
    # RSI
    "rsi_bfq_6", "rsi_bfq_12", "rsi_bfq_24",
    # Bollinger（价格类，转 boll_pct）
    "boll_upper_bfq", "boll_mid_bfq", "boll_lower_bfq",
    # BIAS 6/12/24日（区别于 ma_position 的 5/10/13日）
    "bias1_bfq", "bias2_bfq", "bias3_bfq",
    # CCI / WR / WR1
    "cci_bfq", "wr_bfq", "wr1_bfq",
    # ATR（绝对波动，转 %）
    "atr_bfq",
    # MFI / VR
    "mfi_bfq", "vr_bfq",
    # ASI 振动
    "asi_bfq", "asit_bfq",
    # BBI 多空（价格类，转 BIAS）
    "bbi_bfq",
    # BRAR 情绪
    "brar_ar_bfq", "brar_br_bfq",
    # CR 动量
    "cr_bfq",
    # DFMA
    "dfma_dif_bfq", "dfma_difma_bfq",
    # DMI 趋势强度
    "dmi_adx_bfq", "dmi_adxr_bfq", "dmi_mdi_bfq", "dmi_pdi_bfq",
    # 连涨跌/新高低周期（无复权后缀）
    "downdays", "updays", "lowdays", "topdays",
    # DPO
    "dpo_bfq", "madpo_bfq",
    # EMA（7 周期，价格类，转 BIAS）
    "ema_bfq_5", "ema_bfq_10", "ema_bfq_20", "ema_bfq_30", "ema_bfq_60", "ema_bfq_90", "ema_bfq_250",
    # EMV
    "emv_bfq", "maemv_bfq",
    # EXPMA（价格类，转 BIAS）
    "expma_12_bfq", "expma_50_bfq",
    # Keltner（价格类，转通道位置）
    "ktn_upper_bfq", "ktn_mid_bfq", "ktn_down_bfq",
    # MA（7 周期，价格类，转 BIAS）
    "ma_bfq_5", "ma_bfq_10", "ma_bfq_20", "ma_bfq_30", "ma_bfq_60", "ma_bfq_90", "ma_bfq_250",
    # MASS
    "mass_bfq", "ma_mass_bfq",
    # MTM（绝对价差，转 %）
    "mtm_bfq", "mtmma_bfq",
    # PSY
    "psy_bfq", "psyma_bfq",
    # ROC
    "roc_bfq", "maroc_bfq",
    # Donchian（价格类，转通道位置）
    "taq_up_bfq", "taq_mid_bfq", "taq_down_bfq",
    # TRIX
    "trix_bfq", "trma_bfq",
    # XSII（价格类，转通道位置）
    "xsii_td1_bfq", "xsii_td2_bfq", "xsii_td3_bfq", "xsii_td4_bfq",
]


@feature_registry.register("stk_factor_pro")
class StkFactorProFeature(BaseFeature):
    """
    股票技术面因子特征（Tushare stk_factor_pro）

    注册名：stk_factor_pro
    输出类型：个股级（含 stock_code 列）→ FeatureEngine inner join
    """

    feature_name = "stk_factor_pro"

    factor_columns = [
        "macd_dif", "macd_dea", "macd_bar", "macd_golden_cross",
        "kdj_k", "kdj_d", "kdj_j", "kdj_overbought",
        "rsi_6", "rsi_12", "rsi_24", "rsi_diverge",
        "boll_pct", "boll_width",
        "bias1", "bias2", "bias3",
        "cci", "wr", "wr1",
        "atr_pct", "mfi", "vr",
        "asi", "asit", "bbi_bias",
        "brar_ar", "brar_br", "cr",
        "dfma_dif", "dfma_difma",
        "dmi_adx", "dmi_adxr", "dmi_mdi", "dmi_pdi",
        "downdays", "updays", "lowdays", "topdays",
        "dpo", "madpo",
        "ma_bias_5", "ma_bias_10", "ma_bias_20", "ma_bias_30",
        "ma_bias_60", "ma_bias_90", "ma_bias_250",
        "ema_bias_5", "ema_bias_10", "ema_bias_20", "ema_bias_30",
        "ema_bias_60", "ema_bias_90", "ema_bias_250",
        "expma_bias_12", "expma_bias_50",
        "emv", "maemv",
        "ktn_pct", "mass", "ma_mass",
        "mtm_pct", "mtmma_pct",
        "psy", "psyma", "roc", "maroc",
        "taq_pct", "trix", "trma", "xsii_pct",
    ]

    # ------------------------------------------------------------------ #
    # 主计算入口
    # ------------------------------------------------------------------ #

    def calculate(self, data_bundle) -> tuple:
        """
        :param data_bundle: FeatureDataBundle，需含 trade_date / target_ts_codes /
                            daily_grouped（取 close，用于 BIAS / 通道位置归一化）
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

        # ── 2. 从 DB 批量拉取当日全市场技术因子（仅取 _DB_COLS）──
        # 部分列可能尚未在 DB 中（旧数据或未执行 alter_stk_factor_pro_v2.sql），
        # 使用动态列查询：先获取实际存在的列，再发起查询
        existing_cols = self._get_existing_db_cols(trade_date_fmt)
        query_cols = [c for c in _DB_COLS if c in existing_cols]

        if not query_cols:
            logger.error(f"[StkFactorPro] stk_factor_pro 表无可用列，返回空")
            return pd.DataFrame(), {}

        cols_str = ", ".join(f"`{c}`" for c in query_cols)
        sql = f"SELECT {cols_str} FROM stk_factor_pro WHERE trade_date = %s"
        try:
            raw = db.query(sql, (trade_date_fmt,), return_df=True)
        except Exception as e:
            logger.error(f"[StkFactorPro] DB 查询失败，{trade_date}：{e}")
            raw = pd.DataFrame()

        if raw is None or raw.empty:
            logger.warning(f"[StkFactorPro] {trade_date} DB 无数据，全部填充中性值")
            raw = pd.DataFrame(columns=query_cols)

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

            # 取不复权收盘价（bfq）用于 BIAS / 通道位置计算
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
    # 动态列检测（兼容旧数据/未执行 alter 的环境）
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_existing_db_cols(trade_date_fmt: str) -> set:
        """返回 stk_factor_pro 表实际存在的列名集合（含 ts_code/trade_date）"""
        try:
            result = db.query("DESCRIBE stk_factor_pro", return_df=True)
            if result is not None and not result.empty:
                return set(result["Field"].tolist())
        except Exception:
            pass
        # 降级：返回原始 17 列（旧建表）
        return {
            "ts_code", "trade_date",
            "macd_dif_bfq", "macd_dea_bfq", "macd_bfq",
            "kdj_k_bfq", "kdj_d_bfq", "kdj_bfq",
            "rsi_bfq_6", "rsi_bfq_12", "rsi_bfq_24",
            "boll_upper_bfq", "boll_mid_bfq", "boll_lower_bfq",
            "bias1_bfq", "bias2_bfq", "bias3_bfq",
            "cci_bfq", "atr_bfq", "wr_bfq", "obv_bfq", "mfi_bfq", "vr_bfq",
        }

    # ------------------------------------------------------------------ #
    # 单股特征计算
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_row(ts_code: str, trade_date: str, frow: dict, close: float) -> dict:
        """从 DB 行 + 不复权收盘价计算全部特征列。"""

        def _f(key: str) -> float:
            v = frow.get(key)
            return float(v) if v is not None and not _is_nan(v) else float("nan")

        def _bias(price, ref) -> float:
            """BIAS = (price - ref) / ref × 100；ref 为 0 或 nan 时返回 0"""
            if not price or not ref or np.isnan(price) or np.isnan(ref) or ref == 0:
                return 0.0
            return round((price - ref) / ref * 100, 4)

        def _channel_pct(price, lower, upper) -> float:
            """price 在 [lower, upper] 通道中的位置 [0, 1]"""
            if any(_is_nan(v) or v is None for v in [price, lower, upper]):
                return 0.5
            band = upper - lower
            if band <= 0:
                return 0.5
            return round(max(0.0, min(1.0, (price - lower) / band)), 4)

        def _pct_of_close(val) -> float:
            """绝对量 / close × 100，得到相对 % 值"""
            if _is_nan(val) or not close or np.isnan(val):
                return 0.0
            return round(val / close * 100, 4)

        # ── MACD ──
        macd_dif = _f("macd_dif_bfq")
        macd_dea = _f("macd_dea_bfq")
        macd_bar = _f("macd_bfq")
        macd_golden_cross = 0 if np.isnan(macd_bar) else (1 if macd_bar > 0 else (-1 if macd_bar < 0 else 0))

        # ── KDJ ──
        kdj_k = _f("kdj_k_bfq")
        kdj_d = _f("kdj_d_bfq")
        kdj_j = _f("kdj_bfq")
        if np.isnan(kdj_k) or np.isnan(kdj_d):
            kdj_overbought = 0
        elif kdj_k > 80 and kdj_d > 80:
            kdj_overbought = 1
        elif kdj_k < 20 and kdj_d < 20:
            kdj_overbought = -1
        else:
            kdj_overbought = 0

        # ── RSI ──
        rsi_6  = _f("rsi_bfq_6")
        rsi_12 = _f("rsi_bfq_12")
        rsi_24 = _f("rsi_bfq_24")
        rsi_diverge = round(rsi_6 - rsi_24, 4) if not (np.isnan(rsi_6) or np.isnan(rsi_24)) else 0.0

        # ── Bollinger 位置 ──
        boll_upper = _f("boll_upper_bfq")
        boll_mid   = _f("boll_mid_bfq")
        boll_lower = _f("boll_lower_bfq")
        boll_pct   = _channel_pct(close, boll_lower, boll_upper) if close else 0.5
        boll_width = 0.0
        if not np.isnan(boll_upper) and not np.isnan(boll_lower) and not np.isnan(boll_mid) and boll_mid > 0:
            boll_width = round((boll_upper - boll_lower) / boll_mid, 4)

        # ── BIAS 6/12/24日（来自 API，区别于 ma_position 的本地 5/10/13日）──
        bias1 = _f("bias1_bfq")
        bias2 = _f("bias2_bfq")
        bias3 = _f("bias3_bfq")

        # ── CCI / WR / WR1 ──
        cci = _f("cci_bfq")
        wr  = _f("wr_bfq")
        wr1 = _f("wr1_bfq")

        # ── ATR 波动率（绝对 → %）──
        atr_pct = _pct_of_close(_f("atr_bfq"))

        # ── MFI / VR ──
        mfi = _f("mfi_bfq")
        vr  = _f("vr_bfq")

        # ── ASI 振动 ──
        asi  = _f("asi_bfq")
        asit = _f("asit_bfq")

        # ── BBI 多空（价格类 → BIAS）──
        bbi_bias = _bias(close, _f("bbi_bfq")) if close else 0.0

        # ── BRAR 情绪（原始值可比，AR/BR 均以 100 为基准）──
        brar_ar = _f("brar_ar_bfq")
        brar_br = _f("brar_br_bfq")

        # ── CR 动量 ──
        cr = _f("cr_bfq")

        # ── DFMA ──
        dfma_dif   = _f("dfma_dif_bfq")
        dfma_difma = _f("dfma_difma_bfq")

        # ── DMI ──
        dmi_adx  = _f("dmi_adx_bfq")
        dmi_adxr = _f("dmi_adxr_bfq")
        dmi_mdi  = _f("dmi_mdi_bfq")
        dmi_pdi  = _f("dmi_pdi_bfq")

        # ── 连涨跌/新高低 ──
        downdays = _f("downdays")
        updays   = _f("updays")
        lowdays  = _f("lowdays")
        topdays  = _f("topdays")

        # ── DPO ──
        dpo   = _f("dpo_bfq")
        madpo = _f("madpo_bfq")

        # ── MA BIAS（7 周期，价格类 → %）──
        ma_bias_5   = _bias(close, _f("ma_bfq_5"))   if close else 0.0
        ma_bias_10  = _bias(close, _f("ma_bfq_10"))  if close else 0.0
        ma_bias_20  = _bias(close, _f("ma_bfq_20"))  if close else 0.0
        ma_bias_30  = _bias(close, _f("ma_bfq_30"))  if close else 0.0
        ma_bias_60  = _bias(close, _f("ma_bfq_60"))  if close else 0.0
        ma_bias_90  = _bias(close, _f("ma_bfq_90"))  if close else 0.0
        ma_bias_250 = _bias(close, _f("ma_bfq_250")) if close else 0.0

        # ── EMA BIAS（7 周期，价格类 → %）──
        ema_bias_5   = _bias(close, _f("ema_bfq_5"))   if close else 0.0
        ema_bias_10  = _bias(close, _f("ema_bfq_10"))  if close else 0.0
        ema_bias_20  = _bias(close, _f("ema_bfq_20"))  if close else 0.0
        ema_bias_30  = _bias(close, _f("ema_bfq_30"))  if close else 0.0
        ema_bias_60  = _bias(close, _f("ema_bfq_60"))  if close else 0.0
        ema_bias_90  = _bias(close, _f("ema_bfq_90"))  if close else 0.0
        ema_bias_250 = _bias(close, _f("ema_bfq_250")) if close else 0.0

        # ── EXPMA BIAS（12/50日）──
        expma_bias_12 = _bias(close, _f("expma_12_bfq")) if close else 0.0
        expma_bias_50 = _bias(close, _f("expma_50_bfq")) if close else 0.0

        # ── EMV ──
        emv   = _f("emv_bfq")
        maemv = _f("maemv_bfq")

        # ── Keltner 通道位置 ──
        ktn_pct = _channel_pct(close, _f("ktn_down_bfq"), _f("ktn_upper_bfq")) if close else 0.5

        # ── MASS 梅斯线 ──
        mass    = _f("mass_bfq")
        ma_mass = _f("ma_mass_bfq")

        # ── MTM 动量（绝对 → %）──
        mtm_pct   = _pct_of_close(_f("mtm_bfq"))
        mtmma_pct = _pct_of_close(_f("mtmma_bfq"))

        # ── PSY 心理线 ──
        psy   = _f("psy_bfq")
        psyma = _f("psyma_bfq")

        # ── ROC 变动率（已是 %）──
        roc   = _f("roc_bfq")
        maroc = _f("maroc_bfq")

        # ── Donchian 唐安奇通道位置 ──
        taq_pct = _channel_pct(close, _f("taq_down_bfq"), _f("taq_up_bfq")) if close else 0.5

        # ── TRIX ──
        trix = _f("trix_bfq")
        trma = _f("trma_bfq")

        # ── XSII 薛斯通道位置（td1=下轨，td4=上轨）──
        xsii_pct = _channel_pct(close, _f("xsii_td1_bfq"), _f("xsii_td4_bfq")) if close else 0.5

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
            # BIAS 6/12/24日
            "bias1":             _safe(bias1),
            "bias2":             _safe(bias2),
            "bias3":             _safe(bias3),
            # CCI / WR
            "cci":               _safe(cci),
            "wr":                _safe(wr),
            "wr1":               _safe(wr1),
            # ATR 波动率
            "atr_pct":           atr_pct,
            # MFI / VR
            "mfi":               _safe(mfi),
            "vr":                _safe(vr),
            # ASI
            "asi":               _safe(asi),
            "asit":              _safe(asit),
            # BBI
            "bbi_bias":          bbi_bias,
            # BRAR
            "brar_ar":           _safe(brar_ar),
            "brar_br":           _safe(brar_br),
            # CR
            "cr":                _safe(cr),
            # DFMA
            "dfma_dif":          _safe(dfma_dif),
            "dfma_difma":        _safe(dfma_difma),
            # DMI
            "dmi_adx":           _safe(dmi_adx),
            "dmi_adxr":          _safe(dmi_adxr),
            "dmi_mdi":           _safe(dmi_mdi),
            "dmi_pdi":           _safe(dmi_pdi),
            # 连涨跌/新高低
            "downdays":          _safe(downdays),
            "updays":            _safe(updays),
            "lowdays":           _safe(lowdays),
            "topdays":           _safe(topdays),
            # DPO
            "dpo":               _safe(dpo),
            "madpo":             _safe(madpo),
            # MA BIAS
            "ma_bias_5":         ma_bias_5,
            "ma_bias_10":        ma_bias_10,
            "ma_bias_20":        ma_bias_20,
            "ma_bias_30":        ma_bias_30,
            "ma_bias_60":        ma_bias_60,
            "ma_bias_90":        ma_bias_90,
            "ma_bias_250":       ma_bias_250,
            # EMA BIAS
            "ema_bias_5":        ema_bias_5,
            "ema_bias_10":       ema_bias_10,
            "ema_bias_20":       ema_bias_20,
            "ema_bias_30":       ema_bias_30,
            "ema_bias_60":       ema_bias_60,
            "ema_bias_90":       ema_bias_90,
            "ema_bias_250":      ema_bias_250,
            # EXPMA BIAS
            "expma_bias_12":     expma_bias_12,
            "expma_bias_50":     expma_bias_50,
            # EMV
            "emv":               _safe(emv),
            "maemv":             _safe(maemv),
            # Keltner 通道位置
            "ktn_pct":           ktn_pct,
            # MASS
            "mass":              _safe(mass),
            "ma_mass":           _safe(ma_mass),
            # MTM %
            "mtm_pct":           mtm_pct,
            "mtmma_pct":         mtmma_pct,
            # PSY
            "psy":               _safe(psy),
            "psyma":             _safe(psyma),
            # ROC
            "roc":               _safe(roc),
            "maroc":             _safe(maroc),
            # Donchian 通道位置
            "taq_pct":           taq_pct,
            # TRIX
            "trix":              _safe(trix),
            "trma":              _safe(trma),
            # XSII 通道位置
            "xsii_pct":          xsii_pct,
        }

    @staticmethod
    def _neutral_row(ts_code: str, trade_date: str) -> dict:
        """
        无数据时的中性填充（不对模型引入方向性偏差）。

        中性语义约定：
            - 振荡指标中点（KDJ=50, RSI=50, PSY=50 等）
            - BIAS/% = 0（无偏离）
            - 通道位置 = 0.5（区间中点）
            - 离散信号 = 0（无信号）
            - 连涨跌/周期数 = 0（无信息）
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
            "bias1":             0.0,
            "bias2":             0.0,
            "bias3":             0.0,
            "cci":               0.0,
            "wr":                0.0,
            "wr1":               0.0,
            "atr_pct":           0.0,
            "mfi":               50.0,
            "vr":                1.0,
            "asi":               0.0,
            "asit":              0.0,
            "bbi_bias":          0.0,
            "brar_ar":           100.0,  # BRAR AR 以 100 为基准中性值
            "brar_br":           100.0,
            "cr":                100.0,  # CR 以 100 为基准中性值
            "dfma_dif":          0.0,
            "dfma_difma":        0.0,
            "dmi_adx":           0.0,
            "dmi_adxr":          0.0,
            "dmi_mdi":           0.0,
            "dmi_pdi":           0.0,
            "downdays":          0.0,
            "updays":            0.0,
            "lowdays":           0.0,
            "topdays":           0.0,
            "dpo":               0.0,
            "madpo":             0.0,
            "ma_bias_5":         0.0,
            "ma_bias_10":        0.0,
            "ma_bias_20":        0.0,
            "ma_bias_30":        0.0,
            "ma_bias_60":        0.0,
            "ma_bias_90":        0.0,
            "ma_bias_250":       0.0,
            "ema_bias_5":        0.0,
            "ema_bias_10":       0.0,
            "ema_bias_20":       0.0,
            "ema_bias_30":       0.0,
            "ema_bias_60":       0.0,
            "ema_bias_90":       0.0,
            "ema_bias_250":      0.0,
            "expma_bias_12":     0.0,
            "expma_bias_50":     0.0,
            "emv":               0.0,
            "maemv":             0.0,
            "ktn_pct":           0.5,
            "mass":              25.0,  # MASS 中性约 25（25~27 为震荡区间，>27 预示反转）
            "ma_mass":           25.0,
            "mtm_pct":           0.0,
            "mtmma_pct":         0.0,
            "psy":               50.0,
            "psyma":             50.0,
            "roc":               0.0,
            "maroc":             0.0,
            "taq_pct":           0.5,
            "trix":              0.0,
            "trma":              0.0,
            "xsii_pct":          0.5,
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
