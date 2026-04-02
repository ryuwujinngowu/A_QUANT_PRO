"""
超跌反包策略（OversoldReversalStrategy）
==========================================
候选池构建（D 日收盘后）：
  1. 全市场按 pct_chg 升序，取跌幅最大的初始样本（Top N）
  2. 剔除上市首 5 个交易日新股
  3. 剔除连续跌停（D-1 收盘 ≤ D-1 跌停价）
  4. 剔除缩量大跌（今日 amount < 近5日均 amount × 50%）
  5. 保留：D-1 vs D-11 涨幅 > 13%（10日动量），或 D-1 vs D-31 涨幅 > 90%（30日强势）
  6. 保留：今日 pct_chg 是近10个交易日内最小值（最大单日跌幅）

信号执行：D+1 开盘买入 → D+2 收盘卖出
训练标签：label_d2_5pct（(D+2 close - D+1 open) / D+1 open ≥ 5%）

无未来函数保证：
  - 所有候选池构建、特征计算均使用 D 日及之前数据
  - kline_day 不复权数据，短线2日持仓无需复权
"""
import os
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from features import FeatureEngine
from features.bundle_factory import build_bundle_from_context
from strategies.base_strategy import BaseStrategy
import learnEngine.train_config as cfg
from utils.common_tools import (
    calc_limit_down_price,
    filter_st_stocks,
    get_daily_kline_data,
    get_kline_day_range,
    get_stock_list_date_map,
    get_trade_dates,
)
from utils.log_utils import logger
from utils.xgb_compat import safe_predict_proba


# ── 策略常量 ──────────────────────────────────────────────────────────
_INITIAL_POOL_SIZE   = 300    # 跌幅榜初始取样数（过滤前最大宽度）
_MIN_IPO_DAYS        = 5      # 上市首 N 个交易日视为新股，剔除
_SHRINK_AMOUNT_RATIO = 0.50   # 缩量阈值：今日 amount < 近5日均 × 50%
_GAIN_10D_MIN        = 0.13   # 10日涨幅门槛（D-11→D-1，不含今日）
_GAIN_30D_MIN        = 0.90   # 30日涨幅门槛（D-31→D-1，不含今日）
_MIN_AMOUNT          = 10_000  # 最低流动性（千元，= 1000万，与 dataset.py 对齐）


def _is_limit_down(ts_code: str, close: float, pre_close: float) -> bool:
    """判断该交易日是否封在跌停价（含 ±0.01 容差）"""
    ld = calc_limit_down_price(ts_code, pre_close)
    return ld > 0 and close <= ld + 0.01


class OversoldReversalStrategy(BaseStrategy):
    """
    超跌反包 XGBoost 选股策略

    依赖前置：
        1. 已运行 python learnEngine/dataset.py 生成训练集（含 oversold_reversal 候选池样本）
        2. 已运行 python train.py 训练并保存模型
    """

    @property
    def strategy_id(self) -> str:
        return "oversold_reversal"

    def __init__(self):
        super().__init__()
        self.strategy_name = "超跌反包策略（跌幅榜+历史强势+最大单日跌幅+XGBoost）"
        self.strategy_params = {
            "buy_top_k":   5,
            "min_prob":    0.55,
            "sell_type":   "close",   # D+2 收盘卖出
            "load_minute": True,
            "model_path":  self._get_runtime_model_path(),
        }
        self._model                      = None
        self._required_feature_modules   = None   # None = 全量，与训练口径一致
        self._last_buy_signal_details: List[Dict] = []
        self.initialize()

    def _get_runtime_model_path(self) -> str:
        return cfg.get_strategy_runtime_model_path(self.strategy_id)

    def initialize(self) -> None:
        self.clear_signal()
        self._model = None
        self._last_buy_signal_details = []

    def get_last_buy_signal_details(self) -> List[Dict]:
        return list(self._last_buy_signal_details)

    def supports_ml_training(self) -> bool:
        return True

    def get_training_label_target(self) -> str:
        return "label_d2_5pct"   # (D+2 close - D+1 open) / D+1 open ≥ 5%

    def get_model_registry_info(self) -> Dict[str, str]:
        base_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        return {
            "strategy_id": self.strategy_id,
            "archive_dir": os.path.join(base_dir, "model", self.strategy_id),
            "runtime_dir": cfg.get_strategy_runtime_model_dir(self.strategy_id),
        }

    # ------------------------------------------------------------------ #
    # 候选池构建（dataset.py 和 generate_signal 共用入口）
    # ------------------------------------------------------------------ #

    def build_training_candidates(
        self,
        trade_date: str,
        daily_df: pd.DataFrame = None,
    ) -> Tuple[pd.DataFrame, Dict]:
        if daily_df is None:
            daily_df = get_daily_kline_data(trade_date)
        if daily_df.empty:
            return pd.DataFrame(), {}

        candidates = self._build_candidate_pool(trade_date, daily_df)
        if not candidates:
            return pd.DataFrame(), {}

        candidate_df = pd.DataFrame({
            "ts_code":       candidates,
            "strategy_id":   self.strategy_id,
            "strategy_name": self.strategy_name,
            "sector_name":   "",
            "trade_date":    trade_date,
        })
        context = {"trade_date": trade_date, "target_ts_codes": candidates}
        return candidate_df, context

    # ------------------------------------------------------------------ #
    # 核心过滤逻辑
    # ------------------------------------------------------------------ #

    def _build_candidate_pool(
        self,
        trade_date: str,
        daily_df: pd.DataFrame,
    ) -> List[str]:
        if daily_df.empty or "pct_chg" not in daily_df.columns:
            return []

        # ── Step 0: 基础准入（主板 + 非ST + 最低流动性）──────────────────
        df = daily_df.copy()
        # 排除北交所 / 科创板 / 创业板（短线策略聚焦主板，涨跌停10%规则统一）
        df = df[df["ts_code"].apply(
            lambda c: not (
                c.endswith(".BJ")
                or c.startswith("688")
                or (c.startswith(("300", "301", "302")) and c.endswith(".SZ"))
            )
        )]
        df["pct_chg"] = pd.to_numeric(df["pct_chg"], errors="coerce")
        df = df.dropna(subset=["pct_chg"])
        if "amount" in df.columns:
            df = df[df["amount"].fillna(0) >= _MIN_AMOUNT]

        # 非ST
        try:
            st_ok = set(filter_st_stocks(df["ts_code"].tolist(), trade_date))
            df = df[df["ts_code"].isin(st_ok)]
        except Exception as e:
            logger.warning(f"[OversoldReversal] {trade_date} ST过滤失败: {e}")

        # ── Step 1: 跌幅榜 Top N ────────────────────────────────────────
        pool_df = df.sort_values("pct_chg", ascending=True).head(_INITIAL_POOL_SIZE)
        if pool_df.empty:
            return []

        pool_codes = pool_df["ts_code"].tolist()

        # ── 批量拉取近 80 日历史（5日均量/10日/30日涨幅/10日跌幅全覆盖）──
        try:
            d = datetime.strptime(trade_date, "%Y-%m-%d")
            hist_start = (d - timedelta(days=90)).strftime("%Y-%m-%d")
            hist_df = get_kline_day_range(pool_codes, hist_start, trade_date)
        except Exception as e:
            logger.warning(f"[OversoldReversal] {trade_date} 历史数据拉取失败: {e}")
            return []

        if hist_df.empty:
            return []

        # 标准化 trade_date 为 YYYY-MM-DD
        hist_df = hist_df.copy()
        hist_df["trade_date"] = (
            hist_df["trade_date"].astype(str)
            .str.replace("-", "", regex=False)
            .pipe(lambda s: s.str[:4] + "-" + s.str[4:6] + "-" + s.str[6:8])
        )

        # D-1 日期（用于连续跌停检查）
        try:
            all_dates = get_trade_dates(hist_start, trade_date)
            prev_date = all_dates[-2] if len(all_dates) >= 2 else None
        except Exception as e:
            logger.warning(f"[OversoldReversal] {trade_date} 交易日获取失败: {e}")
            return []

        # ── Step 2: 剔除上市首 5 个交易日新股 ────────────────────────────
        list_date_map = get_stock_list_date_map()
        ipo_cutoff = all_dates[-_MIN_IPO_DAYS] if len(all_dates) >= _MIN_IPO_DAYS else all_dates[0]
        pool_codes = [
            c for c in pool_codes
            if list_date_map.get(c, "").replace("-", "") < ipo_cutoff.replace("-", "")
        ]
        if not pool_codes:
            return []

        # 按股票分组
        grp = hist_df.groupby("ts_code")

        result = []
        for ts_code in pool_codes:
            try:
                stk = grp.get_group(ts_code).sort_values("trade_date")
            except KeyError:
                continue

            d0_rows = stk[stk["trade_date"] == trade_date]
            if d0_rows.empty:
                continue
            d0 = d0_rows.iloc[-1]

            d0_close     = float(d0.get("close",     0) or 0)
            d0_pre_close = float(d0.get("pre_close", 0) or 0)
            d0_amount    = float(d0.get("amount",    0) or 0)
            d0_pct_chg   = float(d0.get("pct_chg",  0) or 0)

            if d0_close <= 0 or d0_pre_close <= 0:
                continue

            # ── Step 3: 剔除连续跌停（D-1 收盘 ≤ D-1 跌停价）───────────
            if prev_date:
                prev_rows = stk[stk["trade_date"] == prev_date]
                if not prev_rows.empty:
                    prev = prev_rows.iloc[-1]
                    prev_c  = float(prev.get("close",     0) or 0)
                    prev_pc = float(prev.get("pre_close", 0) or 0)
                    if prev_c > 0 and prev_pc > 0 and _is_limit_down(ts_code, prev_c, prev_pc):
                        continue   # 昨日封跌停，跳过

            # ── Step 4: 剔除缩量大跌 ─────────────────────────────────────
            hist_5d = stk[stk["trade_date"] < trade_date].tail(5)
            if not hist_5d.empty:
                amt_vals = pd.to_numeric(hist_5d["amount"], errors="coerce").dropna()
                avg_amt  = amt_vals.mean() if not amt_vals.empty else 0.0
                if avg_amt > 0 and d0_amount < avg_amt * _SHRINK_AMOUNT_RATIO:
                    continue   # 成交额不足均量 50%，跳过

            # ── Step 5: 历史动量筛选（10日 > 13% 或 30日 > 90%）──────────
            # 涨幅基准：D-1 收盘（不含今日跌幅），基准日：D-11 / D-31
            hist_before = stk[stk["trade_date"] < trade_date].sort_values("trade_date")
            passes_momentum = False

            if len(hist_before) >= 11:
                c_d1  = float(hist_before.iloc[-1].get("close", 0) or 0)
                c_d11 = float(hist_before.iloc[-11].get("close", 0) or 0)
                if c_d11 > 0 and (c_d1 - c_d11) / c_d11 >= _GAIN_10D_MIN:
                    passes_momentum = True

            if not passes_momentum and len(hist_before) >= 31:
                c_d1  = float(hist_before.iloc[-1].get("close", 0) or 0)
                c_d31 = float(hist_before.iloc[-31].get("close", 0) or 0)
                if c_d31 > 0 and (c_d1 - c_d31) / c_d31 >= _GAIN_30D_MIN:
                    passes_momentum = True

            if not passes_momentum:
                continue

            # ── Step 6: 今日跌幅是近10个交易日内最大单日跌幅 ─────────────
            recent10 = stk[stk["trade_date"] <= trade_date].tail(10)
            chg_vals = pd.to_numeric(recent10["pct_chg"], errors="coerce").dropna()
            if chg_vals.empty or d0_pct_chg > chg_vals.min() + 1e-4:
                continue   # 今日不是近10日最大跌幅，跳过

            result.append(ts_code)

        logger.info(
            f"[OversoldReversal] {trade_date} 候选池: {len(result)} 只 "
            f"（跌幅榜初始 {_INITIAL_POOL_SIZE} → 过滤后 {len(result)} 只）"
        )
        return result

    # ------------------------------------------------------------------ #
    # 买入信号生成（推断入口）
    # ------------------------------------------------------------------ #

    def generate_signal(
        self,
        trade_date: str,
        daily_df: pd.DataFrame,
        positions: Dict[str, any],
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        sell_signal_map: Dict[str, str] = {}
        buy_signal_map = self._generate_buy_signal(trade_date, daily_df)
        logger.info(
            f"[{self.strategy_name}] {trade_date} "
            f"| 买入: {list(buy_signal_map.keys())}"
        )
        return buy_signal_map, sell_signal_map

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        model_path = self.strategy_params.get("model_path", "")
        if not model_path or not os.path.exists(model_path):
            logger.error(f"[OversoldReversal] 模型文件不存在: {model_path}")
            return False
        try:
            with open(model_path, "rb") as f:
                self._model = pickle.load(f)
            logger.info(f"[OversoldReversal] 模型加载成功: {model_path}")
            return True
        except Exception as e:
            logger.error(f"[OversoldReversal] 模型加载失败: {e}")
            return False

    def _generate_buy_signal(
        self,
        trade_date: str,
        daily_df: pd.DataFrame,
    ) -> Dict[str, str]:
        self._last_buy_signal_details = []

        if not self._ensure_model():
            return {}

        try:
            candidate_df, context = self.build_training_candidates(trade_date, daily_df)
        except Exception as e:
            logger.error(
                f"[OversoldReversal] {trade_date} 候选池构建失败: {e}", exc_info=True
            )
            return {}

        if candidate_df.empty:
            logger.warning(f"[OversoldReversal] {trade_date} 候选池为空")
            return {}

        try:
            bundle = build_bundle_from_context(
                context,
                load_minute=self.strategy_params["load_minute"],
                required_modules=self._required_feature_modules,
            )
            engine     = FeatureEngine(self._required_feature_modules)
            feature_df = engine.run_single_date(bundle)
        except Exception as e:
            logger.error(
                f"[OversoldReversal] {trade_date} 特征计算失败: {e}", exc_info=True
            )
            return {}

        if feature_df.empty:
            logger.warning(f"[OversoldReversal] {trade_date} 特征为空")
            return {}

        try:
            expected_cols = list(self._model.feature_names_in_)
            X = (
                feature_df
                .reindex(columns=expected_cols, fill_value=0)
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0)
                .replace([np.inf, -np.inf], 0)
            )
            probs = safe_predict_proba(self._model, X)[:, 1]
            feature_df = feature_df.copy()
            feature_df["_prob"] = probs
        except Exception as e:
            logger.error(
                f"[OversoldReversal] {trade_date} 模型预测失败: {e}", exc_info=True
            )
            return {}

        min_prob = float(self.strategy_params.get("min_prob", 0.0))
        top_k    = int(self.strategy_params["buy_top_k"])
        code_col = "stock_code" if "stock_code" in feature_df.columns else "ts_code"

        selected = (
            feature_df[feature_df["_prob"] >= min_prob]
            .sort_values("_prob", ascending=False)
            .head(top_k)
        )
        if selected.empty:
            logger.info(f"[OversoldReversal] {trade_date} 无超过阈值 {min_prob} 的信号")
            return {}

        buy_signals: Dict[str, str] = {}
        for _, r in selected.iterrows():
            ts_code = str(r.get(code_col, ""))
            if ts_code:
                buy_signals[ts_code] = "open"
                self._last_buy_signal_details.append({
                    "stock_code": ts_code,
                    "buy_type":   "open",
                    "prob":       round(float(r["_prob"]), 4),
                })

        logger.info(
            f"[OversoldReversal] {trade_date} 买入信号: {list(buy_signals.keys())} "
            f"（卖出：D+2收盘）"
        )
        return buy_signals
