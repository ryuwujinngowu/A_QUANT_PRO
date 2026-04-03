"""
趋势股跟随策略（XGBoost 驱动，T+2 持有）
==========================================
候选股准入：
  1. 全市场60日涨幅（kline_day_hfq 后复权）降序，剔除新股（上市≤61交易日），取前100
  2. 再剔除近20日涨幅排名前10（"过热"个股，已在短期大幅拉升，均为非新股）
  3. 再剔除 MA5 < MA30 的个股（均线空头排列）
        均线公共方法：features/ma_indicator.py → MAIndicator.compute_ma_from_hfq_range()
        批量预加载 HFQ kline 后传入 preloaded_kline 参数，零额外 DB 查询

信号执行：
  D 日收盘后运行 → 模型输出买入信号 → D+1 日开盘买入 → D+1 日收盘卖出

无未来函数保证：
  - 所有候选池构建、特征计算均使用 D 日及之前数据
  - kline_day_hfq 后复权数据历史价格不变，无未来函数
"""
import os
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import learnEngine.train_config as cfg
from features import FeatureEngine, FeatureDataBundle
from features.ma_indicator import TechnicalFeatures
from strategies.base_strategy import BaseStrategy
from utils.common_tools import (
    get_trade_dates,
    get_hfq_kline_range,
    get_stock_list_date_map,
    filter_st_stocks,
)
from utils.db_utils import db
from utils.log_utils import logger
from utils.xgb_compat import safe_predict_proba

# ── 策略常量 ──────────────────────────────────────────────────────────────────
_RANK_TOP_N         = 100    # 60日涨幅排名取前N
_HOT_EXCLUDE_TOP_N  = 10     # 近20日涨幅前N名剔除（过热）
_NEW_STOCK_DAYS     = 61     # 上市≤N个交易日视为新股
_MIN_MA_DAYS        = 60     # MA30至少需要的日线数
# 成交额过滤仅用于候选池粗筛（此处保留常量供 build_training_candidates 将来使用，当前未启用）
_MIN_AMOUNT_YI      = 0.5    # 最低日均成交额（亿元）


class TrendFollowStrategy(BaseStrategy):
    """
    趋势股跟随策略
      - 候选池：全市场60日动量排名 + MA5>MA30 + 过热过滤
      - 模型：独立训练的 trend_follow XGBoost
      - 持仓：D+1开盘买入，D+2收盘卖出
    """

    @property
    def strategy_id(self) -> str:
        return "trend_follow"

    def __init__(self):
        super().__init__()
        self.strategy_name = "趋势股跟随策略（60日动量+MA过滤+XGBoost）"
        self.strategy_params = {
            "buy_top_k":   5,
            "min_prob":    0.55,
            "sell_type":   "close",    # D+2 收盘卖出
            "load_minute": True,
            "model_path":  self._get_runtime_model_path(),
        }
        self._ma_calc                    = TechnicalFeatures()
        self._model                      = None
        self._required_feature_modules   = None   # None = 全量特征，与训练时 FeatureEngine() 保持一致
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
        return "label1"   # D+1 日内收益 >= 5%，对应 D+1 开盘买入/收盘卖出的口径

    def get_model_registry_info(self) -> Dict[str, str]:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return {
            "strategy_id": self.strategy_id,
            "archive_dir": os.path.join(base_dir, "model", self.strategy_id),
            "runtime_dir": cfg.get_strategy_runtime_model_dir(self.strategy_id),
        }

    def generate_signal(
        self,
        trade_date: str,
        daily_df: pd.DataFrame,
        positions: Dict[str, any],
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """引擎调用接口（买入 + 卖出信号）"""
        sell_signal_map: Dict[str, str] = {}
        buy_signal_map = self._generate_buy_signal(trade_date)
        logger.info(
            f"[{self.strategy_name}] {trade_date} "
            f"| 买入: {list(buy_signal_map.keys())}"
        )
        return buy_signal_map, sell_signal_map

    # ------------------------------------------------------------------ #
    # 候选池构建
    # ------------------------------------------------------------------ #

    def _get_trade_dates_window(self, trade_date: str, days: int) -> List[str]:
        """获取 trade_date 及之前共 days 个交易日列表（升序）"""
        d = datetime.strptime(trade_date, "%Y-%m-%d")
        start = (d - timedelta(days=days * 3)).strftime("%Y-%m-%d")
        return get_trade_dates(start, trade_date)[-days:]

    def _get_hfq_close_on_date(self, date: str) -> Dict[str, float]:
        """
        查 kline_day_hfq 获取指定交易日所有股票的收盘价。
        trade_date 支持 yyyy-mm-dd 和 yyyymmdd 两种格式。
        """
        date_fmt = date.replace("-", "")
        try:
            rows = db.query(
                "SELECT ts_code, close FROM kline_day_hfq WHERE trade_date = %s",
                params=(date_fmt,),
            ) or []
            return {
                r["ts_code"]: float(r["close"])
                for r in rows
                if r.get("close") and not (
                    r["ts_code"].endswith(".BJ")
                    or r["ts_code"].startswith(("83", "87", "88"))
                )
            }
        except Exception as e:
            logger.warning(f"[TrendFollow] HFQ收盘价查询失败 {date}: {e}")
            return {}

    def build_training_candidates(
        self,
        trade_date: str,
        daily_df: pd.DataFrame = None,
    ) -> Tuple[pd.DataFrame, Dict[str, any]]:
        """
        构建候选池，返回 (candidate_df, context)。
        candidate_df 一行一股票，含 ts_code 列；context 供 FeatureDataBundle 使用。
        """
        trade_dates_60d = self._get_trade_dates_window(trade_date, 60)
        if len(trade_dates_60d) < 2:
            return pd.DataFrame(), {}

        d0_date  = trade_dates_60d[-1]
        d60_date = trade_dates_60d[0]
        d20_date = self._get_trade_dates_window(trade_date, 20)[0]

        # ── Step 1: 60日后复权涨幅，全市场排名 ─────────────────────────────
        close_d0  = self._get_hfq_close_on_date(d0_date)
        close_d60 = self._get_hfq_close_on_date(d60_date)

        if not close_d0 or not close_d60:
            logger.warning(f"[TrendFollow] {trade_date} HFQ数据不足，跳过")
            return pd.DataFrame(), {}

        list_date_map = get_stock_list_date_map()
        # 新股阈值：trade_date 前第61个交易日
        d61_window = self._get_trade_dates_window(trade_date, _NEW_STOCK_DAYS)
        new_stock_cutoff = d61_window[0] if d61_window else ""  # 上市日 > 此日期为新股

        returns_60d = []
        for ts_code, c_d0 in close_d0.items():
            c_d60 = close_d60.get(ts_code)
            if not c_d60 or c_d60 <= 0:
                continue
            list_date = list_date_map.get(ts_code, "")
            # 剔除新股（上市≤61个交易日）
            if list_date and list_date.replace("-", "") >= new_stock_cutoff.replace("-", ""):
                continue
            ret_60d = (c_d0 - c_d60) / c_d60 * 100
            returns_60d.append({"ts_code": ts_code, "return_60d": ret_60d})

        if not returns_60d:
            return pd.DataFrame(), {}

        rank_df = (
            pd.DataFrame(returns_60d)
            .sort_values("return_60d", ascending=False)
            .head(_RANK_TOP_N)
            .reset_index(drop=True)
        )
        top100_codes = rank_df["ts_code"].tolist()
        logger.info(f"[TrendFollow] {trade_date} 60日动量Top{_RANK_TOP_N}选出 {len(top100_codes)} 只")

        # ── Step 2: 剔除近20日过热（20日涨幅前N名）──────────────────────────
        close_d20 = self._get_hfq_close_on_date(d20_date)
        returns_20d = {}
        for ts_code in top100_codes:
            c_d0_ = close_d0.get(ts_code, 0)
            c_d20_ = close_d20.get(ts_code, 0)
            if c_d0_ and c_d20_ > 0:
                returns_20d[ts_code] = (c_d0_ - c_d20_) / c_d20_ * 100

        hot_codes = set(
            sorted(returns_20d, key=returns_20d.get, reverse=True)[:_HOT_EXCLUDE_TOP_N]
        )
        remaining_codes = [c for c in top100_codes if c not in hot_codes]
        logger.info(
            f"[TrendFollow] 剔除过热{len(hot_codes)}只 "
            f"→ 剩余 {len(remaining_codes)} 只"
        )

        if not remaining_codes:
            return pd.DataFrame(), {}

        # ── Step 3: 剔除 ST ───────────────────────────────────────────────
        remaining_codes = filter_st_stocks(remaining_codes, trade_date)

        # ── Step 4: MA5 > MA30 过滤（批量预加载HFQ，复用公共MA方法）─────────
        # 加载候选股 60 日 HFQ kline（MA30 需要至少 60 个数据点保证精度）
        hfq_df = pd.DataFrame()
        if remaining_codes:
            try:
                hfq_df = get_hfq_kline_range(remaining_codes, d60_date, d0_date)
            except Exception as e:
                logger.warning(f"[TrendFollow] HFQ批量查询失败: {e}")

        ma_passed = []
        for ts_code in remaining_codes:
            try:
                ma_res = self._ma_calc.compute_ma_from_hfq_range(
                    ts_code=ts_code,
                    trade_date=trade_date,
                    trade_dates=trade_dates_60d,
                    ma_periods=[5, 30],
                    preloaded_kline=hfq_df if not hfq_df.empty else None,
                )
                ma5  = ma_res.get(5)
                ma30 = ma_res.get(30)
                if ma5 is None or ma30 is None:
                    # 数据不足，保守保留（不因数据缺失错误排除）
                    ma_passed.append(ts_code)
                    continue
                if ma5 > ma30:
                    ma_passed.append(ts_code)
            except Exception:
                ma_passed.append(ts_code)   # 异常时保守保留

        logger.info(
            f"[TrendFollow] MA5>MA30过滤后剩余 {len(ma_passed)} 只 "
            f"（排除 {len(remaining_codes) - len(ma_passed)} 只）"
        )

        if not ma_passed:
            return pd.DataFrame(), {}

        # ── 构建 candidate_df 和 context ──────────────────────────────────
        candidate_df = pd.DataFrame({"ts_code": ma_passed})
        candidate_df["trade_date"]    = trade_date
        candidate_df["strategy_id"]   = self.strategy_id
        candidate_df["strategy_name"] = self.strategy_name
        candidate_df["sector_name"]   = ""

        # FeatureDataBundle 需要 sector_candidate_map / top3_sectors
        # 本策略不依赖板块，传入单虚拟板块供 bundle 构造
        sector_df = candidate_df[["ts_code"]].copy()
        context = {
            "trade_date":          trade_date,
            "top3_sectors":        [self.strategy_id] * 3,
            "adapt_score":         0.0,
            "sector_candidate_map": {self.strategy_id: sector_df},
            "target_ts_codes":     ma_passed,
        }
        return candidate_df, context

    # ------------------------------------------------------------------ #
    # 核心：买入信号
    # ------------------------------------------------------------------ #

    def _ensure_model(self) -> bool:
        """懒加载模型，加载成功返回 True"""
        if self._model is not None:
            return True
        model_path = self.strategy_params.get("model_path", "")
        if not model_path or not os.path.exists(model_path):
            logger.error(f"[TrendFollow] 模型文件不存在: {model_path}")
            return False
        try:
            with open(model_path, "rb") as f:
                self._model = pickle.load(f)
            logger.info(f"[TrendFollow] 模型加载成功: {model_path}")
            return True
        except Exception as e:
            logger.error(f"[TrendFollow] 模型加载失败: {e}")
            return False

    def _generate_buy_signal(self, trade_date: str) -> Dict[str, str]:
        """
        生成 D+1 开盘买入信号，返回 {ts_code: 'open'}。
        卖出：D+2 收盘（由 runner 推送信息告知用户）。
        """
        self._last_buy_signal_details = []

        if not self._ensure_model():
            logger.error(f"[TrendFollow] {trade_date} 模型未就绪")
            return {}

        # ── 候选池 ──────────────────────────────────────────────────────
        try:
            candidate_df, context = self.build_training_candidates(trade_date)
        except Exception as e:
            logger.error(f"[TrendFollow] {trade_date} 候选池构建失败: {e}", exc_info=True)
            return {}

        if candidate_df.empty:
            logger.warning(f"[TrendFollow] {trade_date} 候选池为空")
            return {}

        # ── 特征计算 ──────────────────────────────────────────────────────
        try:
            bundle = FeatureDataBundle(
                trade_date=trade_date,
                target_ts_codes=context["target_ts_codes"],
                sector_candidate_map=context["sector_candidate_map"],
                top3_sectors=context["top3_sectors"],
                adapt_score=context["adapt_score"],
                load_minute=self.strategy_params["load_minute"],
            )
            engine     = FeatureEngine(self._required_feature_modules)
            feature_df = engine.run_single_date(bundle)
        except Exception as e:
            logger.error(f"[TrendFollow] {trade_date} 特征计算失败: {e}", exc_info=True)
            return {}

        if feature_df.empty:
            logger.warning(f"[TrendFollow] {trade_date} 特征为空")
            return {}

        # ── XGBoost 预测 ──────────────────────────────────────────────────
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
            logger.error(f"[TrendFollow] {trade_date} 模型预测失败: {e}", exc_info=True)
            return {}

        # ── 选股 ──────────────────────────────────────────────────────────
        min_prob = float(self.strategy_params.get("min_prob", 0.0))
        top_k    = int(self.strategy_params["buy_top_k"])
        code_col = "stock_code" if "stock_code" in feature_df.columns else "ts_code"

        selected = (
            feature_df[feature_df["_prob"] >= min_prob]
            .sort_values("_prob", ascending=False)
            .head(top_k)
        )

        if selected.empty:
            logger.info(f"[TrendFollow] {trade_date} 无超过阈值 {min_prob} 的信号")
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
            f"[TrendFollow] {trade_date} 买入信号: {list(buy_signals.keys())} "
            f"（卖出：D+2收盘）"
        )
        return buy_signals
