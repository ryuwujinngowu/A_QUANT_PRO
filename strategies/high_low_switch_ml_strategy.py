"""
高低切 XGBoost 选股策略（HighLowSwitchMLStrategy）
===============================================
核心逻辑：
  工程执行时点通常为 D+1 凌晨（与 sector_heat 策略相同）：
       → D 日收盘后，基于 D 日完整盘面数据构建候选池：
           Source A：D 日涨停池（首板/二板，排除三板+）
           Source B：D 日近5日涨幅 10%~35% 的个股（无涨停要求）
       → 过滤：主板 + 非ST + 低流动性
       → FeatureDataBundle 计算全量因子（零未来函数）
       → XGBoost predict_proba 排序选出 Top-K
       → 输出下一个交易日开盘买入信号

卖出逻辑（多条件，任一触发即卖，次日开盘执行）：
  1. 强制止损：日内最低价触及买入价 -10%（T+1，次日开盘无条件止损）
  2. 动态止损：【合规无未来】任意交易日当日收盘跌幅 > -5% → 次日开盘卖出
  3. 均线止盈：收盘价跌破 10 日均线 → 次日开盘卖出
  4. 最大持仓：持仓达 10 个交易日 → 次日开盘卖出

时序对齐原则（与 dataset.py 训练集完全一致）：
  - trade_date=D 的候选池、特征均基于 D 日收盘后已知信息
  - 标签：D+1 日内收益（label1_3pct）
  - 买入执行在 D+1 开盘
  - 卖出信号当日生成 → 次日开盘执行（无未来函数）
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
from position_tracker import TrackerConfig
from utils.common_tools import (
    filter_st_stocks,
    get_daily_kline_data,
    get_kline_day_range,
    get_limit_list_ths,
    get_limit_step,
    get_trade_dates,
    ensure_limit_list_ths_data,
)
from utils.log_utils import logger
from utils.xgb_compat import safe_predict_proba


# ── 策略常量 ──────────────────────────────────────────────────────────
_STOP_LOSS_PCT = -0.10          # 硬止损：-10%
_DYNAMIC_DAY_DROP_PCT = -0.05   # 动态止损：单日收盘跌幅超-5%（修正命名，更清晰）
_MAX_HOLD_DAYS = 10             # 最大持仓天数
_GAIN_5D_MIN = 0.10             # Source B 5日涨幅下限
_GAIN_5D_MAX = 0.35             # Source B 5日涨幅上限
_MIN_AMOUNT = 20_0000            # 低流动性阈值（千元，= 1000万）


def _is_main_board(ts_code: str) -> bool:
    """主板判断（排除创业板/科创板/北交所）"""
    if ts_code.endswith(".BJ"):
        return False
    if ts_code.startswith(("300", "301", "302")) and ts_code.endswith(".SZ"):
        return False
    if ts_code.startswith("688"):
        return False
    return True


class HighLowSwitchMLStrategy(BaseStrategy):
    """
    高低切 XGBoost 选股策略

    依赖前置：
        1. 已运行 python learnEngine/dataset.py 生成训练集
        2. 已运行 python train.py 训练并保存模型
    """

    @property
    def strategy_id(self) -> str:
        return "high_low_switch"

    def __init__(self):
        super().__init__()
        self._tracker_config = TrackerConfig(
            stop_loss_pct=_STOP_LOSS_PCT,
            take_profit_pct=99.0,
            max_hold_days=_MAX_HOLD_DAYS,
        )
        self.strategy_name = "高低切XGBoost盘前选股，开盘买入策略"
        self.strategy_params = {
            "buy_top_k":   5,
            "sell_type":   "open",  # 固定：次日开盘卖出，无未来函数
            "min_prob":    0.6,
            "load_minute": True,
            "model_path":  self._get_runtime_model_path(),
        }

        self._feature_engine: Optional[FeatureEngine] = None
        self._model = None
        self._required_feature_modules = None

        # 持仓管理：{ts_code: {"buy_date": str, "buy_price": float}}
        self.hold_stock_dict: Dict[str, Dict] = {}
        self._last_buy_signal_details: List[Dict] = []

        self.initialize()

    # ------------------------------------------------------------------ #
    # BaseStrategy 强制接口
    # ------------------------------------------------------------------ #

    def _get_runtime_model_path(self) -> str:
        return cfg.get_strategy_runtime_model_path(self.strategy_id)

    def initialize(self) -> None:
        self.clear_signal()
        self.hold_stock_dict.clear()
        self._model = None
        self._feature_engine = None
        self._required_feature_modules = None
        self._last_buy_signal_details = []

    def get_last_buy_signal_details(self) -> List[Dict]:
        return list(self._last_buy_signal_details)

    def generate_signal(
        self,
        trade_date: str,
        daily_df: pd.DataFrame,
        positions: Dict[str, any],
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        # ---- 同步持仓字典 ------------------------------------------------
        for ts_code in list(self.hold_stock_dict.keys()):
            if ts_code not in positions:
                del self.hold_stock_dict[ts_code]

        for ts_code, pos in positions.items():
            if ts_code not in self.hold_stock_dict:
                buy_price = getattr(pos, "avg_cost", 0.0) or getattr(pos, "buy_price", 0.0)
                self.hold_stock_dict[ts_code] = {
                    "buy_date": trade_date,
                    "buy_price": float(buy_price) if buy_price else 0.0,
                }

        # ---- 卖出信号：固定生成 open 类型（次日开盘卖出，无未来函数） ----
        sell_signal_map = self._check_sell_conditions(trade_date, daily_df)

        # ---- 买入信号 ----------------------------------------------------
        buy_signal_map = self._generate_buy_signal(trade_date, daily_df)

        logger.info(
            f"[{self.strategy_name}] {trade_date} "
            f"| 买入: {list(buy_signal_map.keys())} "
            f"| 卖出(次日开盘): {list(sell_signal_map.keys())}"
        )
        return buy_signal_map, sell_signal_map

    # ------------------------------------------------------------------ #
    # 可选引擎回调
    # ------------------------------------------------------------------ #

    def get_tracker_config(self):
        return self._tracker_config

    def on_sell_success(self, ts_code: str) -> None:
        self.hold_stock_dict.pop(ts_code, None)

    def supports_ml_training(self) -> bool:
        return True

    def get_training_label_target(self) -> str:
        return "label1_3pct"

    def get_model_registry_info(self) -> Dict[str, str]:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return {
            "strategy_id": self.strategy_id,
            "archive_dir": os.path.join(base_dir, "model", self.strategy_id),
            "runtime_dir": cfg.get_strategy_runtime_model_dir(self.strategy_id),
        }

    # ------------------------------------------------------------------ #
    # 训练候选池构建（dataset.py 调用）
    # ------------------------------------------------------------------ #

    def build_training_candidates(
        self,
        trade_date: str,
        daily_df: pd.DataFrame = None,
    ) -> Tuple[pd.DataFrame, Dict[str, any]]:
        context: Dict[str, any] = {
            "trade_date": trade_date,
            "target_ts_codes": [],
        }

        if daily_df is None:
            daily_df = get_daily_kline_data(trade_date)
        if daily_df.empty:
            return pd.DataFrame(), context

        target_ts_codes = self._build_candidate_pool(trade_date, daily_df)
        if not target_ts_codes:
            return pd.DataFrame(), context

        context["target_ts_codes"] = target_ts_codes

        candidate_df = pd.DataFrame({
            "ts_code": target_ts_codes,
            "strategy_id": self.strategy_id,
            "strategy_name": self.strategy_name,
            "sector_name": "",
            "trade_date": trade_date,
        })
        return candidate_df, context

    def build_feature_bundle_from_context(self, context: Dict[str, any]):
        return build_bundle_from_context(
            context,
            load_minute=self.strategy_params.get("load_minute", True),
            required_modules=self._required_feature_modules,
        )

    # ------------------------------------------------------------------ #
    # 候选池构建
    # ------------------------------------------------------------------ #

    def _build_candidate_pool(
        self,
        trade_date: str,
        daily_df: pd.DataFrame,
    ) -> List[str]:
        if daily_df.empty:
            return []

        all_codes = daily_df["ts_code"].unique().tolist()
        main_codes = [ts for ts in all_codes if _is_main_board(ts)]
        if not main_codes:
            return []

        try:
            normal_codes = set(filter_st_stocks(main_codes, trade_date))
        except Exception as e:
            logger.warning(f"[高低切ML][{trade_date}] ST 过滤失败: {e}")
            return []
        if not normal_codes:
            return []

        # Source A: 首板/二板
        source_a = set()
        try:
            td_fmt = trade_date.replace("-", "")
            ensure_limit_list_ths_data(td_fmt)
            limit_df = get_limit_list_ths(trade_date, limit_type="涨停池")
            if limit_df is not None and not limit_df.empty:
                step_df = get_limit_step(trade_date)
                step_map: Dict[str, int] = {}
                if not step_df.empty and "ts_code" in step_df.columns and "nums" in step_df.columns:
                    step_map = dict(zip(step_df["ts_code"], step_df["nums"].astype(int)))

                for ts in limit_df["ts_code"].tolist():
                    if ts not in normal_codes:
                        continue
                    cons = step_map.get(ts, 1)
                    if cons <= 2:
                        source_a.add(ts)
        except Exception as e:
            logger.warning(f"[高低切ML][{trade_date}] Source A 涨停池构建失败: {e}")

        # Source B: 5日涨幅10-35%
        source_b = self._get_5d_gain_stocks(trade_date, daily_df, list(normal_codes))

        all_candidates = list(source_a | source_b)

        # 流动性过滤
        if all_candidates and "amount" in daily_df.columns:
            amt_map = dict(zip(daily_df["ts_code"], daily_df["amount"]))
            all_candidates = [
                ts for ts in all_candidates
                if float(amt_map.get(ts, 0) or 0) >= _MIN_AMOUNT
            ]

        logger.info(
            f"[高低切ML][{trade_date}] 候选池: {len(all_candidates)} 只 "
            f"（涨停首/二板={len(source_a)} | 5日涨10-35%={len(source_b)}）"
        )
        return all_candidates

    def _get_5d_gain_stocks(
        self,
        trade_date: str,
        daily_df: pd.DataFrame,
        candidate_codes: List[str],
    ) -> set:
        if not candidate_codes:
            return set()

        d_df = daily_df[daily_df["ts_code"].isin(candidate_codes)]
        close_d_map = dict(zip(d_df["ts_code"], d_df["close"].astype(float)))
        if not close_d_map:
            return set()

        try:
            d_date = datetime.strptime(trade_date, "%Y-%m-%d")
            start = (d_date - timedelta(days=20)).strftime("%Y-%m-%d")
            dates = get_trade_dates(start, trade_date)
            if len(dates) < 6:
                return set()
            d5_date = dates[-6]
        except Exception as e:
            logger.warning(f"[高低切ML][{trade_date}] 获取 D-5 交易日失败: {e}")
            return set()

        try:
            kline_d5 = get_kline_day_range(list(close_d_map.keys()), d5_date, d5_date)
        except Exception as e:
            logger.warning(f"[高低切ML][{trade_date}] 查询 D-5 数据失败: {e}")
            return set()
        if kline_d5.empty:
            return set()

        close_d5_map = dict(zip(kline_d5["ts_code"], kline_d5["close"].astype(float)))

        result = set()
        for ts, close_d in close_d_map.items():
            close_d5 = close_d5_map.get(ts, 0)
            if close_d5 <= 0 or close_d <= 0:
                continue
            gain = (close_d - close_d5) / close_d5
            if _GAIN_5D_MIN <= gain < _GAIN_5D_MAX:
                result.add(ts)

        return result

    # ------------------------------------------------------------------ #
    # 卖出条件检查（核心修正：严格时序 + 无未来函数 + 引擎兼容）
    # ------------------------------------------------------------------ #

    def _check_sell_conditions(
        self,
        trade_date: str,
        daily_df: pd.DataFrame,
    ) -> Dict[str, str]:
        """
        所有卖出信号统一为：open（次日开盘卖出）
        触发条件：
          1. 持仓超期
          2. 日内低点触及-10%止损
          3. 【修正】当日单日跌幅 > -5%（动态止损）
          4. 收盘跌破MA10
        """
        sell_map: Dict[str, str] = {}
        if not self.hold_stock_dict:
            return sell_map

        held_codes = list(self.hold_stock_dict.keys())
        ma10_map = self._batch_get_ma10(held_codes, trade_date)

        for ts_code, info in self.hold_stock_dict.items():
            buy_date = info["buy_date"]
            buy_price = info["buy_price"]

            # T+1 规则：买入当日不卖出
            if buy_date >= trade_date:
                continue

            # 条件1：最大持仓天数
            try:
                hold_days = len(get_trade_dates(buy_date, trade_date)) - 1
            except Exception:
                hold_days = 0
            if hold_days >= _MAX_HOLD_DAYS:
                sell_map[ts_code] = "open"
                logger.debug(f"[高低切ML] {ts_code} 持仓{hold_days}天，次日开盘卖出")
                continue

            # 获取当日行情数据
            row = daily_df[daily_df["ts_code"] == ts_code]
            if row.empty or buy_price <= 0:
                continue
            close_d = float(row["close"].iloc[0])
            low_d = float(row["low"].iloc[0])
            pct_chg = float(row["pct_chg"].iloc[0]) / 100  # 当日涨跌幅（小数）

            # 条件2：硬止损（日内低点-10%）
            stop_price = buy_price * (1 + _STOP_LOSS_PCT)
            if low_d <= stop_price:
                sell_map[ts_code] = "open"
                logger.debug(f"[高低切ML] {ts_code} 触及止损线，次日开盘卖出")
                continue

            # 条件3：【最终合规版】动态止损 → 当日收盘跌幅超-5%，次日开盘卖
            if pct_chg < _DYNAMIC_DAY_DROP_PCT:
                sell_map[ts_code] = "open"
                logger.debug(f"[高低切ML] {ts_code} 当日跌幅{pct_chg*100:.1f}%，次日开盘动态止损")
                continue

            # 条件4：MA10止盈
            ma10 = ma10_map.get(ts_code, 0.0)
            if ma10 > 0 and close_d < ma10:
                sell_map[ts_code] = "open"
                logger.debug(f"[高低切ML] {ts_code} 跌破MA10，次日开盘卖出")

        return sell_map

    def _batch_get_ma10(
        self, ts_codes: List[str], trade_date: str
    ) -> Dict[str, float]:
        if not ts_codes:
            return {}
        try:
            d_date = datetime.strptime(trade_date, "%Y-%m-%d")
            start = (d_date - timedelta(days=25)).strftime("%Y-%m-%d")
            kline = get_kline_day_range(ts_codes, start, trade_date)
        except Exception as e:
            logger.warning(f"[高低切ML] MA10 查询失败: {e}")
            return {}
        if kline.empty:
            return {}

        result: Dict[str, float] = {}
        for ts in ts_codes:
            ts_kline = kline[kline["ts_code"] == ts].sort_values("trade_date")
            closes = ts_kline["close"].astype(float)
            if len(closes) >= 10:
                result[ts] = float(closes.tail(10).mean())
        return result

    # ------------------------------------------------------------------ #
    # 核心：买入信号生成
    # ------------------------------------------------------------------ #

    def _generate_buy_signal(
        self, trade_date: str, daily_df: pd.DataFrame
    ) -> Dict[str, str]:
        if not self._ensure_model():
            logger.error(f"{trade_date} 模型未就绪，跳过买入")
            self._last_buy_signal_details = []
            return {}

        try:
            candidate_df, context = self.build_training_candidates(trade_date, daily_df=daily_df)
        except Exception as e:
            logger.error(f"{trade_date} 候选池构建失败: {e}", exc_info=True)
            self._last_buy_signal_details = []
            return {}

        if candidate_df.empty:
            logger.warning(f"{trade_date} 候选池为空，跳过买入")
            self._last_buy_signal_details = []
            return {}

        try:
            bundle = self.build_feature_bundle_from_context(context)
            if bundle is None:
                logger.warning(f"{trade_date} Feature bundle 为空，跳过买入")
                self._last_buy_signal_details = []
                return {}
            if self._feature_engine is None:
                self._feature_engine = FeatureEngine()
            feature_df = self._feature_engine.run_single_date(bundle)
        except Exception as e:
            logger.error(f"{trade_date} 特征计算失败: {e}", exc_info=True)
            self._last_buy_signal_details = []
            return {}

        if feature_df.empty:
            logger.warning(f"{trade_date} 特征计算结果为空，跳过买入")
            self._last_buy_signal_details = []
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
            logger.error(f"{trade_date} 模型预测失败: {e}", exc_info=True)
            self._last_buy_signal_details = []
            return {}

        held_set = set(self.hold_stock_dict.keys())
        if held_set:
            feature_df = feature_df[~feature_df["stock_code"].isin(held_set)]

        min_prob = float(self.strategy_params.get("min_prob", 0.0))
        top_k = int(self.strategy_params["buy_top_k"])

        selected = (
            feature_df[feature_df["_prob"] >= min_prob]
            .sort_values("_prob", ascending=False)
            .head(top_k)
        )

        if selected.empty:
            self._last_buy_signal_details = []
            return {}

        self._last_buy_signal_details = [
            {
                "stock_code": row["stock_code"],
                "buy_type": "open",
                "prob": round(float(row["_prob"]), 4),
            }
            for _, row in selected.iterrows()
        ]

        buy_signal_map = {item["stock_code"]: "open" for item in self._last_buy_signal_details}
        logger.info(
            f"{trade_date} 最终买入 {len(buy_signal_map)} 只: "
            + " | ".join(
                f"{item['stock_code']}(p={item['prob']:.3f})"
                for item in self._last_buy_signal_details
            )
        )
        return buy_signal_map

    # ------------------------------------------------------------------ #
    # 模型加载
    # ------------------------------------------------------------------ #

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        path = self.strategy_params["model_path"]
        if not os.path.exists(path):
            logger.error(f"模型文件不存在: {path}")
            return False
        try:
            with open(path, "rb") as f:
                self._model = pickle.load(f)
            if not hasattr(self._model, "use_label_encoder"):
                self._model.use_label_encoder = False
            if not hasattr(self._model, "feature_names_in_"):
                logger.error("模型缺少 feature_names_in_ 属性。")
                self._model = None
                return False
            logger.info(
                f"模型加载成功: {path} | 特征数: {len(self._model.feature_names_in_)}"
            )
            return True
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            self._model = None
            return False