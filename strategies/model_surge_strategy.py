"""
模型信号拉涨买入策略（ModelSurgeStrategy）
===========================================
历史回测策略版本 — 对标 agent_stats/agents/model_surge_buy.py

时序：D-1 日生成模型信号 → D 日扫描分钟线追高买入

核心逻辑：
  D-1 日：调用 XGBoost 模型生成买入候选
  D 日：截取 09:30-10:30 分钟线，聚合为 5 分钟切片，
        找到第一个切片振幅 (high-low)/open > 4%，
        以该切片的 close 价买入（custom 类型）
  若窗口内无满足条件的切片，当日不买入

买入价：触发切片的 close（模拟该切片结束时追高成交）
无未来函数：仅使用触发切片之前的信息，不依赖日线 close/low
对齐 agent 层：与 model_surge_buy.py 逻辑完全一致
"""
import os
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from config.config import FILTER_BSE_STOCK
from data.data_cleaner import data_cleaner, TushareRateLimitAbort
from features import FeatureEngine, FeatureDataBundle
from features.sector.sector_heat_feature import SectorHeatFeature
from strategies.base_strategy import BaseStrategy
from position_tracker import TrackerConfig
from utils.common_tools import (
    filter_st_stocks,
    get_daily_kline_data,
    get_stocks_in_sector,
    get_trade_dates,
    has_recent_limit_up_batch,
    ensure_limit_list_ths_data,
    calc_limit_up_price,
)
from utils.log_utils import logger
from utils.xgb_compat import safe_predict_proba

# 与 agent 保持一致
_BUY_TOP_K = 6
_MIN_PROB = 0.60
_MIN_AMOUNT = 10_000
_LOAD_MINUTE = True
_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "model", "sector_heat_xgb_v5.2_auc_first.pkl",
)
_SURGE_AMP_PCT   = 0.04   # 5分钟切片振幅阈值：(high-low)/open > 4%
_WINDOW_START    = "09:30"
_WINDOW_END      = "10:30"
_SLICE_MINUTES   = 5


class ModelSurgeStrategy(BaseStrategy):
    """
    模型信号 + 拉涨买入策略
    """

    def __init__(self):
        super().__init__()
        self.strategy_name = "模型信号拉涨买入策略"
        self.strategy_params = {
            "surge_amp_pct": _SURGE_AMP_PCT,
            "window_start":  _WINDOW_START,
            "window_end":    _WINDOW_END,
            "slice_minutes": _SLICE_MINUTES,
            "buy_top_k":     _BUY_TOP_K,
            "sell_type":     "close",
            "min_prob":      _MIN_PROB,
            "load_minute":   _LOAD_MINUTE,
            "model_path":    _MODEL_PATH,
        }

        self._sector_heat = SectorHeatFeature()
        self._feature_engine = FeatureEngine()
        self._model = None

        # 持仓管理
        self.hold_stock_dict: Dict[str, str] = {}

        # 持仓跟踪配置：短线 -5% 止损 + 8% 止盈
        self._tracker_config = TrackerConfig(
            stop_loss_pct=None,   # 故意设高，不做动态止损
            take_profit_pct=None,   # 故意设高，不做动态止盈
            trailing_stop_pct=None,
            max_hold_days=None,
        )

        self.initialize()

    def initialize(self) -> None:
        self.hold_stock_dict.clear()
        self._model = None

    def generate_signal(
        self,
        trade_date: str,
        daily_df: pd.DataFrame,
        positions: Dict[str, any],
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        :return: (buy_signal_map, sell_signal_map)
        """
        # 同步持仓
        for ts_code in list(self.hold_stock_dict.keys()):
            if ts_code not in positions:
                del self.hold_stock_dict[ts_code]
        for ts_code in positions:
            if ts_code not in self.hold_stock_dict:
                self.hold_stock_dict[ts_code] = trade_date

        # 卖出信号
        sell_type = self.strategy_params["sell_type"]
        sell_signal_map: Dict[str, str] = {
            ts: sell_type
            for ts, buy_date in self.hold_stock_dict.items()
            if buy_date < trade_date
        }

        # 买入信号
        buy_signal_map = self._generate_buy_signal(trade_date, daily_df)

        logger.info(
            f"[{self.strategy_name}] {trade_date} | 买入: {list(buy_signal_map.keys())} | 卖出: {list(sell_signal_map.keys())}"
        )
        return buy_signal_map, sell_signal_map

    def get_tracker_config(self) -> Optional[TrackerConfig]:
        return self._tracker_config

    def on_sell_success(self, ts_code: str) -> None:
        self.hold_stock_dict.pop(ts_code, None)

    def _generate_buy_signal(
        self, trade_date: str, daily_df: pd.DataFrame
    ) -> Dict[str, str]:
        """
        1. 获取 D-1 日模型信号
        2. 对每只候选股，拉取 D 日 5 分钟线
        3. 按时间顺序扫描，找第一根振幅 (high-low)/low >= surge_amp_pct 的 K 线
        4. 以该 K 线 close 价买入（custom 类型）
        """
        if not self._ensure_model():
            return {}

        # 获取 D-1 日
        try:
            trade_dates = get_trade_dates("2020-01-01", trade_date)
            cur_idx = trade_dates.index(trade_date)
            if cur_idx == 0:
                return {}
            prev_date = trade_dates[cur_idx - 1]
        except Exception as e:
            logger.debug(f"{trade_date} 无法获取 D-1: {e}")
            return {}

        # 获取 D-1 日线
        prev_daily = get_daily_kline_data(prev_date)
        if prev_daily is None or prev_daily.empty:
            logger.debug(f"{trade_date} D-1({prev_date}) 日线为空")
            return {}

        # 生成 D-1 日模型信号
        signals = self._get_model_signals(prev_date, prev_daily)
        if not signals:
            logger.debug(f"{trade_date} D-1({prev_date}) 无模型信号")
            return {}

        surge_amp_pct  = self.strategy_params["surge_amp_pct"]
        window_start   = self.strategy_params["window_start"]
        window_end     = self.strategy_params["window_end"]
        slice_minutes  = self.strategy_params["slice_minutes"]
        buy_signal_map: Dict[str, str] = {}

        # D 日开盘价 & 前收价（用于一字板过滤，open/pre_close 均为已知数据）
        open_map: Dict[str, float] = {}
        pre_close_map: Dict[str, float] = {}
        for _, row in daily_df.iterrows():
            ts = row["ts_code"]
            open_map[ts] = float(row.get("open", 0) or 0)
            pre_close_map[ts] = float(row.get("pre_close", 0) or 0)

        for sig in signals:
            ts = sig["ts_code"]
            open_p    = open_map.get(ts, 0.0)
            pre_close = pre_close_map.get(ts, 0.0)

            if open_p <= 0:
                continue

            # 过滤一字板：开盘即涨停，分钟线无波动
            if pre_close > 0:
                limit_up = calc_limit_up_price(ts, pre_close)
                if limit_up > 0 and abs(open_p - limit_up) < 0.015:
                    logger.debug(f"{ts} D 日开盘即涨停，跳过")
                    continue

            # 拉取 D 日分钟线
            try:
                min_df = data_cleaner.get_kline_min_by_stock_date(ts, trade_date)
            except TushareRateLimitAbort:
                raise
            except Exception as e:
                logger.debug(f"{ts} {trade_date} 分钟线获取异常: {e}")
                continue

            if min_df is None or min_df.empty:
                logger.debug(f"{ts} {trade_date} 分钟线为空，跳过")
                continue

            # 截取 09:30-10:30 窗口并按时间排序
            try:
                min_df = min_df.copy()
                min_df["_hm"] = pd.to_datetime(min_df["trade_time"]).dt.strftime("%H:%M")
                window = min_df[
                    (min_df["_hm"] >= window_start) & (min_df["_hm"] <= window_end)
                ].sort_values("_hm").reset_index(drop=True)
            except Exception as e:
                logger.debug(f"{ts} {trade_date} 分钟线时间处理异常: {e}")
                continue

            if window.empty:
                continue

            # 聚合为 5 分钟切片，扫描振幅
            buy_price = None
            n, i = len(window), 0
            while i < n:
                end_i    = min(i + slice_minutes, n)
                slice_df = window.iloc[i:end_i]

                slice_high  = float(slice_df["high"].max())
                slice_low   = float(slice_df["low"].min())
                slice_close = float(slice_df.iloc[-1]["close"])

                amplitude = (slice_high - slice_low) / open_p if open_p > 0 else 0
                if amplitude > surge_amp_pct:
                    buy_price = round(slice_close, 2)
                    logger.info(
                        f"{ts} D 日追高触发: "
                        f"切片第{i//slice_minutes+1}段 "
                        f"high={slice_high:.2f} low={slice_low:.2f} "
                        f"振幅={amplitude:.2%} 买入价={buy_price:.2f}"
                    )
                    break
                i = end_i

            if buy_price is not None:
                buy_signal_map[ts] = "custom"
                buy_signal_map[f"{ts}_custom_price"] = buy_price
            else:
                logger.debug(f"{ts} 09:30-10:30 无切片振幅 > {surge_amp_pct:.0%}，不买入")

        logger.info(f"{trade_date} 追高信号: {[k for k in buy_signal_map if '_custom_price' not in k]}")
        return buy_signal_map

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        if not os.path.exists(self.strategy_params["model_path"]):
            logger.error(f"模型文件不存在")
            return False
        try:
            with open(self.strategy_params["model_path"], "rb") as f:
                self._model = pickle.load(f)
            if not hasattr(self._model, "use_label_encoder"):
                self._model.use_label_encoder = False
            return hasattr(self._model, "feature_names_in_")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False

    def _get_model_signals(self, trade_date: str, daily_df: pd.DataFrame) -> List[Dict]:
        """生成模型信号"""
        try:
            top3_result = self._sector_heat.select_top3_hot_sectors(trade_date)
            top3_sectors = top3_result["top3_sectors"]
            adapt_score = top3_result["adapt_score"]
        except Exception as e:
            logger.debug(f"{trade_date} 板块热度失败: {e}")
            return []

        if not top3_sectors:
            return []

        # 候选池构建
        sector_candidate_map: Dict[str, pd.DataFrame] = {}
        for sector in top3_sectors:
            try:
                raw = get_stocks_in_sector(sector)
                if not raw:
                    continue
                ts_codes = [item["ts_code"] for item in raw]
                ts_codes = [ts for ts in ts_codes if not (ts.endswith(".BJ") or ts.split(".")[0].startswith(("83", "87", "88")))]
                if not ts_codes:
                    continue

                ts_codes = filter_st_stocks(ts_codes, trade_date)
                if not ts_codes:
                    continue

                sector_daily = daily_df[daily_df["ts_code"].isin(ts_codes)].copy()
                if sector_daily.empty:
                    continue

                # 涨停基因过滤
                try:
                    end_dt = datetime.strptime(trade_date, "%Y-%m-%d")
                    pre_end = (end_dt - timedelta(days=1)).strftime("%Y-%m-%d")
                    start_60 = (end_dt - timedelta(days=60)).strftime("%Y-%m-%d")
                    dates = get_trade_dates(start_60, pre_end)[-10:]
                    if dates:
                        ensure_limit_list_ths_data(dates[-1])
                        limit_map = has_recent_limit_up_batch(ts_codes, dates[0], dates[-1])
                        keep = [ts for ts, has in limit_map.items() if has]
                        sector_daily = sector_daily[sector_daily["ts_code"].isin(keep)]
                except Exception:
                    pass

                if sector_daily.empty:
                    continue

                # D 日涨停封板过滤
                keep_mask = []
                for _, row in sector_daily.iterrows():
                    pre_close = float(row.get("pre_close") or 0)
                    close = float(row.get("close") or 0)
                    if pre_close <= 0 or close <= 0:
                        keep_mask.append(True)
                        continue
                    lu = calc_limit_up_price(row["ts_code"], pre_close)
                    keep_mask.append(lu <= 0 or close < lu - 0.01)
                sector_daily = sector_daily[keep_mask]

                # 低流动性过滤
                if "amount" in sector_daily.columns:
                    sector_daily = sector_daily[sector_daily["amount"] >= _MIN_AMOUNT]

                sector_candidate_map[sector] = sector_daily
            except Exception as e:
                logger.debug(f"{trade_date}[{sector}] 候选池失败: {e}")

        target_ts_codes = list({
            ts
            for df in sector_candidate_map.values()
            if not df.empty
            for ts in df["ts_code"].tolist()
        })
        if not target_ts_codes:
            return []

        # 特征计算
        try:
            bundle = FeatureDataBundle(
                trade_date=trade_date,
                target_ts_codes=target_ts_codes,
                sector_candidate_map=sector_candidate_map,
                top3_sectors=top3_sectors,
                adapt_score=adapt_score,
                load_minute=self.strategy_params["load_minute"],
            )
            feature_df = self._feature_engine.run_single_date(bundle)
        except Exception as e:
            logger.debug(f"{trade_date} 特征计算失败: {e}")
            return []

        if feature_df.empty:
            return []

        # XGBoost 预测
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
            logger.debug(f"{trade_date} 预测失败: {e}")
            return []

        # 排序选股
        min_prob = self.strategy_params["min_prob"]
        selected = (
            feature_df[feature_df["_prob"] >= min_prob]
            .sort_values("_prob", ascending=False)
            .head(_BUY_TOP_K)
        )

        if selected.empty:
            return []

        # 构建结果
        daily_sub = daily_df[daily_df["ts_code"].isin(selected["stock_code"].tolist())]
        name_map = {row["ts_code"]: str(row.get("name", "")) for _, row in daily_sub.iterrows()}

        result = []
        for _, row in selected.iterrows():
            ts = row["stock_code"]
            result.append({
                "ts_code": ts,
                "stock_name": name_map.get(ts, ""),
                "prob": round(float(row["_prob"]), 4),
            })

        return result
