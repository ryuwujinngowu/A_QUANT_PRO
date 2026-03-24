"""
板块热度选股尾盘买入策略（V2 — XGBoost 驱动版）
==================================================
核心逻辑：
  D 日：SectorHeatFeature 选出 Top3 板块
       → 候选池筛选（ST / 板块 / 涨停基因 / 低流动性过滤）
       → FeatureEngine 计算全量因子
       → XGBoost predict_proba 排序选出 Top-K
       → 收盘价（'close'）尾盘买入

  D+1：开盘卖出（sell_type='open'），持股不超过 1 个交易日

过滤逻辑与 dataset.py 完全对齐，保证训练/推断口径一致。
"""
import os
import pickle
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config.config import FILTER_BSE_STOCK, FILTER_STAR_BOARD, FILTER_688_BOARD
from data.data_cleaner import data_cleaner
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
)
from utils.log_utils import logger
from risk_penalty_core import (
    RiskPenaltyConfig,
    should_filter_high_risk_stock,
    should_stop_loss,        # 预留：D+1开盘执行层调用（当前策略层仅引入，由执行引擎使用）
    calc_strategy_weight_discount,
)

# 与 dataset.py 对齐的低流动性阈值（单位：千元）
_MIN_AMOUNT = 10_000   # 1000 万元


class SectorHeatStrategy(BaseStrategy):
    """
    板块热度 XGBoost 选股策略

    依赖前置：
        1. 已运行 python learnEngine/dataset.py 生成训练集
        2. 已运行 python train.py 训练并保存模型
    """

    def __init__(self):
        super().__init__()
        self.strategy_name = "板块热度XGBoost选股尾盘买入策略"
        self.strategy_params = {
            "buy_top_k":   6,        # 每日最多买入 N 只
            "sell_type":   "close",   # D+1 卖出类型：open=次日开盘，close=次日收盘
            "min_prob":    0.6,      # 最低买入概率阈值（0 = 不过滤）
            "load_minute": True,     # 是否加载分钟线（保证特征与训练口径一致）
            "model_path": os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "model", "sector_heat_xgb_model.pkl",
            ),
        }

        # 新架构组件（与 dataset.py 使用同一套 FeatureEngine）
        self._sector_heat   = SectorHeatFeature()
        self._feature_engine = FeatureEngine()

        # 模型（懒加载，首次调用 generate_signal 时加载）
        self._model = None

        # 风险惩罚配置（使用选股模型专用预设，所有惩罚参数统一由 risk_penalty_core 管理）
        self._risk_config: RiskPenaltyConfig = RiskPenaltyConfig.for_strategy_model()

        # 持仓管理：{ts_code: buy_date}，用于严格执行 D+1 卖出规则
        self.hold_stock_dict: Dict[str, str] = {}

        # 持仓跟踪配置：D+1 短线策略 -5% 固定止损 + 8% 止盈
        self._tracker_config = TrackerConfig(
            stop_loss_pct=-0.05,         # -5% 止损（短线策略止损线较紧）
            take_profit_pct=0.08,        # +8% 止盈
            trailing_stop_pct=None,      # 短线不启用移动止损
            max_hold_days=None,          # 由策略自身的 D+1 逻辑控制
        )

        self.initialize()

    # ------------------------------------------------------------------ #
    # BaseStrategy 强制接口
    # ------------------------------------------------------------------ #

    def initialize(self) -> None:
        """回测启动 / 重置时由引擎自动调用"""
        self.clear_signal()
        self.hold_stock_dict.clear()
        self._model = None  # 重置模型，下次使用时重新加载

    def generate_signal(
        self,
        trade_date: str,
        daily_df: pd.DataFrame,
        positions: Dict[str, any],
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        引擎每日调用两次：
          第 1 次 → 取 return[0] 作为买入信号，格式 {ts_code: buy_type}
          第 2 次 → 取 return[1] 作为卖出信号，格式 {ts_code: sell_type}
        两次均返回完整 tuple，引擎分别取所需部分。

        :return: (buy_signal_map, sell_signal_map)
        """
        # ---- 同步持仓字典 ------------------------------------------------
        # 移除已不在持仓中的股票（已卖出）
        for ts_code in list(self.hold_stock_dict.keys()):
            if ts_code not in positions:
                del self.hold_stock_dict[ts_code]
        # 记录持仓中尚未登记的股票
        for ts_code in positions:
            if ts_code not in self.hold_stock_dict:
                self.hold_stock_dict[ts_code] = trade_date

        # ---- 卖出信号：买入日 < 当前日 → 次日卖出 -------------------------
        sell_type = self.strategy_params["sell_type"]
        sell_signal_map: Dict[str, str] = {
            ts: sell_type
            for ts, buy_date in self.hold_stock_dict.items()
            if buy_date < trade_date
        }

        # ---- 买入信号 -----------------------------------------------------
        buy_signal_map = self._generate_buy_signal(trade_date, daily_df)

        logger.info(
            f"[{self.strategy_name}] {trade_date} "
            f"| 买入: {list(buy_signal_map.keys())} "
            f"| 卖出: {list(sell_signal_map.keys())}"
        )
        return buy_signal_map, sell_signal_map

    # ------------------------------------------------------------------ #
    # 可选引擎回调
    # ------------------------------------------------------------------ #

    def get_tracker_config(self):
        """返回持仓跟踪配置，由回测引擎自动执行分钟线扫描"""
        return self._tracker_config

    def on_sell_success(self, ts_code: str) -> None:
        """卖出成功后从内部持仓字典移除"""
        self.hold_stock_dict.pop(ts_code, None)

    # ------------------------------------------------------------------ #
    # 核心：买入信号生成
    # ------------------------------------------------------------------ #

    def _generate_buy_signal(
        self, trade_date: str, daily_df: pd.DataFrame
    ) -> Dict[str, str]:
        """
        返回 {ts_code: 'close'}（尾盘买入）
        返回空 dict = 当日不买入
        """
        # ── 模型加载 ──────────────────────────────────────────────────────
        if not self._ensure_model():
            logger.error(f"{trade_date} 模型未就绪，跳过买入")
            return {}

        # ── Step 1: Top3 板块 + 轮动分 ────────────────────────────────────
        try:
            top3_result  = self._sector_heat.select_top3_hot_sectors(trade_date)
            top3_sectors = top3_result["top3_sectors"]
            adapt_score  = top3_result["adapt_score"]
        except Exception as e:
            logger.error(f"{trade_date} 板块热度计算失败: {e}", exc_info=True)
            return {}

        if not top3_sectors:
            logger.warning(f"{trade_date} Top3 板块为空，跳过买入")
            return {}

        logger.info(f"{trade_date} Top3={top3_sectors} | adapt_score={adapt_score}")
        # if adapt_score >=50:
        #     logger.warning(f"{adapt_score} 轮动过快")
        #     return {}

        # ── ST 数据入库（非致命，异常不中断选股）────────────────────────────
        try:
            data_cleaner.insert_stock_st(trade_date=trade_date.replace("-", ""))
        except Exception as e:
            logger.warning(f"{trade_date} ST 数据入库失败（忽略）: {e}")

        # ── Step 2: 候选池构建 ────────────────────────────────────────────
        sector_candidate_map, target_ts_codes = self._build_candidate_pool(
            trade_date, daily_df, top3_sectors
        )
        if not target_ts_codes:
            logger.warning(f"{trade_date} 候选池为空，跳过买入")
            return {}

        # ── Step 3: 特征计算（与训练口径完全一致）────────────────────────────
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
            logger.error(f"{trade_date} 特征计算失败: {e}", exc_info=True)
            return {}

        if feature_df.empty:
            logger.warning(f"{trade_date} 特征计算结果为空，跳过买入")
            return {}

        # ── Step 4: XGBoost 预测 ──────────────────────────────────────────
        try:
            expected_cols = list(self._model.feature_names_in_)
            X = (
                feature_df
                .reindex(columns=expected_cols, fill_value=0)
                # DB 返回的 Decimal/object 列在此强制转为 float（训练 CSV 读回是 float，推断时须对齐）
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0)
                .replace([np.inf, -np.inf], 0)
            )
            probs = self._model.predict_proba(X)[:, 1]
            feature_df = feature_df.copy()
            feature_df["_prob"] = probs
        except Exception as e:
            logger.error(f"{trade_date} 模型预测失败: {e}", exc_info=True)
            return {}

        # ── Step 5: 排序选股 ──────────────────────────────────────────────
        min_prob = float(self.strategy_params.get("min_prob", 0.0))
        top_k    = int(self.strategy_params["buy_top_k"])

        selected = (
            feature_df[feature_df["_prob"] >= min_prob]
            .sort_values("_prob", ascending=False)
            .head(top_k)
        )

        if selected.empty:
            logger.info(f"{trade_date} 无股票超过概率阈值 {min_prob}，跳过买入")
            return {}

        # ── Step 5b: 风险惩罚过滤（高风险市场环境下过滤低流动性个股）────────
        # 从 feature_df 读取宏观数据（若被 EXCLUDE_PATTERNS 过滤则取 0，自动跳过此过滤）
        market_limit_down = int(feature_df.get("market_limit_down_count", pd.Series([0])).iloc[0]
                                if "market_limit_down_count" in feature_df.columns else 0)
        market_index_chg  = (float(feature_df["index_sh_pct_chg"].iloc[0])
                             if "index_sh_pct_chg" in feature_df.columns else None)

        before_risk_filter = len(selected)
        selected = selected[
            ~selected.apply(
                lambda row: should_filter_high_risk_stock(
                    market_limit_down_count=market_limit_down,
                    stock_amount=float(row.get("stock_amount_d0", self._risk_config.min_liquidity_amount + 1)),
                    config=self._risk_config,
                    market_index_chg=market_index_chg,
                ),
                axis=1,
            )
        ]
        if len(selected) < before_risk_filter:
            logger.info(
                f"{trade_date} 风险过滤移除 {before_risk_filter - len(selected)} 只"
                f"（高风险市场+低流动性）| 跌停:{market_limit_down} 只"
            )

        if selected.empty:
            logger.info(f"{trade_date} 风险过滤后候选为空，跳过买入")
            return {}

        # 返回格式 {ts_code: 'close'}，引擎以尾盘收盘价买入
        buy_signal_map = {row["stock_code"]: "close" for _, row in selected.iterrows()}
        logger.info(
            f"{trade_date} 最终买入 {len(buy_signal_map)} 只: "
            + " | ".join(
                f"{row['stock_code']}(p={row['_prob']:.3f})"
                for _, row in selected.iterrows()
            )
        )
        return buy_signal_map

    # ------------------------------------------------------------------ #
    # 模型加载（懒加载 + 属性校验）
    # ------------------------------------------------------------------ #

    def _ensure_model(self) -> bool:
        """加载模型并校验 feature_names_in_ 属性（供 reindex 对齐列序）"""
        if self._model is not None:
            return True
        path = self.strategy_params["model_path"]
        if not os.path.exists(path):
            logger.error(
                f"模型文件不存在: {path}\n"
                f"请先运行：python learnEngine/dataset.py && python train.py"
            )
            return False
        try:
            with open(path, "rb") as f:
                self._model = pickle.load(f)
            if not hasattr(self._model, "feature_names_in_"):
                logger.error(
                    "模型缺少 feature_names_in_ 属性。"
                    "请确认 train.py 使用 pandas DataFrame 作为 X 输入训练 XGBClassifier。"
                )
                self._model = None
                return False
            logger.info(
                f"模型加载成功: {path} "
                f"| 特征数: {len(self._model.feature_names_in_)}"
            )
            return True
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            self._model = None
            return False

    # ------------------------------------------------------------------ #
    # 候选池构建（逻辑与 dataset.py 完全对齐）
    # ------------------------------------------------------------------ #

    def _build_candidate_pool(
        self,
        trade_date: str,
        daily_df: pd.DataFrame,
        top3_sectors: List[str],
    ) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
        """
        逐板块过滤，返回 (sector_candidate_map, target_ts_codes)
        过滤顺序：板块 → ST → 日线数据 → 近10日涨停基因 → D日涨停封板 → 低流动性
        """
        sector_candidate_map: Dict[str, pd.DataFrame] = {}

        for sector in top3_sectors:
            logger.info(f"候选池 | 处理板块: {sector}")
            try:
                raw = get_stocks_in_sector(sector)
                if not raw:
                    logger.warning(f"[{sector}] 无股票，跳过")
                    sector_candidate_map[sector] = pd.DataFrame()
                    continue

                ts_codes = [item["ts_code"] for item in raw]

                # 1. 板块过滤（BSE / 科创 / 创业板）
                ts_codes = self._filter_ts_code_by_board(ts_codes)
                if not ts_codes:
                    sector_candidate_map[sector] = pd.DataFrame()
                    continue

                # 2. ST 过滤
                ts_codes = filter_st_stocks(ts_codes, trade_date)
                if not ts_codes:
                    sector_candidate_map[sector] = pd.DataFrame()
                    continue

                # 3. 日线数据过滤（当日无数据的股票排除）
                sector_daily = daily_df[daily_df["ts_code"].isin(ts_codes)].copy()
                if sector_daily.empty:
                    sector_candidate_map[sector] = pd.DataFrame()
                    continue

                # 4. 近 10 日涨停基因过滤
                candidates    = sector_daily["ts_code"].unique().tolist()
                limit_up_map  = self._check_limit_up_gene(candidates, trade_date, day_count=10)
                keep          = [ts for ts, has in limit_up_map.items() if has]
                sector_daily  = sector_daily[sector_daily["ts_code"].isin(keep)]
                if sector_daily.empty:
                    sector_candidate_map[sector] = pd.DataFrame()
                    continue

                # 5. D 日涨停封板过滤（尾盘无法买入）
                sector_daily = self._filter_limit_up_on_d0(sector_daily)

                # 6. 低流动性过滤（与 dataset.py 阈值对齐）
                if "amount" in sector_daily.columns:
                    sector_daily = sector_daily[sector_daily["amount"] >= _MIN_AMOUNT]

                sector_candidate_map[sector] = sector_daily
                logger.info(f"[{sector}] 最终候选股: {len(sector_daily)}")

            except Exception as e:
                logger.error(f"[{sector}] 候选池构建失败: {e}", exc_info=True)
                sector_candidate_map[sector] = pd.DataFrame()

        # 跨板块去重，汇总候选股票列表
        target_ts_codes = list({
            ts
            for df in sector_candidate_map.values()
            if not df.empty
            for ts in df["ts_code"].tolist()
        })
        return sector_candidate_map, target_ts_codes

    # ------------------------------------------------------------------ #
    # 过滤工具方法
    # ------------------------------------------------------------------ #

    def _filter_ts_code_by_board(self, ts_code_list: List[str]) -> List[str]:
        """过滤北交所 / 科创板 / 创业板"""
        result = []
        for ts in ts_code_list:
            if not ts:
                continue
            if FILTER_BSE_STOCK and (ts.endswith(".BJ") or ts.startswith(("83", "87", "88"))):
                continue
            if FILTER_688_BOARD and ts.startswith("688"):
                continue
            if FILTER_STAR_BOARD and ts.startswith(("300", "301", "302")) and ts.endswith(".SZ"):
                continue
            result.append(ts)
        return result

    def _filter_limit_up_on_d0(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        过滤 D 日涨停封板（close ≥ limit_up - 0.01，尾盘买不进去）
        保守策略：价格数据异常时保留（avoid 误过滤）
        """
        if df.empty:
            return df
        keep = []
        for _, row in df.iterrows():
            pre_close = float(row.get("pre_close") or 0)
            close     = float(row.get("close")     or 0)
            if pre_close <= 0 or close <= 0:
                keep.append(True)    # 数据异常，保守保留
                continue
            lu = self.calc_limit_up_price(row["ts_code"], pre_close)
            keep.append(lu <= 0 or close < lu - 0.01)
        filtered = df[keep].copy()
        removed  = len(df) - len(filtered)
        if removed:
            logger.info(f"[D日涨停过滤] 涨停封板股已过滤: {removed} 只")
        return filtered

    def _check_limit_up_gene(
        self,
        ts_code_list: List[str],
        end_date: str,
        day_count: int = 10,
    ) -> Dict[str, bool]:
        """
        判断近 N 个交易日内是否有涨停（涨停基因过滤）
        保守策略：数据获取失败时全返回 True（不误删股票）
        """
        if not ts_code_list or day_count <= 0 or not end_date:
            return {ts: False for ts in ts_code_list}
        try:
            if len(end_date) == 8 and end_date.isdigit():
                end_dt = datetime.strptime(end_date, "%Y%m%d")
            else:
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            pre_end  = (end_dt - timedelta(days=1)).strftime("%Y-%m-%d")
            start_60 = (end_dt - timedelta(days=60)).strftime("%Y-%m-%d")
            dates    = get_trade_dates(start_60, pre_end)[-day_count:]
            if len(dates) < day_count:
                logger.warning(f"可回溯交易日不足 {day_count} 个，返回全 False（中性值）")
                return {ts: False for ts in ts_code_list}
        except Exception as e:
            logger.error(f"获取交易日失败: {e}")
            return {ts: False for ts in ts_code_list}

        try:
            if not dates:
                logger.warning("回溯交易日为空，返回全 False（中性值）")
                return {ts: False for ts in ts_code_list}

            # 确保 limit_list_ths 有最新数据
            ensure_limit_list_ths_data(dates[-1])
            result = has_recent_limit_up_batch(
                ts_code_list=ts_code_list,
                start_date=dates[0],
                end_date=dates[-1],
            )
        except Exception as e:
            logger.error(f"涨停基因判断失败: {e}，返回全 False（中性值）")
            return {ts: False for ts in ts_code_list}

        logger.info(
            f"涨停基因判断 | 候选: {len(ts_code_list)} "
            f"| 有涨停基因: {sum(result.values())}"
        )
        return result
