"""
板块热度选股开盘买入策略（V3 — XGBoost 驱动版，无未来函数）
==================================================
核心逻辑：
  工程执行时点通常为 D+1 凌晨：
       → 等待 Tushare / 入库链路把 D 日完整盘面写齐
       → 触发策略时，代码里的“上一完整交易日盘面”在业务语义上对应 D 日收盘后数据
       → SectorHeatFeature 以该完整盘面为基准选出 Top3 板块
       → 候选池筛选：基于该完整盘面做 ST / 板块 / 涨停基因 / 低流动性 / 封板过滤
       → FeatureDataBundle 计算全量因子（零未来函数）
       → XGBoost predict_proba 排序选出 Top-K
       → 输出下一个交易日开盘买入信号

关键原则：
  - 业务语义上，每一行样本代表 D 日收盘后可获得的全部信息
  - 工程实现上，因数据入库时效性限制，常在 D+1 凌晨执行，并显式消费最近一个已完整落库的收盘日盘面
  - 不要仅凭 `prev_trade_date` 等命名做字面推断；修改时必须同时考虑“业务基准日 / 数据可得日 / 实际执行时点”
  - 过滤逻辑与 dataset.py 完全对齐，保证训练/推断口径一致
"""
import os
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config.config import FILTER_BSE_STOCK, FILTER_STAR_BOARD, FILTER_688_BOARD
from features import FeatureEngine
from features.bundle_factory import build_bundle_from_context
from features.sector.sector_heat_feature import SectorHeatFeature
from strategies.base_strategy import BaseStrategy
import learnEngine.train_config as cfg
from utils.common_tools import (
    filter_st_stocks,
    get_daily_kline_data,
    get_stocks_in_sector,
    get_trade_dates,
    has_recent_limit_up_batch,
    ensure_limit_list_ths_data,
)
from utils.log_utils import logger
from utils.xgb_compat import safe_predict_proba
# [回退] 亏损惩罚功能回测表现不佳，暂时停用，保留代码以备后续优化后重新启用
# from risk_penalty_core import (
#     RiskPenaltyConfig,
#     should_filter_high_risk_stock,
#     should_stop_loss,
#     calc_strategy_weight_discount,
# )

# 与 dataset.py 对齐的低流动性阈值（单位：千元）
_MIN_AMOUNT = 10_000   # 1000 万元


class SectorHeatStrategy(BaseStrategy):
    """
    板块热度 XGBoost 选股策略

    依赖前置：
        1. 已运行 python learnEngine/dataset.py 生成训练集
        2. 已运行 python train.py 训练并保存模型
    """

    @property
    def strategy_id(self) -> str:
        return "sector_heat"

    def __init__(self):
        super().__init__()
        self._tracker_config = None
        self.strategy_name = "板块热度XGBoost盘前选股，开盘买入策略"
        self.strategy_params = {
            "buy_top_k":   6,        # 每日最多买入 N 只
            "sell_type":   "close",   # D+1 卖出类型：open=次日开盘，close=次日收盘
            "min_prob":    0.6,      # 最低买入概率阈值（0 = 不过滤）
            "load_minute": True,     # 是否加载分钟线（保证特征与训练口径一致）
            "model_path": self._get_runtime_model_path(),
        }

        # 新架构组件（与 dataset.py 使用同一套 FeatureEngine）
        self._sector_heat   = SectorHeatFeature()
        self._feature_engine = None    # 懒加载，模型加载后根据所需模块创建精简版引擎

        # 模型（懒加载，首次调用 generate_signal 时加载）
        self._model = None
        self._required_feature_modules = None   # 推断时所需的最小模块集（从 meta.json 读取）

        # # 风险惩罚配置（使用选股模型专用预设，所有惩罚参数统一由 risk_penalty_core 管理）
        # self._risk_config: RiskPenaltyConfig = RiskPenaltyConfig.for_strategy_model()

        # 持仓管理：{ts_code: buy_date}，用于严格执行 D+1 卖出规则
        self.hold_stock_dict: Dict[str, str] = {}

        # 推送/诊断复用：保留最近一次买入信号的富元数据（不影响主信号返回结构）
        self._last_buy_signal_details: List[Dict[str, any]] = []

        self.initialize()

    # ------------------------------------------------------------------ #
    # BaseStrategy 强制接口
    # ------------------------------------------------------------------ #

    def _get_runtime_model_path(self) -> str:
        return cfg.get_strategy_runtime_model_path(self.strategy_id)

    def initialize(self) -> None:
        """回测启动 / 重置时由引擎自动调用"""
        self.clear_signal()
        self.hold_stock_dict.clear()
        self._model = None  # 重置模型，下次使用时重新加载
        self._feature_engine = None
        self._required_feature_modules = None
        self._last_buy_signal_details = []

    def get_last_buy_signal_details(self) -> List[Dict[str, any]]:
        """返回最近一次买入信号的富元数据，供 runner 推送复用。"""
        return list(self._last_buy_signal_details)

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

    def supports_ml_training(self) -> bool:
        return True

    def get_training_label_target(self) -> str:
        return "label1_3pct"

    def get_model_registry_info(self) -> Dict[str, str]:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return {
            "strategy_id": self.strategy_id,
            "archive_dir": os.path.join(base_dir, "model", self.strategy_id),
            "runtime_dir": cfg.get_strategy_runtime_model_dir(self.strategy_id),
        }

    def _get_prev_trade_date(self, trade_date: str) -> str:
        """获取 D 日对应的前一交易日 D-1。"""
        d_date = datetime.strptime(trade_date, "%Y-%m-%d")
        start_lookback = (d_date - timedelta(days=20)).strftime("%Y-%m-%d")
        all_dates = get_trade_dates(start_lookback, trade_date)
        if len(all_dates) < 2:
            raise ValueError(f"{trade_date} 无法获取前一交易日")
        return all_dates[-2]

    def _get_selection_context(self, trade_date: str) -> Dict[str, any]:
        """构建选股上下文：D-1、Top3 板块、adapt_score。"""
        prev_trade_date = self._get_prev_trade_date(trade_date)
        top3_result = self._sector_heat.select_top3_hot_sectors(prev_trade_date)
        top3_sectors = top3_result["top3_sectors"]
        adapt_score = top3_result["adapt_score"]
        return {
            "trade_date": trade_date,
            "top3_sectors": top3_sectors,
            "adapt_score": adapt_score,
        }

    def build_training_candidates(
        self,
        trade_date: str,
        daily_df: pd.DataFrame = None,
    ) -> Tuple[pd.DataFrame, Dict[str, any]]:
        """
        构建策略训练候选样本，返回 (candidate_df, context)。
        candidate_df 保留一行一股票；context 供特征层与后续全局训练池复用。
        """
        context = self._get_selection_context(trade_date)
        top3_sectors = context["top3_sectors"]
        prev_trade_date = self._get_prev_trade_date(trade_date)

        if not top3_sectors:
            return pd.DataFrame(), context

        prev_daily_df = daily_df if daily_df is not None else get_daily_kline_data(prev_trade_date)
        if prev_daily_df.empty:
            return pd.DataFrame(), context

        sector_candidate_map, target_ts_codes = self._build_candidate_pool(
            trade_date, prev_daily_df, top3_sectors
        )
        context["sector_candidate_map"] = sector_candidate_map
        context["target_ts_codes"] = target_ts_codes

        if not target_ts_codes:
            return pd.DataFrame(), context

        candidate_rows = []
        for sector_name, sector_df in sector_candidate_map.items():
            if sector_df.empty:
                continue
            tmp = sector_df.copy()
            tmp["strategy_id"] = self.strategy_id
            tmp["strategy_name"] = self.strategy_name
            tmp["sector_name"] = sector_name
            tmp["trade_date"] = trade_date
            candidate_rows.append(tmp)

        if not candidate_rows:
            return pd.DataFrame(), context
        candidate_df = pd.concat(candidate_rows, ignore_index=True)
        return candidate_df, context

    def build_feature_bundle_from_context(self, context: Dict[str, any]):
        """基于共享上下文统一构造 FeatureDataBundle（T12: 委托给 bundle_factory）。"""
        return build_bundle_from_context(
            context,
            load_minute=self.strategy_params["load_minute"],
            required_modules=self._required_feature_modules,  # None=全量（训练），List=精简推断
        )

    # ------------------------------------------------------------------ #
    # 核心：买入信号生成
    # ------------------------------------------------------------------ #

    def _generate_buy_signal(
        self, trade_date: str, daily_df: pd.DataFrame
    ) -> Dict[str, str]:
        """
        返回 {ts_code: 'open'}（次日开盘买入）
        所有特征/筛选均基于 D 日收盘后已知数据，无未来函数。
        返回空 dict = 当日不买入
        """
        # ── 模型加载 ──────────────────────────────────────────────────────
        if not self._ensure_model():
            logger.error(f"{trade_date} 模型未就绪，跳过买入")
            self._last_buy_signal_details = []
            return {}

        # ── 共享候选池与上下文构建（与训练口径同源）────────────────────────
        try:
            candidate_df, context = self.build_training_candidates(trade_date, daily_df=None)
        except Exception as e:
            logger.error(f"{trade_date} 候选池构建失败: {e}", exc_info=True)
            self._last_buy_signal_details = []
            return {}

        if candidate_df.empty:
            logger.warning(f"{trade_date} 候选池为空，跳过买入")
            self._last_buy_signal_details = []
            return {}

        logger.info(
            f"{trade_date} Top3={context.get('top3_sectors')} | "
            f"adapt_score={context.get('adapt_score')} (基于 {trade_date} 收盘后口径)"
        )

        # ── Step 3: 特征计算（基于 D 日收盘后数据，与训练口径一致）──────────────────────
        try:
            bundle = self.build_feature_bundle_from_context(context)
            if bundle is None:
                logger.warning(f"{trade_date} Feature bundle 为空，跳过买入")
                self._last_buy_signal_details = []
                return {}
            # _feature_engine 在 _ensure_model 成功后已创建（精简模块版）
            engine = self._feature_engine or FeatureEngine()
            feature_df = engine.run_single_date(bundle)
        except Exception as e:
            logger.error(f"{trade_date} 特征计算失败: {e}", exc_info=True)
            self._last_buy_signal_details = []
            return {}

        if feature_df.empty:
            logger.warning(f"{trade_date} 特征计算结果为空，跳过买入")
            self._last_buy_signal_details = []
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
            probs = safe_predict_proba(self._model, X)[:, 1]
            feature_df = feature_df.copy()
            feature_df["_prob"] = probs
        except Exception as e:
            logger.error(f"{trade_date} 模型预测失败: {e}", exc_info=True)
            self._last_buy_signal_details = []
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
            self._last_buy_signal_details = []
            return {}

        # # ── Step 5b: 风险惩罚过滤（高风险市场环境下过滤低流动性个股）────────
        # # 从 feature_df 读取宏观数据（若被 EXCLUDE_PATTERNS 过滤则取 0，自动跳过此过滤）
        # market_limit_down = int(feature_df.get("market_limit_down_count", pd.Series([0])).iloc[0]
        #                         if "market_limit_down_count" in feature_df.columns else 0)
        # market_index_chg  = (float(feature_df["index_sh_pct_chg"].iloc[0])
        #                      if "index_sh_pct_chg" in feature_df.columns else None)
        #
        # before_risk_filter = len(selected)
        # selected = selected[
        #     ~selected.apply(
        #         lambda row: should_filter_high_risk_stock(
        #             market_limit_down_count=market_limit_down,
        #             stock_amount=float(row.get("stock_amount_d0", self._risk_config.min_liquidity_amount + 1)),
        #             config=self._risk_config,
        #             market_index_chg=market_index_chg,
        #         ),
        #         axis=1,
        #     )
        # ]
        # if len(selected) < before_risk_filter:
        #     logger.info(
        #         f"{trade_date} 风险过滤移除 {before_risk_filter - len(selected)} 只"
        #         f"（高风险市场+低流动性）| 跌停:{market_limit_down} 只"
        #     )
        #
        # if selected.empty:
        #     logger.info(f"{trade_date} 风险过滤后候选为空，跳过买入")
        #     return {}

        self._last_buy_signal_details = [
            {
                "stock_code": row["stock_code"],
                "buy_type": "open",
                "prob": round(float(row["_prob"]), 4),
            }
            for _, row in selected.iterrows()
        ]

        # 返回格式 {ts_code: 'open'}，引擎以次日开盘价买入（信号基于 D 日收盘后数据，无未来函数）
        buy_signal_map = {item["stock_code"]: item["buy_type"] for item in self._last_buy_signal_details}
        logger.info(
            f"{trade_date} 最终买入 {len(buy_signal_map)} 只: "
            + " | ".join(
                f"{item['stock_code']}(p={item['prob']:.3f})"
                for item in self._last_buy_signal_details
            )
        )
        return buy_signal_map

    # ------------------------------------------------------------------ #
    # 模型加载（懒加载 + 属性校验）
    # ------------------------------------------------------------------ #

    def _ensure_model(self) -> bool:
        """
        加载模型并校验 feature_names_in_ 属性（供 reindex 对齐列序）。
        同时推导 _required_feature_modules，用于推断时按需裁剪数据/特征加载。
        """
        if self._model is not None:
            return True
        # 每次重新发现 runtime_model 目录中最新模型，避免缓存旧路径
        path = self._get_runtime_model_path()
        self.strategy_params["model_path"] = path
        if not os.path.exists(path):
            logger.error(
                f"模型文件不存在: {path}\n"
                f"请先运行：python learnEngine/dataset.py && python train.py"
            )
            return False
        try:
            with open(path, "rb") as f:
                self._model = pickle.load(f)
            if not hasattr(self._model, "use_label_encoder"):
                self._model.use_label_encoder = False
            if not hasattr(self._model, "feature_names_in_"):
                logger.error(
                    "模型缺少 feature_names_in_ 属性。"
                    "请确认 train.py 使用 pandas DataFrame 作为 X 输入训练 XGBClassifier。"
                )
                self._model = None
                return False

            # ── 推导所需因子模块（优先读 meta.json，回退到列名反推）──────────
            modules = self._load_required_modules_from_meta(path)
            if modules is None:
                from features.feature_registry import feature_registry
                modules = feature_registry.get_modules_for_columns(
                    list(self._model.feature_names_in_)
                )
                logger.info(
                    f"[{self.strategy_name}] meta.json 不存在，动态推导所需模块: {modules}"
                )
            self._required_feature_modules = modules

            # ── 创建精简版 FeatureEngine（仅含所需模块）──────────────────────
            self._feature_engine = FeatureEngine(modules) if modules else FeatureEngine()

            logger.info(
                f"模型加载成功: {path} "
                f"| 特征数: {len(self._model.feature_names_in_)} "
                f"| 推断模块: {modules}"
            )
            return True
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            self._model = None
            self._feature_engine = None
            self._required_feature_modules = None
            return False

    @staticmethod
    def _load_required_modules_from_meta(model_pkl_path: str):
        """
        读取与模型同目录的 .meta.json，返回 feature_modules 列表。
        文件不存在或字段缺失时返回 None（调用方回退到动态推导）。
        """
        meta_path = model_pkl_path.replace(".pkl", ".meta.json")
        if not os.path.exists(meta_path):
            return None
        try:
            import json as _json
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = _json.load(f)
            modules = meta.get("feature_modules")
            if modules and isinstance(modules, list):
                return modules
        except Exception as e:
            logger.warning(f"meta.json 读取失败，回退动态推导: {e}")
        return None

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
            logger.debug(f"候选池 | 处理板块: {sector}")
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
                logger.debug(f"[{sector}] 最终候选股: {len(sector_daily)}")

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
        过滤 D-1 日涨停封板（close ≥ limit_up - 0.01）
        D-1 收盘涨停的股票次日开盘大概率高开/封板，开盘买入风险高。
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
            logger.debug(f"[D日涨停过滤] 涨停封板股已过滤: {removed} 只")
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

        logger.debug(
            f"涨停基因判断 | 候选: {len(ts_code_list)} "
            f"| 有涨停基因: {sum(result.values())}"
        )
        return result
