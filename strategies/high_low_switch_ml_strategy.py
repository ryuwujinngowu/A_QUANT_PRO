"""
高低切 ML 训练候选策略（HighLowSwitchMLStrategy）
================================================
供 ML 中台训练集生成使用，仅实现候选池生成逻辑。

候选池逻辑（直接复用 long_high_low_switch_strategy.py 的 Step5）：
  - D-1 涨停池（limit_list_ths）
  - 过滤：主板（排除 BJ/创业板/科创板）+ 非 ST + 首板/二板/三板（cons_nums in [1,2,3]）

去掉了原策略中所有"择时"部分（交由模型学习）：
  - 高位池重叠度检查（上涨钝化）
  - MA 乖离率斜率检测
  - 执行窗口状态机
  - 首板涨幅门槛过滤
  - 已持仓去重
"""
from typing import Dict, List, Tuple

import pandas as pd

from strategies.base_strategy import BaseStrategy
from utils.common_tools import (
    filter_st_stocks,
    get_limit_list_ths,
    get_limit_step,
    get_trade_dates,
    ensure_limit_list_ths_data,
)
from utils.log_utils import logger


def _is_main_board(ts_code: str) -> bool:
    """判断是否主板（沪深主板 10cm 涨跌幅），排除创业板/科创板/北交所。"""
    if ts_code.endswith(".BJ"):
        return False
    if ts_code.startswith(("300", "301", "302")) and ts_code.endswith(".SZ"):
        return False
    if ts_code.startswith("688"):
        return False
    return True


class HighLowSwitchMLStrategy(BaseStrategy):
    """高低切 ML 候选策略（仅候选池，不含择时逻辑）"""

    def __init__(self):
        super().__init__()
        self.strategy_name = "高低切ML策略"
        self.strategy_params = {}

    @property
    def strategy_id(self) -> str:
        return "high_low_switch"

    def supports_ml_training(self) -> bool:
        return True

    def build_training_candidates(
        self,
        trade_date: str,
        daily_df: pd.DataFrame = None,
    ) -> Tuple[pd.DataFrame, Dict[str, any]]:
        """
        构建高低切训练候选样本，返回 (candidate_df, context)。

        候选池 = D-1 涨停池（主板 + 非ST + 首/二/三板）。
        feature_trade_date = D-1（特征基于 D-1 收盘数据计算）。
        """
        # ── 获取 D-1 交易日 ────────────────────────────────────────────────
        try:
            lookback_dates = get_trade_dates(
                (pd.Timestamp(trade_date) - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
                trade_date,
            )
            # lookback_dates 包含 trade_date 本身，D-1 是倒数第二个
            if len(lookback_dates) < 2:
                logger.warning(f"[高低切ML][{trade_date}] 无法获取 D-1 交易日，跳过")
                empty_ctx = {"trade_date": trade_date, "feature_trade_date": trade_date,
                             "target_ts_codes": []}
                return pd.DataFrame(), empty_ctx
            prev_date = lookback_dates[-2]
        except Exception as e:
            logger.warning(f"[高低切ML][{trade_date}] 获取交易日历失败: {e}")
            empty_ctx = {"trade_date": trade_date, "feature_trade_date": trade_date,
                         "target_ts_codes": []}
            return pd.DataFrame(), empty_ctx

        context: Dict[str, any] = {
            "trade_date": trade_date,
            "feature_trade_date": prev_date,
            "target_ts_codes": [],
        }

        # ── 确保 D-1 涨停池数据新鲜 ───────────────────────────────────────
        try:
            prev_date_fmt = prev_date.replace("-", "")
            ensure_limit_list_ths_data(prev_date_fmt)
        except Exception as e:
            logger.warning(f"[高低切ML][{trade_date}] ensure_limit_list_ths_data 失败: {e}")

        # ── 获取 D-1 涨停池 ────────────────────────────────────────────────
        try:
            limit_df = get_limit_list_ths(prev_date, limit_type="涨停池")
        except Exception as e:
            logger.warning(f"[高低切ML][{trade_date}] 查涨停池失败: {e}")
            return pd.DataFrame(), context

        if limit_df is None or limit_df.empty:
            logger.info(f"[高低切ML][{trade_date}] D-1({prev_date}) 涨停池为空，跳过")
            return pd.DataFrame(), context

        # ── 主板过滤 ────────────────────────────────────────────────────
        limit_df = limit_df[limit_df["ts_code"].apply(_is_main_board)].copy()
        if limit_df.empty:
            return pd.DataFrame(), context

        # ── ST 过滤 ──────────────────────────────────────────────────────
        all_codes = limit_df["ts_code"].tolist()
        try:
            normal_codes = filter_st_stocks(all_codes, trade_date)
            st_set = set(all_codes) - set(normal_codes)
            limit_df = limit_df[~limit_df["ts_code"].isin(st_set)].copy()
        except Exception as e:
            logger.warning(f"[高低切ML][{trade_date}] ST 过滤失败（保守跳过）: {e}")
            return pd.DataFrame(), context

        if limit_df.empty:
            return pd.DataFrame(), context

        # ── 连板天梯 → 保留首板/二板/三板 ─────────────────────────────────
        try:
            step_df = get_limit_step(prev_date)
            if not step_df.empty and "ts_code" in step_df.columns and "nums" in step_df.columns:
                step_map: Dict[str, int] = dict(
                    zip(step_df["ts_code"], step_df["nums"].astype(int))
                )
            else:
                step_map = {}
        except Exception as e:
            logger.warning(f"[高低切ML][{trade_date}] 获取连板天梯失败（默认首板）: {e}")
            step_map = {}

        limit_df["cons_nums"] = limit_df["ts_code"].map(step_map).fillna(1).astype(int)
        limit_df = limit_df[limit_df["cons_nums"].isin([1, 2, 3])].copy()

        if limit_df.empty:
            return pd.DataFrame(), context

        # ── 构建 candidate_df ──────────────────────────────────────────────
        candidate_df = pd.DataFrame({
            "ts_code": limit_df["ts_code"].tolist(),
            "strategy_id": self.strategy_id,
            "strategy_name": self.strategy_name,
            "sector_name": "",
            "feature_trade_date": prev_date,
        })

        target_ts_codes: List[str] = candidate_df["ts_code"].tolist()
        context["target_ts_codes"] = target_ts_codes

        logger.info(
            f"[高低切ML][{trade_date}] D-1={prev_date} 候选池: {len(target_ts_codes)} 只 "
            f"（首板={int((limit_df['cons_nums']==1).sum())} "
            f"二板={int((limit_df['cons_nums']==2).sum())} "
            f"三板={int((limit_df['cons_nums']==3).sum())}）"
        )
        return candidate_df, context

    def initialize(self) -> None:
        """回测引擎初始化钩子（ML 专用策略不参与回测，无需初始化状态）"""
        pass

    def generate_signal(self, trade_date, daily_df, positions):
        """推理接口（ML 中台专用策略不参与回测引擎，返回空信号）"""
        return {}, {}
