"""
止损止盈引擎 (StopLossEngine)
==============================
在回测引擎的日循环中，对所有可卖持仓拉取当日分钟线，
逐分钟扫描是否触发止损/止盈条件，触发后以该分钟 close 价执行卖出。

设计原则：
  1. 策略声明止损规则（通过 BaseStrategy.get_stop_loss_config），引擎统一执行
  2. 仅对持仓股拉取分钟线（通常 ≤10 只），不拉全市场
  3. 止损优先级最高：在买入信号之前执行，避免"该止损的没止损又加仓"
  4. 向后兼容：策略未实现 get_stop_loss_config 时默认不启用
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from utils.log_utils import logger


@dataclass
class StopLossConfig:
    """
    止损止盈配置（由策略声明，引擎读取执行）

    Attributes:
        enabled:         是否启用分钟线止损止盈
        fixed_stop_loss_pct:  固定止损比例（负数，如 -0.08 = -8%）
        take_profit_pct:      固定止盈比例（正数，如 0.15 = +15%），None = 不启用
        trailing_stop_pct:    移动止损回撤比例（如 0.05 = 从最高点回撤5%触发），None = 不启用
        max_hold_days:        最大持仓天数（超期强平），None = 不限制
    """
    enabled: bool = False
    fixed_stop_loss_pct: float = -0.08
    take_profit_pct: Optional[float] = None
    trailing_stop_pct: Optional[float] = None
    max_hold_days: Optional[int] = None


@dataclass
class StopLossResult:
    """单只股票的止损扫描结果"""
    ts_code: str
    triggered: bool = False
    trigger_type: str = ""          # "stop_loss" / "take_profit" / "trailing_stop" / "max_hold_days"
    trigger_price: float = 0.0      # 触发时的分钟 close 价
    trigger_time: str = ""          # 触发时刻（如 "10:15"）
    pct_from_cost: float = 0.0      # 触发时相对成本的涨跌幅


class StopLossEngine:
    """
    止损止盈扫描引擎

    使用方式（由 MultiStockBacktestEngine 调用）：
        sl_engine = StopLossEngine(config, minute_data_fetcher)
        results = sl_engine.scan_positions(trade_date, positions, account)
    """

    def __init__(self, config: StopLossConfig, minute_data_fetcher=None):
        """
        :param config: 止损配置
        :param minute_data_fetcher: 分钟线获取函数，签名 (ts_code, trade_date) -> pd.DataFrame
                                     返回 DataFrame 须含 trade_time, close, high, low 列
                                     默认使用 data_cleaner.get_kline_min_by_stock_date
        """
        self.config = config
        if minute_data_fetcher is not None:
            self._fetch_minute = minute_data_fetcher
        else:
            from data.data_cleaner import data_cleaner
            self._fetch_minute = data_cleaner.get_kline_min_by_stock_date

    def scan_positions(
        self,
        trade_date: str,
        positions: Dict[str, "Position"],
        daily_df: pd.DataFrame,
    ) -> List[StopLossResult]:
        """
        扫描所有可卖持仓，返回需要止损/止盈的列表。

        :param trade_date: 当前交易日（YYYY-MM-DD 或 YYYYMMDD）
        :param positions:  当前持仓字典 {ts_code: Position}
        :param daily_df:   当日全市场日线（用于 max_hold_days 兜底判断）
        :return: 触发止损/止盈的结果列表
        """
        if not self.config.enabled:
            return []

        results: List[StopLossResult] = []
        trade_date_fmt = trade_date.replace("-", "")

        for ts_code, position in positions.items():
            # T+1：当日买入的不可卖
            if position.available_volume <= 0:
                continue

            cost = position.avg_cost
            if cost <= 0:
                continue

            # ── 超期强平（不需要分钟线） ──
            if self.config.max_hold_days is not None and position.hold_days >= self.config.max_hold_days:
                # 用日线 open 作为强平价
                stock_row = daily_df[daily_df["ts_code"] == ts_code]
                trigger_price = float(stock_row["open"].iloc[0]) if not stock_row.empty else cost
                pct = (trigger_price - cost) / cost
                results.append(StopLossResult(
                    ts_code=ts_code,
                    triggered=True,
                    trigger_type="max_hold_days",
                    trigger_price=trigger_price,
                    trigger_time="09:30",
                    pct_from_cost=pct,
                ))
                logger.info(
                    f"[止损引擎] {trade_date} {ts_code} 超期强平"
                    f"（持仓{position.hold_days}天≥{self.config.max_hold_days}天）"
                    f"| 成本:{cost:.2f} 触发价:{trigger_price:.2f} 涨跌幅:{pct:.2%}"
                )
                continue

            # ── 需要分钟线的止损/止盈检查 ──
            result = self._scan_single_stock(ts_code, trade_date_fmt, cost)
            if result.triggered:
                results.append(result)

        return results

    def _scan_single_stock(
        self,
        ts_code: str,
        trade_date: str,
        cost: float,
    ) -> StopLossResult:
        """
        对单只股票拉取分钟线，逐分钟检查止损/止盈条件。

        扫描顺序（每根分钟K线）：
          1. 固定止损（low 触碰止损价）
          2. 固定止盈（high 触碰止盈价）
          3. 移动止损（从最高点回撤超过阈值）

        :return: StopLossResult（triggered=False 表示未触发）
        """
        result = StopLossResult(ts_code=ts_code)

        try:
            min_df = self._fetch_minute(ts_code, trade_date)
        except Exception as e:
            logger.warning(f"[止损引擎] {ts_code} {trade_date} 分钟线获取失败: {e}，跳过止损检查")
            return result

        if min_df is None or min_df.empty:
            logger.debug(f"[止损引擎] {ts_code} {trade_date} 无分钟线数据，跳过")
            return result

        # 确保按时间排序
        if "trade_time" in min_df.columns:
            min_df = min_df.sort_values("trade_time").reset_index(drop=True)

        # 预计算阈值
        stop_loss_price = cost * (1 + self.config.fixed_stop_loss_pct) if self.config.fixed_stop_loss_pct else None
        take_profit_price = cost * (1 + self.config.take_profit_pct) if self.config.take_profit_pct else None

        # 移动止损状态：追踪盘中最高价
        intraday_high = cost

        for _, bar in min_df.iterrows():
            bar_close = float(bar["close"])
            bar_high = float(bar["high"]) if "high" in bar.columns else bar_close
            bar_low = float(bar["low"]) if "low" in bar.columns else bar_close
            bar_time = self._extract_time_str(bar)

            # 更新盘中最高价（用于移动止损）
            if bar_high > intraday_high:
                intraday_high = bar_high

            # ── 1. 固定止损：bar.low <= 止损价 ──
            if stop_loss_price is not None and bar_low <= stop_loss_price:
                # 触发价取止损价和 bar_close 的较低者（模拟实盘：跌破止损价后挂单成交）
                trigger_price = min(stop_loss_price, bar_close)
                pct = (trigger_price - cost) / cost
                result.triggered = True
                result.trigger_type = "stop_loss"
                result.trigger_price = trigger_price
                result.trigger_time = bar_time
                result.pct_from_cost = pct
                logger.info(
                    f"[止损引擎] {ts_code} {trade_date} {bar_time} 触发固定止损"
                    f" | 成本:{cost:.2f} 止损线:{stop_loss_price:.2f}"
                    f" 触发价:{trigger_price:.2f} 涨跌幅:{pct:.2%}"
                )
                return result

            # ── 2. 固定止盈：bar.high >= 止盈价 ──
            if take_profit_price is not None and bar_high >= take_profit_price:
                trigger_price = max(take_profit_price, bar_close)
                pct = (trigger_price - cost) / cost
                result.triggered = True
                result.trigger_type = "take_profit"
                result.trigger_price = trigger_price
                result.trigger_time = bar_time
                result.pct_from_cost = pct
                logger.info(
                    f"[止损引擎] {ts_code} {trade_date} {bar_time} 触发固定止盈"
                    f" | 成本:{cost:.2f} 止盈线:{take_profit_price:.2f}"
                    f" 触发价:{trigger_price:.2f} 涨跌幅:{pct:.2%}"
                )
                return result

            # ── 3. 移动止损：从盘中最高点回撤超过阈值 ──
            if self.config.trailing_stop_pct is not None and intraday_high > cost:
                trailing_stop_price = intraday_high * (1 - self.config.trailing_stop_pct)
                if bar_low <= trailing_stop_price:
                    trigger_price = min(trailing_stop_price, bar_close)
                    pct = (trigger_price - cost) / cost
                    result.triggered = True
                    result.trigger_type = "trailing_stop"
                    result.trigger_price = trigger_price
                    result.trigger_time = bar_time
                    result.pct_from_cost = pct
                    logger.info(
                        f"[止损引擎] {ts_code} {trade_date} {bar_time} 触发移动止损"
                        f" | 成本:{cost:.2f} 盘中高:{intraday_high:.2f}"
                        f" 回撤线:{trailing_stop_price:.2f}"
                        f" 触发价:{trigger_price:.2f} 涨跌幅:{pct:.2%}"
                    )
                    return result

        return result

    @staticmethod
    def _extract_time_str(bar) -> str:
        """从分钟 bar 中提取时间字符串"""
        if "trade_time" in bar.index:
            t = bar["trade_time"]
            if hasattr(t, "strftime"):
                return t.strftime("%H:%M")
            return str(t)[-5:] if len(str(t)) >= 5 else str(t)
        return ""
