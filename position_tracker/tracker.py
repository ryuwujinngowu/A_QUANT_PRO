"""
持仓跟踪器（核心）
==================
纯计算模块：数据进 → 信号出，不做任何 I/O。

回测模式：调用 scan()，传入全天分钟线，一次性返回所有信号。
实盘模式：调用 scan_bar()，每收到一根 bar 检查一次，流式返回信号。

数据获取由调用方负责：
  - 回测：引擎批量从 DB 拉取分钟线，传入 scan()
  - 实盘：监控层每 30 秒从 API 拉取最新 bar，传入 scan_bar()
"""

from typing import Dict, List, Optional

import pandas as pd

from position_tracker.models import TrackerConfig, TrackedPosition, SellSignal
from position_tracker.rules import SellRule, FixedStopLoss, FixedTakeProfit, TrailingStop
from utils.log_utils import logger


class PositionTracker:
    """
    持仓跟踪器。

    用法（回测）::

        tracker = PositionTracker(config)          # 或 PositionTracker() 使用默认配置
        signals = tracker.scan(positions, minute_data)
        for sig in signals:
            account.sell(sig.ts_code, sig.trigger_price)

    用法（实盘，未来）::

        tracker = PositionTracker(config)
        # 每 30 秒循环
        for pos in positions:
            bar = fetch_latest_bar(pos.ts_code)
            signal = tracker.scan_bar(pos, bar.low, bar.high, bar.close, bar.time)
            if signal:
                wechat_push(signal)

    :param config:       止损止盈配置，None 则使用模块默认值
    :param custom_rules: 自定义规则列表，传入则完全替代默认规则（用于插入模型判断等）
    """

    def __init__(
        self,
        config: Optional[TrackerConfig] = None,
        custom_rules: Optional[List[SellRule]] = None,
    ):
        self.config = config or TrackerConfig()
        self.rules: List[SellRule] = (
            custom_rules if custom_rules is not None else self._build_default_rules()
        )
        # 盘中状态：{ts_code: {"intraday_high": float, ...}}
        # scan() 每次调用自动重置；scan_bar() 需调用方在每日开盘前调用 reset_day()
        self._intraday_state: Dict[str, Dict] = {}

    # ------------------------------------------------------------------ #
    # 公开接口
    # ------------------------------------------------------------------ #

    def scan(
        self,
        positions: List[TrackedPosition],
        minute_data: Dict[str, pd.DataFrame],
    ) -> List[SellSignal]:
        """
        批量扫描（回测模式）。

        :param positions:    被跟踪的持仓列表
        :param minute_data:  {ts_code: DataFrame[trade_time, open, high, low, close]}
                             由调用方从 DB 批量拉取后传入
        :return: 触发的卖出信号列表（每只股票最多一个信号，首次触发即返回）
        """
        self.reset_day()
        signals: List[SellSignal] = []

        for pos in positions:
            # T+1：当日买入不可卖
            if pos.available_volume <= 0:
                continue
            if pos.avg_cost <= 0:
                continue

            # 超期强平（不需要分钟线）
            if self.config.max_hold_days is not None and pos.hold_days >= self.config.max_hold_days:
                price = pos.open_price if pos.open_price > 0 else pos.avg_cost
                pct = (price - pos.avg_cost) / pos.avg_cost
                sig = SellSignal(
                    ts_code=pos.ts_code,
                    trigger_type="max_hold_days",
                    trigger_price=price,
                    trigger_time="09:30",
                    pct_from_cost=pct,
                    reason=f"持仓{pos.hold_days}天>={self.config.max_hold_days}天，超期强平",
                    name=pos.name,
                )
                signals.append(sig)
                logger.info(f"[持仓跟踪] {pos.ts_code} {sig.reason} | 触发价:{price:.2f} 涨跌幅:{pct:.2%}")
                continue

            # 分钟线扫描
            min_df = minute_data.get(pos.ts_code)
            if min_df is None or min_df.empty:
                logger.debug(f"[持仓跟踪] {pos.ts_code} 无分钟线数据，跳过")
                continue

            sig = self._scan_bars(pos, min_df)
            if sig:
                signals.append(sig)

        return signals

    def scan_bar(
        self,
        position: TrackedPosition,
        bar_low: float,
        bar_high: float,
        bar_close: float,
        bar_time: str,
    ) -> Optional[SellSignal]:
        """
        单 bar 扫描（实盘模式）。

        每收到一根新 bar 调用此方法。内部维护盘中状态（intraday_high 等），
        调用方需在每个交易日开盘前调用 reset_day() 重置状态。

        :return: SellSignal 如果触发，否则 None
        """
        if position.available_volume <= 0 or position.avg_cost <= 0:
            return None

        state = self._get_or_init_state(position.ts_code, position.avg_cost)
        if bar_high > state["intraday_high"]:
            state["intraday_high"] = bar_high

        for rule in self.rules:
            signal = rule.check(position, bar_low, bar_high, bar_close, bar_time, state)
            if signal:
                logger.info(
                    f"[持仓跟踪] {position.ts_code} {bar_time} {signal.reason}"
                    f" | 成本:{position.avg_cost:.2f} 触发价:{signal.trigger_price:.2f}"
                    f" 涨跌幅:{signal.pct_from_cost:.2%}"
                )
                return signal
        return None

    def reset_day(self):
        """重置盘中状态。回测模式由 scan() 自动调用；实盘模式需调用方每日开盘前手动调用。"""
        self._intraday_state.clear()

    # ------------------------------------------------------------------ #
    # 内部方法
    # ------------------------------------------------------------------ #

    def _build_default_rules(self) -> List[SellRule]:
        """从 config 构建默认规则列表"""
        rules: List[SellRule] = []
        if self.config.stop_loss_pct is not None:
            rules.append(FixedStopLoss(self.config.stop_loss_pct))
        if self.config.take_profit_pct is not None:
            rules.append(FixedTakeProfit(self.config.take_profit_pct))
        if self.config.trailing_stop_pct is not None:
            rules.append(TrailingStop(self.config.trailing_stop_pct))
        return rules

    def _get_or_init_state(self, ts_code: str, avg_cost: float) -> Dict:
        if ts_code not in self._intraday_state:
            self._intraday_state[ts_code] = {"intraday_high": avg_cost}
        return self._intraday_state[ts_code]

    def _scan_bars(self, position: TrackedPosition, min_df: pd.DataFrame) -> Optional[SellSignal]:
        """对一只股票的全天分钟线逐 bar 扫描，首次触发即返回。"""
        if "trade_time" in min_df.columns:
            min_df = min_df.sort_values("trade_time")

        state = self._get_or_init_state(position.ts_code, position.avg_cost)

        # 用 itertuples 替代 iterrows，性能更优（240 bars/天约快 5x）
        has_high = "high" in min_df.columns
        has_low = "low" in min_df.columns
        has_time = "trade_time" in min_df.columns

        for row in min_df.itertuples(index=False):
            bar_close = float(row.close)
            bar_high = float(row.high) if has_high else bar_close
            bar_low = float(row.low) if has_low else bar_close
            bar_time = self._extract_time(row.trade_time) if has_time else ""

            # 更新盘中最高价
            if bar_high > state["intraday_high"]:
                state["intraday_high"] = bar_high

            # 依次检查每条规则
            for rule in self.rules:
                signal = rule.check(position, bar_low, bar_high, bar_close, bar_time, state)
                if signal:
                    logger.info(
                        f"[持仓跟踪] {position.ts_code} {bar_time} {signal.reason}"
                        f" | 成本:{position.avg_cost:.2f} 触发价:{signal.trigger_price:.2f}"
                        f" 涨跌幅:{signal.pct_from_cost:.2%}"
                    )
                    return signal

        return None

    @staticmethod
    def _extract_time(trade_time) -> str:
        """从 trade_time 字段提取 HH:MM 字符串"""
        if hasattr(trade_time, "strftime"):
            return trade_time.strftime("%H:%M")
        s = str(trade_time)
        return s[-5:] if len(s) >= 5 else s
