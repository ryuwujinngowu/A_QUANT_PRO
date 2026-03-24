"""
卖出规则（可插拔）
=================
每条规则实现 SellRule 接口，逐 bar 检查是否触发。
内置规则：固定止损、固定止盈、移动止损。
扩展方式：继承 SellRule，实现 check()，注册到 PositionTracker。
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional

from position_tracker.models import TrackedPosition, SellSignal


class SellRule(ABC):
    """
    卖出规则抽象基类。

    每条规则在每根分钟 bar 上被调用一次。
    返回 SellSignal 表示触发，返回 None 表示未触发。

    参数说明：
        position: 被跟踪的持仓
        bar_low / bar_high / bar_close: 当前分钟 bar 的最低/最高/收盘价
        bar_time: 当前 bar 时间字符串（"HH:MM"）
        state: 该持仓的盘中状态字典（由 tracker 维护，规则可读写）
               内置 key: "intraday_high"（盘中最高价）
               规则可自行添加 key 存储状态

    设计说明：
        - state 允许规则间共享盘中状态（如盘中最高价），避免重复计算
        - 后续引入模型判断时，可在 state 中传入额外特征
    """

    @abstractmethod
    def check(
        self,
        position: TrackedPosition,
        bar_low: float,
        bar_high: float,
        bar_close: float,
        bar_time: str,
        state: Dict,
    ) -> Optional[SellSignal]:
        pass


class FixedStopLoss(SellRule):
    """固定止损：bar.low 触碰止损价时触发"""

    def __init__(self, pct: float):
        """pct: 负数，如 -0.08 表示跌 8% 止损"""
        self.pct = pct

    def check(self, position, bar_low, bar_high, bar_close, bar_time, state):
        stop_price = position.avg_cost * (1 + self.pct)
        if bar_low <= stop_price:
            price = min(stop_price, bar_close)
            pct = (price - position.avg_cost) / position.avg_cost
            return SellSignal(
                ts_code=position.ts_code,
                trigger_type="stop_loss",
                trigger_price=price,
                trigger_time=bar_time,
                pct_from_cost=pct,
                reason=f"触及固定止损线 {self.pct:.1%}（止损价:{stop_price:.2f}）",
                name=position.name,
            )
        return None


class FixedTakeProfit(SellRule):
    """固定止盈：bar.high 触碰止盈价时触发"""

    def __init__(self, pct: float):
        """pct: 正数，如 0.10 表示涨 10% 止盈"""
        self.pct = pct

    def check(self, position, bar_low, bar_high, bar_close, bar_time, state):
        profit_price = position.avg_cost * (1 + self.pct)
        if bar_high >= profit_price:
            price = max(profit_price, bar_close)
            pct = (price - position.avg_cost) / position.avg_cost
            return SellSignal(
                ts_code=position.ts_code,
                trigger_type="take_profit",
                trigger_price=price,
                trigger_time=bar_time,
                pct_from_cost=pct,
                reason=f"触及固定止盈线 +{self.pct:.1%}（止盈价:{profit_price:.2f}）",
                name=position.name,
            )
        return None


class TrailingStop(SellRule):
    """移动止损：从盘中最高点回撤超过阈值触发"""

    def __init__(self, pct: float):
        """pct: 正数，如 0.05 表示从最高点回撤 5% 触发"""
        self.pct = pct

    def check(self, position, bar_low, bar_high, bar_close, bar_time, state):
        intraday_high = state.get("intraday_high", position.avg_cost)
        # 仅当盘中最高价已超过成本价时才有意义
        if intraday_high <= position.avg_cost:
            return None
        trailing_price = intraday_high * (1 - self.pct)
        if bar_low <= trailing_price:
            price = min(trailing_price, bar_close)
            pct = (price - position.avg_cost) / position.avg_cost
            return SellSignal(
                ts_code=position.ts_code,
                trigger_type="trailing_stop",
                trigger_price=price,
                trigger_time=bar_time,
                pct_from_cost=pct,
                reason=f"盘中高点{intraday_high:.2f}回撤>{self.pct:.1%}（回撤线:{trailing_price:.2f}）",
                name=position.name,
            )
        return None
