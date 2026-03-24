"""
position_tracker — 持仓跟踪模块
================================
逐分钟线跟踪持仓，输出止损/止盈卖出信号。

当前用于回测引擎（批量扫描），后续切换数据源后用于实盘监控（流式扫描）。

核心类：
    PositionTracker  — 跟踪器（纯计算，不做 I/O）
    TrackerConfig    — 止损止盈配置
    TrackedPosition  — 被跟踪的持仓快照
    SellSignal       — 卖出信号输出

规则扩展：
    继承 SellRule，实现 check()，通过 custom_rules 注入 PositionTracker。
"""

from position_tracker.models import TrackerConfig, TrackedPosition, SellSignal
from position_tracker.rules import SellRule, FixedStopLoss, FixedTakeProfit, TrailingStop
from position_tracker.tracker import PositionTracker

__all__ = [
    "PositionTracker",
    "TrackerConfig",
    "TrackedPosition",
    "SellSignal",
    "SellRule",
    "FixedStopLoss",
    "FixedTakeProfit",
    "TrailingStop",
]
