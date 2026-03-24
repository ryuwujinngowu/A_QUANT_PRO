"""
持仓跟踪数据模型
================
TrackerConfig  — 止损止盈配置（模块默认值 + 策略可覆盖）
TrackedPosition — 被跟踪的持仓快照（由调用方构建，传入 tracker）
SellSignal      — 卖出信号输出（tracker 产出，调用方决定执行 or 推送）
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrackerConfig:
    """
    持仓跟踪配置。

    优先级：策略显式声明 > 模块默认值。
    策略未实现 get_tracker_config() 时，模块使用此处默认值。

    Attributes:
        stop_loss_pct:     固定止损比例（负数，如 -0.08 = 跌 8% 止损）
        take_profit_pct:   固定止盈比例（正数，如 0.10 = 涨 10% 止盈）
        trailing_stop_pct: 移动止损回撤比例（从盘中最高点回撤触发），None=不启用
        max_hold_days:     超期强平天数，None=不限制
    """
    stop_loss_pct: float = -0.08
    take_profit_pct: float = 0.10
    trailing_stop_pct: Optional[float] = None
    max_hold_days: Optional[int] = None


@dataclass
class TrackedPosition:
    """
    被跟踪的持仓快照。

    由调用方（回测引擎 / 实盘监控）从自身持仓结构转换而来，
    tracker 不关心底层持仓如何管理。

    Attributes:
        ts_code:          股票代码
        avg_cost:         持仓均价
        available_volume: 当日可卖数量（T+1 后）
        hold_days:        已持仓天数
        name:             股票名称（用于通知推送）
        open_price:       当日开盘价（超期强平成交价）
    """
    ts_code: str
    avg_cost: float
    available_volume: int
    hold_days: int
    name: str = ""
    open_price: float = 0.0


@dataclass
class SellSignal:
    """
    卖出信号（tracker 输出）。

    在回测中：引擎读取后调用 account.sell() 执行。
    在实盘中：监控层读取后调用微信推送程序。

    Attributes:
        ts_code:       股票代码
        name:          股票名称
        trigger_type:  触发类型（stop_loss / take_profit / trailing_stop / max_hold_days）
        trigger_price: 触发价格（建议成交价）
        trigger_time:  触发时刻（"HH:MM"）
        pct_from_cost: 相对成本涨跌幅
        reason:        人类可读的触发原因（用于日志和推送）
    """
    ts_code: str
    trigger_type: str
    trigger_price: float
    trigger_time: str
    pct_from_cost: float
    reason: str
    name: str = ""
