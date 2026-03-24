# strategies/base_strategy.py
"""
所有策略的抽象基类（Base Strategy）
核心作用：
1. 定义引擎与策略之间的统一接口，实现引擎复用、多策略解耦
2. 强制所有自定义策略实现核心方法，避免运行时缺方法报错
3. 统一管理策略通用属性（如信号映射、策略名称）
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
from utils.log_utils import logger
from config.config import (
    MAIN_BOARD_LIMIT_UP_RATE,
    STAR_BOARD_LIMIT_UP_RATE,
    BJ_BOARD_LIMIT_UP_RATE,
)
import pandas as pd
from utils.common_tools import  calc_limit_up_price, calc_limit_down_price
from backtest.stop_loss_engine import StopLossConfig


class BaseStrategy(ABC):
    """
    策略抽象基类：所有自定义策略必须继承此类并实现所有抽象方法
    约定引擎调用的核心接口，确保不同策略与引擎的兼容性
    """

    def __init__(self):
        # ========== 策略通用属性（所有子类共享，无需重复定义） ==========
        # 卖出信号映射：{ts_code: sell_type}，sell_type = "open"/"close"
        self.sell_signal_map: Dict[str, str] = {}
        # 策略名称（子类必须重写，用于回测指标/日志标识）
        # self.strategy_name: str = "UnnamedStrategy"
        # 策略参数（可选，子类可扩展，用于回测结果记录参数）
        self.strategy_params: Dict[str, any] = {}

    # ========== 核心抽象方法（子类必须实现） ==========
    @abstractmethod
    def initialize(self) -> None:
        """
        策略初始化方法（引擎回测启动前调用）
        用途：
        1. 清空信号缓存（如sell_signal_map）
        2. 重置策略内部状态（如持仓天数、指标缓存）
        3. 初始化策略参数
        """
        pass

    @abstractmethod
    def generate_signal(
            self,
            trade_date: str,
            daily_df: pd.DataFrame,
            positions: Dict[str, any]
    ) -> Tuple[List[str], Dict[str, str]]:
        """
        生成买卖信号（引擎每日循环调用的核心方法）
        :param trade_date: 当前交易日（格式：YYYY-MM-DD）
        :param daily_df: 当日全市场日线数据（columns包含ts_code/open/high/low/close等）
        :param positions: 当前账户持仓字典（{ts_code: 持仓对象/持仓信息}）
        :return: 
            - buy_stocks: 当日买入股票列表（[ts_code1, ts_code2, ...]）
            - sell_signal_map: 当日卖出信号字典（{ts_code: sell_type}）
        """
        pass

    def calc_limit_up_price(self, ts_code: str, pre_close: float) -> float:
        """
        计算股票涨停价（适配不同板块涨跌幅限制，融合调试日志+强类型+完整校验）
        :param ts_code: 股票代码（如600000.SH/300001.SZ/831010.BJ）
        :param pre_close: 前一日收盘价
        :return: 涨停价格（保留2位小数，无效值返回0.0）
        """
        return calc_limit_up_price(ts_code, pre_close)

    def calc_limit_down_price(self, ts_code: str, pre_close: float) -> float:
        """
        计算股票跌停价（和涨停价逻辑完全对齐，适配不同板块涨跌幅限制）
        :param ts_code: 股票代码
        :param pre_close: 前一日收盘价
        :return: 跌停价格（保留2位小数，无效值返回0）
        """
        return calc_limit_down_price(ts_code, pre_close)

    # ========== 止损止盈配置接口（子类按需重写） ==========
    def get_stop_loss_config(self) -> Optional[StopLossConfig]:
        """
        返回止损止盈配置。默认返回 None（不启用止损引擎）。
        子类重写此方法以声明自己的止损规则，引擎会在日循环中自动执行。

        示例::

            def get_stop_loss_config(self):
                return StopLossConfig(
                    enabled=True,
                    fixed_stop_loss_pct=-0.08,   # -8% 固定止损
                    take_profit_pct=0.15,         # +15% 止盈
                    trailing_stop_pct=0.05,       # 盘中最高回撤 5% 触发
                    max_hold_days=10,             # 最多持仓 10 天
                )
        """
        return None

    # ========== 可选扩展方法（子类按需重写） ==========
    def get_strategy_info(self) -> Dict[str, any]:
        """
        获取策略完整信息（用于回测结果记录）
        :return: 包含名称、参数的字典
        """
        return {
            "strategy_name": self.strategy_name,
            "strategy_params": self.strategy_params
        }

    def clear_signal(self) -> None:
        """清空卖出信号（通用方法，子类可直接调用）"""
        self.sell_signal_map.clear()