"""
智能体策略基类
所有新增智能体必须继承该类，实现get_signal_stock_pool方法
保证引擎统一调用，完全解耦
"""
from typing import List, Dict
import pandas as pd


class BaseAgent:
    # 必须由子类实现的属性
    agent_id: str = ""    # 智能体唯一标识，和数据库agent_id对应
    agent_name: str = ""  # 智能体中文名称
    agent_desc: str = ""  # 策略详细说明（存入 DB reserve_str_2）

    def __init__(self):
        # 实例级别：记录本次 get_signal_stock_pool 调用中分钟线永久获取失败的股票代码。
        # 引擎在调用完成后读取此列表，将失败信息写入 DB 便于运维追踪。
        # 每次调用 get_signal_stock_pool 前应通过 reset_minute_fetch_state() 重置。
        self._minute_fetch_failures: List[str] = []

    def reset_minute_fetch_state(self) -> None:
        """每次 get_signal_stock_pool 前调用，清空上次的失败记录。"""
        self._minute_fetch_failures = []

    @property
    def minute_fetch_failures(self) -> List[str]:
        """返回本次调用中分钟线永久失败的股票代码列表（10次重试全部失败）。"""
        return list(self._minute_fetch_failures)

    def get_signal_stock_pool(self, trade_date: str, daily_data: pd.DataFrame, context: Dict) -> List[Dict]:
        """
        核心选股方法：必须由子类实现
        :param trade_date: 选股交易日（已经收盘的完整交易日，T日）
        :param daily_data: T日全市场标准化日线数据（含均线、涨跌停、连板数等）
        :param context: 上下文数据，包含：
            - st_stock_list: T日ST股票列表
            - trade_dates: 历史交易日列表
            - pre_close_data: T-1日收盘数据
            - 其他可扩展的上下文数据
        :return: 信号列表，每个元素必须包含以下字段：
            {
                "ts_code": "000001.SZ",
                "stock_name": "平安银行",
                "buy_price": 10.25  # 策略自定义买入价
            }

        注意：
        - 若分钟线拉取失败（接口不稳定），将失败代码写入 self._minute_fetch_failures。
        - 调用前请先调用 reset_minute_fetch_state() 清空上次结果。
        - TushareRateLimitAbort 异常必须向上传播，不得在此方法内吞掉。
        """
        raise NotImplementedError("子类必须实现get_signal_stock_pool方法")
