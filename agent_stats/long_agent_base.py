"""
中长线 Agent 基类
================
继承 BaseAgent，额外要求子类实现 check_sell_signal 方法。

命名规范
--------
中长线 agent 的 agent_id 必须以 "long_" 开头，引擎据此自动分流：
  - AgentStatsEngine    只处理 NOT LIKE 'long_%' 的 agent（短线）
  - AgentLongStatsEngine 只处理 LIKE 'long_%' 的 agent（中长线）

买入流程（与短线完全相同）
  - get_signal_stock_pool() 返回当日买入信号
  - 结果写入 agent_daily_profit_stats（next_day_* 字段全部 NULL）
  - 同时在 agent_long_position_stats 为每只命中股创建开仓记录

卖出流程（中长线特有）
  - check_sell_signal() 每日检查所有持仓，触发时平仓并计算区间统计

区间统计字段（平仓时计算，使用后复权价格）
  sell_date             卖出日期
  sell_price            卖出价（后复权；kline_day_hfq 表；缺失时降级为原始价）
  period_return         区间盈亏 %（相对买入价）
  trading_days          区间交易日天数
  up_days               区间阳线天数（收盘 >= 开盘）
  down_days             区间阴线天数（收盘 < 开盘）
  max_drawdown          区间最大回撤 %（负值，相对买入价最低点）
  max_floating_profit   区间最高浮盈 %（相对买入价最高点）
  daily_detail          逐日明细 JSON（含 date/open/high/low/close/float_pnl）
"""

from typing import Dict, List, Optional
import pandas as pd

from agent_stats.agent_base import BaseAgent


class BaseLongAgent(BaseAgent):
    """
    中长线 Agent 基类。

    子类必须实现：
      1. agent_id       （必须以 "long_" 开头）
      2. agent_name
      3. get_signal_stock_pool()  ← 与短线完全相同
      4. check_sell_signal()      ← 中长线特有：动态卖出信号
    """

    def check_sell_signal(
        self,
        position: Dict,
        today_row: Optional[Dict],
        context: Dict,
    ) -> bool:
        """
        判断持仓是否在今日触发卖出（今日收盘价卖出）。

        :param position: 当前持仓记录（来自 agent_long_position_stats）::

            {
                "agent_id":   "long_breakout_buy",
                "ts_code":    "000001.SZ",
                "stock_name": "平安银行",
                "buy_date":   "2026-01-02",          # YYYY-MM-DD
                "buy_price":  10.25,                  # 买入时后复权价
                "trading_days_so_far": 5,             # 引擎计算后注入，买入后交易日数
            }

        :param today_row: 今日该股票的原始（非复权）日线行，可能为 None（停牌时）::

            {
                "ts_code": "000001.SZ",
                "open": 10.5, "high": 11.0,
                "low": 10.2, "close": 10.8,
                "pct_chg": 2.86,          # 相对昨收的涨跌幅
            }

        :param context: 引擎上下文 {trade_dates: List[str], ...}
        :return: True = 今日收盘卖出；False = 继续持仓
        """
        raise NotImplementedError(
            f"[{self.__class__.__name__}] 必须实现 check_sell_signal()"
        )
