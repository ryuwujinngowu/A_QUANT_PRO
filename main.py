#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from backtest.engine import MultiStockBacktestEngine
from config.config import DEFAULT_INIT_CAPITAL
from strategies.LimitUpPullback_Strategy import LimitUpPullback_Strategy
from strategies.multi_limit_up_strategy import MultiLimitUpStrategy
from strategies.sector_heat_strategy import SectorHeatStrategy                  #热点情绪板块筛选买入策略
from strategies.long_high_low_switch_strategy import HighLowSwitchStrategy      #高低切轮动策略


def main():
    # ===================== 回测参数配置 =====================
    #！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
    #=================有没有新概念，记得维护概念表================
    #========================================================
    START_DATE = "2026-01-01"  # 回测开始日期
    END_DATE = "2026-03-21"    # 回测结束日期
    INIT_CAPITAL = DEFAULT_INIT_CAPITAL  # 初始本金10W元

    # # ===================== 初始化策略与回测引擎 =====================
    # strategy = HighLowSwitchStrategy()
    # engine = MultiStockBacktestEngine(
    #     strategy=strategy,
    #     init_capital=INIT_CAPITAL,
    #     start_date=START_DATE,
    #     end_date=END_DATE
    # )
    strategy = SectorHeatStrategy()
    engine = MultiStockBacktestEngine(
        strategy=strategy,
        init_capital=INIT_CAPITAL,
        start_date=START_DATE,
        end_date=END_DATE
    )
    # strategy = MultiLimitUpStrategy()
    # engine = MultiStockBacktestEngine(
    #     strategy=strategy,
    #     init_capital=INIT_CAPITAL,
    #     start_date=START_DATE,
    #     end_date=END_DATE
    # )


    # strategy = LimitUpPullback_Strategy()
    # engine = MultiStockBacktestEngine(
    #     strategy=strategy,
    #     init_capital=INIT_CAPITAL,
    #     start_date=START_DATE,
    #     end_date=END_DATE
    # )

    engine.run()

if __name__ == "__main__":
    main()