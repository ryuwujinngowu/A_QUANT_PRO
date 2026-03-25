#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from backtest.engine import MultiStockBacktestEngine
from config.config import DEFAULT_INIT_CAPITAL
from strategies.model_dip_strategy import ModelDipStrategy
from strategies.model_surge_strategy import ModelSurgeStrategy
from strategies.multi_limit_up_strategy import MultiLimitUpStrategy
from strategies.sector_heat_strategy import SectorHeatStrategy                  #热点情绪板块筛选买入策略
from strategies.long_high_low_switch_strategy import HighLowSwitchStrategy      #高低切轮动策略


def main():
    # ===================== 回测参数配置 =====================
    #！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
    #=================有没有新概念，记得维护概念表================
    #========================================================
    START_DATE = "2026-01-02"  # 回测开始日期（验证集范围内）
    END_DATE = "2026-03-01"    # 回测结束日期（约2个月）
    INIT_CAPITAL = DEFAULT_INIT_CAPITAL  # 初始本金10W元

    # # ===================== 初始化策略与回测引擎 =====================
    # strategy = HighLowSwitchStrategy()
    # engine = MultiStockBacktestEngine(
    #     strategy=strategy,
    #     init_capital=INIT_CAPITAL,
    #     start_date=START_DATE,
    #     end_date=END_DATE
    # )

    # strategy = SectorHeatStrategy()
    # engine = MultiStockBacktestEngine(
    #     strategy=strategy,
    #     init_capital=INIT_CAPITAL,
    #     start_date=START_DATE,
    #     end_date=END_DATE
    # )

    # strategy = MultiLimitUpStrategy()
    # engine = MultiStockBacktestEngine(
    #     strategy=strategy,
    #     init_capital=INIT_CAPITAL,
    #     start_date=START_DATE,
    #     end_date=END_DATE
    # )


    strategy = ModelDipStrategy()
    engine = MultiStockBacktestEngine(
        strategy=strategy,
        init_capital=INIT_CAPITAL,
        start_date=START_DATE,
        end_date=END_DATE
    )

    # strategy = ModelSurgeStrategy()
    # engine = MultiStockBacktestEngine(
    #     strategy=strategy,
    #     init_capital=INIT_CAPITAL,
    #     start_date=START_DATE,
    #     end_date=END_DATE
    # )

    engine.run()

if __name__ == "__main__":
    main()