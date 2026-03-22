"""
高低切轮动策略 — 回测入口
========================
用法:
    python runner/high_low_switch_backtest.py
    python runner/high_low_switch_backtest.py --start 2025-07-01 --end 2026-03-21
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.engine import MultiStockBacktestEngine
from config.config import DEFAULT_INIT_CAPITAL
from strategies.long_high_low_switch_strategy import HighLowSwitchStrategy


def main():
    parser = argparse.ArgumentParser(description="高低切轮动策略回测")
    parser.add_argument("--start", type=str, default="2025-07-01", help="回测开始日期 YYYY-MM-DD")
    parser.add_argument("--end", type=str, default="2026-03-21", help="回测结束日期 YYYY-MM-DD")
    parser.add_argument("--capital", type=float, default=DEFAULT_INIT_CAPITAL, help="初始资金")
    args = parser.parse_args()

    strategy = HighLowSwitchStrategy()
    engine = MultiStockBacktestEngine(
        strategy=strategy,
        init_capital=args.capital,
        start_date=args.start,
        end_date=args.end,
    )
    result = engine.run()

    print("\n===== 回测结果 =====")
    for k, v in result.items():
        if k not in ("net_value_df", "trade_df"):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
