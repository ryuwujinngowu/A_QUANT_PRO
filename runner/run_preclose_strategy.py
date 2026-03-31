"""
收盘前候选池生成器 (runner/run_preclose_strategy.py)
=====================================================
触发时机：每个交易日 14:55（收盘前 5 分钟）

逻辑：
  1. 拉取全市场实时日线（rt_k）作为 daily_df
  2. 调用 SectorHeatStrategy.generate_signal() 生成买入候选池
  3. 打印 / 推送候选股列表

用法:
  python runner/run_preclose_strategy.py
  python runner/run_preclose_strategy.py --date 2026-03-26  # 手动指定日期
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from datetime import datetime

from data_realtime.realtime_fetcher import RealtimeFetcher
from strategies.sector_heat.sector_heat_strategy import SectorHeatStrategy
from utils.log_utils import logger


def run(trade_date: str = None, index_overrides: dict = None) -> dict:
    """
    主逻辑：拉实时日线 → 跑策略 → 返回买入信号 dict

    Returns:
        {ts_code: buy_type}，如 {"000001.SZ": "close"}
    """
    if trade_date is None:
        trade_date = datetime.now().strftime("%Y-%m-%d")

    logger.info(f"[PreClose] 开始执行收盘前选股，日期={trade_date}")

    # ── Step 1: 拉实时日线 ────────────────────────────────────────────────
    fetcher = RealtimeFetcher()
    daily_df = fetcher.fetch_kline_day()

    if daily_df.empty:
        logger.error("[PreClose] 实时日线为空（市场未开盘 / 接口限流），无法运行策略")
        return {}

    logger.info(f"[PreClose] 实时日线获取成功 {len(daily_df)} 只")

    # ── Step 2: 跑策略 ────────────────────────────────────────────────────
    strategy = SectorHeatStrategy()
    if index_overrides:
        strategy.strategy_params["index_overrides"] = index_overrides
        logger.info(f"[PreClose] 指数覆盖: {index_overrides}")
    buy_signals, sell_signals = strategy.generate_signal(trade_date, daily_df, {})

    # ── Step 3: 输出 ──────────────────────────────────────────────────────
    logger.info(f"[PreClose] 选股完成 | 买入信号: {list(buy_signals.keys())}")

    if buy_signals:
        print("\n" + "=" * 50)
        print(f"  收盘前候选池 ({trade_date})")
        print("=" * 50)
        for ts_code, buy_type in buy_signals.items():
            row = daily_df[daily_df["ts_code"] == ts_code]
            if not row.empty:
                r = row.iloc[0]
                print(f"  {ts_code}  close={r['close']:.2f}  pct={r['pct_chg']:+.2f}%"
                      f"  amount={r['amount']/10000:.1f}亿  → {buy_type}")
            else:
                print(f"  {ts_code}  → {buy_type}")
        print("=" * 50)
    else:
        print(f"\n[PreClose] {trade_date} 无买入信号")

    return buy_signals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="收盘前候选池生成")
    parser.add_argument("--date", type=str, default=None,
                        help="交易日期 yyyy-mm-dd（默认：当日）")
    parser.add_argument("--index", nargs="*", default=None,
                        help="指数涨幅覆盖，格式：000001.SH=-1.5 399001.SZ=-1.6")
    args = parser.parse_args()

    overrides = {}
    if args.index:
        for item in args.index:
            code, val = item.split("=")
            overrides[code.strip()] = float(val.strip())

    run(trade_date=args.date, index_overrides=overrides or None)
