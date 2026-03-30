"""
实时数据获取测试脚本
====================
用法:
  python data_realtime/test_realtime.py              # 全量测试
  python data_realtime/test_realtime.py --kline-day  # 仅测日线
  python data_realtime/test_realtime.py --kline-min  # 仅测分钟线
  python data_realtime/test_realtime.py --strategy   # 测试策略 daily_df 兼容性

说明：
  - 交易时段（9:30-15:00）可获取真实数据，非交易时段接口返回空 DataFrame（正常）
  - 分钟线测试只取少量股票以节省配额
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from utils.log_utils import logger
from data_realtime.realtime_fetcher import RealtimeFetcher

# 测试用股票（流动性好的代表性股票）
_TEST_CODES = ["600000.SH", "000001.SZ", "000002.SZ", "600519.SH", "000858.SZ"]


# ─────────────────────────────────────────────────────────────────────────────
def _sep(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def test_init():
    """T1: 初始化测试（始终可运行）"""
    _sep("T1: 初始化 RealtimeFetcher")
    fetcher = RealtimeFetcher()
    # 触发懒初始化
    _ = fetcher.pro
    print("✓ 初始化成功，Tushare 连接正常")
    return fetcher


def test_kline_day(fetcher: RealtimeFetcher):
    """T2: 实时日线测试"""
    _sep("T2: 实时日线 fetch_kline_day()")

    # 2a. 指定少量股票（减少网络耗时）
    print(f">>> 指定股票: {_TEST_CODES}")
    df = fetcher.fetch_kline_day_codes(_TEST_CODES)

    if df.empty:
        print("⚠ 返回空 DataFrame（非交易时段或无权限，属正常）")
    else:
        print(f"✓ 返回 {len(df)} 行，{len(df.columns)} 列")
        print(f"  列名: {list(df.columns)}")
        print(f"  数据样例:\n{df.head(3).to_string(index=False)}")

        # 验证 Schema 兼容性
        required = {"ts_code", "trade_date", "open", "high", "low", "close",
                    "pre_close", "change1", "pct_chg", "volume", "amount"}
        missing = required - set(df.columns)
        if missing:
            print(f"✗ 缺少 kline_day 兼容字段: {missing}")
        else:
            print("✓ kline_day Schema 兼容性验证通过")

        # 验证 amount 单位（千元，应远小于元）
        avg_amount = df["amount"].mean()
        print(f"  amount 均值 = {avg_amount:.2f} 千元 （等价 {avg_amount/10000:.2f} 亿元）")

    # 2b. 全市场（可选，耗时较长）
    print("\n>>> 全市场实时日线（仅统计行数，不打印明细）")
    df_all = fetcher.fetch_kline_day()
    if df_all.empty:
        print("⚠ 全市场数据为空（非交易时段）")
    else:
        print(f"✓ 全市场 {len(df_all)} 只")
        pct_chg = df_all["pct_chg"]
        up = (pct_chg > 0).sum()
        down = (pct_chg < 0).sum()
        flat = (pct_chg == 0).sum()
        print(f"  涨: {up} | 跌: {down} | 平: {flat}")

    return df


def test_kline_min(fetcher: RealtimeFetcher):
    """T3: 实时分钟线测试"""
    _sep("T3: 实时分钟线 fetch_kline_min()")

    for freq in ["5MIN", "1MIN"]:
        print(f"\n>>> freq={freq}，股票={_TEST_CODES[:3]}")
        df = fetcher.fetch_kline_min(_TEST_CODES[:3], freq=freq)

        if df.empty:
            print(f"⚠ {freq} 返回空（非交易时段或无权限）")
        else:
            print(f"✓ {freq} 返回 {len(df)} 行")
            print(f"  列名: {list(df.columns)}")
            print(f"  数据样例:\n{df.head(3).to_string(index=False)}")

            # 验证 Schema
            required = {"ts_code", "trade_time", "trade_date", "open", "close", "high", "low", "volume", "amount"}
            missing = required - set(df.columns)
            if missing:
                print(f"✗ 缺少 kline_min 兼容字段: {missing}")
            else:
                print(f"✓ kline_min Schema 兼容性验证通过")

            # 验证时间排序
            for code, grp in df.groupby("ts_code"):
                is_sorted = grp["trade_time"].is_monotonic_increasing
                status = "✓" if is_sorted else "✗"
                print(f"  {status} {code}: {len(grp)} 行，时序{'正确' if is_sorted else '异常'}")


def test_strategy_compat(fetcher: RealtimeFetcher):
    """T4: 验证 daily_df 可直接传入 SectorHeatStrategy"""
    _sep("T4: 策略 daily_df 兼容性测试")

    daily_df = fetcher.fetch_kline_day_codes(_TEST_CODES)
    if daily_df.empty:
        print("⚠ 日线数据为空，跳过策略兼容测试（需在交易时段运行）")
        return

    # 模拟策略中 _build_candidate_pool 对 daily_df 的典型操作
    try:
        # 1. 涨幅过滤
        df = daily_df.copy()
        df = df[df["pct_chg"].between(-10, 10)]

        # 2. 成交额过滤（策略中 _MIN_AMOUNT = 10000 千元 = 1亿）
        df = df[df["amount"] >= 0]  # 宽松判断（非交易时段 amount 可能为 0）

        # 3. close / pre_close 字段存在性
        assert "close" in df.columns and "pre_close" in df.columns
        assert "ts_code" in df.columns

        print(f"✓ 策略 daily_df 兼容性验证通过，剩余 {len(df)} 只")
        print(f"  close 范围: {df['close'].min():.2f} ~ {df['close'].max():.2f}")
    except Exception as e:
        print(f"✗ 兼容性测试失败: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="实时数据获取测试")
    parser.add_argument("--kline-day", action="store_true", help="仅测实时日线")
    parser.add_argument("--kline-min", action="store_true", help="仅测实时分钟线")
    parser.add_argument("--strategy",  action="store_true", help="仅测策略兼容性")
    args = parser.parse_args()

    run_all = not (args.kline_day or args.kline_min or args.strategy)

    fetcher = test_init()

    if run_all or args.kline_day:
        test_kline_day(fetcher)

    if run_all or args.kline_min:
        test_kline_min(fetcher)

    if run_all or args.strategy:
        test_strategy_compat(fetcher)

    print("\n" + "="*60)
    print("  测试完成")
    print("="*60)


if __name__ == "__main__":
    main()
