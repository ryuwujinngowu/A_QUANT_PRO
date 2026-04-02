"""
趋势跟随策略 — 日频信号推送 (独立运行脚本)
=============================================
部署路径  : /home/A_QUANT_PRO
Python    : python3.8

功能：
  每日收盘后调用 TrendFollowStrategy 的买入逻辑，
  生成前一交易日信号后通过微信推送，无需启动回测引擎。

与 main.py / backtest/engine 的关系：
  完全解耦。本脚本直接调用策略层的核心方法，不依赖 MultiStockBacktestEngine，
  不维护持仓状态，不执行卖出逻辑（信号仅供人工决策参考）。

─── 策略说明 ─────────────────────────────────────────────────────────────

  趋势跟随策略（TrendFollow）选取全市场60日动量 Top100 个股，
  经新股/ST/过热过滤 + MA5>MA30 趋势确认后，用 XGBoost 模型二次筛选。
  持仓周期：D+1 开盘买入，D+2 收盘卖出（2日持仓）。

─── 运行方式 ─────────────────────────────────────────────────────────────

  # 直接运行（处理上一个完整交易日，即 get_prev_trade_date()）
  python3.8 runner/trend_follow_runner.py

  # 指定日期（补跑历史信号或调试）
  python3.8 runner/trend_follow_runner.py --date 2026-03-13

  # 静默模式（不推送微信，只打印日志，用于调试）
  python3.8 runner/trend_follow_runner.py --dry-run

─── crontab 配置 ─────────────────────────────────────────────────────────

  ⚠ 关键：必须 cd 到项目根目录，否则相对路径（logs/、模型文件等）会出错

  # 方案 A：次日凌晨 5 点触发（推荐）
  # 原因：处理前一交易日（T-1）的数据，凌晨 5 点 Tushare 分钟线已完整入库，
  #       信号结果稳定可靠；周一凌晨 5 点处理上周五数据。
  00 05 * * 1-5 cd /home/A_QUANT_PRO && python3.8 runner/trend_follow_runner.py >> logs/runner.log 2>&1

  # 方案 B：当日 15:30 触发（实时性更高，但有分钟线缺失风险）
  30 15 * * 1-5 cd /home/A_QUANT_PRO && python3.8 runner/trend_follow_runner.py --date $(date +%Y-%m-%d) >> logs/runner.log 2>&1

  ⚠ 如当前 crontab 缺少 cd 命令 或 缺少 5 个时间字段（分时日月周），
    cron 将无法执行或执行后路径错误。

─── 依赖前置 ─────────────────────────────────────────────────────────────
  1. 已运行 python3.8 learnEngine/dataset.py 生成训练集
  2. 已运行 python3.8 train.py 训练并保存归档模型；
     人工晋升后放到 strategies/trend_follow/runtime_model/trend_follow_V*.pkl
"""

import argparse
import os
import sys
from datetime import datetime
from typing import List, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.common_tools import get_trade_dates, get_prev_trade_date
from utils.log_utils import logger
from utils.wechat_push import send_wechat_message_to_multiple_users


# ============================================================
# 辅助工具
# ============================================================

def _is_trade_date(date_str: str) -> bool:
    """判断给定日期是否为 A 股交易日"""
    try:
        dates = get_trade_dates(date_str, date_str)
        return bool(dates)
    except Exception:
        return False


def _get_model_display_name(model_path: str) -> str:
    if not model_path:
        return "trend_follow"
    return os.path.splitext(os.path.basename(model_path))[0]


def _format_signal_message(
    trade_date: str,
    buy_signals: dict,
    signal_details: List[Dict],
    model_name: str,
) -> Tuple[str, str]:
    """
    将买入信号 dict 格式化为 PushPlus 推送标题 + 正文

    :param trade_date:     信号日期
    :param buy_signals:    {ts_code: buy_type} 或 {}
    :param signal_details: [{stock_code, buy_type, prob}, ...]
    :param model_name:     实际运行模型名
    :return: (title, content) 两个字符串
    """
    if not buy_signals:
        title = f"[{model_name}] {trade_date} | 今日无买入信号"
        content = (
            f"模型：{model_name}\n"
            f"趋势跟随策略 | {trade_date}\n"
            f"今日无满足条件的买入标的。\n"
            f"可能原因：候选池无个股通过模型阈值筛选。"
        )
        return title, content

    title = f"[{model_name}] {trade_date} | 发现 {len(buy_signals)} 个买入信号"
    lines = [
        f"📈 趋势跟随 XGBoost 策略",
        f"模型：{model_name}",
        f"信号日期：{trade_date}",
        f"买入时机：次日开盘（D+1）",
        f"卖出时机：D+2 收盘（持仓 2 日）",
        "=" * 32,
    ]
    detail_map = {item["stock_code"]: item for item in signal_details}
    for i, (ts_code, buy_type) in enumerate(buy_signals.items(), 1):
        prob = detail_map.get(ts_code, {}).get("prob")
        prob_txt = f" | p={prob:.4f}" if prob is not None else ""
        lines.append(f"  {i:>2}. {ts_code}   [{buy_type}]{prob_txt}")
    lines += [
        "=" * 32,
        "⚠️  信号仅供参考，注意风险控制，亏了自己受着，赚了发一个",
        "⚠️  仓位建议：均分",
    ]
    content = "\n".join(lines)
    return title, content


# ============================================================
# 主推送逻辑
# ============================================================

def run_daily_signal(trade_date: str, dry_run: bool = False) -> bool:
    """
    单日信号生成 + 微信推送

    :param trade_date: 交易日，格式 YYYY-MM-DD
    :param dry_run:    True=只打印，不推送微信（调试模式）
    :return: 推送成功返回 True
    """
    logger.info(f"[TrendFollowRunner] ===== 开始执行 | 日期: {trade_date} | dry_run={dry_run} =====")

    # ── Step 1: 检查是否为交易日 ──────────────────────────────────────────
    if not _is_trade_date(trade_date):
        msg = f"[TrendFollowRunner] {trade_date} 非交易日，跳过执行"
        logger.info(msg)
        if not dry_run:
            send_wechat_message_to_multiple_users(
                title=f"[trend_follow] {trade_date} 今日非交易日",
                content=msg,
            )
        return True

    # ── Step 2: 调用策略买入信号生成 ────────────────────────────────────
    # TrendFollowStrategy._generate_buy_signal 内部自行查数据，无需外部传 daily_df
    from strategies.trend_follow.trend_follow_strategy import TrendFollowStrategy

    model_name = "trend_follow"
    try:
        strategy = TrendFollowStrategy()
        model_name = _get_model_display_name(strategy.strategy_params.get("model_path", ""))
        buy_signals = strategy._generate_buy_signal(trade_date)
        signal_details = strategy.get_last_buy_signal_details()
    except Exception as e:
        msg = f"[TrendFollowRunner] 策略执行异常: {e}"
        logger.error(msg, exc_info=True)
        if not dry_run:
            send_wechat_message_to_multiple_users(
                title=f"[{model_name}] {trade_date} 策略执行异常",
                content=msg,
            )
        return False

    # ── 无信号时补充诊断提示 ────────────────────────────────────────────
    if not buy_signals:
        now_hour = datetime.now().hour
        if 15 <= now_hour <= 16:
            logger.warning(
                f"[TrendFollowRunner] {trade_date} 无买入信号。"
                f"当前时间 {datetime.now().strftime('%H:%M')}，"
                f"收盘后数据可能未完全入库（Tushare 分钟线约延迟 30-60 分钟）。"
                f"如需确认，建议 16:00 后用 --date {trade_date} 重跑。"
            )
        else:
            logger.info(
                f"[TrendFollowRunner] {trade_date} 无买入信号（策略条件未触发，非数据问题）。"
                f"如有疑问，可用 --dry-run --date {trade_date} 重跑验证。"
            )

    # ── Step 3: 格式化 + 推送 ────────────────────────────────────────────
    title, content = _format_signal_message(trade_date, buy_signals, signal_details, model_name)

    if dry_run:
        logger.info(f"[TrendFollowRunner][DRY-RUN] 标题: {title}")
        logger.info(f"[TrendFollowRunner][DRY-RUN] 正文:\n{content}")
        return True

    success = send_wechat_message_to_multiple_users(title, content)
    if success:
        logger.info(f"[TrendFollowRunner] 推送成功 | {title}")
    else:
        logger.warning(f"[TrendFollowRunner] 推送失败，信号已记录到日志")

    return success


# ============================================================
# CLI 入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="趋势跟随策略日频信号推送（独立脚本，无需启动回测引擎）"
    )
    parser.add_argument(
        "--date",
        type=str,
        default=get_prev_trade_date(),
        help="交易日期，格式 YYYY-MM-DD（默认：最近一个已完成交易日）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="调试模式：只打印信号，不推送微信",
    )
    args = parser.parse_args()

    ok = run_daily_signal(trade_date=args.date, dry_run=args.dry_run)
    sys.exit(0 if ok else 1)
