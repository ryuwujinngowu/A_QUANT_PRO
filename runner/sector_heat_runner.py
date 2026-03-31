"""
板块热度策略 — 日频信号推送 (独立运行脚本)
=============================================
部署路径  : /home/A_QUANT_PRO
Python    : python3.8

功能：
  每日收盘后调用 SectorHeatStrategy 的买入逻辑，
  生成前一交易日信号后通过微信推送，无需启动回测引擎。

与 main.py / backtest/engine 的关系：
  完全解耦。本脚本直接调用策略层的核心方法，不依赖 MultiStockBacktestEngine，
  不维护持仓状态，不执行卖出逻辑（信号仅供人工决策参考）。

─── 运行方式 ─────────────────────────────────────────────────────────────

  # 直接运行（处理上一个完整交易日，即 get_prev_trade_date()）
  python3.8 runner/sector_heat_runner.py

  # 指定日期（补跑历史信号或调试）
  python3.8 runner/sector_heat_runner.py --date 2026-03-13

  # 静默模式（不推送微信，只打印日志，用于调试）
  python3.8 runner/sector_heat_runner.py --dry-run

─── crontab 配置 ─────────────────────────────────────────────────────────

  ⚠ 关键：必须 cd 到项目根目录，否则相对路径（logs/、模型文件等）会出错

  # 方案 A：次日凌晨 5 点触发（推荐）
  # 原因：处理前一交易日（T-1）的数据，凌晨 5 点 Tushare 分钟线已完整入库，
  #       信号结果稳定可靠；周一凌晨 5 点处理上周五数据。
  00 05 * * 1-5 cd /home/A_QUANT_PRO && python3.8 runner/sector_heat_runner.py >> logs/runner.log 2>&1

  # 方案 B：当日 15:30 触发（实时性更高，但有分钟线缺失风险）
  # 原因：Tushare 分钟线在收盘后约 30-60 分钟完成入库。
  #       若在 15:05 等数据未完整时触发，HDI/SEI 特征为 0，信号可能与完整数据有差异。
  #       建议最早 15:30 触发以确保数据完整。
  30 15 * * 1-5 cd /home/A_QUANT_PRO && python3.8 runner/sector_heat_runner.py --date $(date +%Y-%m-%d) >> logs/runner.log 2>&1

  ⚠ 如当前 crontab 缺少 cd 命令 或 缺少 5 个时间字段（分时日月周），
    cron 将无法执行或执行后路径错误。
    请运行 crontab -e 检查完整格式，确保每行格式为：
      分 时 日 月 周 命令

─── 分钟线数据说明 ───────────────────────────────────────────────────────

  本策略特征计算依赖近 5 日分钟线（HDI / SEI 因子）。
  若在收盘后不久运行，当日分钟线可能尚未完全入库，导致：
    · [DataBundle] ⚠ 分钟线当日数据缺失 | ... 日志告警
    · HDI/SEI 特征值为 0，XGBoost 预测概率与完整数据时不同
    · 信号数量/标的与次日凌晨重跑时可能不一致

  这是数据时效性问题，不是 Bug。次日凌晨分钟线完整入库后重跑可获得稳定结果。

─── 依赖前置 ─────────────────────────────────────────────────────────────
  1. 已运行 python3.8 learnEngine/dataset.py 生成训练集
  2. 已运行 python3.8 train.py 训练并保存归档模型；人工晋升后放到 strategies/sector_heat/runtime_model/sector_heat_V*.pkl
"""

import argparse
import os
import sys
from datetime import datetime

from typing import Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.common_tools import get_trade_dates, get_daily_kline_data, get_prev_trade_date
from utils.log_utils import logger
# 修正：保持导入名称正确，后续调用统一使用这个名称
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


def _format_signal_message(trade_date: str, buy_signals: dict) -> Tuple[str, str]:
    """
    将买入信号 dict 格式化为 PushPlus 推送标题 + 正文

    :param trade_date:   信号日期
    :param buy_signals:  {ts_code: buy_type} 或 {}
    :return: (title, content) 两个字符串
    """
    if not buy_signals:
        title   = f"[板块热度筛选策略V5.2] {trade_date} | 今日无买入信号"
        content = (
            f"板块热度策略 | {trade_date}\n"
            f"今日无满足条件的买入标的。\n"
            f"可能原因：这他妈的还买尼玛呢。"
        )
        return title, content

    title = f"[板块热度筛选策略V5.2] {trade_date} | 发现 {len(buy_signals)} 个买入信号"
    lines = [
        f"📈 板块热度 XGBoost 策略-贝叶斯优化",
        f"信号日期：{trade_date}",
        f"买入时机：次日开盘（可参考策略监控台查看近期不同买点收益情况）",
        f"卖出时机：D+1日收盘",
        "=" * 32,
    ]
    for i, (ts_code, buy_type) in enumerate(buy_signals.items(), 1):
        lines.append(f"  {i:>2}. {ts_code}   [{buy_type}]")
    lines += [
        "=" * 32,
        "⚠️  信号仅供参考,注意风险控制,亏了自己受着,赚了发一个",
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
    logger.info(f"[Runner] ===== 开始执行 | 日期: {trade_date} | dry_run={dry_run} =====")

    # ── Step 1: 检查是否为交易日 ──────────────────────────────────────────
    if not _is_trade_date(trade_date):
        msg = f"[Runner] {trade_date} 非交易日，跳过执行"
        logger.info(msg)
        if not dry_run:
            # 修改1：使用正确的函数名 send_wechat_message_to_multiple_users
            # 补充说明：如果你的函数需要指定用户列表，可添加 users 参数，例如：
            send_wechat_message_to_multiple_users(title=f"[板块热度筛选策略V5.2] {trade_date} 今日非交易日", content=msg)
            send_wechat_message_to_multiple_users(
                title=f"[板块热度筛选策略V5.2] {trade_date} 今日非交易日",
                content=msg,
            )
        return True

    # ── Step 2: 获取当日全市场日线（策略构建候选池需要）────────────────────
    daily_df = get_daily_kline_data(trade_date)
    if daily_df.empty:
        msg = f"[Runner] {trade_date} 无法获取日线数据，跳过"
        logger.warning(msg)
        if not dry_run:
            # 修改2：同上，修正函数调用名
            send_wechat_message_to_multiple_users(
                title=f"[量化] {trade_date} 数据异常",
                content=msg
            )
        return False

    logger.info(f"[Runner] 日线数据加载完成 | {len(daily_df)} 只股票")

    # ── Step 3: 调用策略买入信号生成（不依赖引擎，直接调用核心方法）────────
    # 延迟导入：避免在 import 阶段就触发模型加载
    from strategies.sector_heat.sector_heat_strategy import SectorHeatStrategy

    try:
        strategy    = SectorHeatStrategy()
        buy_signals = strategy._generate_buy_signal(trade_date, daily_df)
    except Exception as e:
        msg = f"[Runner] 策略执行异常: {e}"
        logger.error(msg, exc_info=True)
        if not dry_run:
            send_wechat_message_to_multiple_users(
                title=f"[量化] {trade_date} 策略执行异常",
                content=msg
            )
        return False

    # ── 分钟线完整性诊断（信号为空时打印原因参考）────────────────────────
    # FeatureDataBundle 在加载分钟线时会将缺失统计写入 macro_cache；
    # 这里通过 strategy 内部 bundle 读取（bundle 在 _generate_buy_signal 内创建后即释放，
    # 无法直接访问，因此从 _load_minute_data 写入的日志已足够）。
    # 以下仅在信号为空时给出额外的运营提示，帮助区分"策略确实无信号"与"数据不完整导致无信号"。
    if not buy_signals:
        now_hour = datetime.now().hour
        if now_hour >= 15 and now_hour <= 16:
            # 收盘后 1 小时内，分钟线数据可能未完整入库
            logger.warning(
                f"[Runner] {trade_date} 无买入信号。"
                f"当前时间 {datetime.now().strftime('%H:%M')}，"
                f"收盘后数据可能未完全入库（Tushare 分钟线约延迟 30-60 分钟）。"
                f"如需确认，建议 16:00 后用 --date {trade_date} 重跑。"
                f"若次日重跑结果一致，则为策略本身无信号。"
            )
        else:
            logger.info(
                f"[Runner] {trade_date} 无买入信号（策略条件未触发，非数据问题）。"
                f"如有疑问，可用 --dry-run --date {trade_date} 重跑验证。"
            )

    # ── Step 4: 格式化 + 推送 ────────────────────────────────────────────
    title, content = _format_signal_message(trade_date, buy_signals)

    if dry_run:
        logger.info(f"[Runner][DRY-RUN] 标题: {title}")
        logger.info(f"[Runner][DRY-RUN] 正文:\n{content}")
        return True

    # 修改4：核心推送逻辑，修正函数调用名
    success = send_wechat_message_to_multiple_users(title, content)
    if success:
        logger.info(f"[Runner] 推送成功 | {title}")
    else:
        logger.warning(f"[Runner] 推送失败，信号已记录到日志")

    return success


# ============================================================
# CLI 入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="板块热度策略日频信号推送（独立脚本，无需启动回测引擎）"
    )
    parser.add_argument(
        "--date",
        type=str,
        default=get_prev_trade_date(),
        help="交易日期，格式 YYYY-MM-DD（默认：最近一个已完成交易日，即凌晨运行时的前一交易日）",
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