"""
agent_stats — 智能体选股跟踪引擎（服务器运行入口）
=====================================================
部署路径  : /home/A_QUANT_PRO
Python    : python3.8
日志目录  : /home/A_QUANT_PRO/logs/agent_stats.log

─── crontab 配置（每工作日凌晨 3 点，daily 模式）────────────────────────
  0 4 * * 1-5 cd /home/A_QUANT_PRO && python3.8 agent_stats/run.py --mode daily >> logs/agent_stats.log 2>&1

  注意：
  · 必须 cd 到项目根目录，否则模块导入路径错误
  · 凌晨 3 点运行，此时 T-1 日线数据已完整入库
  · 严重限流时进程自动睡眠至次日 00:05 后恢复，无需人工重启

─── 常用场景 ─────────────────────────────────────────────────────────────

# 1. 日常运行（cron 触发，--mode daily 仅处理最新一日，速度最快）
  python3.8 agent_stats/run.py --mode daily

# 2. 首次部署 / 新 agent 历史补全（全量模式，不传 --mode 默认 full）
  python3.8 agent_stats/run.py --start-date 2024-10-01

# 3. 重跑指定 agent（策略逻辑更新后重算历史）
  python3.8 agent_stats/run.py --reset-agent morning_limit_up,afternoon_limit_up --reset-from 2024-10-01

  # 不指定 --reset-from 则从 config.START_DATE 起重跑：
  python3.8 agent_stats/run.py --reset-agent limit_down_buy

  ⚠ --reset-agent 会删除 DB 对应记录并重跑，不可逆，不会自动触发。

# 4. 修复数据不完整记录（分钟线聚合告警 / D+1 未结账）
  python3.8 agent_stats/run.py --repair-incomplete

  修复内容：
  · [MIN_FAIL] 记录：分钟线历史拉取失败导致信号不完整；
    删除后由引擎断点续跑重算（历史数据已缓存到 DB，成功率大幅提升）。
  · D+1 NULL 记录：隔日表现未结账；run_full_flow 的 dates_unclosed 自动处理。

# 5. 查看各 agent 断点续跑状态（不修改任何数据，只打印计划）
  # 直接运行即可，启动时会打印每个 agent 的续跑起点和待处理日数：
  #   [morning_limit_up] 上次有效日期 2026-03-13，已是最新 | 待处理 0 日
  #   [limit_down_buy  ] 无有效记录（含 N 条 [ERR]），从头开始 | 待处理 400 日

─── 异常处理策略 ────────────────────────────────────────────────────────
  · 非限流错误：最多重试 MAX_RETRY_TIMES 次（config.py），每次间隔 RETRY_INTERVAL 秒。
  · Tushare 严重限流（abort）：不消耗重试次数，进程睡眠至次日 00:05 后自动恢复。
  · 最终失败：微信推送告警，退出码 1。
"""
import sys
from pathlib import Path

# 获取当前脚本的绝对路径 (agent_stats/run.py)
current_file = Path(__file__).resolve()
# 项目根目录是 agent_stats 的父目录 (A_QUANT_PRO/)
project_root = current_file.parent.parent

# 将项目根目录加入 sys.path（如果尚未存在）
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# -------------------------------------------
# 先执行模块初始化（路径处理），必须放最前
import agent_stats  # noqa: F401

import argparse
import sys
import time
from datetime import datetime, timedelta

from agent_stats.config import MAX_RETRY_TIMES, RETRY_INTERVAL, START_DATE
from agent_stats.stats_engine import AgentStatsEngine
from agent_stats.long_stats_engine import AgentLongStatsEngine
from agent_stats.wechat_reporter import AgentWechatReporter
from data.data_cleaner import is_rate_limit_aborted
from utils.log_utils import logger
from utils.wechat_push import send_wechat_message_to_multiple_users


def _sleep_until_midnight() -> None:
    """
    睡眠至次日零点（多5分钟缓冲），期间每小时打印一次等待日志。
    用于 Tushare 每日配额耗尽后等待次日配额刷新，进程保持存活不需要人工重启。
    """
    now  = datetime.now()
    next_midnight = (now + timedelta(days=1)).replace(hour=0, minute=5, second=0, microsecond=0)
    total_secs = (next_midnight - now).total_seconds()
    logger.info(
        f"[限流等待] 当日 Tushare 配额耗尽，进程休眠至次日零点后（{next_midnight.strftime('%m-%d %H:%M')}），"
        f"共需等待 {int(total_secs // 3600)}h{int((total_secs % 3600) // 60)}m"
    )
    slept = 0
    check_interval = 3600  # 每小时 log 一次
    while slept < total_secs:
        sleep_this = min(check_interval, total_secs - slept)
        time.sleep(sleep_this)
        slept += sleep_this
        remaining = total_secs - slept
        if remaining > 60:
            logger.info(f"[限流等待] 还需等待约 {int(remaining // 3600)}h{int((remaining % 3600) // 60)}m")


def _parse_args():
    parser = argparse.ArgumentParser(description="Agent 收益统计引擎")
    parser.add_argument(
        "--start-date", type=str, default=None,
        help=f"统计起始日期（YYYY-MM-DD），新 agent 从此日期回溯。不传则用 config.START_DATE={START_DATE}",
    )
    parser.add_argument(
        "--reset-agent", type=str, default=None,
        help="逗号分隔的 agent_id，强制从 --reset-from 日期删除并重跑。"
             "例：--reset-agent morning_limit_up,limit_down_buy",
    )
    parser.add_argument(
        "--reset-from", type=str, default=None,
        help="配合 --reset-agent，指定重跑起始日期（YYYY-MM-DD）。不传则用 --start-date 或 config.START_DATE。",
    )
    parser.add_argument(
        "--repair-incomplete", action="store_true", default=False,
        help="修复历史遗留的数据不完整记录并重新计算。\n"
             "  [MIN_FAIL] 记录（分钟线聚合告警）：删除后由引擎用断点续跑重新生成信号（缓存后成功率高）。\n"
             "  D+1 NULL 记录（D+1 未结账）：由 run_full_flow 的 dates_unclosed 自动重算。\n"
             "用法：python agent_stats/run.py --repair-incomplete",
    )
    parser.add_argument(
        "--mode", type=str, default="full", choices=["full", "daily"],
        help="运行模式：full（默认）= 历史全量补全；daily = 仅处理最新一个交易日（cron 专用，速度快）。",
    )
    return parser.parse_args()


def _build_reset_agents(args) -> dict:
    """将 CLI 参数转换为引擎所需的 {agent_id: from_date} 字典"""
    if not args.reset_agent:
        return {}
    from_date = args.reset_from or args.start_date or START_DATE
    agent_ids = [aid.strip() for aid in args.reset_agent.split(",") if aid.strip()]
    reset = {aid: from_date for aid in agent_ids}
    logger.info(f"手动重置 agent 列表：{reset}")
    return reset


def main():
    args = _parse_args()

    logger.info("=" * 60)
    logger.info(f"[agent_stats] 启动  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  --mode        : {args.mode}")
    if args.start_date:
        logger.info(f"  --start-date  : {args.start_date}")
    if args.reset_agent:
        logger.info(f"  --reset-agent : {args.reset_agent}")
        logger.info(f"  --reset-from  : {args.reset_from or '(使用 start_date)'}")
    if args.repair_incomplete:
        logger.info(f"  --repair-incomplete: 开启，将删除 [MIN_FAIL] 记录后由引擎断点续跑重算")
    logger.info("=" * 60)

    reset_agents = _build_reset_agents(args)
    engine      = AgentStatsEngine(start_date=args.start_date)
    long_engine = AgentLongStatsEngine(start_date=args.start_date)

    # 修复数据不完整记录（在正常运行前执行，run_full_flow 会处理被删除的日期）
    if args.repair_incomplete:
        deleted = engine.repair_incomplete_records()
        logger.info(f"[repair-incomplete] 完成，删除 {deleted} 条 [MIN_FAIL] 记录，"
                    f"继续正常运行引擎（将重算这些日期）...")
    reporter = AgentWechatReporter()

    # ── 短线引擎（含重试 + 限流等待逻辑）────────────────────────────────
    run_success = False
    retry_count = 0

    while retry_count < MAX_RETRY_TIMES and not run_success:
        try:
            run_success = engine.run_full_flow(reset_agents=reset_agents, mode=args.mode)
            if run_success:
                logger.info("短线引擎运行完成")
                # 仅推送最新交易日的统计（历史补全不推，避免刷屏）
                # if engine.all_trade_dates:
                    # reporter.report_latest(engine.all_trade_dates[-1])
            else:
                # 判断是否因 Tushare 当日配额耗尽触发 abort
                if is_rate_limit_aborted():
                    # 配额耗尽：sleep 至次日零点（不消耗 retry_count），
                    # 次日配额刷新后自动恢复，无需人工干预。
                    _sleep_until_midnight()
                    logger.info("[限流等待] 次日零点已到，限流状态自动重置，恢复正常运行...")
                else:
                    retry_count += 1
                    logger.warning(
                        f"短线引擎返回 False，{RETRY_INTERVAL // 60} 分钟后重试"
                        f"（{retry_count}/{MAX_RETRY_TIMES}）"
                    )
                    time.sleep(RETRY_INTERVAL)
        except Exception as e:
            retry_count += 1
            logger.error(f"短线引擎运行异常：{e}", exc_info=True)
            try:
                send_wechat_message_to_multiple_users("【agent_stats 异常】", str(e)[:500])
            except Exception:
                pass
            if retry_count < MAX_RETRY_TIMES:
                time.sleep(RETRY_INTERVAL)

    if not run_success:
        logger.error(f"短线引擎已达最大重试次数 {MAX_RETRY_TIMES}，任务终止")
        try:
            send_wechat_message_to_multiple_users("【agent_stats 最终失败】", "请手动检查日志")
        except Exception:
            pass
        sys.exit(1)

    # ── 中长线引擎（独立运行，不影响短线引擎的退出码）────────────────────
    long_reset = {k: v for k, v in reset_agents.items() if k.startswith("long_")}
    try:
        long_ok = long_engine.run_full_flow(
            reset_agents=long_reset if long_reset else None,
            mode=args.mode,
        )
        if long_ok:
            logger.info("中长线引擎运行完成")
        else:
            logger.warning("中长线引擎返回 False，请检查日志（不影响短线结果）")
    except Exception as e:
        logger.error(f"中长线引擎运行异常：{e}", exc_info=True)
        try:
            send_wechat_message_to_multiple_users("【long_stats 异常】", str(e)[:500])
        except Exception:
            pass

    sys.exit(0)


if __name__ == "__main__":
    main()
