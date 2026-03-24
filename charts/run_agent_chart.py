#!/usr/bin/env python3
"""
Agent 统计图表独立入口
=====================
用法：
    # 单个 agent（自动识别短线/长线）
    python charts/run_agent_chart.py --agent-id sector_heat_v1 --start 2024-07-01 --end 2026-03-01

    # 全部 agent
    python charts/run_agent_chart.py --all --start 2024-07-01 --end 2026-03-01

    # 仅长线
    python charts/run_agent_chart.py --agent-id long_breakout_buy --type long --start 2024-07-01 --end 2026-03-01
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from charts.agent_chart import generate_agent_report, generate_all_agents_report
from utils.log_utils import logger


def main():
    parser = argparse.ArgumentParser(description="Agent 统计可视化")
    parser.add_argument("--agent-id", type=str, help="指定 agent_id")
    parser.add_argument("--all", action="store_true", help="为所有 agent 生成报告")
    parser.add_argument("--start", type=str, required=True, help="起始日期 YYYY-MM-DD")
    parser.add_argument("--end", type=str, required=True, help="截止日期 YYYY-MM-DD")
    parser.add_argument("--type", type=str, default="auto", choices=["short", "long", "auto"],
                        help="agent 类型（默认 auto 自动检测）")
    args = parser.parse_args()

    if args.all:
        paths = generate_all_agents_report(args.start, args.end)
        logger.info(f"共生成 {len(paths)} 份报告")
        for p in paths:
            logger.info(f"  {p}")
    elif args.agent_id:
        path = generate_agent_report(args.agent_id, args.start, args.end, agent_type=args.type)
        if path:
            logger.info(f"报告已生成: {path}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
