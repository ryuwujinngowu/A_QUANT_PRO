"""
Agent 统计结果微信推送
=====================
职责：仅推送最新一个交易日的各 agent 统计汇总，与引擎运行逻辑完全解耦。
历史数据补全时不推送（减少刷屏），只在日常 cron 运行的最后一步调用。

推送格式（示例）：
  ┌──────────────────────────────────┐
  │ [Agent统计] 2024-10-10 收益汇总  │
  │ 早盘打板选手                      │
  │   命中: 12只 | 日内均收: +2.35%  │
  │   次日均收: +1.82% | 最高: +5.6% │
  │   最大亏损: -2.1%                │
  │ ...                              │
  └──────────────────────────────────┘
"""
from typing import List, Dict, Optional
from utils.log_utils import logger
from utils.wechat_push import send_wechat_message_to_multiple_users
from agent_stats.agent_db_operator import AgentStatsDBOperator 


class AgentWechatReporter:
    def __init__(self):
        self.db = AgentStatsDBOperator()

    def report_latest(self, trade_date: str) -> bool:
        """
        查询 trade_date 的所有 agent 统计，格式化后推送微信。
        :param trade_date: 最新交易日（YYYY-MM-DD）
        :return: 推送成功返回 True
        """
        try:
            stats = self.db.get_latest_stats(trade_date)
            if not stats:
                logger.info(f"[WechatReporter] {trade_date} 无统计数据，跳过推送")
                return False

            title, content = self._format_message(trade_date, stats)
            result = send_wechat_message_to_multiple_users(title, content)
            logger.info(f"[WechatReporter] {trade_date} 推送完成")
            return bool(result)
        except Exception as e:
            logger.error(f"[WechatReporter] 推送失败：{e}", exc_info=True)
            return False

    @staticmethod
    def _fmt_pct(val, default="N/A") -> str:
        """将数值格式化为百分比字符串，None 时显示 N/A"""
        if val is None:
            return default
        prefix = "+" if float(val) > 0 else ""
        return f"{prefix}{float(val):.2f}%"

    def _format_message(self, trade_date: str, stats: List[Dict]) -> tuple:
        title = f"[Agent] {trade_date} 汇总"

        lines = [
            f" {trade_date}",
            "=" * 36,
        ]

        for row in stats:
            agent_name = row.get("agent_name", row["agent_id"])
            err = row.get("reserve_str_1")

            if err:
                lines.append(f"⚠ {agent_name} — 计算异常（{str(err)[:60]}）")
                continue

            intra   = self._fmt_pct(row.get("intraday_avg_return"))
            nd_open = self._fmt_pct(row.get("next_day_avg_open_premium"))
            nd_cls  = self._fmt_pct(row.get("next_day_avg_close_return"))
            nd_max  = self._fmt_pct(row.get("next_day_avg_max_premium"))
            nd_dd   = self._fmt_pct(row.get("next_day_avg_max_drawdown"))
            red_min = row.get("next_day_avg_red_minute")
            prof_min= row.get("next_day_avg_profit_minute")

            lines += [
                f"▶ {agent_name}",
                f"   日内均收: {intra}",
                f"   次日开盘: {nd_open}  |  收盘: {nd_cls}",
                f"   最高溢价: {nd_max}  |  最大亏损: {nd_dd}",
            ]
            if red_min is not None and prof_min is not None:
                lines.append(
                    f"   红盘时长: {red_min}min  |  浮盈时长: {prof_min}min"
                )

        lines += [
            "=" * 36,
            "⚠ 以上为模拟跟踪数据，仅供参考。",
        ]

        return title, "\n".join(lines)
