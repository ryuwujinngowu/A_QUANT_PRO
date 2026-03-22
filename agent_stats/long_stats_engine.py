"""
中长线 Agent 统计引擎（v2 — 前向扫描卖出信号）
================================================
职责：处理所有 agent_id 以 "long_" 开头的 Agent，管理"买入信号 → 持仓 → 卖出"全生命周期。

与短线引擎的关键差异
--------------------
  短线（AgentStatsEngine）：T 日买入 → T+1 日强制卖出，跟踪次日各项指标。
  本引擎（AgentLongStatsEngine）：T 日买入 → 前向扫描未来日期检查卖出信号 →
  触发时平仓，跟踪区间盈亏、最大回撤、最高浮盈等中长线指标。

买入记录（两表双写）
--------------------
  agent_daily_profit_stats   同短线方式记录买入信号（next_day_* 字段全部为 NULL）
  agent_long_position_stats  每只命中股单独建仓记录（status=0）

执行模式
--------
1. 历史补全模式（--start-date / --mode full）：
   逐日生成买入信号后，立即对每只买入股票前向扫描所有未来日期（直到触发卖出或
   到达当前最新日期），一次性完成区间收益计算。
   - 触发卖出 → status=1，写入完整区间统计
   - 扫描到最新日期仍未触发卖出 → status=0（未结账），持仓继续跟踪

2. 每日更新模式（--mode daily）：
   仅检查所有 status=0 的未结账持仓，在最新交易日是否触发卖出信号。
   - 触发 → 平仓结账
   - 未触发 → 保持 status=0，等待次日再检查

卖出信号（基于后复权数据，消除未来函数）
------------------------------------------
  引擎在扫描卖出信号前，预取后复权（HFQ）数据并传递 hfq_buy_price。
  Agent 基于 HFQ 价格判断止损和 MA 破位，历史价格不变，无未来函数。
  HFQ 数据获取流程与 QFQ 完全对称：查库 → 请求接口 → 清洗 → 入库。

区间统计（平仓时计算，使用后复权数据）
--------------------------------------
  优先从 kline_day_hfq 表获取后复权价格；
  若该表无数据（尚未入库），降级使用 kline_day 原始价格并记录警告。
"""

import importlib
import inspect
import pkgutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from agent_stats.config import START_DATE, MAX_RETRY_TIMES
from agent_stats.agent_db_operator import AgentStatsDBOperator
from agent_stats.long_agent_base import BaseLongAgent
from agent_stats.long_position_db_operator import LongPositionDBOperator
from agent_stats.engine_shared import (
    get_next_trade_date as _shared_get_next_trade_date,
    build_trade_date_context,
    calc_intraday_stats as _shared_calc_intraday_stats,
)
from utils.common_tools import (
    get_trade_dates,
    get_daily_kline_data,
    get_st_stock_codes,
    get_prev_trade_date,
    get_hfq_kline_range,
    get_kline_day_range,
    get_ex_div_stocks,
    get_ex_div_dates_for_stock,
    get_market_limit_counts,
    get_index_pct_chg,
    ensure_dividend_data,
)
from utils.log_utils import logger
from data.data_cleaner import TushareRateLimitAbort, data_cleaner


class AgentLongStatsEngine:

    def __init__(self, start_date: str = None):
        self.start_date = start_date or START_DATE
        self.signal_db  = AgentStatsDBOperator()      # 买入信号（agent_daily_profit_stats）
        self.pos_db     = LongPositionDBOperator()     # 持仓记录（agent_long_position_stats）
        self.agents: List[BaseLongAgent] = self._auto_load_agents()

        last_trade_date = get_prev_trade_date()

        # 处理窗口：从 START_DATE 到上一个完整交易日
        self.all_trade_dates: List[str] = get_trade_dates(self.start_date, last_trade_date)

        # 含历史回看窗口（供 agent 计算 HFQ 均线等）
        # 250 自然日 ≈ 175 交易日，覆盖 2 倍最大 MA 周期（如 MA60 需回看 120 交易日）
        context_start = (
            datetime.strptime(self.start_date, "%Y-%m-%d") - timedelta(days=250)
        ).strftime("%Y-%m-%d")
        self.context_trade_dates: List[str] = get_trade_dates(context_start, last_trade_date)

        logger.info(
            f"[长线引擎] 初始化完成 | 智能体：{len(self.agents)} 个 | "
            f"处理范围：{self.start_date} ~ {last_trade_date}"
        )

    # ------------------------------------------------------------------ #
    # Agent 自动发现（只加载 long_* agent）
    # ------------------------------------------------------------------ #

    def _auto_load_agents(self) -> List[BaseLongAgent]:
        import agent_stats.agents as agents_pkg
        agents = []
        for _, module_name, _ in pkgutil.iter_modules(agents_pkg.__path__):
            full_module = f"agent_stats.agents.{module_name}"
            try:
                module = importlib.import_module(full_module)
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (
                        issubclass(obj, BaseLongAgent)
                        and obj is not BaseLongAgent
                        and obj.__module__ == full_module
                    ):
                        instance = obj()
                        if not instance.agent_id.startswith("long_"):
                            logger.warning(f"[{name}] agent_id 未以 'long_' 开头，跳过")
                            continue
                        if not instance.agent_id or not instance.agent_name:
                            logger.error(f"[{full_module}.{name}] 缺少 agent_id/agent_name，跳过")
                            continue
                        agents.append(instance)
                        logger.info(f"[长线引擎] 加载智能体：{instance.agent_name}（{instance.agent_id}）")
            except Exception as e:
                logger.error(f"[长线引擎][{full_module}] 加载失败：{e}", exc_info=True)
        if not agents:
            logger.info("[长线引擎] 无中长线智能体，引擎空跑")
        return agents

    # ------------------------------------------------------------------ #
    # 工具方法
    # ------------------------------------------------------------------ #

    def _get_next_trade_date(self, trade_date: str) -> Optional[str]:
        return _shared_get_next_trade_date(self.all_trade_dates, trade_date)

    def _get_context(self, trade_date: str) -> Dict:
        return build_trade_date_context(
            trade_date,
            self.context_trade_dates,
            extra={"ex_div_stocks": get_ex_div_stocks(trade_date)},
        )

    def _ensure_hfq_data(self, ts_code: str, start_date: str, end_date: str) -> None:
        """
        确保后复权数据已入库（查库 → 请求 → 清洗 → 入库）。
        复用 QFQ 完全对称的数据获取流程，仅表名和复权方式不同。
        """
        start_fmt = start_date.replace("-", "")
        end_fmt   = end_date.replace("-", "")
        try:
            data_cleaner.clean_and_insert_kline_day_hfq(
                ts_code, start_date=start_fmt, end_date=end_fmt
            )
        except Exception as e:
            logger.warning(f"[长线引擎][{ts_code}] HFQ 数据入库异常：{e}")

    def _get_hfq_buy_price(self, ts_code: str, buy_date: str) -> float:
        """
        获取买入日的后复权收盘价（用于卖出信号的止损判断，无未来函数）。
        """
        hfq_df = get_hfq_kline_range([ts_code], buy_date, buy_date)
        if not hfq_df.empty:
            return float(hfq_df.iloc[0]["close"])
        return 0.0

    # _prefetch_hfq_for_candidates 已移除。
    # HFQ 数据按需由各 agent 内部拉取（仅对粗筛通过的少量股票），
    # 避免全市场 ~4900 只逐只调用 API 导致超时。
    # 详见 long_breakout_buy.py::get_signal_stock_pool 的两阶段优化。

    # ------------------------------------------------------------------ #
    # 买入信号处理（Step A）
    # ------------------------------------------------------------------ #

    def _process_buy_signal(
        self,
        agent: BaseLongAgent,
        trade_date: str,
        daily_data: pd.DataFrame,
        context: Dict,
    ) -> List[Dict]:
        """
        生成 T 日买入信号，双写：
          1. agent_daily_profit_stats（next_day_* 全部 NULL）
          2. agent_long_position_stats（每只股单独建仓，status=0）

        返回 stock_detail 列表（后续前向扫描需要用到 ts_code / buy_price）。
        """
        try:
            agent.reset_minute_fetch_state()
            signal_list = agent.get_signal_stock_pool(trade_date, daily_data, context)

            avg_ret, stock_detail = self._calc_intraday_stats(signal_list, trade_date)

            self.signal_db.insert_signal_record({
                "agent_id":            agent.agent_id,
                "agent_name":          agent.agent_name,
                "agent_desc":          agent.agent_desc,
                "trade_date":          trade_date,
                "intraday_avg_return": avg_ret,
                "signal_stock_detail": {"stock_list": stock_detail},
            })

            for s in stock_detail:
                self.pos_db.insert_position({
                    "agent_id":   agent.agent_id,
                    "agent_name": agent.agent_name,
                    "ts_code":    s["ts_code"],
                    "stock_name": s.get("stock_name", ""),
                    "buy_date":   trade_date,
                    "buy_price":  s["buy_price"],
                })

            logger.info(
                f"[长线引擎][{agent.agent_id}][{trade_date}] 买入信号入库 | "
                f"命中 {len(signal_list)} 只 | 日内均收益 {avg_ret:.2f}%"
            )
            return stock_detail
        except TushareRateLimitAbort:
            raise
        except Exception as e:
            logger.error(f"[长线引擎][{agent.agent_id}][{trade_date}] 买入信号失败：{e}", exc_info=True)
            self.signal_db.insert_error_record(agent.agent_id, agent.agent_name, trade_date, str(e))
            return []

    def _calc_intraday_stats(
        self, stock_list: List[Dict], trade_date: str
    ) -> Tuple[float, List[Dict]]:
        return _shared_calc_intraday_stats(stock_list, trade_date)

    # ------------------------------------------------------------------ #
    # 前向扫描卖出信号（历史补全模式核心）
    # ------------------------------------------------------------------ #

    def _forward_scan_sell_for_position(
        self,
        agent: BaseLongAgent,
        position: Dict,
        buy_date: str,
        today: str,
    ) -> None:
        """
        对单只持仓从 buy_date+1 起前向扫描所有交易日，检查卖出信号。
        触发卖出 → 计算区间统计并平仓（status=1）。
        扫描到 today 仍未触发 → 保持 status=0（未结账）。

        性能优化：批量预取所有需要的数据，循环内零 DB 查询。
          - 一次性拉取 kline_day（buy_date ~ today）
          - 一次性拉取 kline_day_hfq（context_start ~ today）
          - 一次性查出持仓期间的除权除息日（从 stock_dividend 表）
          - MA 在循环内基于预取的 HFQ 数据在内存中计算

        :param agent:    长线 agent 实例
        :param position: 持仓记录 dict（ts_code, buy_price 等）
        :param buy_date: 买入日期
        :param today:    当前最新交易日（扫描上限）
        """
        ts_code   = position["ts_code"]
        buy_price = float(position["buy_price"])

        try:
            buy_idx = self.all_trade_dates.index(buy_date)
        except ValueError:
            logger.warning(f"[长线引擎][{agent.agent_id}][{ts_code}] buy_date {buy_date} 不在交易日列表中")
            return

        # ── 1. 确保 HFQ 数据入库（分两段，防止部分入库被跳过）──
        context_start = self.context_trade_dates[0] if self.context_trade_dates else buy_date
        self._ensure_hfq_data(ts_code, context_start, buy_date)
        self._ensure_hfq_data(ts_code, buy_date, today)

        # ── 2. 批量预取所有数据（替代逐日逐只 DB 查询）──
        # 2a. 原始日线：buy_date ~ today
        all_daily = get_kline_day_range([ts_code], buy_date, today)
        daily_by_date = {}
        if not all_daily.empty:
            for _, row in all_daily.iterrows():
                d = str(row["trade_date"]).replace("-", "")[:8]
                daily_by_date[d] = row.to_dict()

        # 2b. HFQ 日线：context_start ~ today（用于 MA 计算 + 止损判断）
        all_hfq = get_hfq_kline_range([ts_code], context_start, today)
        hfq_closes_by_date = {}
        hfq_close_list = []  # (trade_date_fmt, close) 有序列表
        if not all_hfq.empty:
            hfq_sorted = all_hfq[all_hfq["ts_code"] == ts_code].sort_values("trade_date")
            for _, row in hfq_sorted.iterrows():
                d = str(row["trade_date"]).replace("-", "")[:8]
                c = float(row["close"])
                hfq_closes_by_date[d] = c
                hfq_close_list.append((d, c))
        #
        # # 2b-check. HFQ 数据完整性校验：对比实际行数与预期交易日数
        # try:
        #     buy_idx_chk = self.all_trade_dates.index(buy_date)
        #     today_idx_chk = self.all_trade_dates.index(today)
        #     expected_hold_days = today_idx_chk - buy_idx_chk + 1
        # except ValueError:
        #     expected_hold_days = 0
        # # 持仓期 HFQ 行数（仅 buy_date ~ today 段，排除 context_start ~ buy_date 的 MA 回看数据）
        # hfq_hold_count = sum(1 for d, _ in hfq_close_list if d >= buy_date.replace("-", ""))
        # hold_coverage = (hfq_hold_count / expected_hold_days) if expected_hold_days > 0 else 0.0
        # if expected_hold_days > 5 and hold_coverage < 0.7:
        #     logger.warning(
        #         f"[长线引擎][{agent.agent_id}][{ts_code}] HFQ 数据严重不足！| "
        #         f"{buy_date}~{today} 预期≈{expected_hold_days}个交易日，"
        #         f"实际仅{hfq_hold_count}行 | 覆盖率{hold_coverage:.1%} | "
        #         f"MA/止损判断将失真，建议检查 Tushare API 或重新入库"
        #     )
        #     position["hfq_data_invalid"] = True

        # 2c. HFQ 买入价
        buy_date_fmt = buy_date.replace("-", "")
        hfq_buy_price = hfq_closes_by_date.get(buy_date_fmt, 0.0)
        position["hfq_buy_price"] = hfq_buy_price

        # 2d. 除权除息日集合（从 stock_dividend 表，一次查出整个持仓期）
        # ensure_dividend_data([ts_code])
        ex_div_dates = get_ex_div_dates_for_stock(ts_code, buy_date, today)

        # 2e. 预构建 HFQ close 序列索引（用于内存 MA 计算）
        hfq_date_list = [item[0] for item in hfq_close_list]
        hfq_val_list  = [item[1] for item in hfq_close_list]

        # ── 3. 前向扫描（循环内零 DB 查询）──
        for scan_idx in range(buy_idx + 1, len(self.all_trade_dates)):
            scan_date = self.all_trade_dates[scan_idx]
            if scan_date > today:
                break

            scan_date_fmt = scan_date.replace("-", "")
            today_row = daily_by_date.get(scan_date_fmt)
            if today_row is None:
                continue  # 停牌日跳过

            position["trading_days_so_far"] = scan_idx - buy_idx

            # 构建轻量卖出上下文（无 DB 查询）
            context = {
                "trade_date":    scan_date,
                "trade_dates":   self.context_trade_dates,
                "ex_div_stocks": {ts_code} if scan_date in ex_div_dates else set(),
                # 预计算的 HFQ MA 数据，供 agent 直接使用
                "_hfq_preloaded": {
                    "hfq_closes_by_date": hfq_closes_by_date,
                    "hfq_date_list":      hfq_date_list,
                    "hfq_val_list":       hfq_val_list,
                },
            }

            try:
                should_sell = agent.check_sell_signal(position, today_row, context)
            except Exception as e:
                logger.warning(
                    f"[长线引擎][{agent.agent_id}][{ts_code}][{scan_date}] "
                    f"check_sell_signal 异常：{e}"
                )
                continue

            if should_sell:
                sell_reason = position.pop("_sell_reason", "未知")
                sell_price = float(today_row["close"]) if today_row else 0.0
                stats = self._calc_period_stats(position, scan_date, sell_price)
                ok = self.pos_db.close_position(agent.agent_id, ts_code, buy_date, stats)
                if ok:
                    logger.info(
                        f"[长线引擎][{agent.agent_id}][{ts_code}] 平仓 | "
                        f"{buy_date} → {scan_date} | "
                        f"持仓 {stats['trading_days']} 日 | "
                        f"区间收益 {stats['period_return']:.2f}% | "
                        f"原因：{sell_reason}"
                    )
                    # 尝试聚合当日所有持仓的统计数据
                    self._try_aggregate_long_stats(agent, buy_date)
                return  # 已平仓，结束扫描

        # 扫描到 today 仍未触发卖出 → 未结账
        logger.debug(
            f"[长线引擎][{agent.agent_id}][{ts_code}] 未结账 | "
            f"买入 {buy_date}，扫描至 {today} 未触发卖出信号"
        )

    # ------------------------------------------------------------------ #
    # Daily 模式：仅检查未结账持仓在最新日期的卖出信号
    # ------------------------------------------------------------------ #

    def _check_unsettled_sell_signals(
        self,
        agent: BaseLongAgent,
        trade_date: str,
        daily_data: pd.DataFrame,
        context: Dict,
    ) -> None:
        """
        仅在 daily 模式下调用。
        遍历该 agent 所有 status=0 的未结账持仓，
        检查是否在 trade_date（最新交易日）触发卖出信号。
        卖出判断基于后复权（HFQ）数据，消除未来函数。
        """
        open_positions = self.pos_db.get_open_positions_before_date(agent.agent_id, trade_date)
        if not open_positions:
            return

        # 构建今日行情索引 {ts_code: row_dict}
        today_rows: Dict[str, Dict] = {}
        for _, row in daily_data.iterrows():
            today_rows[row["ts_code"]] = row.to_dict()

        # 批量确保 HFQ 数据已入库（每只持仓股票）
        context_start = self.context_trade_dates[0] if self.context_trade_dates else trade_date
        for pos in open_positions:
            self._ensure_hfq_data(pos["ts_code"], context_start, trade_date)

        for pos in open_positions:
            ts_code  = pos["ts_code"]
            buy_date = pos["buy_date"]
            today_row = today_rows.get(ts_code)   # None = 今日停牌

            # 计算已持有交易日数
            try:
                buy_idx   = self.all_trade_dates.index(buy_date)
                today_idx = self.all_trade_dates.index(trade_date)
                pos["trading_days_so_far"] = today_idx - buy_idx
            except ValueError:
                pos["trading_days_so_far"] = 0

            # 传递 HFQ 买入价（用于止损判断）
            pos["hfq_buy_price"] = self._get_hfq_buy_price(ts_code, buy_date)

            try:
                should_sell = agent.check_sell_signal(pos, today_row, context)
            except Exception as e:
                logger.warning(f"[长线引擎][{agent.agent_id}][{ts_code}] check_sell_signal 异常：{e}")
                self.pos_db.mark_error(agent.agent_id, ts_code, buy_date, f"sell_check_err:{e}")
                continue

            if not should_sell:
                continue

            sell_reason = pos.pop("_sell_reason", "未知")
            sell_price = float(today_row["close"]) if today_row else 0.0
            stats = self._calc_period_stats(pos, trade_date, sell_price)
            ok = self.pos_db.close_position(agent.agent_id, ts_code, buy_date, stats)
            if ok:
                logger.info(
                    f"[长线引擎][{agent.agent_id}][{ts_code}] 平仓（daily）| "
                    f"{buy_date} → {trade_date} | "
                    f"持仓 {stats['trading_days']} 日 | "
                    f"区间收益 {stats['period_return']:.2f}% | "
                    f"原因：{sell_reason}"
                )
                # 尝试聚合当日所有持仓的统计数据
                self._try_aggregate_long_stats(agent, buy_date)

    # ------------------------------------------------------------------ #
    # 长线聚合：平仓后回写平均盈亏到 agent_daily_profit_stats
    # ------------------------------------------------------------------ #

    def _try_aggregate_long_stats(
        self, agent: BaseLongAgent, buy_date: str
    ) -> None:
        """
        检查 buy_date 当日所有持仓是否全部平仓。
        若全部平仓（status=1），计算聚合统计并回写 agent_daily_profit_stats。
        """
        positions = self.pos_db.get_positions_by_buy_date(agent.agent_id, buy_date)
        if not positions:
            return

        # 检查是否全部平仓
        open_count = sum(1 for p in positions if p.get("status") == 0)
        if open_count > 0:
            return  # 仍有未平仓持仓，暂不聚合

        # 全部平仓，计算聚合统计
        returns = [
            float(p["period_return"])
            for p in positions
            if p.get("period_return") is not None
        ]
        trading_days_list = [
            int(p["trading_days"])
            for p in positions
            if p.get("trading_days") is not None
        ]

        if not returns:
            return

        returns_sorted = sorted(returns)
        n = len(returns_sorted)
        if n % 2 == 0:
            median_return = (returns_sorted[n // 2 - 1] + returns_sorted[n // 2]) / 2
        else:
            median_return = returns_sorted[n // 2]

        agg = {
            "long_median_return":    round(median_return, 4),
            "long_max_return":       round(max(returns), 4),
            "long_min_return":       round(min(returns), 4),
            "long_avg_trading_days": round(
                sum(trading_days_list) / len(trading_days_list), 2
            ) if trading_days_list else 0,
            "long_closed_count":     n,
        }

        ok = self.signal_db.update_long_agg_stats(agent.agent_id, buy_date, agg)
        if ok:
            logger.info(
                f"[长线引擎][{agent.agent_id}][{buy_date}] 聚合回写 | "
                f"平仓 {n} 只 | 中位收益 {agg['long_median_return']:.2f}% | "
                f"最高 {agg['long_max_return']:.2f}% | 最低 {agg['long_min_return']:.2f}% | "
                f"平均持仓 {agg['long_avg_trading_days']:.1f} 日"
            )

    # ------------------------------------------------------------------ #
    # 区间统计计算
    # ------------------------------------------------------------------ #

    def _calc_period_stats(
        self,
        position: Dict,
        sell_date: str,
        sell_price_raw: float,
    ) -> Dict:
        """
        计算区间统计字段（平仓时调用）。
        优先使用后复权数据；HFQ 表无数据时降级为原始价格并记录警告。
        """
        ts_code  = position["ts_code"]
        buy_date = position["buy_date"]
        buy_price_raw = float(position["buy_price"])

        hfq_df = get_hfq_kline_range([ts_code], buy_date, sell_date)
        use_hfq = not hfq_df.empty
        if use_hfq:
            range_df = hfq_df[hfq_df["ts_code"] == ts_code].copy()
            range_df = range_df.sort_values("trade_date")
        else:
            logger.warning(
                f"[长线引擎][{ts_code}] kline_day_hfq 无数据，降级使用原始价格 | "
                f"{buy_date}~{sell_date}"
            )
            range_df = get_kline_day_range([ts_code], buy_date, sell_date)
            if not range_df.empty:
                range_df = range_df[range_df["ts_code"] == ts_code].sort_values("trade_date")

        # 用交易日历计算预期持仓天数（含首尾），不依赖 HFQ 行数
        try:
            _bi = self.all_trade_dates.index(buy_date)
            _si = self.all_trade_dates.index(sell_date)
            expected_days = _si - _bi + 1  # 含首尾（与 range_df 完整时行数口径一致）
        except ValueError:
            expected_days = 0

        if range_df.empty:
            period_return = (sell_price_raw - buy_price_raw) / buy_price_raw * 100 if buy_price_raw else 0.0
            return {
                "sell_date":           sell_date,
                "sell_price":          round(sell_price_raw, 4),
                "period_return":       round(period_return, 4),
                "trading_days":        expected_days if expected_days > 0 else 0,
                "up_days":             0,
                "down_days":           0,
                "max_drawdown":        0.0,
                "max_floating_profit": 0.0,
                "daily_detail":        [],
            }

        # 异常检测：HFQ 数据行数远少于预期交易日（可能 API 限流导致入库不完整）
        if expected_days > 5 and len(range_df) < expected_days * 0.5:
            logger.warning(
                f"[长线引擎][{ts_code}] HFQ 数据严重不足！| "
                f"{buy_date}~{sell_date} 预期≈{expected_days}个交易日，"
                f"实际仅 {len(range_df)} 行，区间统计将失真 | "
                f"可能原因：Tushare API 限流/超时导致入库不完整"
            )
        elif len(range_df) <= 3 and buy_date != sell_date:
            logger.warning(
                f"[长线引擎][{ts_code}] HFQ 数据异常！| "
                f"{buy_date}~{sell_date} 仅 {len(range_df)} 行，区间统计将失真"
            )

        if use_hfq and len(range_df) > 0:
            buy_price = float(range_df.iloc[0]["close"])   # 买入日 HFQ 收盘价（信号基于收盘突破）
            sell_price = float(range_df.iloc[-1]["close"])
        else:
            buy_price  = buy_price_raw
            sell_price = sell_price_raw

        daily_detail = []
        max_high  = buy_price
        min_low   = buy_price
        up_days   = 0
        down_days = 0

        for _, row in range_df.iterrows():
            open_p  = float(row["open"])
            high_p  = float(row["high"])
            low_p   = float(row["low"])
            close_p = float(row["close"])
            trade_d = str(row["trade_date"]).replace("-", "")[:8]
            if len(trade_d) == 8:
                trade_d_fmt = f"{trade_d[:4]}-{trade_d[4:6]}-{trade_d[6:]}"
            else:
                trade_d_fmt = str(row["trade_date"])

            float_pnl = (close_p - buy_price) / buy_price * 100 if buy_price else 0.0
            if high_p > max_high:
                max_high = high_p
            if low_p < min_low:
                min_low = low_p

            if close_p >= open_p:
                up_days += 1
            else:
                down_days += 1

            daily_detail.append({
                "date":      trade_d_fmt,
                "open":      round(open_p,  4),
                "high":      round(high_p,  4),
                "low":       round(low_p,   4),
                "close":     round(close_p, 4),
                "float_pnl": round(float_pnl, 4),
            })

        period_return        = (sell_price - buy_price) / buy_price * 100 if buy_price else 0.0
        max_floating_profit  = (max_high - buy_price)  / buy_price * 100 if buy_price else 0.0
        max_drawdown         = (min_low  - buy_price)  / buy_price * 100 if buy_price else 0.0

        return {
            "sell_date":           sell_date,
            "sell_price":          round(sell_price, 4),
            "period_return":       round(period_return,       4),
            "trading_days":        expected_days if expected_days > 0 else len(range_df),
            "up_days":             up_days,
            "down_days":           down_days,
            "max_drawdown":        round(max_drawdown,        4),
            "max_floating_profit": round(max_floating_profit, 4),
            "daily_detail":        daily_detail,
        }

    # ------------------------------------------------------------------ #
    # 手动重置
    # ------------------------------------------------------------------ #

    def reset_agent(self, agent_id: str, from_date: str) -> None:
        """
        删除指定 agent 从 from_date 起的所有记录（买入信号表 + 持仓表）。
        仅由 run.py --reset-agent 触发，不自动调用。
        """
        self.signal_db.delete_records_from(agent_id, from_date)
        sql = f"DELETE FROM agent_long_position_stats WHERE agent_id = %s AND buy_date >= %s"
        try:
            from utils.db_utils import db
            deleted = db.execute(sql, (agent_id, from_date))
            logger.info(f"[长线引擎] 重置 {agent_id}：删除持仓记录 {deleted} 条（{from_date} 起）")
        except Exception as e:
            logger.error(f"[长线引擎] 重置持仓记录失败：{e}")

    # ------------------------------------------------------------------ #
    # 主流程
    # ------------------------------------------------------------------ #

    def run_full_flow(
        self,
        reset_agents: Optional[Dict[str, str]] = None,
        mode: str = "full",
    ) -> bool:
        """
        完整运行入口。

        :param reset_agents: {agent_id: from_date}，手动重跑指定 agent
        :param mode: "full"（默认）= 历史全量补全（前向扫描卖出信号）；
                     "daily"      = 仅处理最新一个交易日（检查未结账持仓）
        """
        if not self.agents:
            logger.info("[长线引擎] 无中长线 agent，跳过")
            return True

        if not self.all_trade_dates:
            logger.warning("[长线引擎] 交易日列表为空，跳过")
            return True

        today = self.all_trade_dates[-1]
        logger.info(f"===== AgentLongStatsEngine 启动 | 模式：{mode} | 最新交易日：{today} =====")

        # ── 1. 手动重置 ────────────────────────────────────────────────
        if reset_agents:
            for agent_id, from_date in reset_agents.items():
                if agent_id.startswith("long_"):
                    logger.info(f"[长线引擎][重置] {agent_id} 从 {from_date} 起")
                    self.reset_agent(agent_id, from_date)

        # ── daily 模式：仅检查未结账持仓的卖出信号 ──────────────────────
        if mode == "daily":
            logger.info(f"[长线引擎][daily] 最新交易日 {today}：检查未结账持仓 + 新买入信号")

            daily_data = get_daily_kline_data(today)
            if daily_data.empty:
                logger.warning(f"[长线引擎][{today}] 日线数据空，跳过")
                return True

            context = self._get_context(today)

            # Step A：生成新的买入信号
            # HFQ 数据由各 agent 内部按需拉取（仅粗筛通过的少量股票）
            for agent in self.agents:
                try:
                    stock_detail = self._process_buy_signal(agent, today, daily_data, context)
                    # 新买入的股票不需要当天检查卖出（buy_date < trade_date 才检查）
                except TushareRateLimitAbort:
                    logger.error(f"[长线引擎][{today}] Tushare 严重限流，中断")
                    return False
                except Exception as e:
                    logger.error(f"[长线引擎][{agent.agent_id}][{today}] 买入信号异常：{e}")

            # Step B：检查所有未结账持仓在最新日期的卖出信号
            for agent in self.agents:
                try:
                    self._check_unsettled_sell_signals(agent, today, daily_data, context)
                except Exception as e:
                    logger.error(f"[长线引擎][{agent.agent_id}][{today}] 未结账卖出检查异常：{e}")

            logger.info("===== AgentLongStatsEngine（daily）运行完成 =====")
            return True

        # ── full 模式：历史全量补全（前向扫描卖出信号）──────────────────
        # 2. 确定每个 agent 需要补全的买入信号日期
        last_signal_dates = self.signal_db.get_all_agents_last_dates()
        all_td_set        = set(self.all_trade_dates)

        dates_missing_buy: Dict[str, set] = {}
        logger.info("─── 各长线 agent 断点续跑计划 ─────────────────────────────────")
        for agent in self.agents:
            aid      = agent.agent_id
            recorded = set(self.signal_db.get_agent_recorded_dates(aid))
            err_set  = set(self.signal_db.get_agent_err_date_list(aid))
            missing  = (all_td_set - recorded) | (err_set & all_td_set)
            dates_missing_buy[aid] = missing
            last = last_signal_dates.get(aid, "无记录")
            logger.info(f"  [{aid:35s}] 最后有效日期：{last} | 待处理买入信号：{len(missing)} 日")
        logger.info("───────────────────────────────────────────────────────────────")

        # 3. 汇总需要处理的买入信号日期（升序）
        all_buy_dates = sorted(
            set(d for dl in dates_missing_buy.values() for d in dl)
        )

        if not all_buy_dates:
            # 无新买入信号要处理，但可能有遗留的未结账持仓需要前向扫描
            logger.info("[长线引擎] 无新买入信号日期，检查遗留未结账持仓...")
            self._scan_all_unsettled_positions(today)
            logger.info("===== AgentLongStatsEngine 运行完成 =====")
            return True

        logger.info(
            f"[长线引擎] 待处理买入信号日期：{len(all_buy_dates)} 个 | "
            f"{all_buy_dates[0]} ~ {all_buy_dates[-1]}"
        )

        # 4. 逐日处理：生成买入信号 → 立即前向扫描每只股票的卖出信号
        for trade_date in all_buy_dates:
            agents_need_buy = [
                a for a in self.agents
                if trade_date in dates_missing_buy.get(a.agent_id, set())
            ]
            if not agents_need_buy:
                continue

            daily_data = get_daily_kline_data(trade_date)
            if daily_data.empty:
                logger.warning(f"[长线引擎][{trade_date}] 日线数据空，跳过")
                for agent in agents_need_buy:
                    self.signal_db.insert_error_record(
                        agent.agent_id, agent.agent_name, trade_date,
                        "daily_data_empty"
                    )
                continue

            context = self._get_context(trade_date)

            # HFQ 数据由各 agent 内部按需拉取（仅粗筛通过的少量股票）
            for agent in agents_need_buy:
                try:
                    stock_detail = self._process_buy_signal(agent, trade_date, daily_data, context)
                except TushareRateLimitAbort:
                    logger.error(f"[长线引擎][{trade_date}] Tushare 严重限流，中断")
                    return False
                except Exception as e:
                    logger.error(f"[长线引擎][{agent.agent_id}][{trade_date}] 买入信号异常：{e}")
                    continue

                if not stock_detail:
                    continue

                # 立即前向扫描每只买入股票的卖出信号
                for s in stock_detail:
                    position = {
                        "agent_id":  agent.agent_id,
                        "ts_code":   s["ts_code"],
                        "buy_date":  trade_date,
                        "buy_price": s["buy_price"],
                    }
                    try:
                        self._forward_scan_sell_for_position(
                            agent, position, trade_date, today
                        )
                    except Exception as e:
                        logger.error(
                            f"[长线引擎][{agent.agent_id}][{s['ts_code']}] "
                            f"前向扫描卖出信号异常：{e}"
                        )

        # 5. 处理完所有新买入信号后，再扫描遗留的未结账持仓
        # （可能是之前运行中产生的、或 reset 后遗留的）
        self._scan_all_unsettled_positions(today)

        logger.info("===== AgentLongStatsEngine 运行完成 =====")
        return True

    def _scan_all_unsettled_positions(self, today: str) -> None:
        """
        扫描所有 agent 的未结账持仓（status=0），前向扫描卖出信号。
        HFQ 数据预取由 _forward_scan_sell_for_position 内部完成。
        用于：
          1. full 模式末尾：处理之前运行遗留的未结账持仓
          2. 当没有新买入信号时：确保历史未结账持仓也能被处理
        """
        for agent in self.agents:
            open_positions = self.pos_db.get_open_positions(agent.agent_id)
            if not open_positions:
                continue

            logger.info(
                f"[长线引擎][{agent.agent_id}] 扫描 {len(open_positions)} 个未结账持仓"
            )
            for pos in open_positions:
                buy_date = pos["buy_date"]
                try:
                    self._forward_scan_sell_for_position(agent, pos, buy_date, today)
                except Exception as e:
                    logger.error(
                        f"[长线引擎][{agent.agent_id}][{pos['ts_code']}] "
                        f"未结账前向扫描异常：{e}"
                    )
