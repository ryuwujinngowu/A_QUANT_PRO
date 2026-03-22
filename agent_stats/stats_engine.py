"""
核心统计引擎
============
职责：按时间序列，对所有智能体跟踪 T 日选股信号 + T+1 日隔日表现，写入 DB。

执行模式
--------
1. 历史补全（首次/断点续跑/新增 agent）
   - 新 agent（DB 无记录）：从 config.START_DATE 开始逐日处理至最新交易日
   - 已有 agent：从 DB 最后一条记录的次日开始续跑
   - 已处理但缺 D+1 数据的记录：发现后自动补全（结账）
   - 手动重跑：调用方传入 reset_agents 字典（见 run_full_flow 参数说明）

2. 日常运行（每日凌晨 3 点 cron 触发）
   - 与历史补全流程相同，只是缺失日期仅有"今日"（最新交易日）一天

每日处理流程（per agent, per date）
-------------------------------------
  Step A  生成 T 日选股信号 → 立即 INSERT（防崩溃丢数）
  Step B  若 T+1 日数据已存在（历史日期）→ 立即计算并 UPDATE 隔日表现
  Step C  若 T+1 为未来 → 跳过，次日运行时 Step B 会补全

异常处理
--------
- 单日单 agent 异常：跳过该 agent 当天，插入 error 占位记录（reserve_str_1 存错误信息）
- 数据加载异常：同日所有 agent 跳过，统一记录
- 断点续跑：重启后从 DB 末尾继续，已处理的日期不重复处理（ON DUPLICATE KEY 幂等）

关键设计
--------
- 每个交易日的全市场日线只加载一次，所有 agent 共享（减少 IO）
- 分钟线并发拉取（ThreadPoolExecutor，I/O 密集）
- D+1 统计写入时，D 日信号记录已存在，失败只影响 D+1 字段，不丢 D 日数据
"""

import inspect
import importlib
import pkgutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from agent_stats.config import START_DATE, MAX_RETRY_TIMES
from agent_stats.agent_base import BaseAgent
from agent_stats.agent_db_operator import AgentStatsDBOperator
from agent_stats.engine_shared import (
    get_next_trade_date as _shared_get_next_trade_date,
    build_trade_date_context,
    calc_intraday_stats as _shared_calc_intraday_stats,
)
from utils.log_utils import logger
from utils.common_tools import get_trade_dates, get_daily_kline_data, get_st_stock_codes, get_prev_trade_date
from utils.wechat_push import send_wechat_message_to_multiple_users
from data.data_cleaner import data_cleaner, TushareRateLimitAbort


class AgentStatsEngine:
    def __init__(self, start_date: str = None):
        self.start_date = start_date or START_DATE
        self.db_operator = AgentStatsDBOperator()
        self.agents: List[BaseAgent] = self._auto_load_agents()

        # ── 处理日期范围 ──────────────────────────────────────────────────
        # 触发时间为交易日 T 的凌晨 3 点，此时 T-1 日线已入库，T 日尚未开盘。
        # 因此只处理到 T-1（即上一个完整交易日），避免因 T 日数据未入库导致
        # 写入空信号 error 记录、后续无法补跑。
        last_trade_date = get_prev_trade_date()      # 上一个完整交易日

        # all_trade_dates：信号生成的处理窗口（从 START_DATE 到上一交易日）
        self.all_trade_dates: List[str] = get_trade_dates(self.start_date, last_trade_date)

        # context_trade_dates：含历史回看窗口，供 agent 计算 MA / 5日涨幅等。
        # 向前延伸 90 个自然日（≈60 个交易日），覆盖 30日均线等长周期需求。
        context_start = (
            datetime.strptime(self.start_date, "%Y-%m-%d") - timedelta(days=90)
        ).strftime("%Y-%m-%d")
        self.context_trade_dates: List[str] = get_trade_dates(context_start, last_trade_date)

        logger.info(
            f"引擎初始化完成 | 智能体：{len(self.agents)} 个 | "
            f"处理范围：{self.start_date} ~ {last_trade_date} | "
            f"上下文范围：{context_start} ~ {last_trade_date}"
        )

    # ------------------------------------------------------------------ #
    # Agent 自动发现
    # ------------------------------------------------------------------ #

    def _auto_load_agents(self) -> List[BaseAgent]:
        """
        扫描 agent_stats/agents/ 目录，自动实例化所有继承 BaseAgent 的类。
        新增/删除 agent 只需在 agents/ 放置/删除 .py 文件，无需改任何配置。

        注意：agent_id 以 "long_" 开头的中长线 agent 由 AgentLongStatsEngine 处理，
        此处自动跳过，避免被短线流程错误处理。
        """
        from agent_stats.long_agent_base import BaseLongAgent
        import agent_stats.agents as agents_pkg
        agents = []
        for _, module_name, _ in pkgutil.iter_modules(agents_pkg.__path__):
            full_module = f"agent_stats.agents.{module_name}"
            try:
                module = importlib.import_module(full_module)
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (
                        issubclass(obj, BaseAgent)
                        and obj is not BaseAgent
                        and obj.__module__ == full_module
                    ):
                        # 中长线 agent 由 AgentLongStatsEngine 处理，短线引擎跳过
                        if issubclass(obj, BaseLongAgent):
                            continue
                        instance = obj()
                        if not instance.agent_id or not instance.agent_name:
                            logger.error(f"[{full_module}.{name}] 缺少 agent_id/agent_name，跳过")
                            continue
                        agents.append(instance)
                        logger.info(f"加载智能体：{instance.agent_name}（{instance.agent_id}）")
            except Exception as e:
                logger.error(f"[{full_module}] 加载失败：{e}", exc_info=True)
        if not agents:
            logger.warning("⚠ 未发现有效短线智能体，请检查 agent_stats/agents/ 目录")
        return agents

    # ------------------------------------------------------------------ #
    # 工具方法
    # ------------------------------------------------------------------ #

    def _get_next_trade_date(self, trade_date: str) -> Optional[str]:
        return _shared_get_next_trade_date(self.all_trade_dates, trade_date)

    def _get_trade_date_context(self, trade_date: str) -> Dict:
        return build_trade_date_context(trade_date, self.context_trade_dates)

    # ------------------------------------------------------------------ #
    # 日内统计
    # ------------------------------------------------------------------ #

    def _calc_intraday_stats(
        self, stock_list: List[Dict], trade_date: str
    ) -> Tuple[float, List[Dict]]:
        return _shared_calc_intraday_stats(stock_list, trade_date)

    # ------------------------------------------------------------------ #
    # 隔日统计（T+1）
    # ------------------------------------------------------------------ #

    @staticmethod
    def _zero_next_day_stats() -> Dict:
        """
        返回一个全零的 D+1 统计字典，用于：
          1. 当日信号池为空（无命中股票）时，显式将 D+1 字段写为 0，
             标记该记录为"正常结账（空池）"，避免与 DB 默认值混淆。
          2. 与 `get_agents_closed_dates`（IS NOT NULL 判断）配合，
             确保空池记录不被引擎重复处理。
        """
        return {
            "next_day_avg_open_premium":    0.0,
            "next_day_avg_close_return":    0.0,
            "next_day_avg_red_minute":      0,
            "next_day_avg_profit_minute":   0,
            "next_day_avg_intraday_profit": 0.0,
            "next_day_avg_max_premium":     0.0,
            "next_day_avg_max_drawdown":    0.0,
            "next_day_stock_detail":        {"stock_list": []},
        }

    def _calc_next_day_stats(
        self,
        stock_list:     List[Dict],
        trade_date:     str,
        next_trade_date: str,
    ) -> Tuple[Dict, List[Dict]]:
        """
        计算 T+1 日隔日表现。
        注意：调用前必须确认 next_trade_date 已经是过去的完整交易日，
              否则日线/分钟线数据不存在，会返回 (None, [])。
        空信号池直接返回 (None, [])，由调用方（_process_one）写显式零值。
        """
        if not stock_list:
            return None, []

        ts_code_list  = [s["ts_code"]          for s in stock_list]
        buy_price_map = {s["ts_code"]: s["buy_price"]          for s in stock_list}
        t_close_map   = {s["ts_code"]: s.get("intraday_close_price", s["buy_price"]) for s in stock_list}
        name_map      = {s["ts_code"]: s.get("stock_name", "")  for s in stock_list}

        # T+1 日线
        next_df = get_daily_kline_data(next_trade_date, ts_code_list=ts_code_list)
        if next_df.empty:
            logger.warning(f"[{trade_date}→{next_trade_date}] T+1 日线空")
            return {}, []

        # 并发拉取 T+1 分钟线（I/O 密集）
        def _fetch_min(ts_code: str) -> Tuple[str, pd.DataFrame]:
            try:
                return ts_code, data_cleaner.get_kline_min_by_stock_date(ts_code, next_trade_date)
            except Exception as e:
                logger.warning(f"[{ts_code}][{next_trade_date}] 分钟线拉取失败：{e}")
                return ts_code, pd.DataFrame()

        minute_map: Dict[str, pd.DataFrame] = {}
        with ThreadPoolExecutor(max_workers=min(len(ts_code_list), 8)) as pool:
            for code, df in pool.map(_fetch_min, ts_code_list):
                minute_map[code] = df

        # 逐只计算
        detail = []
        open_ret_list, close_ret_list = [], []
        max_prem_list, max_dd_list   = [], []
        red_min_list, profit_min_list, avg_prof_list = [], [], []

        for _, row in next_df.iterrows():
            ts   = row["ts_code"]
            bp   = buy_price_map.get(ts, 0)
            tc   = t_close_map.get(ts, 0)
            if bp <= 0:
                continue
            # 【修复】将 Decimal 类型统一转换为 float，避免类型不匹配
            open_p = float(row["open"])
            close_p = float(row["close"])
            high_p = float(row["high"])
            low_p = float(row["low"])
            amount = float(row["amount"])
            volume = float(row["volume"]) if row["volume"] > 0 else 0.0

            open_ret = (open_p - bp) / bp * 100
            close_ret = (close_p - bp) / bp * 100
            max_prem = (high_p - bp) / bp * 100
            # 最大回撤：负值表示曾跌破买入价（亏损），正值表示全天未跌破买入价
            max_dd = (low_p - bp) / bp * 100
            # VWAP：amount 单位=千元，volume 单位=手（100股），须乘以 10 还原每股价格
            # VWAP = (amount × 1000元) / (volume × 100股) = amount × 10 / volume
            avg_price = (amount * 10) / volume if volume > 0 else close_p
            avg_prof = (avg_price - bp) / bp * 100

            open_dir = (
                "high_open" if open_p > tc
                else ("low_open" if open_p < tc else "flat_open")
            )

            # ...（后续代码保持不变，使用转换后的 open_p/close_p 等变量）...
            min_df = minute_map.get(ts, pd.DataFrame())
            red_min, profit_min = 0, 0
            if not min_df.empty:
                red_min = int((min_df["close"] >= tc).sum())
                profit_min = int((min_df["close"] >= bp).sum())

            open_ret_list.append(open_ret);
            close_ret_list.append(close_ret)
            max_prem_list.append(max_prem);
            max_dd_list.append(max_dd)
            red_min_list.append(red_min);
            profit_min_list.append(profit_min)
            avg_prof_list.append(avg_prof)

            detail.append({
                "ts_code": ts,
                "stock_name": name_map.get(ts, ""),
                "buy_price": bp,
                "next_open_price": open_p,
                "next_close_price": close_p,
                "next_day_avg_price": round(avg_price, 4),
                "open_return": round(open_ret, 4),
                "close_return": round(close_ret, 4),
                "intraday_avg_profit": round(avg_prof, 4),
                "intraday_max_return": round(max_prem, 4),
                "intraday_max_drawdown": round(max_dd, 4),
                "red_minute_total": red_min,
                "profit_minute_total": profit_min,
                "open_direction": open_dir,
            })

        def sm(arr):
            return round(float(np.mean(arr)) if arr else 0.0, 4)

        stats = {
            "next_day_avg_open_premium":   sm(open_ret_list),
            "next_day_avg_close_return":   sm(close_ret_list),
            "next_day_avg_red_minute":     int(np.mean(red_min_list)    if red_min_list    else 0),
            "next_day_avg_profit_minute":  int(np.mean(profit_min_list) if profit_min_list else 0),
            "next_day_avg_intraday_profit":sm(avg_prof_list),
            "next_day_avg_max_premium":    sm(max_prem_list),
            "next_day_avg_max_drawdown":   sm(max_dd_list),
            "next_day_stock_detail":       {"stock_list": detail},
        }
        return stats, detail

    # ------------------------------------------------------------------ #
    # 单 Agent × 单日 处理（原子单元：任何异常只影响该 agent 当天）
    # ------------------------------------------------------------------ #

    def _process_one(
        self,
        agent:       BaseAgent,
        trade_date:  str,
        daily_data:  pd.DataFrame,
        context:     Dict,
        today:       str,
        skip_signal: bool = False,   # True = 信号已存在，只补 D+1
    ) -> bool:
        """
        处理 agent 在 trade_date 的完整任务：
          1. 若 skip_signal=False → 生成信号并 INSERT
          2. 若 T+1 日已是历史日期 → 计算并 UPDATE 隔日表现
        出现异常时插入 error 占位记录，返回 False。
        """
        next_date = self._get_next_trade_date(trade_date)

        # ── Step A: 生成信号 ──────────────────────────────────────────────
        stock_detail: List[Dict] = []
        if not skip_signal:
            try:
                signal_list = agent.get_signal_stock_pool(trade_date, daily_data, context)
                avg_ret, stock_detail = self._calc_intraday_stats(signal_list, trade_date)
                self.db_operator.insert_signal_record({
                    "agent_id":            agent.agent_id,
                    "agent_name":          agent.agent_name,
                    "agent_desc":          agent.agent_desc,
                    "trade_date":          trade_date,
                    "intraday_avg_return": avg_ret,
                    "signal_stock_detail": {"stock_list": stock_detail},
                })
                logger.info(
                    f"[{agent.agent_id}][{trade_date}] 信号入库 | 命中 {len(signal_list)} 只 "
                    f"| 日内均收益 {avg_ret:.2f}%"
                )
                # 检查是否有分钟线永久失败的股票，写入 DB 供运维追踪
                min_failures = agent.minute_fetch_failures
                if min_failures:
                    warn = (
                        f"{len(min_failures)} 只候选分钟线 {10} 次重试失败被跳过："
                        f" {','.join(min_failures[:10])}{'...' if len(min_failures) > 10 else ''}"
                    )
                    self.db_operator.mark_min_fetch_incomplete(agent.agent_id, trade_date, warn)
                    logger.warning(f"[{agent.agent_id}][{trade_date}] ⚠ 已标记数据不完整：{warn}")
            except TushareRateLimitAbort:
                # 严重限流：不写 error 记录（该日未处理，下次运行断点续跑），直接向上传播
                logger.error(
                    f"[{agent.agent_id}][{trade_date}] Tushare 严重限流中断，该日未处理，"
                    f"下次运行将从此日断点续跑"
                )
                raise
            except Exception as e:
                logger.error(f"[{agent.agent_id}][{trade_date}] 信号生成失败：{e}", exc_info=True)
                self.db_operator.insert_error_record(agent.agent_id, agent.agent_name, trade_date, str(e))
                return False
        else:
            # skip_signal=True：D 日信号已在 DB，先读已有信号
            try:
                stock_detail = self.db_operator.get_signal_detail(agent.agent_id, trade_date)
            except Exception as e:
                logger.warning(f"[{agent.agent_id}][{trade_date}] 读取已有信号失败：{e}")
                return False

            # 若已有信号为空，重新执行策略验证：
            # 历史行情数据不变，重跑结果可信；旧代码 bug 可能导致信号错误为空，
            # 修复后重跑可纠正。若重跑后仍为空则为真实空池，写零结账。
            if not stock_detail:
                logger.info(
                    f"[{agent.agent_id}][{trade_date}] DB 中信号为空，用当日行情重新验证..."
                )
                try:
                    signal_list = agent.get_signal_stock_pool(trade_date, daily_data, context)
                    avg_ret, stock_detail = self._calc_intraday_stats(signal_list, trade_date)
                    self.db_operator.insert_signal_record({
                        "agent_id":            agent.agent_id,
                        "agent_name":          agent.agent_name,
                        "agent_desc":          agent.agent_desc,
                        "trade_date":          trade_date,
                        "intraday_avg_return": avg_ret,
                        "signal_stock_detail": {"stock_list": stock_detail},
                    })
                    if signal_list:
                        logger.info(
                            f"[{agent.agent_id}][{trade_date}] 重验：命中 {len(signal_list)} 只"
                            f"（原信号为空，已覆盖）| 日内均收益 {avg_ret:.2f}%"
                        )
                    else:
                        logger.info(
                            f"[{agent.agent_id}][{trade_date}] 重验：策略确认为空池，写零结账"
                        )
                    # 分钟线失败情况同样写入 DB
                    min_failures = agent.minute_fetch_failures
                    if min_failures:
                        warn = (
                            f"{len(min_failures)} 只候选分钟线 {10} 次重试失败被跳过："
                            f" {','.join(min_failures[:10])}{'...' if len(min_failures) > 10 else ''}"
                        )
                        self.db_operator.mark_min_fetch_incomplete(agent.agent_id, trade_date, warn)
                except TushareRateLimitAbort:
                    raise
                except Exception as e:
                    logger.warning(
                        f"[{agent.agent_id}][{trade_date}] 重验信号失败，保留空池：{e}"
                    )
                    stock_detail = []

        # ── Step B: 隔日表现（仅当 T+1 是已完成的历史交易日）────────────
        if next_date and next_date <= today:
            try:
                stats, _ = self._calc_next_day_stats(stock_detail, trade_date, next_date)
                if stats is None:
                    # 信号池为空（无命中股票），显式写零值 D+1，标记为"正常结账（空池）"。
                    # 避免依赖 DB 默认值：IS NOT NULL 检查需要字段为显式写入的值，
                    # 否则 get_agents_closed_dates 无法正确识别"已结账但空池"的记录。
                    self.db_operator.update_next_day_stats(
                        agent.agent_id, trade_date, self._zero_next_day_stats()
                    )
                    logger.info(
                        f"[{agent.agent_id}][{trade_date}] 信号池为空，D+1 写零值结账"
                    )
                elif not stats:
                    # T+1 日线数据缺失（_calc_next_day_stats 返回空 dict），
                    # 本次跳过，下次运行 dates_unclosed 会将其重新纳入结账。
                    logger.warning(
                        f"[{agent.agent_id}][{trade_date}→{next_date}] T+1 日线缺失，跳过结账"
                    )
                else:
                    self.db_operator.update_next_day_stats(agent.agent_id, trade_date, stats)
                    logger.info(
                        f"[{agent.agent_id}][{trade_date}→{next_date}] "
                        f"隔日表现更新 | 均收益 {stats['next_day_avg_close_return']:.2f}%"
                    )
            except Exception as e:
                logger.warning(f"[{agent.agent_id}][{trade_date}] D+1 统计失败：{e}", exc_info=True)
                # 标记 D+1 计算异常（不影响已入库的信号数据）
                self.db_operator.mark_error(
                    agent.agent_id, trade_date, f"[next_day_err]{str(e)[:200]}"
                )
        return True

    # ------------------------------------------------------------------ #
    # 主流程
    # ------------------------------------------------------------------ #

    def repair_incomplete_records(self) -> int:
        """
        修复历史遗留的数据不完整记录（--repair-incomplete 模式入口）。

        修复对象
        --------
        1. reserve_str_1 LIKE '[MIN_FAIL]%' 的记录：
           某日信号生成时有分钟线永久失败，导致信号池不完整。
           修复策略：删除该记录，run_full_flow 断点续跑会将其视为「新日期」
           重新生成信号（此时分钟线已在 DB 缓存，成功率大幅提升）。

        2. next_day_avg_close_return IS NULL 的非错误非空池记录：
           D+1 统计未完成（历史遗留）。
           修复策略：无需额外操作，run_full_flow 的 dates_unclosed 逻辑
           已自动包含这类记录并重算 D+1。

        3. 空信号池 + next_day_avg_close_return IS NULL 的记录：
           旧版引擎未显式写零值结账。
           修复策略：get_empty_pool_unsettled_records 仍存在，
           此处不再写零填充，这类记录在 run_full_flow 的 dates_unclosed
           中会被重走 _process_one(skip_signal=True)，_calc_next_day_stats
           返回 None，进而写显式零值，完成结账。

        返回：删除的 [MIN_FAIL] 记录数（run_full_flow 会重新处理这些日期）
        """
        records = self.db_operator.get_min_fail_records()
        if not records:
            logger.info("[repair-incomplete] 无分钟线聚合告警记录，D+1 NULL 记录由 run_full_flow 自动处理")
            return 0

        logger.info(f"[repair-incomplete] 发现 {len(records)} 条 [MIN_FAIL] 记录，删除后由 run_full_flow 重新处理...")
        deleted = 0
        for rec in records:
            aid  = rec["agent_id"]
            date = rec["trade_date"]
            ok   = self.db_operator.delete_record(aid, date)
            if ok:
                deleted += 1
                logger.info(f"[repair-incomplete][{aid}][{date}] ✓ 已删除，等待重新处理")
            else:
                logger.warning(f"[repair-incomplete][{aid}][{date}] 删除失败")

        logger.info(f"[repair-incomplete] 删除完成：{deleted}/{len(records)} 条，run_full_flow 将重算这些日期")
        return deleted

    def run_full_flow(
        self,
        reset_agents: Optional[Dict[str, str]] = None,
        mode: str = "full",
    ) -> bool:
        """
        完整运行入口（cron 每日凌晨调用 / 手动补全均走此方法）。

        参数
        ----
        reset_agents : {agent_id: from_date} 字典，手动指定需要重跑的 agent。
        mode         : "full"（默认）= 历史全量补全；
                       "daily"       = 仅处理最新一个交易日（cron 模式，速度快）。
                       daily 模式下 D+1 结账仍全量执行（不影响未结账记录补全）。
        """
        today = self.all_trade_dates[-1] if self.all_trade_dates else datetime.now().strftime("%Y-%m-%d")
        logger.info(f"===== AgentStatsEngine 启动 | 模式：{mode} | 最新交易日：{today} =====")

        # ── 1. 手动重置 ────────────────────────────────────────────────
        if reset_agents:
            for agent_id, from_date in reset_agents.items():
                logger.info(f"[重置] {agent_id} 从 {from_date} 起删除并重跑")
                self.db_operator.delete_records_from(agent_id, from_date)

        # ── 2. 查询全局状态（单次批量查询，减少 DB 往返）────────────────
        last_dates   = self.db_operator.get_all_agents_last_dates()
        closed_dates = self.db_operator.get_agents_closed_dates()
        all_td_set   = set(self.all_trade_dates)

        # ── 3. 计算每个 agent 需要处理的日期 ────────────────────────────
        # dates_missing[agent_id]  = 需要生成信号的日期集合（set，O(1) 查找）
        #   = 窗口内无记录的日期（含 START_DATE 往前调整后的历史段）+ [ERR] 日期（重算修复）
        # dates_unclosed[agent_id] = 信号已存在但 D+1 缺失的日期列表
        dates_missing:  Dict[str, set]       = {}
        dates_unclosed: Dict[str, List[str]] = {}

        logger.info("─── 各 agent 断点续跑计划 ───────────────────────────────────")
        for agent in self.agents:
            aid = agent.agent_id

            # 一次性拿到该 agent 的所有已入库日期（含 ERR）和仅 ERR 日期
            recorded_set = set(self.db_operator.get_agent_recorded_dates(aid))
            err_set      = set(self.db_operator.get_agent_err_date_list(aid))

            # 核心公式（三合一）：
            #   窗口内无任何记录的日期  → 新日期续跑 or START_DATE 往前调整后的历史补录
            #   窗口内 [ERR] 日期       → 重算修复（UPSERT 成功时自动清空 reserve_str_1）
            missing = (all_td_set - recorded_set) | (err_set & all_td_set)
            dates_missing[aid] = missing

            # ── 日志描述 ────────────────────────────────────────────────
            last      = last_dates.get(aid)
            err_cnt   = len(err_set & all_td_set)
            gap_dates = sorted(missing - (err_set & all_td_set))   # 纯空白日期（非 ERR）

            if last is None:
                resume_desc = (
                    f"无有效记录（含 {err_cnt} 条 [ERR]），从头开始"
                    if err_cnt else "新 agent，从头开始"
                )
            elif gap_dates and gap_dates[0] < last:
                # START_DATE 往前移，窗口内出现早于最早记录的历史段
                resume_desc = (
                    f"上次有效日期 {last}，补录历史自 {gap_dates[0]}"
                    + (f"（另有 {err_cnt} 条 [ERR] 重算）" if err_cnt else "")
                )
            elif gap_dates:
                resume_desc = (
                    f"上次有效日期 {last}，续跑自 {gap_dates[0]}"
                    + (f"（另有 {err_cnt} 条 [ERR] 重算）" if err_cnt else "")
                )
            else:
                resume_desc = (
                    f"上次有效日期 {last}，已是最新"
                    + (f"（另有 {err_cnt} 条 [ERR] 重算）" if err_cnt else "")
                )

            logger.info(f"  [{aid:30s}] {resume_desc} | 待处理 {len(missing)} 日")

            # unclosed：有效记录（非 ERR）且 D+1 缺失
            valid_recorded = recorded_set - err_set
            agent_closed   = closed_dates.get(aid, set())
            unclosed = [
                d for d in valid_recorded
                if d not in agent_closed
                and d < today
                and self._get_next_trade_date(d) is not None
                and self._get_next_trade_date(d) <= today
            ]
            dates_unclosed[aid] = unclosed

        logger.info("─────────────────────────────────────────────────────────────")

        # ── 4. 汇总需要处理的日期集合，升序遍历 ─────────────────────────
        # daily 模式：买入信号只处理最新一天（D+1 结账仍全量）
        if mode == "daily":
            for aid in dates_missing:
                dates_missing[aid] = {today} if today in dates_missing[aid] else set()

        all_dates_to_process = sorted(
            set(d for dl in dates_missing.values()  for d in dl)
            | set(d for dl in dates_unclosed.values() for d in dl)
        )

        if not all_dates_to_process:
            logger.info("所有 agent 均已是最新，无需处理")
            return True

        logger.info(f"待处理交易日数：{len(all_dates_to_process)}，"
                    f"范围 {all_dates_to_process[0]} ~ {all_dates_to_process[-1]}")

        for trade_date in all_dates_to_process:
            # 找出本日需要执行的 agent（missing 或 unclosed）
            # dates_missing values 为 set，O(1) 查找
            agents_need_signal  = [a for a in self.agents if trade_date in dates_missing.get(a.agent_id, set())]
            agents_need_d1_only = [a for a in self.agents if trade_date in dates_unclosed.get(a.agent_id, [])
                                   and a not in agents_need_signal]

            if not agents_need_signal and not agents_need_d1_only:
                continue

            # 每个交易日只加载一次共享数据
            daily_data = get_daily_kline_data(trade_date)
            if daily_data.empty:
                logger.warning(f"[{trade_date}] 全市场日线为空，跳过所有 agent")
                for agent in agents_need_signal:
                    self.db_operator.insert_error_record(
                        agent.agent_id, agent.agent_name, trade_date,
                        f"daily_data_empty: kline_day 无数据"
                    )
                continue

            context = self._get_trade_date_context(trade_date)

            # 并发处理本日所有 agent（单 agent 异常隔离，互不影响）
            # TushareRateLimitAbort 除外 — 严重限流时中止整个日期的处理
            all_agents_today = [
                (a, False) for a in agents_need_signal
            ] + [
                (a, True)  for a in agents_need_d1_only
            ]

            def _run(task):
                agent, skip_signal = task
                return self._process_one(agent, trade_date, daily_data, context, today, skip_signal)

            try:
                with ThreadPoolExecutor(max_workers=min(len(all_agents_today), 4)) as pool:
                    list(pool.map(_run, all_agents_today))
            except TushareRateLimitAbort as e:
                # 严重限流：未处理的 agent 本日无记录，下次运行将从断点续跑。
                # 已处理的 agent 记录已入库，不受影响。
                logger.error(
                    f"[{trade_date}] Tushare 分钟线严重限流，中断当日历史补全 | {e}"
                )
                try:
                    send_wechat_message_to_multiple_users(
                        "【agent_stats 分钟线限流中断】",
                        f"处理日期 {trade_date} 时触发 Tushare API 严重限流（连续失败超阈值），"
                        f"当日历史补全已中断。\n\n"
                        f"各 agent 支持断点续跑，下次凌晨 3 点 cron 或手动运行时将自动从断点恢复。\n"
                        f"若次日仍失败，请检查 Tushare 账号当日用量是否耗尽（上限 5W 次）。"
                    )
                except Exception:
                    pass
                return False

        logger.info("===== AgentStatsEngine 运行完成 =====")
        return True
