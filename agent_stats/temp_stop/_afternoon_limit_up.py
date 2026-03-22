"""
午盘打板选手（AfternoonLimitUpAgent）
========================================
策略逻辑
--------
模拟午盘阶段（13:00 后）首次触板的涨停标的买入（无仓位限制）。

命中条件：
  1. 当日 high >= 涨停价 * 0.999（日线候选池）
  2. 非 ST / *ST 股票
  3. 排除北交所
  4. 一字板（open >= 涨停价 * 0.999）：必须曾经开板且出现回封；
     回封时间需 >= 13:00 归入午盘；纯一字板全天未开板 → 跳过
  5. 非一字板：首次触板时间（分钟线 high >= 涨停价 * 0.999）需 >= 13:00

买入价：涨停价

与早盘打板的区别
----------------
early = "开盘强势、情绪好" 标的  ↔  afternoon = "尾盘资金发动" 标的
两者统计对比能揭示市场情绪的强弱节奏（早盘/午盘哪个更容易赚钱）。
"""
import concurrent.futures
from typing import List, Dict

import pandas as pd

from agent_stats.agent_base import BaseAgent
from agent_stats.agents._limit_up_helpers import get_min_df, get_first_limit_time, check_reopen, TOL
from data.data_cleaner import TushareRateLimitAbort
from utils.common_tools import calc_limit_up_price
from utils.log_utils import logger

AFTERNOON_START_H = 13


class AfternoonLimitUpAgent(BaseAgent):
    agent_id   = "afternoon_limit_up"
    agent_name = "午盘打板选手"
    agent_desc = (
        "午盘涨停打板策略：当日 high 触板（≥涨停价×0.999），非ST/非北交所；"
        "一字板须曾开板且回封；首次触板/回封时间 ≥ 13:00 为午盘命中；买入价为涨停价。"
    )

    def get_signal_stock_pool(
        self,
        trade_date: str,
        daily_data: pd.DataFrame,
        context: Dict,
    ) -> List[Dict]:
        self.reset_minute_fetch_state()
        st_set = set(context.get("st_stock_list", []))

        # ── 前收价映射 ────────────────────────────────────────────────────
        pre_close_map: Dict[str, float] = {}
        pre_data = context.get("pre_close_data", pd.DataFrame())
        if not pre_data.empty and "ts_code" in pre_data.columns and "close" in pre_data.columns:
            pre_close_map = dict(zip(pre_data["ts_code"], pre_data["close"]))
        if "pre_close" in daily_data.columns:
            for _, row in daily_data.iterrows():
                if row["ts_code"] not in pre_close_map:
                    pre_close_map[row["ts_code"]] = row["pre_close"]

        # ── 候选池 ────────────────────────────────────────────────────────
        candidates = []
        for _, row in daily_data.iterrows():
            ts_code = row["ts_code"]
            if ts_code in st_set:
                continue
            code_prefix = ts_code.split(".")[0]
            if code_prefix.startswith(("83", "87", "88")) or ts_code.endswith(".BJ"):
                continue
            pre_close = pre_close_map.get(ts_code, 0.0)
            if pre_close <= 0:
                continue
            limit_price = calc_limit_up_price(ts_code, pre_close)
            if limit_price <= 0:
                continue
            high = float(row.get("high", 0.0) if hasattr(row, "get") else 0.0)
            if high >= limit_price * TOL:
                candidates.append({
                    "ts_code":     ts_code,
                    "stock_name":  str(row.get("name", "") if hasattr(row, "get") else ""),
                    "buy_price":   limit_price,
                    "open":        float(row.get("open", 0.0) if hasattr(row, "get") else 0.0),
                    "limit_price": limit_price,
                })

        if not candidates:
            logger.info(f"[{self.agent_id}][{trade_date}] 无涨停候选，无信号")
            return []

        # ── 并发拉取分钟线 ────────────────────────────────────────────────
        ts_codes = [c["ts_code"] for c in candidates]
        min_data: Dict[str, pd.DataFrame] = {}
        fetch_failed: List[str] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(get_min_df, ts, trade_date): ts for ts in ts_codes}
            for future in concurrent.futures.as_completed(futures):
                ts = futures[future]
                try:
                    min_data[ts] = future.result()
                except TushareRateLimitAbort:
                    # 严重限流 / 当日配额耗尽，立即向上传播，终止当日处理
                    raise
                except Exception as e:
                    logger.warning(f"[{self.agent_id}][{trade_date}][{ts}] 分钟线永久失败（已重试）：{e}")
                    min_data[ts] = pd.DataFrame()
                    fetch_failed.append(ts)
        # 记录永久失败的股票，引擎会将此信息写入 DB
        self._minute_fetch_failures = fetch_failed
        if fetch_failed:
            logger.warning(
                f"[{self.agent_id}][{trade_date}] ⚠ 分钟线永久失败 {len(fetch_failed)}/{len(ts_codes)} 只"
                f"（均经过 {10} 次重试），这些候选将被跳过并记录至 DB："
                f" {fetch_failed[:5]}{'...' if len(fetch_failed) > 5 else ''}"
            )

        # ── 筛选午盘命中 ─────────────────────────────────────────────────
        result = {}
        for c in candidates:
            ts_code    = c["ts_code"]
            limit_price = c["limit_price"]
            min_df     = min_data.get(ts_code, pd.DataFrame())

            if min_df is None or min_df.empty:
                logger.debug(f"[{self.agent_id}][{trade_date}][{ts_code}] 无分钟线，跳过")
                continue

            is_one_price = c["open"] >= limit_price * TOL

            if is_one_price:
                touch_time = check_reopen(min_df, limit_price)
                if touch_time is None:
                    continue
            else:
                touch_time = get_first_limit_time(min_df, limit_price)
                if touch_time is None:
                    continue

            # 午盘判定：触板/回封时间 >= 13:00
            t = pd.Timestamp(touch_time)
            if t.hour < AFTERNOON_START_H:
                continue

            if ts_code not in result:
                result[ts_code] = {
                    "ts_code":    ts_code,
                    "stock_name": c["stock_name"],
                    "buy_price":  c["buy_price"],
                }

        final = list(result.values())
        logger.info(f"[{self.agent_id}][{trade_date}] 午盘涨停命中 {len(final)} 只（分钟线过滤）")
        return final
