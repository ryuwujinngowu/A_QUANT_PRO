"""
板块热度Rank3-5开盘平铺买入（LongSectorRank3to5OpenBuyAgent）
=============================================================
策略逻辑（时序：D-1 日选股，D 日买入，固定持有 10 交易日）
--------
模拟「热点板块第2梯队（rank3~5）头部股票跟随买入」的统计表现：

1. D-1 日 limit_cpt_list 中 rank=3、4、5 的 3 个板块
2. 每个板块取近 5 日涨幅前 3 名（基于 D-1 收盘，无未来函数）
3. 共 9 只候选股，以 D 日开盘价平铺买入

一字板处理（D 日开盘时判断）：
  - 全天一字板（open ≈ high ≈ close ≈ low ≈ 涨停价）：无法买入，
    在该板块中顺延取下一名（rank4 → 取第4名，以此类推）
  - 一字板开过板（open ≈ close ≈ 涨停，low < 涨停）：以涨停价买入

持仓管理：
  - 固定持有 10 个交易日后强制卖出（收盘价）
  - 不设止损、不判 MA，单纯统计持有期收益

核心观察问题：
  「每个 D 日，做这批买入的交易者，在 10 日持股周期内的平均/中位数收益是多少？」
  → 引擎自动汇总：当一个 D 日的全部持仓平仓后，long_median_return / long_avg_return
    等字段回写至 agent_daily_profit_stats，即反映该 D 日的组合表现。
"""
from typing import Dict, List, Optional

import pandas as pd

from agent_stats.long_agent_base import BaseLongAgent
from agent_stats.agents._position_stock_helpers import is_yizi_limit_up
from utils.common_tools import (
    calc_limit_up_price,
    get_daily_kline_data,
    get_limit_cpt_list,
    get_stocks_in_sector,
    filter_st_stocks,
    sort_by_recent_gain,
)
from utils.log_utils import logger

# ── 策略参数 ──────────────────────────────────────────────────────────────
MAX_HOLD_DAYS    = 10     # 固定持仓交易日数
SECTOR_RANK_MIN  = 3      # 板块热度 rank 起始（含）
SECTOR_RANK_MAX  = 5      # 板块热度 rank 结束（含）
STOCKS_PER_SECTOR = 3     # 每板块取前 N 名
GAIN_DAYS        = 5      # 5 日涨幅排序
FETCH_EXTRA      = 3      # 每板块多拉几名以应对一字板顺延（取前 N+3 名再过滤）
_LIMIT_UP_TOL    = 0.01   # 涨停价容差


class LongSectorRank3to5OpenBuyAgent(BaseLongAgent):
    agent_id   = "long_sector_rank3to5_open_buy"
    agent_name = "板块热度Rank3-5开盘平铺（中长线10日）"
    agent_desc = (
        "D-1日 limit_cpt_list rank3~5 的3个板块，每板块取5日涨幅前3名；"
        "D日开盘平铺买入（全天一字板顺延，开过板以涨停价买）；"
        "固定持有10交易日后强制卖出，无止损/MA判断，纯统计2梯队板块跟随者的10日持有收益。"
    )

    def get_signal_stock_pool(
        self,
        trade_date: str,
        daily_data: pd.DataFrame,
        context: Dict,
    ) -> List[Dict]:
        st_set = set(context.get("st_stock_list", []))
        trade_dates: List[str] = context.get("trade_dates", [])

        # ── 日期格式 ─────────────────────────────────────────────────────
        if len(trade_date) == 8 and trade_date.isdigit():
            trade_date_dash = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}"
        else:
            trade_date_dash = trade_date

        # ── 获取 D-1 日 ──────────────────────────────────────────────────
        if trade_date_dash not in trade_dates:
            logger.warning(f"[{self.agent_id}][{trade_date}] trade_date 不在 trade_dates 中")
            return []
        idx = trade_dates.index(trade_date_dash)
        if idx == 0:
            logger.info(f"[{self.agent_id}][{trade_date}] 无 D-1 交易日，跳过")
            return []
        prev_date = trade_dates[idx - 1]

        # ── Step 1: 查 D-1 limit_cpt_list，取 rank 3/4/5 的板块 ─────────
        cpt_df = get_limit_cpt_list(prev_date)
        if cpt_df is None or cpt_df.empty:
            logger.info(f"[{self.agent_id}][{trade_date}] D-1={prev_date} 无 limit_cpt_list 数据")
            return []
        if "rank" not in cpt_df.columns or "name" not in cpt_df.columns:
            logger.warning(f"[{self.agent_id}][{trade_date}] limit_cpt_list 缺少 rank/name 列")
            return []

        cpt_df["rank"] = pd.to_numeric(cpt_df["rank"], errors="coerce")
        target_sectors = (
            cpt_df[
                (cpt_df["rank"] >= SECTOR_RANK_MIN) & (cpt_df["rank"] <= SECTOR_RANK_MAX)
            ]
            .sort_values("rank")["name"]
            .str.strip()
            .tolist()
        )

        if not target_sectors:
            logger.info(f"[{self.agent_id}][{trade_date}] D-1={prev_date} 无 rank{SECTOR_RANK_MIN}-{SECTOR_RANK_MAX} 板块")
            return []

        logger.info(f"[{self.agent_id}][{trade_date}] 目标板块（D-1={prev_date}）: {target_sectors}")

        # ── Step 2: 每板块取前 N 名（5日涨幅，基于 D-1，无未来函数）────────
        # D 日 open/pre_close map（买入价和涨停价计算用）
        d_open_map: Dict[str, float] = {}
        d_pre_close_map: Dict[str, float] = {}
        d_name_map: Dict[str, str] = {}
        for _, row in daily_data.iterrows():
            ts = row["ts_code"]
            open_p = float(row.get("open", 0) or 0)
            pre_close = float(row.get("pre_close", 0) or 0)
            if open_p > 0:
                d_open_map[ts] = open_p
            if pre_close > 0:
                d_pre_close_map[ts] = pre_close
            d_name_map[ts] = str(row.get("name", "") or "")

        result_map: Dict[str, Dict] = {}  # 去重

        for sector in target_sectors:
            sector_stocks_raw = get_stocks_in_sector(sector)
            if not sector_stocks_raw:
                logger.debug(f"[{self.agent_id}][{trade_date}][{sector}] 板块无股票")
                continue

            ts_codes = [item["ts_code"] for item in sector_stocks_raw]

            # 过滤 ST / 北交所
            ts_codes = filter_st_stocks(ts_codes, trade_date_dash)
            ts_codes = [
                ts for ts in ts_codes
                if not (ts.endswith(".BJ") or ts[:2] in ("83", "87", "88"))
            ]
            if not ts_codes:
                continue

            # 只保留 D 日有开盘数据的股票
            sector_daily = daily_data[daily_data["ts_code"].isin(ts_codes)].copy()
            if sector_daily.empty:
                continue

            # 5 日涨幅排序（用 D-1 收盘价，无未来函数）
            ranked = sort_by_recent_gain(sector_daily, prev_date, day_count=GAIN_DAYS)
            if ranked.empty:
                continue

            # 取前 N+FETCH_EXTRA 名，然后按一字板规则筛到 STOCKS_PER_SECTOR 只
            candidates = ranked.head(STOCKS_PER_SECTOR + FETCH_EXTRA)["ts_code"].tolist()
            picked = 0

            for ts_code in candidates:
                if picked >= STOCKS_PER_SECTOR:
                    break
                if ts_code in result_map:
                    continue  # 跨板块去重

                open_p = d_open_map.get(ts_code, 0)
                if open_p <= 0:
                    continue

                pre_close = d_pre_close_map.get(ts_code, 0)
                limit_up = calc_limit_up_price(ts_code, pre_close) if pre_close > 0 else 0.0

                # 取 D 日 row 数据用于一字板判断
                d_row_series = daily_data[daily_data["ts_code"] == ts_code]
                if d_row_series.empty:
                    continue
                d_row = d_row_series.iloc[0]

                # 一字板检测（用公用方法 is_yizi_limit_up）
                if limit_up > 0 and is_yizi_limit_up(d_row, limit_up):
                    low_p = float(d_row.get("low", 0) or 0)
                    if low_p >= limit_up - _LIMIT_UP_TOL:
                        # 全天一字板，无法买入 → 顺延到下一名
                        logger.debug(
                            f"[{self.agent_id}][{trade_date}][{sector}][{ts_code}] "
                            f"全天一字板，顺延"
                        )
                        continue
                    else:
                        # 开过板（low < 涨停价）→ 以涨停价买入
                        buy_price = round(limit_up, 2)
                        logger.debug(
                            f"[{self.agent_id}][{trade_date}][{sector}][{ts_code}] "
                            f"开过板，涨停价买入 {buy_price}"
                        )
                else:
                    buy_price = round(open_p, 2)

                result_map[ts_code] = {
                    "ts_code":    ts_code,
                    "stock_name": d_name_map.get(ts_code, ""),
                    "buy_price":  buy_price,
                }
                picked += 1

            logger.info(
                f"[{self.agent_id}][{trade_date}][{sector}] 命中 {picked} 只"
            )

        final = list(result_map.values())
        logger.info(
            f"[{self.agent_id}][{trade_date}] 合计 {len(final)} 只"
            f"（目标板块={target_sectors}）"
        )
        return final

    # ------------------------------------------------------------------ #
    # 卖出信号：固定持有 10 交易日后强制卖出，无止损/MA
    # ------------------------------------------------------------------ #

    def check_sell_signal(
        self,
        position: Dict,
        today_row: Optional[Dict],
        context: Dict,
    ) -> bool:
        """
        持有满 MAX_HOLD_DAYS 交易日后强制卖出（收盘价）。
        不设止损、不判 MA，纯统计固定持有期收益。
        """
        if today_row is None:
            return False

        trading_days_held = int(position.get("trading_days_so_far", 0))
        if trading_days_held >= MAX_HOLD_DAYS:
            ts_code = position.get("ts_code", "")
            position["_sell_reason"] = f"超期({trading_days_held}日≥{MAX_HOLD_DAYS}日)"
            logger.debug(
                f"[{self.agent_id}][{ts_code}] 满{MAX_HOLD_DAYS}日，强制卖出"
            )
            return True
        return False
