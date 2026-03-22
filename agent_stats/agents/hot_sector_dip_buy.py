"""
热点板块低吸博弈（HotSectorDipBuyAgent）
=========================================
策略逻辑
--------
模拟「热点板块短线博弈者」的操作：

1. 选出 D 日 Top3 热点板块（沿用 SectorHeatFeature.select_top3_hot_sectors）
2. 每个板块内，按 5 日涨幅降序排序，取前 TOP_N 名 → 候选池（最多 TOP_N × 3 只）
3. 基础过滤：ST / 北交所 / 一字板（无法低吸）
4. 获取 D 日 09:30-10:30 的 5 分钟线：
   - 若任意 bar 的 low ≤ 当日开盘价 × (1 - DIP_PCT)（跌破开盘 3%），则触发低吸信号
   - 买入价 = round(daily_open × (1 - DIP_PCT), 2)，模拟恐慌坑位买入

退出：由短线引擎跟踪 T 日日内收益 + T+1 日隔日表现（开盘/收盘）。

设计意图
--------
核心问题：
  「热点板块中 5 日涨幅居前的个股，在早盘恐慌回调 3% 时，
    主观博弈低吸者的胜率和赔率如何？」
"""
from typing import List, Dict

import pandas as pd

from agent_stats.agent_base import BaseAgent
from data.data_cleaner import data_cleaner, TushareRateLimitAbort
from features.sector.sector_heat_feature import SectorHeatFeature
from utils.common_tools import (
    filter_st_stocks,
    get_stocks_in_sector,
    sort_by_recent_gain,
    calc_limit_up_price,
)
from utils.log_utils import logger

# ── 策略参数 ──────────────────────────────────────────────────────────────
DIP_PCT          = 0.03    # 触发低吸的开盘跌幅阈值（3%）
TOP_N            = 3       # 每个热点板块取 5 日涨幅前 N 名
RECENT_GAIN_DAYS = 5       # 涨幅排序天数
WINDOW_START     = "09:30" # 低吸监测窗口开始（5min bar 的 HH:MM）
WINDOW_END       = "10:30" # 低吸监测窗口结束（含）

_sector_heat = SectorHeatFeature()


class HotSectorDipBuyAgent(BaseAgent):
    agent_id   = "hot_sector_dip_buy"
    agent_name = "热点板块低吸博弈选手"
    agent_desc = (
        "热点板块低吸博弈：选 D 日 Top3 热点板块，每板块取 5 日涨幅前 3 名进候选池；"
        "09:30-10:30 内若价格触及开盘价 -3% 则模拟低吸买入，"
        "跟踪该博弈行为的 T 日日内及 T+1 日隔日表现。"
    )

    def get_signal_stock_pool(
        self,
        trade_date: str,
        daily_data: pd.DataFrame,
        context: Dict,
    ) -> List[Dict]:
        st_set: set = set(context.get("st_stock_list", []))

        # ── 日期格式统一 ──────────────────────────────────────────────────
        # sector_heat 需要 YYYY-MM-DD；data_cleaner 分钟线接口需要 YYYYMMDD
        if len(trade_date) == 8 and trade_date.isdigit():
            trade_date_dash = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}"
            trade_date_8    = trade_date
        else:
            trade_date_dash = trade_date
            trade_date_8    = trade_date.replace("-", "")

        # ── Step 1: Top3 热点板块 ─────────────────────────────────────────
        try:
            heat_result  = _sector_heat.select_top3_hot_sectors(trade_date_dash)
            top3_sectors: List[str] = heat_result.get("top3_sectors", [])
        except Exception as e:
            logger.error(f"[{self.agent_id}][{trade_date}] 板块热度计算失败: {e}")
            return []

        if not top3_sectors:
            logger.warning(f"[{self.agent_id}][{trade_date}] Top3 板块为空，无信号")
            return []

        logger.info(f"[{self.agent_id}][{trade_date}] Top3 热点板块: {top3_sectors}")

        # ── Step 2: 每板块取 5 日涨幅前 TOP_N 名 ──────────────────────────
        candidate_codes: List[str] = []

        for sector in top3_sectors:
            try:
                raw = get_stocks_in_sector(sector)
                if not raw:
                    logger.debug(f"[{self.agent_id}][{trade_date}][{sector}] 板块无股票，跳过")
                    continue

                sector_codes = [item["ts_code"] for item in raw]

                # 过滤 ST
                sector_codes = filter_st_stocks(sector_codes, trade_date_dash)
                if not sector_codes:
                    continue

                # 过滤北交所
                sector_codes = [
                    ts for ts in sector_codes
                    if not (ts.endswith(".BJ") or ts[:2] in ("83", "87", "88"))
                ]
                if not sector_codes:
                    continue

                # 只保留今日有日线数据的股票（已开盘交易）
                sector_daily = daily_data[daily_data["ts_code"].isin(sector_codes)].copy()
                if sector_daily.empty:
                    logger.debug(f"[{self.agent_id}][{trade_date}][{sector}] 日线数据为空")
                    continue

                # 5 日涨幅排序（sort_by_recent_gain 内部查 DB，仅需候选股两天数据）
                sector_daily = sort_by_recent_gain(
                    sector_daily, trade_date_dash, day_count=RECENT_GAIN_DAYS
                )
                if sector_daily.empty:
                    continue

                top_n_codes = sector_daily.head(TOP_N)["ts_code"].tolist()
                gain_col    = f"recent_{RECENT_GAIN_DAYS}d_gain"
                top_n_gains = (
                    sector_daily.head(TOP_N)[gain_col].tolist()
                    if gain_col in sector_daily.columns else []
                )
                logger.info(
                    f"[{self.agent_id}][{trade_date}][{sector}] "
                    f"5日涨幅前{TOP_N}: "
                    + " | ".join(
                        f"{c}({g:.1f}%)" for c, g in zip(top_n_codes, top_n_gains)
                    )
                )
                candidate_codes.extend(top_n_codes)

            except Exception as e:
                logger.warning(
                    f"[{self.agent_id}][{trade_date}][{sector}] 板块处理异常: {e}"
                )
                continue

        # 跨板块去重（保持第一次出现的顺序）
        seen: set = set()
        unique_candidates: List[str] = []
        for ts in candidate_codes:
            if ts not in seen:
                seen.add(ts)
                unique_candidates.append(ts)

        if not unique_candidates:
            logger.info(f"[{self.agent_id}][{trade_date}] 候选池为空，无信号")
            return []

        logger.info(
            f"[{self.agent_id}][{trade_date}] 候选池（去重后）共 {len(unique_candidates)} 只: "
            f"{unique_candidates}"
        )

        # ── Step 3: 从 daily_data 提取开盘价 / 前收价 / 名称，过滤一字板 ──
        daily_sub = daily_data[daily_data["ts_code"].isin(unique_candidates)]

        open_price_map:  Dict[str, float] = {}
        pre_close_map:   Dict[str, float] = {}
        name_map:        Dict[str, str]   = {}

        for _, row in daily_sub.iterrows():
            ts    = row["ts_code"]
            open_p = float(row.get("open", 0) or 0)
            if open_p <= 0:
                continue
            open_price_map[ts] = open_p
            pre_close_map[ts]  = float(row.get("pre_close", 0) or 0)
            name_map[ts]       = str(row.get("name", "") or "")

        filtered_candidates: List[str] = []
        for ts in unique_candidates:
            if ts not in open_price_map:
                logger.debug(f"[{self.agent_id}][{trade_date}][{ts}] 无开盘价数据，跳过")
                continue

            open_p    = open_price_map[ts]
            pre_close = pre_close_map.get(ts, 0)

            # 一字板判断：开盘即封死涨停，且窗口内最低价也贴近涨停价 → 无法低吸
            if pre_close > 0:
                limit_up = calc_limit_up_price(ts, pre_close)
                row_data = daily_sub[daily_sub["ts_code"] == ts]
                if not row_data.empty:
                    low_p = float(row_data.iloc[0].get("low", 0) or 0)
                    if (
                        limit_up > 0
                        and abs(open_p - limit_up) < 0.015
                        and abs(low_p  - limit_up) < 0.015
                    ):
                        logger.debug(
                            f"[{self.agent_id}][{trade_date}][{ts}] 一字板，无低吸空间，跳过"
                        )
                        continue

            filtered_candidates.append(ts)

        if not filtered_candidates:
            logger.info(f"[{self.agent_id}][{trade_date}] 一字板过滤后候选池为空")
            return []

        # ── Step 4: 分钟线低吸信号检测（09:30-10:30 窗口内 low ≤ open×0.97）─
        result: List[Dict] = []

        for ts in filtered_candidates:
            open_price = open_price_map[ts]
            dip_price  = round(open_price * (1 - DIP_PCT), 2)

            # 拉取 D 日分钟线（DB→API→入库 完整链路）
            try:
                min_df = data_cleaner.get_kline_min_by_stock_date(ts, trade_date_8)
            except TushareRateLimitAbort:
                raise   # 必须向上传播，引擎层处理限流
            except Exception as e:
                logger.warning(
                    f"[{self.agent_id}][{trade_date}][{ts}] 分钟线获取失败，跳过: {e}"
                )
                self._minute_fetch_failures.append(ts)
                continue

            if min_df is None or min_df.empty:
                logger.debug(f"[{self.agent_id}][{trade_date}][{ts}] 分钟线为空，跳过")
                continue

            # 解析时间，截取 09:30-10:30 窗口
            try:
                min_df = min_df.copy()
                min_df["_hm"] = pd.to_datetime(min_df["trade_time"]).dt.strftime("%H:%M")
                window = min_df[
                    (min_df["_hm"] >= WINDOW_START) & (min_df["_hm"] <= WINDOW_END)
                ]
            except Exception as e:
                logger.warning(
                    f"[{self.agent_id}][{trade_date}][{ts}] 分钟线时间解析失败: {e}"
                )
                continue

            if window.empty:
                logger.debug(
                    f"[{self.agent_id}][{trade_date}][{ts}] "
                    f"{WINDOW_START}-{WINDOW_END} 无分钟数据，跳过"
                )
                continue

            window_low = float(window["low"].min())

            if window_low <= dip_price:
                logger.info(
                    f"[{self.agent_id}][{trade_date}][{ts}] {name_map.get(ts, '')} "
                    f"✓ 触发低吸: open={open_price:.2f} "
                    f"dip_price={dip_price:.2f} "
                    f"window_low={window_low:.2f}"
                )
                result.append({
                    "ts_code":    ts,
                    "stock_name": name_map.get(ts, ""),
                    "buy_price":  dip_price,
                })
            else:
                logger.debug(
                    f"[{self.agent_id}][{trade_date}][{ts}] "
                    f"未触发: open={open_price:.2f} "
                    f"dip_target={dip_price:.2f} window_low={window_low:.2f}"
                )

        logger.info(
            f"[{self.agent_id}][{trade_date}] 热点板块低吸命中 {len(result)} 只 "
            f"（候选={len(filtered_candidates)} 只，触发 dip={len(result)} 只）"
        )
        return result
