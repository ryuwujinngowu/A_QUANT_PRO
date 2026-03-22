"""
突破买入（LongBreakoutBuyAgent）
==================================

买入信号（get_signal_stock_pool）
---------------------------------
  后复权收盘价突破近 120 日最高价，且成交量放大（> 20 日均量 * 1.5）。
  使用后复权（HFQ）价格计算：历史价格不变，消除未来函数。
  过滤：非 ST、非北交所、有收盘价、近 30 日涨幅不超过 40%（排除连板/暴涨股）、
  近 30 日涨跌停次数不超过 3 次（排除短线情绪股/跟风股）。

卖出信号（check_sell_signal）
------------------------------
  触发任一条件即卖出（今日收盘）：
    1. 止损：后复权浮亏 ≤ -8%（恐慌模式下弹性调整）
    2. 超期：持仓超过 60 交易日
    3. MA 破位：后复权收盘价跌破 MA30 或 MA60
       · 除权除息日跳过 MA 破位检查
    4. 市场恐慌弹性止损：
       · 止损触发时检查市场状态（跌停 > 20 且指数急跌，或跌停 > 涨停）
       · 恐慌模式：止损线下移 1 倍，进入 5 日观察期
       · 观察期恢复/超限退出机制

为什么不用前复权（QFQ）？
  QFQ 以最新交易日为基准回溯调整所有历史价格。每当发生除权除息，
  历史 QFQ 价格全部被重算——未来函数。
  HFQ 从上市日往后调整，历史价格永不改变，回测结果与实盘一致。
"""

from typing import Dict, List, Optional

import pandas as pd

from agent_stats.long_agent_base import BaseLongAgent
from data.data_cleaner import data_cleaner
from features.ma_indicator import technical_features
from utils.common_tools import (
    analyze_stock_follower_strength,
    get_hfq_kline_range,
    get_kline_day_range,
    get_market_limit_counts,
    get_index_pct_chg,
    get_stock_limit_counts_batch,
)
from utils.log_utils import logger


class LongBreakoutBuyAgent(BaseLongAgent):

    agent_id   = "long_breakout_buy"
    agent_name = "突破买入（中长线）"
    agent_desc = (
        "后复权收盘价突破近120日最高价 + 量能放大，排除30日涨幅>40%的短期暴涨股"
        "及30日内涨跌停>3次的情绪股；"
        "止损 -8%（恐慌模式弹性调整）/ MA30/60 破位卖出 / 最长持仓 60 日（全链路后复权）。"
    )

    # ── 策略参数 ──────────────────────────────────────────────────────────
    BREAKOUT_DAYS     = 120      # 突破近 N 日最高价（HFQ）
    VOL_MA_DAYS       = 20       # 成交量均线周期
    VOL_AMPLIFY_RATIO = 1.5      # 当日量 > N 日均量 * ratio 才算放量
    STOP_LOSS_PCT     = -8.0     # 止损线（%，HFQ 浮亏相对买入 HFQ 价）
    MAX_HOLD_DAYS     = 60       # 最长持仓交易日数
    MA_PERIODS        = [30]  # 跌破任一 HFQ 均线即触发卖出

    # ── 候选池过滤参数 ────────────────────────────────────────────────────
    PRE_GAIN_DAYS     = 30       # 突破前 N 日回看窗口
    PRE_GAIN_MAX_PCT  = 65.0     # 前 N 日涨幅上限（%），超过视为连板/暴涨股，排除
    LIMIT_COUNT_DAYS  = 30       # 前 N 日涨跌停次数回看窗口
    LIMIT_COUNT_MAX   = 4       # 前 N 日涨停+跌停总次数上限，超过排除（短线情绪股）

    # ── 市场恐慌弹性止损参数 ──────────────────────────────────────────────
    PANIC_LIMIT_DOWN_THRESHOLD = 20    # 跌停数 > 此值且指数急跌 → 恐慌
    PANIC_INDEX_DROP_THRESHOLD = -1.0  # 上证跌幅 < 此值（%）→ 恐慌条件之一
    PANIC_STOP_LOSS_MULTIPLIER = 2.0   # 恐慌时止损线倍数（原 -8% → -16%）
    PANIC_MA_SHIFT_PCT         = 15.0  # 恐慌时 MA 下移百分比
    PANIC_OBSERVE_DAYS         = 5     # 恐慌观察期（交易日）
    PANIC_FORCE_EXIT_MULT      = 2.5   # 超过此倍数止损线强制退出（-8% * 2.5 = -20%）

    # ------------------------------------------------------------------ #
    # 买入信号
    # ------------------------------------------------------------------ #

    def get_signal_stock_pool(
        self,
        trade_date: str,
        daily_data: pd.DataFrame,
        context: Dict,
    ) -> List[Dict]:
        """
        突破买入选股（后复权价格，无未来函数）。

        两阶段优化：
          Phase 1: 用已有的 kline_day（不复权）做粗筛，将 ~4900 只缩减到 ~100-200 只
          Phase 2: 仅对粗筛通过的股票按需拉取 HFQ 数据，做精确突破判断

        :param trade_date: T 日（已收盘完整交易日），格式 YYYY-MM-DD
        :param daily_data: T 日全市场原始日线 DataFrame（用于候选股过滤）
        :param context:    {st_stock_list, trade_dates, pre_close_data}
        :return: 信号列表 [{ts_code, stock_name, buy_price}, ...]
        """
        if daily_data.empty:
            logger.debug(f"[{self.agent_id}][{trade_date}] 【1/5】输入daily_data为空，直接返回")
            return []

        st_set       = set(context.get("st_stock_list", []))
        trade_dates  = context.get("trade_dates", [])   # YYYY-MM-DD 列表

        # 确定回看起始日期
        try:
            cur_idx        = trade_dates.index(trade_date)
            lookback_idx   = max(0, cur_idx - self.BREAKOUT_DAYS)
            lookback_start = trade_dates[lookback_idx]
        except (ValueError, IndexError):
            logger.warning(f"[{self.agent_id}][{trade_date}] 无法确定回看起始日期，跳过")
            return []

        # 防御性清洗：去除 ts_code 前后空白
        daily_data = daily_data.copy()
        daily_data["ts_code"] = daily_data["ts_code"].astype(str).str.strip()

        total_count = len(daily_data)
        # 逐步过滤并记录每步过滤数量
        mask_st = ~daily_data["ts_code"].isin(st_set)
        mask_bj = ~daily_data["ts_code"].str.endswith(".BJ")
        mask_688 = ~daily_data["ts_code"].str.startswith("688")
        mask_close = daily_data["close"].notna() & (daily_data["close"] > 0)

        st_filtered = total_count - mask_st.sum()
        bj_filtered = (~mask_bj).sum()
        star_filtered = (~mask_688).sum()
        close_filtered = (~mask_close).sum()

        candidate = daily_data[mask_st & mask_bj & mask_688 & mask_close].copy()

        logger.info(
            f"[{self.agent_id}][{trade_date}] 【1/5】基础候选池过滤：{total_count}只 → {len(candidate)}只 | "
            f"ST过滤:{st_filtered}只, 北交所(.BJ)过滤:{bj_filtered}只, "
            f"科创板(688)过滤:{star_filtered}只, 无收盘价过滤:{close_filtered}只"
        )
        # 抽样打印被过滤的 BJ/688 股票，便于确认过滤生效
        if bj_filtered > 0 or star_filtered > 0:
            bj_samples = daily_data[~mask_bj]["ts_code"].head(5).tolist()
            star_samples = daily_data[~mask_688]["ts_code"].head(5).tolist()
            logger.debug(
                f"[{self.agent_id}][{trade_date}] BJ样本(前5):{bj_samples} | 688样本(前5):{star_samples}"
            )
        if candidate.empty:
            return []

        ts_codes = candidate["ts_code"].tolist()
        trade_date_fmt = trade_date.replace("-", "")

        # ── Phase 1: 用 kline_day（不复权）粗筛，大幅减少 HFQ API 调用量 ──
        # 不复权价格与 HFQ 仅在除权除息日有差异，用 5% 容差覆盖绝大多数场景
        raw_range = get_kline_day_range(ts_codes, lookback_start, trade_date)
        rough_candidates = set()
        # 初始化粗筛过滤计数器
        p1_filter_data = 0
        p1_filter_breakout = 0
        p1_filter_vol = 0

        if not raw_range.empty:
            # DB 存储的 trade_date 是 YYYY-MM-DD 字符串（data_cleaner 统一格式），
            # 统一去掉分隔符转为 YYYYMMDD，与 trade_date_fmt 保持一致后再比较
            raw_range["trade_date"] = raw_range["trade_date"].astype(str).str.replace("-", "")
            for ts_code, group in raw_range.groupby("ts_code"):
                group = group.sort_values("trade_date")
                today_row = group[group["trade_date"] == trade_date_fmt]
                if today_row.empty:
                    p1_filter_data += 1
                    continue
                today_close  = float(today_row.iloc[0]["close"])
                today_volume = float(today_row.iloc[0].get("volume", 0) or 0)
                hist = group[group["trade_date"] < trade_date_fmt]
                if len(hist) < self.BREAKOUT_DAYS // 2:
                    p1_filter_data += 1
                    continue
                hist_high = float(hist["high"].max())
                # 粗筛：不复权收盘 > 不复权最高 * 0.95（5% 容差兼容除权调整差异）
                if today_close <= hist_high * 0.95:
                    p1_filter_breakout += 1
                    continue
                # 粗筛量能
                hist_vol_ma = float(hist["volume"].tail(self.VOL_MA_DAYS).mean()) if len(hist) >= self.VOL_MA_DAYS else 0.0
                if hist_vol_ma > 0 and today_volume < hist_vol_ma * self.VOL_AMPLIFY_RATIO * 0.8:
                    p1_filter_vol += 1
                    continue
                rough_candidates.add(ts_code)

        # ====================== 2. Phase1 粗筛过滤 ======================
        logger.info(
            f"[{self.agent_id}][{trade_date}] 【2/5】Phase1粗筛：{len(ts_codes)}只 → {len(rough_candidates)}只 | "
            f"数据不足:{p1_filter_data}只, 未突破:{p1_filter_breakout}只, 量能不足:{p1_filter_vol}只"
        )
        if rough_candidates:
            logger.debug(f"[{self.agent_id}][{trade_date}] Phase1通过列表: {sorted(rough_candidates)}")

        if not rough_candidates:
            logger.debug(f"[{self.agent_id}][{trade_date}] Phase1粗筛无候选，终止筛选")
            return []

        logger.info(
            f"[{self.agent_id}][{trade_date}] 粗筛通过 {len(rough_candidates)} 只"
            f"（原始候选 {len(ts_codes)} 只），开始按需拉取 HFQ"
        )

        # ── Phase 2: 仅对粗筛通过的股票按需拉取 HFQ 并做精确判断 ──
        rough_list = list(rough_candidates)
        start_fmt = lookback_start.replace("-", "")
        end_fmt   = trade_date_fmt

        # # 确保 limit_list_ths 有当日数据（无则 API 补拉）
        # ensure_limit_list_ths_data(trade_date)

        # 批量查询涨跌停次数（一次 DB 查询，从 limit_list_ths 表取精确数据）
        # 回看窗口：trade_date 前 LIMIT_COUNT_DAYS 个交易日
        try:
            limit_start_idx = max(0, cur_idx - self.LIMIT_COUNT_DAYS)
            limit_start = trade_dates[limit_start_idx]
        except (IndexError, ValueError):
            limit_start = lookback_start
        limit_count_map = get_stock_limit_counts_batch(rough_list, limit_start, trade_date)
        try:
            data_cleaner.clean_and_insert_kline_day_hfq(rough_list, start_date=start_fmt, end_date=end_fmt)
        except Exception as e:
            logger.warning(f"[{self.agent_id}][{trade_date}] HFQ 按需入库异常：{e}")

        hfq_range = get_hfq_kline_range(rough_list, lookback_start, trade_date)
        if hfq_range.empty:
            logger.warning(f"[{self.agent_id}][{trade_date}] HFQ 回看区间为空，跳过")
            return []

        # 同样统一 trade_date 格式为 YYYYMMDD
        hfq_range["trade_date"] = hfq_range["trade_date"].astype(str).str.replace("-", "")
        # 初始化精筛过滤计数器
        p2_filter_data = 0
        p2_filter_breakout = 0
        p2_filter_vol = 0
        p2_filter_gain = 0
        p2_filter_limit = 0
        p2_filter_follower = 0
        # 跟踪每只被过滤的股票及原因（debug 级别输出）
        p2_filtered_details = []
        signals = []
        # 记录通过基础筛的股票，后续统一做跟风判断
        pre_follower_candidates = []

        for ts_code, group in hfq_range.groupby("ts_code"):
            group = group.sort_values("trade_date")

            today_row = group[group["trade_date"] == trade_date_fmt]
            if today_row.empty:
                p2_filter_data += 1
                p2_filtered_details.append(f"{ts_code}:无今日HFQ数据")
                continue
            today_close  = float(today_row.iloc[0]["close"])
            today_volume = float(today_row.iloc[0]["volume"] or 0)

            hist = group[group["trade_date"] < trade_date_fmt]
            if len(hist) < self.BREAKOUT_DAYS // 2:
                p2_filter_data += 1
                p2_filtered_details.append(f"{ts_code}:历史数据不足({len(hist)}行<{self.BREAKOUT_DAYS // 2})")
                continue

            hist_high   = float(hist["high"].max())
            hist_vol_ma = float(hist["volume"].tail(self.VOL_MA_DAYS).mean()) if len(hist) >= self.VOL_MA_DAYS else 0.0

            # 精确突破条件：HFQ 收盘价 > 近 N 日 HFQ 最高价
            if today_close <= hist_high:
                p2_filter_breakout += 1
                p2_filtered_details.append(
                    f"{ts_code}:未突破(HFQ close={today_close:.2f}<=high={hist_high:.2f})"
                )
                continue

            # 精确量能条件
            if hist_vol_ma > 0 and today_volume < hist_vol_ma * self.VOL_AMPLIFY_RATIO:
                p2_filter_vol += 1
                p2_filtered_details.append(
                    f"{ts_code}:量能不足(vol={today_volume:.0f}<ma*{self.VOL_AMPLIFY_RATIO}={hist_vol_ma * self.VOL_AMPLIFY_RATIO:.0f})"
                )
                continue

            # 30日涨幅过滤：排除连板股/短期暴涨股
            pre_gain_closes = hist["close"].astype(float).tolist()
            if len(pre_gain_closes) >= self.PRE_GAIN_DAYS:
                recent_close = pre_gain_closes[-1]
                base_close = pre_gain_closes[-self.PRE_GAIN_DAYS]
                if base_close > 0:
                    pre_gain_pct = (recent_close - base_close) / base_close * 100
                    if pre_gain_pct > self.PRE_GAIN_MAX_PCT:
                        p2_filter_gain += 1
                        p2_filtered_details.append(
                            f"{ts_code}:涨幅过高({pre_gain_pct:.1f}%>{self.PRE_GAIN_MAX_PCT}%)"
                        )
                        continue

            # 涨跌停次数过滤：从 limit_list_ths 精确判断（非 pct_chg 近似）
            limit_hit_count = limit_count_map.get(ts_code, 0)
            if limit_hit_count > self.LIMIT_COUNT_MAX:
                p2_filter_limit += 1
                p2_filtered_details.append(
                    f"{ts_code}:涨跌停过多({limit_hit_count}次>{self.LIMIT_COUNT_MAX}次)"
                )
                continue

            # 通过所有基础筛选，加入跟风待判列表
            pre_follower_candidates.append({
                "ts_code": ts_code,
                "today_close": today_close,
            })

        # ====================== 4. Phase2 基础筛结果 ======================
        logger.info(
            f"[{self.agent_id}][{trade_date}] 【3/5】Phase2基础筛：{len(rough_candidates)}只 → {len(pre_follower_candidates)}只 | "
            f"数据不足:{p2_filter_data}, 未突破:{p2_filter_breakout}, "
            f"量能不足:{p2_filter_vol}, 涨幅过高:{p2_filter_gain}, 涨跌停过多:{p2_filter_limit}"
        )
        if p2_filtered_details:
            logger.debug(
                f"[{self.agent_id}][{trade_date}] Phase2过滤明细: {p2_filtered_details}"
            )

        # ====================== 5. 跟风过滤（暂时跳过，已注释） ======================
        # analyze_stock_follower_strength 较慢，暂时跳过跟风判断，直接生成信号
        for item in pre_follower_candidates:
            ts_code = item["ts_code"]
            today_close = item["today_close"]

            # 买入价用原始收盘价（实际成交价），从 daily_data 取
            raw_row = candidate[candidate["ts_code"] == ts_code]
            buy_price = float(raw_row.iloc[0]["close"]) if not raw_row.empty else today_close

            signals.append({
                "ts_code":    ts_code,
                "stock_name": "",
                "buy_price":  round(buy_price, 4),
            })

        # ====================== 6. 最终信号 ======================
        logger.info(
            f"[{self.agent_id}][{trade_date}] 【4/5】跟风过滤：{len(pre_follower_candidates)}只 → {len(signals)}只 | "
            f"跟风排除:{p2_filter_follower}只"
        )
        if signals:
            signal_codes = [s["ts_code"] for s in signals]
            logger.info(
                f"[{self.agent_id}][{trade_date}] 【5/5】最终信号({len(signals)}只): {signal_codes}"
            )
        else:
            logger.info(f"[{self.agent_id}][{trade_date}] 【5/5】最终信号：无")
        return signals

    # ------------------------------------------------------------------ #
    # 卖出信号
    # ------------------------------------------------------------------ #

    def check_sell_signal(
        self,
        position: Dict,
        today_row: Optional[Dict],
        context: Dict,
    ) -> bool:
        """
        卖出信号：超期 / 止损（含恐慌弹性）/ MA 破位（基于后复权数据）。

        恐慌弹性止损机制：
          当正常止损触发时，检查市场是否处于恐慌状态：
          - 恐慌条件：(跌停>20 且 上证跌>2%) 或 (跌停>涨停)
          - 进入恐慌模式：止损线下移 1 倍（-8%→-16%），MA 下移 15%，观察 5 天
          - 观察期内：价格回升到恐慌日开盘价 → 恢复正常规则
          - 继续下跌超 2.5 倍原止损（-20%）→ 强制退出
          - 5 天到期未恢复 → 退出

        position 字段（恐慌模式动态添加）：
          panic_date:         恐慌触发日期
          panic_open_price:   恐慌日 HFQ 开盘价（用于判断恢复）
          panic_observe_count: 已观察天数

        :param position:  持仓记录
        :param today_row: 今日原始日线（停牌时为 None）
        :param context:   引擎上下文
        :return: True = 今日收盘卖出
        """
        if today_row is None:
            return False

        ts_code           = position["ts_code"]
        trading_days_held = int(position.get("trading_days_so_far", 0))
        trade_date        = context.get("trade_date", "")

        # ── 1. 超期 ───────────────────────────────────────────────────
        if trading_days_held >= self.MAX_HOLD_DAYS:
            position["_sell_reason"] = f"超期({trading_days_held}日≥{self.MAX_HOLD_DAYS}日)"
            logger.debug(
                f"[{self.agent_id}][{ts_code}] 超期触发 | "
                f"已持 {trading_days_held} 日 ≥ {self.MAX_HOLD_DAYS} 日"
            )
            return True

        if not trade_date:
            return False

        # ── 计算 HFQ 均线和当日 HFQ 收盘价 ──────────────────────────
        mas = self._compute_hfq_mas(ts_code, trade_date, context)
        hfq_close_today = mas.get("hfq_close_today", 0.0)
        hfq_buy_price = float(position.get("hfq_buy_price", 0))

        # ── 2. 止损（含恐慌弹性机制）─────────────────────────────────
        if hfq_buy_price > 0 and hfq_close_today > 0:
            float_pnl_pct = (hfq_close_today - hfq_buy_price) / hfq_buy_price * 100

            # 检查是否处于恐慌观察期
            panic_date = position.get("panic_date")

            if panic_date:
                # ── 恐慌观察期内 ──
                panic_observe_count = position.get("panic_observe_count", 0) + 1
                position["panic_observe_count"] = panic_observe_count
                panic_open_price = float(position.get("panic_open_price", 0))
                panic_stop_loss  = self.STOP_LOSS_PCT * self.PANIC_STOP_LOSS_MULTIPLIER
                force_exit_pct   = self.STOP_LOSS_PCT * self.PANIC_FORCE_EXIT_MULT

                # 条件 A：价格回升到恐慌日收盘价之上 → 恢复正常规则
                if panic_open_price > 0 and hfq_close_today > panic_open_price:
                    logger.info(
                        f"[{self.agent_id}][{ts_code}] 恐慌恢复 | {trade_date} | "
                        f"观察第{panic_observe_count}天 | "
                        f"HFQ收盘 {hfq_close_today:.2f} > 恐慌基准 {panic_open_price:.2f} | "
                        f"浮亏 {float_pnl_pct:.2f}% → 恢复正常规则"
                    )
                    position.pop("panic_date", None)
                    position.pop("panic_open_price", None)
                    position.pop("panic_observe_count", None)
                    # 恢复后继续走正常止损/MA判断（不return，往下走）

                # 条件 B：继续下跌超过极限 → 强制退出
                elif float_pnl_pct <= force_exit_pct:
                    position["_sell_reason"] = (
                        f"恐慌强制退出(浮亏{float_pnl_pct:.2f}%≤{force_exit_pct:.1f}%"
                        f",观察第{panic_observe_count}天)"
                    )
                    logger.info(
                        f"[{self.agent_id}][{ts_code}] 恐慌强制退出 | {trade_date} | "
                        f"观察第{panic_observe_count}/{self.PANIC_OBSERVE_DAYS}天 | "
                        f"浮亏 {float_pnl_pct:.2f}% ≤ 强制线 {force_exit_pct:.1f}%"
                    )
                    return True

                # 条件 C：观察期到期未恢复 → 退出
                elif panic_observe_count >= self.PANIC_OBSERVE_DAYS:
                    position["_sell_reason"] = (
                        f"恐慌观察期到期({panic_observe_count}天未恢复,浮亏{float_pnl_pct:.2f}%)"
                    )
                    logger.info(
                        f"[{self.agent_id}][{ts_code}] 恐慌观察期到期 | {trade_date} | "
                        f"观察满{panic_observe_count}天未恢复 | "
                        f"浮亏 {float_pnl_pct:.2f}%，退出"
                    )
                    return True
                else:
                    # 仍在观察期内，放宽止损线
                    if float_pnl_pct <= panic_stop_loss:
                        position["_sell_reason"] = (
                            f"恐慌期止损(浮亏{float_pnl_pct:.2f}%≤{panic_stop_loss:.1f}%"
                            f",观察第{panic_observe_count}天)"
                        )
                        logger.info(
                            f"[{self.agent_id}][{ts_code}] 恐慌期止损 | {trade_date} | "
                            f"观察第{panic_observe_count}/{self.PANIC_OBSERVE_DAYS}天 | "
                            f"浮亏 {float_pnl_pct:.2f}% ≤ 恐慌止损 {panic_stop_loss:.1f}%"
                        )
                        return True
                    # 每日输出观察状态（INFO 级）
                    logger.info(
                        f"[{self.agent_id}][{ts_code}] 恐慌观察中 | {trade_date} | "
                        f"第{panic_observe_count}/{self.PANIC_OBSERVE_DAYS}天 | "
                        f"浮亏 {float_pnl_pct:.2f}% | "
                        f"恐慌止损线 {panic_stop_loss:.1f}% | 强制线 {force_exit_pct:.1f}% | "
                        f"恢复基准(HFQ) {panic_open_price:.2f}"
                    )
                    # 恐慌期内不做 MA 破位检查
                    return False

            else:
                # ── 正常模式：止损触发时判断是否进入恐慌模式 ──
                if float_pnl_pct <= self.STOP_LOSS_PCT:
                    # 检查市场状态
                    if self._is_market_panic(trade_date):
                        # 获取当日 HFQ 开盘价
                        preloaded = context.get("_hfq_preloaded")
                        if preloaded:
                            td_fmt = trade_date.replace("-", "")
                            # 从 HFQ 日线取开盘价
                            hfq_open = 0.0
                            hfq_date_list = preloaded.get("hfq_date_list", [])
                            hfq_val_list = preloaded.get("hfq_val_list", [])
                            closes_by_date = preloaded.get("hfq_closes_by_date", {})
                            # 使用 close 近似 open（预加载数据中没有 open）
                            # 实际恐慌日开盘通常低开，用 close 作为恢复基准更保守
                            hfq_open = closes_by_date.get(td_fmt, 0.0)
                        else:
                            hfq_open = hfq_close_today

                        position["panic_date"] = trade_date
                        position["panic_open_price"] = hfq_open
                        position["panic_observe_count"] = 0
                        logger.info(
                            f"[{self.agent_id}][{ts_code}] 恐慌模式触发 | "
                            f"{trade_date} 浮亏 {float_pnl_pct:.2f}%，"
                            f"进入 {self.PANIC_OBSERVE_DAYS} 日观察期"
                        )
                        return False  # 进入观察期，本日不卖出

                    # 非恐慌 → 正常止损
                    position["_sell_reason"] = f"止损(HFQ浮亏{float_pnl_pct:.2f}%≤{self.STOP_LOSS_PCT}%)"
                    logger.debug(
                        f"[{self.agent_id}][{ts_code}] 止损触发（HFQ）| "
                        f"浮亏 {float_pnl_pct:.2f}% ≤ {self.STOP_LOSS_PCT}%"
                    )
                    return True

        # ── 3. MA 破位 ──────────────────────────────────────────────
        if not mas:
            return False

        ex_div_stocks = context.get("ex_div_stocks", set())
        if ts_code in ex_div_stocks:
            return False

        if hfq_close_today <= 0:
            return False

        for period in self.MA_PERIODS:
            ma_val = mas.get(period)
            if ma_val and hfq_close_today < ma_val:
                position["_sell_reason"] = f"MA{period}破位(HFQ收盘{hfq_close_today:.2f}<MA{period} {ma_val:.2f})"
                logger.debug(
                    f"[{self.agent_id}][{ts_code}] MA{period}破位（HFQ）| "
                    f"HFQ收盘 {hfq_close_today:.2f} < MA{period} {ma_val:.2f}"
                )
                return True

        return False

    def _is_market_panic(self, trade_date: str) -> bool:
        """
        判断当日市场是否处于恐慌状态。
        恐慌条件（满足任一）：
          1. 跌停数 > 20 且上证跌幅 < -2%
          2. 跌停数 > 涨停数
        """
        limit_counts = get_market_limit_counts(trade_date)
        limit_down = limit_counts.get("limit_down_count", 0)
        limit_up = limit_counts.get("limit_up_count", 0)

        idx_chg = get_index_pct_chg(trade_date)
        sh_pct_chg = idx_chg.get("000001.SH", 0.0)

        cond1 = limit_down > self.PANIC_LIMIT_DOWN_THRESHOLD and sh_pct_chg < self.PANIC_INDEX_DROP_THRESHOLD
        cond2 = limit_down > limit_up and limit_down > 10  # 至少 10 只跌停才有意义

        if cond1 or cond2:
            logger.info(
                f"[{self.agent_id}][{trade_date}] 市场恐慌 | "
                f"跌停:{limit_down} 涨停:{limit_up} 上证:{sh_pct_chg:.2f}%"
            )
            return True
        return False

    def _compute_hfq_mas(self, ts_code: str, trade_date: str, context: Dict) -> Dict:
        """
        计算指定股票在 trade_date 的各周期后复权 MA。

        优先使用引擎预加载的 HFQ 数据（context["_hfq_preloaded"]），
        在内存中计算 MA，零 DB 查询。
        如果没有预加载数据（如 daily 模式），降级为 DB 查询。

        返回 {30: ma30_val, 60: ma60_val, ..., 'hfq_close_today': float}；
        数据不足时返回空字典。
        """
        preloaded = context.get("_hfq_preloaded")
        if preloaded:
            return self._compute_hfq_mas_from_preloaded(trade_date, preloaded)

        # 降级：无预加载数据时走 DB 查询（daily 模式）
        trade_dates = context.get("trade_dates", [])
        return technical_features.compute_ma_from_hfq_range(
            ts_code, trade_date, trade_dates, self.MA_PERIODS
        )

    def _compute_hfq_mas_from_preloaded(self, trade_date: str, preloaded: Dict) -> Dict:
        """从引擎预加载的 HFQ 数据中计算 MA（纯内存计算，零 DB）。"""
        trade_date_fmt = trade_date.replace("-", "")
        hfq_date_list = preloaded["hfq_date_list"]
        hfq_val_list  = preloaded["hfq_val_list"]

        # 找到 trade_date 在 HFQ 序列中的位置
        try:
            idx = hfq_date_list.index(trade_date_fmt)
        except ValueError:
            return {}

        hfq_close_today = hfq_val_list[idx]
        result = {"hfq_close_today": hfq_close_today}

        for period in self.MA_PERIODS:
            start = max(0, idx - period + 1)
            window = hfq_val_list[start:idx + 1]
            if len(window) >= period:
                result[period] = round(sum(window) / period, 4)

        return result
