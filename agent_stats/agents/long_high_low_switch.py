"""
高低切轮动策略（LongHighLowSwitchAgent）
=========================================
策略逻辑
--------
高位股（近20日涨幅前1%龙头股）在持续拉升后会出现"上涨钝化"：
价格仍高于5日均线（乖离率>0），但乖离率持续收窄（动能衰减）。
此时市场资金倾向于轮换到低位启动的首板/二板/三板（"高低切"）。

触发机制（上涨钝化）
--------
1. 回看 HIGH_POS_LOOKBACK_N 个交易日（含今日），逐日计算高位股池
2. 找出在 ≥ HIGH_POS_OVERLAP_MIN 天中均出现的重合股（持续领涨龙头）
3. 对重合股，用 QFQ 收盘价计算各日5日MA乖离率，再对所有重合股取平均
4. 对平均乖离率序列做线性回归，斜率 < 0 且今日均值 > 0 → 上涨钝化成立

买入池（D日开盘买入）
--------
- 昨日首板：D-1 涨停 & cons_nums=1（即 D-2 非涨停）
  * 过滤：20日涨幅 ≤ 龙头均高 × FIRST_BOARD_GAIN_RATIO（35%）
  * 过滤：不在 mid_position_stock 选股池中（按20日涨幅排名近似）
- 昨日二板：D-1 cons_nums=2
- 昨日三板：D-1 cons_nums=3

通用过滤
--------
- 仅主板（排除创业板/科创板/北交所，即20cm标的）
- 排除 ST 股
- 排除北交所

买入价
------
- D日开盘价买入
- 若 D日一字板开盘（open≈close≈high≈涨停价）→ 涨停价排队买入

卖出条件
--------
- 止损：HFQ浮亏 ≤ -15%（强制卖出）
- 止盈：持满 10 个交易日（到期卖出）
"""
import gc
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from agent_stats.long_agent_base import BaseLongAgent
from agent_stats.agents._position_stock_helpers import (
    BASE_PCT, HIGH_PCT, MIN_HIGH, LOOKBACK_DAYS,
    build_pre_close_map,
)
from features.ma_indicator import technical_features
from utils.common_tools import (
    calc_limit_up_price,
    get_daily_kline_data,
    get_kline_day_range,
    get_limit_list_ths,
    get_limit_step,
)
from utils.log_utils import logger


# MID_PCT_OF_BASE 与 mid_position_stock.py 中定义的参数保持一致（基础池前50%为中位股范围）
MID_PCT_OF_BASE = 0.50


def _is_main_board(ts_code: str) -> bool:
    """
    判断是否主板（沪深主板，10cm限制），排除创业板/科创板/北交所（均为20cm+）。
    复用 common_tools.calc_limit_up_price 中的板块判断逻辑。
    """
    if ts_code.endswith(".BJ"):
        return False
    if ts_code.startswith(("300", "301", "302")) and ts_code.endswith(".SZ"):
        return False
    if ts_code.startswith("688"):
        return False
    return True


class LongHighLowSwitchAgent(BaseLongAgent):
    agent_id = "long_high_low_switch"
    agent_name = "高低切轮动（中长线）"
    agent_desc = (
        "高位股上涨钝化时买入昨日首板/二板/三板（高低切轮动）："
        "回看5日高位股池，找持续重合龙头，检测其5日MA乖离率是否连续下降（线性回归斜率<0）；"
        "触发后以D日开盘价买入昨日首/二/三板（仅主板，排除20cm）；"
        "止损-15%，拿满10个交易日止盈。"
    )

    # ── 上涨钝化参数 ─────────────────────────────────────────────────────────
    HIGH_POS_LOOKBACK_N  = 5   # 回看天数（含今日），同时也是乖离率趋势判断窗口
    HIGH_POS_OVERLAP_MIN = 3   # 重合阈值：股票在 ≥K 天的高位池中出现才算重合
    MA_BIAS_PERIOD       = 5   # 乖离率基准均线天数

    # ── 执行窗口参数 ─────────────────────────────────────────────────────────
    EXEC_WINDOW_DAYS = 5   # 钝化触发后的连续买入窗口天数
    # 新窗口要求：当前龙头与前窗口龙头零重合，才允许开新窗口

    # ── 买入过滤参数 ─────────────────────────────────────────────────────────
    FIRST_BOARD_GAIN_RATIO = 0.35  # 首板20日涨幅上限系数（≤龙头均高×35%）

    # ── 风控参数 ─────────────────────────────────────────────────────────────
    STOP_LOSS_PCT = -15.0   # 止损线（HFQ浮亏%）
    MAX_HOLD_DAYS = 10      # 最长持仓天数（到期止盈）

    def __init__(self):
        super().__init__()
        # 窗口状态（顺序回放时 instance 内存维护；full 模式正确，daily 模式重启会重置）
        self._window_end_idx:       Optional[int] = None  # 窗口最后一天的 trade_dates 索引
        self._window_leader_stocks: Optional[set] = None  # 开窗时的重合龙头股池
        self._prev_leader_stocks:   Optional[set] = None  # 上一个已关闭窗口的龙头（防止立即复开）

    # ────────────────────────────────────────────────────────────────────────
    # 买入信号
    # ────────────────────────────────────────────────────────────────────────
    def get_signal_stock_pool(
        self,
        trade_date: str,
        daily_data: pd.DataFrame,
        context: Dict,
    ) -> List[Dict]:
        if daily_data.empty:
            logger.debug(f"[{self.agent_id}][{trade_date}] daily_data为空，跳过")
            return []

        st_set = set(context.get("st_stock_list", []))
        trade_dates: List[str] = context.get("trade_dates", [])

        try:
            cur_idx = trade_dates.index(trade_date)
        except ValueError:
            logger.warning(f"[{self.agent_id}][{trade_date}] 不在trade_dates中，跳过")
            return []

        min_required = self.HIGH_POS_LOOKBACK_N + self.MA_BIAS_PERIOD + LOOKBACK_DAYS
        if cur_idx < min_required:
            logger.info(f"[{self.agent_id}][{trade_date}] 历史数据不足 {min_required} 日，跳过")
            return []

        # ── Step 1：今日高位池（每天必算，首板过滤参数依赖此处）──────────────
        # 使用 kline_day 原始数据（全量自动更新，无缺口），不用 build_gain_list（QFQ 可能稀疏）
        t21_date_today = trade_dates[cur_idx - LOOKBACK_DAYS]
        all_codes_today = daily_data["ts_code"].tolist()
        t21_df_today = get_kline_day_range(all_codes_today, t21_date_today, t21_date_today)
        t21_close_map_today = (
            dict(zip(t21_df_today["ts_code"], t21_df_today["close"].astype(float)))
            if not t21_df_today.empty else {}
        )
        gain_list_today = []
        for _, _row in daily_data.iterrows():
            _ts = _row["ts_code"]
            if _ts in st_set:
                continue
            if _ts.endswith(".BJ") or _ts.split(".")[0].startswith(("83", "87", "88")):
                continue
            _c_now = float(_row.get("close", 0) or 0)
            _c_t21 = t21_close_map_today.get(_ts, 0.0)
            if _c_now <= 0 or _c_t21 <= 0:
                continue
            gain_list_today.append((_ts, _row.get("name", ""), _c_now / _c_t21 - 1, _row))
        gain_list_today.sort(key=lambda x: -x[2])
        if not gain_list_today:
            logger.info(f"[{self.agent_id}][{trade_date}] 今日无有效gain_list，跳过")
            return []

        n_base_today  = max(MIN_HIGH, int(len(gain_list_today) * BASE_PCT))
        base_pool_today = gain_list_today[:n_base_today]
        n_high_today  = max(MIN_HIGH, int(len(base_pool_today) * HIGH_PCT))

        leader_gains    = [t[2] for t in base_pool_today[:n_high_today]]
        leader_avg_gain = sum(leader_gains) / len(leader_gains) if leader_gains else 0.0
        gain_map: Dict[str, float] = {t[0]: t[2] for t in gain_list_today}
        n_mid_upper  = int(len(base_pool_today) * MID_PCT_OF_BASE)
        mid_pool_set = set(t[0] for t in base_pool_today[:n_mid_upper])

        logger.debug(
            f"[{self.agent_id}][{trade_date}] 今日 有效股={len(gain_list_today)}，"
            f"基础池={n_base_today}，高位={n_high_today}，龙头均涨幅={leader_avg_gain:.1%}"
        )

        # ── 窗口状态机 ───────────────────────────────────────────────────────
        in_window = (
            self._window_end_idx is not None
            and cur_idx <= self._window_end_idx
        )

        # 窗口刚过期：轮转状态，前窗口龙头记录为 _prev
        if self._window_end_idx is not None and not in_window:
            logger.info(
                f"[{self.agent_id}][{trade_date}] 执行窗口已到期，"
                f"前窗口龙头={self._window_leader_stocks}，等待新钝化信号"
            )
            self._prev_leader_stocks   = self._window_leader_stocks
            self._window_end_idx       = None
            self._window_leader_stocks = None

        if in_window:
            days_in = cur_idx - (self._window_end_idx - self.EXEC_WINDOW_DAYS + 1) + 1
            logger.info(
                f"[{self.agent_id}][{trade_date}] 处于钝化执行窗口 "
                f"第{days_in}/{self.EXEC_WINDOW_DAYS}天，直接做板"
            )
        else:
            # ── Step 2：过去 N-1 天高位池（用 kline_day 原始数据，全市场覆盖）──
            # 不使用 build_gain_list（其 QFQ 分支可能只覆盖部分股票）
            # 直接从 kline_day 查两个时间点的原始收盘价计算20日涨幅，与 high_position_stock 对齐
            high_pool_today = set(t[0] for t in base_pool_today[:n_high_today])
            all_high_pools: Dict[str, set] = {trade_date: high_pool_today}

            for i in range(1, self.HIGH_POS_LOOKBACK_N):
                past_date = trade_dates[cur_idx - i]
                try:
                    past_daily = get_daily_kline_data(past_date)
                except Exception as e:
                    logger.warning(f"[{self.agent_id}] 获取{past_date}日线失败：{e}")
                    continue
                if past_daily is None or past_daily.empty:
                    continue
                try:
                    past_idx      = trade_dates.index(past_date)
                    t21_date_past = trade_dates[max(0, past_idx - LOOKBACK_DAYS)]
                    past_codes    = past_daily["ts_code"].tolist()
                    t21_df = get_kline_day_range(past_codes, t21_date_past, t21_date_past)
                    if t21_df.empty:
                        continue
                    t21_close_map = dict(zip(
                        t21_df["ts_code"], t21_df["close"].astype(float)
                    ))
                    past_gains = []
                    for _, row in past_daily.iterrows():
                        ts = row["ts_code"]
                        if ts in st_set:
                            continue
                        if ts.endswith(".BJ") or ts.split(".")[0].startswith(("83", "87", "88")):
                            continue
                        c_today = float(row.get("close", 0) or 0)
                        c_t21   = t21_close_map.get(ts, 0.0)
                        if c_today <= 0 or c_t21 <= 0:
                            continue
                        past_gains.append((ts, c_today / c_t21 - 1))
                    past_gains.sort(key=lambda x: -x[1])
                    n_base_p = max(MIN_HIGH, int(len(past_gains) * BASE_PCT))
                    n_high_p = max(MIN_HIGH, int(n_base_p * HIGH_PCT))
                    all_high_pools[past_date] = set(t[0] for t in past_gains[:n_high_p])
                except Exception as e:
                    logger.warning(f"[{self.agent_id}] {past_date} 高位池计算失败：{e}")
                    continue

            # ── Step 3：重合股 ────────────────────────────────────────────────
            all_stocks_flat = [s for pool in all_high_pools.values() for s in pool]
            occurrence      = Counter(all_stocks_flat)
            overlap_stocks  = [
                ts for ts, cnt in occurrence.items()
                if cnt >= self.HIGH_POS_OVERLAP_MIN
            ]
            overlap_with_cnt = sorted(
                [(ts, occurrence[ts]) for ts in overlap_stocks], key=lambda x: -x[1]
            )
            logger.info(
                f"[{self.agent_id}][{trade_date}] 高位池={len(all_high_pools)}天，"
                f"重合股(≥{self.HIGH_POS_OVERLAP_MIN}天)={len(overlap_stocks)}只：{overlap_with_cnt}"
            )
            if not overlap_stocks:
                logger.info(f"[{self.agent_id}][{trade_date}] 无持续高位重合股，不触发")
                return []

            # ── Step 4：乖离率趋势 ────────────────────────────────────────────
            bias_lookback_start = trade_dates[
                max(0, cur_idx - self.HIGH_POS_LOOKBACK_N - self.MA_BIAS_PERIOD - 2)
            ]
            # 乖离率计算用 kline_day 原始数据（全量自动更新，无缺口风险）
            kline_df = get_kline_day_range(overlap_stocks, bias_lookback_start, trade_date)
            if kline_df is None or kline_df.empty:
                logger.info(f"[{self.agent_id}][{trade_date}] 重合股 kline_day 数据为空，跳过")
                return []

            kline_df["trade_date"] = kline_df["trade_date"].astype(str).str.replace("-", "")
            kline_df = kline_df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
            window_dates = [
                trade_dates[cur_idx - i].replace("-", "")
                for i in range(self.HIGH_POS_LOOKBACK_N - 1, -1, -1)
            ]

            daily_avg_bias: List[Optional[float]] = []
            per_stock_bias: Dict[str, List[Optional[float]]] = {ts: [] for ts in overlap_stocks}
            for d_fmt in window_dates:
                biases = []
                for ts_code in overlap_stocks:
                    stock_df     = kline_df[kline_df["ts_code"] == ts_code].sort_values("trade_date")
                    stock_closes = stock_df["close"].astype(float).tolist()
                    dates_list   = stock_df["trade_date"].tolist()
                    if d_fmt not in dates_list:
                        per_stock_bias[ts_code].append(None)
                        continue
                    d_idx = dates_list.index(d_fmt)
                    if d_idx + 1 < self.MA_BIAS_PERIOD:
                        per_stock_bias[ts_code].append(None)
                        continue
                    bias_series = technical_features.compute_bias_rate_series(
                        stock_closes, ma_period=self.MA_BIAS_PERIOD
                    )
                    bias_val = bias_series[d_idx]
                    per_stock_bias[ts_code].append(bias_val)
                    if bias_val is not None:
                        biases.append(bias_val)
                daily_avg_bias.append(
                    round(sum(biases) / len(biases), 4) if biases else None
                )

            logger.info(
                f"[{self.agent_id}][{trade_date}] 重合股乖离率明细 | 窗口={window_dates}"
            )
            for ts_code, bias_list in per_stock_bias.items():
                formatted = [f"{v:.2f}%" if v is not None else "N/A" for v in bias_list]
                logger.info(f"[{self.agent_id}][{trade_date}]   {ts_code}: {formatted}")

            valid_bias = [(i, v) for i, v in enumerate(daily_avg_bias) if v is not None]
            if len(valid_bias) < 3:
                logger.info(f"[{self.agent_id}][{trade_date}] 有效乖离率不足3个，跳过")
                return []

            xs    = np.array([x for x, _ in valid_bias], dtype=float)
            ys    = np.array([y for _, y in valid_bias], dtype=float)
            slope = float(np.polyfit(xs, ys, 1)[0])
            latest_bias = valid_bias[-1][1]
            is_dulling  = slope < 0 and latest_bias > 0

            logger.info(
                f"[{self.agent_id}][{trade_date}] 乖离率趋势：slope={slope:.4f}，"
                f"今日均值={latest_bias:.2f}%，钝化={is_dulling} | "
                f"各日avg_bias={[v for _, v in valid_bias]}"
            )

            del kline_df
            gc.collect()

            if not is_dulling:
                logger.info(f"[{self.agent_id}][{trade_date}] 上涨钝化未触发")
                return []

            # ── 龙头换手检查：新窗口龙头与前窗口龙头必须零重合 ─────────────
            if self._prev_leader_stocks:
                conflict = set(overlap_stocks) & self._prev_leader_stocks
                if conflict:
                    logger.info(
                        f"[{self.agent_id}][{trade_date}] 龙头未换手，"
                        f"与前窗口重合={conflict}，等待龙头切换"
                    )
                    return []

            # ── 开启执行窗口 ──────────────────────────────────────────────────
            self._window_end_idx       = cur_idx + self.EXEC_WINDOW_DAYS - 1
            self._window_leader_stocks = set(overlap_stocks)
            logger.info(
                f"[{self.agent_id}][{trade_date}] 上涨钝化触发，"
                f"开启{self.EXEC_WINDOW_DAYS}天执行窗口 | 龙头={self._window_leader_stocks}"
            )

        # ── Step 5：D-1 涨停池，分类首板/二板/三板（窗口期每天执行）────────
        pre_date = trade_dates[cur_idx - 1]
        try:
            limit_df = get_limit_list_ths(pre_date, limit_type="涨停池")
        except Exception as e:
            logger.warning(f"[{self.agent_id}][{trade_date}] 查涨停池失败：{e}")
            return []

        if limit_df is None or limit_df.empty:
            logger.info(f"[{self.agent_id}][{trade_date}] D-1({pre_date})涨停池为空")
            return []

        limit_df = limit_df[limit_df["ts_code"].apply(_is_main_board)].copy()
        if limit_df.empty:
            return []
        limit_df = limit_df[~limit_df["ts_code"].isin(st_set)].copy()

        # limit_list_ths 表无 cons_nums 列，从 limit_step（连板天梯）获取连板数
        # limit_step 只收录 2板及以上；不在其中的涨停股即为首板（nums=1）
        # try:
        step_df = get_limit_step(pre_date)
        # except Exception as e:
        #     logger.warning(f"[{self.agent_id}][{trade_date}] 查连板天梯失败：{e}，首板判断降级为全部视为首板")
        #     step_df = pd.DataFrame()

        step_map: Dict[str, int] = (
            dict(zip(step_df["ts_code"], step_df["nums"].astype(int)))
            if not step_df.empty and "ts_code" in step_df.columns and "nums" in step_df.columns
            else {}
        )
        # 涨停池中的股票：在 step_map 里取其连板数，不在的视为首板（1）
        limit_df["cons_nums"] = limit_df["ts_code"].map(step_map).fillna(1).astype(int)

        first_board_df  = limit_df[limit_df["cons_nums"] == 1]
        second_board_df = limit_df[limit_df["cons_nums"] == 2]
        third_board_df  = limit_df[limit_df["cons_nums"] == 3]

        logger.info(
            f"[{self.agent_id}][{trade_date}] D-1涨停池(主板) "
            f"首板={len(first_board_df)} 二板={len(second_board_df)} 三板={len(third_board_df)}"
        )

        # ── Step 6：首板额外过滤 ─────────────────────────────────────────────
        gain_threshold = leader_avg_gain * self.FIRST_BOARD_GAIN_RATIO
        filtered_first = []
        for _, row in first_board_df.iterrows():
            ts_code    = row["ts_code"]
            stock_gain = gain_map.get(ts_code)
            if stock_gain is not None and stock_gain > gain_threshold:
                logger.debug(
                    f"[{self.agent_id}] {ts_code} 首板涨幅{stock_gain:.1%}>"
                    f"阈值{gain_threshold:.1%}，排除"
                )
                continue
            if ts_code in mid_pool_set:
                logger.debug(f"[{self.agent_id}] {ts_code} 在mid_position池，排除")
                continue
            filtered_first.append(row)

        candidate_rows = (
            filtered_first
            + [row for _, row in second_board_df.iterrows()]
            + [row for _, row in third_board_df.iterrows()]
        )
        if not candidate_rows:
            logger.info(f"[{self.agent_id}][{trade_date}] 过滤后无候选股票")
            return []

        # ── Step 7：买入价（仅用 D 日已知数据：open 和 pre_close）────────────
        # 不使用 D 日 close/high（只有 EOD 才知道），避免未来函数
        # 判断逻辑：D 日 open ≈ 涨停价 → 排队买入（涨停价）；否则开盘价买入
        pre_close_map = build_pre_close_map(daily_data, context)
        daily_map: Dict[str, pd.Series] = {
            row["ts_code"]: row for _, row in daily_data.iterrows()
        }

        board_label = {1: "首板", 2: "二板", 3: "三板"}
        result = []
        for lim_row in candidate_rows:
            ts_code    = str(lim_row["ts_code"])
            stock_name = str(lim_row.get("name", ""))
            cons       = int(lim_row.get("cons_nums", 0))

            today_kline = daily_map.get(ts_code)
            if today_kline is None:
                logger.debug(f"[{self.agent_id}] {ts_code} D日无行情，跳过")
                continue

            pre_close = pre_close_map.get(ts_code, 0.0)
            if pre_close <= 0:
                pre_close = float(today_kline.get("pre_close", 0) or 0)

            open_p = float(today_kline.get("open", 0) or 0)
            if open_p <= 0:
                continue

            # D 日 open ≈ 涨停价 → 涨停价排队买入；否则开盘价买入
            limit_up = calc_limit_up_price(ts_code, pre_close) if pre_close > 0 else 0.0
            if limit_up > 0 and abs(open_p - limit_up) < 0.015:
                buy_price = limit_up
            else:
                buy_price = open_p

            if buy_price <= 0:
                continue

            result.append({
                "ts_code":    ts_code,
                "stock_name": stock_name,
                "buy_price":  round(buy_price, 2),
                "_board_type": board_label.get(cons, f"{cons}板"),
            })

        logger.info(
            f"[{self.agent_id}][{trade_date}] 买入信号 {len(result)} 只 "
            f"（首板{len(filtered_first)}，二板{len(second_board_df)}，三板{len(third_board_df)}）"
        )
        return result

    # ────────────────────────────────────────────────────────────────────────
    # 卖出信号
    # ────────────────────────────────────────────────────────────────────────
    def check_sell_signal(
        self,
        position: Dict,
        today_row: Optional[Dict],
        context: Dict,
    ) -> bool:
        trading_days_held = int(position.get("trading_days_so_far", 0))
        ts_code = position.get("ts_code", "")
        trade_date = context.get("trade_date", "")

        # 1. 超期止盈：持满 MAX_HOLD_DAYS 天
        if trading_days_held >= self.MAX_HOLD_DAYS:
            position["_sell_reason"] = f"持满{trading_days_held}日止盈"
            logger.debug(
                f"[{self.agent_id}][{ts_code}] {position['_sell_reason']}"
            )
            return True

        if today_row is None or not trade_date:
            return False

        # 2. 止损：HFQ浮亏 ≤ STOP_LOSS_PCT
        hfq_close = self._get_hfq_close(ts_code, trade_date, context)
        hfq_buy = float(position.get("hfq_buy_price", 0))
        if hfq_buy > 0 and hfq_close > 0:
            float_pnl = (hfq_close - hfq_buy) / hfq_buy * 100
            if float_pnl <= self.STOP_LOSS_PCT:
                position["_sell_reason"] = f"止损(浮亏{float_pnl:.2f}%)"
                logger.debug(
                    f"[{self.agent_id}][{ts_code}] {position['_sell_reason']}"
                )
                return True

        return False

    # ────────────────────────────────────────────────────────────────────────
    # 辅助方法
    # ────────────────────────────────────────────────────────────────────────
    def _get_hfq_close(self, ts_code: str, trade_date: str, context: Dict) -> float:
        """
        获取当日 HFQ 收盘价。
        优先从 context['_hfq_preloaded'] 取（零DB），无预加载时降级查库。
        复用 long_ma_trend_tracking 同款 preloaded 机制。
        """
        preloaded = context.get("_hfq_preloaded")
        if preloaded:
            trade_date_fmt = trade_date.replace("-", "")
            hfq_date_list = preloaded.get("hfq_date_list", [])
            hfq_val_list = preloaded.get("hfq_val_list", [])
            try:
                idx = hfq_date_list.index(trade_date_fmt)
                return float(hfq_val_list[idx])
            except (ValueError, IndexError):
                pass

        # 降级查库
        trade_dates = context.get("trade_dates", [])
        mas = technical_features.compute_ma_from_hfq_range(
            ts_code, trade_date, trade_dates, [5]
        )
        return mas.get("hfq_close_today", 0.0)
