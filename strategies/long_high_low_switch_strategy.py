"""
高低切轮动回测策略（HighLowSwitchStrategy）
============================================
移植自 agent_stats/agents/long_high_low_switch.py，适配回测引擎。

策略逻辑（与 agent 完全一致）
--------
高位股（近20日涨幅前1%龙头股）在持续拉升后会出现"上涨钝化"：
价格仍高于5日均线（乖离率>0），但乖离率持续收窄（动能衰减）。
此时市场资金倾向于轮换到低位启动的首板/二板/三板（"高低切"）。

触发后开启5天执行窗口，每天买入昨日首板/二板/三板（仅主板）。
止损 -15%，持满10个交易日止盈。

买入价：D日开盘价（buy_type="open"）
卖出价：收盘卖出（sell_type="close"）
"""
import gc
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy
from features.ma_indicator import technical_features
from utils.common_tools import (
    filter_st_stocks,
    get_daily_kline_data,
    get_kline_day_range,
    get_limit_list_ths,
    get_limit_step,
    get_trade_dates,
)
from utils.log_utils import logger


# ── 复用 agent 的共享参数 ──────────────────────────────────────────────────
BASE_PCT = 0.01       # 全市场前 1% 为基础池
HIGH_PCT = 0.10       # 基础池中前 10% 为高位股
MIN_HIGH = 3          # 高位股最少 3 只
LOOKBACK_DAYS = 21    # 回看交易日数（计算 20 日涨幅）
MID_PCT_OF_BASE = 0.50  # 基础池前50%为中位股范围


def _is_main_board(ts_code: str) -> bool:
    """判断是否主板（沪深主板，10cm），排除创业板/科创板/北交所。"""
    if ts_code.endswith(".BJ"):
        return False
    if ts_code.startswith(("300", "301", "302")) and ts_code.endswith(".SZ"):
        return False
    if ts_code.startswith("688"):
        return False
    return True


class HighLowSwitchStrategy(BaseStrategy):
    """高低切轮动回测策略（完全复用 agent 选股 + 卖出逻辑）"""

    def __init__(self):
        super().__init__()
        self.strategy_name = "高低切轮动策略"

        # ── 上涨钝化参数（与 agent 一致）─────────────────────────────────────
        self.HIGH_POS_LOOKBACK_N = 5
        self.HIGH_POS_OVERLAP_MIN = 3
        self.MA_BIAS_PERIOD = 5

        # ── 执行窗口参数 ─────────────────────────────────────────────────────
        self.EXEC_WINDOW_DAYS = 5

        # ── 买入过滤参数 ─────────────────────────────────────────────────────
        self.FIRST_BOARD_GAIN_RATIO = 0.35

        # ── 风控参数 ─────────────────────────────────────────────────────────
        self.STOP_LOSS_PCT = -0.15   # 止损线（-15%）
        self.MAX_HOLD_DAYS = 10      # 最长持仓天数

        self.strategy_params = {
            "HIGH_POS_LOOKBACK_N": self.HIGH_POS_LOOKBACK_N,
            "EXEC_WINDOW_DAYS": self.EXEC_WINDOW_DAYS,
            "STOP_LOSS_PCT": self.STOP_LOSS_PCT,
            "MAX_HOLD_DAYS": self.MAX_HOLD_DAYS,
            "FIRST_BOARD_GAIN_RATIO": self.FIRST_BOARD_GAIN_RATIO,
        }

        # ── 内部状态 ─────────────────────────────────────────────────────────
        self._trade_dates: List[str] = []       # 完整交易日历（含回看期）
        self._window_end_idx: Optional[int] = None
        self._window_leader_stocks: Optional[set] = None
        self._prev_leader_stocks: Optional[set] = None
        self._buy_price_map: Dict[str, float] = {}   # {ts_code: 实际买入价}
        # 每日信号缓存（引擎每天调2次 generate_signal，避免重复计算）
        self._cached_date: str = ""
        self._cached_buy: Dict[str, str] = {}
        self._cached_sell: Dict[str, str] = {}

    def initialize(self) -> None:
        self._trade_dates = []
        self._window_end_idx = None
        self._window_leader_stocks = None
        self._prev_leader_stocks = None
        self._buy_price_map.clear()
        self.sell_signal_map.clear()
        self._cached_date = ""
        self._cached_buy = {}
        self._cached_sell = {}
        logger.info(f"【{self.strategy_name}】初始化完成")

    # ────────────────────────────────────────────────────────────────────────
    # 核心接口：generate_signal
    # ────────────────────────────────────────────────────────────────────────
    def generate_signal(
        self,
        trade_date: str,
        daily_df: pd.DataFrame,
        positions: Dict[str, any],
    ) -> Tuple[Dict[str, str], Dict[str, str]]:

        # 引擎每日调2次，第二次直接返回缓存（避免重复计算高位池/乖离率）
        if trade_date == self._cached_date:
            return self._cached_buy, self._cached_sell

        # 懒加载交易日历（回测第一天触发，仅执行一次）
        if not self._trade_dates:
            self._load_trade_dates(trade_date)

        # ── 卖出信号 ──────────────────────────────────────────────────────
        sell_signal_map = self._check_sell_signals(trade_date, daily_df, positions)

        # ── 买入信号 ──────────────────────────────────────────────────────
        buy_signal_map = self._generate_buy_signals(trade_date, daily_df, positions)

        # 缓存当日结果
        self._cached_date = trade_date
        self._cached_buy = buy_signal_map
        self._cached_sell = sell_signal_map

        return buy_signal_map, sell_signal_map

    # ────────────────────────────────────────────────────────────────────────
    # 卖出信号检查
    # ────────────────────────────────────────────────────────────────────────
    def _check_sell_signals(
        self,
        trade_date: str,
        daily_df: pd.DataFrame,
        positions: Dict[str, any],
    ) -> Dict[str, str]:
        sell_map: Dict[str, str] = {}

        for ts_code, pos in positions.items():
            # 1. 超期止盈：持满 MAX_HOLD_DAYS 天
            if pos.hold_days >= self.MAX_HOLD_DAYS:
                sell_map[ts_code] = "close"
                logger.debug(
                    f"[高低切][{ts_code}] 持满{pos.hold_days}日，收盘卖出"
                )
                continue

            # 2. 止损：浮亏 <= STOP_LOSS_PCT
            buy_price = self._buy_price_map.get(ts_code, pos.buy_price)
            if buy_price <= 0:
                continue
            stock_row = daily_df[daily_df["ts_code"] == ts_code]
            if stock_row.empty:
                continue
            close_price = float(stock_row["close"].iloc[0])
            pnl = (close_price - buy_price) / buy_price
            if pnl <= self.STOP_LOSS_PCT:
                sell_map[ts_code] = "close"
                logger.debug(
                    f"[高低切][{ts_code}] 止损(浮亏{pnl:.2%})，收盘卖出"
                )

        return sell_map

    # ────────────────────────────────────────────────────────────────────────
    # 买入信号生成（移植自 agent get_signal_stock_pool）
    # ────────────────────────────────────────────────────────────────────────
    def _generate_buy_signals(
        self,
        trade_date: str,
        daily_df: pd.DataFrame,
        positions: Dict[str, any],
    ) -> Dict[str, str]:
        if daily_df.empty:
            return {}

        trade_dates = self._trade_dates
        try:
            cur_idx = trade_dates.index(trade_date)
        except ValueError:
            logger.warning(f"[高低切][{trade_date}] 不在trade_dates中，跳过")
            return {}

        min_required = self.HIGH_POS_LOOKBACK_N + self.MA_BIAS_PERIOD + LOOKBACK_DAYS
        if cur_idx < min_required:
            return {}

        # ── ST 过滤 ──────────────────────────────────────────────────────
        all_codes = daily_df["ts_code"].unique().tolist()
        normal_codes = filter_st_stocks(all_codes, trade_date)
        st_set = set(all_codes) - set(normal_codes)

        # ── Step 1：今日高位池 ─────────────────────────────────────────────
        t21_date_today = trade_dates[cur_idx - LOOKBACK_DAYS]
        t21_df_today = get_kline_day_range(all_codes, t21_date_today, t21_date_today)
        t21_close_map_today = (
            dict(zip(t21_df_today["ts_code"], t21_df_today["close"].astype(float)))
            if not t21_df_today.empty else {}
        )

        gain_list_today = []
        for _, _row in daily_df.iterrows():
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
            return {}

        n_base_today = max(MIN_HIGH, int(len(gain_list_today) * BASE_PCT))
        base_pool_today = gain_list_today[:n_base_today]
        n_high_today = max(MIN_HIGH, int(len(base_pool_today) * HIGH_PCT))

        leader_gains = [t[2] for t in base_pool_today[:n_high_today]]
        leader_avg_gain = sum(leader_gains) / len(leader_gains) if leader_gains else 0.0
        gain_map: Dict[str, float] = {t[0]: t[2] for t in gain_list_today}
        n_mid_upper = int(len(base_pool_today) * MID_PCT_OF_BASE)
        mid_pool_set = set(t[0] for t in base_pool_today[:n_mid_upper])

        # ── 窗口状态机 ────────────────────────────────────────────────────
        in_window = (
            self._window_end_idx is not None
            and cur_idx <= self._window_end_idx
        )

        if self._window_end_idx is not None and not in_window:
            logger.info(
                f"[高低切][{trade_date}] 执行窗口已到期，等待新钝化信号"
            )
            self._prev_leader_stocks = self._window_leader_stocks
            self._window_end_idx = None
            self._window_leader_stocks = None

        if not in_window:
            # ── Step 2：过去 N-1 天高位池 ─────────────────────────────────
            high_pool_today = set(t[0] for t in base_pool_today[:n_high_today])
            all_high_pools: Dict[str, set] = {trade_date: high_pool_today}

            for i in range(1, self.HIGH_POS_LOOKBACK_N):
                past_date = trade_dates[cur_idx - i]
                try:
                    past_daily = get_daily_kline_data(past_date)
                except Exception as e:
                    logger.warning(f"[高低切] 获取{past_date}日线失败：{e}")
                    continue
                if past_daily is None or past_daily.empty:
                    continue
                try:
                    past_idx = trade_dates.index(past_date)
                    t21_date_past = trade_dates[max(0, past_idx - LOOKBACK_DAYS)]
                    past_codes = past_daily["ts_code"].tolist()
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
                        c_t21 = t21_close_map.get(ts, 0.0)
                        if c_today <= 0 or c_t21 <= 0:
                            continue
                        past_gains.append((ts, c_today / c_t21 - 1))
                    past_gains.sort(key=lambda x: -x[1])
                    n_base_p = max(MIN_HIGH, int(len(past_gains) * BASE_PCT))
                    n_high_p = max(MIN_HIGH, int(n_base_p * HIGH_PCT))
                    all_high_pools[past_date] = set(t[0] for t in past_gains[:n_high_p])
                except Exception as e:
                    logger.warning(f"[高低切] {past_date} 高位池计算失败：{e}")
                    continue

            # ── Step 3：重合股 ────────────────────────────────────────────
            all_stocks_flat = [s for pool in all_high_pools.values() for s in pool]
            occurrence = Counter(all_stocks_flat)
            overlap_stocks = [
                ts for ts, cnt in occurrence.items()
                if cnt >= self.HIGH_POS_OVERLAP_MIN
            ]

            if not overlap_stocks:
                return {}

            # ── Step 4：乖离率趋势 ───────────────────────────────────────
            bias_lookback_start = trade_dates[
                max(0, cur_idx - self.HIGH_POS_LOOKBACK_N - self.MA_BIAS_PERIOD - 2)
            ]
            kline_df = get_kline_day_range(overlap_stocks, bias_lookback_start, trade_date)
            if kline_df is None or kline_df.empty:
                return {}

            kline_df["trade_date"] = kline_df["trade_date"].astype(str).str.replace("-", "")
            kline_df = kline_df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
            window_dates = [
                trade_dates[cur_idx - i].replace("-", "")
                for i in range(self.HIGH_POS_LOOKBACK_N - 1, -1, -1)
            ]

            daily_avg_bias: List[Optional[float]] = []
            for d_fmt in window_dates:
                biases = []
                for ts_code in overlap_stocks:
                    stock_df = kline_df[kline_df["ts_code"] == ts_code].sort_values("trade_date")
                    stock_closes = stock_df["close"].astype(float).tolist()
                    dates_list = stock_df["trade_date"].tolist()
                    if d_fmt not in dates_list:
                        continue
                    d_idx = dates_list.index(d_fmt)
                    if d_idx + 1 < self.MA_BIAS_PERIOD:
                        continue
                    bias_series = technical_features.compute_bias_rate_series(
                        stock_closes, ma_period=self.MA_BIAS_PERIOD
                    )
                    bias_val = bias_series[d_idx]
                    if bias_val is not None:
                        biases.append(bias_val)
                daily_avg_bias.append(
                    round(sum(biases) / len(biases), 4) if biases else None
                )

            valid_bias = [(i, v) for i, v in enumerate(daily_avg_bias) if v is not None]
            if len(valid_bias) < 3:
                del kline_df
                gc.collect()
                return {}

            xs = np.array([x for x, _ in valid_bias], dtype=float)
            ys = np.array([y for _, y in valid_bias], dtype=float)
            slope = float(np.polyfit(xs, ys, 1)[0])
            latest_bias = valid_bias[-1][1]
            is_dulling = slope < 0 and latest_bias > 0

            logger.info(
                f"[高低切][{trade_date}] 乖离率趋势：slope={slope:.4f}，"
                f"今日均值={latest_bias:.2f}%，钝化={is_dulling}"
            )

            del kline_df
            gc.collect()

            if not is_dulling:
                return {}

            # ── 龙头换手检查 ─────────────────────────────────────────────
            if self._prev_leader_stocks:
                conflict = set(overlap_stocks) & self._prev_leader_stocks
                if conflict:
                    logger.info(
                        f"[高低切][{trade_date}] 龙头未换手，与前窗口重合={conflict}"
                    )
                    return {}

            # ── 开启执行窗口 ─────────────────────────────────────────────
            self._window_end_idx = cur_idx + self.EXEC_WINDOW_DAYS - 1
            self._window_leader_stocks = set(overlap_stocks)
            logger.info(
                f"[高低切][{trade_date}] 上涨钝化触发，"
                f"开启{self.EXEC_WINDOW_DAYS}天执行窗口"
            )

        # ── Step 5：D-1 涨停池 ────────────────────────────────────────────
        pre_date = trade_dates[cur_idx - 1]
        try:
            limit_df = get_limit_list_ths(pre_date, limit_type="涨停池")
        except Exception as e:
            logger.warning(f"[高低切][{trade_date}] 查涨停池失败：{e}")
            return {}

        if limit_df is None or limit_df.empty:
            return {}

        limit_df = limit_df[limit_df["ts_code"].apply(_is_main_board)].copy()
        if limit_df.empty:
            return {}
        limit_df = limit_df[~limit_df["ts_code"].isin(st_set)].copy()
        if limit_df.empty:
            return {}

        # 连板天梯获取连板数
        step_df = get_limit_step(pre_date)
        step_map: Dict[str, int] = (
            dict(zip(step_df["ts_code"], step_df["nums"].astype(int)))
            if not step_df.empty and "ts_code" in step_df.columns and "nums" in step_df.columns
            else {}
        )
        limit_df["cons_nums"] = limit_df["ts_code"].map(step_map).fillna(1).astype(int)

        first_board_df = limit_df[limit_df["cons_nums"] == 1]
        second_board_df = limit_df[limit_df["cons_nums"] == 2]
        third_board_df = limit_df[limit_df["cons_nums"] == 3]

        # ── Step 6：首板额外过滤 ─────────────────────────────────────────
        gain_threshold = leader_avg_gain * self.FIRST_BOARD_GAIN_RATIO
        filtered_first = []
        for _, row in first_board_df.iterrows():
            ts_code = row["ts_code"]
            stock_gain = gain_map.get(ts_code)
            if stock_gain is not None and stock_gain > gain_threshold:
                continue
            if ts_code in mid_pool_set:
                continue
            filtered_first.append(row)

        candidate_rows = (
            filtered_first
            + [row for _, row in second_board_df.iterrows()]
            + [row for _, row in third_board_df.iterrows()]
        )
        if not candidate_rows:
            return {}

        # ── Step 7：构建买入信号（开盘价买入）─────────────────────────────
        # 排除已持仓的标的
        held_codes = set(positions.keys())
        buy_signal_map: Dict[str, str] = {}
        for lim_row in candidate_rows:
            ts_code = str(lim_row["ts_code"])
            if ts_code in held_codes:
                continue
            buy_signal_map[ts_code] = "open"

        logger.info(
            f"[高低切][{trade_date}] 买入信号 {len(buy_signal_map)} 只 "
            f"（首板{len(filtered_first)}，二板{len(second_board_df)}，三板{len(third_board_df)}）"
        )
        return buy_signal_map

    # ────────────────────────────────────────────────────────────────────────
    # 回调方法
    # ────────────────────────────────────────────────────────────────────────
    def on_buy_success(self, ts_code: str, buy_price: float):
        self._buy_price_map[ts_code] = buy_price

    def on_sell_success(self, ts_code: str):
        self._buy_price_map.pop(ts_code, None)

    def on_buy_failed(self, ts_code: str, reason: str):
        logger.debug(f"[高低切] {ts_code} 买入失败：{reason}")

    # ────────────────────────────────────────────────────────────────────────
    # 辅助方法
    # ────────────────────────────────────────────────────────────────────────
    def _load_trade_dates(self, first_trade_date: str):
        """
        加载交易日历（回看1年，确保高位池 + 乖离率计算有足够历史数据）。
        """
        # 计算回看起点：1年前
        year = int(first_trade_date[:4]) - 1
        lookback_start = f"{year}{first_trade_date[4:]}"
        # 用当前已知的最远未来日期（回测结束日由引擎控制，此处只需覆盖即可）
        # 多取一些日期不影响正确性，trade_dates.index(trade_date) 仍能找到
        far_end = f"{int(first_trade_date[:4]) + 2}-12-31"
        try:
            self._trade_dates = get_trade_dates(lookback_start, far_end)
            logger.info(
                f"[高低切] 交易日历加载完成：{self._trade_dates[0]} ~ "
                f"{self._trade_dates[-1]}，共{len(self._trade_dates)}天"
            )
        except Exception as e:
            logger.error(f"[高低切] 交易日历加载失败：{e}")
            self._trade_dates = []
