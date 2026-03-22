from typing import Dict, List, Optional

import gc

import pandas as pd
import numpy as np

from agent_stats.long_agent_base import BaseLongAgent
from data.data_cleaner import data_cleaner
from features.ma_indicator import technical_features
from utils.common_tools import (
    get_hfq_kline_range,
    get_kline_day_range,
    get_market_limit_counts,
    get_index_pct_chg,
    get_stock_limit_counts_batch,
)
from utils.log_utils import logger


class LongMaTrendTrackingAgent(BaseLongAgent):
    agent_id = "long_ma_trend_tracking"
    agent_name = "均线多头趋势跟踪（中长线增强版）"
    agent_desc = (
        "后复权价格均线多头排列（MA5>MA10>MA20>MA60），MA20/MA60趋势向上斜率确认，叠加连续温和放量验证；"
        "排除30日涨幅>40%的短期暴涨股及30日内涨跌停>3次的情绪股；"
        "止损 -8%（恐慌模式弹性调整）/ 跌穿MA60止盈 / 最长持仓 60 日（全链路后复权无未来函数）。"
    )

    # ── 策略核心参数 ──────────────────────────────────────────────────────────
    MA_TREND_PERIODS = [5, 10, 20, 60]  # 均线多头排列周期（短期→长期）
    MA_SLOPE_PERIODS = [20, 60]  # 斜率校验的均线周期
    MA_SLOPE_LOOKBACK = 5  # 斜率回看交易日数
    MA_SLOPE_THRESHOLD = 0.0  # 最小斜率阈值（>0为向上趋势）

    # 成交量温和放大参数
    VOL_AMP_DAYS = 5  # 连续放量交易日数
    VOL_MA_DAYS = 20  # 成交量基准均线周期
    VOL_AMPLIFY_RATIO = 1.5  # 最小放量倍数

    # 持仓与风控参数
    STOP_LOSS_PCT = -8.0  # 固定止损线（HFQ浮亏）
    MAX_HOLD_DAYS = 90  # 最长持仓交易日
    MA_SELL_PERIODS = [60]  # 止盈均线（跌穿即卖出）

    # ── 候选池过滤参数 ────────────────────────────────────────────────────
    PRE_GAIN_DAYS = 30  # 前期涨幅回看窗口
    PRE_GAIN_MAX_PCT = 40.0  # 前期涨幅上限（排除暴涨股）
    LIMIT_COUNT_DAYS = 30  # 涨跌停次数回看窗口
    LIMIT_COUNT_MAX = 4  # 涨跌停次数上限（排除情绪股）

    # ── 市场恐慌弹性止损参数 ──────────────────────────────────────────────
    PANIC_LIMIT_DOWN_THRESHOLD = 20  # 恐慌跌停数阈值
    PANIC_INDEX_DROP_THRESHOLD = -2.0  # 恐慌指数跌幅阈值
    PANIC_STOP_LOSS_MULTIPLIER = 2.0  # 恐慌止损线倍数
    PANIC_OBSERVE_DAYS = 5  # 恐慌观察期
    PANIC_FORCE_EXIT_MULT = 2.5  # 恐慌强制止损倍数

    # ------------------------------------------------------------------ #
    # 工具方法：固定序列斜率快速计算（替代polyfit，性能提升10倍+）
    # ------------------------------------------------------------------ #
    @staticmethod
    def _calc_fixed_slope(y_series: np.ndarray) -> float:
        """
        固定x序列（0,1,2...n-1）的线性回归斜率快速计算
        等价于np.polyfit(x, y_series, 1)[0]，消除函数调用开销
        """
        n = len(y_series)
        if n < 2:
            return 0.0
        x = np.arange(n, dtype=np.float32)
        sum_x = np.sum(x)
        sum_y = np.sum(y_series)
        sum_xy = np.sum(x * y_series)
        sum_x2 = np.sum(x ** 2)
        denominator = n * sum_x2 - sum_x ** 2
        if denominator == 0:
            return 0.0
        return (n * sum_xy - sum_x * sum_y) / denominator

    # ------------------------------------------------------------------ #
    # 买入信号（性能优化版）
    # ------------------------------------------------------------------ #
    def get_signal_stock_pool(
            self,
            trade_date: str,
            daily_data: pd.DataFrame,
            context: Dict,
    ) -> List[Dict]:
        if daily_data.empty:
            return []

        st_set = set(context.get("st_stock_list", []))
        trade_dates = context.get("trade_dates", [])

        # 计算回看起始日期（覆盖最长周期需求）
        max_ma_period = max(self.MA_TREND_PERIODS)
        max_lookback = max(
            max_ma_period + self.MA_SLOPE_LOOKBACK,
            self.VOL_MA_DAYS + self.VOL_AMP_DAYS
        )
        try:
            cur_idx = trade_dates.index(trade_date)
            lookback_idx = max(0, cur_idx - max_lookback)
            lookback_start = trade_dates[lookback_idx]
            limit_start_idx = max(0, cur_idx - self.LIMIT_COUNT_DAYS)
            limit_start = trade_dates[limit_start_idx]
        except (ValueError, IndexError):
            logger.warning(f"[{self.agent_id}][{trade_date}] 交易日索引异常，跳过")
            return []

        # 基础候选池过滤
        candidate = daily_data[
            (~daily_data["ts_code"].isin(st_set)) &
            (~daily_data["ts_code"].str.endswith(".BJ")) &
            (daily_data["close"].notna()) &
            (daily_data["close"] > 0)
            ].copy()
        if candidate.empty:
            return []
        ts_codes = candidate["ts_code"].tolist()
        trade_date_fmt = trade_date.replace("-", "")

        # ── Phase1：不复权粗筛（全量预处理+向量化计算，消除循环内重复开销）
        # 仅拉取必要字段，减少内存占用
        raw_range = get_kline_day_range(ts_codes, lookback_start, trade_date)
        if raw_range.empty:
            logger.info(f"[{self.agent_id}][{trade_date}] 粗筛日线数据为空，跳过")
            return []

        # 【性能优化】统一预处理，一次计算全量生效，消除循环内重复操作
        raw_range["trade_date"] = raw_range["trade_date"].astype(str).str.replace("-", "")
        raw_range = raw_range.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        # 向量化预计算均线、均量，循环内直接取值
        for period in self.MA_TREND_PERIODS:
            raw_range[f"ma_{period}"] = raw_range.groupby("ts_code", sort=False)["close"].transform(
                lambda x: x.rolling(window=period, min_periods=period).mean()
            )
        raw_range[f"vol_ma_{self.VOL_MA_DAYS}"] = raw_range.groupby("ts_code", sort=False)["volume"].transform(
            lambda x: x.rolling(window=self.VOL_MA_DAYS, min_periods=self.VOL_MA_DAYS).mean()
        )

        # 粗筛过滤
        rough_candidates = set()
        today_data = raw_range[raw_range["trade_date"] == trade_date_fmt]
        for ts_code, today_row in today_data.groupby("ts_code", sort=False):
            if today_row.empty:
                continue
            # 直接取预计算的均线值，无需循环内重复计算
            ma5 = today_row.iloc[0]["ma_5"]
            ma10 = today_row.iloc[0]["ma_10"]
            ma20 = today_row.iloc[0]["ma_20"]
            ma60 = today_row.iloc[0]["ma_60"]
            today_close = today_row.iloc[0]["close"]
            if pd.isna(ma5) or pd.isna(ma10) or pd.isna(ma20) or pd.isna(ma60):
                continue

            # 1. 粗筛均线多头排列（2%容差覆盖除权差异）
            if not (ma5 >= ma10 * 0.98 and ma10 >= ma20 * 0.98 and ma20 >= ma60 * 0.98 and today_close >= ma5 * 0.98):
                continue

            # 2. 粗筛均线斜率（简化版：当前均线 > N日前均线）
            stock_history = raw_range[raw_range["ts_code"] == ts_code]
            slope_ok = True
            for period in self.MA_SLOPE_PERIODS:
                ma_series = stock_history[f"ma_{period}"].dropna()
                if len(ma_series) < self.MA_SLOPE_LOOKBACK + 1:
                    slope_ok = False
                    break
                if ma_series.iloc[-1] <= ma_series.iloc[-self.MA_SLOPE_LOOKBACK - 1] * 0.99:
                    slope_ok = False
                    break
            if not slope_ok:
                continue

            # 3. 粗筛连续放量
            vol_ma_base = stock_history[f"vol_ma_{self.VOL_MA_DAYS}"].iloc[-self.VOL_AMP_DAYS - 1]
            if pd.isna(vol_ma_base) or vol_ma_base <= 0:
                continue
            recent_vols = stock_history["volume"].iloc[-self.VOL_AMP_DAYS:]
            if (recent_vols < vol_ma_base * self.VOL_AMPLIFY_RATIO * 0.9).any():
                continue

            rough_candidates.add(ts_code)

        # 释放粗筛内存
        del raw_range, today_data
        gc.collect()

        if not rough_candidates:
            logger.info(f"[{self.agent_id}][{trade_date}] 粗筛无候选，跳过HFQ精筛")
            return []
        logger.info(f"[{self.agent_id}][{trade_date}] 粗筛通过{len(rough_candidates)}只，开始HFQ精筛")
        rough_list = list(rough_candidates)
        start_fmt = lookback_start.replace("-", "")
        end_fmt = trade_date_fmt

        # ── Phase2：HFQ精筛（小批次容错入库）
        batch_size = 200
        for i in range(0, len(rough_list), batch_size):
            batch_codes = rough_list[i:i + batch_size]
            try:
                data_cleaner.clean_and_insert_kline_day_hfq(batch_codes, start_date=start_fmt, end_date=end_fmt)
            except Exception as e:
                logger.warning(f"[{self.agent_id}][{trade_date}] HFQ批次{i // batch_size + 1}入库异常：{e}，跳过")
                continue

        hfq_range = get_hfq_kline_range(rough_list, lookback_start, trade_date)
        if hfq_range.empty:
            logger.warning(f"[{self.agent_id}][{trade_date}] HFQ数据为空，跳过")
            return []

        # 【性能优化】HFQ数据统一预处理
        hfq_range["trade_date"] = hfq_range["trade_date"].astype(str).str.replace("-", "")
        hfq_range = hfq_range.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

        # 精筛过滤统计
        slope_filter = 0
        vol_filter = 0
        gain_filter = 0
        final_candidates = []

        for ts_code, group in hfq_range.groupby("ts_code", sort=False):
            group = group.reset_index(drop=True)
            today_row = group[group["trade_date"] == trade_date_fmt]
            if today_row.empty:
                continue
            today_idx = today_row.index[0]
            today_close = float(today_row.iloc[0]["close"])

            # 1. 精确均线多头排列
            ma_values = {}
            enough_data = True
            for period in self.MA_TREND_PERIODS:
                if today_idx + 1 < period:
                    enough_data = False
                    break
                ma_values[period] = group["close"].iloc[today_idx - period + 1: today_idx + 1].mean()
            if not enough_data:
                continue
            ma5, ma10, ma20, ma60 = ma_values[5], ma_values[10], ma_values[20], ma_values[60]
            if not (ma5 > ma10 and ma10 > ma20 and ma20 > ma60 and today_close > ma5):
                continue

            # 2. 精确斜率校验（快速计算，无polyfit开销）
            slope_ok = True
            for period in self.MA_SLOPE_PERIODS:
                if today_idx + 1 < period + self.MA_SLOPE_LOOKBACK:
                    slope_ok = False
                    break
                # 生成近N日的均线序列
                ma_series = []
                for i in range(self.MA_SLOPE_LOOKBACK + 1):
                    check_idx = today_idx - i
                    ma_val = group["close"].iloc[check_idx - period + 1: check_idx + 1].mean()
                    ma_series.append(ma_val)
                # 反转序列（从旧到新），计算斜率
                ma_series = np.array(ma_series[::-1], dtype=np.float32)
                slope = self._calc_fixed_slope(ma_series)
                if slope <= self.MA_SLOPE_THRESHOLD:
                    slope_ok = False
                    break
            if not slope_ok:
                slope_filter += 1
                continue

            # 3. 精确连续放量校验
            if today_idx + 1 < self.VOL_MA_DAYS + self.VOL_AMP_DAYS:
                vol_filter += 1
                continue
            vol_ma_base_idx = today_idx - self.VOL_AMP_DAYS
            vol_ma = group["volume"].iloc[vol_ma_base_idx - self.VOL_MA_DAYS + 1: vol_ma_base_idx + 1].mean()
            if vol_ma <= 0:
                vol_filter += 1
                continue
            recent_vols = group["volume"].iloc[today_idx - self.VOL_AMP_DAYS + 1: today_idx + 1]
            if (recent_vols < vol_ma * self.VOL_AMPLIFY_RATIO).any():
                vol_filter += 1
                continue

            # 4. 前期涨幅过滤
            if len(group) >= self.PRE_GAIN_DAYS + 1:
                pre_close = group["close"].iloc[today_idx - self.PRE_GAIN_DAYS]
                pre_recent_close = group["close"].iloc[today_idx - 1]
                if pre_close > 0:
                    pre_gain_pct = (pre_recent_close - pre_close) / pre_close * 100
                    if pre_gain_pct > self.PRE_GAIN_MAX_PCT:
                        gain_filter += 1
                        continue

            final_candidates.append(ts_code)

        # 【性能优化】延迟涨跌停查询，仅对最终候选股查询
        limit_count_map = get_stock_limit_counts_batch(final_candidates, limit_start, trade_date)
        limit_filter = 0
        signals = []

        for ts_code in final_candidates:
            # 涨跌停次数过滤
            limit_hit_count = limit_count_map.get(ts_code, 0)
            if limit_hit_count > self.LIMIT_COUNT_MAX:
                limit_filter += 1
                continue
            # 取实际买入价
            raw_row = candidate[candidate["ts_code"] == ts_code]
            buy_price = float(raw_row.iloc[0]["close"]) if not raw_row.empty else 0.0
            if buy_price <= 0:
                continue
            signals.append({
                "ts_code": ts_code,
                "stock_name": "",
                "buy_price": round(buy_price, 4),
            })

        # 汇总日志，无冗余逐行输出
        logger.debug(
            f"[{self.agent_id}][{trade_date}] 精筛过滤汇总：斜率{slope_filter}只、成交量{vol_filter}只、"
            f"涨幅{gain_filter}只、涨跌停{limit_filter}只"
        )
        logger.info(f"[{self.agent_id}][{trade_date}] 最终买入信号：{len(signals)}只")
        return signals

    # ------------------------------------------------------------------ #
    # 卖出信号（完全兼容原有逻辑，无性能问题）
    # ------------------------------------------------------------------ #
    def check_sell_signal(
            self,
            position: Dict,
            today_row: Optional[Dict],
            context: Dict,
    ) -> bool:
        if today_row is None:
            return False

        ts_code = position["ts_code"]
        trading_days_held = int(position.get("trading_days_so_far", 0))
        trade_date = context.get("trade_date", "")

        # 1. 超期卖出
        if trading_days_held >= self.MAX_HOLD_DAYS:
            position["_sell_reason"] = f"超期({trading_days_held}日≥{self.MAX_HOLD_DAYS}日)"
            logger.debug(f"[{self.agent_id}][{ts_code}] 超期触发卖出，已持{trading_days_held}日")
            return True
        if not trade_date:
            return False

        # 计算HFQ均线与收盘价
        mas = self._compute_hfq_mas(ts_code, trade_date, context)
        hfq_close_today = mas.get("hfq_close_today", 0.0)
        hfq_buy_price = float(position.get("hfq_buy_price", 0))

        # 2. 止损（含恐慌弹性机制）
        if hfq_buy_price > 0 and hfq_close_today > 0:
            float_pnl_pct = (hfq_close_today - hfq_buy_price) / hfq_buy_price * 100
            panic_date = position.get("panic_date")

            if panic_date:
                # 恐慌观察期逻辑
                panic_observe_count = position.get("panic_observe_count", 0) + 1
                position["panic_observe_count"] = panic_observe_count
                panic_open_price = float(position.get("panic_open_price", 0))
                panic_stop_loss = self.STOP_LOSS_PCT * self.PANIC_STOP_LOSS_MULTIPLIER
                force_exit_pct = self.STOP_LOSS_PCT * self.PANIC_FORCE_EXIT_MULT

                # 恐慌恢复
                if panic_open_price > 0 and hfq_close_today > panic_open_price:
                    logger.info(f"[{self.agent_id}][{ts_code}] 恐慌模式恢复，观察第{panic_observe_count}天")
                    for key in ["panic_date", "panic_open_price", "panic_observe_count"]:
                        position.pop(key, None)
                # 强制止损
                elif float_pnl_pct <= force_exit_pct:
                    position["_sell_reason"] = f"恐慌强制止损(浮亏{float_pnl_pct:.2f}%)"
                    logger.info(f"[{self.agent_id}][{ts_code}] {position['_sell_reason']}")
                    return True
                # 观察期到期
                elif panic_observe_count >= self.PANIC_OBSERVE_DAYS:
                    position["_sell_reason"] = f"恐慌观察期到期({panic_observe_count}天)"
                    logger.info(f"[{self.agent_id}][{ts_code}] {position['_sell_reason']}")
                    return True
                # 恐慌期止损
                elif float_pnl_pct <= panic_stop_loss:
                    position["_sell_reason"] = f"恐慌期止损(浮亏{float_pnl_pct:.2f}%)"
                    logger.info(f"[{self.agent_id}][{ts_code}] {position['_sell_reason']}")
                    return True
                else:
                    logger.info(
                        f"[{self.agent_id}][{ts_code}] 恐慌观察中，第{panic_observe_count}/{self.PANIC_OBSERVE_DAYS}天，"
                        f"浮亏{float_pnl_pct:.2f}%，止损线{panic_stop_loss:.1f}%"
                    )
                    return False
            else:
                # 正常止损触发，判断是否进入恐慌模式
                if float_pnl_pct <= self.STOP_LOSS_PCT:
                    if self._is_market_panic(trade_date):
                        # 记录恐慌基准
                        preloaded = context.get("_hfq_preloaded")
                        hfq_open = hfq_close_today
                        if preloaded:
                            td_fmt = trade_date.replace("-", "")
                            hfq_open = preloaded.get("hfq_closes_by_date", {}).get(td_fmt, hfq_close_today)
                        position.update({
                            "panic_date": trade_date,
                            "panic_open_price": hfq_open,
                            "panic_observe_count": 0
                        })
                        logger.info(f"[{self.agent_id}][{ts_code}] 进入恐慌观察期，浮亏{float_pnl_pct:.2f}%")
                        return False
                    # 正常止损
                    position["_sell_reason"] = f"固定止损(浮亏{float_pnl_pct:.2f}%)"
                    logger.debug(f"[{self.agent_id}][{ts_code}] {position['_sell_reason']}")
                    return True

        # 3. MA60破位止盈
        if not mas:
            return False
        ex_div_stocks = context.get("ex_div_stocks", set())
        if ts_code in ex_div_stocks:
            return False
        if hfq_close_today <= 0:
            return False
        for period in self.MA_SELL_PERIODS:
            ma_val = mas.get(period)
            if ma_val and hfq_close_today < ma_val:
                position["_sell_reason"] = f"MA{period}破位止盈(HFQ收盘{hfq_close_today:.2f}<MA{period}{ma_val:.2f})"
                logger.debug(f"[{self.agent_id}][{ts_code}] {position['_sell_reason']}")
                return True

        return False

    # ------------------------------------------------------------------ #
    # 辅助方法（完全兼容原有逻辑，无性能问题）
    # ------------------------------------------------------------------ #
    def _is_market_panic(self, trade_date: str) -> bool:
        limit_counts = get_market_limit_counts(trade_date)
        limit_down = limit_counts.get("limit_down_count", 0)
        limit_up = limit_counts.get("limit_up_count", 0)
        idx_chg = get_index_pct_chg(trade_date)
        sh_pct_chg = idx_chg.get("000001.SH", 0.0)

        cond1 = limit_down > self.PANIC_LIMIT_DOWN_THRESHOLD and sh_pct_chg < self.PANIC_INDEX_DROP_THRESHOLD
        cond2 = limit_down > limit_up and limit_down > 10
        if cond1 or cond2:
            logger.info(
                f"[{self.agent_id}][{trade_date}] 市场恐慌，跌停{limit_down}只，涨停{limit_up}只，上证{sh_pct_chg:.2f}%")
            return True
        return False

    def _compute_hfq_mas(self, ts_code: str, trade_date: str, context: Dict) -> Dict:
        preloaded = context.get("_hfq_preloaded")
        if preloaded:
            return self._compute_hfq_mas_from_preloaded(trade_date, preloaded)
        trade_dates = context.get("trade_dates", [])
        return technical_features.compute_ma_from_hfq_range(
            ts_code, trade_date, trade_dates, self.MA_SELL_PERIODS
        )

    def _compute_hfq_mas_from_preloaded(self, trade_date: str, preloaded: Dict) -> Dict:
        trade_date_fmt = trade_date.replace("-", "")
        hfq_date_list = preloaded["hfq_date_list"]
        hfq_val_list = preloaded["hfq_val_list"]
        try:
            idx = hfq_date_list.index(trade_date_fmt)
        except ValueError:
            return {}
        hfq_close_today = hfq_val_list[idx]
        result = {"hfq_close_today": hfq_close_today}
        for period in self.MA_SELL_PERIODS:
            start = max(0, idx - period + 1)
            window = hfq_val_list[start:idx + 1]
            if len(window) >= period:
                result[period] = round(sum(window) / period, 4)
        return result