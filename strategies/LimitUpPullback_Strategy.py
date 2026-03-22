import pandas as pd
from typing import List, Dict, Tuple
from strategies.base_strategy import BaseStrategy
from utils.log_utils import logger
from utils.common_tools import get_trade_dates, filter_st_stocks


class LimitUpPullback_Strategy(BaseStrategy):
    """
    涨停回马枪策略（精简版）
    核心逻辑：强势涨停 → 合理回调 → 均线支撑 → 止盈止损
    复用项目工具：ST过滤、涨停价计算、交易日历（均为项目已有实现）
    """

    def __init__(self):
        super().__init__()
        self.strategy_name = "涨停回马枪策略"

        # ========== 核心策略参数（易优化） ==========
        self.limit_up_lookback_days = 10  # 涨停回溯天数
        self.pullback_rate = -0.08  # 回调幅度阈值（下跌≥8%）
        self.support_ma = 10  # 支撑均线周期
        self.ma_tolerance = 0.02  # 价格偏离均线容忍度（±2%）
        self.take_profit_rate = 0.1  # 止盈比例
        self.stop_loss_rate = -0.06  # 止损比例
        self.max_position_count = 6  # 最大持仓数

        # ========== 核心缓存（仅保留必要数据） ==========
        self.limit_up_history = {}  # {ts_code: [{"trade_date": "", "close_price": ""}]}
        self.price_cache = {}  # {ts_code: [close_price1, close_price2...]}
        self.buy_price_map = {}  # {ts_code: 实际买入价}

    def initialize(self) -> None:
        """策略初始化：清空缓存，保证无状态启动"""
        self.limit_up_history.clear()
        self.price_cache.clear()
        self.buy_price_map.clear()
        logger.info(f"【{self.strategy_name}】初始化完成 | 最大持仓{self.max_position_count}只")

    def generate_signal(
            self,
            trade_date: str,
            daily_df: pd.DataFrame,
            positions: Dict[str, any]
    ) -> Tuple[List[str], Dict[str, str]]:
        """
        核心方法：生成当日买卖信号（仅保留核心逻辑，易扩展）
        :return: buy_stocks（买入列表）, sell_signal_map（卖出字典）
        """
        # ========== 步骤1：基础初始化 ==========
        buy_stocks = []
        sell_signal_map = {}
        hold_ts_codes = list(positions.keys())
        available_position = self.max_position_count - len(hold_ts_codes)
        if available_position <= 0:
            logger.info(f"[{trade_date}] 已达最大持仓，跳过选股")
            return buy_stocks, sell_signal_map

        # ========== 步骤2：前置过滤（复用项目工具） ==========
        # 过滤零成交股票
        valid_df = daily_df[daily_df["volume"] > 0].copy()
        # 复用项目ST过滤工具（核心：传入当日日期+股票列表）
        normal_ts_codes = filter_st_stocks(valid_df["ts_code"].unique().tolist(), trade_date)
        valid_df = valid_df[valid_df["ts_code"].isin(normal_ts_codes)]
        if valid_df.empty:
            logger.warning(f"[{trade_date}] 无有效股票数据，返回空信号")
            return buy_stocks, sell_signal_map

        # ========== 步骤3：持仓止盈止损（核心风控） ==========
        for ts_code in hold_ts_codes:
            stock_data = valid_df[valid_df["ts_code"] == ts_code]
            if stock_data.empty:
                continue
            current_price = stock_data["close"].iloc[0]
            buy_price = self.buy_price_map.get(ts_code, current_price)
            pct_change = (current_price / buy_price) - 1

            # 止盈：收盘卖出
            if pct_change >= self.take_profit_rate:
                sell_signal_map[ts_code] = "close"
                self.buy_price_map.pop(ts_code, None)
            # 止损：开盘卖出
            elif pct_change <= self.stop_loss_rate:
                sell_signal_map[ts_code] = "open"
                self.buy_price_map.pop(ts_code, None)

        # ========== 步骤4：更新涨停历史&价格缓存 ==========
        # 计算当日涨跌幅，判断涨停
        valid_df["pct_change"] = (valid_df["close"] / valid_df["pre_close"]) - 1
        for _, row in valid_df.iterrows():
            ts_code = row["ts_code"]
            pre_close = row["pre_close"]
            close_price = row["close"]

            # 初始化缓存
            if ts_code not in self.limit_up_history:
                self.limit_up_history[ts_code] = []
            if ts_code not in self.price_cache:
                self.price_cache[ts_code] = []

            # 更新价格缓存（仅保留最近N天，避免冗余）
            self.price_cache[ts_code].append(close_price)
            if len(self.price_cache[ts_code]) > self.support_ma + 5:
                self.price_cache[ts_code] = self.price_cache[ts_code][-(self.support_ma + 5):]

            # 判断有效涨停（复用基类涨停价计算）
            limit_up_price = self.calc_limit_up_price(ts_code, pre_close)
            is_valid_limit_up = (close_price >= limit_up_price - 0.001) and (row["open"] < limit_up_price - 0.001)
            if is_valid_limit_up:
                self.limit_up_history[ts_code].append({
                    "trade_date": trade_date,
                    "close_price": close_price
                })
                # 清理过期涨停记录（复用交易日历工具）
                start_date = get_trade_dates(
                    start_date=f"{trade_date[:4]}-01-01",
                    end_date=trade_date
                )[-self.limit_up_lookback_days] if len(get_trade_dates(f"{trade_date[:4]}-01-01",
                                                                       trade_date)) >= self.limit_up_lookback_days else trade_date
                self.limit_up_history[ts_code] = [
                    lu for lu in self.limit_up_history[ts_code] if lu["trade_date"] >= start_date
                ]

        # ========== 步骤5：筛选买入标的（核心选股逻辑） ==========
        for ts_code in self.limit_up_history.keys():
            # 跳过持仓/无价格数据/当日无数据的股票
            if ts_code in hold_ts_codes or len(self.price_cache.get(ts_code, [])) < self.support_ma:
                continue
            stock_data = valid_df[valid_df["ts_code"] == ts_code]
            if stock_data.empty:
                continue

            # 核心数据提取
            current_price = stock_data["close"].iloc[0]
            recent_limit_up = self.limit_up_history[ts_code][-self.limit_up_lookback_days:]
            if not recent_limit_up:
                continue
            last_limit_up_price = recent_limit_up[-1]["close_price"]

            # 条件1：回调幅度达标
            pullback_pct = (current_price / last_limit_up_price) - 1
            if pullback_pct > self.pullback_rate:
                continue

            # 条件2：均线支撑+均线向上
            price_series = pd.Series(self.price_cache[ts_code])
            ma_current = price_series.rolling(self.support_ma).mean().iloc[-1]
            ma_prev = price_series.rolling(self.support_ma).mean().iloc[-2]
            price_vs_ma = abs(current_price / ma_current - 1)
            if price_vs_ma > self.ma_tolerance or ma_current <= ma_prev:
                continue

            # 所有条件满足，加入买入列表
            buy_stocks.append(ts_code)
            self.buy_price_map[ts_code] = current_price  # 临时记录，成交后更新
            if len(buy_stocks) >= available_position:
                break

        # ========== 步骤6：日志汇总（关键信息） ==========
        logger.info(
            f"[{trade_date}] 【{self.strategy_name}】信号汇总 | "
            f"买入{len(buy_stocks)}只 | 卖出{len(sell_signal_map)}只 | 持仓{len(hold_ts_codes)}只"
        )
        return buy_stocks, sell_signal_map

    def on_buy_success(self, ts_code: str, buy_price: float):
        """买入成交回调：更新真实买入价（核心，避免预估偏差）"""
        self.buy_price_map[ts_code] = buy_price
        logger.info(f"[{ts_code}] 买入成交 | 成交价：{buy_price:.2f}")