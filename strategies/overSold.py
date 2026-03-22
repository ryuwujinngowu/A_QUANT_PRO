import pandas as pd
from strategies.base_strategy import BaseStrategy
from utils.log_utils import logger


class OversoldRebound_Strategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.strategy_name = "超跌反弹均值回归策略"
        # 策略参数
        self.continuous_down_days = 3  # 连续下跌天数
        self.max_drop_rate = -0.15  # 连续下跌最大跌幅（-15%以上才算超跌）
        self.take_profit_rate = 0.08  # 止盈比例8%
        self.stop_loss_rate = -0.05  # 止损比例-5%
        self.max_position_count = 8  # 最大持仓数量
        # 缓存：每只股票的涨跌幅历史
        self.pct_change_cache = {}
        # 记录每只股票的买入成本，用于止盈止损
        self.buy_price_map = {}

    def initialize(self):
        self.pct_change_cache = {}
        self.buy_price_map = {}
        logger.info(f"【{self.strategy_name}】初始化完成，连续下跌{self.continuous_down_days}天，超跌阈值{self.max_drop_rate*100}%")

    def generate_signal(self, trade_date: str, daily_df: pd.DataFrame, positions: dict):
        buy_stocks = []
        sell_signal_map = {}
        hold_ts_codes = list(positions.keys())

        # 1. 先处理持仓的止盈止损
        for ts_code in hold_ts_codes:
            stock_daily = daily_df[daily_df["ts_code"] == ts_code]
            if stock_daily.empty:
                continue
            current_price = stock_daily["close"].iloc[0]
            buy_price = self.buy_price_map.get(ts_code, current_price)
            # 计算涨跌幅
            pct_change = (current_price / buy_price) - 1

            # 止盈：达到止盈线，当日收盘卖出
            if pct_change >= self.take_profit_rate:
                sell_signal_map[ts_code] = "close"
                logger.info(f"[{trade_date}] {ts_code} 达到止盈线{self.take_profit_rate*100}%，生成收盘卖出信号")
                del self.buy_price_map[ts_code]
            # 止损：跌破止损线，次日开盘卖出（避免跌停卖不出）
            elif pct_change <= self.stop_loss_rate:
                sell_signal_map[ts_code] = "open"
                logger.info(f"[{trade_date}] {ts_code} 跌破止损线{self.stop_loss_rate*100}%，生成开盘卖出信号")
                del self.buy_price_map[ts_code]

        # 2. 筛选超跌股票
        # 基础过滤：流动性充足、非停牌、非一字跌停
        valid_df = daily_df[
            (daily_df["volume"] > 0) &
            (daily_df["amount"] > 5000000) &  # 成交额大于500万
            (daily_df["close"] > daily_df["low"])  # 非一字跌停，有成交机会
        ].copy()
        # 计算当日涨跌幅
        valid_df["pct_change"] = (valid_df["close"] / valid_df["pre_close"]) - 1

        # 3. 筛选连续下跌的超跌股
        for idx, row in valid_df.iterrows():
            ts_code = row["ts_code"]
            pct_change = row["pct_change"]
            # 已有持仓的跳过
            if ts_code in hold_ts_codes:
                continue
            # 初始化缓存
            if ts_code not in self.pct_change_cache:
                self.pct_change_cache[ts_code] = []
            # 补充当日涨跌幅
            self.pct_change_cache[ts_code].append(pct_change)
            # 数据不足的跳过
            if len(self.pct_change_cache[ts_code]) < self.continuous_down_days:
                continue

            # 取最近N天的涨跌幅
            recent_pct = self.pct_change_cache[ts_code][-self.continuous_down_days:]
            # 条件1：连续N天每天都下跌
            is_continuous_down = all(pct < 0 for pct in recent_pct)
            # 条件2：累计跌幅超过阈值
            total_drop = sum(recent_pct)
            is_oversold = total_drop <= self.max_drop_rate

            # 符合超跌条件，加入买入列表
            if is_continuous_down and is_oversold:
                buy_stocks.append(ts_code)
                # 记录买入成本（回测引擎按涨停价/开盘价成交，这里用当日收盘价预记录，成交后更新）
                self.buy_price_map[ts_code] = row["close"]
                # 达到最大持仓数停止
                if len(buy_stocks) >= self.max_position_count - len(hold_ts_codes):
                    break

        logger.info(f"[{trade_date}] 生成买入信号{len(buy_stocks)}只，卖出信号{len(sell_signal_map)}只")
        return buy_stocks, sell_signal_map

    # 【重写】买入成功后更新真实买入成本
    def on_buy_success(self, ts_code: str, buy_price: float):
        """回测引擎买入成功后会调用这个方法，更新真实成本"""
        self.buy_price_map[ts_code] = buy_price