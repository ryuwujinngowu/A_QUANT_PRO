import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
from config.config import MAX_POSITION_COUNT
from utils.common_tools import get_trade_dates, get_daily_kline_data
from backtest.account import Account
from backtest.metrics import BacktestMetrics
from data.data_cleaner import data_cleaner
from data.data_fetcher import data_fetcher
from strategies.base_strategy import BaseStrategy
from utils.db_utils import db
from utils.log_utils import logger


class MultiStockBacktestEngine:
    """全市场多标的回测引擎（V2.0 买入信号格式升级+多价格类型支持）"""

    def __init__(
            self,
            strategy: BaseStrategy,
            init_capital: float,
            start_date: str,
            end_date: str
    ):
        self.strategy = strategy
        self.init_capital = init_capital
        self.start_date = start_date
        self.end_date = end_date
        self.account = Account(init_capital=init_capital, max_position_count=MAX_POSITION_COUNT)
        self.result = {}
        self.account.set_backtest_info(
            strategy_name=self.strategy.strategy_name,
            start_date=str(self.start_date),
            end_date=str(self.end_date)
        )
        # ========== 【新增】支持的买入类型定义 ==========
        self.SUPPORT_BUY_TYPES = ["open", "close", "limit_up", "limit_down", "custom"]
        self.DEFAULT_BUY_TYPE = "limit_up"  # 兼容旧策略的默认类型

    def run(self) -> dict:
        """执行回测核心流程"""
        logger.info(
            f"===== 开始回测，初始本金：{self.init_capital}元，回测时间段：{self.start_date} 至 {self.end_date} =====")
        self.strategy.initialize()
        trade_dates = get_trade_dates(self.start_date, self.end_date)
        for idx, trade_date in enumerate(trade_dates):
            logger.info(f"===== 处理交易日：{trade_date}（第{idx + 1}/{len(trade_dates)}天） =====")
            # 1. 获取当日全市场日线数据（包含open/high/low/close/pre_close，用于一字板判断）
            daily_df = get_daily_kline_data(trade_date)
            if daily_df.empty:
                logger.warning(f"{trade_date} 无有效日线数据，跳过当日")
                continue

            # ========== 开盘卖出信号执行（原有逻辑完全不变） ==========
            open_sell_ts_codes = []
            for ts_code, sell_type in list(self.strategy.sell_signal_map.items()):
                if sell_type == "open":
                    # 1.1 先获取该股票当日数据
                    stock_df = daily_df[daily_df["ts_code"] == ts_code]
                    if stock_df.empty:
                        logger.warning(f"{trade_date} 开盘卖出：{ts_code} 无当日日线数据，跳过")
                        continue

                    # 1.2 【核心拦截】判断当日是否一字跌停，无法卖出
                    pre_close = stock_df["pre_close"].iloc[0]
                    limit_down_price = self.strategy.calc_limit_down_price(ts_code, pre_close)
                    open_price = stock_df["open"].iloc[0]
                    high_price = stock_df["high"].iloc[0]

                    # 一字跌停判断：全天封死跌停，无法卖出
                    is_limit_down_lock = (open_price <= limit_down_price + 0.001) and (
                            high_price <= limit_down_price + 0.001)
                    if is_limit_down_lock:
                        logger.warning(
                            f"{trade_date} 开盘卖出拦截：{ts_code} 当日一字跌停，无法卖出，信号保留至下一交易日")
                        continue  # 不删除信号，保留到下一个交易日继续尝试

                    # 1.3 非一字跌停，正常执行卖出
                    if self.account.sell(trade_date=trade_date, ts_code=ts_code, price=open_price):
                        open_sell_ts_codes.append(ts_code)
                        # ========== 【新增】触发卖出成功回调（和买入对齐） ==========
                        if hasattr(self.strategy, "on_sell_success"):
                            self.strategy.on_sell_success(ts_code)

            # 只删除已成功执行的open信号，未执行的信号保留
            for ts_code in open_sell_ts_codes:
                if ts_code in self.strategy.sell_signal_map:
                    del self.strategy.sell_signal_map[ts_code]

            # ========== 【核心改动1：第一次生成信号，兼容新旧两种买入格式】 ==========
            buy_signal, _ = self.strategy.generate_signal(
                trade_date=trade_date,
                daily_df=daily_df,
                positions=self.account.positions
            )

            # ========== 【核心改动2：解析买入信号，兼容列表/字典两种格式】 ==========
            buy_signal_map = {}
            # 兼容旧格式：列表 → 转为默认字典格式
            if isinstance(buy_signal, list):
                for ts_code in buy_signal:
                    buy_signal_map[ts_code] = self.DEFAULT_BUY_TYPE
                logger.info(f"[{trade_date}] 兼容旧策略列表格式买入信号，共{len(buy_signal_map)}只标的")
            # 新格式：字典 → 直接使用
            elif isinstance(buy_signal, dict):
                buy_signal_map = buy_signal
            # 非法格式：跳过买入
            else:
                logger.error(f"[{trade_date}] 买入信号格式非法，仅支持列表/字典，跳过当日买入")
                buy_signal_map = {}

            # ========== 【核心改动3：买入执行环节，按买入类型动态取价】 ==========
            available_count = self.account.get_available_position_count()
            if available_count > 0 and buy_signal_map:
                # 按可用仓位截取标的，保留字典顺序
                buy_exec_list = list(buy_signal_map.items())[:available_count]
                logger.info(
                    f"[{trade_date}] 开始执行买入操作 | 可用仓位：{available_count} | 买入标的：{[code for code, _ in buy_exec_list]}")

                for ts_code, buy_type in buy_exec_list:
                    # 3.1 获取该股票当日基础数据
                    stock_df = daily_df[daily_df["ts_code"] == ts_code]
                    if stock_df.empty:
                        logger.warning(f"{trade_date} 买入操作：{ts_code} 无当日日线数据，跳过")
                        if hasattr(self.strategy, "on_buy_failed"):
                            self.strategy.on_buy_failed(ts_code, "无当日日线数据")
                        continue

                    # 3.2 计算涨跌停价格（通用基础数据）
                    pre_close = stock_df["pre_close"].iloc[0]
                    limit_up_price = self.strategy.calc_limit_up_price(ts_code, pre_close)
                    limit_down_price = self.strategy.calc_limit_down_price(ts_code, pre_close)
                    open_price = stock_df["open"].iloc[0]
                    close_price = stock_df["close"].iloc[0]
                    low_price = stock_df["low"].iloc[0]
                    high_price = stock_df["high"].iloc[0]

                    # 3.3 【通用拦截】一字涨跌停无法成交，直接跳过（实盘无法买入）
                    is_limit_up_lock = (open_price >= limit_up_price - 0.001) and (low_price >= limit_up_price - 0.001)
                    is_limit_down_lock = (open_price <= limit_down_price + 0.001) and (
                                high_price <= limit_down_price + 0.001)
                    if is_limit_up_lock or is_limit_down_lock:
                        lock_type = "一字涨停" if is_limit_up_lock else "一字跌停"
                        logger.warning(f"{trade_date} 买入拦截：{ts_code} 当日{lock_type}，无法买入，跳过")
                        if hasattr(self.strategy, "on_buy_failed"):
                            self.strategy.on_buy_failed(ts_code, f"当日{lock_type}")
                        continue

                    # 3.4 【核心改动4：根据买入类型，动态计算成交价格】
                    buy_type = buy_type.lower()
                    if buy_type not in self.SUPPORT_BUY_TYPES:
                        logger.warning(
                            f"{trade_date} {ts_code} 不支持的买入类型{buy_type}，使用默认类型{self.DEFAULT_BUY_TYPE}")
                        buy_type = self.DEFAULT_BUY_TYPE

                    # 按类型取价
                    if buy_type == "open":
                        exec_price = open_price
                    elif buy_type == "close":
                        exec_price = close_price
                    elif buy_type == "limit_up":
                        exec_price = limit_up_price
                    elif buy_type == "limit_down":
                        exec_price = limit_down_price
                    elif buy_type == "custom":
                        # 预留自定义价格扩展：从买入信号中取自定义价格，需策略配合传入
                        exec_price = buy_signal_map.get(f"{ts_code}_custom_price", limit_up_price)
                        logger.warning(f"{trade_date} {ts_code} 使用自定义价格买入，价格：{exec_price}")
                    else:
                        exec_price = limit_up_price

                    # 3.5 价格合法性校验
                    if pre_close <= 0 or exec_price <= 0:
                        logger.warning(
                            f"{trade_date} 买入操作：{ts_code} 价格无效（pre_close={pre_close}，exec_price={exec_price}），跳过")
                        if hasattr(self.strategy, "on_buy_failed"):
                            self.strategy.on_buy_failed(ts_code, "价格无效")
                        continue

                    # 3.6 执行买入
                    buy_success = self.account.buy(
                        trade_date=trade_date,
                        ts_code=ts_code,
                        price=exec_price
                    )
                    if buy_success:
                        logger.info(f"[{trade_date}] {ts_code} 买入成功 | 类型：{buy_type} | 成交价格：{exec_price}")
                        # 触发买入成功回调（原有逻辑保留）
                        if hasattr(self.strategy, "on_buy_success"):
                            self.strategy.on_buy_success(ts_code, exec_price)
                    else:
                        logger.warning(f"[{trade_date}] {ts_code} 买入失败，可用资金不足或仓位已满")
                        if hasattr(self.strategy, "on_buy_failed"):
                            self.strategy.on_buy_failed(ts_code, "可用资金不足或仓位已满")
            else:
                logger.info(f"{trade_date} 无可用仓位或无有效买入信号，跳过买入操作")

            # ========== 第二次生成信号，获取卖出信号（原有逻辑完全不变） ==========
            _, sell_signal_map = self.strategy.generate_signal(
                trade_date=trade_date,
                daily_df=daily_df,
                positions=self.account.positions
            )

            # 合并上一日未执行的信号 + 当日新生成的信号（原有逻辑完全不变）
            for ts_code, sell_type in self.strategy.sell_signal_map.items():
                if ts_code not in sell_signal_map:
                    sell_signal_map[ts_code] = sell_type
            self.strategy.sell_signal_map = sell_signal_map

            # ========== 收盘卖出信号执行（原有逻辑完全不变） ==========
            close_sell_ts_codes = []
            for ts_code, sell_type in sell_signal_map.items():
                if sell_type == "close":
                    # 5.1 获取该股票当日数据
                    stock_df = daily_df[daily_df["ts_code"] == ts_code]
                    if stock_df.empty:
                        logger.warning(f"{trade_date} 收盘卖出：{ts_code} 无当日日线数据，跳过")
                        continue

                    # 5.2 【核心拦截】判断当日是否一字跌停，无法卖出
                    pre_close = stock_df["pre_close"].iloc[0]
                    limit_down_price = self.strategy.calc_limit_down_price(ts_code, pre_close)
                    open_price = stock_df["open"].iloc[0]
                    high_price = stock_df["high"].iloc[0]

                    is_limit_down_lock = (open_price <= limit_down_price + 0.001) and (
                            high_price <= limit_down_price + 0.001)
                    if is_limit_down_lock:
                        logger.warning(
                            f"{trade_date} 收盘卖出拦截：{ts_code} 当日一字跌停，无法卖出，信号保留至下一交易日")
                        continue  # 不删除信号，保留到下一个交易日

                    # 5.3 非一字跌停，正常执行卖出
                    close_price = stock_df["close"].iloc[0]
                    if self.account.sell(trade_date=trade_date, ts_code=ts_code, price=close_price):
                        close_sell_ts_codes.append(ts_code)
                        # 触发卖出成功回调
                        if hasattr(self.strategy, "on_sell_success"):
                            self.strategy.on_sell_success(ts_code)

            # 只删除已成功执行的close信号，未执行的信号保留
            for ts_code in close_sell_ts_codes:
                if ts_code in self.strategy.sell_signal_map:
                    del self.strategy.sell_signal_map[ts_code]

            # 每日收盘更新账户资产（原有逻辑完全不变）
            self.account.update_daily_asset(trade_date=trade_date, daily_price_df=daily_df)

        # 回测结束，强制清仓剩余持仓（原有逻辑完全不变）
        logger.info("===== 回测结束，强制清仓剩余持仓 =====")
        last_date = trade_dates[-1]
        last_daily_df = get_daily_kline_data(last_date)
        hold_stocks = list(self.account.positions.keys())
        for ts_code in hold_stocks:
            try:
                if ts_code not in self.account.positions:
                    continue
                stock_df = last_daily_df[last_daily_df["ts_code"] == ts_code]
                if stock_df.empty:
                    logger.warning(f"{last_date} 强制清仓：{ts_code} 无当日日线数据，跳过清仓")
                    continue
                close_price = stock_df["close"].iloc[0]
                if self.account.sell(trade_date=last_date, ts_code=ts_code, price=close_price):
                    if hasattr(self.strategy, "on_sell_success"):
                        self.strategy.on_sell_success(ts_code)
                logger.info(f"{last_date} 强制清仓：{ts_code} 卖出成功，价格：{close_price}")
            except KeyError:
                logger.warning(f"无当日日线数据,检查持仓存在停牌股票")
        self.account.update_daily_asset(trade_date=last_date, daily_price_df=last_daily_df)

        # 计算回测指标（原有逻辑完全不变）
        net_value_df = self.account.get_net_value_df()
        trade_df = self.account.get_trade_df()
        metrics = BacktestMetrics(
            net_value_df=net_value_df,
            init_capital=self.init_capital,
            trade_df=trade_df,
            strategy_name=self.strategy.strategy_name,
            backtest_start_date=self.start_date,
            backtest_end_date=self.end_date
        )
        self.result = metrics.calc_all_metrics()

        # 输出结果（原有逻辑完全不变）
        logger.info("=" * 60)
        logger.info("回测完成，核心指标汇总：")
        for k, v in self.result.items():
            logger.info(f"{k}：{v}")
        logger.info("=" * 60)

        # 新增：打印单标的盈亏排名（原有逻辑完全不变）
        profit_top, loss_top = self.account.get_stock_pnl_ranking()
        if not profit_top.empty:
            logger.info("=" * 60)
            logger.info("【盈利最多的Top10股票】")
            logger.info("=" * 60)
            for idx, row in profit_top.iterrows():
                logger.info(f"  {idx + 1}. {row['ts_code']}：累计净盈亏 {round(row['累计净盈亏'], 2)} 元")
        if not loss_top.empty:
            logger.info("=" * 60)
            logger.info("【亏损最多的Top10股票】")
            logger.info("=" * 60)
            for idx, row in loss_top.iterrows():
                logger.info(f"  {idx + 1}. {row['ts_code']}：累计净盈亏 {round(row['累计净盈亏'], 2)} 元")
        logger.info("=" * 60)

        # 附加详细数据（原有逻辑完全不变）
        self.result["net_value_df"] = net_value_df
        self.result["trade_df"] = trade_df
        return self.result