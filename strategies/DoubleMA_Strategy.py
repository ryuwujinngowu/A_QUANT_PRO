import pandas as pd
from datetime import datetime, timedelta
from strategies.base_strategy import BaseStrategy
from utils.log_utils import logger
from features.ma_indicator import technical_features
from features.macd_indicator import macd_indicator

'''
（1）买入逻辑（三层筛选）
初筛：剔除停牌、成交额 < 1000 万、涨跌停、价格 < 1 元、已有持仓的标的；
双均线金叉：5 日均线上穿 20 日均线，且收盘价站上 20 日均线；
MACD 确认：MACD 金叉或红柱持续放大；
信号格式：字典格式（{"000001.SZ": "open"}），指定开盘价买入。
（2）卖出逻辑（三层风控，优先级从高到低）
固定止损：持仓浮亏 8%，次日开盘卖出（最高优先级，截断亏损）；
移动止盈：持仓浮盈 10% 后回撤 5%，当日收盘卖出（锁定利润，让利润奔跑）；
双死叉确认：均线死叉 + MACD 死叉，当日收盘卖出（趋势结束确认）；
信号格式：字典格式（{"000001.SZ": "open"/"close"}）。'''


class DoubleMA_Strategy(BaseStrategy):
    """
    双均线+MACD双确认趋势跟踪策略
    核心特点：
    1. 严格分层：策略层仅生成买卖信号，不访问account内部数据
    2. 双指标确认：双均线定方向，MACD过滤假突破
    3. 三层风控：固定止损+移动止盈+趋势反转卖出
    4. 完全适配：支持engine的字典格式信号，通过回调管理缓存
    """

    def __init__(self):
        """策略初始化：设置核心参数，初始化缓存容器"""
        super().__init__()
        self.strategy_name = "双均线MACD双确认趋势策略"

        # -------------------- 策略核心参数（可自由调整） --------------------
        self.short_ma_window = 5       # 短期均线窗口（5日线，捕捉短期趋势）
        self.long_ma_window = 20       # 长期均线窗口（20日线，捕捉中期趋势）
        self.max_position_count = 10    # 最大持仓数量（分散风险）
        self.fixed_stop_loss_rate = -0.08  # 固定止损比例（浮亏8%强制卖出）
        self.trailing_stop_profit_rate = 0.10  # 移动止盈触发阈值（浮盈10%启动）

        # -------------------- 策略层缓存容器（仅通过回调管理，不访问account） --------------------
        self.sell_signal_map = {}       # 卖出信号字典（key=股票代码，value=卖出类型open/close）
        self.buy_price_map = {}         # 持仓成本价缓存（仅通过on_buy_success更新）
        self.max_price_map = {}         # 持仓期间最高价缓存（用于移动止盈）

    def initialize(self):
        """
        策略初始化：回测开始前执行一次
        清空所有缓存，确保回测从零开始
        """
        self.sell_signal_map.clear()
        self.buy_price_map.clear()
        self.max_price_map.clear()
        logger.info(f"【{self.strategy_name}】初始化完成 | 参数：短期{self.short_ma_window}日，长期{self.long_ma_window}日，最大持仓{self.max_position_count}只")


    # -------------------- 策略核心方法：生成买卖信号 --------------------
    def generate_signal(self, trade_date: str, daily_df: pd.DataFrame, positions: dict, account_cash=None):
        """
        策略层核心职责：仅生成买卖信号，不访问account内部数据
        :param trade_date: 当前交易日（格式：YYYY-MM-DD）
        :param daily_df: 当日全市场日线数据
        :param positions: 当前持仓字典（仅取keys获取持仓代码列表，不访问内部结构）
        :param account_cash: 账户可用资金（可选，暂不使用）
        :return: (buy_signal_map: 买入信号字典, sell_signal_map: 卖出信号字典)
        """
        # 1. 初始化信号容器和辅助变量
        buy_signal_map = {}  # 买入信号字典（key=代码，value=买入类型）
        hold_ts_codes = list(positions.keys())  # 仅取持仓代码列表，不访问account内部结构
        trade_date_ymd = trade_date.replace("-", "")  # 日期格式转换：YYYY-MM-DD → YYYYMMDD
        # 计算历史数据拉取起始日期（往前推90天，确保均线、MACD计算有足够数据）
        history_start_dt = datetime.strptime(trade_date, "%Y-%m-%d") - timedelta(days=90)
        history_start_date = history_start_dt.strftime("%Y%m%d")

        # ========== 第一步：生成卖出信号（仅基于自己的缓存，不依赖account） ==========
        self.sell_signal_map.clear()  # 清空上一日的旧信号
        for ts_code in hold_ts_codes:
            # 仅处理自己有缓存的持仓（通过on_buy_success确认的真实持仓）
            if ts_code not in self.buy_price_map:
                continue

            # 获取该股票当日日线数据
            stock_daily = daily_df[daily_df["ts_code"] == ts_code]
            if stock_daily.empty:
                continue
            current_close = stock_daily["close"].iloc[0]

            # 从自己的缓存中获取成本价和最高价
            buy_price = self.buy_price_map[ts_code]
            # 更新持仓期间最高价（用于移动止盈）
            self.max_price_map[ts_code] = max(self.max_price_map[ts_code], current_close)
            max_hold_price = self.max_price_map[ts_code]

            # 1.1 固定止损（最高优先级，无条件执行）
            current_pct = (current_close / buy_price) - 1
            if current_pct <= self.fixed_stop_loss_rate:
                self.sell_signal_map[ts_code] = "open"  # "open"表示次日开盘卖出
                logger.info(f"[{trade_date}] {ts_code} 触发固定止损，浮亏{current_pct*100:.2f}%，生成开盘卖出信号")
                continue

            # 1.2 移动止盈（次高优先级，锁定利润）
            max_pct = (max_hold_price / buy_price) - 1
            current_retrace_pct = (max_hold_price - current_close) / max_hold_price
            # 触发条件：最高浮盈≥10% 且 从最高价回撤≥5%
            if max_pct >= self.trailing_stop_profit_rate and current_retrace_pct >= 0.05:
                self.sell_signal_map[ts_code] = "close"  # "close"表示当日收盘卖出
                logger.info(f"[{trade_date}] {ts_code} 触发移动止盈，最高浮盈{max_pct*100:.2f}%，回撤{current_retrace_pct*100:.2f}%，生成收盘卖出信号")
                continue

            # 1.3 均线死叉+MACD死叉双确认（最低优先级，趋势结束确认）
            # 调用MA轮子计算双均线
            ma_df = technical_features.calculate_ma(
                ts_code=ts_code,
                start_date=history_start_date,
                end_date=trade_date_ymd,
                ma_days=[self.short_ma_window, self.long_ma_window]
            )
            if len(ma_df) < 2:  # 数据不足，跳过判断
                continue
            ma_today = ma_df.iloc[-1]  # 当日均线数据
            ma_prev = ma_df.iloc[-2]    # 前一日均线数据
            # 严格死叉判定：前一日短均线≥长均线，当日短均线<长均线
            is_death_cross = (
                (ma_prev[f"ma{self.short_ma_window}"] >= ma_prev[f"ma{self.long_ma_window}"]) and
                (ma_today[f"ma{self.short_ma_window}"] < ma_today[f"ma{self.long_ma_window}"])
            )
            if is_death_cross:
                # 调用MACD轮子二次确认
                macd_df = macd_indicator.calculate_macd(
                    ts_code=ts_code,
                    start_date=history_start_date,
                    end_date=trade_date_ymd
                )
                if not macd_df.empty and macd_df.iloc[-1]["is_death_cross"]:
                    self.sell_signal_map[ts_code] = "close"
                    logger.info(f"[{trade_date}] {ts_code} 均线死叉+MACD死叉双确认，生成收盘卖出信号")

        # ========== 第二步：生成买入信号（字典格式，适配engine） ==========
        # 计算可用仓位（避免负数）
        available_position_count = max(0, self.max_position_count - len(hold_ts_codes))
        if available_position_count > 0:
            # 2.1 初筛有效标的（和engine的拦截逻辑对齐，提前过滤无效标的）
            valid_df = daily_df[
                (daily_df["volume"] > 0) &  # 非停牌（有成交量）
                (daily_df["amount"] > 10000000) &  # 流动性充足（成交额>1000万）
                (abs((daily_df["close"] / daily_df["pre_close"]) - 1) < 0.095) &  # 非涨跌停
                (daily_df["close"] > 1.0) &  # 非仙股（价格>1元）
                (~daily_df["ts_code"].isin(hold_ts_codes))  # 非已有持仓
            ].copy()

            if not valid_df.empty:
                # 2.2 遍历初筛后的标的，双指标确认买入
                for idx, row in valid_df.iterrows():
                    # 仓位已满，停止筛选
                    if len(buy_signal_map) >= available_position_count:
                        break
                    ts_code = row["ts_code"]

                    # 2.2.1 双均线金叉判断
                    ma_df = technical_features.calculate_ma(
                        ts_code=ts_code,
                        start_date=history_start_date,
                        end_date=trade_date_ymd,
                        ma_days=[self.short_ma_window, self.long_ma_window]
                    )
                    if len(ma_df) < 2:
                        continue
                    ma_today = ma_df.iloc[-1]
                    ma_prev = ma_df.iloc[-2]
                    # 严格金叉判定：前一日短均线≤长均线，当日短均线>长均线
                    is_golden_cross = (
                        (ma_prev[f"ma{self.short_ma_window}"] <= ma_prev[f"ma{self.long_ma_window}"]) and
                        (ma_today[f"ma{self.short_ma_window}"] > ma_today[f"ma{self.long_ma_window}"])
                    )
                    # 额外确认：收盘价站上长期均线
                    is_close_above_long_ma = row["close"] > ma_today[f"ma{self.long_ma_window}"]
                    if not (is_golden_cross and is_close_above_long_ma):
                        continue

                    # 2.2.2 MACD二次确认（过滤假突破）
                    macd_df = macd_indicator.calculate_macd(
                        ts_code=ts_code,
                        start_date=history_start_date,
                        end_date=trade_date_ymd
                    )
                    if macd_df.empty:
                        continue
                    macd_today = macd_df.iloc[-1]
                    # MACD确认条件：当日MACD金叉 或 MACD红柱放大
                    is_macd_confirm = (
                        macd_today["is_golden_cross"] or
                        (macd_today["macd_bar"] > 0 and macd_today["macd_bar"] > macd_df.iloc[-2]["macd_bar"])
                    )
                    if is_macd_confirm:
                        # 生成买入信号：字典格式，key=代码，value=买入类型（"open"表示开盘价买入）
                        buy_signal_map[ts_code] = "open"
                        logger.info(f"[{trade_date}] {ts_code} 双信号确认，生成买入信号（字典格式，类型：open）")

        # 始终返回买入字典和卖出字典，不做复杂判断
        return buy_signal_map, self.sell_signal_map

    # -------------------- 策略回调方法（engine执行买卖后调用，管理缓存） --------------------
    def on_buy_success(self, ts_code: str, buy_price: float):
        """
        【engine回调】仅当engine确认买入成功后执行
        更新持仓成本价和最高价缓存
        :param ts_code: 买入成功的股票代码
        :param buy_price: 实际成交价格
        """
        self.buy_price_map[ts_code] = buy_price
        self.max_price_map[ts_code] = buy_price
        logger.info(f"[{self.strategy_name}] {ts_code} 买入成功，更新缓存 | 成本价：{buy_price}")

    def on_sell_success(self, ts_code: str):
        """
        【engine回调】仅当engine确认卖出成功后执行
        清理持仓缓存
        :param ts_code: 卖出成功的股票代码
        """
        if ts_code in self.buy_price_map:
            del self.buy_price_map[ts_code]
        if ts_code in self.max_price_map:
            del self.max_price_map[ts_code]
        logger.info(f"[{self.strategy_name}] {ts_code} 卖出成功，清理缓存")

    def on_buy_failed(self, ts_code: str, reason: str):
        """
        【engine回调】买入失败时执行
        记录失败原因，便于排查
        :param ts_code: 买入失败的股票代码
        :param reason: 失败原因
        """
        logger.error(f"[{self.strategy_name}] {ts_code} 买入失败 | 原因：{reason}")

    def on_reset(self):
        """
        【可选】回测重置时执行
        清空所有缓存
        """
        self.sell_signal_map.clear()
        self.buy_price_map.clear()
        self.max_price_map.clear()
        logger.info(f"[{self.strategy_name}] 策略重置，所有缓存已清空")