import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datetime import datetime
from typing import Dict, Optional, Tuple
import pandas as pd
from collections import defaultdict

from config.config import (
    COMMISSION_RATE, STAMP_DUTY_RATE, SLIPPAGE_RATE,
    T_PLUS_1, MIN_TRADE_VOLUME, MAX_POSITION_COUNT
)
from utils.log_utils import logger


class Position:
    """单只股票持仓类（完全贴合A股合并持仓+加权平均成本规则）"""

    def __init__(self, ts_code: str):
        self.ts_code = ts_code
        # 核心持仓属性（A股合并持仓规则）
        self.total_volume: int = 0  # 总持仓数量
        self.available_volume: int = 0  # 当日可卖数量（=总持仓-当日买入数量）
        self.avg_cost: float = 0.0  # 加权平均持仓成本（A股官方规则）
        self.total_cost: float = 0.0  # 总持仓成本（不含卖出手续费）

        # T+1规则专用属性
        self.today_buy_volume: int = 0  # 当日买入的数量（当日不可卖，收盘后清零）

        # ========== 【新增：完整兼容原有策略的所有属性】 ==========
        self.first_buy_date: str = ""  # 最早的买入日期
        self.has_today_bought: bool = False  # 是否有当日买入的仓位
        self.hold_days: int = 0  # 【新增】持仓天数（兼容原有pos.hold_days）
        self.buy_price: float = 0.0  # 【新增】兼容原有pos.buy_price，返回平均成本价
        self.buy_volume: int = 0  # 【新增】兼容原有pos.buy_volume，返回总持仓数量
        self.can_sell: bool = False  # 【新增】兼容原有pos.can_sell

    def buy(self, buy_volume: int, buy_price: float, buy_total_cost: float, buy_date: str = ""):
        """
        执行买入（支持加仓，自动更新加权平均成本）
        :param buy_volume: 买入数量
        :param buy_price: 买入成交价（含滑点）
        :param buy_total_cost: 买入总成本（含手续费）
        :param buy_date: 【新增兼容参数】买入日期
        """
        # 1. 更新总持仓和总成本
        self.total_volume += buy_volume
        self.total_cost += buy_total_cost

        # 2. 重新计算加权平均成本（A股官方规则）
        if self.total_volume > 0:
            self.avg_cost = self.total_cost / self.total_volume

        # 3. 更新当日买入数量（T+1不可卖）
        self.today_buy_volume += buy_volume

        # 4. 更新可卖数量（当日买入的不可卖）
        self.available_volume = self.total_volume - self.today_buy_volume

        # ========== 【新增：兼容逻辑】 ==========
        if buy_date:
            formatted_date = self._unify_date_format(buy_date)
            if not self.first_buy_date or formatted_date < self.first_buy_date:
                self.first_buy_date = formatted_date
        self.has_today_bought = True
        # 同步兼容属性
        self.buy_price = self.avg_cost
        self.buy_volume = self.total_volume
        self.can_sell = self.available_volume > 0

    # ========== 【新增：完整兼容原有属性访问】 ==========
    @property
    def buy_date(self) -> str:
        """兼容原有策略的pos.buy_date调用，返回最早的买入日期"""
        return self.first_buy_date

    def sell(self, sell_volume: int, sell_price: float, sell_total_income: float) -> Optional[float]:
        """
        执行卖出（支持部分卖出/做T，无顺序限制，只要不超过可卖数量）
        :param sell_volume: 卖出数量
        :param sell_price: 卖出成交价（含滑点）
        :param sell_total_income: 卖出净收入（扣除手续费、印花税）
        :return: 本次卖出的净盈亏，失败返回None
        """
        # 1. 校验可卖数量
        if sell_volume > self.available_volume:
            logger.warning(
                f"[{self.ts_code}] 卖出失败：可卖数量不足，请求卖出{sell_volume}，实际可卖{self.available_volume}")
            return None
        if sell_volume < MIN_TRADE_VOLUME and sell_volume != self.available_volume:
            logger.warning(f"[{self.ts_code}] 卖出失败：数量不足1手，且不是清仓")
            return None

        # 2. 计算本次卖出的净盈亏（按加权平均成本）
        sell_cost = self.avg_cost * sell_volume  # 卖出部分对应的持仓成本
        sell_pnl = sell_total_income - sell_cost  # 本次卖出净盈亏

        # 3. 更新总持仓和总成本
        self.total_volume -= sell_volume
        self.total_cost = self.avg_cost * self.total_volume  # 剩余持仓的总成本

        # 4. 更新可卖数量
        self.available_volume -= sell_volume

        # 5. 清仓时重置成本
        if self.total_volume == 0:
            self.avg_cost = 0.0
            self.total_cost = 0.0
            self.first_buy_date = ""  # 清仓时重置最早买入日期
            self.has_today_bought = False  # 清仓时重置当日买入标记
            self.hold_days = 0  # 清仓时重置持仓天数
            self.buy_price = 0.0
            self.buy_volume = 0
            self.can_sell = False
        else:
            # 同步兼容属性
            self.buy_price = self.avg_cost
            self.buy_volume = self.total_volume
            self.can_sell = self.available_volume > 0

        logger.debug(
            f"[{self.ts_code}] 卖出成功：数量{sell_volume}，净盈亏{round(sell_pnl, 2)}元，剩余持仓{self.total_volume}股")
        return sell_pnl

    def daily_settle(self):
        """
        每日收盘后结算（T+1规则核心：当日买入的持仓，次日可卖）
        必须在每日收盘后调用
        """
        # 1. 当日买入的持仓，次日可卖，加到可卖数量里
        self.available_volume += self.today_buy_volume
        # 2. 清零当日买入数量
        self.today_buy_volume = 0
        # 3. 安全校验：可卖数量不能超过总持仓
        self.available_volume = min(self.available_volume, self.total_volume)
        # ========== 【新增：兼容逻辑】 ==========
        self.has_today_bought = False  # 收盘后重置当日买入标记
        # 更新持仓天数（和原有逻辑一致：有持仓且不是当日买入的才+1）
        if self.total_volume > 0 and self.first_buy_date:
            self.hold_days += 1
        # 同步兼容属性
        self.can_sell = self.available_volume > 0

        logger.debug(f"[{self.ts_code}] 收盘结算完成：总持仓{self.total_volume}，可卖数量{self.available_volume}，持仓天数{self.hold_days}")

    @staticmethod
    def _unify_date_format(date_str: str) -> str:
        """统一日期格式为YYYYMMDD"""
        try:
            return datetime.strptime(date_str.replace("-", ""), "%Y%m%d").strftime("%Y%m%d")
        except Exception as e:
            logger.error(f"日期格式转换失败：{date_str}，错误：{e}")
            return date_str


class Account:
    """多标的分仓账户管理类（完全贴合A股规则，支持加仓/做T/单标的盈亏统计）"""

    def __init__(self, init_capital: float, max_position_count: int = MAX_POSITION_COUNT):
        # 账户核心资产
        self.init_capital = init_capital
        self.available_cash = init_capital  # 可用资金
        self.total_asset = init_capital  # 总资产=可用资金+持仓总市值

        # 分仓配置
        self.max_position_count = max_position_count
        self.per_position_cash = init_capital / max_position_count  # 单份仓位固定资金

        # 持仓管理：key=股票代码，value=Position对象（合并持仓）
        self.positions: Dict[str, Position] = {}

        # 交易记录与净值曲线
        self.trade_history = []
        self.daily_net_value = []

        # 回测信息
        self.strategy_name = "未命名策略"
        self.backtest_start_date = ""
        self.backtest_end_date = ""

        # 盈亏计算辅助
        self.prev_total_asset = init_capital
        self.daily_sold_pnl: Dict[str, Dict[str, float]] = defaultdict(dict)

        # 单标的总盈亏统计（回测结束后看个股贡献）
        self.stock_total_pnl: Dict[str, float] = defaultdict(float)

    def set_backtest_info(self, strategy_name: str, start_date: str, end_date: str):
        """设置回测核心信息（回测引擎初始化时调用）"""
        self.strategy_name = strategy_name
        self.backtest_start_date = self._unify_date_format(start_date)
        self.backtest_end_date = self._unify_date_format(end_date)
        logger.info(f"回测信息已设置：策略名称={strategy_name}，回测时间段={start_date} 至 {end_date}")

    def update_daily_asset(self, trade_date: str, daily_price_df: pd.DataFrame):
        """
        每日收盘后必须调用：更新账户资产、持仓结算、打印盈亏日志
        【核心】先结算持仓T+1状态，再更新资产
        """
        # 1. 所有持仓每日收盘结算（T+1规则：当日买入的次日可卖）
        for position in self.positions.values():
            position.daily_settle()

        # 2. 计算持仓总市值和盈亏明细
        total_position_value = 0.0
        position_pnl_detail = {}

        for ts_code, position in self.positions.items():
            # 获取当日收盘价，无数据则用持仓成本价
            stock_df = daily_price_df[daily_price_df["ts_code"] == ts_code]
            close_price = stock_df["close"].iloc[0] if not stock_df.empty else position.avg_cost
            position_value = position.total_volume * close_price
            total_position_value += position_value

            # 计算持仓浮动盈亏
            if position.avg_cost > 0 and position.total_volume > 0:
                hold_pnl = (close_price - position.avg_cost) * position.total_volume
                hold_pnl_rate = (close_price - position.avg_cost) / position.avg_cost * 100
            else:
                hold_pnl = 0.0
                hold_pnl_rate = 0.0

            position_pnl_detail[ts_code] = {
                "持仓数量": position.total_volume,
                "可卖数量": position.available_volume,
                "平均成本价": round(position.avg_cost, 4),
                "当日收盘价": round(close_price, 4),
                "当日浮动盈亏": round(hold_pnl, 2),
                "持仓累计收益率(%)": round(hold_pnl_rate, 2)
            }

        # 3. 更新总资产
        self.total_asset = self.available_cash + total_position_value

        # 4. 计算当日整体盈亏
        daily_pnl = self.total_asset - self.prev_total_asset
        daily_pnl_rate = daily_pnl / self.prev_total_asset * 100 if self.prev_total_asset > 0 else 0
        total_pnl = self.total_asset - self.init_capital
        total_pnl_rate = total_pnl / self.init_capital * 100 if self.init_capital > 0 else 0

        # 5. 记录每日净值（完全兼容原有逻辑）
        self.daily_net_value.append({
            "策略名称": self.strategy_name,
            "回测开始日期": self.backtest_start_date,
            "回测结束日期": self.backtest_end_date,
            "初始资金": round(self.init_capital, 2),
            "trade_date": trade_date,
            "total_asset": round(self.total_asset, 2),
            "available_cash": round(self.available_cash, 2),
            "position_count": len(self.positions),
            "total_position_value": round(total_position_value, 2),
            "交易日": trade_date,
            "总资产": round(self.total_asset, 2),
            "可用资金": round(self.available_cash, 2),
            "持仓总市值": round(total_position_value, 2),
            "持仓数量": len(self.positions),
            "当日盈亏": round(daily_pnl, 2),
            "当日收益率(%)": round(daily_pnl_rate, 2),
            "累计盈亏": round(total_pnl, 2),
            "累计收益率(%)": round(total_pnl_rate, 2)
        })

        # 6. 结构化打印当日盈亏日志
        logger.info(f"{'=' * 40}")
        logger.info(f"【{trade_date} 每日结算盈亏报告】")
        logger.info(f"{'=' * 40}")
        logger.info(f"📊 账户整体情况：")
        logger.info(f"  当日盈亏：{round(daily_pnl, 2)} 元 | 当日收益率：{round(daily_pnl_rate, 2)} %")
        logger.info(f"  累计盈亏：{round(total_pnl, 2)} 元 | 累计收益率：{round(total_pnl_rate, 2)} %")
        logger.info(f"  账户总资产：{round(self.total_asset, 2)} 元 | 可用资金：{round(self.available_cash, 2)} 元")
        logger.info(f"  持仓总市值：{round(total_position_value, 2)} 元 | 持仓标的数量：{len(self.positions)} 只")

        logger.info(f"📈 持仓标的盈亏明细：")
        if position_pnl_detail:
            for ts_code, pnl_info in position_pnl_detail.items():
                logger.info(
                    f"  {ts_code}：持仓{pnl_info['持仓数量']}股 | 可卖{pnl_info['可卖数量']}股 | 平均成本价{pnl_info['平均成本价']} | 收盘价{pnl_info['当日收盘价']} | 当日盈亏{pnl_info['当日浮动盈亏']}元 | 累计收益率{pnl_info['持仓累计收益率(%)']}%"
                )
        else:
            logger.info(f"  当日无持仓")

        logger.info(f"💸 当日卖出标的盈亏明细：")
        current_sold_pnl = self.daily_sold_pnl.get(trade_date, {})
        if current_sold_pnl:
            total_sold_pnl = sum(current_sold_pnl.values())
            for ts_code, pnl in current_sold_pnl.items():
                logger.info(f"  {ts_code}：卖出净盈亏 {round(pnl, 2)} 元")
            logger.info(f"  👉 当日卖出总盈亏：{round(total_sold_pnl, 2)} 元")
        else:
            logger.info(f"  当日无卖出操作")
        logger.info(f"{'=' * 60}")

        # 7. 更新前一日总资产，清空当日卖出盈亏
        self.prev_total_asset = self.total_asset
        self.daily_sold_pnl.pop(trade_date, None)

    def get_available_position_count(self) -> int:
        """获取剩余可开仓的仓位数量（加仓不占用新仓位）"""
        return self.max_position_count - len(self.positions)

    def buy(self, trade_date: str, ts_code: str, price: float, volume: Optional[int] = None) -> bool:
        """
        执行买入操作（支持加仓，100%向后兼容原有接口）
        :param trade_date: 交易日
        :param ts_code: 股票代码
        :param price: 买入基准价（如涨停价）
        :param volume: 【可选】买入数量，不传则按单份仓位自动计算
        :return: 是否买入成功
        """
        # 1. 新票校验：检查可用仓位
        if ts_code not in self.positions:
            if self.get_available_position_count() <= 0:
                logger.warning(f"[{trade_date}] {ts_code} 买入失败：无可用仓位")
                return False

        # 2. 滑点处理：买入价上浮
        actual_price = price * (1 + SLIPPAGE_RATE)

        # 3. 计算买入数量
        if volume is None:
            # 原有逻辑：按单份仓位计算
            max_can_buy = int(self.per_position_cash / (actual_price * MIN_TRADE_VOLUME)) * MIN_TRADE_VOLUME
        else:
            # 加仓场景：指定数量买入
            max_can_buy = volume

        if max_can_buy < MIN_TRADE_VOLUME:
            logger.warning(f"[{trade_date}] {ts_code} 买入失败：数量不足1手")
            return False

        # 4. 计算手续费和总成本
        commission = max(max_can_buy * actual_price * COMMISSION_RATE, 5)
        total_cost = max_can_buy * actual_price + commission

        if total_cost > self.available_cash:
            logger.warning(f"[{trade_date}] {ts_code} 买入失败：可用资金不足")
            return False

        # 5. 更新持仓
        self.available_cash -= total_cost
        if ts_code not in self.positions:
            self.positions[ts_code] = Position(ts_code=ts_code)

        # 执行买入，自动更新加权平均成本，【新增传递buy_date参数】
        self.positions[ts_code].buy(
            buy_volume=max_can_buy,
            buy_price=actual_price,
            buy_total_cost=total_cost,
            buy_date=trade_date  # 新增：传递买入日期
        )

        # 6. 记录交易（完全兼容原有格式）
        self.trade_history.append({
            "trade_date": trade_date,
            "ts_code": ts_code,
            "direction": "买入",
            "price": round(actual_price, 4),
            "volume": max_can_buy,
            "commission": round(commission, 2),
            "stamp_duty": 0,
            "total_cost": round(total_cost, 2)
        })
        logger.info(f"[{trade_date}] {ts_code} 买入成功，价格：{round(actual_price, 4)}，数量：{max_can_buy}")
        return True

    def sell(self, trade_date: str, ts_code: str, price: float, volume: Optional[int] = None) -> bool:
        """
        执行卖出操作（支持部分卖出/做T，100%向后兼容原有接口）
        :param trade_date: 交易日
        :param ts_code: 股票代码
        :param price: 卖出基准价（如开盘价/收盘价）
        :param volume: 【可选】卖出数量，不传则全卖可卖仓位
        :return: 是否卖出成功
        """
        if ts_code not in self.positions:
            logger.warning(f"[{trade_date}] {ts_code} 卖出失败：无该持仓")
            return False

        position = self.positions[ts_code]
        sellable_vol = position.available_volume

        # 确定卖出数量
        if volume is None:
            # 原有逻辑：全卖可卖仓位
            sell_vol = sellable_vol
        else:
            # 做T场景：指定数量卖出
            sell_vol = min(volume, sellable_vol)

        if sell_vol <= 0:
            logger.warning(f"[{trade_date}] {ts_code} 卖出失败：无可卖仓位（T+1规则）")
            return False

        # 2. 滑点处理：卖出价下浮
        actual_price = price * (1 - SLIPPAGE_RATE)

        # 3. 计算手续费、印花税和净收入
        commission = max(sell_vol * actual_price * COMMISSION_RATE, 5)
        stamp_duty = sell_vol * actual_price * STAMP_DUTY_RATE
        total_income = sell_vol * actual_price - commission - stamp_duty

        # 4. 执行卖出，获取本次盈亏
        sell_pnl = position.sell(
            sell_volume=sell_vol,
            sell_price=actual_price,
            sell_total_income=total_income
        )
        if sell_pnl is None:
            return False

        # 5. 更新账户
        self.available_cash += total_income
        self.daily_sold_pnl[trade_date][ts_code] = sell_pnl
        self.stock_total_pnl[ts_code] += sell_pnl  # 累计单标的盈亏

        # 清仓后删除持仓
        if position.total_volume <= 0:
            del self.positions[ts_code]

        # 6. 记录交易（完全兼容原有格式）
        self.trade_history.append({
            "trade_date": trade_date,
            "ts_code": ts_code,
            "direction": "卖出",
            "price": round(actual_price, 4),
            "volume": sell_vol,
            "commission": round(commission, 2),
            "stamp_duty": round(stamp_duty, 2),
            "total_income": round(total_income, 2),
            "卖出净盈亏": round(sell_pnl, 2)
        })
        logger.info(
            f"[{trade_date}] {ts_code} 卖出成功，价格：{round(actual_price, 4)}，数量：{sell_vol}，净盈亏：{round(sell_pnl, 2)}元")
        return True

    # ========== 单标的盈亏排名功能（原有功能保留） ==========
    def get_stock_pnl_ranking(self, top_n: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """获取单标的盈亏排名（盈利最多/亏损最多的TopN）"""
        if not self.stock_total_pnl:
            return pd.DataFrame(), pd.DataFrame()

        df = pd.DataFrame(list(self.stock_total_pnl.items()), columns=["ts_code", "累计净盈亏"])
        df = df.sort_values("累计净盈亏", ascending=False).reset_index(drop=True)

        profit_top = df.head(top_n).copy()
        loss_top = df.tail(top_n).copy()
        loss_top = loss_top.sort_values("累计净盈亏", ascending=True).reset_index(drop=True)

        return profit_top, loss_top

    def export_backtest_result(self, file_path: str = "回测结果.csv"):
        """导出回测结果（含单标的盈亏排名，完全兼容原有逻辑）"""
        net_value_df = self.get_net_value_df()
        trade_df = self.get_trade_df()

        # 回测汇总表
        summary_df = pd.DataFrame({
            "策略名称": [self.strategy_name],
            "回测开始日期": [self.backtest_start_date],
            "回测结束日期": [self.backtest_end_date],
            "初始资金": [round(self.init_capital, 2)],
            "最终总资产": [round(self.total_asset, 2)],
            "总盈亏": [round(self.total_asset - self.init_capital, 2)],
            "总收益率(%)": [round((self.total_asset - self.init_capital) / self.init_capital * 100, 2)],
            "总交易次数": [len(self.trade_history)],
            "最大持仓数": [self.max_position_count]
        })

        # 单标的盈亏排名
        profit_top, loss_top = self.get_stock_pnl_ranking()

        # 导出文件
        if file_path.endswith(".xlsx"):
            with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
                summary_df.to_excel(writer, sheet_name="回测汇总", index=False)
                net_value_df.to_excel(writer, sheet_name="每日净值", index=False)
                trade_df.to_excel(writer, sheet_name="交易记录", index=False)
                if not profit_top.empty:
                    profit_top.to_excel(writer, sheet_name="盈利最多Top10", index=False)
                if not loss_top.empty:
                    loss_top.to_excel(writer, sheet_name="亏损最多Top10", index=False)
            logger.info(f"回测结果已导出Excel：{file_path}")
        else:
            net_value_df.to_csv(file_path, index=False, encoding="utf-8-sig")
            trade_df.to_csv(file_path.replace(".csv", "_交易记录.csv"), index=False, encoding="utf-8-sig")
            summary_df.to_csv(file_path.replace(".csv", "_回测汇总.csv"), index=False, encoding="utf-8-sig")
            if not profit_top.empty:
                profit_top.to_csv(file_path.replace(".csv", "_盈利最多Top10.csv"), index=False, encoding="utf-8-sig")
            if not loss_top.empty:
                loss_top.to_csv(file_path.replace(".csv", "_亏损最多Top10.csv"), index=False, encoding="utf-8-sig")
            logger.info(f"回测结果已导出CSV：{file_path}")

    def get_net_value_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.daily_net_value)

    def get_trade_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.trade_history)

    @staticmethod
    def _unify_date_format(date_str: str) -> str:
        """统一日期格式为YYYYMMDD"""
        try:
            return datetime.strptime(date_str.replace("-", ""), "%Y%m%d").strftime("%Y%m%d")
        except Exception as e:
            logger.error(f"日期格式转换失败：{date_str}，错误：{e}")
            return date_str