import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
# 导入配置和日志（保留原有依赖）
from config.config import ANNUAL_TRADE_DAYS, RISK_FREE_RATE

from utils.log_utils import logger

# 回测记录文件路径（本目录下，自动创建）
BACKTEST_RECORD_PATH = os.path.join(os.path.dirname(__file__), "回测记录.csv")


class BacktestMetrics:
    """回测指标计算类：内聚指标计算、日志打印、结果写入文件逻辑"""

    def __init__(self,
                 net_value_df: pd.DataFrame,
                 init_capital: float,
                 trade_df: pd.DataFrame,
                 strategy_name: str = "未知策略",
                 backtest_start_date: str = "",
                 backtest_end_date: str = "",
                 strategy_params: Optional[str] = None,
                 benchmark_daily_return: Optional[pd.Series] = None):
        """
        初始化回测指标类
        :param net_value_df: 净值曲线DataFrame（必须含total_asset列）
        :param init_capital: 初始本金
        :param trade_df: 交易记录DataFrame
        :param strategy_name: 策略名称
        :param backtest_start_date: 回测开始日期
        :param backtest_end_date: 回测结束日期
        :param strategy_params: 策略参数摘要字符串（由 BaseStrategy.get_params_summary() 生成）
        :param benchmark_daily_return: 基准日收益率（如沪深300，用于信息比率）
        """
        # 严格参数校验（保留日志提示）
        if net_value_df.empty or "total_asset" not in net_value_df.columns:
            logger.error("净值曲线不能为空，且必须包含total_asset列！")
            raise ValueError("净值曲线不能为空，且必须包含total_asset列！")
        if init_capital <= 0:
            logger.error("初始本金必须大于0！")
            raise ValueError("初始本金必须大于0！")

        # 数据隔离（避免修改原数据）
        self.net_value_df = net_value_df.copy()
        self.init_capital = init_capital
        self.trade_df = trade_df.copy() if not trade_df.empty else pd.DataFrame()
        self.strategy_name = strategy_name
        self.backtest_start_date = backtest_start_date
        self.backtest_end_date = backtest_end_date
        self.strategy_params = strategy_params or "-"
        self.benchmark_daily_return = benchmark_daily_return

        # 计算日收益率（填充首日NaN为0）
        self.net_value_df["daily_return"] = self.net_value_df["total_asset"].pct_change().fillna(0)
        logger.debug(f"Metrics初始化完成 | 策略：{self.strategy_name} | 回测区间：{self.backtest_start_date}~{self.backtest_end_date}")

    def _calc_complete_trades(self) -> pd.DataFrame:
        """修复BUG：按股票代码+日期匹配完整买卖交易"""
        if self.trade_df.empty:
            logger.warning("交易记录为空，交易类指标将为0")
            return pd.DataFrame()

        # 必要字段校验
        required_fields = ["ts_code", "direction", "trade_date", "total_cost", "total_income"]
        if not all(field in self.trade_df.columns for field in required_fields):
            logger.error(f"交易记录缺少必要字段：{required_fields}")
            return pd.DataFrame()

        # 拆分买入/卖出交易
        buy_trades = self.trade_df[self.trade_df["direction"] == "买入"].copy()
        sell_trades = self.trade_df[self.trade_df["direction"] == "卖出"].copy()

        # 按股票代码+日期匹配完整交易
        complete_trades = []
        buy_trades_sorted = buy_trades.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

        for _, buy in buy_trades_sorted.iterrows():
            ts_code = buy["ts_code"]
            # 匹配该股票、日期晚于买入的最早卖出
            matched_sell = sell_trades[
                (sell_trades["ts_code"] == ts_code) &
                (sell_trades["trade_date"] > buy["trade_date"])
            ].sort_values("trade_date").head(1)

            if not matched_sell.empty:
                sell = matched_sell.iloc[0]
                profit = sell["total_income"] - buy["total_cost"]
                complete_trades.append({
                    "ts_code": ts_code,
                    "buy_date": buy["trade_date"],
                    "sell_date": sell["trade_date"],
                    "buy_cost": buy["total_cost"],
                    "sell_income": sell["total_income"],
                    "profit": profit,
                    "is_win": profit > 0
                })
                # 移除已匹配的卖出（避免重复）
                sell_trades = sell_trades.drop(sell.name)

        return pd.DataFrame(complete_trades)

    def _calc_information_ratio(self) -> float:
        """新增：计算信息比率（IR = 年化超额收益 / 年化跟踪误差）"""
        if self.benchmark_daily_return is None:
            logger.warning("未传入基准收益率，信息比率返回NaN")
            return np.nan
        if len(self.benchmark_daily_return) != len(self.net_value_df["daily_return"]):
            logger.error("基准收益率长度与策略日收益长度不匹配")
            return np.nan

        # 计算每日主动收益（策略 - 基准）
        active_return = self.net_value_df["daily_return"] - self.benchmark_daily_return.values
        active_return = active_return.dropna()

        if active_return.std() == 0:
            logger.warning("主动收益无波动，信息比率设为0")
            return 0.0

        # 信息比率公式
        daily_active_mean = active_return.mean()
        daily_active_std = active_return.std()
        information_ratio = (daily_active_mean * ANNUAL_TRADE_DAYS) / (daily_active_std * np.sqrt(ANNUAL_TRADE_DAYS))
        return information_ratio

    def calc_all_metrics(self) -> dict:
        """计算所有核心回测指标（主方法）"""
        # 基础数据
        total_days = len(self.net_value_df)
        final_asset = self.net_value_df["total_asset"].iloc[-1]

        # 1. 收益类指标（防御除零错误）
        total_return = (final_asset - self.init_capital) / self.init_capital * 100
        annual_return = 0.0
        if total_days > 0:
            annual_return = (final_asset / self.init_capital) ** (ANNUAL_TRADE_DAYS / total_days) - 1
        annual_return = annual_return * 100

        # 2. 风险类指标（新增年化波动率）
        self.net_value_df["cum_max"] = self.net_value_df["total_asset"].cummax()
        self.net_value_df["drawdown"] = (self.net_value_df["total_asset"] - self.net_value_df["cum_max"]) / \
                                        self.net_value_df["cum_max"] * 100
        max_drawdown = self.net_value_df["drawdown"].min()
        daily_vol = self.net_value_df["daily_return"].std()
        annual_vol = daily_vol * np.sqrt(ANNUAL_TRADE_DAYS) if daily_vol != 0 else 0

        # 3. 夏普比率（原有逻辑，补充日志）
        excess_return = self.net_value_df["daily_return"] - (RISK_FREE_RATE / ANNUAL_TRADE_DAYS)
        sharpe_ratio = 0.0
        if excess_return.std() != 0:
            sharpe_ratio = np.sqrt(ANNUAL_TRADE_DAYS) * excess_return.mean() / excess_return.std()
        else:
            logger.warning("超额收益无波动，夏普比率设为0")

        # 4. 交易类指标（修复后）
        complete_trades = self._calc_complete_trades()
        total_trade_times = len(complete_trades)
        win_rate = 0.0
        if total_trade_times > 0:
            win_count = complete_trades["is_win"].sum()
            win_rate = (win_count / total_trade_times) * 100

        # 5. 信息比率
        information_ratio = self._calc_information_ratio()

        # 汇总指标（结构化）
        metrics = {
            "策略名称": self.strategy_name,
            "回测开始日期": self.backtest_start_date,
            "回测结束日期": self.backtest_end_date,
            "初始本金(元)": round(self.init_capital, 2),
            "最终资产(元)": round(final_asset, 2),
            "总收益率(%)": round(total_return, 2),
            "年化收益率(%)": round(annual_return, 2),
            "年化波动率(%)": round(annual_vol * 100, 2),
            "最大回撤(%)": round(max_drawdown, 2),
            "夏普比率": round(sharpe_ratio, 2),
            "信息比率": round(information_ratio, 2) if not np.isnan(information_ratio) else "无基准",
            "交易胜率(%)": round(win_rate, 2),
            "总交易次数": total_trade_times,
            "回测交易日数": total_days
        }

        # 保留原有日志（关键指标打印）
        logger.info("=" * 50)
        logger.info(f"回测指标计算完成 | 策略：{self.strategy_name}")
        logger.info(f"核心指标 | 总收益：{metrics['总收益率(%)']}% | 年化：{metrics['年化收益率(%)']}% | 最大回撤：{metrics['最大回撤(%)']}%")
        logger.info(f"风险收益 | 夏普比率：{metrics['夏普比率']} | 信息比率：{metrics['信息比率']}")
        logger.info(f"交易表现 | 胜率：{metrics['交易胜率(%)']}% | 总交易次数：{metrics['总交易次数']}")
        logger.info("=" * 50)

        # 写入文件（完全在metrics内完成，engine不参与）
        self._record_backtest_result(metrics)
        return metrics

    def _record_backtest_result(self, metrics: Dict):
        """核心：将回测结果写入CSV文件（内聚在metrics中）"""
        # 整合策略参数（便于存储）
        record_data = metrics.copy()
        record_data["策略参数"] = self.strategy_params
        record_data["记录时间"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 转为DataFrame
        record_df = pd.DataFrame([record_data])

        # 写入文件（处理编码和追加逻辑）
        try:
            if not os.path.exists(BACKTEST_RECORD_PATH):
                # 文件不存在：创建并写入表头
                record_df.to_csv(BACKTEST_RECORD_PATH, index=False, encoding="utf-8-sig")
                logger.info(f"回测记录文件已创建：{BACKTEST_RECORD_PATH}")
            else:
                # 文件存在：追加（不写表头）
                record_df.to_csv(BACKTEST_RECORD_PATH, index=False, mode="a", header=False, encoding="utf-8-sig")
            logger.info(f"回测结果已成功写入文件：{BACKTEST_RECORD_PATH}")
        except Exception as e:
            logger.error(f"回测结果写入文件失败：{str(e)}")
            # 兜底：控制台提示
            print(f"⚠️  回测记录写入失败：{str(e)}")