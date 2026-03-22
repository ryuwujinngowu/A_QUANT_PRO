import datetime
import logging
from typing import Optional, List, Union

import pandas as pd

from data.data_cleaner import data_cleaner  # 导入数据清洗入库实例
from utils.db_utils import db
from utils.log_utils import logger

COMMON_MA_DAYS = [5, 10, 20, 60, 120, 250]  # 5日(周)、10日(双周)、20日(月)、60日(季)、120日(半年)、250日(年)


class TechnicalFeatures:
    """技术指标特征计算类（量化行业标准口径）"""

    def __init__(self):
        self.logger = logger or logging.getLogger(__name__)

    def _get_qfq_kline_data(
            self,
            ts_code: str,
            start_date: str,
            end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        内部方法：获取并入库前复权K线数据（封装数据获取逻辑）
        :param ts_code: 股票代码（如600000.SH）
        :param start_date: 开始日期（YYYYMMDD）
        :param end_date: 结束日期（YYYYMMDD），不传默认到当日
        :return: 前复权K线DataFrame（空则返回空DF）
        """
        # 1. 先入库前复权数据（确保数据存在）
        # self.logger.info(f"开始获取{ts_code}前复权K线数据（{start_date}~{end_date}）")
        affected_rows = data_cleaner.clean_and_insert_kline_day_qfq(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            table_name="kline_day_qfq"
        )
        sql = """
              SELECT ts_code, trade_date, open, high, low, close, volume, amount
              FROM kline_day_qfq
              WHERE ts_code = %s \
                AND trade_date BETWEEN %s \
                AND %s
              ORDER BY trade_date ASC \
              """
        # 处理默认结束日期
        if not end_date:
            end_date = pd.Timestamp.now().strftime("%Y%m%d")
        # 日期格式转换（数据库中trade_date是DATE类型，需匹配）
        start_date_format = pd.to_datetime(start_date, format="%Y%m%d").strftime("%Y-%m-%d")
        end_date_format = pd.to_datetime(end_date, format="%Y%m%d").strftime("%Y-%m-%d")
        # 从数据库查询数据
        kline_df = db.query(
            sql=sql,
            params=(ts_code, start_date_format, end_date_format),
            return_df=True
        )
        if kline_df.empty:
            self.logger.warning(f"{ts_code}在{start_date}~{end_date}范围内无前复权K线数据")
            return pd.DataFrame()

        # 确保收盘价为数值类型
        kline_df["close"] = pd.to_numeric(kline_df["close"], errors="coerce").fillna(0)
        return kline_df

    def calculate_ma(
            self,
            ts_code: str,
            start_date: str,
            end_date: Optional[str] = None,
            ma_days: Union[int, List[int]] = COMMON_MA_DAYS
    ) -> pd.DataFrame:
        """
        计算简单移动平均线（MA）- 量化行业核心口径（基于前复权收盘价）
        :param ts_code: 股票代码
        :param start_date: 开始日期（YYYYMMDD）
        :param end_date: 结束日期（YYYYMMDD），不传默认到当日
        :param ma_days: 均线天数，支持单个（如5）或列表（如[5,10,20]），默认行业通用口径
        :return: 包含均线的DataFrame（trade_date/ts_code/close/ma5/ma10...）
        """
        start_date_dt = datetime.datetime.strptime(start_date, "%Y%m%d").date()

        # 2. 处理ma_days，确保为列表并取最大值
        if isinstance(ma_days, int):
            ma_days = [ma_days]  # 转为列表统一处理
        max_ma_day = max(ma_days)  # 提取最大均线天数
        new_start_date_dt = start_date_dt - datetime.timedelta(days=max_ma_day*2)
        new_start_date = new_start_date_dt.strftime("%Y%m%d")
        # 校验均线天数合理性
        invalid_days = [d for d in ma_days if d < 1]
        if invalid_days:
            self.logger.error(f"无效均线天数：{invalid_days}，天数需≥1")
            return pd.DataFrame()

        # 2. 获取前复权K线数据
        kline_df = self._get_qfq_kline_data(ts_code, new_start_date, end_date)
        if kline_df.empty:
            return pd.DataFrame()

        # 3. 计算指定天数的均线（基于收盘价，行业标准）
        result_df = kline_df.copy()
        for day in ma_days:
            col_name = f"ma{day}"
            # 简单移动平均：rolling(window=day).mean()，不足天数填充NaN（行业惯例）
            result_df[col_name] = result_df["close"].rolling(window=day, min_periods=1).mean()
            # 保留4位小数（与行情软件精度一致）
            result_df[col_name] = result_df[col_name].round(4)

        # 4. 整理返回字段
        return_cols = ["ts_code", "trade_date", "close"] + [f"ma{day}" for day in ma_days]
        result_df = result_df[return_cols].sort_values("trade_date", ascending=True)
        # ========== 4. 筛选：仅保留用户原始日期范围的数据 ==========
        # 先把trade_date转成日期对象，方便筛选
        result_df['trade_date_dt'] = pd.to_datetime(result_df['trade_date'], format="%Y%m%d").dt.date
        # 筛选条件：trade_date在原始start_date和end_date之间（包含边界）
        if not end_date:
            end_date =  datetime.datetime.strptime(pd.Timestamp.now().strftime("%Y%m%d"), "%Y%m%d").date()
        else:
            end_date =  datetime.datetime.strptime(end_date, "%Y%m%d").date()
        filter_mask = (result_df['trade_date_dt'] >= start_date_dt) & \
                      (result_df['trade_date_dt'] <= end_date)
        result_df = result_df[filter_mask].drop(columns=['trade_date_dt'])  # 删除临时列

        # ========== 5. 整理返回格式 ==========
        return_cols = ["ts_code", "trade_date", "close"] + [f"ma{day}" for day in ma_days]
        result_df = result_df[return_cols].sort_values("trade_date", ascending=True).reset_index(drop=True)
        return result_df



    def compute_ma_from_qfq_range(
            self,
            ts_code: str,
            trade_date: str,
            trade_dates: List[str],
            ma_periods: List[int],
    ) -> dict:
        """
        从已入库的 QFQ 数据计算指定股票在 trade_date 的各周期 MA。

        与 calculate_ma 的区别：
          - 不触发数据入库（假设 QFQ 数据已存在）
          - 接收 trade_dates（YYYY-MM-DD 列表）定位回看范围
          - 返回轻量字典而非 DataFrame，适合逐只检查卖出信号的场景

        :param ts_code:     股票代码
        :param trade_date:  目标日期（YYYY-MM-DD）
        :param trade_dates: 历史交易日列表（YYYY-MM-DD，含回看窗口）
        :param ma_periods:  均线周期列表，如 [10, 20, 30, 60]
        :return: {period: ma_value, ..., 'qfq_close_today': float}；数据不足返回 {}
        """
        from utils.common_tools import get_qfq_kline_range

        max_period = max(ma_periods)
        try:
            today_idx = trade_dates.index(trade_date)
        except ValueError:
            return {}

        lookback_idx   = max(0, today_idx - max_period)
        lookback_start = trade_dates[lookback_idx]

        kline = get_qfq_kline_range([ts_code], lookback_start, trade_date)
        if kline.empty:
            return {}

        kline = kline[kline["ts_code"] == ts_code].sort_values("trade_date")
        closes = kline["close"].astype(float).tolist()
        if not closes:
            return {}

        result = {"qfq_close_today": closes[-1]}
        for period in ma_periods:
            if len(closes) >= period:
                result[period] = round(sum(closes[-period:]) / period, 4)
        return result

    @staticmethod
    def compute_bias_rate_series(close_list: list, ma_period: int = 5) -> list:
        """
        计算收盘价序列对 MAn 的每日乖离率序列（公用轮子）。
        BIAS = (close - MA_n) / MA_n × 100
        数据不足 ma_period 时对应位置返回 None。

        :param close_list: 收盘价列表（按时间升序）
        :param ma_period:  均线周期，默认5日
        :return: 与 close_list 等长的乖离率列表（None 表示数据不足）
        """
        result = []
        for i, close in enumerate(close_list):
            if i + 1 < ma_period:
                result.append(None)
                continue
            window = close_list[i - ma_period + 1: i + 1]
            ma = sum(window) / ma_period
            if not ma or ma == 0:
                result.append(None)
            else:
                result.append(round((close - ma) / ma * 100, 4))
        return result

    def compute_ma_from_hfq_range(
            self,
            ts_code: str,
            trade_date: str,
            trade_dates: List[str],
            ma_periods: List[int],
            preloaded_kline: Optional[pd.DataFrame] = None,
    ) -> dict:
        """
        从已入库的 HFQ（后复权）数据计算指定股票在 trade_date 的各周期 MA。

        与 QFQ 版本的关键差异：
          - 读取 kline_day_hfq 表（后复权数据历史价格不变，无未来函数）
          - 回看窗口为 max_period * 2（后复权数据需要更长窗口保证均线精度）
          - 返回 hfq_close_today 而非 qfq_close_today

        :param ts_code:          股票代码
        :param trade_date:       目标日期（YYYY-MM-DD）
        :param trade_dates:      历史交易日列表（YYYY-MM-DD，含回看窗口）
        :param ma_periods:       均线周期列表，如 [10, 20, 30, 60]
        :param preloaded_kline:  预加载的 HFQ DataFrame（含 ts_code/trade_date/close），
                                 有数据时直接使用不再查库，提升 daily 模式性能
        :return: {period: ma_value, ..., 'hfq_close_today': float}；数据不足返回 {}
        """
        max_period = max(ma_periods)
        try:
            today_idx = trade_dates.index(trade_date)
        except ValueError:
            return {}

        # 回看 2 倍最大周期，确保后复权均线精度
        lookback_idx   = max(0, today_idx - max_period * 2)
        lookback_start = trade_dates[lookback_idx]

        if preloaded_kline is not None and not preloaded_kline.empty:
            # 使用预加载数据（零 DB 查询）
            kline = preloaded_kline[preloaded_kline["ts_code"] == ts_code].copy()
            if not kline.empty:
                # 标准化 trade_date 为 YYYY-MM-DD 格式以便比较
                kline["_td"] = kline["trade_date"].astype(str).str.replace("-", "").str[:8]
                lb_fmt = lookback_start.replace("-", "")
                td_fmt = trade_date.replace("-", "")
                kline = kline[(kline["_td"] >= lb_fmt) & (kline["_td"] <= td_fmt)]
                kline = kline.sort_values("_td")
        else:
            # 降级：查库
            from utils.common_tools import get_hfq_kline_range
            kline = get_hfq_kline_range([ts_code], lookback_start, trade_date)
            if kline.empty:
                return {}
            kline = kline[kline["ts_code"] == ts_code].sort_values("trade_date")

        closes = kline["close"].astype(float).tolist()
        if not closes:
            return {}

        result = {"hfq_close_today": closes[-1]}
        for period in ma_periods:
            if len(closes) >= period:
                result[period] = round(sum(closes[-period:]) / period, 4)
        return result


# 全局实例（供策略模块调用）
technical_features = TechnicalFeatures()

# 测试代码（极简风格，与项目测试逻辑对齐）
if __name__ == "__main__":
    """均线计算测试（行业通用口径）"""
    # 1. 测试简单移动平均（MA）
    ma_result = technical_features.calculate_ma(
        ts_code="300308.SZ",
        start_date="20250305",
        ma_days=[5, 10, 20, 60]  # 测试5/10/20日均线
    )
    if not ma_result.empty:
        logger.info("=====前复权均线数据（MA5/MA10/MA20） =====")
        logger.info(ma_result)  # 打印最后5行数据

    # 2. 测试指数移动平均（EMA）
    # ema_result = technical_features.calculate_ema(
    #     ts_code="600000.SH",
    #     start_date="20250101",
    #     end_date="20250131",
    #     ema_days=5
    # )
    # if not ema_result.empty:
    #     logger.info("===== 600000.SH 前复权EMA5数据 =====")
    #     logger.info(ema_result.tail(5))