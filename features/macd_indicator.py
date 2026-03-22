import datetime
import logging
from typing import Optional, List, Union

import pandas as pd

from data.data_cleaner import data_cleaner  # 导入数据清洗入库实例
from utils.db_utils import db
from utils.log_utils import logger

# MACD行业标准参数（固定，无需自定义）
MACD_PARAMS = {"ema12": 12, "ema26": 26, "dea9": 9}


class MACDIndicator:
    """MACD指标计算类（量化行业标准口径，独立封装）
    新增：支持多股票批量计算金叉/死叉，完全兼容原有单股票调用逻辑
    """

    def __init__(self):
        self.logger = logger or logging.getLogger(__name__)

    def _get_qfq_kline_data(
            self,
            ts_code: str,
            start_date: str,
            end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        内部方法：获取并入库前复权K线数据（与technical.py完全一致，保证数据来源统一）
        :param ts_code: 股票代码（如600000.SH）
        :param start_date: 开始日期（YYYYMMDD）
        :param end_date: 结束日期（YYYYMMDD），不传默认到当日
        :return: 前复权K线DataFrame（空则返回空DF）
        """
        # 1. 先入库前复权数据（确保数据存在）
        self.logger.info(f"开始获取{ts_code}前复权K线数据（{start_date}~{end_date}）")
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

        # 关键修正：将数据库返回的YYYY-MM-DD转回YYYYMMDD（对齐项目格式）
        kline_df["trade_date"] = pd.to_datetime(kline_df["trade_date"]).dt.strftime("%Y%m%d")
        return kline_df

    def _calculate_single_stock_macd(
            self,
            ts_code: str,
            start_date: str,
            end_date: str
    ) -> pd.DataFrame:
        """内部方法：计算单只股票的MACD（抽离原有核心逻辑，便于多股票复用）"""
        # ========== 1. 日期处理 ==========
        start_date_dt = datetime.datetime.strptime(start_date, "%Y%m%d").date()
        max_macd_day = MACD_PARAMS["ema26"]
        new_start_date_dt = start_date_dt - datetime.timedelta(days=max_macd_day * 2)
        new_start_date = new_start_date_dt.strftime("%Y%m%d")

        # ========== 2. 获取K线数据 ==========
        kline_df = self._get_qfq_kline_data(ts_code, new_start_date, end_date)
        if kline_df.empty:
            self.logger.warning(f"{ts_code}MACD计算：K线数据为空，返回空DF")
            return pd.DataFrame()

        # ========== 3. 计算MACD核心指标 ==========
        result_df = kline_df.copy()
        # 计算12/26日EMA
        result_df["ema12"] = result_df["close"].ewm(span=MACD_PARAMS["ema12"], adjust=False, min_periods=1).mean()
        result_df["ema26"] = result_df["close"].ewm(span=MACD_PARAMS["ema26"], adjust=False, min_periods=1).mean()
        # 计算DIF、DEA、MACD柱状线
        result_df["dif"] = result_df["ema12"] - result_df["ema26"]
        result_df["dea"] = result_df["dif"].ewm(span=MACD_PARAMS["dea9"], adjust=False, min_periods=1).mean()
        result_df["macd_bar"] = 2 * (result_df["dif"] - result_df["dea"])

        # ========== 4. 计算金叉/死叉信号 ==========
        result_df["dif_prev"] = result_df["dif"].shift(1)
        result_df["dea_prev"] = result_df["dea"].shift(1)
        result_df["is_golden_cross"] = (result_df["dif"] > result_df["dea"]) & (
                result_df["dif_prev"] <= result_df["dea_prev"])
        result_df["is_death_cross"] = (result_df["dif"] < result_df["dea"]) & (
                result_df["dif_prev"] >= result_df["dea_prev"])

        # ========== 5. 数据精度处理 ==========
        result_df["dif"] = result_df["dif"].round(4)
        result_df["dea"] = result_df["dea"].round(4)
        result_df["macd_bar"] = result_df["macd_bar"].round(4)

        # ========== 6. 筛选原始日期范围 ==========
        # 转换trade_date为date对象用于筛选（内部临时使用）
        result_df['trade_date_dt'] = pd.to_datetime(result_df['trade_date'], format="%Y%m%d").dt.date
        end_date_dt = datetime.datetime.strptime(end_date, "%Y%m%d").date()
        # 筛选原始日期范围
        filter_mask = (result_df['trade_date_dt'] >= start_date_dt) & (result_df['trade_date_dt'] <= end_date_dt)
        result_df = result_df[filter_mask].drop(columns=['trade_date_dt', 'ema12', 'ema26', 'dif_prev', 'dea_prev'])

        # ========== 7. 整理返回格式 ==========
        return_cols = ["ts_code", "trade_date", "close", "dif", "dea", "macd_bar", "is_golden_cross", "is_death_cross"]
        result_df = result_df[return_cols].sort_values("trade_date", ascending=True).reset_index(drop=True)

        return result_df

    def calculate_macd(
            self,
            ts_code: Union[str, List[str]],  # 支持单股票(str)或多股票(list)
            start_date: str,
            end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        计算MACD指标及金叉/死叉信号（量化行业标准口径，基于前复权收盘价）
        新增：支持多股票批量计算，完全兼容原有单股票调用逻辑
        :param ts_code: 股票代码（如688167.SH）或股票代码列表（如["688167.SH", "300308.SZ"]）
        :param start_date: 开始日期（YYYYMMDD）
        :param end_date: 结束日期（YYYYMMDD），不传默认到当日
        :return: 包含MACD及金叉死叉的DataFrame（trade_date为YYYYMMDD格式）
        """
        # 处理默认结束日期
        if not end_date:
            end_date = pd.Timestamp.now().strftime("%Y%m%d")

        # ========== 1. 单股票计算（原有逻辑，完全保留） ==========
        if isinstance(ts_code, str):
            result_df = self._calculate_single_stock_macd(ts_code, start_date, end_date)
            # 原有日志输出格式不变
            self.logger.info(f"{ts_code}MACD计算完成，数据行数：{len(result_df)} | "
                             f"金叉次数：{result_df['is_golden_cross'].sum()} | 死叉次数：{result_df['is_death_cross'].sum()}")
            return result_df

        # ========== 2. 多股票批量计算（新增功能） ==========
        if isinstance(ts_code, list):
            self.logger.info(f"开始批量计算{len(ts_code)}只股票的MACD（{start_date}~{end_date}）")
            all_result_dfs = []
            # 遍历每个股票，复用单股票计算逻辑
            for code in ts_code:
                if not code:  # 跳过空代码
                    continue
                single_df = self._calculate_single_stock_macd(code, start_date, end_date)
                if not single_df.empty:
                    all_result_dfs.append(single_df)
                    self.logger.debug(f"{code}MACD计算完成，数据行数：{len(single_df)} | "
                                      f"金叉次数：{single_df['is_golden_cross'].sum()} | 死叉次数：{single_df['is_death_cross'].sum()}")

            # 合并所有股票结果，保持原有字段结构
            if not all_result_dfs:
                self.logger.warning("多股票MACD计算：所有股票均无有效数据")
                return pd.DataFrame()

            final_df = pd.concat(all_result_dfs, ignore_index=True)
            # 按股票代码+交易日排序，便于查看
            final_df = final_df.sort_values(
                by=["ts_code", "trade_date"],
                ascending=[True, True]
            ).reset_index(drop=True)

            # 批量计算日志（新增，不影响原有日志格式）
            total_stocks = len(final_df["ts_code"].unique())
            self.logger.info(f"多股票MACD批量计算完成 | 有效股票数：{total_stocks} | 总数据行数：{len(final_df)} | "
                             f"总金叉次数：{final_df['is_golden_cross'].sum()} | 总死叉次数：{final_df['is_death_cross'].sum()}")
            return final_df

        # 无效输入处理
        self.logger.error(f"ts_code格式错误：仅支持字符串或列表，当前类型：{type(ts_code)}")
        return pd.DataFrame()

    def get_stock_cross_dates(
            self,
            ts_code: Union[str, List[str]],
            start_date: str,
            end_date: Optional[str] = None
    ) -> dict:
        """
        便捷方法：单独返回每个股票的金叉/死叉日期（YYYYMMDD格式）
        :param ts_code: 股票代码/股票代码列表
        :param start_date: 开始日期
        :param end_date: 结束日期
        :return: 字典，格式：{股票代码: {"golden_cross": [日期列表], "death_cross": [日期列表]}}
        """
        # 先调用calculate_macd获取基础数据
        macd_df = self.calculate_macd(ts_code, start_date, end_date)
        if macd_df.empty:
            return {}

        # 整理每个股票的交叉日期
        cross_date_dict = {}
        for code in macd_df["ts_code"].unique():
            code_df = macd_df[macd_df["ts_code"] == code]
            # 提取金叉/死叉日期（YYYYMMDD字符串）
            golden_dates = code_df[code_df["is_golden_cross"] == True]["trade_date"].tolist()
            death_dates = code_df[code_df["is_death_cross"] == True]["trade_date"].tolist()
            cross_date_dict[code] = {
                "golden_cross": golden_dates,
                "death_cross": death_dates
            }

        return cross_date_dict


# 全局实例
macd_indicator = MACDIndicator()

# 测试代码
if __name__ == "__main__":
    """MACD金叉死叉计算测试（独立文件测试）"""
    # # ========== 测试1：原有单股票功能（完全兼容，无变化） ==========
    # logger.info("===== 测试1：单股票计算 =====")
    # macd_single = macd_indicator.calculate_macd(
    #     ts_code="688167.SH",
    #     start_date="20250105"
    # )
    # if not macd_single.empty:
    #     # 提取单股票交叉日期
    #     golden_cross = macd_single[macd_single["is_golden_cross"] == True]["trade_date"].tolist()
    #     death_cross = macd_single[macd_single["is_death_cross"] == True]["trade_date"].tolist()
    #     logger.info(f"688167.SH 金叉日期：{golden_cross}")
    #     logger.info(f"688167.SH 死叉日期：{death_cross}")

    # ========== 测试2：新增多股票批量计算 ==========
    logger.info("\n===== 测试2：多股票批量计算 =====")
    # 传入股票代码列表
    multi_stocks = ["300502.SZ"]
    macd_multi = macd_indicator.calculate_macd(
        ts_code=multi_stocks,
        start_date="20250305"
    )
    if not macd_multi.empty:
        # 便捷方法：获取每个股票的交叉日期
        cross_dates = macd_indicator.get_stock_cross_dates(multi_stocks, start_date="20250105")
        # 分别输出每个股票的金叉/死叉日期
        for code, dates in cross_dates.items():
            logger.info(f"\n{code}：")
            logger.info(f"  金叉日期：{dates['golden_cross']}")
            logger.info(f"  死叉日期：{dates['death_cross']}")