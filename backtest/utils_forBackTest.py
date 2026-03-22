import uuid
from datetime import datetime

import pandas as pd

from data.data_cleaner import DataCleaner  # 复用已有清洗逻辑
from data.data_fetcher import data_fetcher
from utils.db_utils import db
from utils.log_utils import logger

# 初始化清洗器（复用日线的清洗逻辑）
data_cleaner = DataCleaner()


class BacktestMinuteUtils:
    """分钟线回测专用工具类（按需拉取+临时入库+清空）"""

    def __init__(self):
        self.backtest_id = str(uuid.uuid4())[:8]  # 生成短回测ID（8位）
        self.temp_table = "kline_minute_temp"
        self.metadata_table = "backtest_metadata"

    def fetch_and_save_minute_data(self, ts_code: str, start_time: str, end_time: str) -> bool:
        """
        按需拉取指定股票+时间段分钟线，清洗后写入临时表
        :param ts_code: 股票代码（如000001.SZ）
        :param start_time: 开始时间（YYYY-MM-DD HH:MM:SS）
        :param end_time: 结束时间（YYYY-MM-DD HH:MM:SS）
        :return: 是否成功
        """
        logger.info(f"===== 开始拉取{ts_code}分钟线：{start_time}至{end_time} =====")

        # 1. 调用接口拉取分钟线（适配你的data_fetcher）
        # 注意：Tushare分钟线接口参数需调整为时间范围，示例：
        raw_df = data_fetcher.fetch_kline_minute(
            ts_code=ts_code,
            start_date=start_time.split()[0].replace("-", ""),  # 转YYYYMMDD
            end_date=end_time.split()[0].replace("-", ""),
            freq="1min"  # 1分钟线
        )
        if raw_df.empty:
            logger.warning(f"{ts_code}无{start_time}-{end_time}分钟线数据")
            return False

        # 2. 复用清洗逻辑（适配分钟线）
        cleaned_df = self._clean_minute_data(raw_df)
        if cleaned_df.empty:
            logger.warning(f"{ts_code}分钟线清洗后无数据")
            return False

        # 3. 加回测ID（隔离多线程数据）
        cleaned_df["backtest_id"] = self.backtest_id

        # 4. 批量写入临时表（复用db_utils的batch_insert_df）
        affected_rows = db.batch_insert_df(
            df=cleaned_df,
            table_name=self.temp_table,
            ignore_duplicate=True  # 避免重复写入
        )
        if affected_rows:
            logger.info(f"{ts_code}分钟线写入临时表完成，影响行数：{affected_rows}，回测ID：{self.backtest_id}")
            return True
        return False

    def _clean_minute_data(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """复用清洗逻辑，适配分钟线字段"""
        if raw_df.empty:
            return pd.DataFrame()

        df_cleaned = raw_df.copy()
        # 1. 字段映射（适配临时表）
        field_mapping = {"vol": "volume", "change": "change1"}
        df_cleaned = df_cleaned.rename(columns=field_mapping)

        # 2. 时间格式清洗（trade_time转为datetime）
        df_cleaned["trade_time"] = pd.to_datetime(df_cleaned["trade_time"], errors="coerce")
        df_cleaned = df_cleaned.dropna(subset=["trade_time"])  # 过滤无效时间
        # 提取trade_date（YYYYMMDD）
        df_cleaned["trade_date"] = df_cleaned["trade_time"].dt.strftime("%Y%m%d")

        # 3. 数值字段清洗（复用日线逻辑）
        numeric_fields = ["open", "high", "low", "close", "volume", "amount"]
        for field in numeric_fields:
            df_cleaned[field] = pd.to_numeric(df_cleaned[field], errors="coerce").fillna(0)

        # 4. 只保留临时表需要的字段
        final_fields = ["ts_code", "trade_time", "trade_date", "open", "high", "low", "close", "volume", "amount"]
        df_cleaned = df_cleaned[final_fields]

        logger.info(f"分钟线清洗完成：原始{len(raw_df)}行 → 清洗后{len(df_cleaned)}行")
        return df_cleaned

    def load_minute_data_for_backtest(self) -> pd.DataFrame:
        """从临时表加载当前回测ID的分钟线数据"""
        sql = f"""
            SELECT ts_code, trade_time, trade_date, open, high, low, close, volume, amount
            FROM {self.temp_table}
            WHERE backtest_id = %s
            ORDER BY trade_time ASC
        """
        df = db.query(sql, params=(self.backtest_id,), return_df=True)
        if df.empty:
            logger.warning(f"回测ID{self.backtest_id}无分钟线数据")
        return df

    def record_backtest_result(self, strategy_name: str, ts_code: str, start_time: str, end_time: str, result: dict):
        """记录回测元数据（永久保留）"""
        # 转换时间格式（适配数据库）
        start_time_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
        end_time_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
        # 回测结果转为JSON字符串
        result_str = str(result)  # 可改用json.dumps，需确保序列化正常

        # 写入元数据表
        sql = f"""
            INSERT INTO {self.metadata_table} 
            (backtest_id, strategy_name, ts_code, start_time, end_time, backtest_result)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        params = (self.backtest_id, strategy_name, ts_code, start_time_dt, end_time_dt, result_str)
        db.execute(sql, params)
        logger.info(f"回测元数据记录完成：回测ID{self.backtest_id}，策略{strategy_name}，股票{ts_code}")

    def clear_temp_data(self):
        """清空当前回测ID的临时数据（保留其他回测数据）"""
        sql = f"DELETE FROM {self.temp_table} WHERE backtest_id = %s"
        affected_rows = db.execute(sql, params=(self.backtest_id,))
        logger.info(f"清空回测ID{self.backtest_id}的临时数据，删除行数：{affected_rows}")

    def run_backtest(self, strategy_func, ts_code: str, start_time: str, end_time: str,
                     strategy_name: str = "default_strategy"):
        """
        一站式执行回测流程：拉取数据→加载数据→执行策略→记录结果→清空数据
        :param strategy_func: 策略函数（入参为分钟线DataFrame，返回回测结果dict）
        :param ts_code: 股票代码
        :param start_time: 开始时间（YYYY-MM-DD HH:MM:SS）
        :param end_time: 结束时间（YYYY-MM-DD HH:MM:SS）
        :param strategy_name: 策略名称
        :return: 回测结果
        """
        try:
            # 1. 拉取并保存分钟线
            if not self.fetch_and_save_minute_data(ts_code, start_time, end_time):
                return {"status": "failed", "msg": "拉取分钟线失败"}

            # 2. 加载数据
            minute_df = self.load_minute_data_for_backtest()
            if minute_df.empty:
                return {"status": "failed", "msg": "无可用分钟线数据"}

            # 3. 执行策略回测
            backtest_result = strategy_func(minute_df)
            backtest_result["status"] = "success"

            # 4. 记录元数据
            self.record_backtest_result(strategy_name, ts_code, start_time, end_time, backtest_result)

            # 5. 清空临时数据
            self.clear_temp_data()

            logger.info(f"===== 回测完成：策略{strategy_name}，股票{ts_code} =====")
            return backtest_result

        except Exception as e:
            logger.error(f"回测失败：{str(e)}", exc_info=True)
            # 异常时也清空临时数据
            self.clear_temp_data()
            return {"status": "failed", "msg": str(e)}


# ------------------- 策略函数示例（你可自定义） -------------------
def demo_strategy(minute_df: pd.DataFrame) -> dict:
    """
    示例策略：计算简单收益率
    :param minute_df: 分钟线DataFrame
    :return: 回测结果
    """
    if minute_df.empty:
        return {"return_rate": 0, "max_drawdown": 0}

    # 计算收益率：(最后收盘价 - 第一开盘价)/第一开盘价
    first_open = minute_df["open"].iloc[0]
    last_close = minute_df["close"].iloc[-1]
    return_rate = (last_close - first_open) / first_open * 100

    # 简单计算最大回撤（示例）
    minute_df["cum_max"] = minute_df["close"].cummax()
    minute_df["drawdown"] = (minute_df["close"] - minute_df["cum_max"]) / minute_df["cum_max"] * 100
    max_drawdown = minute_df["drawdown"].min()

    return {
        "return_rate": round(return_rate, 2),
        "max_drawdown": round(max_drawdown, 2),
        "total_trades": len(minute_df),
        "time_range": f"{minute_df['trade_time'].min()}至{minute_df['trade_time'].max()}"
    }


# ------------------- 调用示例 -------------------
if __name__ == "__main__":
    # 初始化回测工具
    backtest_utils = BacktestMinuteUtils()

    # 执行回测（示例：万科A 2024-01-02 09:30至15:00的1分钟线）
    result = backtest_utils.run_backtest(
        strategy_func=demo_strategy,
        ts_code="000002.SZ",
        start_time="2024-01-02 09:30:00",
        end_time="2024-01-02 15:00:00",
        strategy_name="demo_minute_strategy"
    )

    # 打印回测结果
    logger.info(f"回测结果：{result}")