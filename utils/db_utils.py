import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Optional, List, Tuple

import pandas as pd
import pymysql
from dbutils.pooled_db import PooledDB
from dotenv import load_dotenv
from utils.log_utils import logger

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", ".env")
load_dotenv(CONFIG_PATH)


class DBConnector:
    """
    MySQL连接工具类（单例 + 连接池）
    优化点：
    - 减少高频日志开销
    - schema信息缓存
    - 批量SQL分块执行
    - DataFrame批量插入内存优化
    - getenv集中读取
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_pool()
            cls._instance.logger = logger
        return cls._instance

    # =========================
    # 连接池初始化（优化 getenv 读取）
    # =========================
    def _init_pool(self):
        try:
            env = os.environ
            self._db_name = env.get("DB_NAME")

            self.pool = PooledDB(
                creator=pymysql,
                host=env.get("DB_HOST"),
                port=int(env.get("DB_PORT")),
                user=env.get("DB_USER"),
                password=env.get("DB_PASSWORD"),
                database=self._db_name,
                charset="utf8mb4",
                mincached=int(env.get("DB_MIN_CONNS", 2)),
                maxcached=int(env.get("DB_MAX_CONNS", 10)),
                maxconnections=50,
                blocking=True,
                cursorclass=pymysql.cursors.DictCursor,
                connect_timeout=int(env.get("DB_CONNECT_TIMEOUT", 5))
            )

            logger.info("数据库连接池初始化成功")

        except Exception as e:
            logger.critical(f"数据库连接池初始化失败：{e}")
            raise RuntimeError(e)

    # =========================
    # 获取连接（热路径，去日志）
    # =========================
    def get_conn(self):
        conn = self.pool.connection()
        cursor = conn.cursor()
        return conn, cursor

    # =========================
    # 关闭资源（静默失败）
    # =========================
    def close(self, conn, cursor):
        if cursor:
            try:
                cursor.close()
            except Exception:
                pass
        if conn:
            try:
                conn.close()
            except Exception:
                pass

    # =========================
    # 查询
    # =========================
    def query(self, sql, params=None, return_df=False):
        conn = cursor = None
        try:
            conn, cursor = self.get_conn()
            cursor.execute(sql, params or ())
            logger.debug(f"查询sql: {(sql)} ，参数{params}")
            result = cursor.fetchall()
            if return_df:
                return pd.DataFrame.from_records(result)
            return result
        except Exception as e:
            logger.error(f"查询失败: {e}"+sql)
            return None

        finally:
            self.close(conn, cursor)

    # =========================
    # 单条执行
    # =========================
    def execute(self, sql, params=None):
        conn = cursor = None
        try:
            conn, cursor = self.get_conn()
            rows = cursor.execute(sql, params or ())
            conn.commit()
            return rows

        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"执行失败: {e}")
            return None

        finally:
            self.close(conn, cursor)

    # =========================
    # 表字段缓存
    # =========================
    def get_table_columns(self, table_name: str) -> List[str]:
        if not hasattr(self, "_table_columns_cache"):
            self._table_columns_cache = {}

        if table_name in self._table_columns_cache:
            return self._table_columns_cache[table_name]

        sql = """
            SELECT COLUMN_NAME
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
        """

        result = self.query(sql, (self._db_name, table_name))
        cols = [r["COLUMN_NAME"] for r in result] if result else []

        self._table_columns_cache[table_name] = cols
        return cols

    # =========================
    # 新增字段（最终有效版本）
    # =========================
    def add_table_column(self, table_name: str, col_name: str, col_type: str = "VARCHAR(255)",
                         comment: str = "") -> bool:

        table_columns = self.get_table_columns(table_name)

        if col_name in table_columns:
            return True

        try:
            sql = f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type}"
            if comment:
                sql += f" COMMENT '{comment}'"
            else:
                sql += f" DEFAULT '' COMMENT '{col_name}'"

            self.execute(sql)

            # 更新缓存
            self._table_columns_cache.pop(table_name, None)

            return True

        except Exception as e:
            logger.error(f"新增字段失败: {e}")
            return False

    # =========================
    # 批量新增字段
    # =========================
    def batch_add_table_columns(self, table_name: str, col_names: List[str], col_type: str = "VARCHAR(255)") -> bool:
        success = True
        for col in col_names:
            if not self.add_table_column(table_name, col, col_type):
                success = False
        return success

    # =========================
    # 批量执行（分块优化）
    # =========================
    def batch_execute(self, sql: str, params_list: List[Tuple]) -> Optional[int]:

        if not params_list:
            return 0

        CHUNK = 1000
        total = 0

        conn = cursor = None

        try:
            conn, cursor = self.get_conn()

            for i in range(0, len(params_list), CHUNK):
                chunk = params_list[i:i+CHUNK]
                total += cursor.executemany(sql, chunk)

            conn.commit()
            return total

        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"批量执行失败: {e}")
            return None

        finally:
            self.close(conn, cursor)

    # =========================
    # DataFrame批量插入（内存优化）
    # =========================
    def batch_insert_df(self, df: pd.DataFrame, table_name: str, ignore_duplicate: bool = True):

        if df.empty:
            return 0

        try:
            columns = df.columns.tolist()
            placeholders = ", ".join(["%s"] * len(columns))
            sql_columns = ", ".join(f"`{c}`" for c in columns)

            if ignore_duplicate:
                update_clause = ", ".join(f"`{c}`=VALUES(`{c}`)" for c in columns)
                sql = f"INSERT INTO {table_name} ({sql_columns}) VALUES ({placeholders}) ON DUPLICATE KEY UPDATE {update_clause}"
            else:
                sql = f"INSERT INTO {table_name} ({sql_columns}) VALUES ({placeholders})"

            df = df.where(pd.notnull(df), None)  # NaN/NaT → None → SQL NULL
            data = list(df.itertuples(index=False, name=None))  # ✅ 内存优化

            CHUNK = 1000

            conn = cursor = None
            conn, cursor = self.get_conn()

            total = 0
            for i in range(0, len(data), CHUNK):
                total += cursor.executemany(sql, data[i:i+CHUNK])

            conn.commit()
            return total

        except Exception as e:
            if 'conn' in locals() and conn:
                conn.rollback()
            logger.error(f"DF批量插入失败: {e}")
            return None

        finally:
            self.close(conn, cursor)

    # =========================
    # 获取A股代码（避免DF构造）
    # =========================
    def get_all_a_stock_codes(self) -> List[str]:

        sql = "SELECT DISTINCT ts_code FROM stock_basic WHERE list_status = 'L'"

        result = self.query(sql)

        if not result:
            return []

        return [r["ts_code"] for r in result if r["ts_code"]]


# =========================
# 全局单例
# =========================
db = DBConnector()

if __name__ == "__main__":
    pass
        # """全量测试：修正主键重复后的版本"""
        # try:
        #     logger.info("===== 开始执行数据库工具全量测试 =====")
        #     logger.info("===== 1. 测试连接池初始化 =====")
        #     logger.info("连接池初始化成功 ✅")

        #     # ====================== 测试query（查询函数）======================
        #     logger.info("\n===== 2. 测试query查询函数 =====")
        #     # 测试1：查询股票基础信息前5条（返回字典列表）
        #     sql_query_1 = "SELECT ts_code, name, industry FROM stock_basic LIMIT 5"
        #     result_dict = db.query(sql_query_1)
        #     logger.info(f"查询结果（字典列表）：\n{result_dict[:2]}...")  # 只打印前2条避免过长
        #
        #     # 测试2：查询交易日历（返回DataFrame，量化场景常用）
        #     sql_query_2 = "SELECT trade_date, is_open FROM trade_calendar WHERE trade_date >= %s LIMIT 3"
        #     params_query_2 = ("2024-01-01",)
        #     result_df = db.query(sql_query_2, params_query_2, return_df=True)
        #     logger.info(f"查询结果（DataFrame）：\n{result_df}")
        #
        #     # ====================== 测试execute（单条执行）======================
        #     logger.info("\n===== 3. 测试execute单条执行函数（插入+更新+删除） =====")
        #     # 插入前清理kline_day测试数据
        #     sql_clean_kline_day_pre = "DELETE FROM kline_day WHERE ts_code = %s AND trade_date = %s"
        #     db.execute(sql_clean_kline_day_pre, ("600000.SH", "2024-01-02"))
        #
        #     # 测试1：插入测试数据到kline_day（ts_code长度修正为9位）
        #     sql_execute_insert = """
        #         INSERT INTO kline_day (ts_code, trade_date, open, high, low, close, volume, amount)
        #         VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        #         """
        #     params_execute_insert = ("600000.SH", "2024-01-02", 8.0, 8.1, 7.9, 8.05, 10000, 80500.0)
        #     insert_rows = db.execute(sql_execute_insert, params_execute_insert)
        #     logger.info(f"单条插入影响行数：{insert_rows}")
        #
        #     # 测试2：更新测试数据（修改涨跌幅）
        #     sql_execute_update = "UPDATE kline_day SET pct_chg = %s WHERE ts_code = %s AND trade_date = %s"
        #     params_execute_update = (0.625, "600000.SH", "2024-01-02")
        #     update_rows = db.execute(sql_execute_update, params_execute_update)
        #     logger.info(f"单条更新影响行数：{update_rows}")
        #
        #     # ====================== 测试batch_execute（批量执行）======================
        #     logger.info("\n===== 4. 测试batch_execute批量执行函数 =====")
        #     # 插入前清理trade_calendar测试数据（核心修正：避免主键重复）
        #     sql_clean_batch = "DELETE FROM trade_calendar WHERE exchange = %s AND trade_date >= %s"
        #     db.execute(sql_clean_batch, ("SH", "2024-01-01"))
        #
        #     # 测试：批量插入3条交易日历测试数据
        #     sql_batch = "INSERT INTO trade_calendar (exchange, trade_date, is_open, is_holiday, weekday) VALUES (%s, %s, %s, %s, %s)"
        #     params_batch_list = [
        #         ("SH", "2024-01-01", 0, 1, 1),
        #         ("SH", "2024-01-02", 1, 0, 2),
        #         ("SH", "2024-01-03", 1, 0, 3)
        #     ]
        #     batch_rows = db.batch_execute(sql_batch, params_batch_list)
        #     logger.info(f"批量插入影响行数：{batch_rows}")
        #
        #     # ====================== 测试batch_insert_df（DataFrame批量插入）======================
        #     logger.info("\n===== 5. 测试batch_insert_df DataFrame批量插入 =====")
        #     # 插入前清理kline_1min测试数据
        #     sql_clean_kline_1min_pre = "DELETE FROM kline_1min WHERE ts_code = %s AND trade_date = %s"
        #     db.execute(sql_clean_kline_1min_pre, ("600000.SH", "2024-01-02"))
        #
        #     # 构造1分钟K线测试DataFrame
        #     test_df = pd.DataFrame({
        #         "ts_code": ["600000.SH", "600000.SH"],
        #         "trade_time": ["2024-01-02 09:30:00", "2024-01-02 09:31:00"],
        #         "trade_date": ["2024-01-02", "2024-01-02"],
        #         "open": [8.0, 8.05],
        #         "high": [8.02, 8.08],
        #         "low": [7.98, 8.03],
        #         "close": [8.01, 8.06],
        #         "volume": [5000, 6000],
        #         "amount": [40050.0, 48360.0]
        #     })
        #     df_insert_rows = db.batch_insert_df(test_df, "kline_1min")
        #     logger.info(f"DataFrame批量插入影响行数：{df_insert_rows}")
        #
        #     # ====================== 清理测试数据（避免污染真实数据）======================
        #     logger.info("\n===== 6. 清理测试数据 =====")
        #     # 删除kline_day测试数据
        #     sql_clean_kline_day = "DELETE FROM kline_day WHERE ts_code = %s AND trade_date = %s"
        #     db.execute(sql_clean_kline_day, ("600000.SH", "2024-01-02"))
        #
        #     # 删除trade_calendar测试数据
        #     sql_clean_calendar = "DELETE FROM trade_calendar WHERE trade_date >= %s AND exchange = %s"
        #     db.execute(sql_clean_calendar, ("2024-01-01", "SH"))
        #
        #     # 删除kline_1min测试数据
        #     sql_clean_kline_1min = "DELETE FROM kline_1min WHERE ts_code = %s AND trade_date = %s"
        #     db.execute(sql_clean_kline_1min, ("600000.SH", "2024-01-02"))
        #     logger.info("测试数据清理完成 ✅")
        #
        #     logger.info("\n===== 所有测试完成 ✅ =====")
        #
        # except Exception as e:
        #     logger.error(f"测试过程中出现异常：{str(e)} ❌")