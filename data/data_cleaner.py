import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import datetime
import threading
import time
from typing import Optional, List, Union
import numpy as np
import pandas as pd
from data.data_fetcher import data_fetcher
from utils.common_tools import auto_add_missing_table_columns
from utils.common_tools import calc_15_years_date_range
from utils.common_tools import get_trade_dates
from utils.db_utils import db
from utils.log_utils import logger

# ── Tushare 分钟线 API 限流控制 ────────────────────────────────────────────
# 架构说明：
#   get_kline_min_by_stock_date 内置"查DB→拉API→入DB"缓存链。
#   已缓存数据走快速路径（无 API 消耗），仅 DB miss 时进入限流控制区域。
#
# 三种状态（_THROTTLE_STATE["mode"]）：
#   normal    — 正常并发（最多 2 线程同时调 API）
#   throttled — 限流模式（每次 API 调用前额外 sleep 3s，≤20次/分钟）
#   abort     — 中断模式（拒绝新 API 请求，抛 TushareRateLimitAbort）
#
# 失败计数单位（stock_fail_streak）：
#   以「单只股票的所有重试全部耗尽」为一次计数，而非单次 API 调用失败。
#   原因：一只股票重试过程中有指数退避等待（1→2→4…→30s），偶发性失败
#   最终会重试成功，不应累计；只有所有重试全部失败才说明接口持续不可用。
#   任意一只股票成功（无论第几次重试）→ 清零计数。
#
# 状态机迁移：
#   normal → throttled  : 连续 _THROTTLE_NORMAL_STOCK_STREAK 只股票永久失败
#   throttled → abort   : 连续 _THROTTLE_ABORT_STOCK_STREAK  只股票永久失败
#   任意 → normal       : 次日零点自动重置（每日 API 配额刷新）
#   任意 → normal       : 任意股票最终成功（计数清零）

_TUSHARE_MIN_API_SEM = threading.Semaphore(2)   # 最多同时 2 个 API 调用

_THROTTLE_LOCK  = threading.Lock()
_THROTTLE_STATE = {
    "mode":             "normal",  # "normal" | "throttled" | "abort"
    "stock_fail_streak": 0,        # 连续「股票级」永久失败计数（任意股票成功后清零）
    "reset_date":        None,     # 进入非 normal 状态的日期（次日自动重置）
}

# 阈值含义：
#   3 只股票全部 10 次重试失败 → 接口很可能在持续限流，降速
#   6 只股票全部 10 次重试失败 → 即使降速后仍持续失败，中断当日补全
_THROTTLE_NORMAL_STOCK_STREAK = 3   # 连续 N 只股票永久失败 → throttled
_THROTTLE_ABORT_STOCK_STREAK  = 6   # 连续 M 只股票永久失败 → abort（throttled 模式下累计）
_THROTTLE_MIN_INTERVAL  = 3.0       # 限流模式下每次 API 调用最小间隔（秒），约 20次/分钟
_MIN_FETCH_MAX_RETRIES  = 10        # 单只股票最大 API 重试次数（超出后纳入聚合告警）
# _KLINE_MIN_COVERAGE_RATIO = 0.7     # 允许停牌导致的自然缺口，低于该比例视为覆盖严重不足
# _KLINE_HIGH_COVERAGE_RATIO = 0.95   # 覆盖率超过此值时，即使首尾日未覆盖也视为完整（停牌导致的边缘缺口）
# _KLINE_FETCH_MAX_ATTEMPTS = 2        # K线区间覆盖不足时允许整段重拉次数


class TushareRateLimitAbort(Exception):
    """
    Tushare 分钟线接口严重限流或当日配额耗尽，触发当日历史补全中断。
    继承 Exception（非 BaseException），调用方需显式捕获并向上传播。
    次日零点后，_THROTTLE_STATE 自动重置，可恢复正常运行。
    """


def _throttle_get_mode() -> str:
    """线程安全地获取当前限流模式，次日自动重置。"""
    with _THROTTLE_LOCK:
        today = datetime.date.today().isoformat()
        if _THROTTLE_STATE["mode"] != "normal" and _THROTTLE_STATE["reset_date"] != today:
            _THROTTLE_STATE["mode"] = "normal"
            _THROTTLE_STATE["stock_fail_streak"] = 0
            logger.info("[限流控制] 已过零点，自动重置为正常模式")
        return _THROTTLE_STATE["mode"]


def _throttle_on_success():
    """
    某只股票最终成功（无论第几次重试），清零股票级连续失败计数。
    表明接口目前可用，指数退避重试有效，无需降速。
    """
    with _THROTTLE_LOCK:
        _THROTTLE_STATE["stock_fail_streak"] = 0


def _throttle_on_stock_perm_fail() -> str:
    """
    某只股票的所有 _MIN_FETCH_MAX_RETRIES 次重试全部耗尽且均失败时调用。
    计数单位为「只」（stock 级），而非单次 API 调用失败次数，避免指数退避
    期间的等待与计数产生歧义。

    返回更新后的模式（"normal" | "throttled" | "abort"）。
    """
    with _THROTTLE_LOCK:
        _THROTTLE_STATE["stock_fail_streak"] += 1
        streak = _THROTTLE_STATE["stock_fail_streak"]
        today  = datetime.date.today().isoformat()

        if _THROTTLE_STATE["mode"] == "normal" and streak >= _THROTTLE_NORMAL_STOCK_STREAK:
            _THROTTLE_STATE["mode"]       = "throttled"
            _THROTTLE_STATE["reset_date"] = today
            logger.warning(
                f"[限流控制] 已有 {streak} 只股票 {_MIN_FETCH_MAX_RETRIES} 次重试全部失败，"
                f"切换为限流模式（每次请求额外等待 {_THROTTLE_MIN_INTERVAL}s，≤20次/分钟）"
            )
        elif _THROTTLE_STATE["mode"] == "throttled" and streak >= _THROTTLE_ABORT_STOCK_STREAK:
            _THROTTLE_STATE["mode"] = "abort"
            logger.error(
                f"[限流控制] 限流模式下仍有 {streak} 只股票永久失败，触发当日补全中断。"
                f"次日零点后自动恢复。"
            )
        return _THROTTLE_STATE["mode"]


def is_rate_limit_aborted() -> bool:
    """供外部模块查询当前是否处于 abort（严重限流）状态，线程安全。"""
    with _THROTTLE_LOCK:
        return _THROTTLE_STATE["mode"] == "abort"


class DataCleaner:
    """数据清洗+入库核心类（优化版：精简冗余、提升效率、保留核心契约）"""

    def _get_db_columns(self, table_name: str, exclude_columns: List[str] = None) -> List[str]:
        """通用方法：获取数据库表字段（过滤排除字段）"""
        exclude_columns = exclude_columns or ["id", "created_at", "updated_at"]
        db_cols = db.get_table_columns(table_name)
        return [col for col in db_cols if col not in exclude_columns]

    def _align_df_with_db(self, df: pd.DataFrame, table_name: str, exclude_columns: List[str] = None) -> pd.DataFrame:
        """通用方法：过滤DataFrame字段为数据库表共有字段"""
        db_cols = self._get_db_columns(table_name, exclude_columns)
        common_cols = [col for col in df.columns if col in db_cols]
        return df[common_cols].copy()

    def _clean_special_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """通用格式清洗（适配涨跌停池/板块表建表结构，解决1366/1265报错）"""
        df_cleaned = df.copy()
        # 批量重命名关键字段（可扩展）
        reserved_field_mapping = {"change": "change1", "vol": "volume"}
        df_cleaned.rename(columns=reserved_field_mapping, inplace=True)

        # 日期字段格式统一（YYYYMMDD → YYYY-MM-DD）
        date_fields = ["list_date", "delist_date", "end_date", "start_date", "trade_date"]
        for field in date_fields:
            if field in df_cleaned.columns:
                formatted = pd.to_datetime(
                    df_cleaned[field], format="%Y%m%d", errors="coerce"
                )
                # NaT 还原为 None，由数据库层处理
                df_cleaned[field] = formatted.dt.strftime("%Y-%m-%d").where(formatted.notna(), None)

        # 核心NOT NULL字段非空兜底（与建表约束对齐）
        not_null_fields = ["ts_code", "symbol", "name", "exchange", "list_date"]
        for field in not_null_fields:
            if field not in df_cleaned.columns:
                continue
            dtype = df_cleaned[field].dtype
            if pd.api.types.is_object_dtype(dtype):
                df_cleaned[field] = df_cleaned[field].fillna("UNKNOWN")
            elif "date" in field:
                df_cleaned[field] = df_cleaned[field].fillna("1970-01-01")
            elif pd.api.types.is_numeric_dtype(dtype):
                df_cleaned[field] = df_cleaned[field].fillna(0)

        # ========== 核心配置：与建表结构强绑定 ==========
        # 1. 整数字段白名单（建表为INT类型，绝对禁止空字符串，强制填0）
        INTEGER_FIELDS = [
            "open_num", "days", "up_nums", "cons_nums",  # 涨跌停池表
            "nums"  # 连板天梯表
        ]
        # 2. 浮点数字段配置（建表为FLOAT，单精度仅保留2位小数，避免精度溢出）
        FLOAT_FIELDS = [
            "lu_limit_order", "limit_order", "limit_amount", "turnover_rate",
            "free_float", "limit_up_suc_rate", "turnover", "rise_rate", "sum_float",
            "pct_chg"  # 涨跌停池/板块表所有FLOAT字段
        ]
        # 3. 纯字符串字段（建表为VARCHAR，空值填''，与建表默认值对齐）
        STRING_FIELDS = [
            "name", "lu_desc", "tag", "status", "first_lu_time", "last_lu_time",
            "first_ld_time", "last_ld_time", "market_type", "up_stat", "rank"
        ]

        # 其他字段（未在配置中显式声明的，按默认逻辑处理）
        other_cols = [
            col for col in df_cleaned.columns
            if col not in not_null_fields and col not in date_fields
        ]

        for col in other_cols:
            if col not in df_cleaned.columns:
                continue
            dtype = df_cleaned[col].dtype

            # ========== 修复1：整数字段强制转INT，空值填0（解决1366） ==========
            if col in INTEGER_FIELDS:
                # 先转数值（coerce无效值为NaN），再填0，最后转INT
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors="coerce").fillna(0).astype(np.int64)
                continue

            # ========== 修复2：浮点数字段精度控制（解决1265） ==========
            if col in FLOAT_FIELDS:
                # 转数值+填0+保留2位小数（适配MySQL单精度FLOAT）
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors="coerce").fillna(0)
                # 限制2位小数，避免精度溢出
                df_cleaned[col] = df_cleaned[col].round(2)
                # 防止科学计数法（MySQL不兼容科学计数法插入）
                df_cleaned[col] = df_cleaned[col].apply(lambda x: f"{x:.2f}").astype(float)
                continue

            # ========== 修复3：纯字符串字段空值填充（与建表默认值''对齐） ==========
            if col in STRING_FIELDS:
                df_cleaned[col] = df_cleaned[col].fillna("")
                continue

            # ========== 兜底逻辑：兼容未知字段 ==========
            if pd.api.types.is_numeric_dtype(dtype):
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors="coerce").fillna(0)
            elif pd.api.types.is_object_dtype(dtype):
                # 尝试转数值，能转则填0，否则填''
                coerced = pd.to_numeric(df_cleaned[col], errors="coerce")
                if coerced.notna().any():
                    df_cleaned[col] = coerced.fillna(0)
                else:
                    df_cleaned[col] = df_cleaned[col].fillna("")

        return df_cleaned

    # ===================== 业务清洗入库方法 =====================
    def clean_and_insert_stockcompany(self, table_name: str = "stock_company") -> Optional[int]:
        """上市公司工商信息全字段自动化入库（适配tushare stock_company接口）"""
        logger.info("===== 开始上市公司工商信息清洗入库 =====")

        # 1. 获取原始数据（上游已做空值校验，仅判断empty）
        raw_df = data_fetcher.get_stock_company()
        if raw_df.empty:
            logger.warning("stock_company原始数据为空，跳过入库")
            return 0

        # 2. 通用清洗
        cleaned_df = self._clean_special_fields(raw_df)
        if cleaned_df.empty:
            logger.warning("stock_company清洗后数据为空，跳过入库")
            return 0

        # 3. 自动新增缺失字段（复用自身通用方法）
        db_cols = self._get_db_columns(table_name)
        missing_cols = [col for col in cleaned_df.columns if col not in db_cols]
        if missing_cols:
            logger.info(f"表{table_name}缺失字段：{missing_cols}，开始自动新增")
            # stock_company专属字段类型映射
            col_type_map = {
                "ts_code": "CHAR(9)",
                "com_name": "VARCHAR(128)",
                "com_id": "CHAR(18)",
                "exchange": "VARCHAR(8)",
                "chairman": "VARCHAR(32)",
                "manager": "VARCHAR(32)",
                "secretary": "VARCHAR(32)",
                "reg_capital": "DECIMAL(20,2)",
                "setup_date": "DATE",
                "province": "VARCHAR(16)",
                "city": "VARCHAR(16)",
                "introduction": "TEXT",
                "website": "VARCHAR(128)",
                "email": "VARCHAR(64)",
                "office": "VARCHAR(128)",
                "employees": "INT",
                "main_business": "TEXT",
                "business_scope": "TEXT"
            }
            for col in missing_cols:
                col_type = col_type_map.get(col, "VARCHAR(255)")
                db.add_table_column(table_name, col, col_type, comment=f"{col}（stock_company接口字段）")

        # 4. 对齐数据库字段并入库
        final_df = self._align_df_with_db(cleaned_df, table_name)
        try:
            affected_rows = db.batch_insert_df(
                df=final_df,
                table_name=table_name,
                ignore_duplicate=True
            )
            if affected_rows is None:
                logger.error(f"表{table_name}入库失败")
                return 0

            logger.info(f"✅ 表{table_name}入库完成，影响行数：{affected_rows}，字段数：{len(final_df.columns)}")
            # 精简日志示例（保留核心字段）
            if not final_df.empty:
                sample_cols = ["ts_code", "com_name", "province"]
                sample_cols = [col for col in sample_cols if col in final_df.columns]
                logger.info(f"入库示例：\n{final_df[sample_cols].head(3)}")

            return affected_rows
        except Exception as e:
            logger.error(f"表{table_name}入库异常：{str(e)}", exc_info=True)
            return 0

    def clean_and_insert_stockbase(self, table_name: str = "stock_basic") -> Optional[int]:
        """股票基础数据全字段清洗入库（适配tushare stock_basic接口）"""
        logger.info(f"===== 开始股票基础数据清洗入库（目标表：{table_name}） =====")

        # 1. 获取原始数据
        raw_df = data_fetcher.get_stockbase(list_status="L")
        if raw_df.empty:
            logger.warning("stock_basic原始数据为空，跳过入库")
            return 0

        # 2. 通用清洗
        cleaned_df = self._clean_special_fields(raw_df)
        if cleaned_df.empty:
            logger.warning("stock_basic清洗后数据为空，跳过入库")
            return 0

        # 3. 对齐数据库字段并入库
        final_df = self._align_df_with_db(cleaned_df, table_name)
        try:
            affected_rows = db.batch_insert_df(
                df=final_df,
                table_name=table_name,
                ignore_duplicate=True
            )
            if affected_rows is None:
                logger.error(f"表{table_name}入库失败")
                return 0

            logger.info(f"✅ 表{table_name}入库完成，影响行数：{affected_rows}，字段数：{len(final_df.columns)}")
            return affected_rows
        except Exception as e:
            logger.error(f"表{table_name}入库异常：{str(e)}", exc_info=True)
            return 0

    def _clean_kline_day_data(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """日K数据专属清洗（保留核心逻辑，删除上游已做的校验）"""
        if raw_df.empty:
            logger.warning("原始日K数据为空，跳过清洗")
            return pd.DataFrame()

        df_cleaned = raw_df.copy()

        # 1. 字段映射（接口→数据库）
        field_mapping = {"vol": "volume", "change": "change1"}
        df_cleaned.rename(columns=field_mapping, inplace=True)

        # 2. 核心字段列表（匹配数据库表）
        core_fields = [
            "ts_code", "trade_date", "open", "high", "low", "close",
            "pre_close", "change1", "pct_chg", "volume", "amount",
            "turnover_rate", "swing", "limit_up", "limit_down", "update_time", "reserved"
        ]

        # 3. 填充数据库新增字段默认值
        default_vals = {
            "turnover_rate": "",
            "swing": "",
            "limit_up": "",
            "limit_down": "",
            "update_time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "reserved": ""
        }
        for field, val in default_vals.items():
            if field not in df_cleaned.columns:
                df_cleaned[field] = val

        # 4. 日期格式转换（上游已做ts_code校验，删除重复校验）
        df_cleaned["trade_date"] = pd.to_datetime(
            df_cleaned["trade_date"], format="%Y%m%d", errors="coerce"
        ).dt.strftime("%Y-%m-%d")
        df_cleaned["trade_date"] = df_cleaned["trade_date"].fillna("1970-01-01")

        # 5. 数值字段类型转换+空值填充（批量处理，提升效率）
        numeric_fields = {
            "open": float, "high": float, "low": float, "close": float,
            "pre_close": float, "change1": float, "pct_chg": float,
            "volume": "int64", "amount": float,
            "turnover_rate": float, "swing": float, "limit_up": float, "limit_down": float
        }
        for field, dtype in numeric_fields.items():
            if field in df_cleaned.columns:
                df_cleaned[field] = pd.to_numeric(df_cleaned[field], errors="coerce").fillna(0)
                df_cleaned[field] = df_cleaned[field].astype(dtype)

        # 6. 字符串字段填充+去重
        df_cleaned["reserved"] = df_cleaned["reserved"].fillna("").astype(str)
        df_cleaned = df_cleaned.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")

        # 7. 保留核心字段
        df_cleaned = df_cleaned[[col for col in core_fields if col in df_cleaned.columns]]

        logger.debug(f"日K数据清洗完成：原始{len(raw_df)}行 → 清洗后{len(df_cleaned)}行")
        return df_cleaned

    def clean_and_insert_index_daily(
            self,
            ts_code: Optional[str] = None,
            trade_date: Optional[str] = None,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            table_name: str = "index_daily"
    ) -> Optional[int]:
        """指数日线数据清洗入库（精简冗余逻辑，保留核心）"""
        logger.info(f"===== 开始指数日线数据清洗入库（目标表：{table_name}） =====")

        # 1. 获取原始数据（上游已做参数校验）
        raw_df = data_fetcher.fetch_index_daily(
            ts_code=ts_code, trade_date=trade_date, start_date=start_date, end_date=end_date
        )
        if raw_df.empty:
            logger.debug("指数日线原始数据为空，跳过入库")
            return 0
        # 2. 通用清洗
        clean_df = self._clean_special_fields(raw_df)
        if clean_df.empty:
            logger.warning("指数日线数据清洗后为空，跳过入库")
            return 0

        # 3. 对齐数据库字段并入库
        final_df = self._align_df_with_db(clean_df, table_name)
        try:
            affected_rows = db.batch_insert_df(
                df=final_df,
                table_name=table_name,
                ignore_duplicate=True
            )
            logger.info(f"✅ 指数日线数据入库完成，影响行数：{affected_rows}，字段数：{len(final_df.columns)}")
            return affected_rows
        except Exception as e:
            logger.error(f"指数日线数据入库失败：{str(e)}", exc_info=True)
            return None

    def _clean_kline_min_data(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """分钟线数据清洗（精简逻辑，提升效率）"""
        if raw_df.empty:
            return pd.DataFrame()

        df_clean = raw_df.copy()
        # 字段映射（对齐日线命名）
        if "vol" in df_clean.columns:
            df_clean.rename(columns={"vol": "volume"}, inplace=True)

        # 批量替换异常值
        df_clean = df_clean.replace([np.nan, np.inf, -np.inf], 0)

        # 字段类型转换（批量处理）
        type_mapping = {"open": float, "close": float, "high": float, "low": float, "volume": int, "amount": float}
        for col, dtype in type_mapping.items():
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(dtype, errors="ignore")

        # 时间字段格式化
        if "trade_time" in df_clean.columns:
            df_clean["trade_time"] = pd.to_datetime(df_clean["trade_time"], errors="coerce")
            df_clean.dropna(subset=["trade_time"], inplace=True)
            df_clean["trade_date"] = df_clean["trade_time"].dt.date

        # 去重
        df_clean.drop_duplicates(subset=["ts_code", "trade_time"], keep="last", inplace=True)

        logger.debug(f"分钟线数据清洗完成：原始{len(raw_df)}行 → 清洗后{len(df_clean)}行")
        return df_clean

    def clean_and_insert_kline_min(self, raw_df: pd.DataFrame, table_name: str = "kline_min") -> Optional[int]:
        """分钟线数据清洗入库（精简冗余逻辑）"""
        logger.debug(f"分钟线数据入库")
        if raw_df.empty:
            return 0

        clean_df = self._clean_kline_min_data(raw_df)
        if clean_df.empty:
            return 0
        # 对齐数据库字段
        final_df = self._align_df_with_db(clean_df, table_name)

        # 批量入库
        try:
            affected_rows = db.batch_insert_df(
                df=final_df,
                table_name=table_name,
                ignore_duplicate=True
            )
            logger.debug(f"分钟线数据入库完成，影响行数：{affected_rows}")
            return affected_rows
        except Exception as e:
            logger.error(f"分钟线数据入库失败：{str(e)}", exc_info=True)
            return None

    def get_kline_min_by_stock_date(self, ts_code: str, trade_date: str, table_name: str = "kline_min") -> pd.DataFrame:
        """
        获取单只股票单日分钟线数据。

        执行链：查DB缓存 → (miss) → 限流控制 → 带重试的 API 拉取 → 入库 → 再查DB。

        限流机制
        --------
        - 全局状态机管理三种模式：normal / throttled / abort
        - 连续 API 失败 ≥ _THROTTLE_NORMAL_STREAK 次 → throttled（每次 API 多等 3s）
        - 连续失败 ≥ _THROTTLE_ABORT_STREAK 次 → abort（抛 TushareRateLimitAbort）
        - abort 模式下，调用方（agent / engine）负责向上传播异常并发微信告警
        - 次日零点自动重置（每日配额刷新）

        重试机制
        --------
        - 单只股票最多重试 _MIN_FETCH_MAX_RETRIES 次
        - 重试间隔：指数退避（1s, 2s, 4s, ... 上限 30s）
        - 所有重试耗尽后返回空 DataFrame，调用方负责记录聚合告警
        - 若期间进入 abort 模式则立即抛出异常，不再继续重试

        raises
        ------
        TushareRateLimitAbort : 进入 abort 模式时抛出，调用方必须向上传播
        """
        if not ts_code or not trade_date:
            return pd.DataFrame()

        # ── Step 1: 查 DB 缓存（快速路径，无 API 消耗）────────────────────
        sql = """
            SELECT ts_code, trade_time, trade_date, open, close, high, low, volume, amount
            FROM {table}
            WHERE ts_code = %s AND trade_date = %s
            ORDER BY trade_time ASC
        """.format(table=table_name)

        try:
            df = db.query(sql, params=(ts_code, trade_date), return_df=True)
            if not df.empty:
                df["trade_time"] = pd.to_datetime(df["trade_time"])
                logger.debug(f"[{ts_code}-{trade_date}] DB 缓存命中，行数：{len(df)}")
                return df
        except Exception as e:
            logger.error(f"[{ts_code}-{trade_date}] 查库失败：{e}")

        # ── Step 2: DB miss — 进入限流控制区域 ───────────────────────────
        mode = _throttle_get_mode()
        if mode == "abort":
            raise TushareRateLimitAbort(
                f"[{ts_code}][{trade_date}] API 已进入中断模式，拒绝新请求"
            )

        logger.debug(f"[{ts_code}-{trade_date}] DB 无缓存，进入限流控制，准备调用 API")

        for attempt in range(1, _MIN_FETCH_MAX_RETRIES + 1):
            # 每次尝试前检查是否刚进入 abort（其他线程触发）
            mode = _throttle_get_mode()
            if mode == "abort":
                raise TushareRateLimitAbort(
                    f"[{ts_code}][{trade_date}] 第 {attempt} 次重试前检测到 abort 模式"
                )

            # 限流模式：额外等待以确保全局请求率 ≤ 20次/分钟
            extra_wait = _THROTTLE_MIN_INTERVAL if mode == "throttled" else 0.0

            raw_df = pd.DataFrame()
            with _TUSHARE_MIN_API_SEM:
                if extra_wait > 0:
                    time.sleep(extra_wait)
                try:
                    raw_df = data_fetcher.fetch_stk_mins(
                        ts_code=ts_code,
                        freq="1min",
                        start_date=f"{trade_date} 09:25:00",
                        end_date=f"{trade_date} 15:00:00",
                    )
                except Exception as e:
                    logger.warning(
                        f"[{ts_code}-{trade_date}] 第 {attempt}/{_MIN_FETCH_MAX_RETRIES} 次 fetch 异常：{e}"
                    )

            if not raw_df.empty:
                # 成功：清零股票级失败计数，入库，返回
                _throttle_on_success()
                self.clean_and_insert_kline_min(raw_df, table_name)
                try:
                    df = db.query(sql, params=(ts_code, trade_date), return_df=True)
                    if not df.empty:
                        df["trade_time"] = pd.to_datetime(df["trade_time"])
                        logger.debug(
                            f"[{ts_code}-{trade_date}] 第 {attempt} 次拉取成功，入库后行数：{len(df)}"
                        )
                        return df
                except Exception as e:
                    logger.error(f"[{ts_code}-{trade_date}] 入库后查库失败：{e}")
                return pd.DataFrame()

            # 本次 API 返回空 → 仅记录单次警告 + 指数退避等待，不计入股票级失败数
            # （只有所有重试耗尽才算一只股票的「永久失败」，才影响限流状态机）
            backoff = min(2 ** (attempt - 1), 30)
            logger.warning(
                f"[{ts_code}-{trade_date}] 第 {attempt}/{_MIN_FETCH_MAX_RETRIES} 次拉取返回空"
                f"{'（限流模式：已额外等待）' if mode == 'throttled' else ''}，{backoff}s 后重试"
            )
            time.sleep(backoff)

        # ── Step 3: 所有重试耗尽 → 计入股票级永久失败 ───────────────────
        # 此时才触发限流状态机判断：N 只股票永久失败 → throttled / abort
        mode = _throttle_on_stock_perm_fail()
        logger.error(
            f"[{ts_code}-{trade_date}] {_MIN_FETCH_MAX_RETRIES} 次重试全部失败，"
            f"纳入调用方聚合告警，跳过该股票"
        )
        if mode == "abort":
            raise TushareRateLimitAbort(
                f"[{ts_code}][{trade_date}] 触发 abort 模式（连续 {_THROTTLE_ABORT_STOCK_STREAK} 只股票永久失败）"
            )
        return pd.DataFrame()




    def _clean_trade_cal_data(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """交易日历数据清洗（精简逻辑）"""
        if raw_df.empty:
            return pd.DataFrame()

        df_clean = raw_df.copy()
        # 日期格式化
        date_fields = ["cal_date", "pretrade_date"]
        for field in date_fields:
            if field in df_clean.columns:
                df_clean[field] = pd.to_datetime(df_clean[field], errors="coerce").dt.date
                df_clean.dropna(subset=[field], inplace=True)

        # is_open类型转换
        if "is_open" in df_clean.columns:
            df_clean["is_open"] = pd.to_numeric(df_clean["is_open"], errors="coerce").fillna(0).astype(int)

        # 去重
        df_clean.drop_duplicates(subset=["exchange", "cal_date"], keep="last", inplace=True)

        logger.debug(f"交易日历数据清洗完成：原始{len(raw_df)}行 → 清洗后{len(df_clean)}行")
        return df_clean

    def clean_and_insert_trade_cal(self, raw_df: pd.DataFrame, table_name: str = "trade_cal") -> Optional[int]:
        """交易日历数据清洗入库（精简冗余逻辑）"""
        if raw_df.empty:
            return 0

        clean_df = self._clean_trade_cal_data(raw_df)
        if clean_df.empty:
            return 0

        # 对齐数据库字段
        final_df = self._align_df_with_db(clean_df, table_name, exclude_columns=["created_at"])

        # 批量入库
        try:
            affected_rows = db.batch_insert_df(
                df=final_df,
                table_name=table_name,
                ignore_duplicate=True
            )
            logger.debug(f"交易日历数据入库完成，影响行数：{affected_rows}")
            return affected_rows
        except Exception as e:
            logger.error(f"交易日历数据入库失败：{str(e)}", exc_info=True)
            return None

    def get_trade_dates(self, start_date: str, end_date: str, table_name: str = "trade_cal") -> List[str]:
        """获取指定时间段内的交易日列表（精简逻辑）"""
        sql = """
            SELECT cal_date
            FROM {table}
            WHERE cal_date BETWEEN %s AND %s AND is_open = 1
            ORDER BY cal_date ASC
        """.format(table=table_name)

        df = db.query(sql, params=(start_date, end_date), return_df=True)
        if df.empty:
            logger.warning(f"未查询到{start_date}至{end_date}的交易日数据")
            return []

        return df["cal_date"].astype(str).tolist()

    def get_pre_trade_date(self, current_date: str, table_name: str = "trade_cal") -> Optional[str]:
        """获取指定日期的上一个交易日（精简逻辑）"""
        sql = """
            SELECT pretrade_date
            FROM {table}
            WHERE cal_date = %s
        """.format(table=table_name)

        df = db.query(sql, params=(current_date,), return_df=True)
        if df.empty or pd.isna(df["pretrade_date"].iloc[0]):
            return None

        return df["pretrade_date"].iloc[0].strftime("%Y-%m-%d")



    def clean_and_insert_dividend(
            self,
            ts_code: str,
            skip_check: bool = False,
    ) -> Optional[int]:
        """
        分红送股数据入库（stock_dividend 表）。
        按 ts_code 全量拉取，幂等入库（ON DUPLICATE KEY 去重）。

        :param ts_code: 股票代码
        :param skip_check: 跳过 DB 存在性检查（调用方已批量确认不存在时设为 True）
        """
        table_name = "stock_dividend"
        try:
            if not skip_check:
                check_sql = "SELECT 1 FROM stock_dividend WHERE ts_code = %s LIMIT 1"
                if db.query(check_sql, (ts_code,)):
                    return 0

            raw_df = data_fetcher.fetch_dividend(ts_code=ts_code)
            if raw_df.empty:
                return 0

            # ====================== 修复后的日期字段清洗 ======================
            date_fields = ["record_date", "ex_date", "pay_date", "div_listdate", "announce_date"]
            for col in date_fields:
                if col in raw_df.columns:
                    # 步骤1：将 Tushare 返回的 'YYYYMMDD' 字符串转为 datetime
                    # errors='coerce' 会把无效值（如空字符串、'0'）转为 NaT
                    raw_df[col] = pd.to_datetime(raw_df[col], format="%Y%m%d", errors="coerce")

                    # 步骤2：【核心修复】
                    # - 有效日期：格式化为 MySQL 兼容的 'YYYY-MM-DD' 字符串
                    # - 无效日期（NaT）：替换为 None（pandas 会将 None 转为 MySQL 的 NULL）
                    raw_df[col] = raw_df[col].apply(
                        lambda x: x.strftime("%Y-%m-%d") if pd.notna(x) else None
                    )

            # 数值字段（保持原有逻辑）
            num_fields = ["stk_div", "stk_bo_rate", "stk_co_rate", "cash_div", "cash_div_tax"]
            for col in num_fields:
                if col in raw_df.columns:
                    raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce").fillna(0)

            raw_df["update_time"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            final_df = self._align_df_with_db(raw_df, table_name)
            if final_df.empty:
                return 0

            affected = db.batch_insert_df(final_df, table_name, ignore_duplicate=True)
            return affected or 0
        except Exception as e:
            logger.error(f"[dividend] {ts_code} 入库失败：{e}")
            return 0

    def clean_and_insert_kline_day_hfq(
            self,
            ts_code: Union[str, List[str]],
            start_date: str,
            end_date: Optional[str] = None,
    ) -> Optional[int]:
        """
        后复权日K线：查库→请求（adj=hfq）→入库（kline_day_hfq 表）。
        与 qfq 方法完全对称，参数相同，目标表和复权方式不同。
        """
        return self.clean_and_insert_kline_day_qfq(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            table_name="kline_day_hfq",
            adj="hfq",
        )

    def clean_and_insert_kline_day_qfq(
            self,  # 类方法必须有self，调用时由实例自动传递，无需手动传
            ts_code: Union[str, List[str]],
            start_date: str,
            end_date: Optional[str] = None,
            table_name: str = "kline_day_qfq",
            adj: str = "qfq",
    ) -> Optional[int]:
        """
        极简版：先查库再请求（修复参数传递错误）
        核心：仅保留「查库→请求→入库」核心逻辑，无冗余
        """
        # 处理结束日期（必要逻辑）
        if not end_date:
            end_date = datetime.datetime.now().strftime("%Y%m%d")
        else:
            if len(end_date) != 8:
                logger.error("结束日期格式错误（需YYYYMMDD），终止入库")
                return 0

        # 统一转列表+去重（必要逻辑）
        stock_codes = [ts_code] if isinstance(ts_code, str) else ts_code
        stock_codes = list(set(stock_codes))
        total_ingest_rows = 0

        # 核心逻辑：先查库，再请求
        for ts_code in stock_codes:
            try:
                # 1. 查库：检查首尾日期是否覆盖请求范围（仅有部分数据不算完整）
                check_sql = f"""
                    SELECT MIN(trade_date) AS min_d, MAX(trade_date) AS max_d
                    FROM {table_name}
                    WHERE ts_code = %s AND trade_date >= %s AND trade_date <= %s
                """
                check_result = db.query(check_sql, (ts_code, start_date, end_date))
                if check_result and check_result[0].get("min_d") and check_result[0].get("max_d"):
                    min_d = str(check_result[0]["min_d"]).replace("-", "")[:8]
                    max_d = str(check_result[0]["max_d"]).replace("-", "")[:8]
                    if min_d <= start_date and max_d >= end_date:
                        continue  # 首尾覆盖，数据完整

                # 2. 无数据则请求接口
                raw_df = data_fetcher.fetch_kline_day_qfq(ts_code, start_date=start_date, end_date=end_date, adj=adj)
                if raw_df.empty:
                    logger.warning(f"{ts_code} 无接口数据")
                    continue

                # 3. 清洗+入库（复用原有逻辑，无修改）
                cleaned_df = self._clean_kline_day_data(raw_df)
                final_df = self._align_df_with_db(cleaned_df, table_name)
                if final_df.empty:
                    continue

                affected_rows = db.batch_insert_df(final_df, table_name, ignore_duplicate=True)
                total_ingest_rows += affected_rows or 0

            except Exception as e:
                logger.error(f"{ts_code} 处理失败：{str(e)}")

        return total_ingest_rows



    def clean_and_insert_limit_list_ths(
            self,
            trade_date: Optional[str] = None,
            limit_type: Optional[str] = None,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            table_name: str = "limit_list_ths"
    ) -> Optional[int]:
        """涨跌停池数据清洗入库"""
        logger.debug(f"===== 开始涨跌停池数据入库 | trade_date={trade_date} limit_type={limit_type} =====")

        raw_df = data_fetcher.fetch_limit_list_ths(
            trade_date=trade_date, limit_type=limit_type,
            start_date=start_date, end_date=end_date
        )
        if raw_df.empty:
            logger.debug("涨跌停池原始数据为空，跳过入库")
            return 0

        cleaned_df = self._clean_special_fields(raw_df)
        if cleaned_df.empty:
            return 0

        final_df = self._align_df_with_db(cleaned_df, table_name)
        try:
            affected_rows = db.batch_insert_df(df=final_df, table_name=table_name, ignore_duplicate=True)
            logger.debug(f"涨跌停池入库完成 | 影响行数：{affected_rows}")
            return affected_rows
        except Exception as e:
            logger.error(f"涨跌停池入库失败：{str(e)}", exc_info=True)
            return None

    def clean_and_insert_limit_step(
            self,
            trade_date: Optional[str] = None,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            table_name: str = "limit_step"
    ) -> Optional[int]:
        """连板天梯数据清洗入库"""
        logger.info(f"===== 开始连板天梯数据入库 | trade_date={trade_date} =====")

        raw_df = data_fetcher.fetch_limit_step(
            trade_date=trade_date, start_date=start_date, end_date=end_date
        )
        if raw_df.empty:
            logger.debug("连板天梯原始数据为空，跳过入库")
            return 0

        cleaned_df = self._clean_special_fields(raw_df)
        if cleaned_df.empty:
            return 0

        final_df = self._align_df_with_db(cleaned_df, table_name)
        try:
            affected_rows = db.batch_insert_df(df=final_df, table_name=table_name, ignore_duplicate=True)
            logger.info(f"连板天梯入库完成 | 影响行数：{affected_rows}")
            return affected_rows
        except Exception as e:
            logger.error(f"连板天梯入库失败：{str(e)}", exc_info=True)
            return None

    def clean_and_insert_limit_cpt_list(
            self,
            trade_date: Optional[str] = None,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            table_name: str = "limit_cpt_list"
    ) -> Optional[int]:
        """最强板块数据清洗入库"""
        logger.info(f"===== 开始最强板块数据入库 | trade_date={trade_date} =====")

        raw_df = data_fetcher.fetch_limit_cpt_list(
            trade_date=trade_date, start_date=start_date, end_date=end_date
        )
        if raw_df.empty:
            logger.debug("最强板块原始数据为空，跳过入库")
            return 0

        cleaned_df = self._clean_special_fields(raw_df)
        if cleaned_df.empty:
            return 0

        final_df = self._align_df_with_db(cleaned_df, table_name)
        try:
            affected_rows = db.batch_insert_df(df=final_df, table_name=table_name, ignore_duplicate=True)
            logger.info(f"最强板块入库完成 | 影响行数：{affected_rows}")
            return affected_rows
        except Exception as e:
            logger.error(f"最强板块入库失败：{str(e)}", exc_info=True)
            return None

    def insert_stock_st(
            self,
            ts_code: Optional[str] = None,
            trade_date: Optional[str] = None,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None
    ) -> Optional[int]:
        """入库stock_st接口ST股票数据（确保trade_date为YYYY-MM-DD格式）"""
        logger.info(
            f"===== 开始入库ST数据 | ts_code={ts_code} | 日期范围={trade_date or f'{start_date}-{end_date}'} =====")
        rows = data_fetcher.fetch_stock_st(
            ts_code=ts_code,
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date
        )
        if rows.empty:
            logger.warning("ST原始数据为空，跳过入库")
            return 0

        # 复用通用清洗方法，自动转换trade_date为YYYY-MM-DD
        cleaned_df = self._clean_special_fields(rows)
        if cleaned_df.empty:
            logger.warning("ST数据清洗后为空，跳过入库")
            return 0

        # 打印日期格式验证日志（调试用，上线可保留）
        if "trade_date" in cleaned_df.columns:
            sample_dates = cleaned_df["trade_date"].head(3).tolist()
            logger.debug(f"ST数据trade_date格式验证（前3条）：{sample_dates}")

        try:
            affected_rows = db.batch_insert_df(
                df=cleaned_df,
                table_name="stock_risk_warning",
                ignore_duplicate=True
            )
            logger.info(f"✅ ST数据入库完成 | 目标表：stock_risk_warning | 影响行数：{affected_rows}")
            return affected_rows
        except Exception as e:
            logger.error(f"❌ stock_st数据入库失败：{str(e)}", exc_info=True)
            return None

    def clean_and_insert_stk_factor_pro(
            self,
            trade_date: str,
            table_name: str = "stk_factor_pro"
    ) -> Optional[int]:
        """
        全市场技术面因子清洗入库（按交易日，一次拉取当日所有股票）。

        接口文档：股票技术面因子(专业版) - stk_factor_pro（doc_id=328）
        建表 SQL：sql/create_stk_factor_pro.sql

        :param trade_date: 交易日，支持 YYYYMMDD / YYYY-MM-DD
        :param table_name: 目标表名，默认 stk_factor_pro
        :return: 入库行数，失败返回 None
        """
        trade_date_fmt = trade_date.replace("-", "")
        logger.info(f"===== 开始技术面因子数据入库 | trade_date={trade_date_fmt} =====")

        raw_df = data_fetcher.fetch_stk_factor_pro(trade_date=trade_date_fmt)
        if raw_df.empty:
            logger.warning(f"技术面因子原始数据为空，trade_date={trade_date_fmt}，跳过入库")
            return 0

        cleaned_df = self._clean_special_fields(raw_df)
        if cleaned_df.empty:
            return 0

        final_df = self._align_df_with_db(cleaned_df, table_name)
        if final_df.empty:
            logger.warning(f"技术面因子对齐DB列后为空（表{table_name}可能未建立），跳过入库")
            return 0

        try:
            affected_rows = db.batch_insert_df(df=final_df, table_name=table_name, ignore_duplicate=True)
            logger.info(f"技术面因子入库完成 | trade_date={trade_date_fmt} | 影响行数：{affected_rows}")
            return affected_rows
        except Exception as e:
            logger.error(f"技术面因子入库失败 | trade_date={trade_date_fmt}：{str(e)}", exc_info=True)
            return None


    def clean_and_insert_moneyflow_combined(
            self,
            trade_date: str,
            table_name: str = "moneyflow_combined"
    ) -> Optional[int]:
        """
        双源资金流向数据清洗合并入库（按交易日，一次拉取全市场）。

        建表 SQL：sql/create_moneyflow_combined.sql
        数据源：
          THS (moneyflow_ths)  — 大/中/小单净占比，历史更长
          DC  (moneyflow_dc)   — 超大/大/中/小单净占比 + 主力净占比，自 20230911 起

        合并策略：outer join on ts_code，两源各自填充对应字段列，NULL = 该源无当日数据。
        因子计算时对两源均值处理，单源可用时降级使用单源。

        :param trade_date: 交易日，支持 YYYYMMDD / YYYY-MM-DD
        :param table_name: 目标表名，默认 moneyflow_combined
        :return: 入库行数，失败返回 None
        """
        trade_date_fmt = trade_date.replace("-", "")
        logger.info(f"===== 开始资金流向双源入库 | trade_date={trade_date_fmt} =====")

        # ── 拉取两个数据源 ─────────────────────────────────────────────────
        ths_df = data_fetcher.fetch_moneyflow_ths(trade_date=trade_date_fmt)
        dc_df  = data_fetcher.fetch_moneyflow_dc(trade_date=trade_date_fmt)

        if ths_df.empty and dc_df.empty:
            logger.warning(f"资金流向两源均为空，trade_date={trade_date_fmt}，跳过入库")
            return 0

        # ── 清洗 THS ──────────────────────────────────────────────────────
        ths_clean = pd.DataFrame()
        if not ths_df.empty:
            ths_df = self._clean_special_fields(ths_df)
            if not ths_df.empty and "ts_code" in ths_df.columns:
                ths_clean = ths_df[["ts_code", "trade_date"]].copy()
                for src_col, dst_col in [
                    ("buy_lg_amount_rate", "ths_lg_net_rate"),
                    ("buy_md_amount_rate", "ths_md_net_rate"),
                    ("buy_sm_amount_rate", "ths_sm_net_rate"),
                    ("net_amount",         "ths_net_amount"),
                ]:
                    ths_clean[dst_col] = pd.to_numeric(
                        ths_df.get(src_col), errors="coerce"
                    ) if src_col in ths_df.columns else None

        # ── 清洗 DC ───────────────────────────────────────────────────────
        dc_clean = pd.DataFrame()
        if not dc_df.empty:
            dc_df = self._clean_special_fields(dc_df)
            if not dc_df.empty and "ts_code" in dc_df.columns:
                dc_clean = dc_df[["ts_code", "trade_date"]].copy()
                for src_col, dst_col in [
                    ("buy_elg_amount_rate", "dc_elg_net_rate"),
                    ("buy_lg_amount_rate",  "dc_lg_net_rate"),
                    ("buy_md_amount_rate",  "dc_md_net_rate"),
                    ("buy_sm_amount_rate",  "dc_sm_net_rate"),
                    ("net_amount_rate",     "dc_main_net_rate"),
                    ("net_amount",          "dc_net_amount"),
                ]:
                    dc_clean[dst_col] = pd.to_numeric(
                        dc_df.get(src_col), errors="coerce"
                    ) if src_col in dc_df.columns else None

        # ── 双源 outer merge ──────────────────────────────────────────────
        if not ths_clean.empty and not dc_clean.empty:
            merged = pd.merge(
                ths_clean, dc_clean,
                on=["ts_code", "trade_date"], how="outer"
            )
        elif not ths_clean.empty:
            merged = ths_clean
        else:
            merged = dc_clean

        if merged.empty:
            logger.warning(f"资金流向合并后为空，trade_date={trade_date_fmt}，跳过入库")
            return 0

        final_df = self._align_df_with_db(merged, table_name)
        if final_df.empty:
            logger.warning(f"资金流向对齐DB列后为空（表{table_name}可能未建立），跳过入库")
            return 0

        try:
            affected = db.batch_insert_df(df=final_df, table_name=table_name, ignore_duplicate=True)
            logger.info(f"资金流向双源入库完成 | trade_date={trade_date_fmt} | 影响行数：{affected}")
            return affected
        except Exception as e:
            logger.error(f"资金流向双源入库失败 | trade_date={trade_date_fmt}：{e}", exc_info=True)
            return None

    def clean_and_insert_ths_hot(
            self,
            trade_date: str,
            market: str = "热股",
            is_new: str = "Y",
            table_name: str = "ths_hot"
    ) -> Optional[int]:
        """
        同花顺热榜数据清洗入库（按交易日，一次拉取当日热股榜）。

        接口文档：同花顺热榜数据 - ths_hot
        建表 SQL：sql/create_ths_hot.sql

        参数约定：
          历史因子补全：is_new='Y'（22:30后已固定的每日最终榜，无未来函数）
          每日自动更新（19:00）：is_new='N'（盘后最新小时快照，22:30前的最佳近似）

        :param trade_date: 交易日，支持 YYYYMMDD / YYYY-MM-DD
        :param market:     热榜类型，默认 '热股'
        :param is_new:     'Y' 取最终日榜，'N' 取最新小时快照
        :param table_name: 目标表名，默认 ths_hot
        :return: 入库行数，失败返回 None
        """
        trade_date_fmt = trade_date.replace("-", "")
        logger.info(f"===== 开始同花顺热榜数据入库 | trade_date={trade_date_fmt} is_new={is_new} =====")

        raw_df = data_fetcher.fetch_ths_hot(
            trade_date=trade_date_fmt, market=market, is_new=is_new
        )
        if raw_df.empty:
            logger.warning(f"同花顺热榜原始数据为空，trade_date={trade_date_fmt}，跳过入库")
            return 0

        # 旧数据 trade_date 可能含尾部空格（如 '20230928 '），需先 strip
        if "trade_date" in raw_df.columns:
            raw_df = raw_df.copy()
            raw_df["trade_date"] = raw_df["trade_date"].astype(str).str.strip()

        cleaned_df = self._clean_special_fields(raw_df)
        if cleaned_df.empty:
            return 0

        final_df = self._align_df_with_db(cleaned_df, table_name)
        if final_df.empty:
            logger.warning(f"同花顺热榜对齐DB列后为空（表{table_name}可能未建立），跳过入库")
            return 0

        try:
            affected_rows = db.batch_insert_df(df=final_df, table_name=table_name, ignore_duplicate=True)
            logger.info(f"同花顺热榜入库完成 | trade_date={trade_date_fmt} | 影响行数：{affected_rows}")
            return affected_rows
        except Exception as e:
            logger.error(f"同花顺热榜入库失败 | trade_date={trade_date_fmt}：{str(e)}", exc_info=True)
            return None


    def clean_and_insert_ths_daily(
            self,
            trade_date: str,
            table_name: str = "ths_daily",
    ) -> int:
        """
        同花顺板块指数日行情清洗入库（按交易日，一次拉取全量板块当日数据）。

        接口文档：doc_id=260（ths_daily）
        建表 SQL：sql/create_ths_daily.sql

        注意：API 返回字段 `change` 是 MySQL 保留字，入库前重命名为 `change_val`。

        :param trade_date: 交易日，支持 YYYYMMDD / YYYY-MM-DD
        :param table_name: 目标表名，默认 ths_daily
        :return: 入库行数，失败返回 0
        """
        trade_date_fmt = trade_date.replace("-", "")
        logger.info(f"===== 开始 ths_daily 板块日行情入库 | trade_date={trade_date_fmt} =====")

        raw_df = data_fetcher.fetch_ths_daily(trade_date=trade_date_fmt)
        if raw_df.empty:
            logger.warning(f"[ths_daily] 原始数据为空，trade_date={trade_date_fmt}，跳过入库")
            return 0

        # API 返回字段 `change` 是 MySQL 保留字，重命名为 change_val
        if "change" in raw_df.columns:
            raw_df = raw_df.rename(columns={"change": "change_val"})

        cleaned_df = self._clean_special_fields(raw_df)
        if cleaned_df.empty:
            return 0

        final_df = self._align_df_with_db(cleaned_df, table_name)
        if final_df.empty:
            logger.warning(f"[ths_daily] 对齐DB列后为空（表 {table_name} 可能未建立），跳过入库")
            return 0

        try:
            affected = db.batch_insert_df(df=final_df, table_name=table_name, ignore_duplicate=True)
            logger.info(f"[ths_daily] 入库完成 | trade_date={trade_date_fmt} | 影响行数：{affected}")
            return affected or 0
        except Exception as e:
            logger.error(f"[ths_daily] 入库失败 | trade_date={trade_date_fmt}：{e}", exc_info=True)
            return 0

    def clean_and_insert_ths_index(
            self,
            table_name: str = "ths_index",
    ) -> Optional[int]:
        """
        全量拉取同花顺板块指数并入库（ths_index 表）。
        一次 API 调用即可获取全部板块，无需循环，幂等（ON DUPLICATE KEY UPDATE）。

        数据粒度：板块维度（一行一个板块）
        更新策略：全量 upsert，适合每日刷新 count 和新增板块

        :param table_name: 目标表名，默认 ths_index
        :return: 入库行数，失败返回 None
        """
        logger.info("===== 开始同花顺板块指数入库（ths_index）=====")

        raw_df = data_fetcher.fetch_ths_index()
        if raw_df.empty:
            logger.warning("[ths_index] 原始数据为空，跳过入库")
            return 0

        df = raw_df.copy()

        # list_date：YYYYMMDD → YYYY-MM-DD（无效值设 None）
        if "list_date" in df.columns:
            df["list_date"] = pd.to_datetime(
                df["list_date"], format="%Y%m%d", errors="coerce"
            ).apply(lambda x: x.strftime("%Y-%m-%d") if pd.notna(x) else None)

        # count 强制 INT，NaN 填 0
        if "count" in df.columns:
            df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0).astype(int)

        # 字符串字段空值填 ''
        for col in ["name", "exchange", "type"]:
            if col in df.columns:
                df[col] = df[col].fillna("")

        df["update_time"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        final_df = self._align_df_with_db(df, table_name)
        if final_df.empty:
            logger.warning(f"[ths_index] 对齐 DB 列后为空（表 {table_name} 可能未建立），跳过入库")
            return 0

        try:
            affected = db.batch_insert_df(final_df, table_name, ignore_duplicate=True)
            logger.info(f"[ths_index] 入库完成，影响行数：{affected}")
            return affected or 0
        except Exception as e:
            logger.error(f"[ths_index] 入库失败：{e}", exc_info=True)
            return None

    def clean_and_insert_ths_member_batch(
            self,
            ts_code_list: Optional[List[str]] = None,
            table_name: str = "ths_member",
    ) -> int:
        """
        批量拉取同花顺板块成分并入库（ths_member 表）。

        数据粒度：板块-股票关系（一行一个 (板块, 股票) 对）
        更新策略：
          1. 对每个板块，先将该板块所有现有记录标记为 is_new='N'（旧成员退出标记）
          2. 再 upsert 当前最新成员（is_new='Y'），ON DUPLICATE KEY UPDATE
          → 退出板块的股票自动降为 is_new='N'，新入板块的股票被正确写入

        :param ts_code_list: 要更新的板块代码列表；None 则自动从 ths_index 表取全量 A 股板块
        :param table_name:   目标表名，默认 ths_member
        :return: 累计 upsert 行数
        """
        # 若未指定板块列表，从 ths_index 取全量 A 股板块
        if ts_code_list is None:
            try:
                rows = db.query("SELECT ts_code FROM ths_index WHERE exchange = 'A'")
                ts_code_list = [r["ts_code"] for r in rows if r.get("ts_code")]
            except Exception as e:
                logger.error(f"[ths_member] 读取 ths_index 板块列表失败：{e}")
                return 0

        if not ts_code_list:
            logger.warning("[ths_member] 板块列表为空，跳过入库")
            return 0

        total = len(ts_code_list)
        logger.info(f"===== 开始同花顺板块成分入库（ths_member），共 {total} 个板块 =====")

        total_affected = 0
        failed_boards = []

        for idx, board_code in enumerate(ts_code_list, 1):
            try:
                raw_df = data_fetcher.fetch_ths_member(ts_code=board_code)
                if raw_df.empty:
                    logger.debug(f"[ths_member] {board_code} 无成员数据，跳过")
                    continue

                df = raw_df.copy()

                # 日期字段：YYYYMMDD → YYYY-MM-DD（暂无数据，保持鲁棒）
                for col in ["in_date", "out_date"]:
                    if col in df.columns:
                        df[col] = pd.to_datetime(
                            df[col], format="%Y%m%d", errors="coerce"
                        ).apply(lambda x: x.strftime("%Y-%m-%d") if pd.notna(x) else None)

                # weight：数值，NaN 保持 None（暂无数据）
                if "weight" in df.columns:
                    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")

                # 字符串字段
                for col in ["con_name", "is_new"]:
                    if col in df.columns:
                        df[col] = df[col].fillna("")

                df["update_time"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                final_df = self._align_df_with_db(df, table_name)
                if final_df.empty:
                    continue

                # Step 1：将该板块已有记录全部标记为 is_new='N'（先降级旧成员）
                try:
                    db.execute(
                        "UPDATE ths_member SET is_new='N', update_time=NOW() WHERE ts_code=%s",
                        (board_code,)
                    )
                except Exception as e:
                    logger.warning(f"[ths_member] {board_code} 降级旧成员失败：{e}，继续插入")

                # Step 2：upsert 当前最新成员（is_new='Y'）
                affected = db.batch_insert_df(final_df, table_name, ignore_duplicate=True)
                total_affected += affected or 0

                if idx % 100 == 0 or idx == total:
                    logger.info(f"[ths_member] 进度 {idx}/{total}，累计入库 {total_affected} 行")

            except Exception as e:
                logger.warning(f"[ths_member] {board_code} 处理异常：{e}，跳过")
                failed_boards.append(board_code)

        if failed_boards:
            logger.warning(f"[ths_member] {len(failed_boards)} 个板块处理失败：{failed_boards[:10]}...")

        logger.info(f"[ths_member] 全部完成，累计 upsert {total_affected} 行，失败 {len(failed_boards)} 个板块")
        return total_affected


# 全局实例（保持不变，确保下游调用）
data_cleaner = DataCleaner()

