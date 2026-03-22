import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
from typing import Optional, Dict, Any, List, Union
import pandas as pd
from  utils.common_tools import retry_decorator
import tushare as ts
import tushare.pro.client as _ts_client
from utils.log_utils import logger
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===================== 通用常量配置（统一管理，提升可维护性） =====================
API_REQUEST_INTERVAL = 0.2  # Tushare接口限流间隔（秒），统一管理
TS_TOKEN_DEFAULT =  os.getenv("TS_TOKEN_DEFAULT")
TUSHARE_API_URL = "http://tushare.xyz"  # Tushare接口地址，统一配置
DEFAULT_PAGE_LIMIT = 8000  # 分钟线接口分页大小（适配Tushare接口限制）


def _filter_empty_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """通用工具：过滤参数字典中的空值/空字符串，避免Tushare接口报错"""
    return {k: v for k, v in params.items() if v is not None and v != ""}


def _format_date(date_str: Optional[str]) -> Optional[str]:
    """通用工具：将YYYY-MM-DD格式日期转为YYYYMMDD（适配Tushare接口要求），空值返回None"""
    return date_str.replace("-", "") if date_str else None


# ===================== 初始化函数（精简冗余注释，保留核心逻辑） =====================
def initialize_tushare():
    """初始化Tushare Pro API，读取环境变量Token，失败则抛出异常"""
    # 获取Token（优先环境变量，兜底默认值）
    TS_TOKEN = os.getenv("TS_TOKEN", TS_TOKEN_DEFAULT)
    try:
        ts.set_token(TS_TOKEN)
        pro = ts.pro_api()
        pro._DataApi__http_url = TUSHARE_API_URL  # 实例级别（self.pro.xxx 调用时生效）
        _ts_client.DataApi._DataApi__http_url = TUSHARE_API_URL  # 类级别（ts.pro_bar 调用时生效）
        logger.info("Tushare Pro API 初始化成功")
        return pro
    except Exception as e:
        logger.critical(f"Tushare Pro API 初始化失败：{str(e)}")
        raise RuntimeError(f"Tushare初始化失败：{str(e)}")


# ===================== 核心数据获取类（仅修改fetch_stk_mins，其余完全保留） =====================
class DataFetcher:
    """数据请求层核心类（强制返回接口所有字段+保留中文）"""

    def __init__(self):
        self.pro = initialize_tushare()

    def get_stockbase(
            self,
            exchange: Optional[str] = None,
            list_status: str = "L",
            **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        获取股票基础信息（全字段返回）
        接口文档：https://tushare.pro/document/2?doc_id=25

        Args:
            exchange: 交易所代码（如SHSE/SZSE），可选
            list_status: 上市状态 L-上市 D-退市 P-暂停上市，默认L
            **kwargs: 其他扩展参数（接口支持则生效）

        Returns:
            股票基础信息DataFrame，失败/空数据返回None
        """
        # 构造并过滤参数
        params = _filter_empty_params({
            "list_status": list_status,
            "exchange": exchange, **kwargs
        })

        # Tushare stock_basic全字段列表（官方文档v2版本）
        ALL_FIELDS = [
            "ts_code", "symbol", "name", "area", "industry", "cnspell", "market",
            "list_date", "act_name", "act_ent_type", "exchange", "curr_type",
            "list_status", "delist_date", "is_hs", "enname", "fullname"
        ]

        try:
            logger.debug(f"获取股票基础数据，参数：{params}")
            df = self.pro.stock_basic(**params, fields=",".join(ALL_FIELDS))

            if df.empty:
                logger.warning(f"股票基础数据接口返回空，参数：{params}")
            else:
                logger.debug(f"股票基础数据获取成功：行数={len(df)}，字段数={len(df.columns)}")

            return df
        except Exception as e:
            logger.error(f"股票基础数据获取失败，参数：{params}，错误：{str(e)}")
            return None

    def get_stock_company(
            self,
            ts_code: Optional[str] = None,
            exchange: Optional[str] = None,
            **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        获取上市公司基本信息（stock_company接口，doc_id=112）
        接口文档：https://tushare.pro/document/2?doc_id=112

        Args:
            ts_code: 股票代码（如600000.SH），可选
            exchange: 交易所代码（SHSE/SZSE/BSE），可选
            **kwargs: 其他扩展参数（如limit/offset）

        Returns:
            上市公司信息DataFrame，失败/空数据返回None
        """
        params = {
            "ts_code": ts_code,
            "exchange": exchange, **kwargs
        }

        try:
            logger.info(f"获取上市公司基本信息，参数：{params}")
            df = self.pro.stock_company(**params)

            if df.empty:
                logger.warning(f"上市公司基本信息返回空，参数：{params}")
            else:
                logger.info(f"上市公司基本信息获取成功：行数={len(df)}，字段数={len(df.columns)}")
                # 验证核心字段是否存在
                core_fields = ["ts_code", "chairman", "manager"]
                missing_fields = [f for f in core_fields if f not in df.columns]
                if missing_fields:
                    logger.warning(f"核心字段缺失：{missing_fields}（权限/接口版本问题）")

            return df
        except Exception as e:
            logger.error(f"上市公司基本信息获取失败，参数：{params}，错误：{str(e)}")
            return None

    def fetch_kline_day(
            self,
            ts_code: Optional[str] = None,
            trade_date: Optional[str] = None,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        获取A股日K线历史数据（daily接口）
        接口文档：https://tushare.pro/document/2?doc_id=27

        Args:
            ts_code: 股票代码（如600000.SH）
            trade_date: 单交易日（YYYYMMDD）
            start_date: 开始日期（YYYYMMDD）
            end_date: 结束日期（YYYYMMDD）

        Returns:
            日K线DataFrame（空数据返回空DataFrame）
        """
        params = _filter_empty_params({
            "ts_code": ts_code,
            "trade_date": trade_date,
            "start_date": start_date,
            "end_date": end_date
        })
        # logger.debug(f" 调用行情接口: {trade_date}")
        try:
            kline_df = self.pro.daily(**params)
            logger.debug(f"日线数据获取，参数：{params}，行数：{len(kline_df)}")
            return kline_df
        except Exception as e:
            logger.error(f"日线数据获取失败，参数：{params}，错误：{str(e)}")
            return pd.DataFrame()

    @retry_decorator(max_retries=3, retry_interval=2.0)
    def fetch_kline_day_qfq(
            self,
            ts_code: Optional[str] = None,
            trade_date: Optional[str] = None,  # 兼容原参数，接口未使用
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            adj: Optional[str] = None  # 兼容原参数，实际固定为qfq
    ) -> pd.DataFrame:
        """
            前复权日K线DataFrame（空数据返回空DataFrame）
        """
        params = _filter_empty_params({
            "ts_code": ts_code,
            "start_date": start_date,
            "end_date": end_date,
            "adj": adj or "qfq",  # 默认前复权；传入 "hfq" 可获取后复权数据
            "asset": "E",
            "freq": "D",
        })
        try:
            kline_qfq_df = ts.pro_bar(**params)
            logger.debug(f"前复权日线数据获取，参数：{params}，行数：{len(kline_qfq_df)}")
            return kline_qfq_df
        except Exception as e:
            logger.error(f"前复权日线数据获取失败，参数：{params}，错误：{str(e)}")
            return pd.DataFrame()

    @retry_decorator(max_retries=3, retry_interval=1.0)
    def fetch_index_daily(
            self,
            ts_code: Optional[str] = None,
            trade_date: Optional[str] = None,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        获取指数历史日线数据（index_daily接口，doc_id=403）
        接口文档：https://tushare.pro/document/2?doc_id=403

        Args:
            ts_code: 指数代码（如000001.SH）
            trade_date: 单交易日（YYYYMMDD）
            start_date: 开始日期（YYYYMMDD）
            end_date: 结束日期（YYYYMMDD）

        Returns:
            指数日线DataFrame（空数据返回空DataFrame）
        """
        if not ts_code or ts_code.strip() == "":
            ts_code = "000001.SH"
            logger.debug("ts_code未传入或为空值，自动使用默认值：上证指数（000001.SH）")

        params = _filter_empty_params({
            "ts_code": ts_code,
            "trade_date": trade_date,
            "start_date": start_date,
            "end_date": end_date,
        })

        try:
            index_df = self.pro.index_daily(**params)
            logger.debug(f"指数日线数据获取，参数：{params}，行数：{len(index_df)}")

            if index_df.empty:
                logger.debug(f"指数日线数据返回空，参数：{params}")

            return index_df
        except Exception as e:
            logger.error(f"指数日线数据获取失败，参数：{params}，错误：{str(e)}")
            return pd.DataFrame()

    @retry_decorator()  # 应用重试装饰器，自动读取.env配置的参数
    def fetch_stk_mins(
            self,
            ts_code: Union[str, List[str]],
            freq: str = "1min",
            start_date: str = None,
            end_date: str = None
    ) -> pd.DataFrame:
        """
        获取A股股票分钟线数据（stk_mins接口，doc_id=370）
        接口文档：https://tushare.pro/document/2?doc_id=370

        Args:
            ts_code: 股票代码（如600000.SH）
            freq: 分钟频度（1min/5min/15min/30min/60min），默认1min
            start_date: 开始时间（格式：2023-08-25 09:00:00）
            end_date: 结束时间（格式：2023-08-25 15:00:00）

        Returns:
            分钟线DataFrame（空数据返回空DataFrame）
        """

        params = _filter_empty_params({
            "ts_code": ts_code,
            "freq": freq,
            "start_date": start_date,
            "end_date": end_date
        })
        logger.debug(f"获取{ts_code}分钟线数据，参数：{params}")
        time.sleep(API_REQUEST_INTERVAL)
        mins_df = self.pro.stk_mins(**params)

        if mins_df.empty:
            logger.debug(f"{ts_code}分钟线数据为空，参数：{params}")

        logger.debug(f"{ts_code}分钟线数据获取完成，行数：{len(mins_df)}")
        return mins_df


    def fetch_trade_cal(
            self,
            start_date: str,
            end_date: str,
            exchange: str = "SSE",
            is_open: int = None
    ) -> pd.DataFrame:
        """
        获取A股交易日历数据（trade_cal接口，doc_id=26）
        接口文档：https://tushare.pro/document/2?doc_id=26

        Args:
            start_date: 开始日期（YYYY-MM-DD / YYYYMMDD）
            end_date: 结束日期（YYYY-MM-DD / YYYYMMDD）
            exchange: 交易所，默认SSE上交所（A股沪深交易日历一致）
            is_open: 是否交易，1=仅交易日，0=仅休市日，不传返回全部

        Returns:
            交易日历DataFrame（空数据返回空DataFrame）
        """
        # 统一格式化日期，适配接口要求
        params = _filter_empty_params({
            "exchange": exchange,
            "start_date": _format_date(start_date),
            "end_date": _format_date(end_date),
            "is_open": str(is_open) if is_open is not None else None
        })

        try:
            logger.debug(f"获取交易日历数据，参数：{params}")
            time.sleep(API_REQUEST_INTERVAL)  # 接口限流
            cal_df = self.pro.trade_cal(**params)

            if cal_df.empty:
                logger.warning(f"交易日历数据为空，参数：{params}")

            logger.debug(f"交易日历数据获取完成，行数：{len(cal_df)}")
            return cal_df
        except Exception as e:
            logger.error(f"交易日历数据获取失败，参数：{params}，错误：{str(e)}")
            return pd.DataFrame()

    def fetch_stock_daily_basic(
            self,
            ts_code: Optional[str] = None,
            trade_date: Optional[str] = None,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        获取全部股票每日重要的基本面指标，可用于选股分析、报表展示等。单次请求最大返回6000条数据，可按日线循环提取全部历史。
        接口文档：https://tushare.pro/document/2?doc_id=32

        Args:
            ts_code: 指数代码（如000001.SH）
            二选一
            trade_date: 单交易日（YYYYMMDD）
            start_date: 开始日期（YYYYMMDD）
            end_date: 结束日期（YYYYMMDD）

        Returns:
            指数日线DataFrame（空数据返回空DataFrame）
        """
        params = _filter_empty_params({
            "ts_code": ts_code,
            "trade_date": trade_date,
            "start_date": start_date,
            "end_date": end_date,
        })

        try:
            index_df = self.pro.daily_basic(**params)
            logger.debug(f"获取当日交易详细信息，参数：{ts_code}/{trade_date}，行数：{len(index_df)}")

            if index_df.empty:
                logger.warning(f"获取当日交易详细信息为空，参数：{params}")

            return index_df
        except Exception as e:
            logger.error(f"指数日线数据获取失败，参数：{params}，错误：{str(e)}")
            return pd.DataFrame()


    def fetch_stk_factor_pro(
            self,
            ts_code: Optional[str] = None,
            trade_date: Optional[str] = None,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        获取股票每日技术面因子数据（专业版），用于跟踪股票当前走势情况，数据覆盖全历史。
        接口文档：股票技术面因子(专业版) - stk_factor_pro
        限量：单次调取最多返回10000条数据，可以通过日期参数循环
        积分：5000积分每分钟可以请求30次，8000积分以上每分钟500次

        Args:
            ts_code: 股票代码（如000001.SZ、600000.SH）
            二选一
            trade_date: 单交易日（YYYYMMDD）
            start_date: 开始日期（YYYYMMDD）
            end_date: 结束日期（YYYYMMDD）

        Returns:
            股票技术面因子DataFrame（空数据返回空DataFrame）
        """
        params = _filter_empty_params({
            "ts_code": ts_code,
            "trade_date": trade_date,
            "start_date": start_date,
            "end_date": end_date,
        })

        try:
            factor_df = self.pro.stk_factor_pro(**params)
            logger.debug(f"获取股票专业版技术面因子数据，参数：{ts_code}/{trade_date}，行数：{len(factor_df)}")

            if factor_df.empty:
                logger.warning(f"获取股票专业版技术面因子数据为空，参数：{params}")

            return factor_df
        except Exception as e:
            logger.error(f"股票专业版技术面因子数据获取失败，参数：{params}，错误：{str(e)}")
            return pd.DataFrame()


    def fetch_ths_hot(
            self,
            trade_date: Optional[str] = None,
            ts_code: Optional[str] = None,
            market: Optional[str] = None,
            is_new: Optional[str] = None
    ) -> pd.DataFrame:
        """
        获取同花顺热榜数据，涵盖热股、ETF、行业板块等多类型热榜信息
        接口文档：同花顺热榜数据
        限量：未明确标注（保持原有风格）
        积分：未明确标注（保持原有风格）

        Args:
            trade_date: 交易日期（格式未指定，按接口要求传入）
            ts_code: TS代码（如000001.SZ、600000.SH）
            market: 热榜类型(热股、ETF、可转债、行业板块、概念板块、期货、港股、热基、美股)
            is_new: 是否最新（默认Y，如果为N则为盘中和盘后阶段采集，具体时间可参考rank_time字段，状态N每小时更新一次，状态Y更新时间为22：30）

        Returns:
            同花顺热榜数据DataFrame（空数据返回空DataFrame）
        """
        params = _filter_empty_params({
            "trade_date": trade_date,
            "ts_code": ts_code,
            "market": market,
            "is_new": is_new,
        })

        try:
            hot_df = self.pro.ths_hot(** params)  # 适配新接口名ths_hot
            logger.debug(f"获取同花顺热榜数据，参数：{ts_code}/{trade_date}/{market}，行数：{len(hot_df)}")

            if hot_df.empty:
                logger.warning(f"获取同花顺热榜数据为空，参数：{params}")

            return hot_df
        except Exception as e:
            logger.error(f"同花顺热榜数据获取失败，参数：{params}，错误：{str(e)}")
            return pd.DataFrame()

    @retry_decorator(max_retries=3, retry_interval=2.0)
    def fetch_limit_list_ths(
            self,
            trade_date: Optional[str] = None,
            ts_code: Optional[str] = None,
            limit_type: Optional[str] = None,
            market: Optional[str] = None,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        获取涨跌停数据（底层接口已迁移至 limit_list_d，比旧 limit_list_ths 字段更丰富）

        Args:
            trade_date  : 交易日（YYYYMMDD）
            ts_code     : 股票代码
            limit_type  : 类型，支持中文（涨停池/跌停池/炸板池）或 API 原生码（U/D/Z）
            market      : 交易所（SH/SZ/BJ），透传为 exchange 参数
            start_date  : 开始日期（YYYYMMDD）
            end_date    : 结束日期（YYYYMMDD）

        Returns:
            DataFrame，含 limit_type 列（中文，与 DB 存储格式兼容）
        """
        # 中文 → API 参数映射（兼容旧调用方传入中文 limit_type）
        _TO_API = {"涨停池": "U", "跌停池": "D", "炸板池": "Z"}
        _TO_CN  = {"U": "涨停池", "D": "跌停池", "Z": "炸板池"}

        api_limit_type = _TO_API.get(limit_type, limit_type) if limit_type else None

        params = _filter_empty_params({
            "trade_date": trade_date,
            "ts_code":    ts_code,
            "limit_type": api_limit_type,
            "exchange":   market,          # 旧参数 market 映射到新接口的 exchange
            "start_date": start_date,
            "end_date":   end_date,
        })

        try:
            time.sleep(API_REQUEST_INTERVAL)
            df = self.pro.limit_list_d(**params)
            if df is None or df.empty:
                logger.debug(f"limit_list_d 无数据，参数：{params}")
                return pd.DataFrame()

            # 将 'limit' 列（U/D/Z）映射为中文 limit_type，与 DB 存储格式对齐
            if "limit" in df.columns:
                df["limit_type"] = df["limit"].map(_TO_CN).fillna(df["limit"])

            logger.debug(f"limit_list_d 获取成功，参数：{params}，行数：{len(df)}")
            return df
        except Exception as e:
            logger.error(f"limit_list_d 接口获取失败，参数：{params}，错误：{str(e)}")
            return pd.DataFrame()

    def fetch_limit_step(
            self,
            trade_date: Optional[str] = None,
            ts_code: Optional[str] = None,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            nums: Optional[str] = None
    ) -> pd.DataFrame:
        """
        获取涨停连板天梯数据（limit_step 接口）

        Args:
            trade_date: 交易日期（YYYYMMDD）
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            nums: 连板次数，支持多个输入如 nums='2,3'

        Returns:
            连板天梯 DataFrame（空数据返回空 DataFrame）
        """
        params = _filter_empty_params({
            "trade_date": trade_date,
            "ts_code": ts_code,
            "start_date": start_date,
            "end_date": end_date,
            "nums": nums,
        })

        try:
            time.sleep(API_REQUEST_INTERVAL)
            df = self.pro.limit_step(**params)
            logger.debug(f"连板天梯数据获取，参数：{params}，行数：{len(df)}")
            if df.empty:
                logger.debug(f"连板天梯数据为空，参数：{params}")
            return df
        except Exception as e:
            logger.error(f"连板天梯数据获取失败，参数：{params}，错误：{str(e)}")
            return pd.DataFrame()

    def fetch_limit_cpt_list(
            self,
            trade_date: Optional[str] = None,
            ts_code: Optional[str] = None,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        获取最强板块数据（limit_cpt_list 接口）

        Args:
            trade_date: 交易日期（YYYYMMDD）
            ts_code: 板块代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            最强板块 DataFrame（空数据返回空 DataFrame）
        """
        params = _filter_empty_params({
            "trade_date": trade_date,
            "ts_code": ts_code,
            "start_date": start_date,
            "end_date": end_date,
        })

        try:
            time.sleep(API_REQUEST_INTERVAL)
            df = self.pro.limit_cpt_list(**params)
            logger.debug(f"最强板块数据获取，参数：{params}，行数：{len(df)}")
            if df.empty:
                logger.debug(f"最强板块数据为空，参数：{params}")
            return df
        except Exception as e:
            logger.error(f"最强板块数据获取失败，参数：{params}，错误：{str(e)}")
            return pd.DataFrame()

    def fetch_stock_st(
            self,
            ts_code: Optional[str] = None,
            trade_date: Optional[str] = None,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        获取ST股票列表（stock_st接口），支持按日期/股票代码查询
        接口文档：获取ST股票列表，每日9:20更新，20160101起数据
        限量：单次最大1000行
        积分：3000积分起

        Args:
            ts_code: 股票代码
            trade_date: 交易日期（YYYYMMDD）
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            ST股票列表DataFrame，空数据返回空DataFrame
        """
        params = _filter_empty_params({
            "ts_code": ts_code,
            "trade_date": trade_date,
            "start_date": start_date,
            "end_date": end_date,
        })

        try:
            st_df = self.pro.stock_st(**params)
            logger.debug(f"获取stock_st接口数据，参数：{params}，行数：{len(st_df)}")

            if st_df.empty:
                logger.warning(f"获取stock_st接口数据为空，参数：{params}")
            return st_df
        except Exception as e:
            logger.error(f"stock_st接口数据获取失败，参数：{params}，错误：{str(e)}")
            return pd.DataFrame()

    def fetch_st_list(
            self,
            ts_code: Optional[str] = None,
            pub_date: Optional[str] = None,
            imp_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        获取ST风险警示板股票列表（st接口）
        接口文档：ST风险警示板股票列表
        限量：单次最大1000行
        积分：6000积分起

        Args:
            ts_code: 股票代码
            pub_date: 发布日期
            imp_date: 实施日期

        Returns:
            ST风险警示股票列表DataFrame，空数据返回空DataFrame
        """
        params = _filter_empty_params({
            "ts_code": ts_code,
            "pub_date": pub_date,
            "imp_date": imp_date,
        })

        try:
            st_df = self.pro.st(**params)
            logger.debug(f"获取st接口数据，参数：{params}，行数：{len(st_df)}")

            if st_df.empty:
                logger.warning(f"获取st接口数据为空，参数：{params}")

            # 修复接口字段笔误 st_tpye → st_type
            if "st_tpye" in st_df.columns:
                st_df.rename(columns={"st_tpye": "st_type"}, inplace=True)
            return st_df
        except Exception as e:
            logger.error(f"st接口数据获取失败，参数：{params}，错误：{str(e)}")
            return pd.DataFrame()

    @retry_decorator(max_retries=3, retry_interval=2.0)
    def fetch_dividend(
            self,
            ts_code: Optional[str] = None,
            ann_date: Optional[str] = None,
            ex_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        获取分红送股数据（Tushare pro.dividend 接口）。
        :param ts_code: 股票代码（必传其一：ts_code 或 ann_date）
        :param ann_date: 公告日（YYYYMMDD）
        :param ex_date: 除权除息日（YYYYMMDD）
        :return: DataFrame
        """
        params = _filter_empty_params({
            "ts_code": ts_code,
            "ann_date": ann_date,
            "ex_date": ex_date,
            "fields": "ts_code,div_proc,stk_div,stk_bo_rate,stk_co_rate,"
                       "cash_div,cash_div_tax,record_date,ex_date,"
                       "pay_date,div_listdate,announce_date,end_date",
        })
        try:
            df = self.pro.dividend(**params)
            logger.debug(f"[dividend] 参数：{params}，行数：{len(df)}")
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            logger.error(f"[dividend] 获取失败，参数：{params}，错误：{e}")
            return pd.DataFrame()


# ===================== 全局实例（保持原命名，不影响调用） =====================
data_fetcher = DataFetcher()

if __name__ == "__main__":
    # pass
    #     初始化DataFetcher实例
    #     data_fetcher = DataFetcher()
        # # pd.set_option('display.max_columns', None)
        # # pd.set_option('display.width', None)
        # logger.info("--- 测试3.1：按股票代码查询分红 ---")
        # dividend_by_code_df = data_fetcher.fetch_dividend(
        #     ts_code="000001.SZ"  # 平安银行（示例，可替换为任意股票）
        # )
        # if not dividend_by_code_df.empty:
        #     logger.info(f"按股票代码查询分红数据预览（前3行）：\n{dividend_by_code_df.head(3)}")
        # else:
        #     logger.warning("按股票代码查询分红数据为空（可能该股票暂无历史分红记录）")
    #
    # 1. 测试同花顺热榜-热股类型最新数据获取
    #     logger.info("===== 测试1：获取同花顺热榜-热股最新数据 =====")
    #     ths_hot_df = data_fetcher.fetch_ths_hot(
    #         trade_date="20260226",  # 测试日期（按需调整）
    #         market="热股",  # 热榜类型：热股
    #         is_new="Y"  # 获取最新数据
    #     )
    #     if not ths_hot_df.empty:
    #         logger.info(f"同花顺热榜数据预览（前3行）：\n{ths_hot_df}")
    #     else:
    #         logger.warning("同花顺热榜数据为空（可能日期未开盘/无数据）")
    #
    #     # 1. 测试单只股票单日期技术面因子获取
    #     logger.info("===== 测试1：获取单只股票单日期技术面因子 =====")
    #     single_stock_single_day_df = data_fetcher.fetch_stk_factor_pro(
    #         ts_code="000001.SZ",  # 平安银行
    #         trade_date="20260226"  # 测试日期（可根据实际调整）
    #     )
    #     if not single_stock_single_day_df.empty:
    #         logger.info(f"单股票单日因子数据预览（前3行）：\n{single_stock_single_day_df.head(3)}")
    #     else:
    #         logger.warning("单股票单日因子数据为空（可能日期未开盘/无数据）")

        # 2. 测试单只股票一段时间技术面因子获取
#     """测试stock_basic"""
#     try:
#         # 1. 测试股票基础数据获取（修正exchange参数：SHSE=上交所，空则返回全部）
#         logger.info("===== 测试：获取A股上市股票基础数据（全字段） =====")
#         stock_base_df = data_fetcher.get_stockbase(
#             exchange="SSE",  # 可选：仅上交所，注释则返回所有交易所
#             list_status="L",  # 仅上市股票
#         )
#         if stock_base_df is not None:
#             logger.info(f"股票基础数据预览（前3行）：\n{stock_base_df.head(3)}")
#
#         logger.info("\n===== 所有数据获取测试完成 ✅ =====")
#     except Exception as e:
#         logger.error(f"数据获取测试失败，错误信息：{str(e)} ❌")
    # """测试新增的stock_company接口"""
    # try:
        # 1. 测试获取单只股票的公司信息
    #     logger.info("===== 测试：获取单只股票（600000.SH）的公司基本信息 =====")
    #     single_company_df = data_fetcher.get_stock_company(ts_code="600000.SH")
    #     pd.set_option('display.max_columns', None)
    #     pd.set_option('display.width', None)
    #     print(single_company_df)
    #     # 2. 测试获取指定交易所的公司信息
    #     # logger.info("\n===== 测试：获取上交所所有上市公司基本信息 =====")
    #     # exchange_company_df = data_fetcher.get_stock_company(exchange="SSE")
    #     # if exchange_company_df is not None:
    #     #     logger.info(f"上交所公司信息行数：{len(exchange_company_df)}")
    #     #     logger.info(f"董事长字段示例：{exchange_company_df['chairman'].dropna().head(5).tolist()}")
    #
    # #     logger.info("\n===== 所有接口测试完成 ✅ =====")
    # except Exception as e:
    #     logger.error(f"接口测试失败，错误信息：{str(e)} ❌")
    # """测试新增的Kline_day接口"""
    # logger.info("===== 测试：获取日线数据 =====")
    # stock_Kline_day = data_fetcher.fetch_kline_day(
    #     ts_code="000426.SZ",
    #     trade_date="",
    #     start_date='20250516',
    #     end_date= '20250521'
    #
    # )
    # if stock_Kline_day is not None:
    #     logger.info(f"日线数据预览：\n{stock_Kline_day.head(3)}")
    # logger.info("\n=====获取日线数据测试完成 ✅ =====")
    # ===================== 新增：指数日线接口测试代码 =====================
    # """测试新增的index_daily接口"""
    # try:
    #     logger.info("===== 测试：获取指数日线数据 =====")
    #     # 测试1：单指数单日数据
    #     index_daily_single = data_fetcher.fetch_index_daily(
    #         ts_code="399107.SZ",
    #         trade_date="20241231"
    #     )
    #     if not index_daily_single.empty:
    #         logger.info(f"单指数单日数据预览：\n{index_daily_single.head(3)}")

        # 测试2：多指数单日数据（批量获取）
    #     index_daily_multi = data_fetcher.fetch_index_daily(
    #         ts_code="000001.SH",
    #         trade_date="",
    #         start_date= "20090101",
    #         end_date= "20260213"
    #     )
    #     if not index_daily_multi.empty:
    #         logger.info(f"指数数据行数：{len(index_daily_multi)}")
    #         logger.info(f"指数数据预览：\n{index_daily_multi.head(3)}")
    #
    #     logger.info("\n===== 获取指数日线数据测试完成 ✅ =====")
    # except Exception as e:
    #     logger.error(f"指数日线数据测试失败，错误信息：{str(e)} ❌")
# """===================== 分钟线接口（fetch_stk_mins）专项测试用例 ====================="""
#
#
    # try:
    #     # 全局配置：显示所有列，方便查看返回结果
    #     pd.set_option('display.max_columns', None)
    #     pd.set_option('display.width', None)
    # #
    #     logger.info("===== 测试1：有效股票+完整交易日分钟线（1min） =====")
    #     # 测试标的：000001.SZ（平安银行，确保有数据）
    #     # 测试日期：2026-01-05（你的回测日期，交易日）
    #     mins_df_1 = data_fetcher.fetch_stk_mins(
    #         ts_code="302132.SZ",
    #         freq="1min",
    #         start_date="2025-02-17 09:25:00",
    #         end_date="2025-02-17 15:00:00"
    #     )
    #     logger.info(f"✅ 有效股票分钟线获取成功，行数：{len(mins_df_1)}")
    #     logger.info(f"返回字段：{mins_df_1.columns.tolist()}")
    #     logger.info(f"前5行数据：\n{mins_df_1.head()}")
    # except Exception as e:
    #     logger.error(f"❌ 分钟线接口测试崩溃，核心错误：{str(e)}", exc_info=True)


    # logger.info("===== 测试：获取交易日历数据 =====")
    # # 调用fetch_trade_cal方法（参数说明：2025.05.16-2025.05.21，上交所，仅返回交易日）
    # trade_cal_df = data_fetcher.fetch_trade_cal(
    #     start_date='20250516',
    #     end_date='20250521',
    #     exchange="SSE",
    #     is_open=1  # 仅获取交易日，不传则返回全部（含休市）
    # )
    # # 数据预览
    # if not trade_cal_df.empty:
    #     logger.info(f"交易日历数据预览（仅交易日）：\n{trade_cal_df.head(10)}")
    #     logger.info(f"有效交易日数量：{len(trade_cal_df)}")
    # else:
    #     logger.warning("⚠️  未获取到交易日历数据！")
    # logger.info("\n===== 获取交易日历数据测试完成 ✅ =====\n")

    # TUSHARE_TOKEN = ""
    # ts.set_token(TUSHARE_TOKEN)  # 初始化token

    # 2. 调用接口获取数据（示例：浦发银行 2025年1月前复权日线）

    df = data_fetcher.fetch_kline_day_qfq(
        ts_code="000001.SZ",  # 股票代码
        start_date="20260301",# 开始日期
        end_date="20260322"   # 结束日期
    )

    # 3. 打印结果看看
    if not df.empty:
        print("\n===== 获取到的前复权日线数据 =====")
        print(f"数据形状：{df.shape}")  # 行数×列数
        print("\n前5行数据：")
        print(df.head())
        print("\n数据字段：", df.columns.tolist())
    else:
        print("未获取到数据，请检查Token/股票代码/日期是否正确")

    # =================== 测试：获取交易详细信息数据 =====")
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.width', None)
    # df = data_fetcher.fetch_stock_daily_basic(
    #     ts_code= '',
    #     trade_date = '20260226',
    #     start_date='',
    #     end_date='',
    # )
    # print(df)

