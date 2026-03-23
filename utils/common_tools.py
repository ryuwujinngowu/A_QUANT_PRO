import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import re
import threading
from collections import defaultdict
from datetime import datetime, timedelta
import pandas as pd
import time
from pathlib import Path
import functools
from typing import List, Dict, Optional, Set
from typing import Tuple
from config.config import MAIN_BOARD_LIMIT_UP_RATE, STAR_BOARD_LIMIT_UP_RATE, BJ_BOARD_LIMIT_UP_RATE
from utils.db_utils import db
from utils.log_utils import logger
from typing import List, Dict



# 增量更新配置（可统一维护）
UPDATE_RECORD_FILE = Path(__file__).parent.parent / "data" / "update_record.json"


default_exclude = [
    "融资融券", "转融券标的", "标普道琼斯A股", '核准制次新股', '腾讯概念', '阿里巴巴概念', '抖音概念','ST板块','央企国企改革',
    "MSCI概念", "深股通", "沪股通", '一带一路', '新股与次新股', '节能环保','稀缺资源','电子商务','俄乌冲突概念',
    "同花顺漂亮100", "富时罗素概念", "富时罗素概念股", '比亚迪概念', '5G', '小金属概念','参股银行','锂电池','spacex',
    "央企国资改革", "地方国资改革", "证金持股", '新能源汽车', '次新股', '宁德时代概念','人民币贬值受益','中俄贸易概念',
    "汇金持股", "养老金持股", "QFII重仓", "专精特新", 'MSCI中国', '半年报预增', '华为概念', '光伏概念', '储能','一季报预增'
]


def sort_by_recent_gain(df: pd.DataFrame, trade_date: str, day_count: int = 20) -> pd.DataFrame:
    """
    【通用任意天数涨幅排序】仅拉2天数据，性能最优
    计算标准（对齐行情软件）：
        近N日涨幅 = (今日收盘价 - 前第N+1天收盘价) / 前第N+1天收盘价 * 100
        例：近3日涨幅 → 用前第4天收盘价为基准

    :param df: 待排序的股票DataFrame（必须包含ts_code字段）
    :param trade_date: 交易日（兼容YYYY-MM-DD/YYYYMMDD格式）
    :param day_count: 【可选】近N日涨幅的N，默认20
    :return: 按近N日涨幅降序排序后的DataFrame
    """
    # 入参防御
    if df.empty or "ts_code" not in df.columns or day_count <= 0:
        return df

    # 1. 统一今日日期格式
    try:
        if len(trade_date) == 8 and trade_date.isdigit():
            today_dt = datetime.strptime(trade_date, "%Y%m%d")
            today_str = today_dt.strftime("%Y-%m-%d")
        else:
            today_dt = datetime.strptime(trade_date, "%Y-%m-%d")
            today_str = trade_date
    except ValueError:
        return df

    # 2. 取【前第N+1天】的交易日（对齐行情软件计算标准）
    try:
        # 往前推60天，确保能拿到足够的交易日（覆盖节假日）
        start_dt = today_dt - timedelta(days=60)
        trade_days = get_trade_dates(
            start_date=start_dt.strftime("%Y-%m-%d"),
            end_date=today_str
        )
        # 验证交易日数量是否足够
        required_days = day_count + 1
        if len(trade_days) < required_days:
            logger.warning(f"近{day_count}日涨幅排序：可回溯交易日不足{required_days}个，返回原DataFrame")
            return df
        # 取前第N+1天的交易日
        day_ago_str = trade_days[-required_days:][0]
    except Exception:
        return df

    # ========================
    # 核心：仅拉2天数据，仅查询目标股票
    # ========================
    target_codes = df["ts_code"].unique().tolist()
    logger.info(
        f"【近{day_count}日涨幅】仅取2天数据：前第{required_days}天[{day_ago_str}] + 今日[{today_str}]，目标股票{len(target_codes)}只")

    # 第1次拉：前第N+1天的收盘价（仅查询目标股票）
    df_ago = get_daily_kline_data(day_ago_str, ts_code_list=target_codes)
    df_ago = df_ago[["ts_code", "close"]].rename(columns={"close": f"ago_{required_days}d_close"})

    # 第2次拉：今日的收盘价（仅查询目标股票）
    df_today = get_daily_kline_data(today_str, ts_code_list=target_codes)
    df_today = df_today[["ts_code", "close"]].rename(columns={"close": "today_close"})

    # 合并计算（无冗余筛选，因为get_daily_kline_data已过滤）
    df = df.merge(df_today, on="ts_code", how="left").merge(df_ago, on="ts_code", how="left")
    df = df.dropna(subset=["today_close", f"ago_{required_days}d_close"])

    if df.empty:
        return df

    # 计算涨幅并排序（动态列名）
    gain_col = f"recent_{day_count}d_gain"
    df[gain_col] = (df["today_close"] / df[f"ago_{required_days}d_close"] - 1) * 100
    df = df.sort_values(gain_col, ascending=False).reset_index(drop=True)

    logger.info(f"【近{day_count}日涨幅】排序完成：有效股票{len(df)}只，最高涨幅{df[gain_col].iloc[0]:.2f}%")
    return df


def calc_limit_up_price(ts_code: str, pre_close: float) -> float:
    """
    计算股票涨停价（适配不同板块涨跌幅限制，融合调试日志+强类型+完整校验）
    :param ts_code: 股票代码（如600000.SH/300001.SZ/831010.BJ）
    :param pre_close: 前一日收盘价
    :return: 涨停价格（保留2位小数，无效值返回0.0）
    """
    if not pre_close or pre_close <= 0:
        logger.debug(f"[{ts_code}] 前收盘价无效（pre_close={pre_close}），涨停价返回0.0")
        return 0.0
    # 1. 判断板块类型，匹配对应涨跌幅
    if ts_code.endswith(".BJ"):  # 北交所
        limit_rate = BJ_BOARD_LIMIT_UP_RATE
    elif ts_code.startswith(("300", "301", "302")) or (ts_code.startswith("3") and ts_code.endswith(".SZ")):  # 创业板
        limit_rate = STAR_BOARD_LIMIT_UP_RATE
    elif ts_code.startswith("688"):  # 科创板
        limit_rate = STAR_BOARD_LIMIT_UP_RATE  # 科创板和创业板涨跌幅一致（20%）
    else:  # 主板（60/00开头）
        limit_rate = MAIN_BOARD_LIMIT_UP_RATE
    limit_up_price = pre_close * (1 + limit_rate)
    limit_up_price = round(limit_up_price, 2)
    logger.debug(f"[{ts_code}] 前收盘价={pre_close}，涨停幅度={limit_rate}，涨停价={limit_up_price}")
    return round(limit_up_price, 2)


def calc_limit_down_price(ts_code: str, pre_close: float) -> float:
    """
    计算股票跌停价（和涨停价逻辑完全对齐，适配不同板块涨跌幅限制）
    :param ts_code: 股票代码
    :param pre_close: 前一日收盘价
    :return: 跌停价格（保留2位小数，无效值返回0）
    """
    if not pre_close or pre_close <= 0:
        logger.debug(f"[{ts_code}] 前收盘价无效（pre_close={pre_close}），涨停价返回0.0")
        return 0.0
        # 1. 判断板块类型，匹配对应涨跌幅
    if ts_code.endswith(".BJ"):  # 北交所
        limit_rate = BJ_BOARD_LIMIT_UP_RATE
    elif ts_code.startswith(("300", "301", "302")) or (ts_code.startswith("3") and ts_code.endswith(".SZ")):  # 创业板
        limit_rate = STAR_BOARD_LIMIT_UP_RATE
    elif ts_code.startswith("688"):  # 科创板
        limit_rate = STAR_BOARD_LIMIT_UP_RATE  # 科创板和创业板涨跌幅一致（20%）
    else:  # 主板（60/00开头）
        limit_rate = MAIN_BOARD_LIMIT_UP_RATE
    # 跌停价公式：前收盘价 × (1 - 涨跌幅系数)，四舍五入保留2位小数
    limit_down_price = pre_close * (1 - limit_rate)
    logger.debug(f"[{ts_code}] 前收盘价={pre_close}，跌停幅度={limit_rate}，跌停价={limit_down_price}")
    return  round(limit_down_price, 2)


def filter_st_stocks(ts_code_list: List[str], trade_date: str) -> List[str]:
    """
    【唯一ST过滤方法】批量过滤指定交易日的ST/*ST股票（先动态入库当日ST数据，再1次查询比对）
    :param ts_code_list: 待过滤的股票代码列表（如['000001.SZ', '600000.SH']）
    :param trade_date: 交易日（兼容YYYYMMDD/YYYY-MM-DD格式）
    :return: 过滤后的正常股票代码列表（已剔除所有ST股）
    """
    # 3. 1次数据库查询：获取当日最新ST股票代码（入库后查询，确保数据最新）
    try:
        sql = """
              SELECT DISTINCT ts_code
              FROM stock_risk_warning
              WHERE trade_date = %s \
              """
        st_result = db.query(sql, trade_date)
        st_code_set = set([row["ts_code"] for row in st_result]) if st_result else set()
        # 4. 批量比对：保留非ST股票
        normal_codes = [ts_code for ts_code in ts_code_list if ts_code not in st_code_set]
        # 5. 关键日志：明确过滤效果
        filter_count = len(ts_code_list) - len(normal_codes)
        logger.info(
            f"[filter_st_stocks] 交易日{trade_date} ST过滤完成 | "
            f"原始股票数：{len(ts_code_list)} | 剔除ST股数：{filter_count} | 剩余正常股票数：{len(normal_codes)}"
        )
        # 调试日志：打印被剔除的ST股（可选）
        if filter_count > 0:
            st_removed = [ts_code for ts_code in ts_code_list if ts_code in st_code_set]
            logger.debug(f"[filter_st_stocks] 当日被剔除的ST股：{st_removed}")
        return normal_codes
    except Exception as e:
        logger.error(f"[filter_st_stocks] 批量过滤ST股票失败 | 交易日：{trade_date} | 错误：{e}", exc_info=True)
        return []

def get_trade_dates(start_date: str, end_date: str) -> List[str]:
    """
    通用交易日历查询方法：从已入库的trade_cal表中查询指定时间段的有效交易日
    :param start_date: 开始日期，格式yyyy-mm-dd
    :param end_date: 结束日期，格式yyyy-mm-dd
    :return: 按时间升序排列的交易日字符串列表，如["2025-01-01", "2025-01-02"]
    :raise RuntimeError: 查询失败/无有效交易日时抛出异常
    """
    # 1. 基础参数校验
    if not (isinstance(start_date, str) and isinstance(end_date, str)):
        logger.error(f"交易日查询失败：日期格式错误，start_date={start_date}, end_date={end_date}")
        raise RuntimeError("交易日查询失败：日期必须为字符串格式（yyyy-mm-dd）")

    # 2. 执行SQL查询
    sql = """
          SELECT cal_date
          FROM trade_cal
          WHERE cal_date BETWEEN %s AND %s 
            AND is_open = 1
          ORDER BY cal_date ASC
          """
    try:
        df = db.query(sql, params=(start_date, end_date), return_df=True)
    except Exception as e:
        logger.error(f"交易日查询数据库异常：{str(e)}")
        raise RuntimeError(f"交易日查询失败：{str(e)}")

    # 3. 空数据校验
    if df.empty:
        logger.error(f"[{start_date} 至 {end_date}] 时间段内无有效交易日")
        raise RuntimeError(f"[{start_date} 至 {end_date}] 时间段内无有效交易日")

    # 4. 转换为字符串列表返回
    trade_dates = df["cal_date"].astype(str).tolist()
    logger.debug(f"交易日查询成功：[{start_date} 至 {end_date}] 共{len(trade_dates)}个有效交易日")
    return trade_dates


def get_prev_trade_date(ref_date: str = None) -> str:
    """
    返回 ref_date（含）之前最近一个交易日，不含 ref_date 当天。
    主要用于凌晨定时任务中获取"昨天的交易日"。

    :param ref_date: 参考日期，格式 YYYY-MM-DD，默认为今日
    :return: 最近一个已完成的交易日，格式 YYYY-MM-DD
    """
    from datetime import timedelta
    today = ref_date or datetime.now().strftime("%Y-%m-%d")
    start = (datetime.strptime(today, "%Y-%m-%d") - timedelta(days=15)).strftime("%Y-%m-%d")
    try:
        dates = get_trade_dates(start, today)
        # 去掉 ref_date 当天（凌晨运行时当天市场还未开盘）
        dates = [d for d in dates if d < today]
        return dates[-1] if dates else today
    except Exception as e:
        logger.warning(f"[get_prev_trade_date] 查询失败，回退到昨日：{e}")
        from datetime import timedelta
        return (datetime.strptime(today, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")


def get_st_stock_codes(trade_date: str) -> List[str]:
    """
    获取指定交易日所有 ST / *ST 股票代码列表。
    与 filter_st_stocks 区别：本函数直接返回 ST 代码集合，无需传入候选池。

    :param trade_date: 交易日，格式 YYYY-MM-DD
    :return: ST 股票 ts_code 列表
    """
    sql = "SELECT DISTINCT ts_code FROM stock_risk_warning WHERE trade_date = %s"
    try:
        df = db.query(sql, params=(trade_date,), return_df=True)
        return df["ts_code"].tolist() if not df.empty else []
    except Exception as e:
        logger.error(f"[get_st_stock_codes] 查询失败 | 交易日：{trade_date} | 错误：{e}")
        return []


def get_daily_kline_data(trade_date: str, ts_code_list: List[str] = None) -> pd.DataFrame:
    """
    获取指定日期的日线数据（向后兼容优化版）
    【新增功能】支持仅查询指定股票的日线数据，大幅提升性能
    【向后兼容】不传ts_code_list时，保持原有全市场查询逻辑，不影响现有调用

    :param trade_date: 交易日（兼容YYYY-MM-DD/YYYYMMDD格式）
    :param ts_code_list: 【可选】指定股票代码列表，仅查询这些股票的日线数据
    :return: 日线数据DataFrame
    """
    # 1. 日期格式化（兼容两种格式）
    logger.debug(
        f"开始获取日线数据: {trade_date}" + (f"，指定股票数：{len(ts_code_list)}" if ts_code_list else "，全市场"))
    trade_date_format = trade_date.replace("-", "")

    # 2. 构建SQL和参数（根据是否指定股票动态调整）
    if ts_code_list:
        # 【新增】仅查询指定股票
        if not isinstance(ts_code_list, (list, tuple, set)):
            logger.error(f"ts_code_list格式错误，必须是列表/元组/集合，当前类型：{type(ts_code_list)}")
            return pd.DataFrame()
        if not ts_code_list:
            logger.warning("ts_code_list为空，返回空DataFrame")
            return pd.DataFrame()

        # 构建带IN条件的SQL
        sql = """
              SELECT *
              FROM kline_day
              WHERE trade_date = %s 
                AND ts_code IN %s
              """
        params = (trade_date_format, tuple(ts_code_list))
    else:
        # 【原有逻辑】查询全市场（保持向后兼容）
        sql = """
              SELECT *
              FROM kline_day
              WHERE trade_date = %s 
              """
        params = (trade_date_format,)

    # 3. 执行查询（保持原有逻辑不变）
    try:
        df = db.query(sql, params=params, return_df=True)
    except Exception as e:
        logger.error(f"{trade_date} 日线数据查询失败：{str(e)}")
        return pd.DataFrame()
    # 4. 返回结果（保持原有逻辑不变）
    if df is not None and not df.empty:
        logger.debug(f"{trade_date} 日线数据从数据库读取完成，行数：{len(df)}")
        return df
    else:
        logger.error(f"{trade_date} 日线数据拉取失败，跳过当日")
        return pd.DataFrame()


def get_qfq_kline_data(trade_date: str, ts_code_list: List[str] = None) -> pd.DataFrame:
    """
    获取指定日期的前复权日线数据（MA 计算专用）

    :param trade_date:   交易日，兼容 YYYY-MM-DD / YYYYMMDD 格式
    :param ts_code_list: 【可选】仅查询指定股票；为空时查全市场
    :return: 前复权日线 DataFrame（无数据返回空 DF）
    """
    trade_date_fmt = trade_date.replace("-", "")
    if ts_code_list:
        sql    = "SELECT ts_code, trade_date, open, high, low, close, volume, amount FROM kline_day_qfq WHERE trade_date = %s AND ts_code IN %s"
        params = (trade_date_fmt, tuple(ts_code_list))
    else:
        sql    = "SELECT ts_code, trade_date, open, high, low, close, volume, amount FROM kline_day_qfq WHERE trade_date = %s"
        params = (trade_date_fmt,)
    try:
        df = db.query(sql, params=params, return_df=True)
        if df is not None and not df.empty:
            df["trade_date"] = df["trade_date"].astype(str)
            return df
    except Exception as e:
        logger.warning(f"[qfq] {trade_date} 前复权日线查询失败：{e}")
    return pd.DataFrame()


def getStockRank_fortraining(trade_date: str) -> Optional[pd.DataFrame]:
    """
    数据库读取指定日期trade_date
    的全市场历史K线数据（仅用于机器学习训练），筛选符合涨幅阈值的股票并返回


    筛选规则：
    1. 忽略北交所股票（代码含.BJ）；
    2. 创业板（3*.SZ）、科创板（688*.SH）：当日涨跌幅>13%（20%涨停×65%）；
    3. 主板股票（非上述板块）：当日涨跌幅>6.5%（10%涨停×65%）；

    :param trade_date: 待查询日期，格式必须为yyyy-mm-dd（如2025-04-28）
    :return: DataFrame（列：ts_code、pct_chg），按pct_chg正序排列；查询失败/无数据返回None
    """
    # ==================== 2. 构建筛选SQL（精准区分板块涨幅阈值） ====================
    # logger.info('*'*60)
    # logger.info('！！！！！！历史数据，非实时数据，仅供模拟训练！！！！！！！')
    # logger.info('*'*60)
    sql = f"""
        SELECT ts_code, pct_chg
        FROM kline_day
        WHERE trade_date = '{trade_date}'  -- 直接拼接日期字符串
        AND ts_code NOT LIKE '%%.BJ'
        AND pct_chg > CASE
            WHEN ts_code LIKE '3%%.SZ' THEN 13.0
            WHEN ts_code LIKE '688%%.SH' THEN 13.0
            ELSE 6.5
        END;
    """

    # 调用时不传params参数
    result_df = db.query(sql=sql, return_df=True)

    # ==================== 4. 结果处理 ====================
    if result_df is None:
        logger.error(f"查询{trade_date}符合条件的股票失败（数据库异常）")
        return None
    if result_df.empty:
        logger.warning(f"{trade_date}无符合涨幅阈值的股票数据,检查该日是否交易")
        return pd.DataFrame()
    # 确保列名正确（数据库返回的字段名可能大小写/别名问题，强制对齐）
    result_df = result_df.rename(columns=str.lower).loc[:, ['ts_code', 'pct_chg']]
    # 再次确认按pct_chg正序排列（防止SQL排序失效）
    result_df = result_df.sort_values(by='pct_chg', ascending=True).reset_index(drop=True)
    logger.debug(f"{trade_date}共查询到{len(result_df)}  只符合条件的股票")

    return result_df


def getTagRank_daily(
        ts_code_list: List[str],
        exclude_concepts: Optional[List[str]] = None
) -> Optional[pd.DataFrame]:
    """
    接受股票ts_code列表，本地拆分逗号分隔的题材字段，统计题材覆盖情况
    支持传入黑名单数组，过滤掉不具备分析性的题材

    :param ts_code_list: 待分析的股票代码列表（带.SZ/.SH后缀）
    :param exclude_concepts: 需要忽略的题材黑名单数组（可选，不传则使用默认黑名单）
    :return: DataFrame（列：concept_name、cover_stock_count、cover_rate），
             按cover_stock_count降序排列；查询失败/无数据返回None
    """
    # ==================== 0. 初始化默认黑名单 ====================
    # 默认过滤掉常见的无分析性题材，用户可通过参数覆盖
    if exclude_concepts is None:
        exclude_concepts = default_exclude
    else:
        # 如果用户传了黑名单，合并默认黑名单（避免遗漏），也可以直接覆盖
        # exclude_concepts = list(set(exclude_concepts + default_exclude)) # 合并模式
        pass  # 覆盖模式：直接使用用户传入的黑名单

    # ==================== 1. 输入校验 ====================
    if not ts_code_list:
        logger.warning("输入的ts_code_list为空，无法进行题材统计")
        return None

    # 去重并统计输入股票数量
    ts_code_list = list(set(ts_code_list))
    input_stock_count = len(ts_code_list)
    logger.debug(f"开始统计{input_stock_count}只股票的题材覆盖情况，黑名单题材数：{len(exclude_concepts)}")

    # ==================== 2. 极简SQL查询原始数据 ====================
    ts_code_str = "','".join(ts_code_list)
    ts_code_str = f"'{ts_code_str}'"

    sql = f"""
        SELECT ts_code, concept_tags 
        FROM stock_basic 
        WHERE ts_code IN ({ts_code_str})
        AND concept_tags IS NOT NULL
        AND TRIM(concept_tags) != '';
    """

    try:
        raw_df = db.query(sql=sql, return_df=True)
    except Exception as e:
        logger.error(f"题材原始数据查询失败：{str(e)}", exc_info=True)
        return None

    if raw_df is None:
        logger.error("题材原始数据查询返回None（数据库异常）")
        return None
    if raw_df.empty:
        logger.warning("未查询到符合条件的股票题材数据")
        return raw_df

    # ==================== 3. 本地Pandas向量化处理（含黑名单过滤） ====================
    raw_df = raw_df.rename(columns=str.lower)

    # 1. 统一分隔符
    raw_df["concept_tags"] = raw_df["concept_tags"].str.replace("；|，|;", ",", regex=True)
    # 2. 拆分+展开
    exploded_df = raw_df.assign(
        concept_name=raw_df["concept_tags"].str.split(",")
    ).explode("concept_name", ignore_index=True)
    # 3. 去空格、过滤空值
    exploded_df["concept_name"] = exploded_df["concept_name"].str.strip()
    exploded_df = exploded_df[exploded_df["concept_name"] != ""].reset_index(drop=True)

    # ==================== 【新增】4. 黑名单过滤 ====================
    before_filter_count = exploded_df["concept_name"].nunique()
    # 核心过滤逻辑：使用 isin() 匹配黑名单，然后取反 ~
    exploded_df = exploded_df[~exploded_df["concept_name"].isin(exclude_concepts)].reset_index(drop=True)
    after_filter_count = exploded_df["concept_name"].nunique()

    logger.debug(
        f"黑名单过滤完成：过滤前题材数 {before_filter_count}，过滤后 {after_filter_count}，共过滤 {before_filter_count - after_filter_count} 个题材")

    # ==================== 5. 分组统计 ====================
    if exploded_df.empty:
        logger.warning("经过黑名单过滤后，无剩余题材数据")
        return pd.DataFrame(columns=["concept_name", "cover_stock_count", "cover_rate"])

    result_df = exploded_df.groupby("concept_name", as_index=False).agg(
        cover_stock_count=("ts_code", "nunique"),
    )
    result_df["cover_rate"] = round(
        result_df["cover_stock_count"] / input_stock_count * 100,
        2
    )
    result_df = result_df.sort_values(
        by=["cover_stock_count", "cover_rate"],
        ascending=[False, False]
    ).head(5).reset_index(drop=True)

    # ==================== 6. 结果输出 ====================
    logger.debug(f"题材统计完成，前5名题材：\n{result_df.to_string(index=False)}")
    return result_df



def init_update_record():
    """初始化更新记录文件"""
    init_data = {
        "last_update_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "last_update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "updated_tables": []
    }
    with open(UPDATE_RECORD_FILE, "w", encoding="utf-8") as f:
        json.dump(init_data, f, ensure_ascii=False, indent=4)
    return init_data


def read_last_update_record():
    """读取上次更新记录"""
    if not os.path.exists(UPDATE_RECORD_FILE):
        return init_update_record()
    try:
        with open(UPDATE_RECORD_FILE, "r", encoding="utf-8") as f:
            record = json.load(f)
        record["last_update_date"] = record.get("last_update_date", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        return record
    except Exception as e:
        logger.error(f"读取更新记录失败，初始化新记录：{e}")
        return init_update_record()


def write_update_record(update_date: str, updated_tables: list):
    """写入本次更新记录"""
    record_data = {
        "last_update_date": update_date,
        "last_update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "updated_tables": updated_tables
    }
    try:
        with open(UPDATE_RECORD_FILE, "w", encoding="utf-8") as f:
            json.dump(record_data, f, ensure_ascii=False, indent=4)
        logger.info(f"更新记录已保存：本次更新至 {update_date}")
    except Exception as e:
        logger.error(f"写入更新记录失败：{e}")


def calc_incremental_date_range(last_update_date: str, end_date: str = None) -> list:
    """计算增量更新日期列表（YYYYMMDD）"""
    last_dt = datetime.strptime(last_update_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.now()

    if last_dt >= end_dt:
        logger.info("无增量日期，无需更新")
        return []

    date_list = []
    current_dt = last_dt + timedelta(days=1)
    while current_dt <= end_dt:
        date_list.append(current_dt.strftime("%Y%m%d"))
        current_dt += timedelta(days=1)

    logger.info(f"增量更新日期：共{len(date_list)}天 → {date_list}")
    return date_list


def calc_15_years_date_range() -> Tuple[str, str]:
    """
    获取倒推15年内的K线数据
    返回：(start_date, end_date) 均为YYYYMMDD格式字符串
    """
    # 结束日期：当前日期（YYYYMMDD）
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=15*365)).strftime("%Y%m%d")
    logger.info(f"日K数据获取时间范围：{start_date} 至 {end_date}")
    return start_date, end_date

def escape_mysql_reserved_words(field_name: str) -> str:
    """
    转义MySQL保留字字段（给保留字加反引号）
    用于避免字段名与MySQL关键字冲突
    """
    # 常见MySQL保留字（金融数据场景高频）
    reserved_words = {"change", "desc", "order", "select", "insert", "update", "volume"}
    if field_name.lower() in reserved_words:
        return f"`{field_name}`"
    return field_name


def auto_add_missing_table_columns(
    table_name: str,
    missing_columns: List[str],
    col_type_mapping: Dict[str, str] = None
) -> bool:
    """
    通用方法：自动为数据库表新增缺失字段（带默认值，避免插入NULL/NaN报错）
    迁移说明：从DataCleaner类迁移为通用函数，核心逻辑完全不变
    """
    # 默认字段类型映射（金融数据标准化规则）
    default_col_type_mapping = {
        # 日期类字段
        "list_date": "DATE NOT NULL DEFAULT '1970-01-01'",
        "delist_date": "DATE DEFAULT NULL",
        # 数值类字段
        "total_share": "BIGINT DEFAULT 0",
        "float_share": "BIGINT DEFAULT 0",
        "free_share": "BIGINT DEFAULT 0",
        "total_mv": "DECIMAL(20,2) DEFAULT 0.00",
        "circ_mv": "DECIMAL(20,2) DEFAULT 0.00",
        # 核心字符串字段
        "exchange": "VARCHAR(8) NOT NULL DEFAULT 'UNKNOWN'",
        "ts_code": "VARCHAR(9) NOT NULL DEFAULT 'UNKNOWN'",
        "symbol": "VARCHAR(6) NOT NULL DEFAULT 'UNKNOWN'",
        "name": "VARCHAR(32) NOT NULL DEFAULT 'UNKNOWN'",
        # 兜底类型
        "default": "VARCHAR(255) NOT NULL DEFAULT ''"
    }

    # 合并默认映射和自定义映射（自定义优先级更高）
    final_col_map = default_col_type_mapping.copy()
    if col_type_mapping:
        final_col_map.update(col_type_mapping)

    success = True
    for col in missing_columns:
        try:
            col_type = final_col_map.get(col, final_col_map["default"])
            db.add_table_column(table_name, col, col_type)
            logger.info(f"表{table_name}新增字段{col}成功（类型：{col_type}）")
        except Exception as e:
            success = False
            logger.error(f"表{table_name}新增字段{col}失败：{e}")

    return success

# -------------------------- 重试装饰器实现 --------------------------
def retry_decorator(max_retries:  int = 3, retry_interval: float = 1.0):
    """
    ：支持异常重试 + 空DataFrame重试
    :param max_retries: 最大重试次数
    :param retry_interval: 每次重试的间隔（秒）
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            retry_count = 0
            while retry_count < max_retries:
                try:
                    # 执行原方法
                    result = func(self, *args, **kwargs)

                    # 检查返回值是否为空DataFrame
                    if isinstance(result, pd.DataFrame) and result.empty:
                        retry_count += 1
                        # 达到最大重试次数，记录警告并返回空DataFrame
                        if retry_count >= max_retries:
                            logger.warning(
                                f"【{func.__name__}】空数据重试次数已达上限（{max_retries}次），最终返回空数据 "
                                f"参数：args={args}, kwargs={kwargs}"
                            )
                            return result
                        # 未达最大次数，记录警告并等待后重试
                        logger.warning(
                            f"【{func.__name__}】返回空DataFrame，将进行第{retry_count}次重试（剩余{max_retries - retry_count}次） "
                            f"间隔{retry_interval}秒"
                        )
                        time.sleep(retry_interval)
                        continue  # 触发下一次重试
                    # 数据非空，直接返回
                    return result

                except Exception as e:
                    retry_count += 1
                    # 达到最大重试次数，记录错误并返回空DataFrame
                    if retry_count >= max_retries:
                        logger.error(
                            f"【{func.__name__}】异常重试次数已达上限（{max_retries}次），最终执行失败 "
                            f"参数：args={args}, kwargs={kwargs} 错误：{str(e)}"
                        )
                        return pd.DataFrame()
                    # 未达最大次数，降级为 debug 日志后等待重试
                    logger.debug(
                        f"【{func.__name__}】执行异常，将进行第{retry_count}次重试（剩余{max_retries - retry_count}次） "
                        f"间隔{retry_interval}秒 错误：{str(e)}"
                    )
                    time.sleep(retry_interval)
            # 兜底返回空DataFrame
            return pd.DataFrame()
        return wrapper
    return decorator



def get_limit_list_ths(trade_date: str, limit_type: str = None) -> pd.DataFrame:
    """
    查询指定日期的涨跌停池数据
    :param trade_date: 交易日，格式 yyyy-mm-dd
    :param limit_type: 板单类别（涨停池/跌停池/炸板池等），不传返回全部
    :return: DataFrame
    """
    sql = "SELECT * FROM limit_list_ths WHERE trade_date = %s"
    params = [trade_date]
    if limit_type:
        sql += " AND limit_type = %s"
        params.append(limit_type)

    try:
        df = db.query(sql, params=tuple(params), return_df=True)
        logger.debug(f"[涨跌停池] {trade_date} type={limit_type} 行数:{len(df)}")
        return df if df is not None else pd.DataFrame()
    except Exception as e:
        logger.error(f"[涨跌停池] 查询失败: {e}")
        return pd.DataFrame()


def get_limit_step(trade_date: str) -> pd.DataFrame:
    """
    查询指定日期的连板天梯数据
    :param trade_date: 交易日，格式 yyyy-mm-dd
    :return: DataFrame（含 ts_code, name, nums）
    """
    sql = "SELECT * FROM limit_step WHERE trade_date = %s ORDER BY nums DESC"
    try:
        df = db.query(sql, params=(trade_date,), return_df=True)
        logger.debug(f"[连板天梯] {trade_date} 行数:{len(df)}")
        return df if df is not None else pd.DataFrame()
    except Exception as e:
        logger.error(f"[连板天梯] 查询失败: {e}")
        return pd.DataFrame()


def get_limit_cpt_list(trade_date: str) -> pd.DataFrame:
    """
    查询指定日期的最强板块数据
    :param trade_date: 交易日，格式 yyyy-mm-dd
    :return: DataFrame（含 ts_code, name, days, up_stat, cons_nums, up_nums, pct_chg, rank）
    """
    sql = "SELECT * FROM limit_cpt_list WHERE trade_date = %s"
    try:
        df = db.query(sql, params=(trade_date,), return_df=True)
        logger.debug(f"[最强板块] {trade_date} 行数:{len(df)}")
        return df if df is not None else pd.DataFrame()
    except Exception as e:
        logger.error(f"[最强板块] 查询失败: {e}")
        return pd.DataFrame()


def get_index_daily(trade_date: str, ts_code_list: List[str] = None) -> pd.DataFrame:
    """
    查询指定日期的指数日线数据
    :param trade_date: 交易日，格式 yyyy-mm-dd 或 yyyymmdd
    :param ts_code_list: 指定指数代码列表，不传返回全部
    :return: DataFrame
    """
    trade_date_fmt = trade_date.replace("-", "")
    if ts_code_list:
        sql = "SELECT * FROM index_daily WHERE trade_date = %s AND ts_code IN %s"
        params = (trade_date_fmt, tuple(ts_code_list))
    else:
        sql = "SELECT * FROM index_daily WHERE trade_date = %s"
        params = (trade_date_fmt,)

    try:
        df = db.query(sql, params=params, return_df=True)
        logger.debug(f"[指数日线] {trade_date} 行数:{len(df)}")
        return df if df is not None else pd.DataFrame()
    except Exception as e:
        logger.error(f"[指数日线] 查询失败: {e}")
        return pd.DataFrame()


def get_market_total_volume(dates: List[str]) -> pd.DataFrame:
    """
    批量查询多个交易日的全市场成交量（DB 侧 GROUP BY 聚合，比代码侧汇总快得多）
    :param dates: 交易日列表，格式 yyyy-mm-dd
    :return: DataFrame，列：trade_date(yyyymmdd), market_total_vol
    """
    if not dates:
        return pd.DataFrame(columns=["trade_date", "market_total_vol"])
    dates_fmt = [d.replace("-", "") for d in dates]
    placeholders = ", ".join(["%s"] * len(dates_fmt))
    sql = f"""
          SELECT trade_date, SUM(amount) AS market_total_vol
          FROM kline_day
          WHERE trade_date IN ({placeholders})
          GROUP BY trade_date
          """
    try:
        df = db.query(sql, params=tuple(dates_fmt), return_df=True)
        logger.debug(f"[全市场成交量] 查询完成 | 日期数:{len(dates)} | 返回行数:{len(df)}")
        return df if df is not None else pd.DataFrame(columns=["trade_date", "market_total_vol"])
    except Exception as e:
        logger.error(f"[全市场成交量] 查询失败: {e}")
        return pd.DataFrame(columns=["trade_date", "market_total_vol"])


def get_stocks_in_sector(sector_name: str) -> List[str]:
    """
    【通用工具】从stock_basic表查询指定板块/概念对应的所有股票代码
    精准匹配逗号分隔的concept_text字段，避免模糊匹配错误
    :param sector_name: 板块/概念完整名称（如"人工智能"、"算力"）
    :return: 该板块下的所有股票ts_code列表，查询失败/无结果返回空列表
    """
    sector_name_clean = str(sector_name).strip()
    # 核心SQL：用MySQL原生FIND_IN_SET精准匹配逗号分隔的概念
    # 加入LOWER兼容大小写不一致的场景，DISTINCT去重避免重复数据
    sql = """
        SELECT DISTINCT ts_code 
        FROM stock_basic 
        WHERE FIND_IN_SET(%s, concept_tags) > 0
          """
    # 执行参数化查询（防SQL注入，完全适配项目db工具的调用方式）
    result_df = db.query(sql, params=(sector_name_clean,))
    return result_df



def analyze_stock_follower_strength(
    ts_code: str,
    trade_date: str,
    *,
    daily_rank_top_pct: float = 0.20,
    amp_rank_top_pct: float = 0.30,
    minute_bucket: str = "5min",
) -> Dict[str, object]:
    """
    判断目标股票在指定交易日是否属于"跟风票"。

    两步判断：
    1. 日线层：peer 池 = 目标股所有概念（排除 default_exclude）下股票合并去重，按当日涨幅排序。
               目标股涨幅排不进前 daily_rank_top_pct（默认 20%）→ 直接返回跟风。
    2. 分钟线层（仅排进前 20% 的股票执行）：
       - 目标股：找 5 分钟线振幅（high/首根开盘 - 1）最大的第一根 K 线 → T_target, A_target
         （涨停场景：首次涨停时刻即为 T_target，因为振幅最大的第一根就是首次触及涨停的 K 线）
       - 对涨幅高于目标股的每只股票：
           * 找自身振幅最大的第一根 K 线 → T_i（用于计算时间中位数）
           * 取 T_target 时刻的 5 分钟振幅 → A_i（"板块在 T_target 时刻的强度"）
       - 不跟风需同时满足：
           * 时间：T_target < median(T_i)   ← 目标股先于大多数强势股到达峰值
           * 振幅：A_target 排在 {A_i} 前 amp_rank_top_pct（默认 30%）← 目标股振幅够强
       - 任一不满足 → 跟风

    分钟线获取失败（所有重试耗尽）时用中性值：不标记为跟风（保守保留股票）。

    返回字段：
      {
        "is_follower": bool,
        "trade_date": str,
        "target_pct_chg": float,
        "concept_tags": List[str],            # 过滤 default_exclude 后的有效概念
        "peer_pool_size": int,                # peer 池大小（当日有涨幅数据且 pct_chg > 0）
        "daily_rank_pct": float,              # 排名百分位（越小越强）
        "stronger_peer_count": int,           # 涨幅高于目标股的 peer 数量
        "target_peak_time": str,              # T_target（HH:MM）
        "target_peak_amp": float,             # A_target（%，相对首根开盘）
        "stronger_median_peak_time": str,     # median(T_i)（HH:MM）
        "target_amp_rank_pct": float,         # A_target 在 {A_i} 中的排名百分位
        "timing_ok": bool,                    # T_target < median(T_i)
        "amplitude_ok": bool,                 # A_target 在前 30%
        "judgement_summary": List[str],
      }
    """
    import time as _time
    from data.data_cleaner import data_cleaner

    _t0 = _time.time()
    trade_date_fmt = trade_date.replace("-", "")
    result: Dict[str, object] = {
        "is_follower": False,
        "trade_date": trade_date,
        "target_pct_chg": 0.0,
        "concept_tags": [],
        "peer_pool_size": 0,
        "daily_rank_pct": 1.0,
        "stronger_peer_count": 0,
        "target_peak_time": "",
        "target_peak_amp": 0.0,
        "stronger_median_peak_time": "",
        "target_amp_rank_pct": 1.0,
        "timing_ok": False,
        "amplitude_ok": False,
        "amplitude_top3": False,
        "leading_concepts": [],
        "judgement_summary": [],
    }

    if not ts_code or not trade_date_fmt:
        result["judgement_summary"].append("invalid_input")
        return result

    _exclude_set = set(default_exclude)

    def _normalize_tags(raw_tags: str) -> List[str]:
        if not raw_tags:
            return []
        tags = re.split(r"[,，；;]+", str(raw_tags))
        return [t.strip() for t in tags if t.strip() and t.strip() not in _exclude_set]

    # ── Step 1: 加载目标股概念（排除 default_exclude）──────────────────────
    rows = db.query("SELECT concept_tags FROM stock_basic WHERE ts_code = %s LIMIT 1", params=(ts_code,)) or []
    if not rows:
        result["judgement_summary"].append("no_stock_basic")
        return result
    valid_concepts = _normalize_tags(rows[0].get("concept_tags", ""))
    result["concept_tags"] = valid_concepts
    if not valid_concepts:
        result["judgement_summary"].append("no_valid_concepts")
        return result

    # ── Step 2: 拉取所有概念下的 peer 股票，合并去重 ────────────────────────
    peer_codes: set = set()
    for concept in valid_concepts:
        stocks = get_stocks_in_sector(concept) or []
        for s in stocks:
            code = s.get("ts_code") if isinstance(s, dict) else s
            code = str(code).strip()
            if code:
                peer_codes.add(code)
    peer_codes.add(ts_code)

    # ── Step 3: 拉取当日日线，按涨幅排序，确认目标股排名 ────────────────────
    daily_df = get_daily_kline_data(trade_date, ts_code_list=list(peer_codes))
    if daily_df.empty or "ts_code" not in daily_df.columns:
        result["judgement_summary"].append("missing_daily")
        result["is_follower"] = True
        return result

    daily_df = daily_df.copy()
    daily_df["ts_code"] = daily_df["ts_code"].astype(str)
    daily_df["pct_chg"] = pd.to_numeric(daily_df["pct_chg"], errors="coerce")
    daily_df = daily_df.dropna(subset=["pct_chg"])

    target_row = daily_df[daily_df["ts_code"] == ts_code]
    if target_row.empty:
        result["judgement_summary"].append("target_missing_daily")
        result["is_follower"] = True
        return result

    target_pct = float(target_row.iloc[0]["pct_chg"])
    result["target_pct_chg"] = target_pct

    active_df = daily_df[daily_df["pct_chg"] > 0].sort_values("pct_chg", ascending=False).reset_index(drop=True)
    result["peer_pool_size"] = int(len(active_df))

    if active_df.empty or ts_code not in set(active_df["ts_code"].tolist()):
        result["judgement_summary"].append("target_not_positive")
        result["is_follower"] = True
        return result

    target_rank_idx = int(active_df.index[active_df["ts_code"] == ts_code][0])  # 0-based
    daily_rank_pct = (target_rank_idx + 1) / len(active_df)
    result["daily_rank_pct"] = daily_rank_pct

    stronger_df = active_df.iloc[:target_rank_idx].copy()
    result["stronger_peer_count"] = int(len(stronger_df))

    # ── Step 4: 日线层判断 ────────────────────────────────────────────────
    if daily_rank_pct > daily_rank_top_pct:
        result["is_follower"] = True
        result["judgement_summary"].append(f"daily_rank_fail rank_pct={daily_rank_pct:.3f}")
        return result

    _t1 = _time.time()
    result["judgement_summary"].append(f"daily_rank_ok rank_pct={daily_rank_pct:.3f}")

    # 批量查询强势股名称+概念（1次SQL）
    stronger_codes_all = stronger_df["ts_code"].astype(str).tolist()
    _name_rows = db.query(
        f"SELECT ts_code, name, concept_tags FROM stock_basic WHERE ts_code IN ({','.join(['%s']*len(stronger_codes_all))})",
        params=tuple(stronger_codes_all),
    ) or []
    _name_map = {r["ts_code"]: r["name"] for r in _name_rows}
    _concept_map = {r["ts_code"]: _normalize_tags(r.get("concept_tags", "")) for r in _name_rows}
    # 加入目标股自身
    _concept_map[ts_code] = valid_concepts

    logger.debug(
        f"[follower:{ts_code}] 日线层通过 rank={daily_rank_pct:.3f} "
        f"peer池={result['peer_pool_size']} 更强股={len(stronger_codes_all)} 耗时={_t1-_t0:.1f}s"
    )
    for _sc in stronger_codes_all:
        _sc_row = stronger_df[stronger_df["ts_code"] == _sc]
        _sc_pct = float(_sc_row.iloc[0]["pct_chg"]) if not _sc_row.empty else 0.0
        logger.debug(f"  [stronger] {_sc} {_name_map.get(_sc, '?')} pct_chg={_sc_pct:.2f}%")

    if stronger_df.empty:
        # 排第一，没有更强的 peer
        result["is_follower"] = False
        result["judgement_summary"].append("rank1_no_stronger_peers")
        return result

    # ── Step 5: 分钟线层 ─────────────────────────────────────────────────
    def _get_first_open(min_df: pd.DataFrame) -> float:
        if min_df is None or min_df.empty:
            return 0.0
        df = min_df.copy()
        df["trade_time"] = pd.to_datetime(df["trade_time"])
        df = df.sort_values("trade_time")
        opens = pd.to_numeric(df["open"], errors="coerce").dropna()
        return float(opens.iloc[0]) if not opens.empty else 0.0

    def _find_peak_bar(min_df: pd.DataFrame, first_open: float) -> tuple:
        """
        找振幅（high/first_open - 1）最大的第一根 K 线。
        涨停场景：最大振幅会在首次涨停时出现，之后维持（high 不再更高），
        取第一根（即首次触及最高振幅的时刻），天然等同于"涨停时间"。
        返回 (time_str "HH:MM", amplitude_pct)。
        """
        if min_df is None or min_df.empty or first_open <= 0:
            return None, 0.0
        df = min_df.copy()
        df["trade_time"] = pd.to_datetime(df["trade_time"])
        df = df.sort_values("trade_time").reset_index(drop=True)
        df["high"] = pd.to_numeric(df["high"], errors="coerce")
        df = df.dropna(subset=["high"])
        if df.empty:
            return None, 0.0
        df["amp"] = (df["high"] / first_open - 1) * 100
        max_amp = float(df["amp"].max())
        # 取第一根达到 max_amp 的 K 线（0.05% 容差应对浮点误差）
        peak_rows = df[df["amp"] >= max_amp - 0.05]
        first_peak = peak_rows.iloc[0]
        bucket = pd.Timestamp(first_peak["trade_time"]).floor(minute_bucket)
        return bucket.strftime("%H:%M"), max_amp

    def _get_amp_at_time(min_df: pd.DataFrame, target_time_str: str, first_open: float) -> float:
        """
        取 target_time_str 所在 5 分钟 bucket 的振幅（high/first_open - 1）%。
        若该 bucket 无数据，取最近更早的 bucket。
        """
        if min_df is None or min_df.empty or first_open <= 0 or not target_time_str:
            return 0.0
        df = min_df.copy()
        df["trade_time"] = pd.to_datetime(df["trade_time"])
        df["bucket"] = df["trade_time"].dt.floor(minute_bucket)
        df["high"] = pd.to_numeric(df["high"], errors="coerce")
        df = df.dropna(subset=["high"])
        if df.empty:
            return 0.0
        date_part = trade_date if "-" in trade_date else f"{trade_date_fmt[:4]}-{trade_date_fmt[4:6]}-{trade_date_fmt[6:]}"
        target_bucket = pd.Timestamp(f"{date_part} {target_time_str}").floor(minute_bucket)
        matched = df[df["bucket"] == target_bucket]
        if matched.empty:
            earlier = df[df["bucket"] <= target_bucket]
            if earlier.empty:
                return 0.0
            matched = earlier.tail(1)
        amp = float((matched["high"].iloc[0] / first_open - 1) * 100)
        return max(amp, 0.0)

    def _time_to_minutes(t_str: Optional[str]) -> int:
        if not t_str:
            return 10 ** 9
        hh, mm = t_str.split(":")
        return int(hh) * 60 + int(mm)

    # 拉目标股分钟线
    target_min_df = data_cleaner.get_kline_min_by_stock_date(ts_code, trade_date_fmt)
    target_first_open = _get_first_open(target_min_df)
    if target_first_open <= 0:
        result["judgement_summary"].append("target_minute_missing_neutral")
        result["is_follower"] = False
        return result

    t_target, a_target = _find_peak_bar(target_min_df, target_first_open)
    if t_target is None:
        result["judgement_summary"].append("target_peak_not_found_neutral")
        result["is_follower"] = False
        return result

    result["target_peak_time"] = t_target
    result["target_peak_amp"] = round(a_target, 3)

    _t2 = _time.time()
    logger.debug(
        f"[follower:{ts_code}] 目标股分钟线完成 T_target={t_target} A_target={a_target:.2f}% 耗时={_t2-_t1:.1f}s"
    )

    # 对每只 stronger stock：找 T_i（自身峰值时间）和 A_i（T_target 时刻的振幅）
    stronger_codes = stronger_df["ts_code"].astype(str).tolist()
    peak_times_minutes: List[int] = []
    amps_at_t: List[float] = []
    # 保留完整明细（含股票代码），用于排名日志
    stronger_details: List[Dict] = []

    for sc in stronger_codes:
        _t_sc = _time.time()
        sc_min_df = data_cleaner.get_kline_min_by_stock_date(sc, trade_date_fmt)
        sc_first_open = _get_first_open(sc_min_df)
        if sc_first_open <= 0:
            logger.debug(f"  [stronger_min] {sc} {_name_map.get(sc,'?')} 无分钟线数据")
            continue
        sc_peak_time, sc_peak_amp = _find_peak_bar(sc_min_df, sc_first_open)
        if sc_peak_time:
            peak_times_minutes.append(_time_to_minutes(sc_peak_time))
        sc_amp_at_t = _get_amp_at_time(sc_min_df, t_target, sc_first_open)
        amps_at_t.append(sc_amp_at_t)
        stronger_details.append({
            "ts_code": sc,
            "name": _name_map.get(sc, "?"),
            "peak_time": sc_peak_time,
            "amp_at_t": sc_amp_at_t,
        })
        logger.debug(
            f"  [stronger_min] {sc} {_name_map.get(sc,'?')} "
            f"peak_time={sc_peak_time} peak_amp={sc_peak_amp:.2f}% "
            f"amp@T_target={sc_amp_at_t:.2f}% 耗时={_time.time()-_t_sc:.1f}s"
        )

    if not peak_times_minutes or not amps_at_t:
        result["judgement_summary"].append("stronger_minute_insufficient_neutral")
        result["is_follower"] = False
        return result

    # 时间判断：T_target < median(T_i)
    median_minutes = float(pd.Series(peak_times_minutes).median())
    median_hh = int(median_minutes) // 60
    median_mm = int(median_minutes) % 60
    median_peak_time_str = f"{median_hh:02d}:{median_mm:02d}"
    result["stronger_median_peak_time"] = median_peak_time_str
    timing_ok = _time_to_minutes(t_target) < median_minutes

    # 振幅判断：A_target 排在 {A_i} 前 amp_rank_top_pct
    n = len(amps_at_t)
    rank_above = sum(1 for a in amps_at_t if a > a_target)   # amps_at_t 不含目标股自身
    target_amp_rank_pct = (rank_above + 1) / (n + 1)          # 分母含目标股
    result["target_amp_rank_pct"] = round(target_amp_rank_pct, 3)
    amplitude_ok = target_amp_rank_pct <= amp_rank_top_pct
    amplitude_top3 = rank_above < 3   # 目标股振幅在所有股（stronger + 自身）中排前3

    result["amplitude_top3"] = amplitude_top3

    # ── Debug：T_target 时刻振幅排名，含目标股，输出前 10% ──────────────────
    _all_at_t = stronger_details + [{"ts_code": ts_code, "name": "【目标股】", "peak_time": t_target, "amp_at_t": a_target}]
    _all_at_t_sorted = sorted(_all_at_t, key=lambda x: x["amp_at_t"], reverse=True)
    _top10_count = max(1, int(len(_all_at_t_sorted) * 0.10))
    logger.debug(
        f"[follower:{ts_code}] T_target={t_target} 时刻振幅排名（共{len(_all_at_t_sorted)}只，前10%={_top10_count}只，前3={amplitude_top3}）："
    )
    for _i, _item in enumerate(_all_at_t_sorted, 1):
        _marker = " ◀ TOP10%" if _i <= _top10_count else ""
        _top3_mark = " ◀ TOP3" if _i <= 3 else ""
        _target_marker = " ★目标股" if _item["ts_code"] == ts_code else ""
        logger.debug(
            f"  #{_i:2d} {_item['ts_code']} {_item['name']}"
            f"  amp@{t_target}={_item['amp_at_t']:.2f}%  peak_time={_item['peak_time']}"
            f"{_marker}{_top3_mark}{_target_marker}"
        )

    # ── T_target 时刻振幅前3共有概念 ────────────────────────────────────────
    _top3 = _all_at_t_sorted[:3]
    _top3_concept_sets = []
    for _item in _top3:
        _c = _concept_map.get(_item["ts_code"], [])
        if _c:
            _top3_concept_sets.append(set(_c))
    if _top3_concept_sets:
        _common = _top3_concept_sets[0]
        for _s in _top3_concept_sets[1:]:
            _common = _common & _s
        leading_concepts = sorted(_common)
    else:
        leading_concepts = []
    result["leading_concepts"] = leading_concepts
    logger.debug(
        f"[follower:{ts_code}] T_target={t_target} 振幅前3共有概念（{len(leading_concepts)}个）: {leading_concepts}"
    )

    # ── 最终判断 ─────────────────────────────────────────────────────────────
    # 不跟风条件：(时间早 AND 振幅强) OR 振幅绝对前3
    not_follower = (timing_ok and amplitude_ok) or amplitude_top3
    result["timing_ok"] = timing_ok
    result["amplitude_ok"] = amplitude_ok
    result["is_follower"] = not not_follower
    _total = _time.time() - _t0
    result["judgement_summary"].append(
        f"timing={'ok' if timing_ok else 'fail'} T_target={t_target} median_T={median_peak_time_str} | "
        f"amp={'ok' if amplitude_ok else 'fail'} A_target={a_target:.2f}% "
        f"amp_rank={target_amp_rank_pct:.3f} amp_top3={amplitude_top3} | "
        f"total_time={_total:.1f}s"
    )
    logger.debug(
        f"[follower:{ts_code}] 完成 is_follower={result['is_follower']} "
        f"timing_ok={timing_ok} amplitude_ok={amplitude_ok} amplitude_top3={amplitude_top3} 总耗时={_total:.1f}s"
    )
    return result


def get_sector_stock_daily_data(sector_name: str, trade_date: str) -> pd.DataFrame:
    """
    【核心工具】查询指定板块在指定交易日的所有股票日线数据
    完全匹配需求：先查板块对应股票→再查这些股票当日日线
    :param sector_name: 板块/概念完整名称（如"人工智能"、"算力"）
    :param trade_date: 交易日，格式严格为YYYYMMDD（如"20260101"，与项目全局格式对齐）
    :return: 该板块指定交易日的全量日线数据DataFrame，查询失败/无结果返回空DataFrame
    """
    # 入参合法性校验
    if not sector_name or not str(sector_name).strip():
        logger.warning("[get_sector_stock_daily_data] 板块名称不能为空")
        return pd.DataFrame()

    trade_date_clean = str(trade_date).strip()
    if len(trade_date_clean) != 8 or not trade_date_clean.isdigit():
        logger.warning(f"[get_sector_stock_daily_data] 交易日期格式错误，要求YYYYMMDD，传入：{trade_date}")
        return pd.DataFrame()

    sector_name_clean = str(sector_name).strip()

    try:
        # 步骤1：获取板块对应的全量股票代码
        ts_code_list = get_stocks_in_sector(sector_name_clean)
        if not ts_code_list:
            logger.warning(f"[get_sector_stock_daily_data] 板块[{sector_name_clean}]无对应股票，返回空数据")
            return pd.DataFrame()

        # 步骤2：批量查询这些股票在指定交易日的日线数据
        # 表名kline_day与项目全局命名对齐，如需修改请调整表名即可
        sql = """
              SELECT *
              FROM kline_day
              WHERE ts_code IN %s
                AND trade_date = %s \
              """
        # IN查询必须传元组，适配Python MySQL参数化规范
        result_df = db.query(sql, params=(tuple(ts_code_list), trade_date_clean))

        if result_df.empty:
            logger.warning(f"[get_sector_stock_daily_data] 板块[{sector_name_clean}]在{trade_date_clean}无有效日线数据")
            return pd.DataFrame()

        logger.info(
            f"[get_sector_stock_daily_data] 板块[{sector_name_clean}]在{trade_date_clean}查询到{len(result_df)}条日线数据")
        return result_df

    except Exception as e:
        logger.error(
            f"[get_sector_stock_daily_data] 查询板块日线失败，板块：{sector_name_clean}，日期：{trade_date_clean}，错误：{str(e)}",
            exc_info=True
        )
        return pd.DataFrame()

def get_hfq_kline_range(
    ts_code_list: List[str],
    date_start: str,
    date_end: str,
) -> pd.DataFrame:
    """
    批量查询多只股票在指定日期范围内的后复权日线数据（kline_day_hfq 表）。
    用于中长线持仓区间盈亏计算，准确反映除权除息影响。

    :param ts_code_list: 股票代码列表
    :param date_start:   起始日期（含），支持 YYYY-MM-DD 或 YYYYMMDD
    :param date_end:     结束日期（含），支持 YYYY-MM-DD 或 YYYYMMDD
    :return: DataFrame，含 ts_code / trade_date / open / high / low / close 等列；
             若 kline_day_hfq 表无数据，返回空 DataFrame（调用方应降级为原始价格）
    """
    if not ts_code_list:
        return pd.DataFrame()
    start_fmt = date_start.replace("-", "")
    end_fmt   = date_end.replace("-", "")
    try:
        sql = """
            SELECT ts_code, trade_date, open, high, low, close, pre_close, volume, amount
            FROM kline_day_hfq
            WHERE ts_code IN %s
              AND trade_date >= %s
              AND trade_date <= %s
            ORDER BY ts_code, trade_date
        """
        df = db.query(sql, params=(tuple(ts_code_list), start_fmt, end_fmt), return_df=True)
        return df if df is not None else pd.DataFrame()
    except Exception as e:
        logger.warning(f"[get_hfq_kline_range] 查询失败（kline_day_hfq 表可能尚未建立）| {e}")
        return pd.DataFrame()


def get_qfq_kline_range(
    ts_code_list: List[str],
    date_start: str,
    date_end: str,
) -> pd.DataFrame:
    """
    批量查询多只股票在指定日期范围内的前复权日线数据（kline_day_qfq 表）。
    用于 MA 计算、突破信号等需要连续价格（消除除权跳空影响）的场景。

    :param ts_code_list: 股票代码列表
    :param date_start:   起始日期（含），支持 YYYY-MM-DD 或 YYYYMMDD
    :param date_end:     结束日期（含），支持 YYYY-MM-DD 或 YYYYMMDD
    :return: DataFrame，含 ts_code / trade_date / open / high / low / close / volume 等列
    """
    if not ts_code_list:
        return pd.DataFrame()
    start_fmt = date_start.replace("-", "")
    end_fmt   = date_end.replace("-", "")
    try:
        sql = """
            SELECT ts_code, trade_date, open, high, low, close, volume, amount
            FROM kline_day_qfq
            WHERE ts_code IN %s
              AND trade_date >= %s
              AND trade_date <= %s
            ORDER BY ts_code, trade_date
        """
        df = db.query(sql, params=(tuple(ts_code_list), start_fmt, end_fmt), return_df=True)
        return df if df is not None else pd.DataFrame()
    except Exception as e:
        logger.error(f"[get_qfq_kline_range] 查询失败 | {date_start}~{date_end} | {e}")
        return pd.DataFrame()


def get_ex_div_stocks(trade_date: str) -> set:
    """
    返回指定交易日发生除权除息的股票代码集合。

    判断依据：原始价格涨跌幅（kline_day.pct_chg）与前复权涨跌幅（kline_day_qfq.pct_chg）的差值。
    除权除息日，原始价格因除息而下跌，但前复权价格连续（无跳空），两者差值明显为负。
    阈值 -3.0% 可覆盖绝大多数送转股/派息场景，误判率极低。

    :param trade_date: 交易日，格式 YYYY-MM-DD 或 YYYYMMDD
    :return: 当日发生除权除息的 ts_code 集合（无数据时返回空集合）
    """
    trade_date_fmt = trade_date.replace("-", "")
    sql = """
        SELECT k.ts_code
        FROM kline_day k
        JOIN kline_day_qfq q ON k.ts_code = q.ts_code AND k.trade_date = q.trade_date
        WHERE k.trade_date = %s
          AND k.pct_chg IS NOT NULL
          AND q.pct_chg IS NOT NULL
          AND (k.pct_chg - q.pct_chg) < -3.0
    """
    try:
        rows = db.query(sql, params=(trade_date_fmt,)) or []
        return {r["ts_code"] for r in rows}
    except Exception as e:
        logger.warning(f"[get_ex_div_stocks] 查询失败 | {trade_date} | {e}")
        return set()


def get_ex_div_dates_for_stock(ts_code: str, date_start: str, date_end: str) -> set:
    """
    从 stock_dividend 表查询指定股票在日期范围内的除权除息日集合。
    替代 get_ex_div_stocks 的 JOIN 查询，性能提升数十倍。

    :param ts_code:    股票代码
    :param date_start: 起始日期（含），支持 YYYY-MM-DD 或 YYYYMMDD
    :param date_end:   结束日期（含）
    :return: 除权除息日集合（YYYY-MM-DD 格式）
    """
    start_fmt = date_start.replace("-", "")
    end_fmt = date_end.replace("-", "")
    sql = """
        SELECT ex_date FROM stock_dividend
        WHERE ts_code = %s AND ex_date >= %s AND ex_date <= %s AND ex_date IS NOT NULL
    """
    try:
        rows = db.query(sql, params=(ts_code, start_fmt, end_fmt)) or []
        result = set()
        for r in rows:
            d = r["ex_date"]
            if d:
                result.add(d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d))
        return result
    except Exception as e:
        logger.warning(f"[get_ex_div_dates_for_stock] 查询失败 | {ts_code} | {e}")
        return set()


def get_market_limit_counts(trade_date: str) -> Dict:
    """
    查询当日涨停/跌停数量（从 limit_list_ths 表）。
    轻量查询，供长线引擎市场恐慌判断使用。

    :param trade_date: 交易日，支持 YYYY-MM-DD 或 YYYYMMDD
    :return: {"limit_up_count": int, "limit_down_count": int}
    """
    trade_date_fmt = trade_date.replace("-", "")
    sql = """
        SELECT limit_type, COUNT(*) AS cnt
        FROM limit_list_ths
        WHERE trade_date = %s AND limit_type IN ('涨停池', '跌停池')
        GROUP BY limit_type
    """
    result = {"limit_up_count": 0, "limit_down_count": 0}
    try:
        rows = db.query(sql, params=(trade_date_fmt,)) or []
        for r in rows:
            if r.get("limit_type") == "涨停池":
                result["limit_up_count"] = int(r["cnt"])
            elif r.get("limit_type") == "跌停池":
                result["limit_down_count"] = int(r["cnt"])
    except Exception as e:
        logger.warning(f"[get_market_limit_counts] 查询失败 | {trade_date} | {e}")
    return result


def get_stock_limit_counts_batch(
    ts_code_list: List[str], start_date: str, end_date: str
) -> Dict[str, int]:
    """
    批量查询多只股票在日期区间内的涨停+跌停总次数（从 limit_list_ths 表）。
    一次 DB 请求覆盖所有股票，比逐只查询或 pct_chg 近似判断更准确。

    :param ts_code_list: 股票代码列表
    :param start_date:   开始日期，YYYY-MM-DD 或 YYYYMMDD
    :param end_date:     结束日期，YYYY-MM-DD 或 YYYYMMDD
    :return: {ts_code: limit_hit_count}，未上榜的股票不在字典中（视为 0）
    """
    if not ts_code_list:
        return {}
    start_fmt = start_date.replace("-", "")
    end_fmt   = end_date.replace("-", "")
    placeholders = ",".join(["%s"] * len(ts_code_list))
    sql = f"""
        SELECT ts_code, COUNT(*) AS cnt
        FROM limit_list_ths
        WHERE ts_code IN ({placeholders})
          AND trade_date BETWEEN %s AND %s
          AND limit_type IN ('涨停池', '跌停池')
        GROUP BY ts_code
    """
    try:
        rows = db.query(sql, params=tuple(ts_code_list) + (start_fmt, end_fmt)) or []
        return {r["ts_code"]: int(r["cnt"]) for r in rows}
    except Exception as e:
        logger.warning(f"[get_stock_limit_counts_batch] 查询失败：{e}")
        return {}


def has_recent_limit_up_batch(
    ts_code_list: List[str],
    start_date: str,
    end_date: str,
    *,
    conservative_on_error: bool = False,
) -> Dict[str, bool]:
    """
    批量判断多只股票在日期区间内是否出现过涨停（仅统计 limit_list_ths 的"涨停池"）。

    设计目标：
    1. 替代"逐日拉 kline_day + 逐行算涨停价"的高成本实现；
    2. 数据中性原则：查询异常时默认全 False（无涨停信号），避免引入偏向性正样本；
    3. 向后兼容旧调用方的返回格式：{ts_code: bool}

    :param ts_code_list: 股票代码列表
    :param start_date:   开始日期（含），支持 YYYY-MM-DD / YYYYMMDD
    :param end_date:     结束日期（含），支持 YYYY-MM-DD / YYYYMMDD
    :param conservative_on_error: 查询异常时是否全返回 True（默认 True）
    :return: {ts_code: True=区间内至少有一次涨停, False=没有涨停记录}
    """
    if not ts_code_list:
        return {}

    start_fmt = start_date.replace("-", "")
    end_fmt = end_date.replace("-", "")
    placeholders = ",".join(["%s"] * len(ts_code_list))
    sql = f"""
        SELECT DISTINCT ts_code
        FROM limit_list_ths
        WHERE ts_code IN ({placeholders})
          AND trade_date BETWEEN %s AND %s
          AND limit_type = '涨停池'
    """

    try:
        rows = db.query(sql, params=tuple(ts_code_list) + (start_fmt, end_fmt)) or []
        hit_set = {r["ts_code"] for r in rows if r.get("ts_code")}
        return {ts_code: ts_code in hit_set for ts_code in ts_code_list}
    except Exception as e:
        logger.warning(
            f"[has_recent_limit_up_batch] 查询失败 | {start_date}~{end_date} | {e}"
        )
        default_value = True if conservative_on_error else False
        return {ts_code: default_value for ts_code in ts_code_list}


def get_index_pct_chg(trade_date: str, index_codes: List[str] = None) -> Dict[str, float]:
    """
    查询当日核心指数涨跌幅（从 index_daily 表）。
    轻量查询，供长线引擎市场恐慌判断使用。

    :param trade_date: 交易日
    :param index_codes: 指数代码列表，默认上证+深证+创业板
    :return: {ts_code: pct_chg, ...}
    """
    if index_codes is None:
        index_codes = ["000001.SH", "399001.SZ", "399006.SZ"]
    trade_date_fmt = trade_date.replace("-", "")
    sql = "SELECT ts_code, pct_chg FROM index_daily WHERE trade_date = %s AND ts_code IN %s"
    try:
        rows = db.query(sql, params=(trade_date_fmt, tuple(index_codes))) or []
        return {r["ts_code"]: float(r.get("pct_chg", 0) or 0) for r in rows}
    except Exception as e:
        logger.warning(f"[get_index_pct_chg] 查询失败 | {trade_date} | {e}")
        return {}


def get_kline_day_range(
    ts_code_list: List[str],
    date_start: str,
    date_end: str,
) -> pd.DataFrame:
    """
    批量查询多只股票在指定日期范围内的日线数据。
    date_start / date_end 支持 YYYY-MM-DD 或 YYYYMMDD 两种格式，内部统一转 YYYYMMDD。

    :param ts_code_list: 股票代码列表
    :param date_start:   起始日期（含）
    :param date_end:     结束日期（含）
    :return: DataFrame，含 ts_code / trade_date / open / high / low / close / pre_close 等列
    """
    if not ts_code_list:
        return pd.DataFrame()
    start_fmt = date_start.replace("-", "")
    end_fmt   = date_end.replace("-", "")
    try:
        sql = """
            SELECT ts_code, trade_date, open, high, low, close, pre_close, volume, amount
            FROM kline_day
            WHERE ts_code IN %s
              AND trade_date >= %s
              AND trade_date <= %s
            ORDER BY ts_code, trade_date
        """
        df = db.query(sql, params=(tuple(ts_code_list), start_fmt, end_fmt), return_df=True)
        return df if df is not None else pd.DataFrame()
    except Exception as e:
        logger.error(f"[get_kline_day_range] 查询失败 | {date_start}~{date_end} | {e}")
        return pd.DataFrame()


def ensure_dividend_data(ts_code_list: List[str]) -> None:
    """
    确保 stock_dividend 表有指定股票的分红数据（DB→API→DB 链路）。

    一次批量 IN 查询找出哪些股票已入库，仅对 missing 股票调 API 补拉（skip_check=True
    跳过 clean_and_insert_dividend 内部的冗余 SELECT 1 检查）。
    """
    if not ts_code_list:
        return
    from data.data_cleaner import data_cleaner

    # 一次批量查哪些股票已有记录
    try:
        sql = "SELECT DISTINCT ts_code FROM stock_dividend WHERE ts_code IN %s"
        rows = db.query(sql, params=(tuple(ts_code_list),)) or []
        existing = {r["ts_code"] for r in rows if r.get("ts_code")}
    except Exception as e:
        logger.warning(f"[ensure_dividend_data] 查库失败：{e}，跳过补拉")
        return

    missing = [ts for ts in ts_code_list if ts not in existing]
    if not missing:
        return

    logger.info(f"[ensure_dividend_data] {len(missing)}只股票需补拉分红数据")
    for ts_code in missing:
        try:
            data_cleaner.clean_and_insert_dividend(ts_code, skip_check=True)
        except Exception as e:
            logger.warning(f"[ensure_dividend_data] {ts_code} 补拉失败：{e}")


def get_dividend_check_batch(
    ts_code_list: List[str],
    start_date: str,
    end_date: str,
) -> Set[str]:
    """
    查询区间内有除权除息事件的股票集合（stock_dividend 表单次范围查询）。

    调用前须先调用 ensure_dividend_data(ts_code_list) 保证数据已入库。
    之后不在返回集合中的股票，即可视为区间内无除权：
    - 无除权 → 原始日线与 HFQ 的相对关系（MA/突破/涨幅）完全一致，可直接用 kline_day
    - 有除权 → 必须用 HFQ 消除价格跳空

    :param ts_code_list: 股票代码列表（仅用于 Python 层交集，不传入 SQL）
    :param start_date:   开始日期（含），支持 YYYY-MM-DD / YYYYMMDD
    :param end_date:     结束日期（含）
    :return: 区间内有 ex_date 记录的股票代码集合
    """
    if not ts_code_list:
        return set()
    start_fmt = start_date.replace("-", "")
    end_fmt = end_date.replace("-", "")
    candidate_set = set(ts_code_list)

    try:
        sql = """
            SELECT DISTINCT ts_code FROM stock_dividend
            WHERE ex_date BETWEEN %s AND %s
              AND ex_date IS NOT NULL
        """
        rows = db.query(sql, params=(start_fmt, end_fmt)) or []
        all_div_stocks = {r["ts_code"] for r in rows if r.get("ts_code")}
        return all_div_stocks & candidate_set
    except Exception as e:
        logger.warning(f"[get_dividend_check_batch] 查询失败：{e}，全部保守走 HFQ")
        # 查询失败时返回整个候选集，调用方将所有股票走 HFQ（保守兜底）
        return candidate_set


def ensure_qfq_data(
    ts_code_list: List[str],
    date_start: str,
    date_end: str,
    adj: str = "qfq",
    max_retries: int = 3,
) -> None:
    """
    确保 kline_day_qfq（或 kline_day_hfq）数据在指定区间内完整。

    覆盖率基准：kline_day 该股该区间的行数（停牌日 kline_day 无记录，自然不计入）。
    这比按交易日历计算更准确，因为个股停牌不反映在复权表里也是正确的。

    流程（每只不足的股票）：
      1. 一条 SQL JOIN 找出哪些股票 adj 行数 < kline_day 行数
      2. 对每只缺口股票：找缺失日期的最小/最大值，请求该子区间
      3. 带重试；重试耗尽后仍不足 → 抛 RuntimeError（调用方决定如何处理）

    :param ts_code_list: 股票代码列表
    :param date_start:   起始日期（含），支持 YYYY-MM-DD / YYYYMMDD
    :param date_end:     结束日期（含）
    :param adj:          "qfq"（前复权）或 "hfq"（后复权）
    :param max_retries:  每只股票最大重试次数
    """
    from data.data_cleaner import data_cleaner

    if not ts_code_list:
        return

    table   = "kline_day_qfq" if adj == "qfq" else "kline_day_hfq"
    start_f = date_start.replace("-", "")
    end_f   = date_end.replace("-", "")

    # ── Step 1：批量找出哪些股票 adj 数据不足（一条 JOIN SQL）────────────
    check_sql = f"""
        SELECT
            k.ts_code,
            COUNT(k.trade_date)  AS kline_count,
            COUNT(q.trade_date)  AS adj_count
        FROM kline_day k
        LEFT JOIN {table} q
            ON k.ts_code = q.ts_code AND k.trade_date = q.trade_date
        WHERE k.ts_code IN %s
          AND k.trade_date >= %s
          AND k.trade_date <= %s
        GROUP BY k.ts_code
        HAVING kline_count > adj_count
    """
    try:
        deficient = db.query(
            check_sql,
            params=(tuple(ts_code_list), start_f, end_f),
            return_df=True,
        )
    except Exception as e:
        logger.warning(f"[ensure_qfq_data] 覆盖率检查 SQL 失败：{e}")
        return

    if deficient is None or deficient.empty:
        return  # 全部已完整

    logger.info(
        f"[ensure_qfq_data][{table}] 发现 {len(deficient)} 只股票数据不足，开始补拉"
    )

    # ── Step 2：逐只补拉 ─────────────────────────────────────────────────
    for _, row in deficient.iterrows():
        ts_code  = row["ts_code"]
        expected = int(row["kline_count"])
        actual   = int(row["adj_count"])

        # 找缺失日期区间（缺失的最早~最晚，作为补拉 start/end）
        missing_sql = f"""
            SELECT MIN(k.trade_date) AS miss_start, MAX(k.trade_date) AS miss_end
            FROM kline_day k
            LEFT JOIN {table} q
                ON k.ts_code = q.ts_code AND k.trade_date = q.trade_date
            WHERE k.ts_code = %s
              AND k.trade_date >= %s
              AND k.trade_date <= %s
              AND q.trade_date IS NULL
        """
        try:
            miss_df = db.query(missing_sql, params=(ts_code, start_f, end_f), return_df=True)
        except Exception as e:
            logger.warning(f"[ensure_qfq_data] {ts_code} 缺失日期查询失败：{e}")
            continue

        if miss_df is None or miss_df.empty or miss_df.iloc[0]["miss_start"] is None:
            continue

        miss_start = str(miss_df.iloc[0]["miss_start"]).replace("-", "")
        miss_end   = str(miss_df.iloc[0]["miss_end"]).replace("-", "")

        logger.info(
            f"[ensure_qfq_data] {ts_code} {table} 缺 {expected - actual} 天 "
            f"({actual}/{expected})，补拉区间 {miss_start}~{miss_end}"
        )

        success = False
        for attempt in range(1, max_retries + 1):
            try:
                if adj == "qfq":
                    data_cleaner.clean_and_insert_kline_day_qfq(
                        ts_code=ts_code, start_date=miss_start, end_date=miss_end
                    )
                else:
                    data_cleaner.clean_and_insert_kline_day_hfq(
                        ts_code=ts_code, start_date=miss_start, end_date=miss_end
                    )

                # 验证是否补全
                verify = db.query(
                    f"SELECT COUNT(*) AS cnt FROM {table} "
                    f"WHERE ts_code = %s AND trade_date >= %s AND trade_date <= %s",
                    params=(ts_code, start_f, end_f),
                    return_df=True,
                )
                actual_new = int(verify.iloc[0]["cnt"]) if (verify is not None and not verify.empty) else 0
                if actual_new >= expected:
                    logger.info(f"[ensure_qfq_data] {ts_code} 补拉成功（{actual_new}/{expected}）")
                    success = True
                    break

                logger.warning(
                    f"[ensure_qfq_data] {ts_code} 第{attempt}次补拉后仍不足 "
                    f"({actual_new}/{expected})，继续重试"
                )
                time.sleep(1.5)

            except Exception as e:
                logger.warning(f"[ensure_qfq_data] {ts_code} 第{attempt}次补拉异常：{e}")
                if attempt < max_retries:
                    time.sleep(1.5)

        if not success:
            raise RuntimeError(
                f"[ensure_qfq_data] {ts_code} {table} 数据补拉失败："
                f"{max_retries} 次重试后仍不足 ({actual}/{expected})，"
                f"缺失区间 {miss_start}~{miss_end}，请检查 Tushare 数据源"
            )


def ensure_limit_list_ths_data(trade_date: str) -> None:
    """
    确保 limit_list_ths 表有指定交易日的涨跌停数据。
    若 DB 无数据，走 API 补拉（与 DataBundle 相同的 DB→API→DB 链路）。

    :param trade_date: 交易日，支持 YYYY-MM-DD / YYYYMMDD
    """
    from data.data_cleaner import data_cleaner

    trade_date_fmt = trade_date.replace("-", "")
    check_sql = "SELECT 1 FROM limit_list_ths WHERE trade_date = %s LIMIT 1"
    try:
        existing = db.query(check_sql, (trade_date_fmt,))
        if existing:
            return
        logger.info(f"[ensure_limit_list_ths_data] {trade_date} 无涨跌停数据，触发 API 补拉")
        data_cleaner.clean_and_insert_limit_list_ths(trade_date=trade_date_fmt)
    except Exception as e:
        logger.warning(f"[ensure_limit_list_ths_data] {trade_date} 补拉失败：{e}")


def ensure_stk_factor_pro_data(trade_date: str) -> None:
    """
    确保 stk_factor_pro 表有指定交易日的技术面因子数据。
    若 DB 无数据，走 API 补拉（DB→API→DB 链路）。

    :param trade_date: 交易日，支持 YYYY-MM-DD / YYYYMMDD
    """
    from data.data_cleaner import data_cleaner

    trade_date_fmt = trade_date.replace("-", "")
    check_sql = "SELECT 1 FROM stk_factor_pro WHERE trade_date = %s LIMIT 1"
    try:
        existing = db.query(check_sql, (trade_date_fmt,))
        if existing:
            return
        logger.info(f"[ensure_stk_factor_pro_data] {trade_date} 无技术面因子数据，触发 API 补拉")
        data_cleaner.clean_and_insert_stk_factor_pro(trade_date=trade_date_fmt)
    except Exception as e:
        logger.warning(f"[ensure_stk_factor_pro_data] {trade_date} 补拉失败：{e}")


if __name__ == "__main__":
    import json
    import logging
    logging.getLogger("a_quant").setLevel(logging.DEBUG)
    result = analyze_stock_follower_strength("300724.SZ", "20260320")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    # pass
