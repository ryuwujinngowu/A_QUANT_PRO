from datetime import datetime
from typing import Dict
import os
import pandas as pd
# 导入核心配置：涨停率、最大持仓数 + 新增可配置的股票类型过滤开关
from config.config  import (
    MAIN_BOARD_LIMIT_UP_RATE,
    STAR_BOARD_LIMIT_UP_RATE,
    BJ_BOARD_LIMIT_UP_RATE,
    MAX_POSITION_COUNT,
    FILTER_BSE_STOCK,
    FILTER_STAR_BOARD,
    FILTER_MAIN_BOARD
)

from data.data_cleaner import data_cleaner
from strategies import BaseStrategy
from utils.log_utils import logger

# 分钟线缓存开关（全局配置）
CACHE_ENABLE = True
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "config.py")

class MultiLimitUpStrategy(BaseStrategy):
    """多标的涨停策略（核心：首次触板/一字板回封选股，分仓控制）"""

    def __init__(self):
        """策略初始化：加载配置+初始化缓存/信号容器"""
        super().__init__()

        self.max_position_count = MAX_POSITION_COUNT  # 最大持仓数量（分仓个数）
        self.limit_up_price_tolerance = 0.999  # 涨停价容忍度（避免价格微小误差漏判）
        self.sell_signal_map: Dict[str, str] = {}  # 卖出信号映射：{股票代码: 卖出类型(open/close)}
        self.min_data_cache: Dict[tuple, pd.DataFrame] = {}  # 分钟线缓存：{(ts_code, trade_date): 分钟线DF}
        self.SELL_TYPE_AFTER_BLOWUP = "open"  # 止损触发,默认开盘卖，可配置为close
        self.strategy_name = f"多标的涨停打板策略(持仓直到断板,炸板次日{self.SELL_TYPE_AFTER_BLOWUP}止损)"
        self.initialize()  # 重置信号/缓存



    def initialize(self):
        """策略重置（回测/实盘启动前执行）"""
        self.sell_signal_map = {}  # 清空卖出信号（避免跨周期残留）
        if CACHE_ENABLE:
            self.min_data_cache.clear()  # 清空分钟线缓存

    # ========= 涨停幅度计算（核心工具方法） =========
    def get_limit_up_rate_by_ts_code(self, ts_code: str) -> float:
        """
        根据股票代码判断涨停幅度（区分主板/双创板/北交所）
        :param ts_code: 股票代码（如600000.SH、300001.SZ、830001.BJ）
        :return: 涨停幅度（0=北交所/10%=主板/20%=双创板）
        """
        code, exch = ts_code.split('.')
        # 北交所股票：无涨停限制
        if code.startswith(('83', '87', '88')) or exch == 'BJ':
            return BJ_BOARD_LIMIT_UP_RATE
        # 双创板（创业板300/科创板688）：20%涨停
        if code.startswith(('30', '688')):
            return STAR_BOARD_LIMIT_UP_RATE
        # 主板（沪市60/深市00）：10%涨停
        return MAIN_BOARD_LIMIT_UP_RATE



    def _filter_stock_by_type(self, ts_code: str) -> bool:
        """
        问题4：可配置的股票类型过滤（替代原_is_not_bse_stock）
        :param ts_code: 股票代码
        :return: True=保留该股票（符合过滤规则），False=过滤掉
        """
        code, exch = ts_code.split('.')
        # 北交所股票过滤逻辑
        is_bse = code.startswith(('83', '87', '88')) or exch == 'BJ'
        if FILTER_BSE_STOCK and is_bse:
            return False  # 过滤北交所

        # 双创板（创业板/科创板）过滤逻辑
        is_star_board = code.startswith(('300', '688'))
        if FILTER_STAR_BOARD and is_star_board:
            return False  # 过滤双创板

        # 主板过滤逻辑
        is_main_board = not (is_bse or is_star_board)
        if FILTER_MAIN_BOARD and is_main_board:
            return False  # 过滤主板

        # 保留该股票
        return True

    # ========= 分钟线数据获取（核心工具方法） =========
    def _get_min_df(self, ts_code, trade_date):
        key = (ts_code, trade_date)
        if key in self.min_data_cache:
            logger.debug(f"[{ts_code}-{trade_date}] 分钟线缓存命中，直接返回")
            return self.min_data_cache[key]

        # 第一步：优先调用cleaner已有方法获取分钟线
        min_df = data_cleaner.get_kline_min_by_stock_date(ts_code, trade_date)
        if not min_df.empty:
            self.min_data_cache[key] = min_df
            logger.debug(f"[{ts_code}-{trade_date}] 从数据库读取分钟线数据，行数={len(min_df)}")
            return min_df


    # ========= 首次触板判断（核心选股逻辑） =========
    def get_first_limit_time(self, min_df, limit_price):
        """
        问题2：改用high判断触板（分钟内high触涨停但close回落也需算触板）
        :param min_df: 分钟线DF
        :param limit_price: 涨停价（元）
        :return: 首次触板时间（datetime/None）
        """
        # 确保trade_time为datetime类型（避免字符串排序错误）
        if not pd.api.types.is_datetime64_any_dtype(min_df["trade_time"]):
            min_df["trade_time"] = pd.to_datetime(min_df["trade_time"])

        # 问题2修复：用high代替close，因为分钟内最高价触板即算触板（即使收盘回落）
        hit = min_df[min_df.high >= limit_price * self.limit_up_price_tolerance]
        logger.debug(f"触板判断：涨停价={limit_price}，容忍度={self.limit_up_price_tolerance}，触板分钟数={len(hit)}")

        if hit.empty:
            return None

        # 按时间排序，取首次触板时间
        first_hit_time = hit.sort_values("trade_time").trade_time.iloc[0]
        logger.debug(f"首次触板时间={first_hit_time}")
        return first_hit_time


    # ========= 核心选股逻辑（生成买入列表） =========
    def select_buy_stocks(self, trade_date, daily_df, available_pos):
        """
        筛选买入股票（首次触板/一字板回封）- 性能优化版（保留全量遍历，保证排序准确性）
        :param trade_date: 交易日
        :param daily_df: 当日全市场日线DF
        :param available_pos: 可用仓位数量
        :return: 买入股票列表（按触板/回封时间排序）
        """
        if available_pos <= 0:
            logger.debug(f"[{trade_date}] 可用仓位={available_pos}，无买入")
            return []

        # 过滤股票类型 + 过滤无效前收盘价（保留copy消除警告）
        daily_df_filtered = daily_df.loc[
            daily_df.ts_code.apply(self._filter_stock_by_type) & (daily_df["pre_close"] > 0)
            ].copy()
        logger.debug(f"[{trade_date}] 过滤后股票数量={len(daily_df_filtered)}")

        # 向量化计算涨停价（原逻辑不变）
        rates = daily_df_filtered.ts_code.map(self.get_limit_up_rate_by_ts_code)
        daily_df_filtered["limit_up_price"] = (daily_df_filtered.pre_close * (1 + rates)).round(2)
        logger.debug(f"[{trade_date}] 向量化计算涨停价完成，有效股票数={len(daily_df_filtered)}")

        # 候选池生成（原逻辑不变）
        candidates = daily_df_filtered[
            daily_df_filtered.high >= daily_df_filtered.limit_up_price * self.limit_up_price_tolerance
            ]
        candidate_count = len(candidates)
        logger.info(f"{trade_date} 候选池数量: {candidate_count}")

        # ========== 核心优化1：批量+多线程获取所有候选股分钟线（解决接口耗时） ==========
        min_data_dict = {}
        if candidate_count > 0:
            candidate_ts_codes = candidates.ts_code.unique()

            logger.debug(f"候选股：{candidate_ts_codes}")
            min_data_dict = self._batch_get_min_df(candidate_ts_codes, trade_date)


        # ========== 优化2：预计算常量（减少循环内属性访问） ==========
        tolerance = self.limit_up_price_tolerance
        hits = {}  # 存储所有符合条件股票的触板/回封时间（全量收集）

        # ========== 保留全量遍历（保证排序准确性） ==========
        for row in candidates.itertuples(index=False):
            ts = row.ts_code
            limit_price = row.limit_up_price

            # 从批量结果中取分钟线（无IO，直接读内存字典）
            min_df = min_data_dict.get(ts, pd.DataFrame())
            if min_df is None or min_df.empty:
                logger.debug(f"[{ts}-{trade_date}] 无分钟线数据，跳过该股票")
                continue

            # ========== 原核心逻辑完全不变（全量校验） ==========
            is_limit_open = row.open >= limit_price * tolerance
            if is_limit_open:
                logger.debug(f"[{ts}-{trade_date}] 判定为一字板，需校验开板+回封")
                t = self.check_reopen(min_df, limit_price, row, ts)
                if not t:
                    logger.debug(f"[{ts}-{trade_date}] 一字板但未开板/未回封，实盘无法买入，跳过")
                    continue
                hits[ts] = t  # 全量收集，不提前终止
            else:
                t = self.get_first_limit_time(min_df, limit_price)
                if t:
                    logger.debug(f"[{ts}-{trade_date}] 非一字板，首次触板时间={t}，纳入买入候选")
                    hits[ts] = t  # 全量收集，不提前终止
                else:
                    logger.debug(f"[{ts}-{trade_date}] 非一字板且无首次触板，跳过")

        # ========== 原排序逻辑不变（基于全量hits排序，保证准确性） ==========
        ordered = sorted(hits.items(), key=lambda x: x[1])

        buy_stocks = [x[0] for x in ordered[:available_pos]]

        # logger.info(f"{trade_date} 可用仓位:{available_pos}，最终买入列表（符合实盘逻辑）：{buy_stocks}，数量={len(buy_stocks)}")
        return buy_stocks

    # ========== 保留：批量+多线程获取分钟线的辅助方法（核心优化，不影响排序） ==========
    def _batch_get_min_df(self, ts_codes, trade_date):
        """
        多线程批量获取分钟线（复用原_get_min_df逻辑，仅并行执行，保证数据完整）
        :param ts_codes: 股票代码列表
        :param trade_date: 交易日
        :return: {ts_code: min_df}
        """
        import concurrent.futures
        min_data_dict = {}

        # 第一步：先查缓存（复用原缓存逻辑，无IO）
        cache_hits = []
        cache_misses = []
        for ts in ts_codes:
            key = (ts, trade_date)
            if key in self.min_data_cache and not self.min_data_cache[key].empty:
                min_data_dict[ts] = self.min_data_cache[key]
                cache_hits.append(ts)
            else:
                cache_misses.append(ts)
        logger.debug(f"[{trade_date}] 分钟线缓存命中{len(cache_hits)}只，未命中{len(cache_misses)}只")

        # 第二步：多线程获取未命中缓存的股票（核心：并行IO，不影响数据完整性）
        if cache_misses:
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(self._get_min_df, ts, trade_date): ts for ts in cache_misses}
                for future in concurrent.futures.as_completed(futures):
                    # time.sleep(1)
                    ts = futures[future]
                    try:
                        min_df = future.result()
                        min_data_dict[ts] = min_df
                        # 存入缓存（复用原逻辑）
                        if min_df is not None and not min_df.empty:
                            self.min_data_cache[(ts, trade_date)] = min_df
                    except Exception as e:
                        logger.error(f"[{ts}-{trade_date}] 批量获取分钟线失败：{str(e)}")
                        min_data_dict[ts] = pd.DataFrame()

        return min_data_dict

    # ========== 核心改动2：重构check_reopen方法，强化开板校验 ==========
    def check_reopen(self, min_df, limit_price, daily_row, ts):
        """
        校验一字板开板回封（严格符合实盘：必须开板且回封）
        :param ts:
        :param min_df: 分钟线DF
        :param limit_price: 涨停价
        :param daily_row: 日线数据（NamedTuple/Series）
        :return: 回封时间（datetime/None）
        """
        tolerance = self.limit_up_price_tolerance
        threshold = limit_price * tolerance

        # 1. 再次确认一字板（双层校验，避免漏判）
        open_price = daily_row.open if hasattr(daily_row, 'open') else 0
        if open_price < threshold:
            logger.debug(f"{ts}非一字板（开盘价={open_price} < 阈值={threshold}），跳过回封校验")
            return None

        # 2. 校验是否开板（核心：存在至少1分钟的最低价 < 阈值）
        min_low = min_df['low'].min()
        if min_low >= threshold:
            logger.debug(f"{ts}一字板全天未开板（最低low={min_low} ≥ 阈值={threshold}），实盘无法买入，返回None")
            return None

        # 3. 校验开板后是否回封
        df = min_df.sort_values("trade_time").reset_index(drop=True)
        for i, row in df.iterrows():
            # 开板判定：本分钟最低价 < 阈值
            if row.low < threshold:
                logger.debug(f"[{ts}{row.trade_time}] 一字板开板（low={row.low} < 阈值={threshold}）")
                # 情况1：本分钟回封（收盘价≥阈值）
                if row.close >= threshold:
                    logger.debug(f"[{ts}，{row.trade_time}] 本分钟回封，返回该时间")
                    return row.trade_time
                # 情况2：下一分钟回封（跨分钟回封）
                if i + 1 < len(df):
                    next_row = df.iloc[i + 1]
                    if next_row.close >= threshold:
                        logger.debug(f"[{ts}，{next_row.trade_time}] 下一分钟回封，返回该时间")
                        return next_row.trade_time

        # 开板后未回封（实盘也无法买入）
        logger.debug(f"{ts},一字板开板后未回封，返回None")
        return None


    def _unify_date_format(self, date_str: str) -> str:
        """统一日期格式为YYYYMMDD（和Account/Position类保持一致）"""
        try:
            return datetime.strptime(date_str.replace("-", ""), "%Y%m%d").strftime("%Y%m%d")
        except Exception as e:
            logger.error(f"策略层日期格式转换失败：{date_str}，错误：{e}")
            return date_str

    # ========= 买卖信号生成（核心入口方法） =========
    def generate_signal(self, trade_date, daily_df, positions):
        """
        生成买卖信号（回测引擎核心调用方法）
        :param trade_date: 交易日
        :param daily_df: 当日全市场日线DF
        :param positions: 当前持仓：{ts_code: 持仓对象（含hold_days字段）}
        :return: (buy_stocks, sell_signal_map)
        """
        sell_signal_map = {}

        # 遍历当前持仓，生成卖出信号
        for ts_code, pos in positions.items():
            # 筛选该股票当日日线数据
            row = daily_df[daily_df.ts_code == ts_code]
            # 问题9：row.empty时打印debug日志
            if row.empty:
                logger.debug(f"[{ts_code}-{trade_date}] 无当日日线数据，跳过卖出判断")
                continue
            row = row.iloc[0]  # 转为Series，方便取值

            # 计算涨停价，判断是否涨停
            limit_price = self.calc_limit_up_price(ts_code, row["pre_close"])
            is_limit = row["close"] >= limit_price * self.limit_up_price_tolerance
            """
            hold_days业务逻辑（确保回测引擎正确接收信号）：
            1. hold_days=0：当日买入的股票（持仓天数0）
               - 未涨停 → 生成"open"信号 → 回测引擎次日开盘执行卖出
            2. hold_days>0：持仓超1天的股票
               - 未涨停 → 生成"close"信号 → 回测引擎当日收盘执行卖出
            信号传递：sell_signal_map会被回测引擎保存，次日处理"open"信号，当日处理"close"信号
            """
            if pos.buy_date == self._unify_date_format(trade_date) and not is_limit:
                # 仅当「当日买入（buy_date=当前交易日）」且未涨停时，生成次日炸板卖出信号
                sell_signal_map[ts_code] = self.SELL_TYPE_AFTER_BLOWUP
                logger.info(
                    f"[{ts_code}-{trade_date}] 当日（{trade_date}）买入未涨停，生成次日{self.SELL_TYPE_AFTER_BLOWUP}卖出信号（配置项控制）")
            elif pos.buy_date <= self._unify_date_format(trade_date) and not is_limit:
                # 持仓超1天（或D+1日盘中hold_days=0但非当日买入）未涨停，生成当日收盘卖出信号
                sell_signal_map[ts_code] = "close"
                logger.info(f"[{ts_code}-{trade_date}] 持仓超{pos.hold_days}天（买入日期：{pos.buy_date}）未涨停，生成止损卖出信号")
        # 计算可用仓位，生成买入信号
        available_pos = max(0, self.max_position_count - len(positions))
        buy_stocks = self.select_buy_stocks(trade_date, daily_df, available_pos)
        return buy_stocks, sell_signal_map