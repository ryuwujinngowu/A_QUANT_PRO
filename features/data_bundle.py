"""
特征数据容器 (FeatureDataBundle)
==================================
设计原则：
    1. 由外部（dataset.py）在特征计算前统一构建，所有因子类共享同一份数据
    2. 日线 / 分钟线各只发起一次 IO，因子内部禁止再自行拉数据
    3. load_minute=False 可跳过分钟线加载，适用于纯日线因子调试场景
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import List, Dict
import pandas as pd

from utils.common_tools import (
    get_trade_dates, get_daily_kline_data,
    get_limit_list_ths, get_limit_step, get_limit_cpt_list, get_index_daily,
    get_market_total_volume,
)
from data.data_cleaner import data_cleaner
from utils.log_utils import logger

# 并发加载线程数（IO 密集型，可设较大值）
_IO_WORKERS = 8


class FeatureDataBundle:
    """
    特征计算统一数据容器

    构造参数：
        trade_date          : D 日，格式 yyyy-mm-dd
        target_ts_codes     : 所有候选股代码列表（三个板块合并去重后的完整列表）
        sector_candidate_map: {板块名: 候选股 DataFrame}（含 ts_code 等列）
        top3_sectors        : 当日 Top3 板块名称列表，顺序即板块 ID（1/2/3）
        adapt_score         : 板块轮动分（0-100），由 dataset.py 调用板块热度后传入，
                              避免 FeatureEngine 内重复调用 select_top3_hot_sectors
        load_minute         : 是否加载分钟线（默认 True），不需要 SEI 时可设 False 提速

    预加载属性（构造后即可使用）：
        lookback_dates_5d   : 含 D 日在内的最近 5 个交易日列表
        lookback_dates_20d  : 含 D 日在内的最近 20 个交易日列表
        daily_grouped       : dict，key=(ts_code, trade_date)，value=该行日线数据 dict
                              O(1) 查找，是所有因子计算的核心加速手段
        minute_cache        : dict，key=(ts_code, trade_date)，value=分钟线 DataFrame
        macro_cache         : dict，预加载的宏观数据（涨跌停池/连板/最强板块/指数日线）
    """

    def __init__(
            self,
            trade_date: str,
            target_ts_codes: List[str],
            sector_candidate_map: Dict[str, pd.DataFrame],
            top3_sectors: List[str],
            adapt_score: float = 0.0,
            load_minute: bool = True,
    ):
        self.trade_date = trade_date
        self.target_ts_codes = target_ts_codes
        self.sector_candidate_map = sector_candidate_map
        self.top3_sectors = top3_sectors
        self.adapt_score = adapt_score      # 透传给 SectorHeatFeature.calculate()

        self.lookback_dates_5d: List[str] = []
        self.lookback_dates_20d: List[str] = []
        self.daily_grouped: Dict[tuple, dict] = {}
        self.minute_cache: Dict[tuple, pd.DataFrame] = {}
        self.macro_cache: Dict[str, pd.DataFrame] = {}

        self._load_trade_dates()
        self._load_daily_data()
        self._load_macro_data()
        if load_minute:
            self._load_minute_data()

    def _load_trade_dates(self):
        try:
            d_date = datetime.strptime(self.trade_date, "%Y-%m-%d")
            start_5d = (d_date - timedelta(days=20)).strftime("%Y-%m-%d")
            self.lookback_dates_5d = get_trade_dates(start_5d, self.trade_date)[-5:]
            start_20d = (d_date - timedelta(days=40)).strftime("%Y-%m-%d")
            self.lookback_dates_20d = get_trade_dates(start_20d, self.trade_date)[-20:]
            logger.info(f"[DataBundle] {self.trade_date} 交易日加载完成 | 5日: {self.lookback_dates_5d}")
        except Exception as e:
            logger.error(f"[DataBundle] 交易日加载失败：{e}")
            raise

    def _load_daily_data(self):
        """批量加载日线（仅查候选股，多线程并发拉取各日期数据）"""
        try:
            all_dates = list(set(self.lookback_dates_5d + self.lookback_dates_20d))

            def _fetch_one(date):
                df = get_daily_kline_data(trade_date=date, ts_code_list=self.target_ts_codes)
                if not df.empty:
                    df["trade_date"] = df["trade_date"].astype(str)
                return df

            frames = []
            with ThreadPoolExecutor(max_workers=_IO_WORKERS) as pool:
                futures = {pool.submit(_fetch_one, d): d for d in all_dates}
                for fut in as_completed(futures):
                    df = fut.result()
                    if not df.empty:
                        frames.append(df)

            if frames:
                all_df = pd.concat(frames, ignore_index=True)
                self.daily_grouped = (
                    all_df.groupby(["ts_code", "trade_date"]).first().to_dict(orient="index")
                )
            logger.info(f"[DataBundle] 日线加载完成 | 日期数:{len(all_dates)} | 记录数:{len(self.daily_grouped)}")
        except Exception as e:
            logger.error(f"[DataBundle] 日线数据加载失败：{e}")
            raise

    def _load_macro_data(self):
        """
        预加载 D 日市场宏观数据。
        访问链路：DB → API（DB 无数据时自动通过 cleaner 补拉并写入 DB，下次直接走 DB）
        limit_list / limit_step / limit_cpt / index_daily 均有 API 兜底；
        market_vol 来自 kline_day 聚合，依赖 kline_day 已落库，无单独 API。
        """
        try:
            td     = self.trade_date
            td_fmt = td.replace("-", "")     # YYYYMMDD，data_cleaner / data_fetcher 格式

            # ── 涨跌停池（合并补拉，一次接口同时覆盖两张池）──────────────────
            limit_up_df   = get_limit_list_ths(td, limit_type="涨停池")
            limit_down_df = get_limit_list_ths(td, limit_type="跌停池")
            if limit_up_df.empty and limit_down_df.empty:
                logger.info(f"[DataBundle] {td} 涨跌停池 DB无数据，接口补拉入库...")
                try:
                    data_cleaner.clean_and_insert_limit_list_ths(trade_date=td_fmt)
                    limit_up_df   = get_limit_list_ths(td, limit_type="涨停池")
                    limit_down_df = get_limit_list_ths(td, limit_type="跌停池")
                    logger.info(
                        f"[DataBundle] 涨跌停池补拉完成 | "
                        f"涨停:{len(limit_up_df)} 跌停:{len(limit_down_df)}"
                    )
                except Exception as e:
                    logger.warning(f"[DataBundle] 涨跌停池接口补拉失败（本次用空数据）：{e}")
            self.macro_cache["limit_up_df"]   = limit_up_df
            self.macro_cache["limit_down_df"] = limit_down_df

            # ── 连板天梯 ──────────────────────────────────────────────────────
            limit_step_df = get_limit_step(td)
            if limit_step_df.empty:
                logger.info(f"[DataBundle] {td} 连板天梯 DB无数据，接口补拉入库...")
                try:
                    data_cleaner.clean_and_insert_limit_step(trade_date=td_fmt)
                    limit_step_df = get_limit_step(td)
                    logger.info(f"[DataBundle] 连板天梯补拉完成 | {len(limit_step_df)} 行")
                except Exception as e:
                    logger.warning(f"[DataBundle] 连板天梯接口补拉失败（本次用空数据）：{e}")
            self.macro_cache["limit_step_df"] = limit_step_df

            # ── 最强板块 ──────────────────────────────────────────────────────
            limit_cpt_df = get_limit_cpt_list(td)
            if limit_cpt_df.empty:
                logger.info(f"[DataBundle] {td} 最强板块 DB无数据，接口补拉入库...")
                try:
                    data_cleaner.clean_and_insert_limit_cpt_list(trade_date=td_fmt)
                    limit_cpt_df = get_limit_cpt_list(td)
                    logger.info(f"[DataBundle] 最强板块补拉完成 | {len(limit_cpt_df)} 行")
                except Exception as e:
                    logger.warning(f"[DataBundle] 最强板块接口补拉失败（本次用空数据）：{e}")
            self.macro_cache["limit_cpt_df"] = limit_cpt_df

            # ── 指数日线 ──────────────────────────────────────────────────────
            index_codes = ["000001.SH", "399001.SZ", "399006.SZ"]
            index_df    = get_index_daily(td, ts_code_list=index_codes)
            if index_df.empty:
                logger.info(f"[DataBundle] {td} 指数日线 DB无数据，接口补拉入库...")
                try:
                    for code in index_codes:
                        data_cleaner.clean_and_insert_index_daily(
                            ts_code=code, start_date=td_fmt, end_date=td_fmt
                        )
                    index_df = get_index_daily(td, ts_code_list=index_codes)
                    logger.info(f"[DataBundle] 指数日线补拉完成 | {len(index_df)} 行")
                except Exception as e:
                    logger.warning(f"[DataBundle] 指数日线接口补拉失败（本次用空数据）：{e}")
            self.macro_cache["index_df"] = index_df

            # ── 全市场成交量（kline_day 聚合，依赖 kline_day 已落库）──────────
            self.macro_cache["market_vol_df"] = get_market_total_volume(self.lookback_dates_5d)

            # ── 5日历史涨停数量 / 最大连板数（d1-d4，用于派生趋势因子）────────
            # d0 已有完整数据，直接从已加载结果读取；d1-d4 并发查询历史
            _d0_up_count   = len(limit_up_df)
            _d0_max_consec = 0
            if not limit_step_df.empty and "nums" in limit_step_df.columns:
                _nums = pd.to_numeric(limit_step_df["nums"], errors="coerce").dropna()
                _d0_max_consec = int(_nums.max()) if len(_nums) > 0 else 0

            limit_up_counts_5d: dict = {td: _d0_up_count}
            consec_max_5d:      dict = {td: _d0_max_consec}

            hist_dates = self.lookback_dates_5d[:-1]   # d1~d4（不含d0）

            def _fetch_hist_macro(date):
                up_df_h   = get_limit_list_ths(date, limit_type="涨停池")
                step_df_h = get_limit_step(date)
                # 同 d0 一样：DB 无数据时尝试接口补拉，保证历史趋势因子有效
                if up_df_h.empty:
                    try:
                        data_cleaner.clean_and_insert_limit_list_ths(
                            trade_date=date.replace("-", "")
                        )
                        up_df_h = get_limit_list_ths(date, limit_type="涨停池")
                    except Exception:
                        pass
                if step_df_h.empty:
                    try:
                        data_cleaner.clean_and_insert_limit_step(
                            trade_date=date.replace("-", "")
                        )
                        step_df_h = get_limit_step(date)
                    except Exception:
                        pass
                up_cnt = len(up_df_h)
                max_c  = 0
                if not step_df_h.empty and "nums" in step_df_h.columns:
                    _n = pd.to_numeric(step_df_h["nums"], errors="coerce").dropna()
                    max_c = int(_n.max()) if len(_n) > 0 else 0
                return date, up_cnt, max_c

            if hist_dates:
                with ThreadPoolExecutor(max_workers=min(4, len(hist_dates))) as pool:
                    for _date, _up_cnt, _max_c in pool.map(_fetch_hist_macro, hist_dates):
                        limit_up_counts_5d[_date] = _up_cnt
                        consec_max_5d[_date]      = _max_c

            self.macro_cache["limit_up_counts_5d"] = limit_up_counts_5d
            self.macro_cache["consec_max_5d"]      = consec_max_5d

            logger.debug(
                f"[DataBundle] 宏观数据加载完成 | "
                f"涨停:{len(self.macro_cache['limit_up_df'])} "
                f"跌停:{len(self.macro_cache['limit_down_df'])} "
                f"连板:{len(self.macro_cache['limit_step_df'])} "
                f"板块:{len(self.macro_cache['limit_cpt_df'])} "
                f"指数:{len(self.macro_cache['index_df'])} "
                f"市场成交量:{len(self.macro_cache['market_vol_df'])}"
            )
        except Exception as e:
            logger.warning(f"[DataBundle] 宏观数据加载异常（非致命）：{str(e)[:120]}")

    def _load_minute_data(self):
        """
        加载候选股近 5 日分钟线（多线程并发，HDI/SEI 因子必需）。

        缺失原因说明
        -----------
        · 当天交易刚结束时触发：Tushare 分钟线通常有 30–60 分钟延迟，
          若在收盘后立即运行（如 15:05），数据可能尚未写入 Tushare。
        · 分钟线数据会先写入本地 DB 缓存，下次访问直接读 DB，
          因此在数据完全入库后重跑结果会与首次运行有差异。
        缺失时后果：HDI/SEI 等依赖分钟线的特征值为 0，影响模型概率预测，
        可能导致信号有无和数量与完整数据时不同。
        """
        try:
            tasks = [
                (ts_code, date)
                for ts_code in self.target_ts_codes
                for date in self.lookback_dates_5d
            ]
            if not tasks:
                return

            def _fetch_one(pair):
                ts_code, date = pair
                return pair, data_cleaner.get_kline_min_by_stock_date(ts_code, date)

            with ThreadPoolExecutor(max_workers=_IO_WORKERS) as pool:
                for (ts_code, date), df in pool.map(_fetch_one, tasks):
                    self.minute_cache[(ts_code, date)] = df

            # 统计缺失情况，当天数据缺失时发出 WARNING
            total       = len(tasks)
            empty_pairs = [(c, d) for (c, d), df in self.minute_cache.items() if df.empty]
            empty_count = len(empty_pairs)
            # 只统计当天（trade_date）的缺失，历史日期缺失影响较小
            today_empty = [(c, d) for c, d in empty_pairs if d == self.trade_date]

            if today_empty:
                codes_str = ",".join(c for c, _ in today_empty[:10])
                suffix    = f"...等{len(today_empty)}只" if len(today_empty) > 10 else f"{len(today_empty)}只"
                logger.warning(
                    f"[DataBundle] ⚠ 分钟线当日数据缺失 | {self.trade_date} 共 {suffix} 无分钟线数据"
                    f"（{codes_str}{'...' if len(today_empty) > 10 else ''}）"
                    f"| 可能原因：收盘后数据尚未入库（Tushare 约延迟 30-60 分钟）"
                    f"| HDI/SEI 特征将为 0，当前信号仅供参考，建议 15:30+ 或次日凌晨再跑以获得完整结果"
                )
                # 将当日缺失信息存入 macro_cache 供 runner 层展示诊断
                self.macro_cache["_minute_today_missing"] = len(today_empty)
                self.macro_cache["_minute_today_total"]   = len(self.target_ts_codes)
            else:
                self.macro_cache["_minute_today_missing"] = 0
                self.macro_cache["_minute_today_total"]   = len(self.target_ts_codes)

            logger.info(
                f"[DataBundle] 分钟线加载完成 | 总对数:{total} "
                f"| 有数据:{total - empty_count} | 无数据:{empty_count}"
                f"（其中当日缺失:{len(today_empty)}）"
            )
        except Exception as e:
            logger.warning(f"[DataBundle] 分钟线加载异常（非致命）：{str(e)[:120]}")