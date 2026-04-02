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
    get_st_set, get_stock_list_date_map,
    get_hp_cycle_slice_avg_gain, get_active_stock_stats,
    get_kline_day_range, get_market_breadth_liquidity_stats,
)
from data.data_cleaner import data_cleaner
from utils.log_utils import logger
from features.utils.high_position_utils import (
    compute_high_pos_selection, HP_BASE_PCT,
)

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

    # ── 各数据加载方法对应的最小因子模块集 ─────────────────────────────────────
    # None（required_modules=None）= 训练路径，加载全部数据
    _MODULES_NEEDING_MACRO       = frozenset({
        "sector_heat", "market_macro", "individual", "agent_return_stats", "limit_emotion",
    })
    _MODULES_NEEDING_HP_EXT      = frozenset({"hp_stage", "hp_style", "hp_cycle", "active_stats"})
    _MODULES_NEEDING_LIMIT_TOUCH = frozenset({"limit_emotion"})
    _MODULES_NEEDING_MINUTE      = frozenset({"sector_stock"})

    def __init__(
            self,
            trade_date: str,
            target_ts_codes: List[str],
            sector_candidate_map: Dict[str, pd.DataFrame],
            top3_sectors: List[str],
            adapt_score: float = 0.0,
            load_minute: bool = True,
            required_modules=None,   # None=全量（训练路径）；List[str]=推断路径按需裁剪
    ):
        self.trade_date = trade_date
        self.target_ts_codes = target_ts_codes
        self.sector_candidate_map = sector_candidate_map
        self.top3_sectors = top3_sectors
        self.adapt_score = adapt_score      # 透传给 SectorHeatFeature.calculate()

        self.lookback_dates_5d: List[str] = []
        self.lookback_dates_20d: List[str] = []
        self.lookback_dates_22d: List[str] = []   # D-21 ~ D0（高位股20日涨幅计算所需）
        self.lookback_dates_60d: List[str] = []   # D-59 ~ D0（trend_pullback等60日窗口因子用）
        self.daily_grouped: Dict[tuple, dict] = {}
        self.minute_cache: Dict[tuple, pd.DataFrame] = {}
        self.macro_cache: Dict[str, pd.DataFrame] = {}
        self.hp_ext_cache: Dict = {}               # 高位股情绪 + 市场广度因子扩展数据

        # required_modules=None → 全量模式（训练），不裁剪
        _req = frozenset(required_modules) if required_modules is not None else None

        self._load_trade_dates()
        self._load_daily_data()

        if _req is None or _req & self._MODULES_NEEDING_MACRO:
            self._load_macro_data()

        if _req is None or _req & self._MODULES_NEEDING_HP_EXT:
            self._load_hp_ext_cache()

        if _req is None or _req & self._MODULES_NEEDING_LIMIT_TOUCH:
            self._load_limit_touch_data()   # 依赖 hp_ext_cache["st_set"]（缺失时安全降级）

        if load_minute and (_req is None or _req & self._MODULES_NEEDING_MINUTE):
            self._load_minute_data()

    def _load_trade_dates(self):
        try:
            d_date = datetime.strptime(self.trade_date, "%Y-%m-%d")
            start_5d = (d_date - timedelta(days=20)).strftime("%Y-%m-%d")
            self.lookback_dates_5d = get_trade_dates(start_5d, self.trade_date)[-5:]
            start_20d = (d_date - timedelta(days=40)).strftime("%Y-%m-%d")
            self.lookback_dates_20d = get_trade_dates(start_20d, self.trade_date)[-20:]
            # lookback_dates_22d：含 D-21 到 D0 共 22 个交易日，用于高位股 20 日涨幅基准
            start_22d = (d_date - timedelta(days=50)).strftime("%Y-%m-%d")
            self.lookback_dates_22d = get_trade_dates(start_22d, self.trade_date)[-22:]
            # lookback_dates_60d：含 D-59 到 D0 共 60 个交易日，用于 trend_pullback 等60日窗口因子
            start_60d = (d_date - timedelta(days=120)).strftime("%Y-%m-%d")
            self.lookback_dates_60d = get_trade_dates(start_60d, self.trade_date)[-60:]
            logger.info(f"[DataBundle] {self.trade_date} 交易日加载完成 | 5日: {self.lookback_dates_5d}")
        except Exception as e:
            logger.error(f"[DataBundle] 交易日加载失败：{e}")
            raise

    def _load_daily_data(self):
        """批量加载日线（仅查候选股，多线程并发拉取各日期数据）"""
        try:
            all_dates = list(set(self.lookback_dates_5d + self.lookback_dates_20d + self.lookback_dates_60d))

            def _fetch_one(date):
                df = get_daily_kline_data(trade_date=date, ts_code_list=self.target_ts_codes)
                if not df.empty and "trade_date" in df.columns:
                    td = df["trade_date"].astype(str).str.replace("-", "", regex=False)
                    df["trade_date"] = td.str.slice(0, 4) + "-" + td.str.slice(4, 6) + "-" + td.str.slice(6, 8)
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

            # ── 炸板池（limit_list_ths API 一次覆盖全类型，补拉已在上方完成）──
            zhaban_df = get_limit_list_ths(td, limit_type="炸板池")
            self.macro_cache["zhaban_df"] = zhaban_df

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

            limit_up_counts_5d:  dict = {td: _d0_up_count}
            consec_max_5d:       dict = {td: _d0_max_consec}
            zhaban_counts_5d:    dict = {td: len(zhaban_df)}
            # D0 涨停+炸板 codes（供 _load_limit_touch_data 计算分钟线触板成交额）
            _d0_touch_codes: list = []
            for _df in [limit_up_df, zhaban_df]:
                if not _df.empty and "ts_code" in _df.columns:
                    _d0_touch_codes.extend(_df["ts_code"].tolist())
            limit_touch_codes_5d: dict = {td: list(set(_d0_touch_codes))}

            hist_dates = self.lookback_dates_5d[:-1]   # d1~d4（不含d0）

            def _fetch_hist_macro(date):
                up_df_h     = get_limit_list_ths(date, limit_type="涨停池")
                step_df_h   = get_limit_step(date)
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
                # 炸板池（补拉已覆盖，直接查）
                zhaban_df_h = get_limit_list_ths(date, limit_type="炸板池")

                up_cnt     = len(up_df_h)
                zhaban_cnt = len(zhaban_df_h)
                max_c      = 0
                if not step_df_h.empty and "nums" in step_df_h.columns:
                    _n = pd.to_numeric(step_df_h["nums"], errors="coerce").dropna()
                    max_c = int(_n.max()) if len(_n) > 0 else 0

                # 历史日涨停+炸板 codes（供 _load_limit_touch_data 算分钟线触板成交额）
                _h_codes: list = []
                for _hdf in [up_df_h, zhaban_df_h]:
                    if not _hdf.empty and "ts_code" in _hdf.columns:
                        _h_codes.extend(_hdf["ts_code"].tolist())

                return date, up_cnt, max_c, zhaban_cnt, list(set(_h_codes))

            if hist_dates:
                with ThreadPoolExecutor(max_workers=min(4, len(hist_dates))) as pool:
                    for _date, _up_cnt, _max_c, _zhaban_cnt, _codes in pool.map(
                        _fetch_hist_macro, hist_dates
                    ):
                        limit_up_counts_5d[_date]  = _up_cnt
                        consec_max_5d[_date]        = _max_c
                        zhaban_counts_5d[_date]     = _zhaban_cnt
                        limit_touch_codes_5d[_date] = _codes

            self.macro_cache["limit_up_counts_5d"]  = limit_up_counts_5d
            self.macro_cache["consec_max_5d"]        = consec_max_5d
            self.macro_cache["zhaban_counts_5d"]     = zhaban_counts_5d
            self.macro_cache["limit_touch_codes_5d"] = limit_touch_codes_5d

            # ── 20日上证指数涨跌幅（个股强势因子 factor 11 用）─────────────
            # 批量查询全部 20d 日期，单次 SQL，存为 {date: pct_chg} dict
            try:
                hist_sh_pct_chg: dict = {}
                # d0 已在 index_df 中
                d0_idx_df = self.macro_cache.get("index_df", pd.DataFrame())
                if not d0_idx_df.empty and "ts_code" in d0_idx_df.columns:
                    d0_row = d0_idx_df[d0_idx_df["ts_code"] == "000001.SH"]
                    if not d0_row.empty:
                        hist_sh_pct_chg[td] = float(d0_row.iloc[0].get("pct_chg", 0.0) or 0.0)
                # 批量查询其余历史日期
                hist_20d_dates = [d for d in self.lookback_dates_20d if d != td]
                if hist_20d_dates:
                    dates_fmt_list = [d.replace("-", "") for d in hist_20d_dates]
                    placeholders   = ", ".join(["%s"] * len(dates_fmt_list))
                    from utils.db_utils import db as _db
                    hist_idx_rows = _db.query(
                        f"SELECT trade_date, pct_chg FROM index_daily "
                        f"WHERE ts_code = '000001.SH' AND trade_date IN ({placeholders})",
                        params=tuple(dates_fmt_list),
                    ) or []
                    for r in hist_idx_rows:
                        raw_td = str(r.get("trade_date", "")).replace("-", "")
                        # 统一转成 yyyy-mm-dd 匹配 lookback_dates_20d 格式
                        if len(raw_td) == 8:
                            std_td = f"{raw_td[:4]}-{raw_td[4:6]}-{raw_td[6:]}"
                        else:
                            std_td = str(r.get("trade_date", ""))
                        hist_sh_pct_chg[std_td] = float(r.get("pct_chg", 0.0) or 0.0)
                self.macro_cache["hist_sh_pct_chg"] = hist_sh_pct_chg
                logger.debug(f"[DataBundle] 20日上证指数涨跌幅加载完成 | {len(hist_sh_pct_chg)} 条")
            except Exception as e:
                logger.warning(f"[DataBundle] 20日指数历史加载失败（非致命）：{e}")
                self.macro_cache["hist_sh_pct_chg"] = {}

            # ── 5日全市场广度/流动性统计（宏观扩展因子用）──────────
            try:
                self.macro_cache["breadth_liquidity_5d"] = get_market_breadth_liquidity_stats(self.lookback_dates_5d)
            except Exception as e:
                logger.warning(f"[DataBundle] 5日广度/流动性统计加载失败（非致命）：{e}")
                self.macro_cache["breadth_liquidity_5d"] = {}

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

    def _load_hp_ext_cache(self):
        """
        预加载高位股情绪 + 市场广度因子所需的扩展数据。

        分两轮 IO：
          Round 1（并发）：全市场关键日期日线、ST集合、上市日期、120日切片统计、活跃股统计
          Round 2（顺序）：识别高位股基础池后，针对性拉取其近5日日线（用于 MA5 乖离率计算）

        hp_ext_cache 结构：
          market_all_d0          : D0 全市场日线 DataFrame
          market_all_d10         : D-10 全市场日线 DataFrame（10日涨幅基准）
          market_all_d21         : D-21 全市场日线 DataFrame（20日涨幅基准/高位股定义）
          st_set                 : 当日 ST 股票代码集合
          list_date_map          : {ts_code: list_date(YYYYMMDD)}
          hp_base_pool           : 高位股基础池 DataFrame（全市场前1%，约50只）
          hp_base_pool_recent5d  : 基础池近5日日线（用于 MA5 + 量比计算）
          hp_cycle_slices        : 12个切片的高位股平均涨幅 List[float]（索引0=D0切片）
          active_stats           : 活跃股广度统计 Dict
          hp_high_pos_minute_5d  : 高位股近5日分钟线缓存 Dict[(ts_code, trade_date)] -> DataFrame
          key_dates              : 各关键日期 Dict（d0/d1/d4/d10/d21/d60）
        """
        try:
            td = self.trade_date

            # ── 校验关键日期 ─────────────────────────────────────────────────
            if len(self.lookback_dates_5d) < 5 or len(self.lookback_dates_22d) < 22:
                logger.warning(f"[DataBundle] hp_ext_cache: 历史日期不足，跳过高位股因子")
                self.hp_ext_cache = {}
                return

            d0_date  = self.lookback_dates_5d[-1]   # D0
            d1_date  = self.lookback_dates_5d[-2]   # D-1
            d4_date  = self.lookback_dates_5d[0]    # D-4（5日均量区间起始）
            d10_date = self.lookback_dates_20d[-11] if len(self.lookback_dates_20d) >= 11 else None
            d21_date = self.lookback_dates_22d[0]   # D-21

            if not d10_date:
                logger.warning(f"[DataBundle] hp_ext_cache: lookback_dates_20d 不足11天，跳过")
                self.hp_ext_cache = {}
                return

            # D-60：用于60日新高/低区间（额外获取扩展历史）
            d_date   = datetime.strptime(td, "%Y-%m-%d")
            start_ext = (d_date - timedelta(days=280)).strftime("%Y-%m-%d")
            trade_dates_ext = get_trade_dates(start_ext, td)
            d60_date = trade_dates_ext[-61] if len(trade_dates_ext) >= 61 else None

            self.hp_ext_cache["key_dates"] = {
                "d0": d0_date, "d1": d1_date, "d4": d4_date,
                "d10": d10_date, "d21": d21_date, "d60": d60_date,
            }

            # ── 构建120日切片任务（12次 SQL 聚合，并发执行）────────────────
            slice_tasks: List[tuple] = []   # (k, slice_end_date, d21_for_slice)
            if len(trade_dates_ext) >= 132:
                for k in range(12):
                    s_idx = -(k * 10 + 1)      # 切片末日索引（-1=D0, -11=D-10, ...）
                    d_idx = s_idx - 21           # 对应 D-21 索引
                    if abs(d_idx) > len(trade_dates_ext):
                        break
                    slice_tasks.append((k, trade_dates_ext[s_idx], trade_dates_ext[d_idx]))

            # ── Round 1：全部并发提交 ─────────────────────────────────────
            with ThreadPoolExecutor(max_workers=_IO_WORKERS) as pool:
                fut_d0     = pool.submit(get_daily_kline_data, d0_date)
                fut_d10    = pool.submit(get_daily_kline_data, d10_date)
                fut_d21    = pool.submit(get_daily_kline_data, d21_date)
                fut_st     = pool.submit(get_st_set,               td)
                fut_ldmap  = pool.submit(get_stock_list_date_map)

                # 活跃股统计（需要 d60 和 d1）
                if d60_date:
                    fut_liquid = pool.submit(
                        get_active_stock_stats, d0_date, d4_date, d60_date, d1_date
                    )
                else:
                    fut_liquid = None

                # 120日切片统计（并发提交12个 SQL 聚合）
                slice_futures: Dict = {}
                for k, s_date, d21_s in slice_tasks:
                    f = pool.submit(get_hp_cycle_slice_avg_gain, s_date, d21_s)
                    slice_futures[f] = k

                # 收集结果（在 with 块内等待所有任务完成）
                # 注意：DataFrame 不能用 or 运算符（会触发 "truth value is ambiguous"），
                #       改用显式 None 判断
                def _df_or_empty(r):
                    return r if isinstance(r, pd.DataFrame) else pd.DataFrame()

                market_d0    = _df_or_empty(fut_d0.result())
                market_d10   = _df_or_empty(fut_d10.result())
                market_d21   = _df_or_empty(fut_d21.result())
                st_set       = fut_st.result()     or set()
                ldmap        = fut_ldmap.result()  or {}
                active_stats = fut_liquid.result() if fut_liquid else {}

                hp_cycle_slices: List[float] = [0.0] * 12
                for fut, k in slice_futures.items():
                    try:
                        hp_cycle_slices[k] = fut.result() or 0.0
                    except Exception as e:
                        logger.warning(f"[DataBundle] hp_cycle 切片{k}查询失败：{e}")

            self.hp_ext_cache.update({
                "market_all_d0":    market_d0,
                "market_all_d10":   market_d10,
                "market_all_d21":   market_d21,
                "st_set":           st_set,
                "list_date_map":    ldmap,
                "active_stats":     active_stats,
                "hp_cycle_slices":  [g for g in hp_cycle_slices if g is not None],
                "d21_date_str":     d21_date.replace("-", ""),
            })

            # ── 识别高位股基础池（基于 Round 1 结果，纯内存计算）────────────
            base_pool, high_pos = compute_high_pos_selection(
                market_d0, market_d21, st_set, ldmap,
                d21_date.replace("-", ""),
            )
            self.hp_ext_cache["hp_base_pool"]  = base_pool
            self.hp_ext_cache["hp_high_pos"]   = high_pos

            # ── Round 2：拉取基础池近5日日线（约50只 × 5天，单次批量查询）──
            if not base_pool.empty:
                base_codes = base_pool["ts_code"].tolist()
                try:
                    recent5d = get_kline_day_range(base_codes, d4_date, d0_date)
                    self.hp_ext_cache["hp_base_pool_recent5d"] = recent5d
                except Exception as e:
                    logger.warning(f"[DataBundle] hp_ext_cache Round2近5日数据失败：{e}")
                    self.hp_ext_cache["hp_base_pool_recent5d"] = pd.DataFrame()
            else:
                self.hp_ext_cache["hp_base_pool_recent5d"] = pd.DataFrame()

            # ── Round 3：拉取高位股近5日分钟线（用于高位股触顶时间阶段因子）──
            hp_minute_cache = {}
            if not high_pos.empty:
                hp_codes = high_pos["ts_code"].tolist()
                for code in hp_codes:
                    for minute_date in self.lookback_dates_5d:
                        try:
                            hp_minute_cache[(code, minute_date)] = data_cleaner.get_kline_min_by_stock_date(
                                code, minute_date
                            )
                        except Exception as e:
                            logger.warning(f"[DataBundle] hp_ext_cache 高位股分钟线失败 | {code} {minute_date} | {e}")
                            hp_minute_cache[(code, minute_date)] = pd.DataFrame()
            self.hp_ext_cache["hp_high_pos_minute_5d"] = hp_minute_cache

            logger.info(
                f"[DataBundle] hp_ext_cache加载完成 | "
                f"市场D0:{len(market_d0)}行 "
                f"基础池:{len(base_pool)}只 高位股:{len(high_pos)}只 "
                f"切片统计:{len(self.hp_ext_cache['hp_cycle_slices'])}个 "
                f"活跃股:{active_stats.get('active_total', 0)}只"
            )

        except Exception as e:
            logger.warning(f"[DataBundle] hp_ext_cache加载异常（非致命，高位股因子将返回中性值）：{str(e)[:200]}")
            self.hp_ext_cache = {}

    def _load_limit_touch_data(self):
        """
        计算 D0~D4 每日触板分钟K线成交额（涨停池 + 炸板池，剔除ST，首次触板那根K线 amount 之和）。

        算法：
          1. 从 macro_cache["limit_touch_codes_5d"] 取每日涨停+炸板 codes
          2. 剔除 ST（hp_ext_cache["st_set"]）
          3. 批量查对应日期的日线 high（作为涨停价参照：触板股 daily_high = 涨停价）
          4. 并发加载所有 (ts_code, date) 对的分钟线
          5. 每只股票找首次触板（high >= daily_high - 0.01）那根K线，累加 amount

        结果：macro_cache["limit_touch_amount_5d"] = {trade_date: amount_千元}
        依赖：hp_ext_cache["st_set"]，需在 _load_hp_ext_cache 之后调用。
        所有异常均为非致命，失败时对应日期写入 0.0（中性值）。
        """
        try:
            st_set           = self.hp_ext_cache.get("st_set", set())
            touch_codes_5d   = self.macro_cache.get("limit_touch_codes_5d", {})
            lookback_5d      = self.lookback_dates_5d

            if not touch_codes_5d or not lookback_5d:
                self.macro_cache["limit_touch_amount_5d"] = {d: 0.0 for d in lookback_5d}
                return

            # ── Step1：收集所有 (date, ts_code) 对，剔除ST ─────────────────
            tasks: list = []   # [(date, ts_code), ...]
            for date in lookback_5d:
                codes = touch_codes_5d.get(date, [])
                for c in codes:
                    if c not in st_set:
                        tasks.append((date, c))

            if not tasks:
                self.macro_cache["limit_touch_amount_5d"] = {d: 0.0 for d in lookback_5d}
                return

            # ── Step2：批量查日线 high（所有日期，一次 SQL per date）──────────
            # daily_high_map[(ts_code, date_no_dash)] = high
            daily_high_map: dict = {}
            all_dates_needed = list(set(d for d, _ in tasks))
            all_codes_needed = list(set(c for _, c in tasks))
            # lookback_5d 跨度很短（5天），一次 get_kline_day_range 覆盖全部
            try:
                _daily_range_df = get_kline_day_range(
                    all_codes_needed,
                    min(all_dates_needed),
                    max(all_dates_needed),
                )
                if not _daily_range_df.empty and "ts_code" in _daily_range_df.columns:
                    for _, _r in _daily_range_df.iterrows():
                        _td = str(_r.get("trade_date", "")).replace("-", "")
                        daily_high_map[(_r["ts_code"], _td)] = float(_r.get("high", 0) or 0)
            except Exception as _e:
                logger.warning(f"[DataBundle] 触板股日线 high 批量查询失败：{_e}")

            # ── Step3：并发加载分钟线 ─────────────────────────────────────────
            def _fetch_min_pair(pair):
                date, ts_code = pair
                return pair, data_cleaner.get_kline_min_by_stock_date(ts_code, date)

            minute_cache_local: dict = {}
            with ThreadPoolExecutor(max_workers=_IO_WORKERS) as pool:
                for pair, min_df in pool.map(_fetch_min_pair, tasks):
                    minute_cache_local[pair] = min_df

            # ── Step4：按日期累加触板成交额 ───────────────────────────────────
            touch_amount_5d: dict = {d: 0.0 for d in lookback_5d}
            for date, ts_code in tasks:
                td_no_dash  = date.replace("-", "")
                limit_price = daily_high_map.get((ts_code, td_no_dash), 0.0)
                if limit_price <= 0:
                    continue
                min_df = minute_cache_local.get((date, ts_code))
                if min_df is None or min_df.empty or "high" not in min_df.columns:
                    continue
                min_df = min_df.sort_values("trade_time").reset_index(drop=True)
                highs = min_df["high"].astype(float)
                touch_mask = highs >= limit_price - 0.01   # 容差 0.01 元防浮点误差
                if not touch_mask.any():
                    continue
                first_idx = int(touch_mask.idxmax())
                if "amount" in min_df.columns:
                    touch_amount_5d[date] = touch_amount_5d.get(date, 0.0) + float(
                        min_df.loc[first_idx, "amount"] or 0
                    ) / 1000   # stk_mins amount 单位为元，÷1000 转换为千元与日线口径一致

            self.macro_cache["limit_touch_amount_5d"] = touch_amount_5d
            d0_amt = touch_amount_5d.get(self.trade_date, 0.0)
            logger.info(
                f"[DataBundle] 5日触板分钟成交额 | 任务:{len(tasks)}对 "
                f"| D0:{d0_amt / 1e9:.4f}万亿元"
            )
        except Exception as e:
            logger.warning(f"[DataBundle] 触板分钟成交额加载失败（非致命）：{e}")
            self.macro_cache["limit_touch_amount_5d"] = {d: 0.0 for d in self.lookback_dates_5d}

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