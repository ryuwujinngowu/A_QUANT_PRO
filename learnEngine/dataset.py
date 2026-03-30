"""
训练集生成主流程 (dataset.py)
==============================
运行方式：python dataset.py

整体流程（每日原子性处理）：
  1. SectorHeatFeature.select_top3_hot_sectors(date)
     → top3_sectors + adapt_score
  2. 构建板块候选池 sector_candidate_map（过滤ST/北交所/无涨停基因）
  3. FeatureDataBundle(... adapt_score=adapt_score) 统一预加载数据
  4. FeatureEngine.run_single_date(data_bundle) → feature_df（含 adapt_score）
  5. LabelEngine.generate_single_date → label_df
  6. 合并、清洗、追加写入 CSV
  7. ProcessedDatesManager 标记已处理（写入成功后才标记，保证幂等）
"""

import json
import os
import sys
import warnings
from collections import defaultdict
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config import FILTER_BSE_STOCK, FILTER_STAR_BOARD, FILTER_688_BOARD
from data.data_cleaner import data_cleaner
from features import FeatureEngine, FeatureDataBundle
import learnEngine.train_config as _cfg
from learnEngine.label import LabelEngine
from learnEngine.split_spec import write_split_spec
from strategies.sector_heat_strategy import SectorHeatStrategy
from strategies.high_low_switch_ml_strategy import HighLowSwitchMLStrategy
from utils.common_tools import (
    get_stocks_in_sector,
    filter_st_stocks,
    sort_by_recent_gain,
    get_trade_dates,
    get_daily_kline_data,
    has_recent_limit_up_batch,
    calc_limit_up_price,
    ensure_limit_list_ths_data,
)
from utils.log_utils import logger


# ============================================================
# 已处理日期管理（原子性读写，保证断点续跑幂等）
# ============================================================

class ProcessedDatesManager:
    """管理已处理日期的读写，确保原子性"""

    def __init__(self, file_path: str, factor_version: str):
        self.file_path      = file_path
        self.factor_version = factor_version
        self.processed_dates = self._load()

    def _load(self) -> List[str]:
        """加载已处理日期；因子版本不一致则清空（强制重跑）"""
        if not os.path.exists(self.file_path):
            return []
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("factor_version") != self.factor_version:
                    logger.warning("因子版本变更，清空已处理记录（将重新生成全量数据）")
                    return []
                return sorted(data.get("processed_dates", []))
        except Exception as e:
            logger.error(f"加载已处理日期失败: {e}")
            return []

    def is_processed(self, date: str) -> bool:
        return date in self.processed_dates

    def add(self, date: str):
        """添加并立即持久化（写入成功后调用，保证幂等）"""
        if date not in self.processed_dates:
            self.processed_dates.append(date)
            self._save()

    def reset(self):
        """清空已处理记录（如训练集 CSV 被删除，需重新生成时调用）"""
        self.processed_dates = []
        if os.path.exists(self.file_path):
            os.remove(self.file_path)
        logger.warning("已处理日期记录已重置，将重新生成全量数据")

    def _save(self):
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(
                {"factor_version": self.factor_version,
                 "processed_dates": sorted(self.processed_dates)},
                f, ensure_ascii=False, indent=2
            )


# ============================================================
# 数据集清洗器
# ============================================================

class DataSetAssembler:
    """单日数据校验 & 清洗"""

    # 价格为 0 必属异常的核心列（停牌/数据缺失导致，宁可丢行也不污染模型）
    _PRICE_SANITY_COL = "stock_close_d0"

    @staticmethod
    def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()

        required = ["sample_id", "strategy_id", "stock_code", "trade_date", "label1", "label2"]
        missing  = [c for c in required if c not in df.columns]
        if missing:
            logger.error(f"[DataSetAssembler] 缺失核心列: {missing}")
            return pd.DataFrame()

        # T6: 使用 sample_id 去重（允许同一股票同一日期出现在不同策略/板块）
        df = df.drop_duplicates(subset=["sample_id"])
        # label1 是唯一必需列（D+1 停牌时 LabelEngine 跳过，dropna 正常）
        # label2 依赖 D+2 数据，最新日期可能 D+2 尚未收盘，保留 NaN 行而非丢弃
        # 训练时按 TARGET_LABEL 各自 dropna，在此只保证 label1 非空
        df = df.dropna(subset=["label1"])

        # ── 丢弃 D 日收盘价缺失或为零的行 ───────────────────────────────
        # 该列为零必然是停牌 / 数据入库异常，宁可损失训练样本，
        # 也不能让"收盘价=0"这种明显错误值进入特征矩阵污染模型。
        if DataSetAssembler._PRICE_SANITY_COL in df.columns:
            bad = df[DataSetAssembler._PRICE_SANITY_COL].isna() | \
                  (df[DataSetAssembler._PRICE_SANITY_COL] <= 0)
            bad_count = int(bad.sum())
            if bad_count:
                logger.warning(
                    f"[DataSetAssembler] 丢弃 {DataSetAssembler._PRICE_SANITY_COL} "
                    f"异常行（停牌/数据缺失）: {bad_count} 行"
                )
                df = df[~bad]

        # ── 特征 NaN 显式中性值填充 ────────────────────────────────────────
        # 注意：label1/label2 已在上方 dropna 保证，此处 fillna 不会影响标签
        #
        # 语义说明：不同类型的因子有不同的"中性"含义，统一填 0 会扭曲语义：
        #   pos_20d = 0 → "处在20日最低价"（强烈看空），正确中性应为 0.5（区间中点）
        #   stock_cpr = 0 → "收于日内最低价"（强烈看空），正确中性应为 0.5
        #   stock_trend_r2 = 0 → "纯震荡无趋势"，正确中性应为 0.5（无信息）
        #   boll_pct = 0 → "触及布林下轨"，正确中性应为 0.5（区间中点）
        #
        # 规则：先按列名精确/模式匹配填充有语义的中性值，其余列统一填 0
        _exact_neutral = {
            "pos_20d":           0.5,   # 20日价格区间中点
            "pos_5d":            0.5,   # 5日价格区间中点
            "from_high_20d":     0.1,   # 距最高点跌幅：保守填10%，避免偏向0（当前就是最高点）
            "boll_pct":          0.5,   # 布林带区间中点
        }
        _pattern_neutral = {
            "_cpr_":       0.5,   # 收盘位置比：区间中点
            "_trend_r2_":  0.5,   # 分钟线趋势R²：中性（无趋势信息）
        }
        fill_map = {}
        for col in df.columns:
            if col in _exact_neutral:
                fill_map[col] = _exact_neutral[col]
            else:
                for pat, val in _pattern_neutral.items():
                    if pat in col:
                        fill_map[col] = val
                        break
        if fill_map:
            df = df.fillna(fill_map)
        df = df.fillna(0)   # 其余列（pct_chg/bias/sei/hdi/计数类等）0 即为正确中性值
        return df.reset_index(drop=True)


# ============================================================
# 私有辅助函数
# ============================================================

def _filter_ts_code_by_board(ts_code_list: List[str]) -> List[str]:
    """过滤北交所 / 科创板 / 创业板"""
    result = []
    for ts in ts_code_list:
        if not ts:
            continue
        if FILTER_BSE_STOCK and (ts.endswith(".BJ") or ts.startswith(("83", "87", "88"))):
            continue
        if FILTER_688_BOARD and ts.startswith("688"):
            continue
        if FILTER_STAR_BOARD and ts.startswith(("300", "301", "302")) and ts.endswith(".SZ"):
            continue
        result.append(ts)
    return result


def _check_stock_has_limit_up(
        ts_code_list: List[str], end_date: str, day_count: int = 10
) -> Dict[str, bool]:
    """批量判断近 N 日是否有涨停（中性值兜底：异常时返回 False，不引入虚假正样本）"""
    if not ts_code_list or day_count <= 0 or not end_date:
        return {ts: False for ts in ts_code_list}

    try:
        if len(end_date) == 8 and end_date.isdigit():
            end_dt = datetime.strptime(end_date, "%Y%m%d")
        else:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        end_fmt = end_dt.strftime("%Y-%m-%d")
    except ValueError as e:
        logger.error(f"日期解析失败: {end_date} | {e}")
        return {ts: False for ts in ts_code_list}

    try:
        pre_end  = (end_dt - timedelta(days=1)).strftime("%Y-%m-%d")
        start_dt = (end_dt - timedelta(days=60)).strftime("%Y-%m-%d")
        dates    = get_trade_dates(start_dt, pre_end)[-day_count:]
        if len(dates) < day_count:
            logger.warning(f"回溯交易日不足 {day_count} 个，返回全 False（中性值）")
            return {ts: False for ts in ts_code_list}
    except Exception as e:
        logger.error(f"获取交易日失败: {e}")
        return {ts: False for ts in ts_code_list}

    if not dates:
        logger.warning("回溯交易日为空，返回全 False（中性值）")
        return {ts: False for ts in ts_code_list}

    # 确保 limit_list_ths 有最新数据（DB→API→DB 链路）
    ensure_limit_list_ths_data(dates[-1])

    try:
        result = has_recent_limit_up_batch(
            ts_code_list=ts_code_list,
            start_date=dates[0],
            end_date=dates[-1],
        )
    except Exception as e:
        logger.error(f"涨停池批量查询失败: {e}")
        return {ts: False for ts in ts_code_list}

    logger.info(
        f"近 {day_count} 日涨停判断完成 | 候选: {len(ts_code_list)} | 有涨停基因: {sum(result.values())}"
    )
    return result


def _filter_limit_up_on_d0(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    过滤 D 日涨停封板的股票（收盘价 == 涨停价，买不进去）
    :param daily_df: D 日候选股日线 DataFrame（含 ts_code, pre_close, close）
    :return: 过滤后的 DataFrame
    """
    if daily_df.empty:
        return daily_df

    keep_mask = []
    for _, row in daily_df.iterrows():
        ts_code   = row["ts_code"]
        pre_close = row.get("pre_close", 0)
        close     = row.get("close", 0)
        if pre_close <= 0 or close <= 0:
            keep_mask.append(True)  # 数据异常，保守保留
            continue
        limit_up = calc_limit_up_price(ts_code, pre_close)
        if limit_up > 0 and close >= limit_up - 0.01:
            keep_mask.append(False)  # D 日涨停封板，过滤
        else:
            keep_mask.append(True)

    filtered = daily_df[keep_mask].copy()
    removed  = len(daily_df) - len(filtered)
    if removed > 0:
        logger.info(f"[D日涨停过滤] 过滤涨停封板股: {removed} 只")
    return filtered


# 低流动性过滤阈值（成交额，单位：万元；tushare amount 单位为千元）
MIN_AMOUNT_THRESHOLD = 10000  # 1000万元 = 10000千元


def _filter_low_liquidity(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    过滤成交额极低的股票（策略买不进去，加入训练集只会引入噪声）
    :param daily_df: D 日候选股日线 DataFrame（含 amount 列，单位千元）
    :return: 过滤后的 DataFrame
    """
    if daily_df.empty or "amount" not in daily_df.columns:
        return daily_df

    before = len(daily_df)
    filtered = daily_df[daily_df["amount"] >= MIN_AMOUNT_THRESHOLD].copy()
    removed = before - len(filtered)
    if removed > 0:
        logger.info(f"[低流动性过滤] 过滤成交额 < {MIN_AMOUNT_THRESHOLD}千元股: {removed} 只")
    return filtered


def validate_train_dataset(csv_path: str) -> pd.DataFrame:
    """最终训练集全量校验"""
    if not os.path.exists(csv_path):
        logger.warning(f"训练集文件不存在，跳过校验: {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    logger.info(f"【最终校验】总行数: {len(df)}")

    required = ["sample_id", "strategy_id", "stock_code", "trade_date", "label1", "label2", "adapt_score"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"缺失核心列: {missing}")

    # T6: 使用 sample_id 去重
    dup = df.duplicated(subset=["sample_id"]).sum()
    if dup:
        df = df.drop_duplicates(subset=["sample_id"])
        logger.warning(f"移除重复行: {dup}")

    # 只对核心二分类 label 做 dropna；涉及 D+2/D0 的标签允许 NaN（训练时按需处理）
    null = df[["label1", "label2"]].isnull().sum().sum()
    if null:
        df = df.dropna(subset=["label1", "label2"])
        logger.warning(f"移除标签空值行: {null}")

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"【最终校验】有效行数: {len(df)}")
    return df


# ============================================================
# 主流程入口
# ============================================================

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # ==================== 可配置参数 ====================
    # 支持多时间段：每个元素为 (start_date, end_date)，格式 yyyy-mm-dd
    # 各段内的交易日会合并去重后统一处理（跨段间隙不纳入）
    DATE_RANGES: List[Tuple[str, str]] = [
        ("2023-01-01", "2023-09-30"),
        ("2024-11-01", "2026-03-10"),
    ]
    OUTPUT_CSV_PATH       = os.path.join(os.getcwd(), "datasets", "train_dataset_latest.csv")
    PROCESSED_DATES_FILE  = "processed_dates.json"
    # 因子逻辑有变更（新增列、修改计算公式）时必须更新版本号，否则旧数据不会重跑
    FACTOR_VERSION        = "v5.1_individual_factors_fixed"
    # =====================================================

    # ---------- 从多段日期范围收集全部交易日 ----------
    _date_set: set = set()
    for _s, _e in DATE_RANGES:
        _date_set.update(get_trade_dates(_s, _e))
    all_trade_dates: List[str] = sorted(_date_set)

    # LabelEngine 需要覆盖所有段的完整跨度（计算 D+1/D+2 标签）
    _label_start = min(s for s, _ in DATE_RANGES)
    _label_end   = max(e for _, e in DATE_RANGES)

    # ---------- 初始化核心组件 ----------
    feature_engine    = FeatureEngine()          # 使用 features/__init__.py 的新引擎
    label_engine      = LabelEngine(_label_start, _label_end)
    # 多策略注册：所有 supports_ml_training()==True 的策略均参与训练集生成
    _strategies       = [SectorHeatStrategy(), HighLowSwitchMLStrategy()]
    _sector_heat_strat = _strategies[0]          # 用于构建共享 FeatureDataBundle
    dates_manager     = ProcessedDatesManager(PROCESSED_DATES_FILE, FACTOR_VERSION)

    # ---------- 确定待处理日期 ----------
    to_process      = [d for d in all_trade_dates if not dates_manager.is_processed(d)]

    # ── 启动一致性检查 ─────────────────────────────────────────────────────
    # 场景：进程在"CSV 写入成功"与"标记已处理"之间崩溃
    # 结果：数据已落盘但未标记 → 下次启动会重跑该日，写入重复行
    # 修复：读取 CSV 中已有的日期，对未标记但已有数据的日期补充标记，
    #       避免重复写入（最终校验的 deduplicate 作为兜底）
    if os.path.exists(OUTPUT_CSV_PATH) and to_process:
        try:
            csv_dates = set(
                pd.read_csv(OUTPUT_CSV_PATH, usecols=["trade_date"])["trade_date"]
                .astype(str).unique()
            )
            retroactive = csv_dates & set(all_trade_dates) - set(dates_manager.processed_dates)
            if retroactive:
                logger.info(
                    f"启动一致性修复：CSV 中已有数据但未标记完成的日期 → {sorted(retroactive)}，"
                    f"自动补充标记（避免重复写入）"
                )
                for d in sorted(retroactive):
                    dates_manager.add(d)
                to_process = [d for d in all_trade_dates if not dates_manager.is_processed(d)]
        except Exception as e:
            logger.warning(f"启动一致性检查失败（忽略，继续正常处理）: {e}")

    # ── CSV 删除但全部日期已标记 → 重置 ────────────────────────────────────
    if not to_process and not os.path.exists(OUTPUT_CSV_PATH):
        logger.warning("训练集 CSV 不存在但所有日期已标记为处理完成，重置记录并重新生成")
        dates_manager.reset()
        to_process = list(all_trade_dates)
    if not to_process:
        logger.info("✅ 所有日期已处理完成！")
        validate_train_dataset(OUTPUT_CSV_PATH)
        exit(0)
    logger.info(f"待处理日期（共 {len(to_process)} 个）: {to_process}")

    # ── 连续失败计数器：超阈值直接退出，避免系统性故障下静默空跑 ──────────────
    MAX_CONSECUTIVE_FAILS = 5   # 连续 5 个日期失败 → 视为系统性异常，终止
    consecutive_fails = 0

    # ---------- CSV 写入模式 ----------
    first_write   = not os.path.exists(OUTPUT_CSV_PATH)
    fixed_columns = None
    if not first_write:
        fixed_columns = pd.read_csv(OUTPUT_CSV_PATH, nrows=0).columns.tolist()
        logger.info(f"断点续跑 | 固定列数: {len(fixed_columns)}")

    # ==================== 逐日原子性处理 ====================
    for date in to_process:
        logger.info(f"\n========== 处理日期: {date} ==========")
        try:
            # ---- Step 1: 预拉取当日宏观数据入库 ----
            date_fmt = date.replace("-", "")
            try:
                data_cleaner.clean_and_insert_limit_list_ths(trade_date=date_fmt, limit_type="涨停池")
                data_cleaner.clean_and_insert_limit_list_ths(trade_date=date_fmt, limit_type="跌停池")
                data_cleaner.clean_and_insert_limit_step(trade_date=date_fmt)
                data_cleaner.clean_and_insert_limit_cpt_list(trade_date=date_fmt)
                data_cleaner.clean_and_insert_index_daily(trade_date=date_fmt)
            except Exception as e:
                logger.error(f"{date} 宏观数据入库失败: {e}", exc_info=True)

            # ---- Step 2: 多策略候选池收集 ----
            daily_df = get_daily_kline_data(date)
            all_candidate_dfs = []
            sector_heat_context = None

            for _strat in _strategies:
                try:
                    _cdf, _ctx = _strat.build_training_candidates(date, daily_df=daily_df)
                except Exception as _e:
                    logger.warning(f"{date} [{_strat.strategy_id}] 候选池生成失败: {_e}")
                    continue
                if _strat.strategy_id == "sector_heat":
                    sector_heat_context = _ctx
                if not _cdf.empty:
                    all_candidate_dfs.append(_cdf)

            # sector_heat 候选池作为 FeatureDataBundle 的构建基础（提供板块上下文）
            if sector_heat_context is None or not sector_heat_context.get("top3_sectors"):
                logger.warning(f"{date} sector_heat Top3 板块为空，跳过")
                consecutive_fails = 0
                continue

            if not all_candidate_dfs:
                logger.warning(f"{date} 所有策略候选池为空，跳过")
                consecutive_fails = 0
                continue

            # ---- Step 3: 构建共享数据容器（union 所有策略的 ts_codes，一次 IO 覆盖所有因子）----
            # 将其他策略的 ts_codes 合并到 sector_heat context，bundle 一次性预加载
            _all_ts_codes = set(sector_heat_context.get("target_ts_codes") or [])
            for _cdf in all_candidate_dfs:
                _all_ts_codes.update(_cdf["ts_code"].tolist())
            sector_heat_context["target_ts_codes"] = list(_all_ts_codes)

            data_bundle = _sector_heat_strat.build_feature_bundle_from_context(sector_heat_context)
            if data_bundle is None:
                logger.warning(f"{date} FeatureDataBundle 构建失败，跳过")
                consecutive_fails = 0
                continue

            # ---- Step 4: 特征计算（一次计算，覆盖所有策略的候选股）----
            feature_df_base = feature_engine.run_single_date(data_bundle)
            if feature_df_base.empty:
                logger.warning(f"{date} 特征计算失败，跳过")
                continue

            # ---- Step 4.5: T6 — 多策略元数据注入 & sample_id 生成 ----
            # 对每个策略的 candidate_df 分别 merge 到 feature_df_base，再 concat
            # 同一股票可在多个策略/板块中重复出现（sample_id 包含 strategy_id + sector_name 保证唯一）
            strategy_slices = []
            for _cdf in all_candidate_dfs:
                _meta = _cdf[
                    ["ts_code", "strategy_id", "strategy_name", "sector_name", "feature_trade_date"]
                ].rename(columns={"ts_code": "stock_code"})
                _slice = _meta.merge(feature_df_base, on="stock_code", how="left")
                strategy_slices.append(_slice)

            feature_df = pd.concat(strategy_slices, ignore_index=True)
            # sample_id = "strategy_id__stock_code__trade_date__sector_name"
            feature_df["sample_id"] = (
                feature_df["strategy_id"] + "__"
                + feature_df["stock_code"] + "__"
                + feature_df["trade_date"].astype(str) + "__"
                + feature_df["sector_name"].fillna("")
            )
            logger.info(
                f"{date} 策略元数据注入完成 | 行数: {len(feature_df)} "
                f"| 唯一股票: {feature_df['stock_code'].nunique()} "
                f"| 策略分布: {feature_df['strategy_id'].value_counts().to_dict()}"
            )

            # ---- Step 5: 标签生成（用 feature_trade_date = D-1，与 feature_df.trade_date 对齐）----
            # feature_df["trade_date"] = D-1（由 FeatureDataBundle.trade_date = feature_trade_date 决定）
            # label_engine 以 D-1 为基准：D（D-1 的下一交易日）开盘买入，收盘卖出 → label1
            _feature_trade_date = sector_heat_context["feature_trade_date"]
            label_df = label_engine.generate_single_date(
                _feature_trade_date, feature_df["stock_code"].unique().tolist()
            )
            if label_df.empty:
                logger.warning(f"{date} 标签生成失败，跳过")
                continue

            # ---- Step 7: 合并 & 清洗 ----
            merged   = pd.merge(feature_df, label_df, on=["stock_code", "trade_date"], how="left")
            clean_df = DataSetAssembler.validate_and_clean(merged)
            if clean_df.empty:
                logger.warning(f"{date} 清洗后无有效数据，跳过")
                continue

            # ---- Step 8: 列对齐（断点续跑时保持列顺序一致）----
            if first_write:
                fixed_columns = clean_df.columns.tolist()
            else:
                clean_df = clean_df.reindex(columns=fixed_columns, fill_value=0)

            # ---- Step 9: 原子性写入 ----
            clean_df.to_csv(
                OUTPUT_CSV_PATH,
                mode="a", header=first_write,
                index=False, encoding="utf-8-sig"
            )
            first_write = False

            # ---- Step 10: 标记已处理（写入成功后才标记，保证幂等）----
            # 用独立 try 包裹：若 JSON 写盘失败（磁盘满等），不应影响数据，
            # 下次启动时由"启动一致性检查"补充标记即可
            try:
                dates_manager.add(date)
            except Exception as mark_err:
                logger.warning(
                    f"{date} 标记已处理失败（数据已写入，下次启动将自动补偿）: {mark_err}"
                )

            consecutive_fails = 0  # 本日成功，重置计数器
            logger.info(f"✅ {date} 处理完成，写入 {len(clean_df)} 行")

        except Exception as e:
            consecutive_fails += 1
            logger.error(
                f"{date} 处理失败 (连续失败 {consecutive_fails}/{MAX_CONSECUTIVE_FAILS}): {e}",
                exc_info=True,
            )
            if consecutive_fails >= MAX_CONSECUTIVE_FAILS:
                logger.critical(
                    f"连续 {MAX_CONSECUTIVE_FAILS} 个日期处理失败，"
                    f"疑似系统性故障（DB 断连 / 数据异常），终止训练集生成"
                )
                raise RuntimeError(
                    f"训练集生成异常退出：连续 {MAX_CONSECUTIVE_FAILS} 个日期失败"
                ) from e
            continue

    # ==================== 最终校验 ====================
    logger.info("\n========== 全量处理完成 ==========")
    if os.path.exists(OUTPUT_CSV_PATH):
        validate_train_dataset(OUTPUT_CSV_PATH)

        # T7: 写出 frozen split spec（按日期边界冻结 train/val，供 selector/train 统一读取）
        try:
            spec = write_split_spec(
                csv_path=OUTPUT_CSV_PATH,
                val_ratio=_cfg.VAL_RATIO,
                output_path=_cfg.SPLIT_SPEC_PATH,
            )
            logger.info(
                f"✅ Split spec 已写出: {_cfg.SPLIT_SPEC_PATH}\n"
                f"   train: {spec['train_start_date']} ~ {spec['train_end_date']} "
                f"({spec['train_rows']} 行)\n"
                f"   val:   {spec['val_start_date']} ~ {spec['val_end_date']} "
                f"({spec['val_rows']} 行)"
            )
        except Exception as e:
            logger.error(f"Split spec 写出失败（不影响训练集）: {e}")
    else:
        logger.error("❌ 训练集生成失败！")
