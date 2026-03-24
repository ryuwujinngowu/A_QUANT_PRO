"""
模型信号共享 Helper
====================
为所有基于 SectorHeatStrategy 模型输出的短线 Agent 提供统一的信号获取接口。

核心职责：
    在 D 日调用 SectorHeatStrategy 的完整选股流程（板块热度 → 候选池 → 特征计算 → XGBoost 预测），
    返回模型输出的买入信号列表（ts_code, stock_name, close_price, prob）。

设计意图：
    3 个模型信号 Agent（平铺开盘买、拉涨买入、恐慌低吸）共享同一套信号源，
    只在 D+1 日买点时机上有差异。提取公共逻辑避免 3 份重复代码。
"""
import os
import pickle
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config.config import FILTER_BSE_STOCK, FILTER_STAR_BOARD, FILTER_688_BOARD
from data.data_cleaner import data_cleaner
from features import FeatureEngine, FeatureDataBundle
from features.sector.sector_heat_feature import SectorHeatFeature
from utils.common_tools import (
    filter_st_stocks,
    get_daily_kline_data,
    get_stocks_in_sector,
    get_trade_dates,
    has_recent_limit_up_batch,
    ensure_limit_list_ths_data,
)
from utils.log_utils import logger

# ── 策略参数（与 sector_heat_strategy.py 保持一致）──────────────────────────
_BUY_TOP_K = 6        # 每日最多信号数
_MIN_PROB  = 0.6       # 最低买入概率阈值
_MIN_AMOUNT = 10_000   # 低流动性阈值（千元，= 1000 万元）
_LOAD_MINUTE = True    # 特征计算是否加载分钟线（与训练口径一致）

# 模型路径（与 sector_heat_strategy.py 一致）
_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "model", "sector_heat_xgb_model.pkl",
)

# 模块级单例（避免每个 agent 各自加载一份模型和特征引擎）
_sector_heat = SectorHeatFeature()
_feature_engine = FeatureEngine()
_model = None


def _ensure_model():
    """懒加载模型，返回模型实例或 None"""
    global _model
    if _model is not None:
        return _model
    if not os.path.exists(_MODEL_PATH):
        logger.error(f"[model_signal] 模型文件不存在: {_MODEL_PATH}")
        return None
    try:
        with open(_MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        if not hasattr(_model, "feature_names_in_"):
            logger.error("[model_signal] 模型缺少 feature_names_in_ 属性")
            _model = None
            return None
        logger.info(f"[model_signal] 模型加载成功 | 特征数: {len(_model.feature_names_in_)}")
        return _model
    except Exception as e:
        logger.error(f"[model_signal] 模型加载失败: {e}")
        _model = None
        return None


def _filter_ts_code_by_board(ts_code_list: List[str]) -> List[str]:
    """过滤北交所 / 科创板 / 创业板（与 sector_heat_strategy 一致）"""
    result = []
    for ts in ts_code_list:
        if not ts:
            continue
        if FILTER_BSE_STOCK and (ts.endswith(".BJ") or ts.startswith(("83", "87", "88"))):
            continue
        # if FILTER_688_BOARD and ts.startswith("688"):
        #     continue
        # if FILTER_STAR_BOARD and ts.startswith(("300", "301", "302")) and ts.endswith(".SZ"):
        #     continue
        result.append(ts)
    return result


def _calc_limit_up_price(ts_code: str, pre_close: float) -> float:
    """计算涨停价"""
    if pre_close <= 0:
        return 0.0
    if ts_code.startswith(("300", "301", "688")):
        return round(pre_close * 1.19, 2)
    if ts_code.startswith(("83", "87", "88")) or ts_code.endswith(".BJ"):
        return round(pre_close * 1.30, 2)
    return round(pre_close * 1.09, 2)


def get_model_signal_stocks(
    trade_date: str,
    daily_data: pd.DataFrame,
    caller_agent_id: str = "",
) -> List[Dict]:
    """
    获取 D 日模型输出的买入信号股列表。

    :param trade_date: D 日（YYYY-MM-DD 格式）
    :param daily_data: D 日全市场日线 DataFrame
    :param caller_agent_id: 调用方 agent_id（用于日志前缀）
    :return: [{"ts_code": ..., "stock_name": ..., "close_price": ..., "prob": ...}, ...]
             close_price 为 D 日收盘价（策略尾盘买入价），
             各 Agent 可根据自身买点逻辑使用 D+1 分钟线确定实际买入价。
    """
    tag = f"[{caller_agent_id}]" if caller_agent_id else "[model_signal]"

    # ── 模型加载 ──────────────────────────────────────────────────────────
    model = _ensure_model()
    if model is None:
        logger.error(f"{tag}[{trade_date}] 模型未就绪，无信号")
        return []

    # ── 日期格式 ──────────────────────────────────────────────────────────
    if len(trade_date) == 8 and trade_date.isdigit():
        trade_date_dash = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}"
    else:
        trade_date_dash = trade_date

    # ── Step 1: Top3 板块 ─────────────────────────────────────────────────
    try:
        top3_result  = _sector_heat.select_top3_hot_sectors(trade_date_dash)
        top3_sectors = top3_result["top3_sectors"]
        adapt_score  = top3_result["adapt_score"]
    except Exception as e:
        logger.error(f"{tag}[{trade_date}] 板块热度计算失败: {e}")
        return []

    if not top3_sectors:
        logger.warning(f"{tag}[{trade_date}] Top3 板块为空")
        return []

    logger.info(f"{tag}[{trade_date}] Top3={top3_sectors} | adapt_score={adapt_score}")

    # ── ST 数据入库 ───────────────────────────────────────────────────────
    try:
        data_cleaner.insert_stock_st(trade_date=trade_date_dash.replace("-", ""))
    except Exception:
        pass

    # ── Step 2: 候选池构建 ────────────────────────────────────────────────
    sector_candidate_map: Dict[str, pd.DataFrame] = {}
    from datetime import datetime, timedelta

    for sector in top3_sectors:
        try:
            raw = get_stocks_in_sector(sector)
            if not raw:
                sector_candidate_map[sector] = pd.DataFrame()
                continue

            ts_codes = [item["ts_code"] for item in raw]
            ts_codes = _filter_ts_code_by_board(ts_codes)
            if not ts_codes:
                sector_candidate_map[sector] = pd.DataFrame()
                continue

            ts_codes = filter_st_stocks(ts_codes, trade_date_dash)
            if not ts_codes:
                sector_candidate_map[sector] = pd.DataFrame()
                continue

            sector_daily = daily_data[daily_data["ts_code"].isin(ts_codes)].copy()
            if sector_daily.empty:
                sector_candidate_map[sector] = pd.DataFrame()
                continue

            # 涨停基因过滤（近 10 日）
            candidates = sector_daily["ts_code"].unique().tolist()
            try:
                end_dt = datetime.strptime(trade_date_dash, "%Y-%m-%d")
                pre_end = (end_dt - timedelta(days=1)).strftime("%Y-%m-%d")
                start_60 = (end_dt - timedelta(days=60)).strftime("%Y-%m-%d")
                dates = get_trade_dates(start_60, pre_end)[-10:]
                if dates:
                    ensure_limit_list_ths_data(dates[-1])
                    limit_map = has_recent_limit_up_batch(candidates, dates[0], dates[-1])
                    keep = [ts for ts, has in limit_map.items() if has]
                    sector_daily = sector_daily[sector_daily["ts_code"].isin(keep)]
            except Exception:
                pass

            if sector_daily.empty:
                sector_candidate_map[sector] = pd.DataFrame()
                continue

            # D 日涨停封板过滤
            keep_mask = []
            for _, row in sector_daily.iterrows():
                pre_close = float(row.get("pre_close") or 0)
                close = float(row.get("close") or 0)
                if pre_close <= 0 or close <= 0:
                    keep_mask.append(True)
                    continue
                lu = _calc_limit_up_price(row["ts_code"], pre_close)
                keep_mask.append(lu <= 0 or close < lu - 0.01)
            sector_daily = sector_daily[keep_mask]

            # 低流动性过滤
            if "amount" in sector_daily.columns:
                sector_daily = sector_daily[sector_daily["amount"] >= _MIN_AMOUNT]

            sector_candidate_map[sector] = sector_daily

        except Exception as e:
            logger.error(f"{tag}[{trade_date}][{sector}] 候选池构建失败: {e}")
            sector_candidate_map[sector] = pd.DataFrame()

    target_ts_codes = list({
        ts
        for df in sector_candidate_map.values()
        if not df.empty
        for ts in df["ts_code"].tolist()
    })
    if not target_ts_codes:
        logger.warning(f"{tag}[{trade_date}] 候选池为空")
        return []

    # ── Step 3: 特征计算 ──────────────────────────────────────────────────
    try:
        bundle = FeatureDataBundle(
            trade_date=trade_date_dash,
            target_ts_codes=target_ts_codes,
            sector_candidate_map=sector_candidate_map,
            top3_sectors=top3_sectors,
            adapt_score=adapt_score,
            load_minute=_LOAD_MINUTE,
        )
        feature_df = _feature_engine.run_single_date(bundle)
    except Exception as e:
        logger.error(f"{tag}[{trade_date}] 特征计算失败: {e}")
        return []

    if feature_df.empty:
        logger.warning(f"{tag}[{trade_date}] 特征计算结果为空")
        return []

    # ── Step 4: XGBoost 预测 ──────────────────────────────────────────────
    try:
        expected_cols = list(model.feature_names_in_)
        X = (
            feature_df
            .reindex(columns=expected_cols, fill_value=0)
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .replace([np.inf, -np.inf], 0)
        )
        probs = model.predict_proba(X)[:, 1]
        feature_df = feature_df.copy()
        feature_df["_prob"] = probs
    except Exception as e:
        logger.error(f"{tag}[{trade_date}] 模型预测失败: {e}")
        return []

    # ── Step 5: 排序选股 ──────────────────────────────────────────────────
    selected = (
        feature_df[feature_df["_prob"] >= _MIN_PROB]
        .sort_values("_prob", ascending=False)
        .head(_BUY_TOP_K)
    )

    if selected.empty:
        logger.info(f"{tag}[{trade_date}] 无股票超过概率阈值 {_MIN_PROB}")
        return []

    # ── 构建结果 ──────────────────────────────────────────────────────────
    # 需要从 daily_data 取 D 日收盘价和名称
    daily_sub = daily_data[daily_data["ts_code"].isin(selected["stock_code"].tolist())]
    close_map = {row["ts_code"]: float(row["close"]) for _, row in daily_sub.iterrows()}
    name_map  = {row["ts_code"]: str(row.get("name", "")) for _, row in daily_sub.iterrows()}

    result = []
    for _, row in selected.iterrows():
        ts = row["stock_code"]
        result.append({
            "ts_code":     ts,
            "stock_name":  name_map.get(ts, ""),
            "close_price": close_map.get(ts, 0.0),  # D 日收盘价
            "prob":        round(float(row["_prob"]), 4),
        })

    logger.info(
        f"{tag}[{trade_date}] 模型输出 {len(result)} 只信号: "
        + " | ".join(f"{s['ts_code']}(p={s['prob']:.3f})" for s in result)
    )
    return result
