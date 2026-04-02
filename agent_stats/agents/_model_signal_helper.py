"""
模型信号共享 Helper
====================
为所有基于 SectorHeatStrategy 模型输出的短线 Agent 提供统一的信号获取接口。

核心职责：
    在 D 日调用 SectorHeatStrategy 的共享候选池 + 特征计算流程，
    返回模型输出的买入信号列表（ts_code, stock_name, close_price, prob）。
"""
import os
import pickle
from typing import Dict, List

import numpy as np
import pandas as pd

from config.config import FILTER_BSE_STOCK
import learnEngine.train_config as cfg
from strategies import SectorHeatStrategy
from utils.log_utils import logger
from utils.xgb_compat import safe_predict_proba

# ── 策略参数（与 sector_heat_strategy.py 保持一致）──────────────────────────
_BUY_TOP_K = 6        # 每日最多信号数
_MIN_PROB  = 0.60      # 最低买入概率阈值
_MIN_AMOUNT = 10_000   # 低流动性阈值（千元，= 1000 万元）
_LOAD_MINUTE = True    # 特征计算是否加载分钟线（与训练口径一致）

# 模型路径（推理层读取策略目录下的 runtime_model；要求目录中恰好只有一个版本模型）
_MODEL_PATH = cfg.get_strategy_runtime_model_path("sector_heat")

# 模块级单例（避免每个 agent 各自加载一份模型和策略对象）
_strategy = SectorHeatStrategy()
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
        if not hasattr(_model, "use_label_encoder"):
            _model.use_label_encoder = False
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

    model = _ensure_model()
    if model is None:
        logger.error(f"{tag}[{trade_date}] 模型未就绪，无信号")
        return []

    if len(trade_date) == 8 and trade_date.isdigit():
        trade_date_dash = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}"
    else:
        trade_date_dash = trade_date

    try:
        candidate_df, context = _strategy.build_training_candidates(trade_date_dash, daily_df=daily_data)
    except Exception as e:
        logger.error(f"{tag}[{trade_date}] 共享候选池构建失败: {e}")
        return []

    if candidate_df.empty:
        logger.warning(f"{tag}[{trade_date}] 候选池为空")
        return []

    logger.info(
        f"{tag}[{trade_date}] Top3={context.get('top3_sectors')} | "
        f"adapt_score={context.get('adapt_score')}"
    )

    try:
        bundle = _strategy.build_feature_bundle_from_context(context)
        if bundle is None:
            logger.warning(f"{tag}[{trade_date}] Feature bundle 为空")
            return []
        feature_df = _strategy._feature_engine.run_single_date(bundle)
    except Exception as e:
        logger.error(f"{tag}[{trade_date}] 特征计算失败: {e}")
        return []

    if feature_df.empty:
        logger.warning(f"{tag}[{trade_date}] 特征计算结果为空")
        return []

    try:
        expected_cols = list(model.feature_names_in_)
        X = (
            feature_df
            .reindex(columns=expected_cols, fill_value=0)
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .replace([np.inf, -np.inf], 0)
        )
        probs = safe_predict_proba(model, X)[:, 1]
        feature_df = feature_df.copy()
        feature_df["_prob"] = probs
    except Exception as e:
        logger.error(f"{tag}[{trade_date}] 模型预测失败: {e}")
        return []

    selected = (
        feature_df[feature_df["_prob"] >= _MIN_PROB]
        .sort_values("_prob", ascending=False)
        .head(_BUY_TOP_K)
    )
    if selected.empty:
        logger.info(f"{tag}[{trade_date}] 无股票超过概率阈值 {_MIN_PROB}")
        return []

    daily_sub = daily_data[daily_data["ts_code"].isin(selected["stock_code"].tolist())]
    close_map = {row["ts_code"]: float(row["close"]) for _, row in daily_sub.iterrows()}
    name_map = {row["ts_code"]: str(row.get("name", "")) for _, row in daily_sub.iterrows()}

    result = []
    for _, row in selected.iterrows():
        ts = row["stock_code"]
        result.append({
            "ts_code": ts,
            "stock_name": name_map.get(ts, ""),
            "close_price": close_map.get(ts, 0.0),
            "prob": round(float(row["_prob"]), 4),
        })

    logger.info(
        f"{tag}[{trade_date}] 模型输出 {len(result)} 只信号: "
        + " | ".join(f"{s['ts_code']}(p={s['prob']:.3f})" for s in result)
    )
    return result
