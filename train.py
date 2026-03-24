"""
模型训练入口 (train.py)
========================
运行方式：python train.py

前置条件：
    已运行 python learnEngine/dataset.py 生成 train_dataset.csv

流程：
    1. 加载训练集 CSV
    2. 数据预处理（清洗、特征/标签分离）
    3. 时间序列切分 train / val（避免未来数据泄漏）
    4. 训练 XGBoost 模型
    5. 输出评估指标 + 特征重要性
    6. 保存模型
"""

import os
import sys
import warnings
from fnmatch import fnmatch

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    classification_report, confusion_matrix, precision_recall_curve, f1_score,
)
from typing import List

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from learnEngine.model import SectorHeatXGBModel
# [回退] 亏损惩罚功能回测表现不佳，暂时停用，保留代码以备后续优化后重新启用
# from risk_penalty_core import train_with_risk_penalty, RiskPenaltyConfig
from utils.log_utils import logger


# ============================================================
# 可配置参数
# ============================================================
# 数据集路径（与 dataset.py OUTPUT_CSV_PATH 保持一致）
TRAIN_CSV_PATH   = os.path.join(os.getcwd(), "learnEngine", "datasets", "train_dataset_final.csv")

# 模型版本号（与 dataset.py FACTOR_VERSION 保持一致，修改因子时两处同步更新）
MODEL_VERSION    = "v4.0"  # [回退] 去除亏损惩罚后恢复基线版本号
_MODEL_DIR       = os.path.join(os.getcwd(), "model")
# 版本化存档：每次训练生成独立文件，不覆盖历史模型
MODEL_SAVE_PATH  = os.path.join(_MODEL_DIR, f"sector_heat_xgb_{MODEL_VERSION}.pkl")
# 稳定路径：策略/回测统一加载此文件，train.py 训练完后自动同步覆盖
MODEL_LATEST_PATH = os.path.join(_MODEL_DIR, "sector_heat_xgb_latest.pkl")  # 【修改2】改回 .pkl，保持兼容

TARGET_LABEL     = "label1"       # 训练目标：label1 (日内 5% 收益) 或 label2 (隔夜高开)
VAL_RATIO        = 0.2            # 验证集占比（按时间序列尾部切分）

# 需要排除的非特征列（主键 + 标签 + 辅助信息）
EXCLUDE_COLS = [
    "stock_code", "trade_date",
    "label1", "label2",
    "label_raw_return",   # D+1实际收益率，仅用于样本权重生成，不作为训练特征
    "sector_name", "top3_sectors",
]

# 因子过滤模式（fnmatch 通配符）：匹配到的列将被排除在训练特征之外
# 修改此处即可在不重新生成 CSV 的情况下切换因子组合，无需重跑 dataset.py
EXCLUDE_PATTERNS: List[str] = [
    # ===== 原始价格类：无跨股可比性，归一化信息已由 bias/ma_slope 携带 =====
    "stock_open_*",
    "stock_high_*",
    "stock_low_*",
    "stock_close_*",
    "ma5",
    "ma10",
    # "ma13",  # 保留：旧模型使用，对趋势判断有增量

    # ===== 全 NaN / 无数据因子 =====
    "market_*",     # 市场宏观类：全为 NaN，完全无预测力
    "sector_id",    # 板块分类 ID：ICIR≈0

    # ===== d1-d4 滞后因子 =====
    "*_d[1-4]",     # 仅 d0 当日因子有效，d1-d4 预测力严重衰减

    # ===== 冗余因子 =====
    "bias5",        # 与 bias13 高度相关
    "bias10",       # 与 bias13 高度相关
    "pos_5d",       # 与 pos_20d 高度相关
    "index_*",      # 指数涨跌幅：与 market_* 同源，训练集中全 NaN

    # ===== 以下因子保留给模型/factor_search 自动判断 =====
    # 注释 = 保留（旧模型使用，对 AUC/Recall 有贡献）
    # 'sector_avg_profit_*',
    # 'sector_avg_loss_*',
    # 'stock_sector_20d_rank',
    # 'stock_vol_ratio_*',
    # 'ma_align',
    # 'from_high_20d',
    # 'stock_vwap_dev_*',

    # ===== 低 ICIR 因子（旧模型也排除） =====
    "stock_pct_chg_*",          # 涨跌幅原始值
    "stock_hdi_*",              # HDI 持股难度
    "stock_profit_*",           # 盈利情绪分（与 max_dd 高度相关）
    "stock_seal_times_*",       # 涨停封板次数
    "stock_break_times_*",      # 涨停打开次数
    # "stock_lift_times_*",     # 保留：旧模型使用
    "stock_trend_r2_*",         # 趋势拟合
    "stock_cpr_*",              # K 线比率
    "stock_candle_*",           # K 线形态
    # "stock_gap_return_*",     # 保留：旧模型使用
    # "stock_lower_shadow_*",   # 保留：旧模型使用
    "stock_red_time_ratio_*",
    "stock_float_profit_time_ratio_*",
    "stock_red_session_pm_ratio_*",
    "stock_float_session_pm_ratio_*",
]


# ============================================================
# 数据加载与预处理
# ============================================================
def load_and_prepare(csv_path: str, target_label: str):
    """
    加载训练集 CSV 并预处理

    :return: (X, y, feature_cols, df) — 特征矩阵、标签、特征列名、原始 DataFrame
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"训练集文件不存在: {csv_path}\n请先运行 python learnEngine/dataset.py 生成训练集")

    df = pd.read_csv(csv_path)
    logger.info(f"加载训练集: {csv_path} | 行数: {len(df)} | 列数: {len(df.columns)}")

    # 检查标签列
    if target_label not in df.columns:
        raise ValueError(f"目标标签列 '{target_label}' 不存在于训练集中，可用列: {df.columns.tolist()}")

    # 删除标签缺失的行
    before = len(df)
    df = df.dropna(subset=[target_label])
    if len(df) < before:
        logger.info(f"删除 {target_label} 缺失行: {before - len(df)}")

    # 去重
    dup = df.duplicated(subset=["stock_code", "trade_date"]).sum()
    if dup > 0:
        df = df.drop_duplicates(subset=["stock_code", "trade_date"])
        logger.info(f"移除重复行: {dup}")

    # 分离特征与标签
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    # 仅保留数值型列
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]

    # EXCLUDE_PATTERNS 过滤（fnmatch 通配符，不影响 CSV 生成）
    if EXCLUDE_PATTERNS:
        feature_cols = [c for c in feature_cols
                        if not any(fnmatch(c, pat) for pat in EXCLUDE_PATTERNS)]
        logger.info(f"EXCLUDE_PATTERNS 过滤后特征列数: {len(feature_cols)}")

    X = df[feature_cols].copy()
    y = df[target_label].astype(int).values

    # 缺失值填 0（与 DataSetAssembler 一致）
    X = X.fillna(0)

    # inf 替换为 0
    X = X.replace([np.inf, -np.inf], 0)

    logger.info(f"特征列数: {len(feature_cols)} | 正样本率: {y.mean():.2%}")
    return X, y, feature_cols, df


# ============================================================
# 时间序列切分（严格按 trade_date 排序，避免未来数据泄漏）
# ============================================================
def time_series_split(X, y, df, val_ratio: float):
    """
    按 trade_date 排序后尾部切分验证集（不随机打乱）

    :return: (X_train, X_val, y_train, y_val)
    """
    # 按 trade_date 排序
    dates = df["trade_date"].values
    sort_idx = np.argsort(dates)
    X = X.iloc[sort_idx].reset_index(drop=True)
    y = y[sort_idx]

    split_point = int(len(X) * (1 - val_ratio))
    X_train, X_val = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_val = y[:split_point], y[split_point:]

    # 获取切分日期信息
    sorted_dates = dates[sort_idx]
    train_end   = sorted_dates[split_point - 1] if split_point > 0 else "N/A"
    val_start   = sorted_dates[split_point]      if split_point < len(dates) else "N/A"

    logger.info(
        f"时间序列切分 | 训练集: {len(X_train)} 行 (截至 {train_end}) | "
        f"验证集: {len(X_val)} 行 (从 {val_start} 起)"
    )
    logger.info(f"训练集正样本率: {y_train.mean():.2%} | 验证集正样本率: {y_val.mean():.2%}")

    return X_train, X_val, y_train, y_val


# ============================================================
# 详细评估
# ============================================================
def evaluate_model(model, X_val, y_val, feature_cols):
    """输出详细评估指标"""
    y_pred  = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    acc       = accuracy_score(y_val, y_pred)
    auc       = roc_auc_score(y_val, y_proba)
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall    = recall_score(y_val, y_pred, zero_division=0)

    logger.info("=" * 60)
    logger.info("模型评估结果")
    logger.info("=" * 60)
    logger.info(f"  准确率 (Accuracy):  {acc:.4f}")
    logger.info(f"  AUC:                {auc:.4f}")
    logger.info(f"  精确率 (Precision): {precision:.4f}")
    logger.info(f"  召回率 (Recall):    {recall:.4f}")
    logger.info("")

    # 混淆矩阵
    cm = confusion_matrix(y_val, y_pred)
    logger.info("混淆矩阵:")
    logger.info(f"  TN={cm[0][0]}  FP={cm[0][1]}")
    logger.info(f"  FN={cm[1][0]}  TP={cm[1][1]}")
    logger.info("")

    # 分类报告
    logger.info("分类报告:")
    logger.info("\n" + classification_report(y_val, y_pred, target_names=["不买", "买入"]))

    # # 决策阈值优化（默认 0.5 对不平衡数据往往太高）
    # logger.info("决策阈值搜索:")
    # prec_curve, rec_curve, thresholds = precision_recall_curve(y_val, y_proba)
    # # 计算每个阈值的 F1
    # f1_array = 2 * (prec_curve[:-1] * rec_curve[:-1]) / (prec_curve[:-1] + rec_curve[:-1] + 1e-10)
    # best_f1_idx = np.argmax(f1_array)
    # best_threshold = thresholds[best_f1_idx]
    # best_f1 = f1_array[best_f1_idx]
    #
    # # 用最优阈值重新计算指标
    # y_pred_opt = (y_proba >= best_threshold).astype(int)
    # prec_opt = precision_score(y_val, y_pred_opt, zero_division=0)
    # rec_opt  = recall_score(y_val, y_pred_opt, zero_division=0)
    # f1_opt   = f1_score(y_val, y_pred_opt, zero_division=0)
    #
    # logger.info(f"  默认阈值 0.5:  Precision={precision:.4f}  Recall={recall:.4f}  F1={2*precision*recall/(precision+recall+1e-10):.4f}")
    # logger.info(f"  最优阈值 {best_threshold:.3f}: Precision={prec_opt:.4f}  Recall={rec_opt:.4f}  F1={f1_opt:.4f}")
    #
    # # 展示多个候选阈值的效果
    # logger.info("  阈值敏感度分析:")
    # for t in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    #     y_t = (y_proba >= t).astype(int)
    #     p_t = precision_score(y_val, y_t, zero_division=0)
    #     r_t = recall_score(y_val, y_t, zero_division=0)
    #     f_t = f1_score(y_val, y_t, zero_division=0)
    #     n_pred = y_t.sum()
    #     logger.info(f"    threshold={t:.2f}: Prec={p_t:.4f} Rec={r_t:.4f} F1={f_t:.4f} 预测买入数={n_pred}")
    # logger.info("")

    # 特征重要性 Top 20
    importances = model.feature_importances_
    fi = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    logger.info("Top 20 重要特征:")
    for i, row in fi.head(20).iterrows():
        logger.info(f"  {row['feature']:40s} {row['importance']:.4f}")

    return {
        "accuracy": acc, "auc": auc, "precision": precision, "recall": recall,
        # "best_threshold": float(best_threshold),
        # "precision_at_best": prec_opt, "recall_at_best": rec_opt, "f1_at_best": f1_opt,
    }


# ============================================================
# 主流程
# ============================================================
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    logger.info("=" * 60)
    logger.info("开始模型训练（基线版：标准 XGBoost）")
    logger.info("=" * 60)

    # 1. 加载数据
    X, y, feature_cols, df = load_and_prepare(TRAIN_CSV_PATH, TARGET_LABEL)

    if len(X) < 50:
        logger.error(f"训练集样本不足（{len(X)} 行），至少需要 50 行，请扩大日期范围重新生成训练集")
        sys.exit(1)

    # 2. 时间序列切分
    X_train, X_val, y_train, y_val = time_series_split(X, y, df, VAL_RATIO)

    # 3. 训练模型
    xgb_model = SectorHeatXGBModel(model_save_path=MODEL_SAVE_PATH)
    xgb_model.train(X_train, X_val, y_train, y_val, feature_cols)

    # [回退] 亏损惩罚回测表现不佳，暂停使用。保留代码以备后续优化后重新启用。
    # ========== 亏损惩罚配置（已停用） ==========
    # sort_idx    = np.argsort(df["trade_date"].values)
    # split_point = int(len(df) * (1 - VAL_RATIO))
    # df_train_partition = df.iloc[sort_idx[:split_point]]
    #
    # config = RiskPenaltyConfig.for_strategy_model()
    # config.loss_weight_multiplier = 1.5
    # config.high_risk_env_extra_multiplier = 1.15
    # config.loss_severity_tiers = (
    #     (-0.07, 1.8),
    #     (-0.03, 1.4),
    #     ( 0.00, 1.1),
    #     ( 0.03, 1.0),
    # )
    # train_with_risk_penalty(
    #     xgb_model, X_train, X_val, y_train, y_val, feature_cols,
    #     df_train=df_train_partition, label_col=TARGET_LABEL, config=config,
    # )

    # 4. 详细评估
    metrics = evaluate_model(xgb_model.model, X_val, y_val, feature_cols)

    # 5. 同步 latest（策略/回测稳定加载路径）
    import shutil
    shutil.copy2(MODEL_SAVE_PATH, MODEL_LATEST_PATH)
    logger.info(f"已同步至稳定路径: {MODEL_LATEST_PATH}")

    # 6. 完成
    logger.info("=" * 60)
    logger.info(f"训练完成！版本化模型: {MODEL_SAVE_PATH}")
    logger.info(f"  训练集: {len(X_train)} 行 | 验证集: {len(X_val)} 行")
    logger.info(f"  AUC: {metrics['auc']:.4f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}")
    logger.info("=" * 60)