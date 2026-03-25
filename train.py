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

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from learnEngine.model import SectorHeatXGBModel
import learnEngine.train_config as cfg
from utils.log_utils import logger


# ============================================================
# 可配置参数（详细配置见 learnEngine/train_config.py）
# ============================================================
TRAIN_CSV_PATH    = cfg.TRAIN_CSV_PATH
MODEL_VERSION     = cfg.MODEL_VERSION
TARGET_LABEL      = cfg.TARGET_LABEL
VAL_RATIO         = cfg.VAL_RATIO
EXCLUDE_COLS      = cfg.EXCLUDE_COLS

_MODEL_DIR        = cfg.MODEL_DIR
MODEL_SAVE_PATH   = os.path.join(_MODEL_DIR, f"sector_heat_xgb_{MODEL_VERSION}.pkl")
MODEL_LATEST_PATH = os.path.join(_MODEL_DIR, "sector_heat_xgb_latest.pkl")


# ============================================================
# 数据加载与预处理
# ============================================================
def _load_selector_result(json_path: str, all_cols) -> tuple:
    """
    从 factor_selector.py 输出的 JSON 加载最优因子列表 + Optuna 最优超参。

    :return: (available_features: list, override_params: dict | None)
             override_params=None 表示 JSON 无超参信息（降级到 base_params）
    """
    if not os.path.exists(json_path):
        return [], None
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        selected  = data.get("selected_features", [])
        available = [c for c in selected if c in set(all_cols)]
        missing   = [c for c in selected if c not in set(all_cols)]
        if missing:
            logger.warning(f"selected_features.json 中 {len(missing)} 个因子在 CSV 中不存在，已忽略: {missing[:5]}...")
        logger.info(f"从 selected_features.json 加载 {len(available)} 个最优因子")

        # 读取 Optuna 搜到的最优 XGBoost 超参（含 scale_pos_weight）
        xgb_params = data.get("xgb_params", {})
        metrics    = data.get("metrics", {})
        spw        = metrics.get("scale_pos_weight")
        override   = None
        if xgb_params:
            override = dict(xgb_params)
            if spw is not None:
                override["scale_pos_weight"] = spw
                logger.info(f"Optuna 最优超参已加载 | scale_pos_weight={spw:.2f} | "
                            f"max_depth={xgb_params.get('max_depth')} | "
                            f"lr={xgb_params.get('learning_rate', '?'):.3f}")

        return available, override
    except Exception as e:
        logger.warning(f"加载 selected_features.json 失败 ({e})，降级为全量因子+base_params 模式")
        return [], None


def load_and_prepare(csv_path: str, target_label: str):
    """
    加载训练集 CSV 并预处理。

    特征选择优先级：
      1. 若 cfg.SELECTED_FEATURES_PATH 存在（factor_selector.py 输出）→ 使用最优因子子集 + Optuna超参
      2. 否则 → 使用 CSV 中全部数值型因子（排除 EXCLUDE_COLS）

    :return: (X, y, feature_cols, df, override_params)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"训练集文件不存在: {csv_path}\n"
            "请先运行 python learnEngine/dataset.py 生成训练集"
        )

    df = pd.read_csv(csv_path)
    logger.info(f"加载训练集: {csv_path} | 行数: {len(df)} | 列数: {len(df.columns)}")

    if target_label not in df.columns:
        raise ValueError(f"目标标签列 '{target_label}' 不存在，可用列: {df.columns.tolist()}")

    before = len(df)
    df = df.dropna(subset=[target_label])
    if len(df) < before:
        logger.info(f"删除 {target_label} 缺失行: {before - len(df)}")

    dup = df.duplicated(subset=["stock_code", "trade_date"]).sum()
    if dup > 0:
        df = df.drop_duplicates(subset=["stock_code", "trade_date"])
        logger.info(f"移除重复行: {dup}")

    # 所有候选数值因子（排除固定非特征列）
    all_numeric = [
        c for c in df.columns
        if c not in EXCLUDE_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]

    # 优先使用 factor_selector 搜出的最优子集 + Optuna 超参
    selected, override_params = _load_selector_result(cfg.SELECTED_FEATURES_PATH, all_numeric)
    if selected:
        feature_cols = selected
        logger.info(f"[模式] factor_selector 最优子集，{len(feature_cols)} 个因子")
    else:
        feature_cols = all_numeric
        logger.info(f"[模式] 全量因子（未找到 selected_features.json），{len(feature_cols)} 个因子")

    X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y = df[target_label].astype(int).values

    logger.info(f"特征列数: {len(feature_cols)} | 正样本率: {y.mean():.2%}")
    return X, y, feature_cols, df, override_params


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
def _precision_at_k(y_true, y_proba, top_k_pct: float) -> tuple:
    """
    Precision@K：取预测概率最高的前 top_k_pct 比例样本，计算其中正样本率。
    直接对应实盘行为：只买置信度最高的那批票，看实际胜率。

    :return: (precision_at_k, top_k_count, top_k_positives)
    """
    k = max(1, int(len(y_true) * top_k_pct))
    top_idx = np.argsort(y_proba)[-k:]
    top_labels = np.array(y_true)[top_idx]
    pak = top_labels.mean()
    return pak, k, int(top_labels.sum())


def evaluate_model(model, X_val, y_val, feature_cols):
    """输出详细评估指标"""
    y_pred  = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    acc       = accuracy_score(y_val, y_pred)
    auc       = roc_auc_score(y_val, y_proba)
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall    = recall_score(y_val, y_pred, zero_division=0)

    # Precision@K（与 factor_selector 目标函数对齐）
    pak, k, k_pos = _precision_at_k(y_val, y_proba, cfg.TOP_K_PCT)

    logger.info("=" * 60)
    logger.info("模型评估结果")
    logger.info("=" * 60)
    logger.info(f"  准确率 (Accuracy):  {acc:.4f}")
    logger.info(f"  AUC:                {auc:.4f}")
    logger.info(f"  精确率 (Precision): {precision:.4f}")
    logger.info(f"  召回率 (Recall):    {recall:.4f}")
    logger.info(f"  Precision@{cfg.TOP_K_PCT:.0%}:      {pak:.4f}  (Top-{k} 样本中正例 {k_pos} 个)")
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
        "precision_at_k": pak, "top_k_pct": cfg.TOP_K_PCT,
    }


# ============================================================
# 主流程
# ============================================================
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    logger.info("=" * 60)
    logger.info("开始模型训练（基线版：标准 XGBoost）")
    logger.info("=" * 60)

    # 1. 加载数据（同时读取 Optuna 最优超参）
    X, y, feature_cols, df, override_params = load_and_prepare(TRAIN_CSV_PATH, TARGET_LABEL)

    if len(X) < 50:
        logger.error(f"训练集样本不足（{len(X)} 行），至少需要 50 行，请扩大日期范围重新生成训练集")
        sys.exit(1)

    # 2. 时间序列切分
    X_train, X_val, y_train, y_val = time_series_split(X, y, df, VAL_RATIO)

    # 3. 训练模型（若有 Optuna 最优超参则直接使用，否则降级到 base_params）
    xgb_model = SectorHeatXGBModel(model_save_path=MODEL_SAVE_PATH)
    xgb_model.train(X_train, X_val, y_train, y_val, feature_cols,
                    override_params=override_params)

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
    logger.info(f"  AUC: {metrics['auc']:.4f} | Precision: {metrics['precision']:.4f} | Precision@{metrics['top_k_pct']:.0%}: {metrics['precision_at_k']:.4f}")
    logger.info("=" * 60)