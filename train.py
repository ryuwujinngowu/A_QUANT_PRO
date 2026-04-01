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

from learnEngine.model import StrategyXGBModel
import learnEngine.train_config as cfg
from utils.log_utils import logger


def _derive_feature_modules(feature_cols: list) -> list:
    """从模型特征列名反推所需因子模块（用于推断时按需裁剪数据加载）。"""
    try:
        from features.feature_registry import feature_registry
        return feature_registry.get_modules_for_columns(feature_cols)
    except Exception as e:
        logger.warning(f"[train] feature_modules 推导失败，meta.json 中留空: {e}")
        return []


# ============================================================
# 可配置参数（详细配置见 learnEngine/train_config.py）
# ============================================================
TRAIN_CSV_PATH    = cfg.TRAIN_CSV_PATH
MODEL_VERSION     = cfg.MODEL_VERSION
TARGET_LABEL      = cfg.TARGET_LABEL
VAL_RATIO         = cfg.VAL_RATIO
EXCLUDE_COLS      = cfg.EXCLUDE_COLS


def resolve_training_artifacts(strategy_id: str = None, version: str = None) -> dict:
    strategy_id = strategy_id or getattr(cfg, "STRATEGY_ID", None)
    version = version or MODEL_VERSION
    dataset_dir = cfg.get_latest_valid_dataset_dir()
    return {
        "dataset_dir": dataset_dir,
        "csv_path": cfg.get_dataset_csv_path(dataset_dir),
        "split_spec_path": cfg.get_split_spec_path(dataset_dir),
        "selected_features_path": cfg.get_selected_features_path(strategy_id, dataset_dir),
        "model_dir": cfg.get_model_version_dir(strategy_id, version) if strategy_id else cfg.MODEL_DIR,
        "model_pkl_path": cfg.get_model_pkl_path(strategy_id, version) if strategy_id else os.path.join(cfg.MODEL_DIR, f"strategy_xgb_{version}.pkl"),
        "model_json_path": cfg.get_model_json_path(strategy_id, version) if strategy_id else os.path.join(cfg.MODEL_DIR, f"strategy_xgb_{version}.json"),
        "model_meta_path": cfg.get_model_meta_path(strategy_id, version) if strategy_id else os.path.join(cfg.MODEL_DIR, f"strategy_xgb_{version}.meta.json"),
        "model_config_snapshot_path": cfg.get_model_config_snapshot_path(strategy_id, version) if strategy_id else os.path.join(cfg.MODEL_DIR, f"strategy_xgb_{version}.config.json"),
        "runtime_model_path": cfg.get_strategy_runtime_model_path(strategy_id) if strategy_id else None,
        "runtime_model_meta_path": cfg.get_strategy_runtime_model_meta_path(strategy_id) if strategy_id else None,
    }


# T11: 路径由策略 ID 驱动，训练只写版本归档，runtime 人工晋升
# 兼容旧路径：若 STRATEGY_ID 未设置，退回到原 model/ 根目录
def _resolve_model_paths(strategy_id, version):
    artifacts = resolve_training_artifacts(strategy_id=strategy_id, version=version)
    return artifacts["model_pkl_path"], artifacts["runtime_model_path"]


# ============================================================
# 数据加载与预处理
# ============================================================
def _selector_result_is_usable(json_path: str, csv_path: str, target_label: str) -> bool:
    """检查因子筛选结果是否存在、可读，且与当前训练任务匹配。"""
    if not os.path.exists(json_path):
        return False

    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning(f"selected_features.json 读取失败: {e}")
        return False

    selected = data.get("selected_features", [])
    if not selected:
        logger.warning("selected_features.json 中没有 selected_features，视为无效")
        return False

    saved_target = data.get("target_label")
    if saved_target and saved_target != target_label:
        logger.warning(
            f"selected_features.json 目标标签不匹配: 当前={target_label} | 文件={saved_target}"
        )
        return False

    saved_csv = data.get("csv_path")
    if saved_csv and os.path.abspath(saved_csv) != os.path.abspath(csv_path):
        logger.warning(
            "selected_features.json 对应的训练集与当前配置不一致，"
            f"当前={os.path.abspath(csv_path)} | 文件={saved_csv}"
        )
        return False

    return True


def _ensure_selector_result(csv_path: str, target_label: str, selector_json: str = None):
    """
    确保 factor_selector 结果存在且与当前训练集匹配。

    若配置允许，则自动运行一遍 FactorSelector，避免静默回退到全因子训练。
    """
    selector_json = selector_json or cfg.SELECTED_FEATURES_PATH
    need_refresh = cfg.FACTOR_SELECTOR_FORCE_REFRESH or not _selector_result_is_usable(
        selector_json, csv_path, target_label
    )
    if not need_refresh:
        return

    if not cfg.AUTO_RUN_FACTOR_SELECTOR:
        return

    logger.info(
        "未发现可用的因子筛选结果，开始自动运行 FactorSelector..."
        f" | csv={csv_path} | target={target_label} | stage={cfg.FACTOR_SELECTOR_STAGE}"
    )
    from learnEngine.factor_selector import FactorSelector

    selector = FactorSelector(csv_path=csv_path, target=target_label)
    selector.run(stage=cfg.FACTOR_SELECTOR_STAGE, output_path=selector_json)


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


def load_and_prepare(
    csv_path:    str,
    target_label: str,
    strategy_id: str = None,
):
    """
    加载训练集 CSV 并预处理。

    T9 新增:
      - strategy_id 过滤（按策略 ID 取对应行，支持多策略训练池）
      - 去重键改为 sample_id（全局训练池中同股同日可属不同策略）
      - 加载 frozen split spec（若存在）

    特征选择优先级：
      1. 若 cfg.SELECTED_FEATURES_PATH 存在（factor_selector.py 输出）→ 使用最优因子子集 + Optuna超参
      2. 否则 → 使用 CSV 中全部数值型因子（排除 EXCLUDE_COLS）

    :return: (X, y, feature_cols, df, override_params, split_spec)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"训练集文件不存在: {csv_path}\n"
            "请先运行 python learnEngine/dataset.py 生成训练集"
        )

    df = pd.read_csv(csv_path)
    logger.info(f"加载训练集: {csv_path} | 行数: {len(df)} | 列数: {len(df.columns)}")

    # 过滤 stk_factor_pro API 超时导致的中性值污染行
    if {'rsi_6', 'kdj_k', 'cci'}.issubset(df.columns):
        bad_mask = (df['rsi_6'] == 50.0) & (df['kdj_k'] == 50.0) & (df['cci'] == 0.0)
        if bad_mask.sum() > 0:
            logger.info(f"过滤 stk_factor_pro 中性值污染行: {bad_mask.sum()} 行")
            df = df[~bad_mask].reset_index(drop=True)

    # T9: strategy_id 过滤（支持多策略训练池）
    if strategy_id and "strategy_id" in df.columns:
        before_sid = len(df)
        df = df[df["strategy_id"] == strategy_id].reset_index(drop=True)
        logger.info(f"[T9] strategy_id='{strategy_id}' 过滤: {before_sid} → {len(df)} 行")
    elif strategy_id:
        logger.warning(f"[T9] strategy_id='{strategy_id}' 但训练集无 strategy_id 列，跳过过滤")

    if target_label not in df.columns:
        raise ValueError(f"目标标签列 '{target_label}' 不存在，可用列: {df.columns.tolist()}")

    before = len(df)
    df = df.dropna(subset=[target_label])
    if len(df) < before:
        logger.info(f"删除 {target_label} 缺失行: {before - len(df)}")

    # T9: 去重键改为 sample_id（存在时），兼容旧数据集的 stock+date 去重
    _dedup_key = ["sample_id"] if "sample_id" in df.columns else ["stock_code", "trade_date"]
    dup = df.duplicated(subset=_dedup_key).sum()
    if dup > 0:
        df = df.drop_duplicates(subset=_dedup_key)
        logger.info(f"移除重复行（按 {_dedup_key}）: {dup}")

    # T9 + strategy_configs: 按策略 ID 动态计算有效排除列
    # → 自动排除其他策略的专属列（如 adapt_score 对非板块热度策略无意义）
    from learnEngine.strategy_configs import get_effective_exclude_cols
    _effective_exclude = get_effective_exclude_cols(strategy_id, EXCLUDE_COLS)

    all_numeric = [
        c for c in df.columns
        if c not in _effective_exclude and pd.api.types.is_numeric_dtype(df[c])
    ]

    artifacts = resolve_training_artifacts(strategy_id=strategy_id)
    _ensure_selector_result(csv_path, target_label, selector_json=artifacts["selected_features_path"])
    selected, override_params = _load_selector_result(artifacts["selected_features_path"], all_numeric)
    if selected:
        feature_cols = selected
        logger.info(f"[模式] factor_selector 最优子集，{len(feature_cols)} 个因子")
    else:
        if cfg.REQUIRE_SELECTED_FEATURES:
            raise RuntimeError(
                "未获取到可用的 selected_features.json，已阻止全因子盲搜。\n"
                f"请先运行 python learnEngine/factor_selector.py --csv {csv_path}"
            )
        feature_cols = all_numeric
        logger.info(f"[模式] 全量因子（未找到 selected_features.json），{len(feature_cols)} 个因子")

    X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y = df[target_label].astype(int).values

    # T9: 加载 frozen split spec
    split_spec = None
    split_spec_path = artifacts["split_spec_path"]
    if os.path.exists(split_spec_path):
        from learnEngine.split_spec import load_split_spec
        try:
            split_spec = load_split_spec(split_spec_path)
            logger.info(
                f"[T9] Frozen split spec 加载: "
                f"train ≤ {split_spec['train_end_date']} | "
                f"val ≥ {split_spec['val_start_date']}"
            )
        except Exception as e:
            logger.warning(f"[T9] split spec 加载失败（降级用 val_ratio 切）: {e}")
    else:
        logger.info(f"[T9] split spec 不存在，将用 val_ratio={VAL_RATIO} 按行切")

    logger.info(f"特征列数: {len(feature_cols)} | 正样本率: {y.mean():.2%}")
    return X, y, feature_cols, df, override_params, split_spec


# ============================================================
# 时间序列切分（严格按 trade_date 排序，避免未来数据泄漏）
# ============================================================
def time_series_split(X, y, df, val_ratio: float, split_spec=None):
    """
    按 trade_date 排序后尾部切分验证集（不随机打乱）。

    T9: 有 frozen split_spec 时按日期边界切分；无 spec 时降级用 val_ratio 按行切。

    :return: (X_train, X_val, y_train, y_val)
    """
    # 按 trade_date 排序
    dates = df["trade_date"].values
    sort_idx = np.argsort(dates)
    X = X.iloc[sort_idx].reset_index(drop=True)
    y = y[sort_idx]
    sorted_dates = dates[sort_idx]
    sorted_df = df.iloc[sort_idx].reset_index(drop=True)

    if split_spec:
        from learnEngine.split_spec import apply_split_spec
        tr_df, val_df = apply_split_spec(sorted_df, split_spec)
        tr_idx  = tr_df.index
        val_idx = val_df.index
        X_train, X_val = X.loc[tr_idx], X.loc[val_idx]
        y_train, y_val = y[tr_idx], y[val_idx]
        train_end = split_spec["train_end_date"]
        val_start = split_spec["val_start_date"]
        logger.info(f"[T9] Frozen split 边界: train ≤ {train_end} | val ≥ {val_start}")
    else:
        split_point = int(len(X) * (1 - val_ratio))
        X_train, X_val = X.iloc[:split_point], X.iloc[split_point:]
        y_train, y_val = y[:split_point], y[split_point:]
        train_end = sorted_dates[split_point - 1] if split_point > 0 else "N/A"
        val_start = sorted_dates[split_point]      if split_point < len(sorted_dates) else "N/A"

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

    # T9: 策略 ID（从 cfg 读取，None = 全策略池）
    _strategy_id = getattr(cfg, "STRATEGY_ID", None)

    logger.info("=" * 60)
    logger.info(f"开始模型训练 | strategy_id={_strategy_id or '全策略'} | label={TARGET_LABEL}")
    logger.info("=" * 60)

    artifacts = resolve_training_artifacts(strategy_id=_strategy_id, version=MODEL_VERSION)
    logger.info(f"dataset_dir={artifacts['dataset_dir']}")
    logger.info(f"selected_features={artifacts['selected_features_path']}")

    # 1. 加载数据（同时读取 Optuna 最优超参 + frozen split spec）
    X, y, feature_cols, df, override_params, split_spec = load_and_prepare(
        artifacts["csv_path"], TARGET_LABEL, strategy_id=_strategy_id
    )

    if len(X) < 50:
        logger.error(f"训练集样本不足（{len(X)} 行），至少需要 50 行，请扩大日期范围重新生成训练集")
        sys.exit(1)

    # 2. 时间序列切分（T9: 优先使用 frozen split spec）
    X_train, X_val, y_train, y_val = time_series_split(X, y, df, VAL_RATIO, split_spec=split_spec)

    # 3. 训练模型（T10/T11: StrategyXGBModel 写入版本目录）
    archive_path, _runtime_path = _resolve_model_paths(_strategy_id, MODEL_VERSION)
    os.makedirs(os.path.dirname(archive_path), exist_ok=True)
    xgb_model = StrategyXGBModel(model_save_path=archive_path)
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

    # 5. 完成（训练只写 model/<strategy>/<version>，runtime 需人工晋升到策略目录）
    try:
        model_meta = {
            "strategy_id": _strategy_id,
            "model_version": MODEL_VERSION,
            "trained_at": pd.Timestamp.now().isoformat(),
            "dataset_dir": artifacts["dataset_dir"],
            "dataset_csv": artifacts["csv_path"],
            "split_spec_path": artifacts["split_spec_path"],
            "selected_features_path": artifacts["selected_features_path"],
            "selected_features": feature_cols,
            "feature_modules": _derive_feature_modules(feature_cols),
            "target_label": TARGET_LABEL,
            "xgb_params": override_params or getattr(xgb_model, "base_params", {}),
            "train_rows": int(len(X_train)),
            "val_rows": int(len(X_val)),
            "train_date_range": {
                "start": str(df["trade_date"].min()) if not df.empty else None,
                "end": split_spec.get("train_end_date") if split_spec else None,
            },
            "val_date_range": {
                "start": split_spec.get("val_start_date") if split_spec else None,
                "end": split_spec.get("val_end_date") if split_spec else None,
            },
            "metrics": metrics,
            "feature_count": len(feature_cols),
            "candidate_pool_rule_summary": f"strategy={_strategy_id}; build_training_candidates() shared strategy logic",
            "runtime_promotion_target": artifacts["runtime_model_path"],
        }
        with open(artifacts["model_meta_path"], "w", encoding="utf-8") as f:
            json.dump(model_meta, f, ensure_ascii=False, indent=2)
        with open(artifacts["model_config_snapshot_path"], "w", encoding="utf-8") as f:
            json.dump({
                "strategy_id": _strategy_id,
                "model_version": MODEL_VERSION,
                "target_label": TARGET_LABEL,
                "val_ratio": VAL_RATIO,
                "exclude_cols": EXCLUDE_COLS,
                "dataset_dir": artifacts["dataset_dir"],
                "selected_features_path": artifacts["selected_features_path"],
            }, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"模型元数据写出失败: {e}")

    logger.info("=" * 60)
    logger.info(f"训练完成！model 版本目录: {os.path.dirname(archive_path)}")
    logger.info(f"  runtime 手工晋升目标: {_runtime_path}")
    logger.info(f"  训练集: {len(X_train)} 行 | 验证集: {len(X_val)} 行")
    logger.info(f"  AUC: {metrics['auc']:.4f} | Precision: {metrics['precision']:.4f} | Precision@{metrics['top_k_pct']:.0%}: {metrics['precision_at_k']:.4f}")
    logger.info("=" * 60)
