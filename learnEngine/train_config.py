"""
learnEngine/train_config.py
===========================
统一配置入口 — 所有可调参数集中在此处。
train.py 和 factor_selector.py 均从此处读取，避免参数分散在多个文件。

修改参数只需改此文件，无需碰 train.py / factor_selector.py 的逻辑代码。
"""
import json
import os
import re
from datetime import datetime
from typing import Optional

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录
BASE_DIR = os.path.join(_BASE_DIR, "learnEngine")
DATASET_ROOT_DIR = os.path.join(BASE_DIR, "datasets")
MODEL_DIR = os.path.join(_BASE_DIR, "model")
STRATEGY_ROOT_DIR = os.path.join(_BASE_DIR, "strategies")


# ═══════════════════════════════════════════════
# dataset artifact 目录
# ═══════════════════════════════════════════════
DATASET_CSV_NAME = "train_dataset.csv"
SPLIT_SPEC_NAME = "split_spec.json"
PROCESSED_DATES_NAME = "processed_dates.json"
DATASET_MANIFEST_NAME = "dataset_manifest.json"
TRAIN_CONFIG_SNAPSHOT_NAME = "train_config.snapshot.json"


def create_dataset_run_id(prefix: Optional[str] = None) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    pid = os.getpid()
    rand = os.urandom(2).hex()
    base = f"{ts}_p{pid}_{rand}"
    return f"{prefix}_{base}" if prefix else base


def get_dataset_dir(run_id: str) -> str:
    return os.path.join(DATASET_ROOT_DIR, run_id)


def get_dataset_csv_path(dataset_dir: str) -> str:
    return os.path.join(dataset_dir, DATASET_CSV_NAME)


def get_split_spec_path(dataset_dir: str) -> str:
    return os.path.join(dataset_dir, SPLIT_SPEC_NAME)


def get_processed_dates_path(dataset_dir: str) -> str:
    return os.path.join(dataset_dir, PROCESSED_DATES_NAME)


def get_dataset_manifest_path(dataset_dir: str) -> str:
    return os.path.join(dataset_dir, DATASET_MANIFEST_NAME)


def get_dataset_config_snapshot_path(dataset_dir: str) -> str:
    return os.path.join(dataset_dir, TRAIN_CONFIG_SNAPSHOT_NAME)


def get_selected_features_path(strategy_id: Optional[str], dataset_dir: Optional[str] = None) -> str:
    base_dir = dataset_dir or get_latest_valid_dataset_dir()
    name = f"selected_features_{strategy_id}.json" if strategy_id else "selected_features_all.json"
    return os.path.join(base_dir, name)


def _dataset_dir_is_valid(dataset_dir: str) -> bool:
    csv_path = get_dataset_csv_path(dataset_dir)
    split_path = get_split_spec_path(dataset_dir)
    manifest_path = get_dataset_manifest_path(dataset_dir)
    if not (
        os.path.isdir(dataset_dir)
        and os.path.exists(csv_path)
        and os.path.exists(split_path)
        and os.path.exists(manifest_path)
    ):
        return False
    try:
        with open(split_path, encoding="utf-8") as f:
            spec = json.load(f)
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

        saved_csv = spec.get("dataset_csv")
        manifest_csv = manifest.get("dataset_csv")
        manifest_status = manifest.get("status")
        validation_status = manifest.get("validation_status")

        return (
            bool(saved_csv)
            and bool(manifest_csv)
            and os.path.abspath(saved_csv) == os.path.abspath(csv_path)
            and os.path.abspath(manifest_csv) == os.path.abspath(csv_path)
            and manifest_status == "completed"
            and validation_status == "passed"
        )
    except Exception:
        return False


def get_latest_valid_dataset_dir() -> str:
    if not os.path.isdir(DATASET_ROOT_DIR):
        raise FileNotFoundError(f"dataset 根目录不存在: {DATASET_ROOT_DIR}")
    candidates = []
    for name in sorted(os.listdir(DATASET_ROOT_DIR), reverse=True):
        path = os.path.join(DATASET_ROOT_DIR, name)
        if _dataset_dir_is_valid(path):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"未找到有效 dataset 目录: {DATASET_ROOT_DIR}")
    return candidates[0]


def _safe_default_dataset_dir() -> str:
    try:
        return get_latest_valid_dataset_dir()
    except Exception:
        return DATASET_ROOT_DIR


_DEFAULT_DATASET_DIR = _safe_default_dataset_dir()

# 默认消费最新有效 dataset 目录
TRAIN_CSV_PATH = get_dataset_csv_path(_DEFAULT_DATASET_DIR)
SPLIT_SPEC_PATH = get_split_spec_path(_DEFAULT_DATASET_DIR)
SELECTED_FEATURES_PATH = get_selected_features_path("sector_heat", _DEFAULT_DATASET_DIR)


# ═══════════════════════════════════════════════
# 模型产物目录结构约定（按策略 / 版本固化）
# ═══════════════════════════════════════════════
# 目录规范：
#   model/<strategy_id>/<version>/               ← 训练归档（完整版本固化）
#   strategies/<strategy_id>/runtime_model/      ← 手工晋升后的运行模型


def get_model_version_dir(strategy_id: str, version: str) -> str:
    return os.path.join(MODEL_DIR, strategy_id, version)


def get_model_pkl_path(strategy_id: str, version: str) -> str:
    return os.path.join(get_model_version_dir(strategy_id, version), "model.pkl")


def get_model_json_path(strategy_id: str, version: str) -> str:
    return os.path.join(get_model_version_dir(strategy_id, version), "model.json")


def get_model_meta_path(strategy_id: str, version: str) -> str:
    return os.path.join(get_model_version_dir(strategy_id, version), "model_meta.json")


def get_model_config_snapshot_path(strategy_id: str, version: str) -> str:
    return os.path.join(get_model_version_dir(strategy_id, version), "train_config.snapshot.json")


def get_strategy_runtime_model_dir(strategy_id: str) -> str:
    return os.path.join(STRATEGY_ROOT_DIR, strategy_id, "runtime_model")


def get_strategy_runtime_model_pattern(strategy_id: str) -> str:
    return rf"^{re.escape(strategy_id)}_V[^/\\]+\\.pkl$"


def get_strategy_runtime_model_path(strategy_id: str) -> str:
    """
    返回 runtime_model 目录中最新（mtime 最大）的模型文件路径。
    若目录不存在或无匹配文件，返回占位路径（运行时会报 FileNotFoundError）。
    多文件时自动选最新，并打印 warning，不再抛异常。
    """
    runtime_dir = get_strategy_runtime_model_dir(strategy_id)
    if not os.path.isdir(runtime_dir):
        return os.path.join(runtime_dir, f"{strategy_id}_V1.pkl")

    pattern = re.compile(get_strategy_runtime_model_pattern(strategy_id))
    candidates = [
        os.path.join(runtime_dir, name)
        for name in os.listdir(runtime_dir)
        if pattern.match(name)
    ]
    if not candidates:
        return os.path.join(runtime_dir, f"{strategy_id}_V1.pkl")
    # 多个候选时按 mtime 降序，取最新
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    if len(candidates) > 1:
        import warnings
        warnings.warn(
            f"[train_config] runtime_model 目录存在多个候选模型，自动选最新: "
            f"{os.path.basename(candidates[0])}  "
            f"(其余: {[os.path.basename(p) for p in candidates[1:]]})",
            stacklevel=2,
        )
    return candidates[0]


def get_strategy_runtime_model_meta_path(strategy_id: str) -> str:
    model_path = get_strategy_runtime_model_path(strategy_id)
    return model_path.replace(".pkl", ".meta.json")


# ═══════════════════════════════════════════════
# 训练基础配置
# ═══════════════════════════════════════════════
STRATEGY_ID = "sector_heat"
MODEL_VERSION = "acceptance_v1"
TARGET_LABEL  = "label1_3pct"
VAL_RATIO     = 0.2

# 未来两阶段开盘架构预留列（当前仅占位/排除，不参与训练与筛因）
RESERVED_FUTURE_LABEL_COLS = [
    "label_open_regime_stage1",
    "label_open_regime_stage1_bin",
    "label_open_regime_stage2",
    "label_open_regime_stage2_bin",
]

RESERVED_FUTURE_FEATURE_COLS = [
    "feat_pred_open_regime_stage1",
    "feat_pred_open_regime_stage1_prob_low",
    "feat_pred_open_regime_stage1_prob_mid",
    "feat_pred_open_regime_stage1_prob_high",
    "feat_pred_open_regime_stage2",
    "feat_pred_open_regime_stage2_conf",
]

EXCLUDE_COLS = [
    "stock_code", "trade_date",
    "label1", "label2", "label1_3pct", "label1_8pct", "label_d2_limit_down",
    "label_raw_return", "label_open_gap", "label_d1_high", "label_d1_low",
    "label_d1_pct_chg", "label_d2_return", "label_d2_5pct", "label_5d_30pct", "label_d1_open_up",
    "sector_name", "top3_sectors",
    "sample_id", "strategy_id", "strategy_name",
    *RESERVED_FUTURE_LABEL_COLS,
    *RESERVED_FUTURE_FEATURE_COLS,
]

IC_MIN_ICIR     = 0.15
IC_MIN_WIN_RATE = 0.48
IC_MAX_PVALUE   = 0.20
CORR_MAX = 0.75
N_TRIALS            = 400
TIMEOUT             = 7200
TOP_K_PCT           = 0.30
MIN_TOP_K_POSITIVES = 10
MIN_FEATURES        = 5
N_CV_FOLDS          = 1
XGB_MAX_DEPTH_RANGE        = (3, 7)
XGB_LR_RANGE               = (0.02, 0.20)
XGB_N_EST_RANGE            = (80, 260)
XGB_SUBSAMPLE_RANGE        = (0.65, 1.00)
XGB_COLSAMPLE_RANGE        = (0.60, 1.00)
XGB_MIN_CHILD_WEIGHT_RANGE = (1, 10)
XGB_GAMMA_RANGE            = (0.0, 4.0)
XGB_REG_ALPHA_RANGE        = (0.0, 3.0)
XGB_REG_LAMBDA_RANGE       = (0.5, 8.0)
FACTOR_SELECTOR_FORCE_REFRESH = False
AUTO_RUN_FACTOR_SELECTOR      = False
FACTOR_SELECTOR_STAGE         = "12"
REQUIRE_SELECTED_FEATURES     = False
