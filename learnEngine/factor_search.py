"""
因子组合自动搜索 (factor_search.py)
====================================
运行方式：python -m learnEngine.factor_search [--n-trials 200] [--timeout 7200]

定位：learnEngine 层的因子筛选工具，与 factor_ic.py（单因子分析）互补。
      factor_ic.py 回答"单个因子好不好"，factor_search.py 回答"哪些因子组合在一起最好"。

原理：
    因子之间存在非线性交互效应——单因子 ICIR 高不代表组合后好，
    低 ICIR 因子也可能在组合中提供互补信息。因此需要自动搜索。

    搜索策略：
      1. 按「因子家族」分组（如 stock_max_dd 家族 = d0~d4），每组 4 种选择
      2. 独立因子单独 on/off
      3. 同时搜索 XGBoost 超参数
      → 搜索空间 ≈ 4^N_families × 2^N_standalone × 连续超参数
      → Optuna TPE 采样器可在 200~500 轮内收敛

    动态因子感知：
      不硬编码因子名，而是从训练集 CSV 的列名动态发现因子家族和独立因子。
      新增因子只需重新生成 CSV（dataset.py），本脚本自动纳入搜索范围。

输出：
    - learnEngine/search_results/best_config.json：最优因子列表 + 参数
    - 可直接粘贴到 train.py 的 EXCLUDE_PATTERNS
"""

import argparse
import json
import os
import re
import sys
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)

# 延迟导入重量级依赖（optuna, xgboost, sklearn）以减少内存峰值。
# 在 module scope 仅导入轻量包，重量级包在函数内 import。
from utils.log_utils import logger


# ============================================================
# 全局配置
# ============================================================

# 数据集路径（与 train.py / dataset.py 保持一致）
_DATASET_DIR = os.path.join(os.path.dirname(__file__), "datasets")
TRAIN_CSV_PATH = os.path.join(_DATASET_DIR, "train_dataset_latest.csv")

# 搜索结果输出目录
SEARCH_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "search_results")

# 训练目标
TARGET_LABEL = "label1"
VAL_RATIO = 0.2

# 非特征列（绝对排除，不参与搜索）
_META_COLS = frozenset([
    "stock_code", "trade_date",'label_d1_pct_chg','label_d1_high'
    "label1", "label2","label1_3pct",'label1_8pct','label_raw_return'
    "label_raw_return",'label_d2_return','label_d1_low'
    "sector_name", "top3_sectors",
])

# 绝对排除的因子模式（原始价格无跨股可比性，不应参与搜索）
_ALWAYS_EXCLUDE_PATTERNS = [
    "stock_open_d*",
    "stock_high_d*",
    "stock_low_d*",
    "stock_close_d*",
    "ma5",
    "ma10",
    "sector_id",
]


# ============================================================
# 因子动态分组引擎
# ============================================================

class FactorGroupEngine:
    """
    从训练集列名动态发现因子结构，无需硬编码因子名称。

    分组规则：
      - 匹配 `xxx_d{0-9}` 后缀的列 → 归入家族 `xxx`（如 stock_max_dd_d0~d4 → 家族 stock_max_dd）
      - 不匹配后缀的数值列 → 独立因子

    新增因子自动纳入：
      只需在 dataset.py 中新增因子列，重新生成 CSV，本引擎自动发现并纳入搜索。
    """

    DAY_SUFFIX_PATTERN = re.compile(r'^(.+)_d(\d)$')

    def __init__(self, all_features: List[str]):
        self.family_groups: Dict[str, List[str]] = defaultdict(list)
        self.standalone_features: List[str] = []
        self._build(all_features)

    def _build(self, all_features: List[str]):
        for feat in sorted(all_features):
            m = self.DAY_SUFFIX_PATTERN.match(feat)
            if m:
                family_name = m.group(1)
                self.family_groups[family_name].append(feat)
            else:
                self.standalone_features.append(feat)

        # 排序家族内部成员（d0, d1, d2, d3, d4）
        self.family_groups = {
            k: sorted(v) for k, v in sorted(self.family_groups.items())
        }

    def select_features(self, family_choices: Dict[str, str],
                        standalone_choices: Dict[str, bool]) -> List[str]:
        """根据选择方案生成最终因子列表"""
        selected = []

        for family_name, members in self.family_groups.items():
            choice = family_choices.get(family_name, "exclude")
            if choice == "d0_only":
                selected.extend(m for m in members if m.endswith("_d0"))
            elif choice == "d0_d1":
                selected.extend(m for m in members if m.endswith("_d0") or m.endswith("_d1"))
            elif choice == "all_days":
                selected.extend(members)
            # "exclude" → 不加入

        for feat in self.standalone_features:
            if standalone_choices.get(feat, False):
                selected.append(feat)

        return selected

    def summary(self) -> str:
        lines = [
            f"因子家族: {len(self.family_groups)} 组 | 独立因子: {len(self.standalone_features)} 个",
            "",
            "家族因子（每组可选 exclude/d0_only/d0_d1/all_days）:",
        ]
        for name, members in self.family_groups.items():
            days = ", ".join(m.split("_")[-1] for m in members)
            lines.append(f"  {name}: {len(members)} 个 ({days})")

        lines.append("")
        lines.append(f"独立因子（on/off）: {', '.join(self.standalone_features)}")
        return "\n".join(lines)


# ============================================================
# 数据加载
# ============================================================

def load_data(csv_path: str = TRAIN_CSV_PATH) -> Tuple[pd.DataFrame, List[str]]:
    """
    加载训练集，动态发现可搜索因子。

    新因子自动纳入：只要 CSV 中有新的数值列（不在 _META_COLS 和 _ALWAYS_EXCLUDE_PATTERNS 中），
    就会被自动发现并加入搜索范围。
    """
    from fnmatch import fnmatch

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"训练集不存在: {csv_path}\n请先运行 python -m learnEngine.dataset 生成训练集"
        )

    # 注意：pandas C parser 在导入 xgboost+sklearn 后可能 OOM（已知 bug），
    # 使用 python engine 作为稳定方案（22K 行性能差异可忽略）。
    df = pd.read_csv(csv_path, engine="python")
    # 数值列转 float32 节省内存
    numeric_cols = df.select_dtypes(include=[np.float64]).columns
    df[numeric_cols] = df[numeric_cols].astype(np.float32)
    logger.info(f"加载训练集: {len(df)} 行 × {len(df.columns)} 列 | 内存: {df.memory_usage(deep=True).sum()/1e6:.1f}MB")

    df = df.dropna(subset=[TARGET_LABEL])
    df = df.drop_duplicates(subset=["stock_code", "trade_date"])

    # 动态发现所有数值特征列
    all_features = [
        c for c in df.columns
        if c not in _META_COLS
        and pd.api.types.is_numeric_dtype(df[c])
        and not any(fnmatch(c, pat) for pat in _ALWAYS_EXCLUDE_PATTERNS)
    ]

    logger.info(f"可搜索因子: {len(all_features)} 个（从 CSV 动态发现）")
    return df, all_features


def time_series_split(df: pd.DataFrame, feature_cols: List[str]):
    """时间序列切分（与 train.py 逻辑一致）"""
    X = df[feature_cols].copy().fillna(0).replace([np.inf, -np.inf], 0)
    y = df[TARGET_LABEL].astype(int).values

    dates = df["trade_date"].values
    sort_idx = np.argsort(dates)
    X = X.iloc[sort_idx].reset_index(drop=True)
    y = y[sort_idx]

    split_point = int(len(X) * (1 - VAL_RATIO))
    X_train, X_val = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_val = y[:split_point], y[split_point:]

    df_sorted = df.iloc[sort_idx].reset_index(drop=True)
    df_train = df_sorted.iloc[:split_point]  # 保留以备后续扩展

    return X_train, X_val, y_train, y_val, df_train


# ============================================================
# 搜索引擎
# ============================================================

class FactorSearchEngine:
    """
    Optuna 驱动的因子组合 + 超参数联合搜索引擎。

    搜索维度：
      1. 因子家族选择（exclude / d0_only / d0_d1 / all_days）
      2. 独立因子 on/off
      3. XGBoost 超参数（max_depth, learning_rate, n_estimators 等）

    优化目标：
      综合分 = w_recall × Recall + w_precision × Precision + w_auc × AUC
      当 Recall < min_recall_floor 时施加惩罚，避免模型完全不预测正样本。
    """

    # 优化目标权重（可调）
    WEIGHT_RECALL    = 0.4
    WEIGHT_PRECISION = 0.4
    WEIGHT_AUC       = 0.2
    MIN_RECALL_FLOOR = 0.05   # Recall 低于此值时 score 减半
    MIN_FEATURES     = 3      # 至少保留的因子数

    def __init__(self, df: pd.DataFrame, all_features: List[str]):
        self.df = df
        self.group_engine = FactorGroupEngine(all_features)
        logger.info(self.group_engine.summary())

    def _objective(self, trial) -> float:
        """Optuna 目标函数（trial 类型: optuna.Trial）"""
        import xgboost as xgb
        from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score

        # ── 1. 因子选择 ────────────────────────────────────────
        family_choices = {}
        for family_name in self.group_engine.family_groups:
            family_choices[family_name] = trial.suggest_categorical(
                f"fam_{family_name}",
                ["exclude", "d0_only", "d0_d1", "all_days"],
            )

        standalone_choices = {}
        for feat in self.group_engine.standalone_features:
            standalone_choices[feat] = trial.suggest_categorical(
                f"feat_{feat}", [True, False],
            )

        selected = self.group_engine.select_features(family_choices, standalone_choices)

        if len(selected) < self.MIN_FEATURES:
            return 0.0

        # ── 2. XGBoost 超参数 ──────────────────────────────────
        xgb_params = {
            "objective":          "binary:logistic",
            "eval_metric":        "auc",
            "max_depth":          trial.suggest_int("max_depth", 2, 6),
            "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators":       trial.suggest_int("n_estimators", 100, 800, step=100),
            "subsample":          trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight":   trial.suggest_int("min_child_weight", 1, 10),
            "gamma":              trial.suggest_float("gamma", 0.0, 0.5),
            "reg_alpha":          trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda":         trial.suggest_float("reg_lambda", 0.5, 3.0),
            "scale_pos_weight":   1.0,  # 类别平衡由 sample_weight 承担
            "early_stopping_rounds": 20,
            "n_jobs":             -1,
            "random_state":       42,
            "verbosity":          0,
        }

        # ── 3. 准备数据 ────────────────────────────────────────
        X_train, X_val, y_train, y_val, _ = time_series_split(self.df, selected)

        # ── 4. 训练 ────────────────────────────────────────────
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # ── 5. 评估 ────────────────────────────────────────────
        y_pred  = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]

        recall    = recall_score(y_val, y_pred, zero_division=0)
        precision = precision_score(y_val, y_pred, zero_division=0)
        auc       = roc_auc_score(y_val, y_proba)
        f1        = f1_score(y_val, y_pred, zero_division=0)

        # 综合评分
        score = (
            self.WEIGHT_RECALL * recall
            + self.WEIGHT_PRECISION * precision
            + self.WEIGHT_AUC * auc
        )
        if recall < self.MIN_RECALL_FLOOR:
            score *= 0.5

        # 记录指标
        trial.set_user_attr("n_features", len(selected))
        trial.set_user_attr("recall", round(recall, 4))
        trial.set_user_attr("precision", round(precision, 4))
        trial.set_user_attr("auc", round(auc, 4))
        trial.set_user_attr("f1", round(f1, 4))
        trial.set_user_attr("features", selected)

        return score

    def run(self, n_trials: int = 200, timeout: Optional[int] = None,
            study_name: str = "factor_search"):
        """运行搜索，返回 optuna.Study"""
        import optuna
        from optuna.samplers import TPESampler

        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=TPESampler(seed=42, n_startup_trials=20),
        )

        study.optimize(
            self._objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
        )

        return study


# ============================================================
# 结果输出
# ============================================================

class SearchResultExporter:
    """将搜索结果导出为 JSON 和可粘贴的 train.py 配置"""

    def __init__(self, study, group_engine: FactorGroupEngine):
        self.study = study
        self.group_engine = group_engine

    def export_json(self, output_dir: str = SEARCH_RESULTS_DIR) -> dict:
        """导出最优配置到 JSON"""
        os.makedirs(output_dir, exist_ok=True)

        best = self.study.best_trial

        # 提取因子选择
        factor_families = {}
        for family_name in self.group_engine.family_groups:
            key = f"fam_{family_name}"
            if key in best.params:
                factor_families[family_name] = best.params[key]

        standalone_factors = {}
        for feat in self.group_engine.standalone_features:
            key = f"feat_{feat}"
            if key in best.params:
                standalone_factors[feat] = best.params[key]

        result = {
            "best_score": round(best.value, 4),
            "metrics": {
                "recall": best.user_attrs.get("recall"),
                "precision": best.user_attrs.get("precision"),
                "auc": best.user_attrs.get("auc"),
                "f1": best.user_attrs.get("f1"),
                "n_features": best.user_attrs.get("n_features"),
            },
            "xgb_params": {
                k: round(best.params[k], 4) if isinstance(best.params.get(k), float) else best.params.get(k)
                for k in ["max_depth", "learning_rate", "n_estimators", "subsample",
                           "colsample_bytree", "min_child_weight", "gamma", "reg_alpha", "reg_lambda"]
                if k in best.params
            },
            "factor_families": factor_families,
            "standalone_factors": standalone_factors,
            "selected_features": sorted(best.user_attrs.get("features", [])),
        }

        output_path = os.path.join(output_dir, "best_config.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        logger.info(f"最优配置已保存: {output_path}")
        return result

    def print_summary(self):
        """打印搜索结果摘要"""
        best = self.study.best_trial

        logger.info("=" * 70)
        logger.info("搜索完成")
        logger.info("=" * 70)
        logger.info(f"总试验次数: {len(self.study.trials)}")

        logger.info(f"\n最优试验 (Trial #{best.number}):")
        logger.info(f"  综合评分:   {best.value:.4f}")
        logger.info(f"  Recall:     {best.user_attrs.get('recall', 0):.4f}")
        logger.info(f"  Precision:  {best.user_attrs.get('precision', 0):.4f}")
        logger.info(f"  AUC:        {best.user_attrs.get('auc', 0):.4f}")
        logger.info(f"  F1:         {best.user_attrs.get('f1', 0):.4f}")
        logger.info(f"  因子数:     {best.user_attrs.get('n_features', 0)}")

        # Top 10 试验
        valid_trials = [t for t in self.study.trials if t.value is not None]
        sorted_trials = sorted(valid_trials, key=lambda t: t.value, reverse=True)

        logger.info(f"\nTop 10 试验:")
        header = f"  {'#':>4s}  {'Score':>7s}  {'Recall':>7s}  {'Prec':>7s}  {'AUC':>7s}  {'F1':>7s}  {'Nfeat':>5s}"
        logger.info(header)
        for t in sorted_trials[:10]:
            logger.info(
                f"  {t.number:4d}  {t.value:7.4f}  "
                f"{t.user_attrs.get('recall', 0):7.4f}  "
                f"{t.user_attrs.get('precision', 0):7.4f}  "
                f"{t.user_attrs.get('auc', 0):7.4f}  "
                f"{t.user_attrs.get('f1', 0):7.4f}  "
                f"{t.user_attrs.get('n_features', 0):5d}"
            )

        # 输出最优因子列表
        features = best.user_attrs.get("features", [])
        if features:
            logger.info(f"\n最优因子组合 ({len(features)} 个):")
            for feat in sorted(features):
                logger.info(f"  - {feat}")

    def print_train_config(self, result: dict):
        """输出可直接粘贴到 train.py 的配置代码"""
        logger.info("\n" + "=" * 70)
        logger.info("以下配置可直接粘贴到 train.py")
        logger.info("=" * 70)

        # EXCLUDE_PATTERNS
        excluded_families = [
            fname for fname, choice in result["factor_families"].items()
            if choice == "exclude"
        ]
        excluded_standalone = [
            feat for feat, use in result["standalone_factors"].items()
            if not use
        ]
        # 部分选择的家族（d0_only / d0_d1）→ 排除不选的 day 后缀
        partial_excludes = []
        for fname, choice in result["factor_families"].items():
            if choice == "d0_only":
                partial_excludes.append(f'    "{fname}_d[1-4]",  # 仅保留 d0')
            elif choice == "d0_d1":
                partial_excludes.append(f'    "{fname}_d[2-4]",  # 仅保留 d0, d1')

        logger.info("")
        logger.info("# --- EXCLUDE_PATTERNS（粘贴到 train.py）---")
        logger.info("EXCLUDE_PATTERNS = [")
        logger.info('    # 原始价格（无跨股可比性）')
        logger.info('    "stock_open_*", "stock_high_*", "stock_low_*", "stock_close_*",')
        logger.info('    "ma5", "ma10", "sector_id",')
        if excluded_families:
            logger.info('    # Optuna 搜索排除的因子家族')
            for fname in sorted(excluded_families):
                logger.info(f'    "{fname}_*",')
        if excluded_standalone:
            logger.info('    # Optuna 搜索排除的独立因子')
            for feat in sorted(excluded_standalone):
                logger.info(f'    "{feat}",')
        if partial_excludes:
            logger.info('    # Optuna 搜索：部分日期排除')
            for line in sorted(partial_excludes):
                logger.info(line)
        logger.info("]")

        # XGBoost base_params
        xp = result["xgb_params"]
        logger.info("")
        logger.info("# --- XGBoost base_params（粘贴到 learnEngine/model.py）---")
        for key, val in xp.items():
            logger.info(f'    "{key}": {val},')


# ============================================================
# 主入口
# ============================================================

def main():
    # 先导入重量级依赖（在 CSV 加载前），避免 pandas C parser OOM
    logger.info("加载依赖...")
    import xgboost  # noqa: F401
    import optuna    # noqa: F401
    logger.info("依赖加载完成")

    parser = argparse.ArgumentParser(description="Optuna 因子组合自动搜索")
    parser.add_argument("--n-trials", type=int, default=200,
                        help="搜索试验次数（默认 200，建议首次 100 快速验证）")
    parser.add_argument("--timeout", type=int, default=None,
                        help="超时秒数（默认不限，建议设置 3600~7200）")
    parser.add_argument("--study-name", type=str, default="factor_search",
                        help="Optuna study 名称")
    parser.add_argument("--csv", type=str, default=TRAIN_CSV_PATH,
                        help="训练集 CSV 路径（默认 latest）")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("Optuna 因子组合 + 超参数联合搜索")
    logger.info(f"试验次数: {args.n_trials} | 超时: {args.timeout}s")
    logger.info("=" * 70)

    # 1. 加载数据
    df, all_features = load_data(args.csv)

    # 2. 创建搜索引擎并运行
    engine = FactorSearchEngine(df, all_features)
    study = engine.run(
        n_trials=args.n_trials,
        timeout=args.timeout,
        study_name=args.study_name,
    )

    # 3. 输出结果
    exporter = SearchResultExporter(study, engine.group_engine)
    exporter.print_summary()
    result = exporter.export_json()
    exporter.print_train_config(result)

    logger.info("\n完成！请根据上述输出更新 train.py 的配置，然后重新训练验证。")


if __name__ == "__main__":
    main()
