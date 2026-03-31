"""
learnEngine/factor_selector.py
===============================
三阶段因子筛选器

  Stage 1: IC 过滤    — 复用 factor_ic.py，剔除三项指标均差的无效因子
  Stage 2: 相关性去重 — 高度相关的因子只保留 |ICIR| 最高的代表
  Stage 3: Optuna精调 — 在精简候选集上联合搜索因子子集 + XGBoost 超参
                        目标函数：Precision@K + AUC（纯盈利，无亏损惩罚）

运行方式：
    # 全流程（推荐）
    python learnEngine/factor_selector.py

    # 快速验证（20轮）
    python learnEngine/factor_selector.py --n-trials 20

    # 只跑 Stage 1+2（查看哪些因子被剔除）
    python learnEngine/factor_selector.py --stage 12

    # 基于上次 Stage 1+2 结果只重跑 Stage 3
    python learnEngine/factor_selector.py --stage 3

输出：learnEngine/search_results/selected_features.json
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import learnEngine.train_config as cfg
from utils.log_utils import logger


# ═══════════════════════════════════════════════════════════════════════
# 辅助函数（无副作用，易单独测试）
# ═══════════════════════════════════════════════════════════════════════

def _precision_at_k(y_true: np.ndarray, y_proba: np.ndarray, k_pct: float) -> float:
    """取预测概率最高的前 k_pct 比例样本，计算其实际正样本率。"""
    k = max(1, int(len(y_true) * k_pct))
    top_k_idx = np.argsort(y_proba)[::-1][:k]
    return float(y_true[top_k_idx].mean())


def _time_split(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    val_ratio: float,
    split_spec: Optional[Dict] = None,
):
    """
    按 trade_date 严格时序切分，返回 (X_tr, X_val, y_tr, y_val)。

    T8: 优先使用 frozen split_spec（按日期边界），无 spec 时降级用 val_ratio 按行切。
    T8: 去重键改为 sample_id（存在时），兼容旧数据的 stock_code+trade_date 去重。
    """
    _dedup_key = ["sample_id"] if "sample_id" in df.columns else ["stock_code", "trade_date"]
    df = (
        df.dropna(subset=[target])
        .drop_duplicates(subset=_dedup_key)
        .sort_values("trade_date")
        .reset_index(drop=True)
    )
    X = df[features].fillna(0).replace([np.inf, -np.inf], 0)
    y = df[target].astype(int).values

    if split_spec:
        from learnEngine.split_spec import apply_split_spec
        tr_df, val_df = apply_split_spec(df, split_spec)
        tr_idx  = tr_df.index
        val_idx = val_df.index
        return X.loc[tr_idx], X.loc[val_idx], y[tr_idx], y[val_idx]

    split = int(len(df) * (1 - val_ratio))
    return X.iloc[:split], X.iloc[split:], y[:split], y[split:]


def _single_fold_eval(
    X_tr, X_val, y_tr, y_val,
    xgb_params: Dict,
    top_k_pct: float,
    min_top_k_pos: int,
    scale_pos_weight: float,
) -> Dict:
    """单折训练评估，供 _train_eval 的 CV 模式调用。"""
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    model = xgb.XGBClassifier(
        **xgb_params,
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=30,
        n_jobs=-1,
        random_state=42,
        verbosity=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    proba     = model.predict_proba(X_val)[:, 1]
    pak       = _precision_at_k(y_val, proba, top_k_pct)
    auc       = float(roc_auc_score(y_val, proba)) if len(np.unique(y_val)) > 1 else 0.5
    k         = max(1, int(len(y_val) * top_k_pct))
    top_k_pos = int(y_val[np.argsort(proba)[::-1][:k]].sum())
    score     = (0.3 * auc) if top_k_pos < min_top_k_pos else (0.6 * auc + 0.4 * pak)
    return {"score": score, "auc": auc, "precision_at_k": pak, "model": model}


def _train_eval(
    df: pd.DataFrame,
    features: List[str],
    xgb_params: Dict,
    target: str,
    val_ratio: float,
    top_k_pct: float,
    min_top_k_pos: int,
    scale_pos_weight: float = 1.0,
    n_cv_folds: int = 1,
    split_spec: Optional[Dict] = None,
) -> Dict:
    """
    时间序列 CV 训练评估（n_cv_folds>1 时对多个时间窗口取均值）。

    评分：score = 0.6 × AUC + 0.4 × Precision@K（AUC主导，保证开仓勇气）
    CV 的意义：防止 Optuna 过拟合到单一验证期的市场行情，泛化更好。
    T8: 有 split_spec 时使用 frozen 日期边界切分；去重键改为 sample_id。
    """
    # T8: dedup by sample_id if available
    _dedup_key = ["sample_id"] if "sample_id" in df.columns else ["stock_code", "trade_date"]
    df_s = (
        df.dropna(subset=[target])
        .drop_duplicates(subset=_dedup_key)
        .sort_values("trade_date")
        .reset_index(drop=True)
    )
    n = len(df_s)
    if n < 100:
        return {"score": 0.0, "precision_at_k": 0.0, "auc": 0.5, "n_features": len(features)}

    X_all = df_s[features].fillna(0).replace([np.inf, -np.inf], 0)
    y_all = df_s[target].astype(int).values

    # T8: frozen split spec 优先（单折，日期边界冻结）
    if split_spec and n_cv_folds <= 1:
        from learnEngine.split_spec import apply_split_spec
        tr_df, val_df = apply_split_spec(df_s, split_spec)
        tr_idx  = np.array(tr_df.index)
        val_idx = np.array(val_df.index)
        folds = [(tr_idx, val_idx)]
        _use_spec_fold = True
    elif n_cv_folds <= 1:
        split = int(n * (1 - val_ratio))
        folds = [(slice(None, split), slice(split, None))]
        _use_spec_fold = False
    else:
        _use_spec_fold = False
        # 扩展窗口 CV：最后一折 = 实际部署折（训练80%，验证最后20%）
        # 前 n-1 折覆盖更早的时间段，让 Optuna 感知跨时间段泛化能力
        # 例如 val_ratio=0.2, n_cv_folds=3:
        #   fold0: train [0, 60%), val [60%, 80%)  ← 早期泛化
        #   fold1: train [0, 70%), val [70%, 90%)  ← 中期泛化（若数据够）
        #   fold2: train [0, 80%), val [80%, 100%) ← 实际部署折（必须包含）
        folds = []
        for i in range(n_cv_folds):
            train_end = 1.0 - val_ratio * (n_cv_folds - i) / n_cv_folds
            val_start = train_end
            val_end   = min(1.0, val_start + val_ratio)
            folds.append((
                slice(None, int(n * train_end)),
                slice(int(n * val_start), int(n * val_end)),
            ))

    scores, aucs, paks = [], [], []
    last_model = None
    for tr_sl, val_sl in folds:
        # T8: spec 折使用 .loc（整数 index 数组），其余用 .iloc（slice）
        if _use_spec_fold:
            X_tr, X_val = X_all.loc[tr_sl], X_all.loc[val_sl]
            y_tr, y_val = y_all[tr_sl], y_all[val_sl]
        else:
            X_tr, X_val = X_all.iloc[tr_sl], X_all.iloc[val_sl]
            y_tr, y_val = y_all[tr_sl], y_all[val_sl]
        if len(X_tr) < 50 or len(X_val) < 20:
            continue
        res = _single_fold_eval(X_tr, X_val, y_tr, y_val,
                                xgb_params, top_k_pct, min_top_k_pos, scale_pos_weight)
        scores.append(res["score"])
        aucs.append(res["auc"])
        paks.append(res["precision_at_k"])
        last_model = res["model"]

    if not scores:
        return {"score": 0.0, "precision_at_k": 0.0, "auc": 0.5, "n_features": len(features)}

    return {
        "score":           float(np.mean(scores)),
        "precision_at_k":  float(np.mean(paks)),
        "auc":             float(np.mean(aucs)),
        "n_features":      len(features),
        "top_k_positives": min_top_k_pos + 1,
        "model":           last_model,
    }


# ═══════════════════════════════════════════════════════════════════════
# FactorSelector
# ═══════════════════════════════════════════════════════════════════════

class FactorSelector:
    """
    三阶段因子筛选器。

    所有阈值和超参范围默认从 train_config 读取，支持构造时单独覆盖，
    方便在 notebook 或脚本中做参数实验。
    """

    def __init__(
        self,
        csv_path:         Optional[str]   = None,
        target:           Optional[str]   = None,
        val_ratio:        Optional[float] = None,
        # T8: 按 strategy_id 过滤（None = 使用全部数据）
        strategy_id:      Optional[str]   = None,
        # T8: frozen split spec 路径（None = 降级用 val_ratio 按行切）
        split_spec_path:  Optional[str]   = None,
        # Stage 1
        ic_min_icir:      Optional[float] = None,
        ic_min_win_rate:  Optional[float] = None,
        ic_max_pvalue:    Optional[float] = None,
        # Stage 2
        corr_max:         Optional[float] = None,
        # Stage 3
        n_trials:         Optional[int]   = None,
        timeout:          Optional[int]   = None,
        top_k_pct:        Optional[float] = None,
        min_top_k_pos:    Optional[int]   = None,
        min_features:     Optional[int]   = None,
        n_cv_folds:       Optional[int]   = None,
    ):
        def _v(val, default):
            return val if val is not None else default

        self.csv_path        = _v(csv_path,        cfg.TRAIN_CSV_PATH)
        self.target          = _v(target,           cfg.TARGET_LABEL)
        self.val_ratio       = _v(val_ratio,        cfg.VAL_RATIO)
        # T8
        self.strategy_id     = strategy_id if strategy_id is not None else getattr(cfg, "STRATEGY_ID", None)
        self.split_spec_path = _v(split_spec_path, cfg.SPLIT_SPEC_PATH)
        self.ic_min_icir     = _v(ic_min_icir,      cfg.IC_MIN_ICIR)
        self.ic_min_win_rate = _v(ic_min_win_rate,  cfg.IC_MIN_WIN_RATE)
        self.ic_max_pvalue   = _v(ic_max_pvalue,    cfg.IC_MAX_PVALUE)
        self.corr_max        = _v(corr_max,         cfg.CORR_MAX)
        self.n_trials        = _v(n_trials,         cfg.N_TRIALS)
        self.timeout         = _v(timeout,          cfg.TIMEOUT)
        self.top_k_pct       = _v(top_k_pct,        cfg.TOP_K_PCT)
        self.min_top_k_pos   = _v(min_top_k_pos,    cfg.MIN_TOP_K_POSITIVES)
        self.min_features    = _v(min_features,     cfg.MIN_FEATURES)
        self.n_cv_folds      = _v(n_cv_folds,       cfg.N_CV_FOLDS)

        self._df:           Optional[pd.DataFrame] = None
        self._all_features: List[str]              = []
        self._icir_map:     Dict[str, float]       = {}  # Stage 2 排序用
        self._split_spec:   Optional[Dict]         = None  # T8: frozen split

    # ─── 数据加载 ────────────────────────────────────────────────────────

    def _load(self) -> pd.DataFrame:
        """懒加载 CSV；T8 新增：strategy_id 过滤 + split spec 加载。"""
        if self._df is not None:
            return self._df

        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(
                f"训练集不存在: {self.csv_path}\n"
                "请先运行 python learnEngine/dataset.py 生成训练集"
            )

        logger.info(f"加载训练集: {self.csv_path}")
        df = pd.read_csv(self.csv_path)

        # 过滤 stk_factor_pro API 超时导致的中性值污染行
        if {'rsi_6', 'kdj_k', 'cci'}.issubset(df.columns):
            bad_mask = (df['rsi_6'] == 50.0) & (df['kdj_k'] == 50.0) & (df['cci'] == 0.0)
            if bad_mask.sum() > 0:
                logger.info(f"过滤 stk_factor_pro 中性值污染行: {bad_mask.sum()} 行")
                df = df[~bad_mask].reset_index(drop=True)

        # T8: 按 strategy_id 过滤（支持多策略训练池）
        if self.strategy_id and "strategy_id" in df.columns:
            before = len(df)
            df = df[df["strategy_id"] == self.strategy_id].reset_index(drop=True)
            logger.info(
                f"[T8] strategy_id 过滤: '{self.strategy_id}' | "
                f"{before} → {len(df)} 行"
            )
        elif self.strategy_id:
            logger.warning(
                f"[T8] strategy_id='{self.strategy_id}' 但训练集无 strategy_id 列，跳过过滤"
            )

        # T8: 加载 frozen split spec
        if self.split_spec_path and os.path.exists(self.split_spec_path):
            from learnEngine.split_spec import load_split_spec
            try:
                self._split_spec = load_split_spec(self.split_spec_path)
                logger.info(
                    f"[T8] Frozen split spec 加载: "
                    f"train ≤ {self._split_spec['train_end_date']} | "
                    f"val ≥ {self._split_spec['val_start_date']}"
                )
            except Exception as e:
                logger.warning(f"[T8] split spec 加载失败（降级用 val_ratio 切）: {e}")
                self._split_spec = None
        else:
            self._split_spec = None
            logger.info(f"[T8] split spec 不存在，将用 val_ratio={self.val_ratio} 按行切")

        # T8 + strategy_configs: 按策略 ID 动态计算有效排除列
        # → 自动排除其他策略的专属列（如 adapt_score 对非板块热度策略是无效列）
        from learnEngine.strategy_configs import get_effective_exclude_cols
        effective_exclude = get_effective_exclude_cols(self.strategy_id, cfg.EXCLUDE_COLS)

        self._all_features = [
            c for c in df.columns
            if c not in effective_exclude
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        logger.info(
            f"发现 {len(self._all_features)} 个因子列 "
            f"（CSV {len(df.columns)} 列 | 排除: {len(effective_exclude)} | "
            f"strategy_id={self.strategy_id or '全策略'}）"
        )
        self._df = df
        return df

    # ─── Stage 1: IC 过滤 ────────────────────────────────────────────────

    def stage1_ic_filter(self, features: List[str]) -> List[str]:
        """
        用 IC/ICIR 过滤明确无效因子。
        剔除条件：|ICIR| < ic_min_icir  AND  胜率 < ic_min_win_rate  AND  p > ic_max_pvalue
        三条件同时满足才剔除（宽松策略，避免误删有潜力因子）。
        """
        df = self._load()
        logger.info(f"[Stage 1] IC 过滤开始，输入 {len(features)} 个因子")

        try:
            from learnEngine.factor_ic import calc_factor_ic_report
            # 只传入候选因子列 + 标签列，避免不必要的计算
            cols_needed = ["trade_date"] + [c for c in features if c in df.columns] + [self.target]
            report = calc_factor_ic_report(df[cols_needed].copy(), return_col=self.target)
        except Exception as e:
            logger.warning(f"[Stage 1] factor_ic 调用失败（{e}），跳过 Stage 1，返回全部因子")
            return features

        # 适配不同可能的列名（factor_ic.py 会把列名重命名为 "icir(IC信息比率-综合评分)" 等带中文描述的形式）
        report = report.reset_index()
        factor_col = next(
            (c for c in report.columns if c.lower().startswith(("factor", "factor_name", "column", "feature"))),
            report.columns[0],
        )
        icir_col = next((c for c in report.columns if c.lower().startswith("icir")), None)
        wr_col   = next((c for c in report.columns if c.lower().startswith("win_rate")), None)
        pv_col   = next((c for c in report.columns if c.lower().startswith("p_value")), None)

        if icir_col is None:
            logger.warning("[Stage 1] 未找到 ICIR 列，跳过 Stage 1")
            return features

        # 记录 ICIR 绝对值，供 Stage 2 按质量排序
        for _, row in report.iterrows():
            fn = str(row[factor_col])
            self._icir_map[fn] = abs(float(row.get(icir_col, 0) or 0))

        # 三条件 AND 策略
        remove_set = set()
        for _, row in report.iterrows():
            fn   = str(row[factor_col])
            icir = abs(float(row.get(icir_col, 0) or 0))
            wr   = float(row.get(wr_col,  1.0) or 1.0) if wr_col else 1.0
            pv   = float(row.get(pv_col,  0.0) or 0.0) if pv_col else 0.0
            if icir < self.ic_min_icir and wr < self.ic_min_win_rate and pv > self.ic_max_pvalue:
                remove_set.add(fn)

        kept = [f for f in features if f not in remove_set]
        logger.info(f"[Stage 1] 剔除 {len(remove_set)} 个无效因子 → 剩余 {len(kept)} 个")
        return kept

    # ─── Stage 2: 相关性去重 ─────────────────────────────────────────────

    def stage2_corr_dedup(self, features: List[str]) -> List[str]:
        """
        贪心去除高度相关的冗余因子。
        按 |ICIR| 降序处理：两个因子高度相关时，保留 ICIR 更高的那个。
        """
        df = self._load()
        if len(features) <= 1:
            return features
        logger.info(f"[Stage 2] 相关性去重开始，输入 {len(features)} 个因子，阈值 |corr|>{self.corr_max}")

        # 计算相关系数矩阵（只用候选列）
        valid_feats = [f for f in features if f in df.columns]
        X     = df[valid_feats].fillna(0).replace([np.inf, -np.inf], 0)
        corr  = X.corr().abs()

        # 按 |ICIR| 降序排列（ICIR 未知的放末尾），先加入的优先保留
        sorted_feats = sorted(valid_feats, key=lambda f: self._icir_map.get(f, 0.0), reverse=True)

        selected = []
        for feat in sorted_feats:
            if not selected:
                selected.append(feat)
                continue
            # 与已选因子的最大相关系数
            max_corr = max(
                (corr.at[feat, s] for s in selected if feat in corr.index and s in corr.columns),
                default=0.0,
            )
            if max_corr <= self.corr_max:
                selected.append(feat)

        removed = len(features) - len(selected)
        logger.info(f"[Stage 2] 剔除 {removed} 个冗余因子 → 剩余 {len(selected)} 个")
        return selected

    # ─── Stage 3: Optuna 精调 ────────────────────────────────────────────

    def stage3_optuna(self, features: List[str]) -> Tuple[List[str], Dict, Dict]:
        """
        Optuna 联合搜索：因子子集（每个 on/off）+ XGBoost 超参。
        目标：Precision@K + AUC（纯盈利目标，无亏损惩罚参数）。
        返回：(selected_features, best_xgb_params, metrics_dict)
        """
        import optuna

        df = self._load()
        logger.info(
            f"[Stage 3] Optuna 搜索开始，候选 {len(features)} 个因子，"
            f"n_trials={self.n_trials}，timeout={self.timeout}s，"
            f"Precision@{self.top_k_pct:.0%}"
        )

        feat_list       = features  # 闭包引用
        top_k_pct       = self.top_k_pct
        min_top_k_pos   = self.min_top_k_pos
        min_features    = self.min_features
        val_ratio       = self.val_ratio
        target          = self.target
        n_cv_folds      = self.n_cv_folds
        split_spec      = self._split_spec  # T8: frozen spec（None = 降级用 val_ratio）

        # T8: scale_pos_weight 按训练集（frozen split train 部分）自然正负比计算
        if split_spec:
            from learnEngine.split_spec import apply_split_spec
            _df_train, _ = apply_split_spec(df.dropna(subset=[target]), split_spec)
        else:
            _df_train = df.dropna(subset=[target]).iloc[:int(len(df) * (1 - val_ratio))]
        _pos = int(_df_train[target].sum())
        _neg = int(len(_df_train) - _pos)
        _spw = min(round(_neg / max(_pos, 1), 3), 4.0)  # cap 4.0，防止过度激进
        logger.info(f"[Stage 3] scale_pos_weight={_spw:.2f}（neg={_neg}/pos={_pos}，已 cap 4.0）")

        def objective(trial: optuna.Trial) -> float:
            # ── 因子子集选择（每个因子独立 on/off）──
            selected = [f for f in feat_list if trial.suggest_categorical(f"f::{f}", [True, False])]
            if len(selected) < min_features:
                return 0.0

            # ── XGBoost 超参（不含 scale_pos_weight，由外部按数据比例固定）──
            params = {
                "max_depth":        trial.suggest_int(   "max_depth",        *cfg.XGB_MAX_DEPTH_RANGE),
                "learning_rate":    trial.suggest_float( "learning_rate",    *cfg.XGB_LR_RANGE),
                "n_estimators":     trial.suggest_int(   "n_estimators",     *cfg.XGB_N_EST_RANGE),
                "subsample":        trial.suggest_float( "subsample",        *cfg.XGB_SUBSAMPLE_RANGE),
                "colsample_bytree": trial.suggest_float( "colsample_bytree", *cfg.XGB_COLSAMPLE_RANGE),
                "min_child_weight": trial.suggest_int(   "min_child_weight", *cfg.XGB_MIN_CHILD_WEIGHT_RANGE),
                "gamma":            trial.suggest_float( "gamma",            *cfg.XGB_GAMMA_RANGE),
                "reg_alpha":        trial.suggest_float( "reg_alpha",        *cfg.XGB_REG_ALPHA_RANGE),
                "reg_lambda":       trial.suggest_float( "reg_lambda",       *cfg.XGB_REG_LAMBDA_RANGE),
            }

            result = _train_eval(df, selected, params, target, val_ratio, top_k_pct, min_top_k_pos,
                                 scale_pos_weight=_spw, n_cv_folds=n_cv_folds,
                                 split_spec=split_spec)  # T8
            return result["score"]

        def _progress_cb(study: optuna.Study, trial: optuna.Trial):
            if trial.value is None:
                return
            if (trial.number + 1) % 20 == 0 or trial.number == 0:
                logger.info(
                    f"[Stage 3] Trial {trial.number + 1}/{self.n_trials} | "
                    f"best={study.best_value:.4f} | "
                    f"this={trial.value:.4f}"
                )

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=30),
        )
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            callbacks=[_progress_cb],
        )

        best = study.best_trial
        best_features = [f for f in feat_list if best.params.get(f"f::{f}", False)]
        best_xgb      = {k: v for k, v in best.params.items() if not k.startswith("f::")}

        # 用最优配置重新完整评估一次（单折，与部署时评估口径一致）
        final = _train_eval(
            df, best_features, best_xgb, target, val_ratio, top_k_pct, min_top_k_pos,
            scale_pos_weight=_spw, n_cv_folds=1,
            split_spec=split_spec,  # T8
        )

        logger.info(
            f"[Stage 3] 完成！{len(best_features)} 个因子 | "
            f"Precision@{top_k_pct:.0%}={final['precision_at_k']:.4f} | "
            f"AUC={final['auc']:.4f} | Score={final['score']:.4f}"
        )
        metrics = {
            "precision_at_k":    final["precision_at_k"],
            "auc":               final["auc"],
            "score":             final["score"],
            "k_pct":             top_k_pct,
            "n_trials_done":     len(study.trials),
            "best_trial_no":     best.number,
            "scale_pos_weight":  _spw,   # 数据自然正负比，训练时直接使用
        }
        return best_features, best_xgb, metrics

    # ─── 主流程 ──────────────────────────────────────────────────────────

    def run(self, stage: str = "all", output_path: Optional[str] = None) -> Dict:
        """
        运行三阶段筛选流程并保存结果。

        :param stage:
            "all"  — 完整三阶段（推荐）
            "12"   — 只跑 Stage 1+2（快速查看哪些因子被剔除，不运行 Optuna）
            "3"    — 只跑 Stage 3（复用已有 Stage 1+2 结果，适合重调超参）
        :param output_path: 结果 JSON 路径，默认 cfg.SELECTED_FEATURES_PATH
        """
        output_path = output_path or cfg.get_selected_features_path(self.strategy_id)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        self._load()  # 触发懒加载，确认文件存在
        features   = list(self._all_features)
        s1_removed: List[str] = []
        s2_removed: List[str] = []

        if stage in ("all", "12"):
            s1_in     = len(features)
            features  = self.stage1_ic_filter(features)
            s1_removed = [f for f in self._all_features if f not in set(features)]

            features  = self.stage2_corr_dedup(features)
            s2_removed = [
                f for f in self._all_features
                if f not in set(features) and f not in set(s1_removed)
            ]

        elif stage == "3":
            # 尝试从已有 JSON 恢复 Stage 1+2 的剔除列表
            if os.path.exists(output_path):
                with open(output_path, encoding="utf-8") as f:
                    prev = json.load(f)
                s1_removed = prev.get("stage1_removed", [])
                s2_removed = prev.get("stage2_removed", [])
                features   = [
                    f for f in self._all_features
                    if f not in set(s1_removed) and f not in set(s2_removed)
                ]
                logger.info(
                    f"[Stage 3] 复用上次 Stage 1+2 结果，候选 {len(features)} 个因子"
                )
            else:
                logger.warning("[Stage 3] 未找到上次结果，对全部因子直接运行 Stage 3")

        # T8: 公共元数据（写入 JSON，供 train.py 读取时识别对应策略和 split）
        _common = {
            "dataset_dir":       os.path.dirname(os.path.abspath(self.csv_path)),
            "csv_path":          os.path.abspath(self.csv_path),
            "strategy_id":       self.strategy_id,   # T8: 策略标识（None = 全策略池）
            "split_spec_path":   self.split_spec_path,
            "split_train_end":   (self._split_spec or {}).get("train_end_date"),
            "split_val_start":   (self._split_spec or {}).get("val_start_date"),
        }

        if stage == "12":
            # 不运行 Optuna，只输出筛选结果
            result = {
                **_common,
                "stage1_input_count": len(self._all_features),
                "stage1_removed":     s1_removed,
                "stage2_input_count": len(self._all_features) - len(s1_removed),
                "stage2_removed":     s2_removed,
                "stage3_input_count": len(features),
                "stage3_candidates":  features,
                "feature_count":      len(features),
                "xgb_params":         {},
                "metrics":            {},
                "selected_features":  features,  # Stage 1+2 后的候选，未经 Optuna 精调
                "target_label":       self.target,
                "search_timestamp":   datetime.now().isoformat(timespec="seconds"),
                "note":               "stage=12，未运行 Optuna。selected_features 为 Stage 1+2 后候选，非最终选择。",
            }
        else:
            best_features, best_xgb, metrics = self.stage3_optuna(features)
            result = {
                **_common,
                "stage1_input_count": len(self._all_features),
                "stage1_removed":     s1_removed,
                "stage2_input_count": len(self._all_features) - len(s1_removed),
                "stage2_removed":     s2_removed,
                "stage3_input_count": len(features),
                "selected_features":  best_features,
                "feature_count":      len(best_features),
                "xgb_params":         best_xgb,
                "metrics":            metrics,
                "target_label":       self.target,
                "search_timestamp":   datetime.now().isoformat(timespec="seconds"),
            }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        logger.info(f"结果已保存: {output_path}")
        metrics = result.get("metrics", {})
        logger.info(
            f"\n{'='*60}\n"
            f"  Stage 1 剔除: {len(s1_removed)} 个（无效因子）\n"
            f"  Stage 2 剔除: {len(s2_removed)} 个（冗余因子）\n"
            f"  Stage 3 入选: {result['feature_count']} 个因子\n"
            + (
                f"  Precision@{metrics.get('k_pct', 0):.0%} = {metrics.get('precision_at_k', 0):.4f}\n"
                f"  AUC           = {metrics.get('auc', 0):.4f}\n"
                if metrics else ""
            )
            + f"  选中因子: {result['selected_features']}\n"
            f"{'='*60}"
        )
        return result


# ═══════════════════════════════════════════════════════════════════════
# CLI 入口
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="三阶段因子筛选器（IC过滤 → 相关性去重 → Optuna精调）",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--csv",         default=None, help="训练集 CSV 路径（默认 train_config.TRAIN_CSV_PATH）")
    parser.add_argument("--output",      default=None, help="输出 JSON 路径（默认 search_results/selected_features.json）")
    parser.add_argument("--n-trials",    type=int,   default=None, help="Optuna 试验次数（默认 200）")
    parser.add_argument("--timeout",     type=int,   default=None, help="超时秒数（默认 3600）")
    parser.add_argument("--top-k",       type=float, default=None, help="Precision@K 的 K 值，如 0.3（默认 0.30）")
    parser.add_argument("--strategy-id", default=None, help="策略 ID 过滤（如 sector_heat）；None = 使用全策略训练池")  # T8
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "12", "3"],
        help=(
            "运行阶段：\n"
            "  all — 完整三阶段（推荐）\n"
            "  12  — 只跑 Stage 1+2（查看因子剔除情况）\n"
            "  3   — 只跑 Stage 3（复用已有 Stage 1+2 结果）"
        ),
    )
    args = parser.parse_args()

    selector = FactorSelector(
        csv_path    = args.csv,
        n_trials    = args.n_trials,
        timeout     = args.timeout,
        top_k_pct   = args.top_k,
        strategy_id = args.strategy_id,  # T8
    )
    selector.run(stage=args.stage, output_path=args.output)
