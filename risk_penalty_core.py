"""
risk_penalty_core.py — 全链路亏损惩罚核心底层模块
====================================================
定位：整个系统所有模型、所有模块的唯一亏损惩罚与风险控制规则中枢。
      一处定义、全链路复用、零侵入现有代码、高兼容、高可拓展。

架构分层（与未来四层大架构对齐）：
  Module 1 │ RiskPenaltyConfig          — 全局统一风险配置（预设 + 自定义）
  Module 2 │ generate_sample_weights    — 训练时样本权重生成（现阶段核心落地）
           │ train_with_risk_penalty    — 零侵入训练包装器（直接替换 model.train 调用）
  Module 3 │ calc_strategy_weight_discount  ─╮ 实盘实时风险规则
           │ should_filter_high_risk_stock   ─┤ [预留未来大架构：Regime->策略权重分配层]
           │ should_stop_loss                ─╯
  Module 4 │ check_no_future_data       — 无未来函数合规校验（独立可调用）
           │ normalize_weights          — 权重归一化与异常值处理
           │ _validate_config           — 配置参数合法性校验（内部工具）

快速接入（现阶段落地）：
    # train.py 中替换原有 xgb_model.train() 调用：
    from learnEngine.risk_penalty_core import train_with_risk_penalty, RiskPenaltyConfig
    train_with_risk_penalty(xgb_model, X_train, X_val, y_train, y_val, feature_cols, df_train)

时序合规约定（本项目统一口径）：
    trade_date   = 信号日（D），特征基于 D 及 D-N 历史数据（_d0.._d4）
    label1/label2 = D+1 日收益（买入价=D+1开盘，卖出价=D+1收盘）
    训练集切分   = 按 trade_date 时间序列切分，验证集严格晚于训练集（train.py 已实现）
    本模块权重   = 仅在训练集分区上生成，绝不跨越切分点
"""

import os
import sys
# 文件位于项目根目录，确保同级模块（utils/）可正常导入
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.log_utils import logger


# ============================================================
# MODULE 1: 全局风险配置类
# ============================================================

@dataclass
class RiskPenaltyConfig:
    """
    全链路风险惩罚统一配置类。
    所有亏损惩罚、风险控制参数全部收拢于此，一处修改全链路生效。

    三种使用方式：
        1. 直接使用预设配置（推荐）：
               config = RiskPenaltyConfig.for_strategy_model()
               config = RiskPenaltyConfig.for_regime_model()
               config = RiskPenaltyConfig.default()

        2. 自定义参数（不改类内部代码，仅传参）：
               config = RiskPenaltyConfig(loss_weight_multiplier=4.0, open_15min_stop_loss_pct=-0.025)

        3. 在预设基础上微调：
               config = RiskPenaltyConfig.for_strategy_model()
               config.loss_weight_multiplier = 4.0  # 覆盖单个参数
    """

    # ── 样本权重生成（训练时）─────────────────────────────────
    loss_weight_multiplier: float = 3.0
    """
    亏损样本（label=0）基础权重倍数。
    当 label_raw_return 列不存在（旧数据）时，所有亏损样本统一使用此倍数。
    当 label_raw_return 列存在时，此值作为分级惩罚的「基准乘子」，
    最终权重 = loss_weight_multiplier × 分级系数（见 loss_severity_tiers）。
    """

    win_weight_base: float = 1.0
    """盈利样本（label=1）基础权重。通常保持1.0，与 loss_weight_multiplier 形成相对比。"""

    high_risk_env_extra_multiplier: float = 1.5
    """
    高风险市场环境下，亏损样本额外惩罚系数（叠加在分级权重上相乘）。
    最终亏损权重 = loss_weight_multiplier × 分级系数 × high_risk_env_extra_multiplier（高风险时）
    """

    # ── 亏损分级惩罚（需要 label_raw_return 列）──────────────────
    # 格式：[(return_threshold, severity_multiplier), ...]，按 threshold 从小到大排列
    # 匹配规则：raw_return < threshold 时使用对应 multiplier，取最严重（最小threshold）匹配
    # 示例默认值含义：
    #   raw_return < -0.07 → ×2.5（跌停级，-7%以下，最重惩罚）
    #   raw_return < -0.03 → ×1.8（重亏，-3%~-7%）
    #   raw_return < 0.00  → ×1.2（小亏，0~-3%）
    #   raw_return < 0.03  → ×0.8（错过目标但未亏，0~3%，轻微惩罚）
    #   raw_return >= 0.03 → label1=1，走盈利分支，不在此处
    loss_severity_tiers: tuple = (
        (-0.07, 2.5),   # 跌停级：-7% 以下（实际跌停约-10%），最高惩罚
        (-0.03, 1.8),   # 重亏：-3% ~ -7%
        ( 0.00, 1.2),   # 小亏：0% ~ -3%
        ( 0.03, 0.8),   # 错过目标：0% ~ 3%，未亏但未达目标，轻微降权
    )
    """
    亏损分级惩罚配置，需配合训练集中的 label_raw_return 列使用。
    若该列不存在，自动回退到 loss_weight_multiplier 统一惩罚。

    自定义示例（更激进的惩罚）：
        config = RiskPenaltyConfig.for_strategy_model()
        config.loss_severity_tiers = (
            (-0.09, 3.0),   # 接近跌停（-9%以下）
            (-0.05, 2.0),   # 重亏（-5%~-9%）
            (-0.02, 1.3),   # 小亏（-2%~-5%）
            ( 0.03, 0.9),   # 错过目标
        )
    """

    # ── 高风险市场环境判定阈值（来自训练集宏观特征列）────────────
    high_risk_limit_down_threshold: int = 50
    """
    market_limit_down_count（跌停家数）超过此值时，判定为高风险市场环境。
    数据来源：训练集中的 market_limit_down_count 列。
    """

    high_risk_index_drop_threshold: float = -0.02
    """
    index_sh_pct_chg（上证指数涨跌幅）低于此值时，叠加高风险判定。
    满足 limit_down OR index_drop 任一即视为高风险环境。
    数据来源：训练集中的 index_sh_pct_chg 列（train.py 的 EXCLUDE_PATTERNS 中包含 index_*，
    若该列被排除则自动跳过此判定，不影响权重生成）。
    """

    # ── 实盘硬止损（推断/策略时）─────────────────────────────────
    open_15min_stop_loss_pct: float = -0.03
    """
    开盘15分钟涨跌幅低于此值时触发硬止损，锁死单笔最大亏损。
    对应 should_stop_loss() 函数的默认阈值。
    """

    # ── 流动性门槛（实盘风险过滤）────────────────────────────────
    min_liquidity_amount: float = 10_000.0
    """
    最低流动性门槛（单位：千元），即1000万元。
    与 dataset.py 的 MIN_AMOUNT_THRESHOLD 对齐，高风险环境下过滤低于此值的个股。
    对应 should_filter_high_risk_stock() 函数参数。
    """

    # ── 近期策略表现折扣（[预留] 未来策略权重动态分配层）─────────
    recent_loss_window: int = 20
    """近期亏损率统计窗口（天），用于 calc_strategy_weight_discount()。"""

    recent_loss_high_threshold: float = 0.60
    """近期亏损率超过此值 -> 大幅折扣（策略严重失效信号）。"""

    recent_loss_medium_threshold: float = 0.45
    """近期亏损率超过此值（但低于 high）-> 中等折扣。"""

    recent_loss_high_discount: float = 0.3
    """高亏损率折扣系数：strategy_weight × 0.3。"""

    recent_loss_medium_discount: float = 0.6
    """中等亏损率折扣系数：strategy_weight × 0.6。"""

    # ================================================================
    # 预设配置工厂方法
    # ================================================================

    @classmethod
    def default(cls) -> "RiskPenaltyConfig":
        """
        基础通用配置（适合大多数场景的默认起点）。

        示例：
            config = RiskPenaltyConfig.default()
        """
        return cls()

    @classmethod
    def for_regime_model(cls) -> "RiskPenaltyConfig":
        """
        Regime 市场状态模型专用配置。
        核心逻辑：重点惩罚「把熊市误判为牛市」的致命错误。
        熊市时所有买入均直接亏损，误判代价极高，因此亏损惩罚大幅提升。

        [预留未来大架构：Regime 预判层 -> 策略权重动态分配层]

        示例：
            config = RiskPenaltyConfig.for_regime_model()
            weights = generate_sample_weights(df_train, config=config)
        """
        return cls(
            loss_weight_multiplier=5.0,           # 熊市误判惩罚极高
            win_weight_base=1.0,
            high_risk_env_extra_multiplier=2.0,   # 极端市场环境下再翻倍
            high_risk_limit_down_threshold=30,    # 更敏感：30家跌停即触发
            high_risk_index_drop_threshold=-0.015, # 更敏感：-1.5% 即触发
            open_15min_stop_loss_pct=-0.02,        # 更保守止损线
            recent_loss_high_threshold=0.50,
            recent_loss_medium_threshold=0.35,
            recent_loss_high_discount=0.2,         # 失效时大幅压仓
            recent_loss_medium_discount=0.5,
        )

    @classmethod
    def for_strategy_model(cls) -> "RiskPenaltyConfig":
        """
        Strategy 选股模型专用配置（当前 sector_heat_strategy 对应此配置）。
        核心逻辑：重点惩罚「高风险环境下的错误买入」和「下跌亏损样本」。

        示例：
            config = RiskPenaltyConfig.for_strategy_model()
            train_with_risk_penalty(model, X_train, X_val, y_train, y_val, cols, df_train, config=config)
        """
        return cls(
            loss_weight_multiplier=3.0,
            win_weight_base=1.0,
            high_risk_env_extra_multiplier=1.5,
            high_risk_limit_down_threshold=50,
            high_risk_index_drop_threshold=-0.02,
            open_15min_stop_loss_pct=-0.03,
            recent_loss_high_threshold=0.60,
            recent_loss_medium_threshold=0.45,
            recent_loss_high_discount=0.3,
            recent_loss_medium_discount=0.6,
        )

    def validate(self) -> None:
        """参数合法性校验，不合法时抛出 ValueError（含明确错误描述）。"""
        _validate_config(self)


# ============================================================
# MODULE 2: 样本权重生成 + 零侵入训练包装器
# ============================================================

# 训练集中宏观特征列名（与 dataset.py / train_dataset_final.csv 对齐）
_LIMIT_DOWN_COL = "market_limit_down_count"
_INDEX_CHG_COL  = "index_sh_pct_chg"


def generate_sample_weights(
    df: pd.DataFrame,
    label_col: str = "label1",
    date_col: str = "trade_date",
    config: Optional[RiskPenaltyConfig] = None,
    normalize: bool = True,
    clip_max_multiplier: float = 10.0,
) -> np.ndarray:
    """
    根据标签和市场环境生成样本权重，直接传入 XGBoost/sklearn 的 sample_weight 参数。

    权重逻辑（三层叠加）：
        1. 基础权重：盈利样本 = win_weight_base，亏损样本 = win_weight_base × loss_weight_multiplier
        2. 市场环境加成：亏损样本处于高风险市场时，额外乘以 high_risk_env_extra_multiplier
           高风险判定（满足任一）：
             - market_limit_down_count > high_risk_limit_down_threshold
             - index_sh_pct_chg < high_risk_index_drop_threshold
           注意：若宏观特征列被 train.py EXCLUDE_PATTERNS 排除，则此层自动跳过，不影响结果
        3. 归一化：均值归一化为 1.0（可选），避免整体梯度量级变化

    [!] 时序合规强制检查：
        - 自动校验训练集 trade_date 是否可解析为日期
        - 自动检测是否包含未来数据特征列（命名含 d-1 等负d索引）
        - 违规时抛出 RuntimeError，禁止生成权重

    :param df:                   训练集 DataFrame（仅传训练集分区，不含验证集！）
    :param label_col:            标签列名，必须是 0/1 二值（默认 "label1"）
    :param date_col:             交易日列名（用于合规校验，默认 "trade_date"）
    :param config:               RiskPenaltyConfig 实例，None 时使用 for_strategy_model() 预设
    :param normalize:            是否均值归一化（默认 True，保持梯度量级稳定）
    :param clip_max_multiplier:  权重最大截断倍数（相对均值），防止极端权重主导训练
    :return:                     1D np.ndarray，shape=(len(df),)，可直接传入 sample_weight

    使用示例：
        # 最简接入（使用预设配置，与 train.py 的数据流对齐）：
        from learnEngine.risk_penalty_core import generate_sample_weights

        # df_train = df.iloc[sort_idx[:split_point]]  ← 与 train.py 的时序切分对齐
        weights = generate_sample_weights(df_train, label_col="label1")

        # 传入 XGBoost fit：
        model.fit(X_train, y_train, sample_weight=weights, eval_set=[(X_val, y_val)])

        # 自定义配置：
        from learnEngine.risk_penalty_core import RiskPenaltyConfig
        config = RiskPenaltyConfig(loss_weight_multiplier=4.0)
        weights = generate_sample_weights(df_train, config=config)
    """
    if config is None:
        config = RiskPenaltyConfig.for_strategy_model()
    config.validate()

    # ── 合规校验：禁止传入带未来数据的 DataFrame ─────────────────
    _check_no_future_data_internal(df, date_col)

    # ── 标签列校验 ───────────────────────────────────────────────
    if label_col not in df.columns:
        raise ValueError(
            f"[RiskPenalty] 标签列 '{label_col}' 不在 DataFrame 中。"
            f"可用列: {df.columns.tolist()[:10]}..."
        )
    labels = df[label_col].values
    if not set(np.unique(labels[~np.isnan(labels)])).issubset({0, 1, 0.0, 1.0}):
        raise ValueError(
            f"[RiskPenalty] 标签列 '{label_col}' 必须是 0/1 二值标签，"
            f"检测到: {np.unique(labels[~np.isnan(labels)])}"
        )

    n = len(df)
    weights = np.ones(n, dtype=float)

    # ── 检测宏观特征列是否可用（train.py 可能通过 EXCLUDE_PATTERNS 排除）──
    has_limit_down = _LIMIT_DOWN_COL in df.columns
    has_index_chg  = _INDEX_CHG_COL  in df.columns
    use_macro_env  = has_limit_down or has_index_chg

    if not use_macro_env:
        logger.warning(
            f"[RiskPenalty] 宏观特征列 {_LIMIT_DOWN_COL} / {_INDEX_CHG_COL} 均不在 DataFrame 中"
            f"（可能被 EXCLUDE_PATTERNS 过滤），跳过高风险市场环境加成，仅使用基础权重。"
        )

    # ── 检测 label_raw_return 列是否可用 ─────────────────────────
    raw_return_col = "label_raw_return"
    has_raw_return = raw_return_col in df.columns and df[raw_return_col].notna().any()
    raw_returns    = df[raw_return_col].values if has_raw_return else None

    # 预排序分级阈值（从小到大），便于快速匹配
    sorted_tiers: list = sorted(config.loss_severity_tiers, key=lambda x: x[0])

    if has_raw_return:
        logger.info(f"[RiskPenalty] 检测到 label_raw_return，启用分级惩罚模式")
    else:
        logger.info(f"[RiskPenalty] 未检测到 label_raw_return，使用统一惩罚（loss_multiplier={config.loss_weight_multiplier}）")

    # ── 逐行生成权重 ─────────────────────────────────────────────
    loss_base_weight = config.win_weight_base * config.loss_weight_multiplier

    for i in range(n):
        label_val = labels[i]
        if np.isnan(label_val):
            weights[i] = config.win_weight_base   # NaN 标签按中性处理
            continue

        if int(label_val) == 1:
            # 盈利样本：基础权重（未来可在此按盈利幅度加成）
            weights[i] = config.win_weight_base
        else:
            # 亏损样本：分级惩罚 or 统一惩罚
            if has_raw_return:
                raw_ret = raw_returns[i]
                severity = _get_severity_multiplier(raw_ret, sorted_tiers)
                w = config.win_weight_base * config.loss_weight_multiplier * severity
            else:
                w = loss_base_weight

            # 市场环境加成（叠加在分级权重上）
            if use_macro_env and _is_high_risk_env(df.iloc[i], config, has_limit_down, has_index_chg):
                w *= config.high_risk_env_extra_multiplier

            weights[i] = w

    # ── 归一化与截断 ─────────────────────────────────────────────
    weights = normalize_weights(weights, clip_max_multiplier=clip_max_multiplier)
    if not normalize:
        pass  # 跳过均值归一化，仅保留截断后的绝对权重

    pos_count  = int((labels == 1).sum())
    loss_count = int((labels == 0).sum())

    # 分级统计摘要（仅在有原始收益时输出）
    if has_raw_return:
        tier_stats = []
        for thresh, mult in sorted_tiers:
            mask = (labels == 0) & (raw_returns < thresh)
            if tier_stats:
                prev_thresh = sorted_tiers[sorted_tiers.index((thresh, mult)) - 1][0]
                mask = (labels == 0) & (raw_returns >= prev_thresh) & (raw_returns < thresh)
            tier_stats.append(f"<{thresh*100:.0f}%: {mask.sum()}样本×{mult}")
        logger.info(f"[RiskPenalty] 分级分布 | " + " | ".join(tier_stats))

    logger.info(
        f"[RiskPenalty] 权重生成完成 | 样本数:{n} | 盈利:{pos_count} 亏损:{loss_count} | "
        f"权重范围:[{weights.min():.3f}, {weights.max():.3f}] | "
        f"权重均值:{weights.mean():.3f} | "
        f"模式:{'分级惩罚' if has_raw_return else '统一惩罚'} | "
        f"宏观加成:{'启用' if use_macro_env else '跳过'}"
    )
    return weights


def train_with_risk_penalty(
    model_instance,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    feature_cols: List[str],
    df_train: pd.DataFrame,
    label_col: str = "label1",
    config: Optional[RiskPenaltyConfig] = None,
) -> object:
    """
    零侵入训练包装器：不修改 SectorHeatXGBModel / model.py 任何代码，
    直接在外层生成 sample_weight 并接管 fit 调用，完全替代 model.train() 调用。

    与 model.train() 行为完全一致（动态 scale_pos_weight、early stopping、评估、保存），
    唯一区别是在 fit() 中注入了 sample_weight。

    :param model_instance: SectorHeatXGBModel 实例（未训练状态）
    :param X_train:        训练集特征 DataFrame（与 train.py 中的 X_train 一致）
    :param X_val:          验证集特征 DataFrame
    :param y_train:        训练集标签 numpy array
    :param y_val:          验证集标签 numpy array
    :param feature_cols:   特征列名列表
    :param df_train:       训练集原始 DataFrame（含 label_col 和宏观特征，用于生成权重）
                           [!] 必须仅包含训练集行（时序切分后的前80%），不含验证集！
    :param label_col:      标签列名（默认 "label1"）
    :param config:         风险配置实例，None 时使用 for_strategy_model() 预设
    :return:               训练完成的 XGBClassifier 模型对象

    使用示例（在 train.py 中替换原有的 xgb_model.train() 调用）：

        # ─── 原有代码（保留不变）─────────────────────────────────
        X, y, feature_cols, df = load_and_prepare(TRAIN_CSV_PATH, TARGET_LABEL)
        X_train, X_val, y_train, y_val = time_series_split(X, y, df, VAL_RATIO)
        xgb_model = SectorHeatXGBModel(model_save_path=MODEL_SAVE_PATH)

        # ─── 新增：替换下面这一行 ─────────────────────────────────
        # 原来：xgb_model.train(X_train, X_val, y_train, y_val, feature_cols)
        # 替换为：
        from learnEngine.risk_penalty_core import train_with_risk_penalty
        split_point = int(len(df) * (1 - VAL_RATIO))
        sort_idx = df["trade_date"].argsort().values
        df_train_partition = df.iloc[sort_idx[:split_point]]   # 时序训练集分区
        train_with_risk_penalty(
            xgb_model, X_train, X_val, y_train, y_val, feature_cols,
            df_train=df_train_partition,
            label_col=TARGET_LABEL,
        )
        # ─── 后续评估代码无需修改 ────────────────────────────────
    """
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, roc_auc_score

    if config is None:
        config = RiskPenaltyConfig.for_strategy_model()
    config.validate()

    # ── Step 1: 生成样本权重 ─────────────────────────────────────
    weights = generate_sample_weights(df_train, label_col=label_col, config=config)

    # ── Step 2: 动态计算 scale_pos_weight（复刻 model.py 逻辑）──
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    scale_pos_weight = round(neg / pos, 2) if pos > 0 else 1.0
    scale_pos_weight = min(scale_pos_weight, 4.0)
    logger.info(
        f"[RiskPenalty] 风险惩罚训练启动 | "
        f"loss_multiplier={config.loss_weight_multiplier} | "
        f"high_risk_extra={config.high_risk_env_extra_multiplier} | "
        f"scale_pos_weight={scale_pos_weight} | "
        f"样本数:训练={len(X_train)} 验证={len(X_val)}"
    )

    # ── Step 3: 初始化并训练（注入 sample_weight）──────────────
    params = {**model_instance.base_params, "scale_pos_weight": scale_pos_weight}
    model_instance.model = xgb.XGBClassifier(**params)
    model_instance.model.fit(
        X_train, y_train,
        sample_weight=weights,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    actual_trees = (
        model_instance.model.best_iteration + 1
        if hasattr(model_instance.model, "best_iteration") and model_instance.model.best_iteration is not None
        else params["n_estimators"]
    )
    logger.info(f"[RiskPenalty] 实际训练轮数: {actual_trees} / {params['n_estimators']}")

    # ── Step 4: 评估（与 model.py 保持一致的指标输出）────────────
    y_val_pred  = model_instance.model.predict(X_val)
    y_val_proba = model_instance.model.predict_proba(X_val)[:, 1]
    acc = accuracy_score(y_val, y_val_pred)
    auc = roc_auc_score(y_val, y_val_proba)
    logger.info("=" * 55)
    logger.info(f"[RiskPenalty] 模型训练完成（含亏损惩罚）")
    logger.info(f"[RiskPenalty] 验证集 AUC:      {auc:.4f}")
    logger.info(f"[RiskPenalty] 验证集 Accuracy: {acc:.2%}")
    logger.info("=" * 55)

    # ── Step 5: 保存模型 ─────────────────────────────────────────
    model_instance.save_model()
    return model_instance.model


# ============================================================
# MODULE 3: 实盘实时风险规则
# [预留未来大架构：Regime预判->策略权重动态分配->选股子模型->T+1动态持仓决策]
# ============================================================

def calc_strategy_weight_discount(
    recent_loss_rate: float,
    config: Optional[RiskPenaltyConfig] = None,
) -> float:
    """
    [预留未来大架构：策略权重动态分配层]
    根据策略近期亏损率计算权重折扣系数，用于动态降低失效策略的仓位权重。

    折扣逻辑（三档）：
        loss_rate >= high_threshold  -> 大幅折扣（策略严重失效，大幅压仓）
        loss_rate >= medium_threshold -> 中等折扣（策略表现偏弱，适度减仓）
        loss_rate < medium_threshold  -> 无折扣（正常运行）

    :param recent_loss_rate: 近期亏损率，范围 [0, 1]。
                             计算方式：近 N 个信号中亏损信号数 / 总信号数
                             （N = config.recent_loss_window，默认20天）
    :param config:           RiskPenaltyConfig 实例，None 时使用 for_strategy_model() 预设
    :return:                 折扣系数 (0, 1.0]，策略权重 × 折扣系数 = 实际分配权重

    使用示例：
        from learnEngine.risk_penalty_core import calc_strategy_weight_discount, RiskPenaltyConfig

        # 近20个信号中有13个亏损（65%亏损率）-> 大幅折扣
        discount = calc_strategy_weight_discount(recent_loss_rate=0.65)
        actual_weight = base_weight * discount  # base_weight × 0.3

        # 配合 sector_heat_strategy 的 min_prob 动态调整：
        config = RiskPenaltyConfig.for_strategy_model()
        discount = calc_strategy_weight_discount(recent_loss_rate, config=config)
        adjusted_min_prob = min(0.9, self.strategy_params["min_prob"] / discount)
    """
    if config is None:
        config = RiskPenaltyConfig.for_strategy_model()

    if not (0.0 <= recent_loss_rate <= 1.0):
        raise ValueError(
            f"[RiskPenalty] recent_loss_rate 必须在 [0, 1] 范围内，当前值: {recent_loss_rate}"
        )

    if recent_loss_rate >= config.recent_loss_high_threshold:
        discount = config.recent_loss_high_discount
        logger.warning(
            f"[RiskPenalty] 策略近期亏损率 {recent_loss_rate:.1%} ≥ "
            f"高阈值 {config.recent_loss_high_threshold:.1%}，"
            f"大幅折扣: ×{discount}"
        )
        return discount

    if recent_loss_rate >= config.recent_loss_medium_threshold:
        discount = config.recent_loss_medium_discount
        logger.info(
            f"[RiskPenalty] 策略近期亏损率 {recent_loss_rate:.1%} ≥ "
            f"中阈值 {config.recent_loss_medium_threshold:.1%}，"
            f"中等折扣: ×{discount}"
        )
        return discount

    return 1.0  # 正常运行，无折扣


def should_filter_high_risk_stock(
    market_limit_down_count: int,
    stock_amount: float,
    config: Optional[RiskPenaltyConfig] = None,
    market_index_chg: Optional[float] = None,
) -> bool:
    """
    [预留未来大架构：T+1动态持仓决策层]
    高风险个股过滤：在高风险市场环境下自动过滤低流动性个股。

    过滤逻辑：
        高风险市场（跌停家数 > threshold OR 指数跌幅 < threshold）
        AND 个股流动性 < min_liquidity_amount
        -> 过滤（返回 True 表示应过滤该股）

    :param market_limit_down_count: 当日市场跌停家数（来自 market_limit_down_count）
    :param stock_amount:            个股当日成交额（千元，与 dataset.py 口径一致）
    :param config:                  RiskPenaltyConfig 实例
    :param market_index_chg:        上证指数涨跌幅（可选，None 时不参与判断）
    :return:                        True = 应过滤此股，False = 可正常买入

    使用示例（在 sector_heat_strategy.py 的选股过滤阶段）：
        from learnEngine.risk_penalty_core import should_filter_high_risk_stock, RiskPenaltyConfig

        config = RiskPenaltyConfig.for_strategy_model()
        for ts_code, row in candidates.iterrows():
            if should_filter_high_risk_stock(
                market_limit_down_count=market_down_count,
                stock_amount=row["amount"],
                market_index_chg=sh_index_chg,
                config=config,
            ):
                continue  # 跳过此股
    """
    if config is None:
        config = RiskPenaltyConfig.for_strategy_model()

    # 判断是否高风险市场环境
    is_high_risk = market_limit_down_count > config.high_risk_limit_down_threshold
    if market_index_chg is not None:
        is_high_risk = is_high_risk or (market_index_chg < config.high_risk_index_drop_threshold)

    if not is_high_risk:
        return False  # 正常市场，不过滤

    # 高风险环境下，过滤低流动性个股
    return stock_amount < config.min_liquidity_amount


def should_stop_loss(
    open_15min_pct_chg: float,
    config: Optional[RiskPenaltyConfig] = None,
) -> bool:
    """
    [预留未来大架构：T+1开盘动态持仓决策层]
    单笔硬止损决策：开盘15分钟涨跌幅超过阈值时触发止损，锁死单笔最大亏损。

    :param open_15min_pct_chg: 开盘15分钟涨跌幅，如 -0.035 表示下跌3.5%
                                计算方式：(当前价 - 昨收) / 昨收
    :param config:             RiskPenaltyConfig 实例
    :return:                   True = 触发止损（立即卖出），False = 继续持有

    使用示例（在开盘15分钟后的执行层调用）：
        from learnEngine.risk_penalty_core import should_stop_loss

        pct_chg = (current_price - pre_close) / pre_close
        if should_stop_loss(pct_chg):
            execute_sell(ts_code, reason="risk_penalty_hard_stop")
    """
    if config is None:
        config = RiskPenaltyConfig.for_strategy_model()

    trigger = open_15min_pct_chg < config.open_15min_stop_loss_pct
    if trigger:
        logger.warning(
            f"[RiskPenalty] 硬止损触发 | 开盘15分钟涨跌幅={open_15min_pct_chg:.2%} < "
            f"阈值={config.open_15min_stop_loss_pct:.2%}"
        )
    return trigger


# ============================================================
# MODULE 4: 辅助工具模块
# ============================================================

def check_no_future_data(
    df: pd.DataFrame,
    feature_date_col: str = "trade_date",
    label_col: Optional[str] = "label1",
    feature_cols: Optional[List[str]] = None,
) -> Dict:
    """
    无未来函数合规校验（独立可调用）。
    检查训练集是否违反时序因果规范，输出详细校验报告。

    校验项目：
        1. 特征日期列（trade_date）是否可解析为日期格式
        2. 特征日期列是否单调递增（无时序混乱）
        3. 特征列名中是否包含负 d 索引（如 *_d-1 表示未来数据）
        4. 标签列是否为二值（0/1），不含前向数据命名
        5. 时序连续性：相邻日期跳跃是否合理（> 7天可能是数据缺失）

    :param df:               训练集 DataFrame
    :param feature_date_col: 特征日期列名（默认 "trade_date"，表示信号日 D）
    :param label_col:        标签列名（用于验证不含未来数据命名，可传 None 跳过）
    :param feature_cols:     待校验的特征列名列表，None 时自动检测所有非日期非标签列
    :return:                 校验报告 dict，含以下字段：
                               is_compliant   : bool，True 表示合规
                               warnings       : List[str]，警告信息（合规但建议检查）
                               errors         : List[str]，错误信息（不合规，必须修复）
                               date_range     : Tuple[str, str]，数据日期范围
                               sample_count   : int，样本总数

    使用示例：
        from learnEngine.risk_penalty_core import check_no_future_data
        import pandas as pd

        df = pd.read_csv("learnEngine/history/csv/train_dataset_final.csv")
        report = check_no_future_data(df, feature_date_col="trade_date", label_col="label1")

        if not report["is_compliant"]:
            for err in report["errors"]:
                print(f"[ERR] {err}")
            raise RuntimeError("训练集存在未来数据，禁止训练！")

        for warn in report["warnings"]:
            print(f"[!]  {warn}")
        print(f"[OK] 合规校验通过 | 日期范围: {report['date_range'][0]} ~ {report['date_range'][1]}")
    """
    errors   : List[str] = []
    warnings : List[str] = []
    date_range = ("N/A", "N/A")
    sample_count = len(df)

    # ── 校验1：日期列存在性 ─────────────────────────────────────
    if feature_date_col not in df.columns:
        errors.append(
            f"日期列 '{feature_date_col}' 不存在于 DataFrame 中。"
            f"可用列: {df.columns.tolist()[:10]}..."
        )
        return {
            "is_compliant": False, "errors": errors, "warnings": warnings,
            "date_range": date_range, "sample_count": sample_count,
        }

    # ── 校验2：日期列可解析性 ──────────────────────────────────
    try:
        parsed_dates = pd.to_datetime(df[feature_date_col], errors="coerce")
        invalid_count = parsed_dates.isna().sum()
        if invalid_count > 0:
            errors.append(
                f"日期列 '{feature_date_col}' 中有 {invalid_count} 个无法解析的日期值，"
                f"请检查数据格式（应为 YYYY-MM-DD 或 YYYYMMDD）。"
            )
        else:
            min_date = parsed_dates.min().strftime("%Y-%m-%d")
            max_date = parsed_dates.max().strftime("%Y-%m-%d")
            date_range = (min_date, max_date)
    except Exception as e:
        errors.append(f"日期列 '{feature_date_col}' 解析失败: {e}")

    # ── 校验3：日期单调性（时序混乱检测）──────────────────────
    if not errors:
        date_vals = pd.to_datetime(df[feature_date_col], errors="coerce").dropna()
        if not date_vals.is_monotonic_increasing:
            warnings.append(
                f"日期列 '{feature_date_col}' 不是单调递增（时序可能混乱）。"
                f"虽然 generate_sample_weights 不依赖顺序，但训练时请确保时序切分正确。"
            )

    # ── 校验4：特征列名中的负 d 索引（未来数据特征）─────────────
    check_cols = feature_cols if feature_cols is not None else df.columns.tolist()
    import re
    future_pattern = re.compile(r"_d-\d+")   # 匹配 _d-1, _d-2 等负索引
    future_cols = [c for c in check_cols if future_pattern.search(str(c))]
    if future_cols:
        errors.append(
            f"检测到含负 d 索引的特征列（未来数据）: {future_cols}。"
            f"本项目约定 _d0=当日, _d1~d4=历史，负索引表示未来，违反时序因果规范。"
        )

    # ── 校验5：标签列合法性 ──────────────────────────────────
    if label_col is not None and label_col in df.columns:
        unique_vals = set(df[label_col].dropna().unique())
        if not unique_vals.issubset({0, 1, 0.0, 1.0}):
            warnings.append(
                f"标签列 '{label_col}' 包含非 0/1 值: {unique_vals}。"
                f"若为连续收益率，generate_sample_weights 将仍以是否 > 0 判断盈亏，请确认。"
            )
    elif label_col is not None:
        warnings.append(f"标签列 '{label_col}' 不在 DataFrame 中，已跳过标签校验。")

    # ── 校验6：时序连续性（大跨度日期跳跃检测）──────────────────
    if not errors and len(date_vals) > 1:
        date_sorted = date_vals.sort_values()
        gaps = (date_sorted.diff().dt.days.dropna() > 7)
        large_gap_count = gaps.sum()
        if large_gap_count > 5:
            warnings.append(
                f"日期序列中有 {large_gap_count} 处超过7天的跳跃，"
                f"可能存在较长数据缺失期（节假日+停盘），建议核查数据完整性。"
            )

    is_compliant = len(errors) == 0
    if is_compliant:
        logger.info(
            f"[RiskPenalty] 合规校验通过 [OK] | "
            f"日期范围:{date_range[0]}~{date_range[1]} | 样本:{sample_count}"
        )
    else:
        logger.error(
            f"[RiskPenalty] 合规校验失败 ✗ | "
            f"错误数:{len(errors)} | 首错:{errors[0]}"
        )

    return {
        "is_compliant":  is_compliant,
        "errors":        errors,
        "warnings":      warnings,
        "date_range":    date_range,
        "sample_count":  sample_count,
    }


def normalize_weights(
    weights: np.ndarray,
    clip_max_multiplier: float = 10.0,
) -> np.ndarray:
    """
    权重归一化与异常值处理：先截断极端权重，再均值归一化为 1.0。

    :param weights:            原始权重 1D numpy array
    :param clip_max_multiplier: 截断阈值 = mean × clip_max_multiplier，防止极端权重主导训练
    :return:                   归一化后的权重 1D numpy array，均值 ≈ 1.0

    使用示例：
        weights = np.array([1.0, 3.0, 4.5, 50.0, 1.0])
        normalized = normalize_weights(weights, clip_max_multiplier=5.0)
        # 50.0 被截断，均值归一化为1.0
    """
    weights = np.asarray(weights, dtype=float)
    if len(weights) == 0:
        return weights

    # 截断极端值
    mean_w = weights.mean()
    if mean_w > 0:
        clip_threshold = mean_w * clip_max_multiplier
        clipped_count  = int((weights > clip_threshold).sum())
        if clipped_count > 0:
            weights = np.clip(weights, 0, clip_threshold)
            logger.debug(f"[RiskPenalty] 截断极端权重 {clipped_count} 个（阈值={clip_threshold:.3f}）")

    # 均值归一化：均值归一化为 1.0，保持梯度量级稳定
    mean_w = weights.mean()
    if mean_w > 0:
        weights = weights / mean_w

    return weights


# ============================================================
# 内部工具函数（非公开接口）
# ============================================================

def _validate_config(config: RiskPenaltyConfig) -> None:
    """
    配置参数合法性校验，不合法时抛出 ValueError（含明确字段名和建议值）。
    内部被 config.validate() 和 generate_sample_weights 调用。
    """
    errors = []

    if config.loss_weight_multiplier < 1.0:
        errors.append(
            f"loss_weight_multiplier={config.loss_weight_multiplier} 不合法，"
            f"必须 ≥ 1.0（等于1.0表示不惩罚，建议范围: 2.0~6.0）"
        )

    if config.win_weight_base <= 0:
        errors.append(
            f"win_weight_base={config.win_weight_base} 不合法，必须 > 0（建议值: 1.0）"
        )

    if config.high_risk_env_extra_multiplier < 1.0:
        errors.append(
            f"high_risk_env_extra_multiplier={config.high_risk_env_extra_multiplier} 不合法，"
            f"必须 ≥ 1.0（等于1.0表示不额外加成）"
        )

    if not (-0.20 <= config.open_15min_stop_loss_pct < 0):
        errors.append(
            f"open_15min_stop_loss_pct={config.open_15min_stop_loss_pct} 不合法，"
            f"必须在 [-0.20, 0) 范围内（如 -0.03 表示 -3%）"
        )

    if not (0 < config.recent_loss_medium_threshold < config.recent_loss_high_threshold <= 1.0):
        errors.append(
            f"亏损率阈值配置不合法：medium={config.recent_loss_medium_threshold} "
            f"high={config.recent_loss_high_threshold}，"
            f"必须满足 0 < medium < high ≤ 1.0"
        )

    if not (0 < config.recent_loss_high_discount <= config.recent_loss_medium_discount <= 1.0):
        errors.append(
            f"折扣系数配置不合法：high_discount={config.recent_loss_high_discount} "
            f"medium_discount={config.recent_loss_medium_discount}，"
            f"必须满足 0 < high_discount ≤ medium_discount ≤ 1.0"
        )

    if errors:
        raise ValueError(
            f"[RiskPenalty] RiskPenaltyConfig 参数校验失败，共 {len(errors)} 个问题：\n"
            + "\n".join(f"  {i+1}. {e}" for i, e in enumerate(errors))
        )


def _get_severity_multiplier(raw_return: float, sorted_tiers: list) -> float:
    """
    内部：根据实际收益率查找对应的分级惩罚系数。
    tiers 已按 threshold 从小到大排序，取第一个满足 raw_return < threshold 的系数。
    全部不满足（收益率高于所有 threshold）时返回 1.0（无额外惩罚，属于盈利区间）。
    """
    if np.isnan(raw_return):
        return 1.0
    for threshold, multiplier in sorted_tiers:
        if raw_return < threshold:
            return multiplier
    return 1.0


def _is_high_risk_env(
    row: pd.Series,
    config: RiskPenaltyConfig,
    has_limit_down: bool,
    has_index_chg: bool,
) -> bool:
    """
    内部：判断单行样本是否处于高风险市场环境。
    满足任一条件即视为高风险：跌停家数超阈值 OR 指数跌幅超阈值。
    """
    if has_limit_down:
        limit_down_val = row.get(_LIMIT_DOWN_COL, 0)
        if pd.notna(limit_down_val) and int(limit_down_val) > config.high_risk_limit_down_threshold:
            return True

    if has_index_chg:
        index_chg_val = row.get(_INDEX_CHG_COL, 0)
        if pd.notna(index_chg_val) and float(index_chg_val) < config.high_risk_index_drop_threshold:
            return True

    return False


def _check_no_future_data_internal(df: pd.DataFrame, date_col: str) -> None:
    """
    内部合规校验（generate_sample_weights 调用）。
    发现违规时直接抛出 RuntimeError，禁止生成权重。
    """
    import re
    future_pattern = re.compile(r"_d-\d+")
    future_cols = [c for c in df.columns if future_pattern.search(str(c))]
    if future_cols:
        raise RuntimeError(
            f"[RiskPenalty] 合规校验失败：检测到含未来数据的特征列 {future_cols}。"
            f"本项目约定 _d0=当日, _d1~d4=历史，负索引（_d-1等）表示未来，"
            f"这些列必须从训练集中移除后才能生成权重。"
        )


# ============================================================
# 快速自测（直接运行验证模块正确性）
# ============================================================

if __name__ == "__main__":
    """
    直接运行自测：
        python learnEngine/risk_penalty_core.py

    自测内容：
        1. 配置类（预设 + 自定义 + 校验）
        2. 样本权重生成（含宏观环境加成）
        3. 合规校验
        4. 实盘规则函数
        5. 归一化工具
    """
    warnings.filterwarnings("ignore")
    print("=" * 65)
    print("risk_penalty_core.py — 自测开始")
    print("=" * 65)

    # ── Test 1: 配置类 ────────────────────────────────────────────
    print("\n[Test 1] 配置类")
    default_cfg  = RiskPenaltyConfig.default()
    regime_cfg   = RiskPenaltyConfig.for_regime_model()
    strategy_cfg = RiskPenaltyConfig.for_strategy_model()
    custom_cfg   = RiskPenaltyConfig(loss_weight_multiplier=4.0, open_15min_stop_loss_pct=-0.025)

    assert default_cfg.loss_weight_multiplier  == 3.0,  "default 配置错误"
    assert regime_cfg.loss_weight_multiplier   == 5.0,  "regime 配置错误"
    assert strategy_cfg.loss_weight_multiplier == 3.0,  "strategy 配置错误"
    assert custom_cfg.loss_weight_multiplier   == 4.0,  "custom 配置错误"

    strategy_cfg.validate()   # 正常配置不应抛出异常

    try:
        bad_cfg = RiskPenaltyConfig(loss_weight_multiplier=0.5)  # 非法
        bad_cfg.validate()
        assert False, "应抛出 ValueError"
    except ValueError as e:
        pass  # 预期异常

    print("  [OK] 配置类测试通过（预设 + 自定义 + 校验）")

    # ── Test 2: 样本权重生成 ──────────────────────────────────────
    print("\n[Test 2] 样本权重生成")
    np.random.seed(42)
    n = 200
    mock_df = pd.DataFrame({
        "trade_date":              pd.date_range("2024-11-01", periods=n, freq="B").strftime("%Y-%m-%d"),
        "stock_code":              [f"{i:06d}.SH" for i in range(n)],
        "label1":                  np.random.choice([0, 1], size=n, p=[0.7, 0.3]),
        _LIMIT_DOWN_COL:           np.random.randint(0, 150, size=n),
        _INDEX_CHG_COL:            np.random.uniform(-0.05, 0.03, size=n),
        "stock_amount_d0":         np.random.uniform(5000, 50000, size=n),
        "bias13":                  np.random.uniform(-0.1, 0.2, size=n),
    })

    weights = generate_sample_weights(mock_df, label_col="label1", config=strategy_cfg)

    assert weights.shape == (n,),          f"权重维度错误: {weights.shape}"
    assert np.all(weights > 0),            "存在非正权重"
    assert abs(weights.mean() - 1.0) < 0.1, f"均值归一化失败: {weights.mean():.4f}"

    # 验证亏损样本权重 > 盈利样本权重
    loss_mask = mock_df["label1"].values == 0
    win_mask  = mock_df["label1"].values == 1
    assert weights[loss_mask].mean() > weights[win_mask].mean(), "亏损样本权重应大于盈利样本"

    print(f"  [OK] 权重生成正常 | shape={weights.shape} | 均值={weights.mean():.3f} "
          f"| 亏损均重={weights[loss_mask].mean():.3f} 盈利均重={weights[win_mask].mean():.3f}")

    # ── Test 3: 宏观特征缺失时的容错 ──────────────────────────────
    print("\n[Test 3] 宏观特征列缺失容错")
    mock_df_no_macro = mock_df.drop(columns=[_LIMIT_DOWN_COL, _INDEX_CHG_COL])
    weights_no_macro = generate_sample_weights(mock_df_no_macro, label_col="label1")
    assert weights_no_macro.shape == (n,), "缺失宏观特征时权重维度错误"
    print(f"  [OK] 宏观特征缺失时正常降级 | 均值={weights_no_macro.mean():.3f}")

    # ── Test 4: 合规校验 ──────────────────────────────────────────
    print("\n[Test 4] 合规校验")
    report = check_no_future_data(mock_df, feature_date_col="trade_date", label_col="label1")
    assert report["is_compliant"], f"合规校验误报: {report['errors']}"
    print(f"  [OK] 合法数据通过校验 | 日期:{report['date_range']}")

    # 测试未来数据列检测
    mock_df_bad = mock_df.copy()
    mock_df_bad["stock_close_d-1"] = np.random.randn(n)  # 人为加入未来数据列
    report_bad = check_no_future_data(mock_df_bad)
    assert not report_bad["is_compliant"], "未来数据列应被检测出"
    print(f"  [OK] 未来数据列检测正常 | 错误:{report_bad['errors'][0][:60]}...")

    # 测试 generate_sample_weights 对未来数据的强制拒绝
    try:
        generate_sample_weights(mock_df_bad, label_col="label1")
        assert False, "应抛出 RuntimeError"
    except RuntimeError:
        print("  [OK] 未来数据强制拒绝正常")

    # ── Test 5: 实盘规则函数 ──────────────────────────────────────
    print("\n[Test 5] 实盘规则函数")
    # 策略权重折扣
    assert calc_strategy_weight_discount(0.65) == strategy_cfg.recent_loss_high_discount
    assert calc_strategy_weight_discount(0.50) == strategy_cfg.recent_loss_medium_discount
    assert calc_strategy_weight_discount(0.30) == 1.0
    print("  [OK] 策略权重折扣三档逻辑正确")

    # 高风险股票过滤
    assert should_filter_high_risk_stock(80, 5000.0, strategy_cfg) is True   # 高风险+低流动性 -> 过滤
    assert should_filter_high_risk_stock(80, 50000.0, strategy_cfg) is False  # 高风险+高流动性 -> 不过滤
    assert should_filter_high_risk_stock(20, 5000.0, strategy_cfg) is False   # 低风险 -> 不过滤
    print("  [OK] 高风险个股过滤逻辑正确")

    # 硬止损
    assert should_stop_loss(-0.04) is True   # -4% < -3% 阈值 -> 止损
    assert should_stop_loss(-0.02) is False  # -2% > -3% 阈值 -> 持有
    assert should_stop_loss(0.01)  is False  # 正收益 -> 持有
    print("  [OK] 硬止损决策逻辑正确")

    # ── Test 6: 归一化工具 ───────────────────────────────────────
    print("\n[Test 6] 归一化工具")
    raw_weights = np.array([1.0, 3.0, 4.5, 100.0, 1.0])  # 100.0 为极端值
    norm = normalize_weights(raw_weights, clip_max_multiplier=5.0)
    assert abs(norm.mean() - 1.0) < 0.05, f"归一化均值异常: {norm.mean()}"
    assert norm.max() < 100.0, "极端值截断失效"
    print(f"  [OK] 归一化正常 | 原始max={raw_weights.max()} -> 截断后max={norm.max():.3f}")

    print("\n" + "=" * 65)
    print("[OK] 全部自测通过 (6/6)")
    print("=" * 65)
    print("\n接入指南：")
    print("  训练时（train.py）：")
    print("    from learnEngine.risk_penalty_core import train_with_risk_penalty")
    print("    # 替换 xgb_model.train(...) 为：")
    print("    train_with_risk_penalty(xgb_model, X_train, X_val, y_train, y_val,")
    print("                            feature_cols, df_train_partition)")
    print()
    print("  策略时（sector_heat_strategy.py）：")
    print("    from learnEngine.risk_penalty_core import should_filter_high_risk_stock, should_stop_loss")
