"""
learnEngine/strategy_configs.py
=================================
多策略训练池的每策略配置中心。

设计原则
--------
核心问题：全局训练集是一张宽表，每一行的 ``strategy_id`` 字段标识该样本来自哪个策略的候选池。
不同策略的候选股逻辑完全不同，但所有行都共享同一套已注册因子（全因子宽表）。

每个策略在自己的候选池生成阶段可能会额外写入该策略专属的列（如 sector_heat 的
``adapt_score``）。这些专属列对其他策略的样本来说是 NaN / 0，如果不排除会引入偏差。

解决方案
--------
每个策略声明自己的 ``strategy_specific_cols``（自己额外写入的专属列）。
训练/因子筛选时调用 :func:`get_effective_exclude_cols`，自动把**其他所有策略**的
专属列加入排除列表。

使用方式
--------
- dataset.py / 策略类：在候选池中额外写入策略专属列时，在本文件注册
- factor_selector.py / train.py：用 ``get_effective_exclude_cols(strategy_id, ...)``
  替换原来的 ``EXCLUDE_COLS``，实现按策略隔离特征

新增策略只需在 ``STRATEGY_CONFIGS`` 里加一条记录，**不需要修改其他策略的配置**。
"""

from __future__ import annotations
from typing import Dict, List, Optional

# ─── 每策略配置表 ─────────────────────────────────────────────────────────────

STRATEGY_CONFIGS: Dict[str, Dict] = {

    "sector_heat": {
        # 默认训练标签（可在 train_config.py 中覆盖）
        "label": "label1",

        # 本策略候选池生成时额外写入的专属列（对其他策略样本为 NaN 或错误中性值）
        # 训练其他策略时，这些列会被自动排除
        "strategy_specific_cols": [
            "adapt_score",          # 板块热度适应分（SectorHeatStrategy 独有）
            # sector_stock 板块内特征：只对 sector_heat 候选股有意义；
            # 其他策略的样本经 LEFT JOIN 后这些列为 NaN，fillna(0) 后语义错误
            # （stock_vol_ratio=0 等同于停牌信号，stock_hdi=0 极端低难度，均非中性）
            "stock_sector_20d_rank",
            *[f"stock_hdi_d{i}" for i in range(5)],
            *[f"stock_vol_ratio_d{i}" for i in range(5)],
        ],
    },

    # ── 后续策略占位（候选池逻辑实现后填入） ──────────────────────────────────

    "high_pos_tracking": {
        "label": "label1",
        "strategy_specific_cols": [
            # e.g. "hp_breakout_score",  # 高位突破评分
        ],
    },

    "mid_pos_tracking": {
        "label": "label1",
        "strategy_specific_cols": [
            # e.g. "mid_pullback_score",
        ],
    },

    "high_low_switch": {
        "label": "label1",
        "strategy_specific_cols": [
            # e.g. "hl_switch_signal",
        ],
    },

    "momentum": {
        "label": "label1",
        "strategy_specific_cols": [
            # e.g. "momentum_score_5d",
        ],
    },

    "oversold_reversal": {
        # 超跌反包：D+1开盘买入，D+2收盘卖出 → 二分类标签（≥5%为正样本）
        # 与 OversoldReversalStrategy.get_training_label_target() 保持一致
        "label": "label_d2_5pct",
        "strategy_specific_cols": [],
    },

    "trend_follow": {
        # 趋势跟踪：D+1日内收益 ≥ 5% 为正样本
        # 与 TrendFollowStrategy.get_training_label_target() 保持一致
        "label": "label1",
        "strategy_specific_cols": [],
    },
}


# ─── 公开接口 ─────────────────────────────────────────────────────────────────

def get_effective_exclude_cols(
    strategy_id: Optional[str],
    base_exclude_cols: List[str],
) -> List[str]:
    """
    返回训练特定策略时应排除的完整列列表。

    = base_exclude_cols（全局 label/元数据列）
    + 所有**其他策略**的 strategy_specific_cols（避免跨策略列污染）

    :param strategy_id:      当前训练的策略 ID；None = 全策略池训练，不额外排除任何列
    :param base_exclude_cols: 全局基础排除列（来自 train_config.EXCLUDE_COLS）
    :return: 去重后的排除列列表
    """
    if not strategy_id:
        # 全策略训练：只排除全局 base，不额外排除（适用于跨策略研究）
        return list(base_exclude_cols)

    other_specific: List[str] = []
    for sid, cfg in STRATEGY_CONFIGS.items():
        if sid != strategy_id:
            other_specific.extend(cfg.get("strategy_specific_cols", []))

    return list(dict.fromkeys(list(base_exclude_cols) + other_specific))  # 去重保序


def get_strategy_label(strategy_id: str, fallback: str = "label1") -> str:
    """返回策略的默认训练标签列名（可在 train_config.py 中覆盖）。"""
    return STRATEGY_CONFIGS.get(strategy_id, {}).get("label", fallback)


def list_registered_strategies() -> List[str]:
    """返回所有已注册策略 ID 列表。"""
    return list(STRATEGY_CONFIGS.keys())
