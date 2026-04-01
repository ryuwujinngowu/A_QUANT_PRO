"""
features/bundle_factory.py
===========================
FeatureDataBundle 统一构造工厂（T12）。

外部调用方通过此工厂构建 bundle，屏蔽 FeatureDataBundle 的策略专属构造细节。
FeatureDataBundle 本身不改动，仅在外部增加一层包装，统一外部构造口径。

使用示例
--------
# 非 sector_heat 策略（最简调用）：
bundle = build_bundle(trade_date, ts_codes)

# sector_heat 策略（传入专属上下文）：
bundle = build_bundle(
    trade_date, ts_codes,
    strategy_context={
        "sector_candidate_map": ...,
        "top3_sectors": ...,
        "adapt_score": ...,
    },
)

# 从策略共享 context 直接构建（替代 strategy.build_feature_bundle_from_context）：
bundle = build_bundle_from_context(context, load_minute=True)
"""
from typing import Dict, List, Optional, Set

from features.data_bundle import FeatureDataBundle


def build_bundle(
    trade_date: str,
    target_ts_codes: List[str],
    strategy_context: Optional[Dict] = None,
    load_minute: bool = True,
    required_modules: Optional[List[str]] = None,
) -> FeatureDataBundle:
    """
    统一 FeatureDataBundle 构造入口。

    :param trade_date:       D 日，格式 yyyy-mm-dd
    :param target_ts_codes:  候选股代码列表（去重后）
    :param strategy_context: 策略专属上下文字典（可选），支持以下 key：
                               - sector_candidate_map : dict  (sector_heat 专用)
                               - top3_sectors         : list  (sector_heat 专用)
                               - adapt_score          : float (sector_heat 专用)
                             未传入的 key 自动使用中性默认值，不影响非 sector_heat 因子计算。
    :param load_minute:      是否加载分钟线（默认 True，纯日线调试可设 False 提速）
    :return: 已完成数据预加载的 FeatureDataBundle 实例
    """
    ctx = strategy_context or {}
    return FeatureDataBundle(
        trade_date=trade_date,
        target_ts_codes=target_ts_codes,
        sector_candidate_map=ctx.get("sector_candidate_map", {}),
        top3_sectors=ctx.get("top3_sectors", []),
        adapt_score=ctx.get("adapt_score", 0.0),
        load_minute=load_minute,
        required_modules=required_modules,
    )


def build_bundle_from_context(
    context: Dict,
    load_minute: bool = True,
    required_modules: Optional[List[str]] = None,
) -> Optional[FeatureDataBundle]:
    """
    从策略共享 context 字典直接构建 bundle。

    与 SectorHeatStrategy.build_feature_bundle_from_context() 等价，
    但策略无关，可被所有支持 ML 训练的策略复用。

    :param context:     包含 trade_date / target_ts_codes 以及可选策略专属 key 的上下文字典
    :param load_minute: 是否加载分钟线
    :return: FeatureDataBundle，或 None（target_ts_codes 为空时）
    """
    target_ts_codes = context.get("target_ts_codes") or []
    if not target_ts_codes:
        return None
    return build_bundle(
        trade_date=context["trade_date"],
        target_ts_codes=target_ts_codes,
        strategy_context=context,
        load_minute=load_minute,
        required_modules=required_modules,
    )
