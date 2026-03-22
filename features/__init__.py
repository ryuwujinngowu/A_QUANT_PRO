"""
特征层统一入口 (features/__init__.py)
======================================
外部调用方式（dataset.py）：

    from features import FeatureEngine, SectorHeatFeature
    from features.data_bundle import FeatureDataBundle

    # 全部因子
    engine = FeatureEngine()

    # 按需选择因子
    engine = FeatureEngine(["sector_heat", "sector_stock", "ma_position"])

    # 构建数据容器后计算
    data_bundle = FeatureDataBundle(trade_date, ts_codes, sector_map, top3, adapt_score)
    feature_df  = engine.run_single_date(data_bundle)

已注册因子（按导入顺序）：
    sei_emotion   → SEIFeature（由 sector_stock 内部调用，不单独运行）
    sector_heat   → SectorHeatFeature（板块热度 + adapt_score 全局因子）
    sector_stock  → SectorStockFeature（个股维度全量情绪因子）
    ma_position   → MAPositionFeature（均线 + 乖离率 + 个股位置，D 日截面）
    market_macro  → MarketMacroFeature（涨跌停 + 连板 + 最强板块 + 指数，全局因子）
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
import pandas as pd

from features.base_feature import BaseFeature
from features.feature_registry import feature_registry, FeatureRegistry
from features.data_bundle import FeatureDataBundle
from utils.log_utils import logger

# ──────────────────────────────────────────────────────────────────────
# 因子注册区（"导入即注册"机制）
#
# 工作原理：
#   每个因子文件顶部都有 @feature_registry.register("xxx") 装饰器。
#   Python 执行 import 语句时，该装饰器立即运行，将类写入注册中心的
#   全局字典。因此「import = 注册」，无需手动调用任何注册函数。
#
# 新增因子 — 只需 2 步：
#   1. 在对应子目录下新建 .py 文件，继承 BaseFeature 并加装饰器
#   2. 在下方加一行 import（noqa 注释抑制"未使用变量"IDE 警告）
#      示例：from features.your_dir.your_file import YourFeature  # noqa: F401
#
# 删除因子 — 只需 1 步：
#   删除或注释掉对应的 import 行，该因子将从注册中心消失，
#   FeatureEngine 不会再调用它。（旧训练集列会在下次重跑后自然消失）
#
# 顺序说明：
#   import 的顺序决定 get_all_features() 的返回顺序，
#   进而影响 FeatureEngine 并行调度时的日志打印顺序（不影响结果）。
# ──────────────────────────────────────────────────────────────────────
from features.emotion.sei_feature import SEIFeature                        # noqa: F401  # SEI/HDI 情绪引擎（内部工具，不单独注册）
from features.sector.sector_heat_feature import SectorHeatFeature          # noqa: F401  # 板块热度 + adapt_score（全局因子）
from features.sector.sector_stock_feature import SectorStockFeature        # noqa: F401  # 板块个股全量情绪因子（个股因子）
from features.technical.ma_position_feature import MAPositionFeature       # noqa: F401  # 均线 + 乖离率 + 位置（个股因子，用前复权数据）
from features.macro.market_macro_feature import MarketMacroFeature         # noqa: F401  # 市场宏观因子：涨跌停/连板/指数（全局因子）

__all__ = [
    "FeatureEngine", "FeatureDataBundle",
    "SectorHeatFeature", "SectorStockFeature", "SEIFeature", "MAPositionFeature",
    "MarketMacroFeature",
    "feature_registry",
]


class FeatureEngine:
    """
    特征引擎，统一调度所有已注册因子

    :param feature_name_list: 指定因子名称列表，None 则运行全部已注册因子
                               可用值：sei_emotion, sector_heat, sector_stock
    """

    def __init__(self, feature_name_list: List[str] = None):
        if feature_name_list is None:
            self.features = feature_registry.get_all_features()
        else:
            self.features = feature_registry.get_features(feature_name_list)
        self.logger = logger
        self.logger.info(
            f"[FeatureEngine] 初始化完成，已加载：{[f.feature_name for f in self.features]}"
        )

    def run_single_date(self, data_bundle: FeatureDataBundle) -> pd.DataFrame:
        """
        单日全量特征计算（多线程并行调度各因子）

        :param data_bundle: 预加载的数据容器（只读，线程安全）
        :return: stock_code + trade_date 为主键的特征 DataFrame
        """
        trade_date = data_bundle.trade_date
        stock_dfs: List[pd.DataFrame] = []   # 含 stock_code（个股级）
        global_dfs: List[pd.DataFrame] = []  # 不含 stock_code（全局级，如 adapt_score）

        def _run_one(feature):
            return feature.feature_name, feature.calculate(data_bundle)

        with ThreadPoolExecutor(max_workers=len(self.features)) as pool:
            futures = {pool.submit(_run_one, f): f for f in self.features}
            for fut in as_completed(futures):
                feature = futures[fut]
                try:
                    name, (feature_df, _) = fut.result()
                    if feature_df.empty:
                        self.logger.warning(f"[FeatureEngine] {name} 返回空 DataFrame，跳过")
                        continue
                    if "stock_code" in feature_df.columns:
                        stock_dfs.append(feature_df)
                    else:
                        global_dfs.append(feature_df)
                except Exception as e:
                    self.logger.error(f"[FeatureEngine] {feature.feature_name} 失败：{e}", exc_info=True)
                    return pd.DataFrame()

        if not stock_dfs:
            self.logger.warning(f"[FeatureEngine] {trade_date} 无个股级特征数据")
            return pd.DataFrame()

        # 个股级 inner join（保证所有特征均有值）
        full_df = stock_dfs[0]
        for df in stock_dfs[1:]:
            full_df = pd.merge(full_df, df, on=["stock_code", "trade_date"], how="inner")

        # 全局级 left join（广播到所有行），修复原有 merge 逻辑 bug：
        # 原代码在 stock_dfs[0] 无 stock_code 时会产生笛卡尔积
        for df in global_dfs:
            full_df = pd.merge(full_df, df, on=["trade_date"], how="left")

        full_df = full_df.loc[:, ~full_df.columns.duplicated()]
        self.logger.info(
            f"[FeatureEngine] {trade_date} 合并完成 | 行:{len(full_df)} | 列:{len(full_df.columns)}"
        )
        return full_df