# learnEngine/__init__.py
# from .mock_data_generator import generate_full_mock_dataset
from .label import LabelEngine
from .dataset import ProcessedDatesManager, DataSetAssembler, validate_train_dataset
from .model import SectorHeatXGBModel

__all__ = [
    # "generate_full_mock_dataset",
    "LabelEngine",
    "ProcessedDatesManager",
    "DataSetAssembler",
    "validate_train_dataset",
    "SectorHeatXGBModel",
    # factor_search 按需导入（避免启动时加载 optuna 等重量级依赖）：
    # from learnEngine.factor_search import FactorSearchEngine
    "FactorSearchEngine",
    "FactorGroupEngine",
    "SearchResultExporter",
]
