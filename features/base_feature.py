# features/base_feature.py
import pandas as pd
from abc import ABC, abstractmethod
from  utils.log_utils import logger



class BaseFeature(ABC):
    def __init__(self, data_api=None):
        self.data_api = data_api
        self.feature_name = self.__class__.__name__
        self._locked = True

    @abstractmethod
    def calculate(self, *args, **kwargs) -> pd.DataFrame:
        """
        【核心抽象方法】所有特征类必须实现
        统一返回：pd.DataFrame，每行=单个股-单交易日，包含该特征计算的所有列
        """
        pass

    def run(self, *args, **kwargs) -> pd.DataFrame:
        """统一执行入口"""
        logger.info(f"[{self.feature_name}] 开始计算特征")
        result_df = self.calculate(*args, **kwargs)
        logger.info(f"[{self.feature_name}] 特征计算完成，共{len(result_df)}行")
        return result_df

    def __setattr__(self, key, value):
        if hasattr(self, "_locked") and self._locked and key != "_locked":
            if key in self.__dict__:
                raise RuntimeError(f"参数 {key} 已锁定，保证特征口径统一")
        super().__setattr__(key, value)