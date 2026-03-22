"""
因子注册中心：统一管理所有特征类，支持灵活选择、批量调度

────────────────────────────────────────────────
设计模式：单例 + 装饰器注册
────────────────────────────────────────────────

单例模式（Singleton）：
    FeatureRegistry 只存在一个实例（feature_registry），
    _registry 字典在整个进程生命周期内全局共享，
    所有模块拿到的都是同一份注册表。

装饰器注册流程（以新增因子为例）：

    # your_feature.py
    @feature_registry.register("my_factor")   # ← 第一步：给类贴标签
    class MyFeature(BaseFeature):
        ...

    # features/__init__.py
    from features.xxx.your_feature import MyFeature  # noqa: F401
    #  ↑ 第二步：import 触发装饰器执行，"my_factor" 写入 _registry

    # 之后任何代码都可以：
    engine = FeatureEngine(["my_factor"])   # 按名字取用
    engine = FeatureEngine()               # 运行全部已注册因子
────────────────────────────────────────────────
"""
from typing import List, Dict, Type
from features.base_feature import BaseFeature


class FeatureRegistry:
    """
    因子注册器（单例）

    _registry: { "factor_name" -> FeatureClass } 的全局字典
    所有读写均通过类方法，不依赖实例状态，线程安全（GIL 保护字典操作）。
    """
    _instance = None
    _registry: Dict[str, Type[BaseFeature]] = {}  # 注册表：名字 → 类（注意是类本身，不是实例）

    def __new__(cls, *args, **kwargs):
        # 单例保证：整个进程只创建一个 FeatureRegistry 对象
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    @classmethod
    def register(cls, feature_name: str):
        """
        因子注册装饰器。

        用法：在因子类定义上方加一行
            @feature_registry.register("唯一名字")
        然后在 features/__init__.py 中 import 该文件，注册即自动完成。

        :param feature_name: 因子唯一标识（字符串），不可与已注册因子重名
        """
        def wrapper(feature_class: Type[BaseFeature]):
            if feature_name in cls._registry:
                raise ValueError(f"因子{feature_name}已注册，不可重复注册")
            feature_class.feature_name = feature_name   # 反向注入名字到类属性，方便日志打印
            cls._registry[feature_name] = feature_class
            return feature_class
        return wrapper

    @classmethod
    def get_feature(cls, feature_name: str) -> BaseFeature:
        """根据因子名称实例化并返回一个因子对象"""
        if feature_name not in cls._registry:
            raise ValueError(f"因子{feature_name}未注册，可用因子：{list(cls._registry.keys())}")
        return cls._registry[feature_name]()   # 每次调用都新建实例，避免状态污染

    @classmethod
    def get_features(cls, feature_name_list: List[str]) -> List[BaseFeature]:
        """
        按名称列表批量获取因子实例。
        用于 FeatureEngine(["sector_heat", "ma_position"]) 这类按需选取场景。
        """
        return [cls.get_feature(name) for name in feature_name_list]

    @classmethod
    def get_all_features(cls) -> List[BaseFeature]:
        """获取所有已注册的因子实例（顺序 = __init__.py 的 import 顺序）"""
        return [cls() for cls in cls._registry.values()]

    @classmethod
    def list_all_features(cls) -> List[str]:
        """
        列出当前已注册的所有因子名称，用于调试。
        示例：
            from features.feature_registry import feature_registry
            print(feature_registry.list_all_features())
            # ['sector_heat', 'sector_stock', 'ma_position', 'market_macro']
        """
        return list(cls._registry.keys())


# 全局单例实例 — 所有模块通过这一个对象访问注册中心
feature_registry = FeatureRegistry()