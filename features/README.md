# 特征层（features/）

> 因子完整口径、已知问题及 Review 记录，请查阅：**[features/FACTOR_REVIEW.md](./FACTOR_REVIEW.md)**

---

## 扩展指南

新增因子只需 3 步：

```python
# Step 1：创建因子文件
# features/your_category/your_feature.py

from features.base_feature import BaseFeature
from features.feature_registry import feature_registry

@feature_registry.register("your_feature_name")   # 全局唯一注册键
class YourFeature(BaseFeature):
    feature_name = "your_feature_name"
    factor_columns = ["col_a", "col_b"]            # 输出列声明（文档用）

    def calculate(self, data_bundle) -> tuple:
        """
        data_bundle 已预加载所有数据，此处禁止发起任何 IO。

        返回：
          个股级因子 → feature_df 含 stock_code + trade_date 列
          全局级因子 → feature_df 仅含 trade_date 列（FeatureEngine 自动 left join 广播）
        """
        trade_date = data_bundle.trade_date
        # ... 计算逻辑 ...
        return feature_df, {}

# Step 2：在 features/__init__.py 末尾添加 import（导入即注册）
from features.your_category.your_feature import YourFeature  # noqa: F401

# Step 3：更新 learnEngine/dataset.py 的 FACTOR_VERSION（触发训练集重跑）
```

### 可用数据（data_bundle 属性）

| 属性 | 内容 | 访问方式 |
|------|------|---------|
| `trade_date` | D 日，`yyyy-mm-dd` | 直接用 |
| `lookback_dates_5d` | D-4~D0 共5个交易日，升序 | `[-1]`=D0，`[:-1]`=D1~D4 |
| `lookback_dates_20d` | D-19~D0 共20个交易日 | 同上 |
| `daily_grouped` | `{(ts_code, trade_date): row_dict}` | O(1) 查找 |
| `minute_cache` | `{(ts_code, trade_date): DataFrame}` | 分钟线，None 表示无数据 |
| `macro_cache` | 涨跌停池/炸板池/连板/最强板块/指数/量统计等 | `macro_cache["limit_up_df"]` 等 |
| `hp_ext_cache` | 高位股基础池/近5日日线/ST集合/市场广度数据 | `hp_ext_cache["st_set"]` 等 |

### 全局级 vs 个股级

| 类型 | 特征 | join 方式 |
|------|------|---------|
| 全局级 | `feature_df` 无 `stock_code` 列，每日一行 | `left join on trade_date`，自动广播到所有个股 |
| 个股级 | `feature_df` 含 `stock_code` 列，每股一行 | `inner join on [stock_code, trade_date]` |

### 数据完整性原则

- **零 IO 原则**：`calculate()` 内不查库、不调 API，所有数据从 `data_bundle` 读取
- **中性值原则**：数据缺失时返回不引入方向偏差的中性值（数值=0 或历史均值，bool=False）
- **自引用防护**：比率因子的分子用 D0，分母用 D1~D4 均值，不含 D0 自身
