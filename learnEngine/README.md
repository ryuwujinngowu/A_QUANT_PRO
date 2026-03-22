# learnEngine — 机器学习层

本层负责**训练集生成 → 模型训练 → 因子有效性分析**三个环节。

---

## 文件一览

| 文件 | 作用 |
|------|------|
| `dataset.py` | 训练集生成（逐日原子性处理，支持断点续跑） |
| `label.py` | 标签定义（label1 / label2） |
| `model.py` | XGBoost 模型类（训练、推理、保存、加载） |
| `factor_ic.py` | 因子 IC 分析工具（评估每个因子对标签的预测力） |

---

## 一、dataset.py — 训练集生成

### 调用顺序（完整流程图）

```
python learnEngine/dataset.py
        │
        ▼
[初始化] FeatureEngine()          ← 加载全部已注册因子
         LabelEngine()            ← 读取 START_DATE ~ END_DATE 的 D+1/D+2 标签数据
         SectorHeatFeature()      ← 板块热度计算器
         ProcessedDatesManager()  ← 读取/写入 processed_dates.json（断点续跑用）
        │
        ▼
[启动检查] get_trade_dates(START, END)
           已处理日期 → 跳过；CSV 与标记不一致 → 自动修复
        │
        ▼ 对每个待处理日期 date 循环：
        │
        ├─ Step 1: sector_heat.select_top3_hot_sectors(date)
        │          → top3_sectors（本日最热 3 个板块名称列表）
        │          → adapt_score（板块轮动速度 0-100）
        │
        ├─ Step 2: 宏观数据入库（涨停池 / 跌停池 / 连板天梯 / 最强板块 / 指数日线）
        │          data_cleaner.clean_and_insert_*(date_fmt)
        │
        ├─ Step 3: 构建板块候选池 sector_candidate_map
        │          对 top3_sectors 中每个板块：
        │          get_stocks_in_sector(sector)
        │          → _filter_ts_code_by_board()   过滤北交所/科创/创业板（可配置）
        │          → filter_st_stocks()            过滤 ST
        │          → _check_stock_has_limit_up()   保留近 10 日有涨停基因的股票
        │          → _filter_limit_up_on_d0()      过滤当日封板（买不进去）
        │          → _filter_low_liquidity()        过滤低流动性（amount < 1000万）
        │
        ├─ Step 4: FeatureDataBundle(date, ts_codes, sector_map, top3, adapt_score)
        │          一次性预加载所有数据到内存：
        │          _load_trade_dates()    → lookback_5d / lookback_20d
        │          _load_daily_data()     → daily_grouped（不复权，O(1)查找）
        │          _load_qfq_data()       → qfq_daily_grouped（前复权，MA专用）
        │          _load_macro_data()     → macro_cache（涨跌停/连板/指数等）
        │          _load_minute_data()    → minute_cache（候选股近5日分钟线）
        │
        ├─ Step 5: feature_engine.run_single_date(data_bundle)
        │          多线程并行计算所有因子：
        │          sector_heat.calculate()    → 全局因子（adapt_score + 板块效应）
        │          sector_stock.calculate()   → 个股因子（d0-d4，~140列）
        │          ma_position.calculate()    → 个股因子（MA/位置，11列）
        │          market_macro.calculate()   → 全局因子（涨跌停/指数，9+列）
        │          inner join（个股级） + left join（全局级） → feature_df
        │
        ├─ Step 6: label_engine.generate_single_date(date, ts_codes)
        │          → label_df（stock_code, trade_date, label1, label2）
        │
        ├─ Step 7: 合并 feature_df + label_df → merged_df
        │          DataSetAssembler.validate() 数据校验（类型/范围/极值处理）
        │          追加写入 train_dataset.csv（列对齐，header 仅首次写）
        │
        └─ Step 8: dates_manager.add(date)  ← 写入成功后才标记，保证幂等性
```

### 可配置参数（dataset.py 底部 `if __name__ == "__main__":`）

| 参数 | 说明 | 修改时机 |
|------|------|----------|
| `START_DATE` / `END_DATE` | 训练集日期范围 | 需要延伸历史数据时 |
| `FACTOR_VERSION` | 因子版本号 | **每次修改因子计算逻辑或新增/删除因子后必须更新** |
| `OUTPUT_CSV_PATH` | 训练集 CSV 路径 | 默认在运行目录 |
| `MAX_CONSECUTIVE_FAILS` | 连续失败多少次终止 | 一般不改 |

> ⚠️ `FACTOR_VERSION` 是最高频的改动点。只要特征列、计算公式有变化，就必须更新（如 `"v3.2_xxx"`），否则旧的训练数据不会被重跑，会与新模型的列不一致。

### 断点续跑机制

- 每日处理完写入 `processed_dates.json`，下次启动自动跳过已处理日期
- 如果程序在"CSV写入成功"和"标记已处理"之间崩溃：下次启动会自动检测并修复（不会写重复数据）
- 如需强制重跑某段时间：删除 `processed_dates.json` 或手动修改其中的日期列表

---

## 二、train.py — 模型训练

### 调用顺序

```
python train.py
        │
        ▼
load_and_prepare("train_dataset.csv")
        │  pd.read_csv → 去重 → 分离 X（特征）和 y（标签）
        │  X = 所有数值列（排除 stock_code / trade_date / label1 / label2 / sector_name 等）
        │  y = label1（默认）或 label2
        │
        ▼
time_series_split(X, y, val_ratio=0.2)
        │  按时间排序，前 80% 为训练集，后 20% 为验证集
        │  ⚠️ 不做随机打乱，避免未来数据泄露
        │
        ▼
SectorHeatXGBModel.train(X_train, y_train, X_val, y_val)
        │  动态计算 scale_pos_weight = neg样本数 / pos样本数（处理 A 股标签不平衡）
        │  XGBoost 训练（early_stopping_rounds=50，监控验证集 AUC）
        │  保存模型到 sector_heat_xgb_model.pkl
        │
        ▼
evaluate_model(model, X_val, y_val)
        │  Accuracy / AUC / Precision / Recall / Confusion Matrix
        └─ Top-20 特征重要性（feature_importances_）
```

### 可配置参数（train.py 顶部）

| 参数 | 说明 | 修改时机 |
|------|------|----------|
| `TRAIN_CSV_PATH` | 训练集路径 | 路径变动时 |
| `MODEL_SAVE_PATH` | 模型保存路径 | 路径变动时 |
| `TARGET_LABEL` | `"label1"` 或 `"label2"` | 切换预测目标时 |
| `VAL_RATIO` | 验证集比例（默认 0.2） | 一般不改 |
| `EXCLUDE_COLS` | 排除在特征之外的列 | 新增非特征列时补充 |

### 标签说明（label.py）

| 标签 | 定义 | 语义 |
|------|------|------|
| `label1 = 1` | D+1 日收盘 > D+1 日开盘（日内盈利） | 次日做多能赚钱 |
| `label2 = 1` | label1=1 **且** D+2 开盘 > D+1 收盘（隔夜高开） | 值得持仓过夜 |

> `label2 ⊆ label1`：label2=1 必然满足 label1=1。label2 是 label1 的子集（更严苛的条件）。

---

## 三、factor_ic.py — 因子 IC 分析

IC（Information Coefficient）衡量"一个因子能在多大程度上预测未来收益"。

### 核心概念

| 指标 | 含义 | 参考标准 |
|------|------|----------|
| IC 均值 | 因子与收益的平均秩相关系数 | \|IC\| > 0.03 有参考意义 |
| ICIR | IC均值 / IC标准差（信噪比） | \|ICIR\| > 0.5 认为有效 |
| 胜率 | IC > 0 的日期占比 | > 55% 较好 |
| p值 | t检验显著性 | < 0.05 认为统计显著 |
| **effective** | \|ICIR\| > 0.5 **且** p < 0.05 | 两个条件同时满足 |

### 调用顺序

```
python learnEngine/factor_ic.py
        │
        ▼
pd.read_csv("train_dataset.csv")
        │
        ▼
calc_factor_ic_report(df, exclude_cols=EXCLUDE_COLS, return_col="label1")
        │  对每个因子列：
        │  calc_ic_series()   → 按 trade_date 分组，每日计算截面 Spearman IC
        │  calc_icir()        → 汇总 IC 序列：均值 / 标准差 / ICIR / 胜率 / t检验
        │  标记 effective（|ICIR|>0.5 且 p<0.05）
        │  按 |ICIR| 降序排列
        │
        ▼
打印报告 + 保存 factor_ic_report.csv
```

### 如何只计算特定因子的 IC

**方法一：直接指定 `factor_cols`（推荐）**

```python
from learnEngine.factor_ic import calc_factor_ic_report
import pandas as pd

df = pd.read_csv("train_dataset.csv")

# 只计算你关心的因子
my_factors = [
    "stock_red_time_ratio_d0",
    "stock_float_profit_time_ratio_d0",
    "stock_red_session_pm_ratio_d0",
    "ma5_slope",
    "market_limit_up_count",
]
report = calc_factor_ic_report(df, factor_cols=my_factors, return_col="label1")
print(report)
```

**方法二：通过前缀过滤（筛选某一类因子）**

```python
# 只分析所有 d0 日的因子
d0_factors = [c for c in df.columns if c.endswith("_d0") and df[c].dtype != object]
report = calc_factor_ic_report(df, factor_cols=d0_factors, return_col="label1")

# 只分析时间持续类因子
time_factors = [c for c in df.columns if "time_ratio" in c or "session_pm" in c]
report = calc_factor_ic_report(df, factor_cols=time_factors, return_col="label1")
```

**方法三：分析单个因子的 IC 时间序列**

```python
from learnEngine.factor_ic import calc_ic_series, calc_icir

ic_series = calc_ic_series(df, factor_col="stock_red_time_ratio_d0", return_col="label1")
print(ic_series)         # 每个交易日的 IC 值
print(calc_icir(ic_series))   # 汇总统计

# 画出 IC 时间序列（需要 matplotlib）
ic_series.plot(title="red_time_ratio_d0 IC序列", figsize=(12, 4))
```

**方法四：分析因子的 IC 衰减（持仓 1~5 天的预测力变化）**

```python
from learnEngine.factor_ic import calc_ic_decay

# 需要 df 里有多个前向收益列（label_1d / label_2d / label_3d 等）
decay = calc_ic_decay(df, factor_col="stock_red_time_ratio_d0",
                      return_cols=["label1", "label_2d", "label_3d"])
print(decay)
```

---

## 四、新增因子完整指南（小白版）

> 下面以"新增一个自定义因子 `my_factor`"为例，完整走一遍流程。

### 第 1 步：新建因子文件

在 `features/` 的某个子目录下新建文件。选择哪个子目录取决于因子类型：

```
features/
├── sector/      ← 与板块/情绪相关
├── technical/   ← 技术指标（MA、MACD等）
├── macro/       ← 市场宏观（指数、涨停等）
└── emotion/     ← 情绪因子（SEI/HDI等，一般不用新增）
```

例如，新建 `features/technical/my_factor_feature.py`：

```python
# features/technical/my_factor_feature.py

from features.base_feature import BaseFeature
from features.feature_registry import feature_registry   # ← 必须导入注册中心
import pandas as pd

# ↓↓↓ 这个装饰器就是"注册"——给你的因子起一个唯一名字 ↓↓↓
@feature_registry.register("my_factor")
class MyFactorFeature(BaseFeature):

    feature_name = "my_factor"   # 与装饰器里的名字一致

    def calculate(self, data_bundle) -> tuple:
        """
        参数：data_bundle（数据容器，已预加载好所有数据，禁止在这里再发起 IO）
        返回：(feature_df, {})
            feature_df：
              - 个股因子：必须含 stock_code + trade_date 列
              - 全局因子：只含 trade_date 列（会广播到所有个股）
        """
        trade_date    = data_bundle.trade_date
        daily_grouped = data_bundle.daily_grouped   # 不复权日线
        ts_codes      = data_bundle.target_ts_codes

        rows = []
        for ts_code in ts_codes:
            key     = (ts_code, trade_date)
            day_row = daily_grouped.get(key, {})

            # ── 在这里写你的计算逻辑 ──────────────────────────────
            close   = float(day_row.get("close",   0) or 0)
            vol     = float(day_row.get("vol",     0) or 0)
            # 示例：简单的价格/成交量因子
            my_val  = close * vol if vol > 0 else 0.0
            # ─────────────────────────────────────────────────────

            rows.append({
                "stock_code": ts_code,
                "trade_date": trade_date,
                "my_factor":  my_val,      # 因子值
            })

        return pd.DataFrame(rows), {}
```

### 第 2 步：在 `features/__init__.py` 添加 import

打开 `features/__init__.py`，找到"导入即完成注册"那一段（约第 36-41 行），
仿照已有的那几行添加一行：

```python
# 导入即完成注册，顺序决定 get_all_features 的返回顺序
from features.emotion.sei_feature import SEIFeature                        # noqa: F401
from features.sector.sector_heat_feature import SectorHeatFeature          # noqa: F401
from features.sector.sector_stock_feature import SectorStockFeature        # noqa: F401
from features.technical.ma_position_feature import MAPositionFeature       # noqa: F401
from features.macro.market_macro_feature import MarketMacroFeature         # noqa: F401
from features.technical.my_factor_feature import MyFactorFeature           # noqa: F401  ← 加这行
```

> 只要 import 进来，`@feature_registry.register(...)` 装饰器就会自动执行，
> 因子就注册好了。不需要改任何其他地方（除了 FACTOR_VERSION）。

### 第 3 步：更新 `FACTOR_VERSION`

打开 `learnEngine/dataset.py`，找到：

```python
FACTOR_VERSION = "v3.2_qfq_ma_red_time"
```

改成一个新的版本号（版本号是你自己随便起的，只要和旧的不同就行）：

```python
FACTOR_VERSION = "v3.3_my_factor"   # ← 改成新名字
```

这会让 `ProcessedDatesManager` 检测到版本变更，自动清空已处理记录，
下次运行 `dataset.py` 时重新生成所有训练数据（包含新因子列）。

### 完整检查清单

```
☐ 1. 新建因子文件，写好 @feature_registry.register("xxx") + calculate()
☐ 2. features/__init__.py 添加一行 import
☐ 3. dataset.py 更新 FACTOR_VERSION
☐ 4. 重跑 python learnEngine/dataset.py（生成包含新因子的训练集）
☐ 5. 重跑 python train.py（用新训练集重新训练模型）
☐ 6. （可选）运行 python learnEngine/factor_ic.py 验证新因子的 IC 有效性
```

---

## 五、删除因子

> 删除比新增更简单，只需 2 步（不需要改因子文件，直接隐藏掉就行）：

1. 在 `features/__init__.py` 注释掉或删除对应的 import 行
2. 更新 `dataset.py` 的 `FACTOR_VERSION` 并重跑

> 如果因子涉及 `data_bundle.py` 中的专用数据加载（如 `qfq_daily_grouped`），
> 也需要考虑能否同步移除，但这不影响系统运行（保留加载也没有副作用）。

---

## 六、高频调整点汇总

以下是开发中最常需要修改的地方，按频率从高到低：

| 调整内容 | 修改位置 | 是否需要更新 FACTOR_VERSION |
|----------|----------|-----------------------------|
| 修改训练日期范围 | `dataset.py` START_DATE / END_DATE | 否（只影响数据范围，不影响列） |
| 新增 / 删除因子 | 因子文件 + `features/__init__.py` | **是** |
| 修改因子计算公式 | 对应因子文件 | **是** |
| 修改候选股过滤规则 | `dataset.py` 内 `_filter_*` 函数 | 否（不影响特征列定义） |
| 修改标签定义 | `learnEngine/label.py` | **是**（标签变了需重跑） |
| 修改 XGBoost 超参数 | `learnEngine/model.py` base_params | 否（只需重新 train） |
| 切换预测目标 label1↔label2 | `train.py` TARGET_LABEL | 否（只需重新 train） |
| 修改 IC 有效性阈值 | `learnEngine/factor_ic.py` 第 229-233 行 `effective` 判断 | 否 |
| 修改板块筛选规则 | `features/sector/sector_heat_feature.py` | 否 |
| 添加宏观数据类型 | `features/data_bundle.py` + `macro/market_macro_feature.py` | **是** |

---

## 七、数据容器速查（data_bundle 可用字段）

在因子的 `calculate(data_bundle)` 里，可以使用以下数据：

```python
data_bundle.trade_date          # str, D 日，格式 "YYYY-MM-DD"
data_bundle.target_ts_codes     # List[str], 本日所有候选股代码
data_bundle.top3_sectors        # List[str], Top3 板块名称
data_bundle.adapt_score         # float, 板块轮动速度分 0-100
data_bundle.sector_candidate_map  # Dict[板块名, DataFrame], 板块候选池

data_bundle.lookback_dates_5d   # List[str], 含 D 日在内最近 5 个交易日（升序）
data_bundle.lookback_dates_20d  # List[str], 含 D 日在内最近 20 个交易日（升序）

# 不复权日线，key=(ts_code, "YYYY-MM-DD"), value=该行 dict
data_bundle.daily_grouped       # Dict[tuple, dict]
# 常用字段: open/high/low/close/pre_close/vol/amount/pct_chg

# 前复权日线（MA 专用），结构同 daily_grouped
data_bundle.qfq_daily_grouped   # Dict[tuple, dict]

# 分钟线，key=(ts_code, "YYYY-MM-DD"), value=DataFrame(trade_time/open/high/low/close/volume)
data_bundle.minute_cache        # Dict[tuple, pd.DataFrame]

# 宏观缓存，各 key 对应预加载好的 DataFrame
data_bundle.macro_cache["limit_up_df"]    # 涨停池
data_bundle.macro_cache["limit_down_df"]  # 跌停池
data_bundle.macro_cache["limit_step_df"]  # 连板天梯
data_bundle.macro_cache["limit_cpt_df"]   # 最强板块
data_bundle.macro_cache["index_df"]       # 指数日线（上证/深证/创业板）
data_bundle.macro_cache["market_vol_df"]  # 全市场成交量（近 5 日，kline_day 聚合）
```

> **禁止在 `calculate()` 内部发起数据库查询或 API 调用**。
> 所有数据必须通过 data_bundle 获取，这是架构的核心约束（保证单次 IO）。
