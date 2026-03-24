# learnEngine — 机器学习层

本层负责**训练集生成 → 因子有效性分析 → 因子组合搜索 → 模型训练（含亏损惩罚）**四个环节。

---

## 文件一览

| 文件 | 作用 | 运行方式 |
|------|------|----------|
| `dataset.py` | 训练集生成（逐日原子性处理，支持断点续跑） | `python learnEngine/dataset.py` |
| `label.py` | 标签定义（label1 / label2 / label_raw_return） | 被 dataset.py 调用 |
| `model.py` | XGBoost 模型类（训练、推理、保存、加载） | 被 train.py 调用 |
| `factor_ic.py` | 因子 IC 分析（单因子维度的预测力评估） | `python learnEngine/factor_ic.py` |
| `factor_search.py` | **Optuna 因子组合搜索**（多因子组合 + 超参数 + 惩罚参数联合优化） | `python learnEngine/factor_search.py` |

项目根目录的关联文件：

| 文件 | 作用 |
|------|------|
| `train.py` | 模型训练入口（调用 model.py + risk_penalty_core.py） |
| `risk_penalty_core.py` | 亏损惩罚核心模块（样本权重生成 + 类别平衡） |

---

## 整体流程

```
dataset.py          → 生成训练集 CSV（因子 + 标签 + 原始收益率）
       ↓
factor_ic.py        → 单因子 IC 分析：哪些因子有预测力？
       ↓
factor_search.py    → 多因子组合搜索：哪些因子组合在一起效果最好？最优超参数和惩罚参数是什么？
       ↓                （输出 best_config.json，含可粘贴到 train.py 的配置）
train.py            → 用最优配置训练模型（含亏损惩罚样本权重）
       ↓
model/              → 保存训练好的模型（.pkl + .json）
       ↓
strategies/         → 策略层加载模型进行实盘推理
```

### factor_ic.py vs factor_search.py 的区别

| 维度 | factor_ic.py | factor_search.py |
|------|-------------|-----------------|
| 回答什么问题 | "单个因子好不好" | "哪些因子组合在一起最好" |
| 分析方式 | 单因子 IC/ICIR 统计 | Optuna 贝叶斯搜索 |
| 能否发现交互效应 | 不能（独立评估每个因子） | 能（因子组合后通过 XGBoost 训练评估） |
| 用途 | 初筛：淘汰明确无效因子 | 精调：找最优组合 + 超参数 + 惩罚参数 |
| 运行时间 | 几秒 | 30~120 分钟（200 轮） |

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
        │          → label_df（stock_code, trade_date, label1, label2, label_raw_return）
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

> `FACTOR_VERSION` 是最高频的改动点。只要特征列、计算公式有变化，就必须更新（如 `"v3.2_xxx"`），否则旧的训练数据不会被重跑，会与新模型的列不一致。

### 断点续跑机制

- 每日处理完写入 `processed_dates.json`，下次启动自动跳过已处理日期
- 如果程序在"CSV写入成功"和"标记已处理"之间崩溃：下次启动会自动检测并修复（不会写重复数据）
- 如需强制重跑某段时间：删除 `processed_dates.json` 或手动修改其中的日期列表

---

## 二、train.py + risk_penalty_core.py — 模型训练与亏损惩罚

### 2.1 亏损惩罚设计思路

#### 问题背景

A 股量化场景有两个特点：
1. **标签极度不平衡**：正样本（次日盈利）约 10-15%，负样本约 85-90%
2. **亏损代价不对称**：一次 -7% 的亏损需要 +7.5% 才能回本，重亏的实际代价远大于同幅度盈利

朴素的 XGBoost 训练会因为负样本数量压倒性多，倾向于预测"不买"——Accuracy 很高但 Recall 极低（不产生任何买入信号），毫无实用价值。

#### 解决方案演进

**第一阶段：scale_pos_weight（类别平衡）**

XGBoost 原生 `scale_pos_weight = neg_count / pos_count`，放大正样本梯度。
问题：只解决了类别平衡，未区分"小亏"和"重亏"，模型无法学到"什么情况容易跌停"。

**第二阶段：sample_weight + scale_pos_weight（双重叠加，已废弃）**

在 `scale_pos_weight` 基础上，额外通过 `sample_weight` 对亏损样本施加惩罚。
问题：**两个机制方向相反，互相竞争**——`scale_pos_weight` 放大正样本，`sample_weight` 放大负样本。
结果：净效果不可控。惩罚参数稍大，模型完全不敢买（Recall=0.03）；稍小，模型乱买（Accuracy 暴跌）。

**第三阶段：统一由 sample_weight 承担（当前方案）**

将类别平衡和亏损惩罚统一到 `sample_weight` 中，`scale_pos_weight` 固定为 1.0：

```
正样本权重 = class_balance_ratio（= min(neg/pos, 4.0)，补偿正样本数量劣势）
负样本权重 = loss_weight_multiplier × severity_tier
             ├── raw_return < -7%  : severity_heavy（跌停级，最高惩罚）
             ├── raw_return < -3%  : severity_medium（中亏）
             ├── raw_return < 0%   : severity_light（小亏）
             └── raw_return < +3%  : 1.0（微赚未达标，中性）

高风险市场环境额外加成：
  满足任一条件时，亏损样本权重 × high_risk_env_extra_multiplier
  - 跌停家数 > 50
  - 上证指数跌幅 > 2%
```

权重全部在同一个维度竞争，正负样本的平衡点由参数精确控制。

#### 参数关系图

```
                   ┌───────────────────────────────────┐
                   │       generate_sample_weights     │
                   │    （risk_penalty_core.py）         │
                   └──────────┬────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
     正样本 (label=1)              负样本 (label=0)
              │                               │
   weight = class_balance_ratio    weight = loss_multiplier × severity
           (≈ neg/pos, cap 4.0)              │
              │                    ┌──────────┼──────────┐
              │               <-7%: heavy  <-3%: med   <0%: light
              │                    │          │          │
              │                    └──────────┴──────────┘
              │                               │
              │                  高风险环境？ × extra_multiplier
              │                               │
              └───────────┬───────────────────┘
                          │
                   normalize（均值=1.0）
                          │
                          ▼
                   传入 XGBoost fit(sample_weight=...)
                   scale_pos_weight = 1.0（不再二次叠加）
```

#### 参数调优的影响

| 场景 | loss_multiplier | 效果 |
|------|----------------|------|
| 过高（如 3.0） | 负样本权重压倒正样本 | Recall 极低，模型不敢买 |
| 过低（如 1.0） | 正样本权重 >> 负样本 | Recall 高但 Precision/Accuracy 下降 |
| 甜点（如 1.3~1.8） | 正负样本适度平衡 | Recall 和 Precision 均衡 |

甜点的精确值取决于数据分布，由 `factor_search.py` 自动搜索确定。

### 2.2 训练流程

```
python train.py
        │
        ▼
load_and_prepare("train_dataset_latest.csv")
        │  pd.read_csv → 去重 → 分离 X（特征）和 y（标签）
        │  EXCLUDE_COLS：排除主键/标签/辅助列
        │  EXCLUDE_PATTERNS：排除无效因子（fnmatch 通配符过滤）
        │
        ▼
time_series_split(X, y, val_ratio=0.2)
        │  按时间排序，前 80% 为训练集，后 20% 为验证集
        │  不做随机打乱，避免未来数据泄露
        │
        ▼
RiskPenaltyConfig（配置亏损惩罚参数）
        │  loss_weight_multiplier、severity tiers、high_risk_env 等
        │  参数来源：factor_search.py 搜索结果 或 手工设定
        │
        ▼
train_with_risk_penalty(xgb_model, X_train, ..., config)
        │  1. generate_sample_weights()：生成含类别平衡+亏损分级的样本权重
        │  2. scale_pos_weight=1.0（类别平衡已由 sample_weight 承担）
        │  3. XGBClassifier.fit(sample_weight=weights, eval_set=[(X_val, y_val)])
        │  4. early_stopping_rounds=20，监控验证集 AUC
        │
        ▼
evaluate_model(model, X_val, y_val)
        │  Accuracy / AUC / Precision / Recall / 混淆矩阵
        │  决策阈值搜索：遍历 0.10~0.50 找 F1 最优阈值
        │  Top-20 特征重要性
        │
        ▼
保存模型
        │  版本化存档：model/sector_heat_xgb_{VERSION}.pkl
        └─ 稳定路径同步：model/sector_heat_xgb_latest.pkl（策略层加载用）
```

### 2.3 可配置参数

**train.py 顶部：**

| 参数 | 说明 | 修改时机 |
|------|------|----------|
| `TRAIN_CSV_PATH` | 训练集路径 | 路径变动时 |
| `MODEL_VERSION` | 模型版本号 | 每次改因子/参数时更新 |
| `TARGET_LABEL` | `"label1"` 或 `"label2"` | 切换预测目标时 |
| `VAL_RATIO` | 验证集比例（默认 0.2） | 一般不改 |
| `EXCLUDE_PATTERNS` | 因子排除模式（fnmatch） | factor_search.py 搜索后更新 |

**train.py 中的 RiskPenaltyConfig：**

| 参数 | 说明 | 当前值 | 来源 |
|------|------|--------|------|
| `loss_weight_multiplier` | 亏损样本基础权重 | 1.5（临时） | 待 Optuna 搜索 |
| `high_risk_env_extra_multiplier` | 高风险环境加成 | 1.15 | 待 Optuna 搜索 |
| `loss_severity_tiers` | 亏损分级倍数 | 见代码 | 待 Optuna 搜索 |

### 标签说明（label.py）

| 标签 | 定义 | 语义 |
|------|------|------|
| `label1 = 1` | D+1 日收盘 > D+1 日开盘 + 5%（日内盈利） | 次日做多能赚钱 |
| `label2 = 1` | label1=1 **且** D+2 开盘 > D+1 收盘（隔夜高开） | 值得持仓过夜 |
| `label_raw_return` | D+1 日的实际收益率（连续值） | 用于亏损分级惩罚的样本权重计算 |

> `label_raw_return` 不作为训练特征（在 EXCLUDE_COLS 中），仅在 `generate_sample_weights` 中用于区分亏损程度。

---

## 三、factor_search.py — 因子组合自动搜索

### 3.1 为什么需要自动搜索

手工尝试因子组合的问题：
1. 因子之间存在**非线性交互效应**——单因子 ICIR 高不代表组合后好
2. 低 ICIR 因子可能在组合中提供互补信息
3. 因子选择、XGBoost 超参数、亏损惩罚参数三者互相影响，手工调参无法同时优化
4. 因子库持续扩大，每次新增因子后需要重新评估最优组合

### 3.2 搜索策略

166 个因子逐个 on/off 搜索 2^166 不现实。采用**因子家族分组**策略压缩搜索空间：

```
因子家族分组规则：
  stock_max_dd_d0, stock_max_dd_d1, ..., stock_max_dd_d4
  └── 归入家族 "stock_max_dd"

每个家族 4 种选择：
  exclude   → 全部排除
  d0_only   → 仅保留当日因子（d0）
  d0_d1     → 保留 d0 + d1（前一日）
  all_days  → 保留全部（d0~d4）

独立因子（无 d0~d4 后缀）：on / off
```

搜索空间：`4^24(家族) × 2^23(独立) × 连续超参数 ≈ 可管理`

### 3.3 搜索维度

| 类别 | 变量数 | 说明 |
|------|--------|------|
| 因子家族 | 24 组 × 4 选项 | 按家族控制天数粒度 |
| 独立因子 | 23 个 × 2 选项 | on/off |
| XGBoost 超参数 | 9 个 | max_depth, learning_rate, n_estimators 等 |
| 亏损惩罚参数 | 4 个 | loss_multiplier + 三档 severity |
| **合计** | **60 个决策变量** | TPE 采样器 200 轮可收敛 |

### 3.4 优化目标

```python
score = 0.4 × Recall + 0.4 × Precision + 0.2 × AUC

if Recall < 0.05:
    score *= 0.5   # 极低 Recall 重罚（避免模型完全不预测正样本）
```

量化场景需要 Recall（不遗漏机会）和 Precision（预测准确）并重。AUC 作为排序能力的辅助指标。

### 3.5 动态因子感知

`factor_search.py` **不硬编码任何因子名称**，而是从训练集 CSV 的列名动态发现：

```python
class FactorGroupEngine:
    DAY_SUFFIX_PATTERN = re.compile(r'^(.+)_d(\d)$')

    def _build(self, all_features):
        for feat in all_features:
            m = self.DAY_SUFFIX_PATTERN.match(feat)
            if m:
                self.family_groups[m.group(1)].append(feat)
            else:
                self.standalone_features.append(feat)
```

新增因子只需：
1. 在 `features/` 中实现因子 → 重跑 `dataset.py` 生成新 CSV
2. 重新运行 `factor_search.py` → 新因子自动纳入搜索范围

### 3.6 使用方式

```bash
# 正式搜索（200 轮，约 30~60 分钟）
python learnEngine/factor_search.py --n-trials 200

# 快速验证（10 轮，约 5 分钟）
python learnEngine/factor_search.py --n-trials 10

# 限时搜索（最多跑 1 小时）
python learnEngine/factor_search.py --n-trials 500 --timeout 3600

# 指定训练集
python learnEngine/factor_search.py --csv learnEngine/datasets/train_dataset_v5.csv
```

### 3.7 输出

搜索完成后输出三部分：

**1. 控制台摘要**
```
最优试验 (Trial #87):
  综合评分:   0.3842
  Recall:     0.2156
  Precision:  0.4012
  AUC:        0.6534
  因子数:     34

Top 10 试验:
   #  Score  Recall   Prec    AUC     F1  Nfeat  loss_m
  87  0.384  0.2156  0.4012  0.6534  0.28    34    1.72
  ...
```

**2. JSON 配置文件**（`learnEngine/search_results/best_config.json`）
```json
{
  "metrics": { "recall": 0.2156, "precision": 0.4012, "auc": 0.6534 },
  "xgb_params": { "max_depth": 4, "learning_rate": 0.08, ... },
  "risk_penalty": { "loss_multiplier": 1.72, "severity_heavy": 1.85, ... },
  "factor_families": { "stock_max_dd": "d0_only", "stock_vwap_dev": "d0_d1", ... },
  "selected_features": ["stock_max_dd_d0", "stock_vwap_dev_d0", "stock_vwap_dev_d1", ...]
}
```

**3. 可粘贴的 train.py 配置**
```
# --- EXCLUDE_PATTERNS ---
EXCLUDE_PATTERNS = [
    "stock_open_*", "stock_high_*", "stock_low_*", "stock_close_*",
    "ma5", "ma10", "sector_id",
    # Optuna 搜索排除的因子家族
    "stock_trend_r2_*",
    "stock_seal_times_*",
    ...
]

# --- RiskPenaltyConfig ---
config.loss_weight_multiplier = 1.72
config.loss_severity_tiers = (
    (-0.07, 1.85),
    (-0.03, 1.42),
    ( 0.00, 1.15),
    ( 0.03, 1.0),
)
```

---

## 四、factor_ic.py — 因子 IC 分析

IC（Information Coefficient）衡量"一个因子能在多大程度上预测未来收益"。

### 核心概念

| 指标 | 含义 | 参考标准 |
|------|------|----------|
| IC 均值 | 因子与收益的平均秩相关系数 | \|IC\| > 0.03 有参考意义 |
| ICIR | IC均值 / IC标准差（信噪比） | \|ICIR\| > 0.5 认为有效 |
| 胜率 | IC > 0 的日期占比 | > 55% 较好 |
| p值 | t检验显著性 | < 0.05 认为统计显著 |
| **effective** | \|ICIR\| > 0.5 **且** p < 0.05 | 两个条件同时满足 |

### 使用方式

```python
from learnEngine.factor_ic import calc_factor_ic_report
import pandas as pd

df = pd.read_csv("learnEngine/datasets/train_dataset_latest.csv")

# 全量因子分析
report = calc_factor_ic_report(df, return_col="label1")
print(report.head(20))

# 只分析特定因子
my_factors = ["stock_max_dd_d0", "stock_vwap_dev_d0", "bias13"]
report = calc_factor_ic_report(df, factor_cols=my_factors, return_col="label1")
```

---

## 五、新增因子完整指南

### 第 1 步：新建因子文件

在 `features/` 的某个子目录下新建文件（按因子类型选择子目录）：

```python
# features/technical/my_factor_feature.py

from features.base_feature import BaseFeature
from features.feature_registry import feature_registry
import pandas as pd

@feature_registry.register("my_factor")
class MyFactorFeature(BaseFeature):

    feature_name = "my_factor"

    def calculate(self, data_bundle) -> tuple:
        trade_date    = data_bundle.trade_date
        daily_grouped = data_bundle.daily_grouped
        ts_codes      = data_bundle.target_ts_codes

        rows = []
        for ts_code in ts_codes:
            key     = (ts_code, trade_date)
            day_row = daily_grouped.get(key, {})
            close   = float(day_row.get("close", 0) or 0)
            vol     = float(day_row.get("vol", 0) or 0)
            my_val  = close * vol if vol > 0 else 0.0

            rows.append({
                "stock_code": ts_code,
                "trade_date": trade_date,
                "my_factor":  my_val,
            })

        return pd.DataFrame(rows), {}
```

### 第 2 步：在 `features/__init__.py` 添加 import

```python
from features.technical.my_factor_feature import MyFactorFeature  # noqa: F401
```

### 第 3 步：更新 FACTOR_VERSION 并重跑

```bash
# 1. 修改 dataset.py 的 FACTOR_VERSION
# 2. 重跑训练集生成
python learnEngine/dataset.py
# 3. 重跑因子搜索（新因子自动纳入搜索范围）
python learnEngine/factor_search.py --n-trials 200
# 4. 根据搜索结果更新 train.py 配置，重新训练
python train.py
```

### 完整检查清单

```
1. 新建因子文件，写好 @feature_registry.register("xxx") + calculate()
2. features/__init__.py 添加一行 import
3. dataset.py 更新 FACTOR_VERSION
4. 重跑 python learnEngine/dataset.py
5. 重跑 python learnEngine/factor_search.py（新因子自动纳入搜索）
6. 根据搜索输出更新 train.py 的 EXCLUDE_PATTERNS 和 RiskPenaltyConfig
7. 重跑 python train.py
```

---

## 六、删除因子

1. 在 `features/__init__.py` 注释掉或删除对应的 import 行
2. 更新 `dataset.py` 的 `FACTOR_VERSION` 并重跑 dataset.py
3. 重跑 factor_search.py + train.py（删除的因子自动从搜索空间消失）

---

## 七、高频调整点汇总

| 调整内容 | 修改位置 | 需要重跑什么 |
|----------|----------|-------------|
| 修改训练日期范围 | `dataset.py` START_DATE / END_DATE | dataset.py |
| 新增 / 删除因子 | 因子文件 + `features/__init__.py` + FACTOR_VERSION | dataset.py → factor_search.py → train.py |
| 修改因子计算公式 | 对应因子文件 + FACTOR_VERSION | dataset.py → factor_search.py → train.py |
| 调整因子组合 | `train.py` EXCLUDE_PATTERNS | train.py（或跑 factor_search.py 自动搜索） |
| 调整亏损惩罚参数 | `train.py` RiskPenaltyConfig | train.py（或跑 factor_search.py 自动搜索） |
| 修改 XGBoost 超参数 | `learnEngine/model.py` base_params | train.py |
| 切换预测目标 label1/label2 | `train.py` TARGET_LABEL | train.py |
| 修改标签定义 | `learnEngine/label.py` + FACTOR_VERSION | dataset.py → train.py |

---

## 八、数据容器速查（data_bundle 可用字段）

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

---

## 九、目录结构

```
learnEngine/
├── README.md              ← 本文档
├── __init__.py            ← 模块导出
├── dataset.py             ← 训练集生成
├── label.py               ← 标签定义
├── model.py               ← XGBoost 模型类
├── factor_ic.py           ← 单因子 IC 分析
├── factor_search.py       ← Optuna 因子组合搜索
├── datasets/              ← 训练集 CSV 存放
│   └── train_dataset_latest.csv
├── search_results/        ← factor_search.py 输出
│   └── best_config.json
├── processed_dates.json   ← 断点续跑标记
└── history/               ← 历史版本存档
```
