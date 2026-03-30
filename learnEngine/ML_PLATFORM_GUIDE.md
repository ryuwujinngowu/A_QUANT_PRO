# ML 中台使用指南

> 版本：2026-03-30（T1-T14 全完成后）
> 适用对象：新增策略 / 新增因子 / 日常训练调参 / 理解整体架构

---

## 1. 整体架构一览

```
┌──────────────────────────────────────────────────────────────┐
│                     ML 中台全链路                              │
│                                                              │
│  策略层              learnEngine              特征层           │
│  (strategies/)       (训练/选因子)             (features/)     │
│                                                              │
│  BaseStrategy ──► dataset.py ──► factor_selector.py         │
│  (候选池生成)    (全局训练池CSV)    (IC+相关性+Optuna选因子)     │
│                      │                      │               │
│                      ▼                      ▼               │
│               split_spec.json ──────► train.py              │
│               (时序边界冻结)          (策略级训练)             │
│                                            │                │
│                                            ▼                │
│                              model/<strategy_id>/           │
│                                archive/<version>/model.pkl  │
│                                   (人工验证后)               │
│                                runtime/model.pkl            │
│                                   (推理层读取)               │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. 核心模块与文件位置

| 模块 | 文件 | 职责 |
|------|------|------|
| 统一配置 | `learnEngine/train_config.py` | 所有可调参数（路径/阈值/超参搜索范围），**只改这里** |
| 多策略注册 | `learnEngine/strategy_configs.py` | 各策略的专属列声明 + 列隔离逻辑 |
| 训练集生成 | `learnEngine/dataset.py` | 遍历历史日期 → 候选池 → 特征计算 → 打标 → 输出 CSV |
| 时序划分 | `learnEngine/split_spec.py` | 写/读/应用 frozen split spec（日期边界） |
| 因子筛选 | `learnEngine/factor_selector.py` | 三阶段：IC 过滤 → 相关性去重 → Optuna 超参搜索 |
| 模型训练 | `train.py` | 策略级训练入口，写 archive，不碰 runtime |
| 模型 Wrapper | `learnEngine/model.py` | `StrategyXGBModel`，策略无关 XGBoost 封装 |
| Bundle 工厂 | `features/bundle_factory.py` | 统一构造 `FeatureDataBundle`，屏蔽策略专属参数 |
| 策略基类 | `strategies/base_strategy.py` | ML hook：`supports_ml_training / build_training_candidates` |
| 推理辅助 | `agent_stats/agents/_model_signal_helper.py` | 加载模型 → 推理 → 买入候选排序 |

---

## 3. 完整调用流程

### 3.1 训练集生成（dataset.py）

```bash
python learnEngine/dataset.py
```

内部流程（每个 trade_date 循环）：

```
SectorHeatStrategy.select_top3_hot_sectors(date)
    → top3_sectors, adapt_score, sector_candidate_map

SectorHeatStrategy.build_training_candidates(date, daily_df)
    → candidate_df（含 strategy_id / strategy_name / sector_name / feature_trade_date 列）

bundle_factory.build_bundle_from_context(context)
    → FeatureDataBundle（预加载日线/分钟线/宏观数据）

FeatureEngine.run_single_date(bundle)
    → feature_df（全因子宽表）

LabelEngine.generate_single_date(date, stock_list)
    → label_df（D+1 日内收益标签）

feature_df merge label_df
    → 注入 sample_id / strategy_id / strategy_name 等元数据列
    → 追加写入 TRAIN_CSV_PATH

# 最后一次循环结束后：
split_spec.write_split_spec(csv_path, val_ratio=0.2)
    → 写出 split_spec.json（日期边界冻结）
```

**输出文件**：
- `learnEngine/datasets/train_dataset_<version>.csv`
- `learnEngine/datasets/split_spec.json`

---

### 3.2 因子筛选（factor_selector.py）

```bash
# 完整三阶段（400轮 Optuna）
python learnEngine/factor_selector.py

# 快速验证（只跑 Stage 1+2，不搜超参）
python learnEngine/factor_selector.py --stage 12

# 只重跑 Stage 3 Optuna（已有 Stage 1+2 结果时）
python learnEngine/factor_selector.py --stage 3

# 指定策略（默认读 train_config.STRATEGY_ID）
python learnEngine/factor_selector.py --strategy-id sector_heat

# 快速验证（10 轮 Optuna）
python learnEngine/factor_selector.py --n-trials 10
```

三阶段内部流程：

```
Stage 1 — IC 过滤
  对每个候选因子计算 Spearman IC：
    |ICIR| < IC_MIN_ICIR
    AND 胜率 < IC_MIN_WIN_RATE
    AND p值 > IC_MAX_PVALUE
  → 三条件同时满足才剔除（宽松策略）

Stage 2 — 相关性去重
  |Pearson 相关| > CORR_MAX(0.75) 的因子对
  → 保留 |ICIR| 更高的那个

Stage 3 — Optuna 搜索
  目标：Precision@30% + AUC 加权
  同时搜索：因子子集 + XGBoost 超参
  搜索 N_TRIALS=400 轮
```

**输出文件**：`learnEngine/search_results/selected_features_<version>.json`

内容包含：
- `selected_features`：筛出的因子列表
- `best_params`：最优 XGBoost 超参
- `_common`：strategy_id / split_spec 元数据

---

### 3.3 策略模型训练（train.py）

```bash
python train.py
```

参数均从 `learnEngine/train_config.py` 读取，**不需要命令行传参**。

内部流程：

```
load_and_prepare(TRAIN_CSV_PATH, TARGET_LABEL, strategy_id=STRATEGY_ID)
  1. 读取 CSV
  2. 过滤 strategy_id == STRATEGY_ID 的行
  3. 去重：drop_duplicates(subset=["sample_id"])
  4. 排除列：get_effective_exclude_cols(STRATEGY_ID, EXCLUDE_COLS)
     → 自动排除其他策略的专属列
  5. 读取 selected_features.json（Optuna 最优超参 + 因子列表）
  6. 读取 split_spec.json（frozen 日期边界）
  → 返回 X, y, feature_cols, df, override_params, split_spec

time_series_split(X, y, df, VAL_RATIO, split_spec=split_spec)
  → 优先用 frozen 日期边界；无 spec 时降级 val_ratio 尾部切分

_resolve_model_paths(STRATEGY_ID, MODEL_VERSION)
  → archive_path = model/sector_heat/archive/<version>/model.pkl
  → runtime_path = model/sector_heat/runtime/model.pkl

os.makedirs(archive_dir, exist_ok=True)

StrategyXGBModel(model_save_path=archive_path).train(
    X_train, X_val, y_train, y_val, feature_cols,
    override_params=override_params  # Optuna 最优超参
)
  → 保存 archive_path（pickle + JSON 副本）
  → 不自动同步 runtime（需人工晋升）

evaluate_model(model, X_val, y_val, feature_cols)
  → AUC / Precision / Precision@K / 特征重要性
```

**输出文件**：
- `model/sector_heat/archive/<version>/model.pkl`
- `model/sector_heat/archive/<version>/model.json`（XGBoost 原生格式，供调试）

---

### 3.4 模型晋升（手动）

训练完成并通过回测验证后，手动晋升：

```bash
# 创建 runtime 目录（首次）
mkdir -p model/sector_heat/runtime/

# 晋升
cp model/sector_heat/archive/<version>/model.pkl model/sector_heat/runtime/model.pkl
```

推理层（`_model_signal_helper.py`）从 `model/sector_heat/runtime/model.pkl` 读取。

> ⚠️ 当前 `_model_signal_helper.py` 仍用旧路径 `model/sector_heat_xgb_model.pkl`，
> 在推理层接入 T11 路径之前，晋升时需额外 cp 一份到旧路径，或修改 helper。

---

### 3.5 推理流程（实盘 / 回测）

```
SectorHeatStrategy.generate_signal(trade_date, context)
  → build_training_candidates(trade_date, daily_df)
     → candidate_df + context（含 sector_candidate_map / top3_sectors / adapt_score）
  → build_feature_bundle_from_context(context)
     → bundle_factory.build_bundle_from_context(context)
     → FeatureDataBundle（预加载数据）
  → FeatureEngine.run_single_date(bundle)
     → feature_df
  → model.predict_proba(feature_df[feature_names_in_])
     → 按概率排序 Top-K（K=6，概率 > 0.60）
  → 返回买入候选列表
```

---

## 4. 新增因子

### 步骤

1. **实现因子类**（`features/<category>/<name>_feature.py`）
   ```python
   from features.feature_registry import feature_registry

   @feature_registry.register("my_factor")
   class MyFactorFeature:
       def calculate(self, data_bundle) -> pd.DataFrame:
           # 返回含 stock_code + trade_date + 因子列 的 DataFrame
           ...
   ```

2. **在 `features/__init__.py` 添加 import**（触发注册）

3. **在 `learnEngine/train_config.py` 的 `EXCLUDE_COLS` 中确认新因子不需要排除**（默认不需要操作，除非这列不是特征）

4. **重新生成训练集**：`python learnEngine/dataset.py`

5. **重跑因子筛选**：`python learnEngine/factor_selector.py`

6. **重跑训练**：`python train.py`

### 全局因子 vs 个股因子

| 类型 | 返回内容 | FeatureEngine 处理方式 |
|------|---------|----------------------|
| 个股因子 | 含 stock_code 列 | 按 stock_code 合并 |
| 全局因子 | 无 stock_code（只有 trade_date） | 按 trade_date left join（广播到所有个股） |

---

## 5. 新增策略（接入 ML 中台）

### 步骤

1. **在策略类中实现 ML hook**：

```python
class MyNewStrategy(BaseStrategy):

    @property
    def strategy_id(self) -> str:
        return "my_strategy"

    def supports_ml_training(self) -> bool:
        return True

    def build_training_candidates(self, trade_date: str, daily_df) -> tuple:
        """返回 (candidate_df, context_dict)
        candidate_df 必须包含列：
          ts_code, strategy_id, strategy_name, sector_name（无板块填""）,
          feature_trade_date
        """
        ...
        return candidate_df, context

    def build_feature_bundle_from_context(self, context):
        from features.bundle_factory import build_bundle_from_context
        return build_bundle_from_context(context, load_minute=True)
```

2. **在 `learnEngine/strategy_configs.py` 注册**：

```python
STRATEGY_CONFIGS = {
    ...
    "my_strategy": {
        "label": "label1",
        "strategy_specific_cols": ["my_exclusive_col"],  # 只有该策略才有的列
    },
}
```

3. **在 `learnEngine/dataset.py` 的策略循环中添加**（当前只有 sector_heat，多策略时扩展此处）：

```python
for strategy in [SectorHeatStrategy(), MyNewStrategy()]:
    if strategy.supports_ml_training():
        candidates = strategy.build_training_candidates(date, daily_df)
        ...
```

4. **配置训练参数**：修改 `learnEngine/train_config.py` 的 `STRATEGY_ID` 为新策略 ID，然后跑完整训练流程。

---

## 6. 修改配置 / 调参

**所有参数集中在 `learnEngine/train_config.py`，只改这一个文件。**

```python
# 切换训练目标
TARGET_LABEL = "label1_3pct"   # 或 label1 / label1_8pct / label2 等

# 切换版本号（用于区分不同训练实验）
MODEL_VERSION = "v7.0_test"

# 调整 Optuna 搜索规模
N_TRIALS = 50        # 快速验证
N_TRIALS = 400       # 正式训练

# 调整 IC 过滤阈值（放宽：给 Optuna 更多候选因子）
IC_MIN_ICIR = 0.10

# 切换训练集 CSV（换数据集时修改）
TRAIN_CSV_PATH = os.path.join(...)

# 强制重跑因子筛选（即使已有结果文件）
FACTOR_SELECTOR_FORCE_REFRESH = True
```

---

## 7. 时序划分说明

### 设计原则
- **按日期边界**划分，不按行数，避免因数据密度不均匀导致信息泄露
- split spec 由 `dataset.py` 在生成训练集后**一次性写出**
- `factor_selector.py` 和 `train.py` **只读取**，保证两者用同一批 train/val 样本

### 文件位置
`learnEngine/datasets/split_spec.json`

```json
{
  "dataset_csv": "...路径...",
  "train_start_date": "2024-11-01",
  "train_end_date": "2026-01-31",
  "val_start_date": "2026-02-01",
  "val_end_date": "2026-02-28",
  "train_rows": 590,
  "val_rows": 180,
  "val_ratio": 0.2
}
```

### 降级机制
- 有 `split_spec.json` → 用日期边界
- 无 `split_spec.json` → 降级为 `val_ratio=0.2` 尾部切分（兼容旧流程）

---

## 8. 多策略列隔离机制

### 问题背景
不同策略的候选池生成逻辑不同，有些列只有特定策略才有：
- `adapt_score`：只有 `sector_heat` 策略有（板块轮动分）
- 其他策略样本的 `adapt_score` 列为 NaN 或 0

如果训练 `high_pos_tracking` 时不排除 `adapt_score`，
XGBoost 会把"大量 NaN"当作负信号，产生虚假学习。

### 解决方案
`learnEngine/strategy_configs.py` 中每个策略声明自己的专属列：

```python
STRATEGY_CONFIGS = {
    "sector_heat":        {"strategy_specific_cols": ["adapt_score"]},
    "high_pos_tracking":  {"strategy_specific_cols": []},
}
```

训练时：
```python
effective_exclude = get_effective_exclude_cols("high_pos_tracking", EXCLUDE_COLS)
# → EXCLUDE_COLS + ["adapt_score"]（sector_heat 的专属列被自动排除）
```

---

## 9. 现有文档索引

| 文档 | 内容 |
|------|------|
| `learnEngine/README.md` | TODO 列表 + 进度日志（实时维护） |
| `learnEngine/HANDOFF_ML_PLATFORM.md` | AI 接手文档（最低 token 成本快速上下文恢复）|
| `learnEngine/ML_PLATFORM_GUIDE.md` | **本文件**：使用说明 + 调用流程 |
| `learnEngine/ACCEPTANCE_TODO.md` | 验收 TODO 清单（含测试完成状态）|
| `CLAUDE.md`（项目根） | 项目全局记忆（数据原则 / 架构 / 修复历史）|
| `memory/project_ml_platform.md` | Claude 跨会话记忆（核心决策 + 文件定位）|
