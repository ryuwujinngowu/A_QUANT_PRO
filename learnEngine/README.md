# learnEngine — 机器学习层

> 最后更新：2026-03-25
> 当前架构版本：v2.0（三阶段因子筛选 + 纯盈利优化）

本层负责**训练集生成 → 因子有效性分析 → 因子组合搜索 → 模型训练**四个环节。
亏损控制已迁移至 `position_tracker`（动态止损），模型只专注预测赚钱概率。

---

## 文件一览

| 文件 | 作用 | 运行方式 |
|------|------|----------|
| `dataset.py` | 训练集生成（逐日原子性处理，支持断点续跑） | `python learnEngine/dataset.py` |
| `label.py` | 标签定义（label1 / label2 / label_raw_return） | 被 dataset.py 调用 |
| `model.py` | XGBoost 模型类（训练、推理、保存、加载） | 被 train.py 调用 |
| `factor_ic.py` | 因子 IC 分析（单因子维度的预测力评估） | `python learnEngine/factor_ic.py` |
| `factor_selector.py` | **三阶段因子筛选**（IC过滤→相关性去重→Optuna搜索） | `python learnEngine/factor_selector.py` |
| `factor_search.py` | *(旧版)* Optuna 因子组合搜索（含亏损惩罚，已废弃） | 保留备查 |
| `train_config.py` | 统一配置入口（因子筛选/模型超参/训练参数） | 被 train.py 导入 |

项目根目录的关联文件：

| 文件 | 作用 |
|------|------|
| `train.py` | 模型训练入口（读 train_config.py，调用 model.py） |
| `risk_penalty_core.py` | *(已停用)* 亏损惩罚模块，保留供参考 |

---

## 整体流程（v2.0）

```
dataset.py              → 生成训练集 CSV（因子 + 标签 + 原始收益率）
       ↓
factor_ic.py            → 单因子 IC 分析（初筛）
       ↓
factor_selector.py      → 三阶段因子筛选（输出 selected_features.json）
  Stage 1: IC 过滤      → 140 因子 → ~60（剔除 |ICIR|<0.3 的无效因子）
  Stage 2: 相关性去重   → ~60 → ~35（同族高相关因子只保留代表）
  Stage 3: Optuna 精调  → ~35 → 最优组合 + 超参数（纯盈利目标，无惩罚参数）
       ↓
train.py                → 用 train_config.py + selected_features.json 训练模型
       ↓
model/                  → 保存训练好的模型（.pkl）
       ↓
strategies/             → 策略层加载模型进行实盘推理
```

---

## 核心设计决策

### 决策 1：移除亏损惩罚（2026-03-25）

**背景**：亏损惩罚（sample_weight 分级）使模型过度谨慎，实测造成性能下降。

**新分工**：
- **模型层**：只负责预测"次日开盘后盈利概率"，不感知亏损
- **止损层**：由 `position_tracker`（stop_loss_pct / trailing_stop）控制风险
- **类别平衡**：保留 `scale_pos_weight = neg/pos`，解决标签不平衡，与亏损无关

**配置变化**：
- `model.py`：`scale_pos_weight` 动态计算（唯一权重机制）
- `train.py`：删除 `RiskPenaltyConfig`，不再调用 `risk_penalty_core.py`
- `factor_search.py`（v2.0）：搜索空间删掉 4 个 loss penalty 变量

### 决策 2：换 Objective 函数

**旧版**：`0.4×Recall + 0.4×Precision + 0.2×AUC`
- 问题：高 Recall 可以通过"全部预测为正"轻松实现，诱导模型乱买

**新版（Precision@K + AUC）**：
```python
# K = 验证集按概率降序取前 30%（模拟实际只买最自信的股票）
precision_at_k = actual_win_rate_of_top_30pct_predictions
score = 0.6 × precision_at_k + 0.4 × AUC

# 保护项：top-30% 预测中至少有 10 个正样本（防止空信号过拟合）
if positive_count_in_top_k < 10:
    score *= 0.3
```

**理由**：
- `Precision@K` 直接对应真实交易效果：只买模型最有把握的那批
- `AUC` 保证整体排序能力，防止模型退化为固定阈值
- 去掉 Recall：实盘不需要抓住所有机会，只需要抓住的机会有较高胜率

### 决策 3：三阶段因子筛选

**为什么不直接用 Optuna 搜全部 140 因子**：
- 搜索空间 ≈ 4^24（家族）× 2^23（独立）≈ 天文数字
- 即使 200 轮 Optuna，在高维空间中本质是随机游走，不能收敛
- 噪声因子会干扰梯度提升，即使 Optuna "排除"了它们，前几轮的噪声仍会污染搜索历史

**三阶段设计**：

```
Stage 1: IC 过滤（秒级）
  输入：140 因子
  过滤条件：|ICIR| < 0.25 AND 胜率 < 50% AND p > 0.15（三条件同时满足才剔除）
  输出：~60-80 因子（剔除明确无效因子）

Stage 2: 相关性去重（秒级）
  输入：~60-80 因子
  方法：计算因子间 Pearson 相关系数矩阵
        对 |corr| > 0.75 的因子对，保留 |ICIR| 较高的一个
  输出：~25-40 因子（剔除信息冗余）

Stage 3: Optuna 精调（分钟级）
  输入：~25-40 因子候选
  搜索：每个候选因子 on/off + XGBoost 6 个超参
  目标：Precision@K + AUC（纯盈利，无惩罚参数）
  输出：最优因子子集（通常 15-30 个）+ 最优超参数
  用时：~10-30 分钟（vs 原来 30-120 分钟）
```

### 决策 4：暂不引入深度学习

**评估结论**：当前数据规模（~10 万行 × 40 特征）不适合 DL。

| 方案 | 优势 | 劣势 | 适用规模 |
|------|------|------|----------|
| XGBoost（当前） | 快、可解释、小数据强 | 特征交互靠人工构造 | 10K-1M 行 |
| LightGBM（可替换） | 比 XGB 快 3-5x，相近精度 | 同上 | 10K-1M 行 |
| TabNet（预留接口） | 自动学习特征交互注意力 | 需要 >50 万行才有优势 | >500K 行 |
| FT-Transformer | SOTA 表格 DL | 极难调参，超大数据才能赢 XGB | >1M 行 |

**预留设计**：`model.py` 中 `BaseModel` 抽象基类接口，未来接入 LightGBM 或 TabNet 无需改动 `train.py`。

**跨平台约束**（Windows + M1 Mac）：
- XGBoost：✅ 原生支持两平台
- LightGBM：✅ 原生支持两平台
- PyTorch（未来 DL 用）：✅ Windows（CUDA/CPU），✅ M1 Mac（MPS 后端）
- **禁止**：CUDA-only 库、TensorFlow（M1 适配较差）

---

## 新文件设计

### factor_selector.py（待实现）

```python
# 使用方式
python learnEngine/factor_selector.py \
    --csv learnEngine/datasets/train_dataset_latest.csv \
    --n-trials 200 \
    --output learnEngine/search_results/selected_features.json

# 参数
--csv         训练集路径
--n-trials    Optuna 轮数（默认 200）
--timeout     超时秒数（默认 3600）
--stage       1/2/3/all（指定只跑哪个阶段，默认 all）
--ic-min      Stage 1 ICIR 阈值（默认 0.25）
--corr-max    Stage 2 相关性阈值（默认 0.75）
--top-k-pct   Precision@K 的 K 值（默认 0.3 = 前 30%）
```

输出文件 `selected_features.json`：
```json
{
  "stage1_removed": ["factor_x", "factor_y", ...],
  "stage2_removed": ["factor_z", ...],
  "selected_features": ["stock_vwap_dev_d0", "stock_max_dd_d0", ...],
  "feature_count": 22,
  "xgb_params": { "max_depth": 4, "learning_rate": 0.05, ... },
  "metrics": { "precision_at_k": 0.48, "auc": 0.67 },
  "search_timestamp": "2026-03-25T14:30:00"
}
```

### train_config.py（待实现）

统一配置入口，消除 `train.py` / `factor_search.py` / `model.py` 中散落的参数：

```python
# 所有可调参数集中在这里
class FactorSelectorConfig:
    # Stage 1: IC 过滤
    IC_MIN_ICIR: float = 0.25         # |ICIR| 低于此值纳入"待剔除候选"
    IC_MIN_WIN_RATE: float = 0.50     # 胜率低于此值纳入"待剔除候选"
    IC_MAX_PVALUE: float = 0.15       # p 值高于此值纳入"待剔除候选"
    # 注：三条件同时满足才剔除（宽松策略，避免误删有价值因子）

    # Stage 2: 相关性去重
    CORR_MAX: float = 0.75            # 相关性高于此值视为冗余

    # Stage 3: Optuna
    N_TRIALS: int = 200
    TIMEOUT: int = 3600
    TOP_K_PCT: float = 0.30           # Precision@K 的 K 值


class ModelConfig:
    MODEL_TYPE: str = "xgboost"       # "xgboost" | "lightgbm" | "ensemble"
    # XGBoost 超参范围（Optuna 搜索用）
    MAX_DEPTH_RANGE: tuple = (3, 6)
    LR_RANGE: tuple = (0.03, 0.15)
    N_EST_RANGE: tuple = (200, 1000)
    SUBSAMPLE_RANGE: tuple = (0.6, 1.0)
    COLSAMPLE_RANGE: tuple = (0.6, 1.0)
    MIN_CHILD_WEIGHT_RANGE: tuple = (1, 10)


class TrainConfig:
    TRAIN_CSV_PATH: str = "learnEngine/datasets/train_dataset_latest.csv"
    SELECTED_FEATURES_PATH: str = "learnEngine/search_results/selected_features.json"
    MODEL_VERSION: str = "v5.0"
    TARGET_LABEL: str = "label1"
    VAL_RATIO: float = 0.2
    # 亏损惩罚：已移除，由 position_tracker 负责
    USE_LOSS_PENALTY: bool = False
```

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
        │
        ├─ Step 3: 构建板块候选池 sector_candidate_map
        │          过滤：北交所 / ST / 无涨停基因 / 当日封板 / 低流动性
        │
        ├─ Step 4: FeatureDataBundle — 一次性预加载所有数据到内存
        │
        ├─ Step 5: feature_engine.run_single_date(data_bundle)
        │          多线程并行计算所有因子 → feature_df（~140 列）
        │
        ├─ Step 6: label_engine.generate_single_date(date, ts_codes)
        │          → label_df（label1 / label2 / label_raw_return）
        │
        ├─ Step 7: 合并 feature_df + label_df，DataSetAssembler 校验，写入 CSV
        │
        └─ Step 8: dates_manager.add(date)  ← 写入成功后才标记，保证幂等性
```

### 可配置参数

| 参数 | 说明 | 修改时机 |
|------|------|----------|
| `START_DATE` / `END_DATE` | 训练集日期范围 | 需要延伸历史数据时 |
| `FACTOR_VERSION` | 因子版本号 | **每次修改因子计算逻辑或新增/删除因子后必须更新** |
| `OUTPUT_CSV_PATH` | 训练集 CSV 路径 | 默认在运行目录 |
| `MAX_CONSECUTIVE_FAILS` | 连续失败多少次终止 | 一般不改 |

> `FACTOR_VERSION` 是最高频的改动点。只要特征列、计算公式有变化，就必须更新（如 `"v5.0_xxx"`），否则旧的训练数据不会被重跑，会与新模型的列不一致。

---

## 二、train.py — 模型训练（v2.0）

### 2.1 设计原则

**v2.0 的核心变化**：移除亏损惩罚，train.py 只做以下三件事：
1. 读取 `train_config.py` 中的配置
2. 从 `selected_features.json` 加载最优因子列表
3. 用干净的 XGBoost（只有 scale_pos_weight 做类别平衡）训练模型

### 2.2 训练流程

```
python train.py
        │
        ▼
读取 TrainConfig + selected_features.json
        │  加载最优因子列表（factor_selector.py 的输出）
        │  加载最优 XGB 超参数（factor_selector.py 的输出）
        │
        ▼
load_and_prepare(TRAIN_CSV_PATH, selected_features)
        │  pd.read_csv → 去重 → 只保留 selected_features 列
        │  时间序列分割：前 80% 训练 / 后 20% 验证（不打乱）
        │
        ▼
计算 scale_pos_weight = neg_count / pos_count
        │  唯一的权重机制，解决标签不平衡（正样本约 10-15%）
        │  不再使用 sample_weight / RiskPenaltyConfig
        │
        ▼
XGBClassifier.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        │  early_stopping_rounds=20，监控验证集 AUC
        │
        ▼
evaluate_model(model, X_val, y_val)
        │  AUC / Precision@K / 混淆矩阵
        │  决策阈值搜索（遍历 0.10~0.50 找 Precision@K 最优阈值）
        │  Top-20 特征重要性
        │
        ▼
保存模型
        model/sector_heat_xgb_{VERSION}.pkl  ← 版本化存档
        model/sector_heat_xgb_latest.pkl     ← 策略层加载用
```

### 2.3 可配置参数（均在 train_config.py）

| 参数 | 说明 |
|------|------|
| `TRAIN_CSV_PATH` | 训练集路径 |
| `SELECTED_FEATURES_PATH` | factor_selector 输出的因子列表 |
| `MODEL_VERSION` | 模型版本号 |
| `TARGET_LABEL` | `"label1"` 或 `"label2"` |
| `VAL_RATIO` | 验证集比例（默认 0.2） |

### 2.4 标签说明（label.py）

| 标签 | 定义 | 语义 |
|------|------|------|
| `label1 = 1` | (D+1 close - D+1 open) / D+1 open ≥ 3% | 次日日内盈利 |
| `label2 = 1` | label1=1 **且** D+2 开盘 > D+1 收盘 | 值得持仓过夜 |
| `label_raw_return` | D+1 日的实际收益率（连续值） | 保留在 CSV 中，不作为训练特征 |

---

## 三、factor_selector.py — 三阶段因子筛选（v2.0）

### 3.1 为什么三阶段而不是直接 Optuna

单阶段 Optuna 在 140 因子上搜索的问题：
1. 搜索空间 ≈ `4^24(家族) × 2^23(独立)` → 200 轮相当于随机游走
2. 噪声因子污染 Optuna 搜索历史（TPE 采样器学到错误先验）
3. 把"IC为0"的因子和"IC高"的因子混在一起搜索，收益极不对称

三阶段的效果：每个阶段用 O(n) 时间剔除明确无效因子，最后 Optuna 在干净的搜索空间中运行。

### 3.2 Stage 1：IC 过滤

```python
# 从 factor_ic.py 的输出读取
# 过滤条件（三者同时满足才剔除，宽松策略）
keep_factor = NOT (
    abs(icir) < IC_MIN_ICIR    # 0.25
    AND win_rate < 0.50        # 胜率
    AND p_value > 0.15         # 统计不显著
)
```

**重要**：条件宽松（AND 而非 OR）——只有明确无效的因子才剔除，避免误删有潜力的因子。
低 ICIR 因子可能在组合中提供互补信息（XGBoost 的交互学习能力），Stage 3 会最终决定。

### 3.3 Stage 2：相关性去重

```python
# 计算因子间相关系数矩阵
corr_matrix = feature_df[candidate_factors].corr()

# 贪心聚类：按 |ICIR| 降序排列，逐个加入，若与已选因子 |corr| > 0.75 则丢弃
selected = []
for factor in sorted_by_icir:
    if max(|corr(factor, s)| for s in selected) <= CORR_MAX:
        selected.append(factor)
```

**效果**：同族的 d0~d4 因子高度相关，Stage 2 自动只保留最有效的天数。

### 3.4 Stage 3：Optuna 精调

**搜索空间**（~25-40 候选）：
- 每个候选因子：`on` / `off`
- XGBoost 超参：max_depth / learning_rate / n_estimators / subsample / colsample_bytree / min_child_weight

**优化目标（Precision@K）**：
```python
# 按预测概率降序，取前 K% 的样本
k = int(len(y_val) * TOP_K_PCT)  # 默认前 30%
top_k_mask = prob_rank <= k
precision_at_k = y_val[top_k_mask].mean()  # 实际正样本率

# 保护项
if y_val[top_k_mask].sum() < 10:  # top-K 中正样本太少
    return 0.3 * auc              # 惩罚空信号过拟合

score = 0.6 * precision_at_k + 0.4 * auc
```

### 3.5 与旧 factor_search.py 的对比

| 维度 | 旧 factor_search.py | 新 factor_selector.py |
|------|--------------------|-----------------------|
| 搜索前预处理 | 无 | IC过滤 + 相关性去重 |
| 搜索变量数 | 60 个（含惩罚参数） | ~30 个（纯因子+超参） |
| 优化目标 | Recall + Precision + AUC | Precision@K + AUC |
| 亏损惩罚参数 | 4 个（在搜索空间中） | 无 |
| 收敛质量 | 200 轮可能未收敛 | 200 轮大概率收敛 |
| 运行时间 | 30~120 分钟 | 10~30 分钟 |

### 3.6 使用方式

```bash
# 全流程（推荐）
python learnEngine/factor_selector.py --n-trials 200

# 只跑 Stage 1+2（快速看哪些因子被剔除）
python learnEngine/factor_selector.py --stage 12

# 用已有 Stage 1+2 结果只跑 Stage 3
python learnEngine/factor_selector.py --stage 3

# 快速验证（10 轮）
python learnEngine/factor_selector.py --n-trials 10
```

### 3.7 输出

`learnEngine/search_results/selected_features.json`：
```json
{
  "stage1_input_count": 140,
  "stage1_removed": ["factor_a", "factor_b"],
  "stage2_input_count": 78,
  "stage2_removed": ["stock_close_d1", "stock_close_d2"],
  "stage3_input_count": 35,
  "selected_features": ["stock_vwap_dev_d0", "stock_max_dd_d0", ...],
  "feature_count": 22,
  "xgb_params": { "max_depth": 4, "learning_rate": 0.05, ... },
  "metrics": {
    "precision_at_k": 0.48,
    "auc": 0.67,
    "k_pct": 0.30
  },
  "search_timestamp": "2026-03-25T14:30:00"
}
```

---

## 四、factor_ic.py — 因子 IC 分析

IC（Information Coefficient）衡量"一个因子能在多大程度上预测未来收益"。

### 核心概念

| 指标 | 含义 | 参考标准 |
|------|------|----------|
| IC 均值 | 因子与收益的平均秩相关系数 | \|IC\| > 0.03 有参考意义 |
| ICIR | IC均值 / IC标准差（信噪比） | \|ICIR\| > 0.25 进入 Stage 1 候选 |
| 胜率 | IC > 0 的日期占比 | > 50% 较好 |
| p值 | t检验显著性 | < 0.15 认为有参考意义 |

### 使用方式

```python
from learnEngine.factor_ic import calc_factor_ic_report
import pandas as pd

df = pd.read_csv("learnEngine/datasets/train_dataset_latest.csv")
report = calc_factor_ic_report(df, return_col="label1")
print(report.head(20))  # 按 |ICIR| 降序
```

---

## 五、新增因子完整指南

### 第 1 步：新建因子文件

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
            key    = (ts_code, trade_date)
            row    = daily_grouped.get(key, {})
            close  = float(row.get("close", 0) or 0)
            vol    = float(row.get("vol", 0) or 0)
            my_val = close * vol if vol > 0 else 0.0
            rows.append({"stock_code": ts_code, "trade_date": trade_date, "my_factor": my_val})

        return pd.DataFrame(rows), {}
```

> **禁止在 `calculate()` 内部发起数据库查询或 API 调用**。所有数据通过 `data_bundle` 获取。

### 第 2 步：在 `features/__init__.py` 添加 import

```python
from features.technical.my_factor_feature import MyFactorFeature  # noqa: F401
```

### 第 3 步：更新 FACTOR_VERSION 并重跑

```bash
# 1. 修改 dataset.py 的 FACTOR_VERSION（如 "v5.1_my_factor"）
# 2. 重跑训练集生成
python learnEngine/dataset.py
# 3. 重跑三阶段因子筛选（新因子自动纳入搜索范围）
python learnEngine/factor_selector.py --n-trials 200
# 4. 根据输出更新 train_config.py，重新训练
python train.py
```

### 完整检查清单

```
□ 1. 新建因子文件，写好 @feature_registry.register + calculate()
□ 2. features/__init__.py 添加一行 import
□ 3. dataset.py 更新 FACTOR_VERSION
□ 4. python learnEngine/dataset.py（重跑生成数据）
□ 5. python learnEngine/factor_selector.py（新因子自动纳入，重跑搜索）
□ 6. 查看 selected_features.json，确认新因子是否被选中
□ 7. 更新 train_config.py 的 MODEL_VERSION
□ 8. python train.py
```

---

## 六、删除因子

1. 在 `features/__init__.py` 注释掉或删除对应的 import 行
2. 更新 `dataset.py` 的 `FACTOR_VERSION` 并重跑 dataset.py
3. 重跑 factor_selector.py + train.py（删除的因子自动从搜索空间消失）

---

## 七、高频调整点汇总（v2.0）

| 调整内容 | 修改位置 | 需要重跑什么 |
|----------|----------|-------------|
| 修改训练日期范围 | `dataset.py` START_DATE / END_DATE | dataset.py |
| 新增 / 删除因子 | 因子文件 + `features/__init__.py` + FACTOR_VERSION | dataset.py → factor_selector.py → train.py |
| 修改因子计算公式 | 对应因子文件 + FACTOR_VERSION | dataset.py → factor_selector.py → train.py |
| 调整因子筛选阈值 | `train_config.py` FactorSelectorConfig | factor_selector.py → train.py |
| 调整 Precision@K 的 K 值 | `train_config.py` TOP_K_PCT | factor_selector.py → train.py |
| 修改 XGBoost 超参范围 | `train_config.py` ModelConfig | factor_selector.py（Stage 3）→ train.py |
| 切换预测目标 label1/label2 | `train_config.py` TARGET_LABEL | train.py |
| 修改标签定义 | `label.py` + FACTOR_VERSION | dataset.py → train.py |

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

# 宏观缓存
data_bundle.macro_cache["limit_up_df"]    # 涨停池
data_bundle.macro_cache["limit_down_df"]  # 跌停池
data_bundle.macro_cache["limit_step_df"]  # 连板天梯
data_bundle.macro_cache["limit_cpt_df"]   # 最强板块
data_bundle.macro_cache["index_df"]       # 指数日线（上证/深证/创业板）
data_bundle.macro_cache["market_vol_df"]  # 全市场成交量（近 5 日，kline_day 聚合）
```

---

## 九、目录结构（v2.0）

```
learnEngine/
├── README.md                  ← 本文档
├── __init__.py                ← 模块导出
├── dataset.py                 ← 训练集生成
├── label.py                   ← 标签定义
├── model.py                   ← XGBoost 模型类（预留 BaseModel 接口）
├── factor_ic.py               ← 单因子 IC 分析
├── factor_selector.py         ← 三阶段因子筛选（v2.0，替代 factor_search.py）
├── factor_search.py           ← (旧版，保留备查)
├── train_config.py            ← 统一配置入口（v2.0 新增）
├── datasets/                  ← 训练集 CSV 存放
│   └── train_dataset_latest.csv
├── search_results/            ← factor_selector.py 输出
│   ├── selected_features.json ← 最优因子列表 + 超参（v2.0）
│   └── best_config.json       ← (旧版输出，保留)
├── processed_dates.json       ← 断点续跑标记
└── history/                   ← 历史版本存档
```

---

## 十、实施路线图

### Phase 1（立即可做）：换 Objective + 移除惩罚

```bash
# 修改 factor_search.py 的 objective 为 Precision@K
# 删除 4 个 loss_penalty 搜索变量
# 修改 train.py 只用 scale_pos_weight
python learnEngine/factor_search.py --n-trials 200
python train.py
```
预期收益：Precision@K 提升（不再因 Recall 目标乱预测），训练更快。

### Phase 2（核心改进）：实现 factor_selector.py + train_config.py

```
新建 learnEngine/factor_selector.py（三阶段流程）
新建 learnEngine/train_config.py（统一配置）
修改 train.py 读 train_config
```
预期收益：搜索空间收缩 4x，收敛质量大幅提升，运行时间减半。

### Phase 3（可选增强）：LightGBM 支持

```
修改 model.py 新增 BaseModel 抽象 + LightGBMModel 实现
修改 train_config.py MODEL_TYPE = "lightgbm" | "xgboost" | "ensemble"
```
预期收益：训练速度 3x，精度持平或微升。

### Phase 4（未来，数据量 > 50 万行时）：DL 接入

```
新建 learnEngine/model_tabnet.py（PyTorch TabNet，跨平台）
train_config.py MODEL_TYPE = "tabnet"
```
当前不实施，接口预留好。
