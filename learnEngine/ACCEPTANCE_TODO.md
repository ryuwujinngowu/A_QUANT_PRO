# ML 中台验收 TODO

> 创建时间：2026-03-30
> 目的：逐项验收各模块，确保主流程可端到端运行
> 使用方式：每完成一项在 `[ ]` 改为 `[x]`，并在下方记录测试结果
> 接手指令：先读本文件，再读 `ML_PLATFORM_GUIDE.md`，按顺序执行未完成项

---

## 状态说明

- `[x]` = 已通过（含记录）
- `[ ]` = 未测试
- `[!]` = 发现问题，需处理

---

## A. 模块级静态验证（语法 + import）

```bash
# 一键执行所有 py_compile
python -m py_compile \
  learnEngine/dataset.py \
  learnEngine/split_spec.py \
  learnEngine/strategy_configs.py \
  learnEngine/factor_selector.py \
  learnEngine/model.py \
  learnEngine/train_config.py \
  train.py \
  features/bundle_factory.py \
  strategies/sector_heat_strategy.py \
  agent_stats/agents/_model_signal_helper.py
```

- [x] A1. `learnEngine/train_config.py` — py_compile 通过
- [x] A2. `learnEngine/strategy_configs.py` — py_compile 通过
- [x] A3. `learnEngine/split_spec.py` — py_compile 通过
- [x] A4. `learnEngine/dataset.py` — py_compile 通过
- [x] A5. `learnEngine/factor_selector.py` — py_compile 通过
- [x] A6. `learnEngine/model.py` — py_compile 通过（StrategyXGBModel + 别名）
- [x] A7. `train.py` — py_compile 通过
- [x] A8. `features/bundle_factory.py` — py_compile 通过
- [x] A9. `strategies/sector_heat_strategy.py` — py_compile 通过
- [ ] A10. `agent_stats/agents/_model_signal_helper.py` — py_compile（待确认）
- [x] A11. `strategies/high_low_switch_ml_strategy.py` — py_compile 通过（2026-03-31）
- [x] A12. `learnEngine/dataset.py` — py_compile 通过（多策略修改后，2026-03-31）

---

## B. 核心逻辑单元验证

### B1. strategy_configs.py — 多策略列隔离

```python
from learnEngine.strategy_configs import get_effective_exclude_cols, list_registered_strategies
from learnEngine.train_config import EXCLUDE_COLS

cols_sh = get_effective_exclude_cols("sector_heat", EXCLUDE_COLS)
cols_hp = get_effective_exclude_cols("high_pos_tracking", EXCLUDE_COLS)

assert "adapt_score" not in cols_sh, "sector_heat 不应排除 adapt_score"
assert "adapt_score" in cols_hp,     "high_pos_tracking 应排除 adapt_score"
print("B1 通过，注册策略:", list_registered_strategies())
```

- [x] B1. 列隔离逻辑正确（已在 T13 验证通过）

### B2. train_config.py — archive/runtime 路径

```python
from learnEngine.train_config import get_archive_model_path, get_runtime_model_path

arch = get_archive_model_path("sector_heat", "v6.4")
rt   = get_runtime_model_path("sector_heat")

assert "archive" in arch and "sector_heat" in arch
assert "runtime" in rt   and "sector_heat" in rt
print("B2 通过:", arch)
```

- [x] B2. archive/runtime 路径生成正确（已在 T13 验证通过）

### B3. model.py — StrategyXGBModel 别名兼容

```python
from learnEngine.model import StrategyXGBModel, SectorHeatXGBModel
assert SectorHeatXGBModel is StrategyXGBModel
print("B3 通过")
```

- [x] B3. 别名兼容正确（已在 T13 验证通过）

### B4. split_spec.py — 读写一致性

```python
from learnEngine.split_spec import write_split_spec, load_split_spec, apply_split_spec
import pandas as pd

# 用现有 CSV 测试
from learnEngine.train_config import TRAIN_CSV_PATH, SPLIT_SPEC_PATH
import os

if os.path.exists(TRAIN_CSV_PATH):
    spec = write_split_spec(TRAIN_CSV_PATH, val_ratio=0.2, output_path="/tmp/test_spec.json")
    loaded = load_split_spec("/tmp/test_spec.json")
    assert spec["train_end_date"] == loaded["train_end_date"]
    print("B4 通过，train_end:", spec["train_end_date"], "val_start:", spec["val_start_date"])
else:
    print("B4 跳过：CSV 不存在")
```

- [ ] B4. split_spec 读写一致性验证（需有训练集 CSV）

### B5. bundle_factory.py — 构造接口

```python
# 仅验证接口可 import，不做实际 IO
from features.bundle_factory import build_bundle, build_bundle_from_context
print("B5 通过：bundle_factory 可 import")
```

- [ ] B5. bundle_factory import 正常（需在项目解释器环境下执行）

---

## C. 因子层验证

### C1. 因子注册表完整性

```python
from features.feature_registry import feature_registry
registered = list(feature_registry._registry.keys())
print("已注册因子:", registered)
# 预期包含：sector_heat / sector_stock / ma_position / market_macro /
#           individual / hp_stage / hp_style / hp_cycle / active_stats 等
```

- [ ] C1. 所有预期因子已注册（需在项目解释器环境下执行）

### C2. 全局因子 vs 个股因子 tag 正确

```python
# 检查全局因子（hp_stage / hp_style / hp_cycle / market_macro 等）的 is_global 标记
from features.feature_registry import feature_registry
for name, cls in feature_registry._registry.items():
    instance = cls()
    is_global = getattr(instance, "is_global", False)
    print(f"  {name}: is_global={is_global}")
```

- [ ] C2. 全局因子 is_global 标记正确

### C3. 单日因子计算 smoke test（需数据库连接）

```python
# 使用一个历史日期做小规模测试
# 预期：feature_df 包含全因子列，无全列 NaN
from learnEngine.train_config import STRATEGY_ID
# ... 参考 learnEngine/dataset.py 的单日调用链
```

- [ ] C3. 单日因子计算可正常运行（需 DB + 项目解释器）

---

## D. 数据集生成流程（dataset.py）

### D1. 单日 smoke test

> ⚠️ 2026-03-31 修复说明：
> - **label date bug 已修复**：`label_engine.generate_single_date(date, ...)` 改为 `label_engine.generate_single_date(context["feature_trade_date"], ...)`
>   - 原因：feature_df.trade_date = D-1（FeatureDataBundle.trade_date = feature_trade_date），label_df 必须也用 D-1 才能 merge 成功
> - **多策略循环已实现**：`_strategies = [SectorHeatStrategy(), HighLowSwitchMLStrategy()]`
> - **新文件**：`strategies/high_low_switch_ml_strategy.py`（候选池 = D-1涨停池，主板+非ST+首/二/三板）

```bash
# 先把 DATE_RANGES 改为只包含1-2个交易日，运行验证
# 例如：DATE_RANGES = [("2025-01-10", "2025-01-10")]
python learnEngine/dataset.py
```

检查点：
- CSV 中包含 `sample_id` / `strategy_id` / `strategy_name` / `sector_name` / `feature_trade_date` 列
- `strategy_id` 列同时包含 "sector_heat" 和 "high_low_switch" 的行（如当日两种策略都有候选股）
- `sample_id` 无重复
- label1 / label2 不全为 NaN（label date bug 已修复后应有值）
- 行数 >= 10

- [ ] D1. 单日 dataset 生成 ≥10 行，两种策略均有 rows，labels 非空（需 DB）
- [ ] D2. split_spec.json 内容验证（train/val 不重叠，边界日期合理）

---

## E. 因子筛选流程（factor_selector.py）

### E1. Stage 1+2 smoke test（无 Optuna，快速）

```bash
python learnEngine/factor_selector.py --stage 12
```

检查点：
- 输出 IC 表（各因子 ICIR / 胜率 / p 值）
- 相关性去重后因子数 < Stage 1 后因子数
- `strategy_id` 过滤有效（日志输出过滤后样本量）
- frozen split spec 被正确加载（日志输出 train_end_date / val_start_date）

- [x] E1. Stage 12 smoke test 通过（T8 测试，但为旧版 CSV；需用新 CSV 重验）

### E2. Stage 3 Optuna 10轮快速搜索

```bash
python learnEngine/factor_selector.py --n-trials 10
```

检查点：
- 输出 `selected_features_<version>.json`
- JSON 包含 `selected_features` 列表 + `best_params` 字典 + `_common.strategy_id`
- `best_params` 包含有效 XGBoost 超参（max_depth / learning_rate / n_estimators 等）

- [ ] E2. Stage 3 Optuna 10轮运行正常，输出 JSON 结构完整

---

## F. 模型训练流程（train.py）

### F1. 完整训练 smoke test

```bash
python train.py
```

检查点：
- 日志输出：strategy 过滤后样本量 / frozen split 边界 / 训练集行数 / 验证集行数
- `model/sector_heat/archive/<version>/model.pkl` 文件存在
- `model/sector_heat/archive/<version>/model.json` 文件存在
- 日志输出 AUC > 0.5（随机基线）
- **不存在** `shutil.copy2` 自动同步到 runtime（T11 约定）

- [ ] F1. 完整训练 smoke test（需有 selected_features.json + 训练集 CSV）
- [ ] F2. archive 模型文件正确生成，路径符合 T11 约定
- [ ] F3. runtime 目录未被自动修改（只有 archive 有新文件）

---

## G. 多策略列隔离验证（运行级）

### G1. 训练时列隔离实际生效

```python
# 运行 sector_heat 训练时，检查 feature_cols 中包含 adapt_score
# 运行 high_pos_tracking 训练时，检查 feature_cols 中不含 adapt_score
# （train.py load_and_prepare 的返回值）
from train import load_and_prepare
from learnEngine.train_config import TRAIN_CSV_PATH, TARGET_LABEL

# sector_heat
X, y, feature_cols, _, _, _ = load_and_prepare(TRAIN_CSV_PATH, TARGET_LABEL, strategy_id="sector_heat")
print("sector_heat feature_cols:", [c for c in feature_cols if "adapt" in c])  # 应包含 adapt_score
```

- [ ] G1. sector_heat 训练时 adapt_score 在 feature_cols 中
- [ ] G2. high_pos_tracking 训练时 adapt_score 不在 feature_cols 中（如有对应训练集）

---

## H. 推理层与训练层对齐验证

### H1. 推理特征列与训练特征列一致

推理侧用 `model.feature_names_in_` reindex，理论上已自动对齐。
但需确认：
- 推理时用的候选池逻辑与 `dataset.py` 中一致（同一个 `build_training_candidates`）
- 推理时调用 `build_bundle_from_context` 与训练时一致

- [ ] H1. 检查 `_model_signal_helper.py` 的推理路径与 `dataset.py` 的训练路径是否共用同一候选池逻辑

### H2. _model_signal_helper.py 模型路径更新

当前 `_model_signal_helper.py` 仍用旧路径：
```python
_MODEL_PATH = os.path.join(..., "model", "sector_heat_xgb_model.pkl")
```
T11 后推理层应读 `model/sector_heat/runtime/model.pkl`。

- [ ] H2. 将 `_model_signal_helper.py` 的 `_MODEL_PATH` 改为 `cfg.get_runtime_model_path("sector_heat")`

### H3. 端到端推理 smoke test

```python
# 用一个历史日期测试完整推理链：候选池 → bundle → 特征 → 预测概率
# 不需要真实下单，只需验证输出为 [(ts_code, prob), ...] 格式
```

- [ ] H3. 端到端推理 smoke test（需 DB + runtime 模型文件）

---

## I. 数据层对接验证

### I1. 训练集标签日期对齐

- `dataset.py` 用 `feature_trade_date`（D-1）计算特征，用 `trade_date`（D 日开盘买入日）做标签
- 确认 `LabelEngine.generate_single_date(trade_date, stock_list)` 里的 buy_price = D 日开盘，sell_price = D 日收盘

- [ ] I1. 抽查 5 条样本，确认特征日期 = 标签日期 - 1 个交易日

### I2. 推理特征不含未来数据

- 推理时 `feature_trade_date` = 昨日（T-1），`trade_date`（买入日）= 今日
- 因子计算不使用任何 T 日之后的数据

- [ ] I2. 随机抽查 3 个因子，确认 calculate() 只用 data_bundle.trade_date 及更早数据

---

## J. 回归测试（重构前后行为不变）

### J1. SectorHeatStrategy 推理结果稳定性

用相同日期跑一次推理，对比重构前后（T2-T14）输出的候选股列表是否相同或合理偏差范围内。

- [ ] J1. 用固定历史日期验证推理结果与重构前一致（可接受小幅排序差异）

### J2. 旧模型 pickle 可被 StrategyXGBModel 加载

```python
from learnEngine.model import StrategyXGBModel
model = StrategyXGBModel(model_save_path="model/sector_heat_xgb_latest.pkl")
model.load_model()
print("J2 通过：旧模型可加载")
```

- [ ] J2. 旧 pkl 文件可被新 StrategyXGBModel 正常加载

---

## K. 配置一致性验证

### K1. train_config.EXCLUDE_COLS 覆盖所有 label 列

```python
from learnEngine.train_config import EXCLUDE_COLS
from learnEngine.label import LabelEngine  # 或查看 label.py 的输出列

expected_labels = [
    "label1", "label2", "label1_3pct", "label1_8pct", "label_d2_limit_down",
    "label_raw_return", "label_open_gap", "label_d1_high", "label_d1_low",
    "label_d1_pct_chg", "label_d2_return",
]
missing = [l for l in expected_labels if l not in EXCLUDE_COLS]
assert not missing, f"EXCLUDE_COLS 缺少 label 列: {missing}"
print("K1 通过")
```

- [ ] K1. EXCLUDE_COLS 覆盖所有 label 列（无遗漏）

### K2. TRAIN_CSV_PATH 存在且非空

```python
import os, pandas as pd
from learnEngine.train_config import TRAIN_CSV_PATH
assert os.path.exists(TRAIN_CSV_PATH), f"训练集不存在: {TRAIN_CSV_PATH}"
df = pd.read_csv(TRAIN_CSV_PATH, nrows=5)
print(f"K2 通过，列数: {len(df.columns)}, 前5列: {list(df.columns[:5])}")
```

- [ ] K2. TRAIN_CSV_PATH 存在，包含预期列结构

---

## 验收结论

| 模块 | 状态 | 备注 |
|------|------|------|
| A. 静态验证 | 大部分通过 | A10 待确认 |
| B. 核心逻辑 | B1-B3 通过，B4-B5 待运行 | 需项目解释器环境 |
| C. 因子层 | 未测试 | 需 DB 连接 |
| D. dataset.py | 未运行级测试 | 需 DB 连接 |
| E. factor_selector | E1 旧版通过，E2 未测 | 需完整 CSV |
| F. train.py | 未运行级测试 | 需 selected_features.json |
| G. 多策略列隔离 | 单元测试通过，运行级未测 | |
| H. 推理层对接 | H2 路径问题待修复 | _model_signal_helper 旧路径 |
| I. 数据层对接 | 未测试 | |
| J. 回归测试 | 未测试 | |
| K. 配置一致性 | 未测试 | |

**优先处理**：
1. **H2**（推理层路径）— 阻塞实盘运行
2. **F1-F3**（训练 smoke test）— 验证训练主流程
3. **D1**（单日 dataset smoke test）— 验证训练集生成
4. **E2**（Optuna 10轮）— 验证因子筛选输出

---

## 测试日志

> 每次测试后在此追加记录，格式：
> `[日期] [测试项] 结果：通过/失败，说明：...`

- [2026-03-30] T13 全链路 import + 逻辑断言：A1-A9/B1-B3 通过
- [2026-03-30] T8 Stage 12 smoke test：E1 通过（旧版 CSV，train=590 val=180）
