# ML 中台重构接手文档

> 更新时间：2026-03-30
> 用途：给下一次 Claude / AI **最低 token 成本**快速接手当前重构进度
> 原则：这里只写**接手必需信息**，详细背景仍看 `learnEngine/README.md`

---

## 1. 当前结论：做到哪了

### 已完成
- **T2** `BaseStrategy` 已加入 ML 中台非破坏式接口
- **T3** `SectorHeatStrategy` 已抽出共享候选池 / context / bundle 构造逻辑
- **T4** `agent_stats/agents/_model_signal_helper.py` 已切到共享候选链路
- **T5** `learnEngine/dataset.py` 已切到共享的 `sector_heat` 候选逻辑
- **T6** `learnEngine/dataset.py` 全局训练池 schema 完成：新增 `sample_id / strategy_id / strategy_name / sector_name / feature_trade_date`；旧 `(stock_code, trade_date)` 去重已替换为 `sample_id` 去重
- **T7** `learnEngine/split_spec.py` 新增 frozen split spec 模块；`train_config.py` 新增 `SPLIT_SPEC_PATH`；`dataset.py` 全量生成完后自动写 spec；split 按日期边界冻结
- **T8** `factor_selector.py` 新增 `strategy_id` 过滤 + frozen split；dedup 改 sample_id；CLI 加 `--strategy-id`
- **T9** `train.py` 新增 `strategy_id` 过滤 + frozen split；`train_config.py` 补全缺失属性
- **strategy_configs.py（核心）** `learnEngine/strategy_configs.py` — 多策略动态列隔离：每策略声明 `strategy_specific_cols`，`get_effective_exclude_cols()` 训练时自动排除其他策略专属列；`factor_selector.py` / `train.py` 均已切换
- **T10** `learnEngine/model.py` 类名从 `SectorHeatXGBModel` 改为 `StrategyXGBModel`；旧名保留为别名；`train.py` 已切换
- **T11** `train_config.py` 新增 `get_archive_model_path / get_runtime_model_path`；`train.py` 训练只写 archive（`model/<strategy_id>/archive/<version>/model.pkl`），不再自动同步 runtime；runtime 需人工晋升

### 已确认的关键事实
- 当前训练集默认唯一键仍是 **`(stock_code, trade_date)`**，这会阻碍全局多策略训练池
- 当前 split 逻辑仍在 `train.py` 和 `learnEngine/factor_selector.py` 各自维护，且还是**按行尾切分**，不是 frozen spec
- `SectorHeatStrategy.build_training_candidates()` 已经能产出：
  - `strategy_id`
  - `strategy_name`
  - `sector_name`
  - `feature_trade_date`
- 推理端当前安全：
  - `strategies/sector_heat_strategy.py`
  - `agent_stats/agents/_model_signal_helper.py`
  都是按 `model.feature_names_in_` reindex，所以新增元数据列只要**不进入训练特征**，不会破坏现有推理

---

## 2. 下一步从哪开始

下一步直接从 **T12** 开始，不要重做前面工作。

### T6（已完成）
定义全局训练池 schema：
- 增加 `sample_id`
- 增加 `strategy_id`
- 增加 `strategy_name`
- 增加 `feature_trade_date`
- 去掉旧的 `(stock_code, trade_date)` 唯一假设

### T7
新增 frozen split spec：
- dataset 生成时写 split artifact
- `factor_selector.py` / `train.py` 统一读取同一份 split spec
- split 必须按**日期边界冻结**，不能再按行切

### T8
改造 `learnEngine/factor_selector.py`：
- 按 `strategy_id` 过滤
- 使用 frozen split
- 去重键改为 `sample_id`

### T9
改造 `train.py`：
- 按 `strategy_id` 过滤
- 使用 frozen split
- 去重键改为 `sample_id`
- 训练输出路径开始向策略级入口过渡

---

## 3. 推荐实施顺序（严格按这个来）

1. **先读本文件**（2分钟内拿到当前状态）
2. **再读 `learnEngine/README.md` 的第 4、5、6 节**
   - 第 4 节：TODO
   - 第 5 节：执行顺序
   - 第 5.1 节：增量测试原则
   - 第 6 节：进度日志
3. **然后读计划文件**：
   - `/Users/liusonghao/.claude/plans/happy-brewing-wren.md`
4. **最后再打开代码**，顺序如下：
   - `strategies/base_strategy.py`
   - `strategies/sector_heat_strategy.py`
   - `learnEngine/dataset.py`
   - `learnEngine/factor_selector.py`
   - `train.py`
   - `agent_stats/agents/_model_signal_helper.py`

> 最省 token 的关键：**不要先通读 README 前 1-3 节**，那些是背景和目标，接手当前重构时不是第一优先级。

---

## 4. 关键代码定位（只看这些）

### 已完成 hook / 共享链路
- `strategies/base_strategy.py:111`
- `strategies/sector_heat_strategy.py:206`
- `strategies/sector_heat_strategy.py:256`
- `agent_stats/agents/_model_signal_helper.py:117`

### 当前旧假设的落点
- `learnEngine/dataset.py:120` → 仍按 `stock_code + trade_date` 去重
- `train.py:127` → 仍按 `stock_code + trade_date` 去重
- `learnEngine/factor_selector.py:64` → 仍按 `stock_code + trade_date` 去重
- `train.py:157` → 本地 `time_series_split`
- `learnEngine/factor_selector.py:55` → 本地 `_time_split`
- `learnEngine/factor_selector.py:108` → `_train_eval` 内部仍自带 split 逻辑

---

## 5. 必须遵守的工作方式

用户已经明确要求：

1. **每完成一个 TODO，立即更新文档**
   - 先更新 `learnEngine/README.md`
   - 再更新本接手文档（如果状态有变化）

2. **每完成一个 TODO，立即做模块测试**
   - 先 `py_compile`
   - 再做该 TODO 对应的最小 smoke test

3. **不要一口气做完 T6-T9 再统一测试**
   - 必须逐项完成、逐项验证、逐项写日志

---

## 6. 每个 TODO 的最低测试要求

### T6 完成后
- `python -m py_compile learnEngine/dataset.py strategies/sector_heat_strategy.py`
- dataset 最小 smoke：确认输出里出现 `sample_id` / `strategy_id`
- 确认没有再按 `(stock_code, trade_date)` 去重

### T7 完成后
- `python -m py_compile learnEngine/split_spec.py learnEngine/dataset.py learnEngine/factor_selector.py train.py`
- 生成一次 split spec
- 确认 selector/train 读的是同一份 split

### T8 完成后
- `python -m py_compile learnEngine/factor_selector.py`
- `python learnEngine/factor_selector.py --stage 12`
- `python learnEngine/factor_selector.py --n-trials 10`

### T9 完成后
- `python -m py_compile train.py learnEngine/model.py`
- `python train.py`
- 检查日志里是否包含：
  - strategy 过滤
  - frozen split 边界
  - 训练/验证样本量
  - 指标输出

---

## 7. 当前未解决的问题清单

- `dataset.py` 多策略循环：当前只调 `SectorHeatStrategy`，多策略时需遍历所有 `supports_ml_training()==True` 的策略
- `_model_signal_helper.py` 推理层改用 `get_runtime_model_path()` 路径加载模型（当前仍用旧路径）
- `strategy_configs.py`：后续策略接入时填入各自的 `strategy_specific_cols`
- `SectorHeatStrategy.build_training_candidates()` 的运行级 smoke 还没在正确解释器环境下补做
- `strategy_configs.py` 里后续策略的 `strategy_specific_cols` 待各策略实现后填入
- `dataset.py` 目前只循环调用 `SectorHeatStrategy`；多策略时需改为遍历所有支持 ML 的策略

---

## 8. 一句话接手指令

如果你是下一次 Claude：

**T1-T14 全部完成。接手任务是验收，读 `learnEngine/ACCEPTANCE_TODO.md`，按优先级从 H2 → F1 → D1 → E2 顺序推进未完成验收项。**

---

## 9. 文档索引

| 文档 | 用途 |
|------|------|
| `learnEngine/README.md` | TODO 列表 + 进度日志 |
| `learnEngine/HANDOFF_ML_PLATFORM.md` | 本文件：AI 接手上下文 |
| `learnEngine/ML_PLATFORM_GUIDE.md` | 使用说明：调用流程 / 新增因子 / 新增策略 / 调参 |
| `learnEngine/ACCEPTANCE_TODO.md` | 验收 TODO 清单（逐项勾选）|
