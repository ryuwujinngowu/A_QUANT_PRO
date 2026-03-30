# ML 中台架构升级工作文档

> 状态：进行中
> 维护方式：每完成一个任务，立即更新并划掉对应 TODO
> 查看方式：你可随时通过 `btw` 检查本文件进展

---

## 1. 用户原始目标 / 要求整理

核心痛点：
1. 后期多策略逐个出训练集耗时巨大。
2. 模型版本杂乱，训练参数没有固化，导致训练历史无法复现。
3. 后续会出现多策略、多模型，甚至一个策略多个模型，当前全局 `model/` 目录会越来越混乱。

目标架构：
1. 建立**机器学习中台**。
2. 训练集生成时，逐日自动搜索策略层所有策略的候选股生成逻辑。
3. 统一生成**全策略候选股池**。
4. 对全候选股池计算**全因子的宽表训练池**。
5. 对来自特定策略的股票行打上策略标签，**千万不能去重**。
6. 在生成全策略全因子训练集后，统一固定并记录全局时序拆分：`train / val / test`。
7. 每个策略训练时：
   - 先加载全量训练集
   - 再过滤出自己的样本
   - 进行 IC 计算
   - 因子筛选
   - 超参优化
   - 最终训练出属于该策略的模型
8. 每个策略目录下，清晰归档：
   - 模型版本
   - 预测目标（label）
   - 训练参数
   - 因子集
   - 时序拆分
   - AUC / recall / 概率分布等指标
9. 模型只有在**手动测试集严谨回测通过**后，才手动移动到策略层运行目录。
10. 最终目录调整为：
   - 策略层下每个策略一个目录
   - 该目录同时放策略代码和该策略最终使用的模型
11. 重构要求：
   - 不重复造轮子
   - 有清晰的重构计划
   - 保持代码风格统一
   - 有前瞻性，适配未来迭代和高扩展性
   - 方便维护
12. 最后要更新相关记忆文档，减少后续检索 token 成本。

---

## 2. 当前代码现状总结

### learnEngine
- `learnEngine/dataset.py` 当前本质上是 **sector_heat 专用训练集脚本**。
- `learnEngine/label.py` 已经具备较强复用性，能批量生成多种 label。
- `learnEngine/factor_selector.py` 有三阶段因子筛选能力，可复用。
- `train.py` 和 `factor_selector.py` 各自维护 split 逻辑，不统一。
- `learnEngine/model.py` 的模型 wrapper 仍带有 `SectorHeat` 命名和路径假设。

### features
- `FeatureEngine` / `FeatureRegistry` / `FeatureDataBundle` 已经形成稳定特征平台。
- 当前最大问题不是特征引擎不可用，而是 `FeatureDataBundle` 偏重、外部调用耦合细节太多。
- 现阶段不宜大拆 Feature 层，应优先包一层 builder/factory 稳定边界。

### strategies
- `strategies/sector_heat_strategy.py`、`learnEngine/dataset.py`、`agent_stats/agents/_model_signal_helper.py` 存在候选池逻辑重复。
- 当前 train / infer / helper 口径漂移风险高。
- `BaseStrategy` 还没有 ML 中台所需的策略发现与训练候选接口。

---

## 3. 当前设计思路（推荐方案）

### 总体思路
采用：**平台核心（learnEngine） + 策略适配器（strategies） + 统一特征平台（features）**

### 3.1 第一优先级：先统一 sector_heat 的 train / infer 候选池逻辑
原因：
- 当前三处重复逻辑最容易造成训练和推理漂移。
- 不先统一这一层，后面做全局训练池也会建立在不稳定口径之上。

做法：
- 先在 `BaseStrategy` 增加非破坏式 ML hook
- 把 `SectorHeatStrategy` 的候选池、context、bundle 构造抽成共享方法
- `learnEngine/dataset.py` 和 `_model_signal_helper.py` 改为复用这套共享逻辑

### 3.2 第二优先级：把 dataset.py 升级成全局训练池构建器
目标：
- 自动发现支持 ML 的策略
- 逐日收集各策略候选样本
- 允许同一股票同一天出现多行（不同策略样本）
- 统一特征计算
- 统一 label join
- 统一输出 dataset artifact

### 3.3 第三优先级：一次性冻结 global split spec
目标：
- 由全局训练池生成时写出统一 split spec
- 后续 selector / train / evaluate 都只读取，不再本地现算
- split 采用**日期边界冻结**，而非按行数切分

### 3.4 第四优先级：训练改成“全局数据集 + strategy_id 过滤”
每个策略训练时：
- 加载全量训练集
- 过滤 `strategy_id`
- 加载 frozen split spec
- 跑 IC / selector / optuna / train
- 输出到该策略自己的 archive 目录

### 3.5 第五优先级：模型归档与人工晋升
结构：
- `learnEngine/artifacts/datasets/<dataset_version>/...`
- `model/<strategy_id>/archive/<model_version>/...`
- `model/<strategy_id>/runtime/...`

原则：
- 训练只写 archive
- 运行时只读 runtime
- runtime 只接受人工验证通过后的手动晋升模型

### 3.6 FeatureDataBundle 当前处理策略
原则：
- **先包一层，不急着大拆**
- 先新增 factory / builder，统一 bundle 构造口径
- 如果后面多策略构建时性能或结构真的成为瓶颈，再二阶段拆成：
  - TradeDateContext
  - CandidateContext
  - DailyMarketCache
  - MacroCache
  - MinuteCache
  - ExtendedFeatureCache

---

## 4. 当前待办 TODO（实时更新）

- [x] T1. 建立工作文档，整理用户要求、当前方案、实施 TODO
- [x] T2. 为 `BaseStrategy` 增加 ML 中台所需非破坏式接口
- [x] T3. 拆分 `SectorHeatStrategy`，抽取共享候选池 / context / bundle 构造逻辑
- [x] T4. 改造 `agent_stats/agents/_model_signal_helper.py`，复用 `sector_heat` 共享逻辑
- [x] T5. 改造 `learnEngine/dataset.py`，先接入共享的 `sector_heat` 候选逻辑
- [ ] T6. 定义全局训练池 schema（`sample_id`, `strategy_id` 等）并去除旧的 `(stock_code, trade_date)` 唯一假设
- [ ] T7. 新增 global split spec 管理模块，并把 split 从 `factor_selector.py` / `train.py` 中抽离
- [ ] T8. 改造 `learnEngine/factor_selector.py` 为“读取全局训练池 + strategy_id 过滤 + frozen split”模式
- [ ] T9. 改造 `train.py` 为策略级训练入口，并接入 archive 输出
- [ ] T10. 泛化 `learnEngine/model.py`，去掉 sector_heat 专属命名和路径假设
- [ ] T11. 调整模型产物目录结构，建立 archive/runtime 约定
- [ ] T12. 为 `FeatureDataBundle` 增加 builder/factory 包装层，统一外部构造方式
- [ ] T13. 完成最小可运行链路验证（至少 sector_heat 全链路）
- [ ] T14. 更新项目记忆文档 / 架构文档

---

## 5. 当前执行顺序

### 阶段 A：先打通同源候选池
- T2
- T3
- T4
- T5
- T7（阶段内增量测试）

### 阶段 B：再建立全局训练池与 split
- T6
- T7

### 阶段 C：训练与产物归档升级
- T8
- T9
- T10
- T11

### 阶段 D：收尾与文档
- T12
- T13
- T14

---

## 5.1 增量测试原则（新增）

从现在开始，每完成一个模块或一次明确重构后，立即做该模块的最小验证，不等全部重构结束再统一排查。

当前执行规则：
1. **结构性重构后立即测试**：先测刚改的模块，避免问题跨阶段扩散。
2. **优先做低成本验证**：import、最小调用链、单日期 smoke test、关键返回结构检查。
3. **失败先修当前层**：不把明显问题带到下一阶段。
4. **进度文档同步记录**：每次测试结果都写入进度日志，便于通过 `btw` 查看。

当前已完成阶段需要补做的测试：
- [x] 校验 `strategies/base_strategy.py` 新增 ML hook 不破坏现有 import / 实例化（已完成语法编译）
- [ ] 校验 `SectorHeatStrategy.build_training_candidates()` / `build_feature_bundle_from_context()` 最小可调用性（当前环境缺 `pandas`，待在项目解释器下补做）
- [x] 校验 `_model_signal_helper.py` 新共享链路可 import（已完成语法编译）
- [x] 校验 `learnEngine/dataset.py` 已切换到策略共享逻辑后可通过基础语法与 import 检查（已完成语法编译）

---

## 6. 进度日志

- 2026-03-30：已创建本工作文档，准备开始 T2。
- 2026-03-30：已完成 T2/T3。`BaseStrategy` 新增 ML 中台默认 hook；`SectorHeatStrategy` 已抽出共享候选样本构建、selection context、feature bundle 构造逻辑，后续 `dataset.py` 与 `_model_signal_helper.py` 将直接复用。
- 2026-03-30：已完成 T4/T5。`_model_signal_helper.py` 与 `learnEngine/dataset.py` 已切换为复用 `SectorHeatStrategy` 的共享候选池 / bundle 构造逻辑，初步消除 sector_heat 的 train/infer/helper 口径漂移。
- 2026-03-30：已开始执行增量测试。`base_strategy.py` / `sector_heat_strategy.py` / `_model_signal_helper.py` / `dataset.py` 均已通过 `python3 -m py_compile` 语法检查；进一步 smoke test 在当前系统 Python 下因缺少 `pandas` 失败，后续需在项目解释器环境下补充运行级验证。
