# A_QUANT Regime-Strategy 分层模型 ROADMAP

> **Last Updated**: 2026-03-24
> **Branch**: `claude/check-latest-commit-y3f6a`
> **Architecture**: Regime 决策 → Strategy 权重动态分配 → 买入信号
> **核心原则**: 极简，清晰的数据流，每一步可独立验收

---

## 📋 项目目标

构建两层决策系统，用于优化每日买入信号：

1. **Regime 层**: 市场状态三分类（BULL/BEAR/CHOPPY），输出置信度和仓位限制
2. **Strategy 层**: 三个独立的选股策略（短线隔日/D+2/D+3），基于 Regime 动态权重分配
3. **最终输出**: 买入信号（考虑了市场风险和多策略融合）

---

## 🎯 执行阶段

### **Phase 1: Regime 标签定义 & 训练集生成**

**任务序号**: `M1.1 ~ M1.4`

#### M1.1: 定义 Regime 标签生成规则
- **输入**: T+1 当日收盘数据（limit_up_count, limit_down_count, index_pct_chg, adapt_score）
- **输出**: regime 标签 ∈ {0:BULL, 1:BEAR, 2:CHOPPY}
- **核心规则**（写成函数 `label_regime(row) -> int`）:
  ```python
  BEAR (高风险): index_pct_chg < -2.0 OR limit_down_count > 20
  BULL (低风险): index_pct_chg > 1.5 AND limit_up_count > 80 AND adapt_score > 60
  CHOPPY (中等风险): 其他
  ```
- **验收标准**:
  - [ ] 新建文件 `learnEngine/labels/regime_label.py`，包含 `label_regime(row) -> int` 函数
  - [ ] 编写单元测试，至少 5 个边界用例（极端行情、正常行情、震荡）
  - [ ] 回测数据 2024-11-01 ~ 2026-03-24，验证标签分布合理（3 个类别均衡）

---

#### M1.2: 改造 dataset.py 生成 regime 标签列
- **现状**: dataset.py 生成 label1/label2（个股标签）
- **改造**: 在 CSV 中新增 `regime` 列（全局标签，每天一个值）
- **具体改动**:
  - 在 `dataset.py` 的主循环中，每日调用 `label_regime(market_stats)` 生成标签
  - 将标签列添加到 CSV，与特征并列
- **输入数据来源**:
  - 现有的 `market_macro_feature` 已经计算了 limit_up/down_count, index_pct_chg, adapt_score
  - 这些数据已在 CSV 的特征列中，无需额外 API 调用
- **验收标准**:
  - [ ] `train_dataset_final.csv` 新增 `regime` 列
  - [ ] 运行一个完整日期（如 2026-03-23），验证 regime 列有值且≠NaN
  - [ ] 检查 CSV 行数和列数无异常

---

#### M1.3: 设计 Regime 训练集的样本加权策略
- **目标**: 通过加权，让模型学到"极端行情和误判都很关键"
- **权重规则**（写成函数 `compute_regime_sample_weight(row, regime_label) -> float`）:
  ```python
  # 极端行情（limit_up > 80 或 limit_down > 20）
  if limit_up_count > 80 OR limit_down_count > 20:
      weight = 2.0

  # 致命误判：预测BULL但实际BEAR（反向操作风险最大）
  if predicted_regime == 0 AND true_regime == 1:
      weight = 3.0

  # 其他情况
  else:
      weight = 1.0
  ```
- **验收标准**:
  - [ ] 新建文件 `learnEngine/objectives/regime_sample_weight.py`
  - [ ] 函数签名、文档、单元测试完整
  - [ ] 统计样本权重分布（平均值、中位数、max），验证权重合理

---

#### M1.4: 整合 M1.1~M1.3，生成最终 Regime 训练集
- **目标**: 准备好 (X_train, y_train, sample_weight) 供 XGBoost 训练
- **步骤**:
  1. 从 CSV 读取 regime 列和全局特征（market_macro 系列）
  2. 分离特征 X 和标签 y
  3. 计算样本权重
  4. 时序 80/20 分割成 train/val
  5. 输出：train_regime_X.npz, train_regime_y.npy, train_regime_weight.npy 等（或存为 pickle）
- **新建文件**: `learnEngine/dataset_regime.py`
  - 函数: `prepare_regime_dataset(csv_path: str) -> (X_train, X_val, y_train, y_val, weight_train, weight_val, feature_names)`
- **验收标准**:
  - [ ] 脚本运行无错误
  - [ ] 输出数据形状正确（train: 80%行数，val: 20%行数）
  - [ ] 样本权重统计合理（无NaN/inf）
  - [ ] 特征列数 = market_macro 特征数（~10列）

---

### **Phase 2: Regime 模型训练 & 校准**

**任务序号**: `M2.1 ~ M2.3`

#### M2.1: 基础 Regime XGBoost 模型训练
- **目标**: 训练三分类 Regime 模型，使用加权损失
- **新建文件**: `learnEngine/models/regime_model.py`
  - 类: `RegimeModel(BaseModel)` (继承一个简单的 BaseModel 抽象类或直接写)
  - 方法:
    - `__init__(params=None)`: 初始化参数（max_depth=4, learning_rate=0.1, objective='multi:softprob' 等）
    - `train(X_train, X_val, y_train, y_val, sample_weight_train=None)`: 用加权损失训练
    - `predict_with_confidence(X)`: 返回 (predicted_class, max_probability) 元组
    - `save_model(path)`, `load_model(path)`: 持久化
- **核心参数**:
  ```python
  params = {
      'objective': 'multi:softprob',  # 三分类
      'eval_metric': 'mlogloss',       # 多分类对数损失
      'max_depth': 4,
      'learning_rate': 0.1,
      'subsample': 0.8,
      'colsample_bytree': 0.8,
      'num_class': 3,  # BULL/BEAR/CHOPPY
  }
  ```
- **加权损失**: XGBoost 原生支持 `sample_weight` 参数
- **验收标准**:
  - [ ] 模型文件 `learnEngine/models/regime_model.py` 完整
  - [ ] 训练脚本可运行: `python -c "from learnEngine.models.regime_model import RegimeModel; rm = RegimeModel()"`
  - [ ] 输出：模型文件、训练日志（train/val loss）
  - [ ] 评估指标：val 准确率 > 65%（基准）

---

#### M2.2: 温度校准（Temperature Scaling）
- **目标**: 让模型的输出概率与实际准确率匹配，避免 overconfident
- **方法**: 在验证集上找最优的温度参数 T，使得 ECE (Expected Calibration Error) < 5%
- **新建函数**: `learnEngine/objectives/regime_calibration.py`
  - 函数: `calibrate_temperature(model, X_val, y_val, search_range=[0.5, 2.0]) -> (best_T, calibrated_model)`
  - 内部使用 Platt Scaling 或简单的温度搜索
- **验收标准**:
  - [ ] 校准脚本完整
  - [ ] 输出最优温度参数 T（如 T=1.2）
  - [ ] ECE 计算正确，< 5% 为合格
  - [ ] 校准前后的概率对比（可视化或数值比对）

---

#### M2.3: 历史 Regime 推理 & 置信度计算
- **目标**: 对全历史数据（2024-11-01 ~ 2026-03-24）推理 Regime，计算置信度
- **新建脚本**: `learnEngine/train_regime.py`
  - 函数: `infer_regime_history(model, calibration_T, csv_path, output_path)`
  - 逻辑:
    1. 加载已训练的 Regime 模型 + 校准参数
    2. 逐日推理，得到 (regime, p_raw)
    3. 应用温度校准：p_calibrated = softmax(logits / T)
    4. 计算历史准确率（过去 20 天）
    5. 最终置信度 = 0.7 * p_calibrated + 0.3 * acc_history_20d
    6. 生成输出 CSV：trade_date, regime, p_bull, p_bear, p_choppy, regime_confidence
- **输出文件**: `learnEngine/regime_history.csv`
  - 列: `trade_date, regime, p_bull, p_bear, p_choppy, regime_confidence, position_size_coeff`
  - position_size_coeff = 1.0 if confidence > 0.75 else (0.5 if confidence > 0.60 else 0.0)
- **验收标准**:
  - [ ] 脚本运行无错误
  - [ ] 输出 CSV 有数据，行数 = 交易日数
  - [ ] regime_confidence ∈ [0, 1] 且无 NaN
  - [ ] 置信度分布合理（不是全 0.9x 或全 0.5x）

---

### **Phase 3: Strategy 模型架构 & 训练集拆分**

**任务序号**: `M3.1 ~ M3.3`

#### M3.1: 设计 Strategy 训练集的 Regime 分层
- **目标**: 按 Regime 拆分训练数据，生成 3 个纯净的子集
- **输入**: `train_dataset_final.csv` + 来自 M2.3 的 `regime_history.csv`
- **步骤**:
  1. 合并两个 CSV（按 trade_date）
  2. 按 regime 值分组：data_bull, data_bear, data_choppy
  3. 每个子集内部做 80/20 时序分割
- **新建脚本**: `learnEngine/dataset_strategy.py`
  - 函数: `prepare_strategy_datasets(csv_path, regime_csv_path) -> Dict[str, (X_train, X_val, y_train, y_val)]`
  - 返回: {'bull': (...), 'bear': (...), 'choppy': (...)}
- **验收标准**:
  - [ ] 脚本运行无错误
  - [ ] 三个子集的行数都 > 100（保证样本充足）
  - [ ] 每个子集的 label1 分布（正样本率）输出到日志（用于理解数据特性）

---

#### M3.2: 设计 Strategy 的风险敏感样本加权
- **目标**: 高风险行情下的预测错误要更重地惩罚
- **权重规则**（写成函数 `compute_strategy_sample_weight(row, regime, label1) -> float`）:
  ```python
  # 高风险行情（BEAR 或 CHOPPY 阶段）
  if regime in [1, 2]:  # BEAR or CHOPPY
      if label1 == 0:  # 预测错（应该涨但跌了）
          weight = 2.0
      else:
          weight = 1.0

  # 低风险行情（BULL 阶段）
  elif regime == 0:  # BULL
      if label1 == 0:  # 负样本在牛市中也要重视（可能是高风险个股）
          weight = 1.5
      else:
          weight = 1.0

  return weight
  ```
- **新建文件**: `learnEngine/objectives/strategy_sample_weight.py`
- **验收标准**:
  - [ ] 函数实现完整
  - [ ] 单元测试（至少 3 个 case）
  - [ ] 样本权重统计输出

---

#### M3.3: 整合 M3.1~M3.2，生成 3 个 Strategy 训练集
- **新建脚本**: `learnEngine/dataset_strategy_final.py`
  - 函数: `prepare_regime_wise_strategy_datasets(...) -> Dict[str, Dict]`
  - 返回每个 Regime 子集的：X_train, X_val, y_train, y_val, weight_train, weight_val, feature_names
- **验收标准**:
  - [ ] 脚本运行无错误
  - [ ] 输出 3 个子集，每个都有完整的 train/val + weights
  - [ ] 数据形状和类型检查无异常

---

### **Phase 4: Strategy 模型训练**

**任务序号**: `M4.1 ~ M4.2`

#### M4.1: 构建 BaseModel 抽象基类
- **目标**: 为 RegimeModel 和 StrategyModel 提供统一接口
- **新建文件**: `learnEngine/models/base_model.py`
  - 类: `BaseModel(ABC)`
  - 抽象方法: `train()`, `predict()`, `predict_proba()`, `save_model()`, `load_model()`
  - 通用方法: `get_feature_importance()` 等
- **验收标准**:
  - [ ] 抽象类定义完整，可继承
  - [ ] RegimeModel 改为继承 BaseModel

---

#### M4.2: 训练 3 个独立的 Strategy 子模型
- **目标**: 分别在 BULL/BEAR/CHOPPY 数据上训练二分类模型
- **新建文件**: `learnEngine/models/strategy_model.py`
  - 类: `StrategyModel(BaseModel)`
  - 方法: 同 RegimeModel，但 objective='binary:logistic'（二分类）
- **新建脚本**: `learnEngine/train_strategy.py`
  - 函数: `train_all_regime_wise_models(dataset_dict) -> Dict[str, StrategyModel]`
  - 逻辑:
    1. 对 'bull', 'bear', 'choppy' 分别创建一个 StrategyModel
    2. 用各自的 train/val 数据和权重训练
    3. 评估 AUC, Precision, Recall（分别输出）
    4. 保存 3 个模型文件
- **验收标准**:
  - [ ] 3 个模型都训练成功
  - [ ] 每个模型的 val AUC > 55%（基准）
  - [ ] 模型文件可加载
  - [ ] 特征重要性输出（top 20）

---

### **Phase 5: 实盘推理链路**

**任务序号**: `M5.1 ~ M5.3`

#### M5.1: Regime 每日推理模块
- **目标**: 每日收盘后，推理当日的 Regime 及置信度
- **新建模块**: `learnEngine/inference/regime_inference.py`
  - 类: `RegimeInference`
  - 方法:
    - `__init__(model_path, calibration_T)`
    - `infer(market_features) -> (regime, confidence, position_size_coeff)`
  - 逻辑:
    1. 加载今日市场特征（limit_up/down_count 等）
    2. 推理得到 (regime, p_raw)
    3. 应用温度校准和历史准确率加权
    4. 根据 confidence 计算仓位系数
- **验收标准**:
  - [ ] 模块可导入
  - [ ] 模拟推理一次，输出合理的 regime 和 confidence
  - [ ] 返回值类型和范围正确

---

#### M5.2: Strategy 权重动态分配模块
- **目标**: 基于 Regime，决定 3 个 Strategy 的权重
- **新建模块**: `learnEngine/inference/strategy_weight_allocator.py`
  - 函数: `allocate_weights(regime: int, regime_confidence: float, strategy_perf_history: Dict) -> Dict[str, float]`
  - 逻辑:
    ```python
    if regime == 0:  # BULL
        weights = {'short_d1': 0.3, 'short_d2': 0.35, 'short_d3': 0.35}
    elif regime == 1:  # BEAR
        weights = {'short_d1': 0.5, 'short_d2': 0.3, 'short_d3': 0.2}
    else:  # CHOPPY
        weights = {'short_d1': 0.4, 'short_d2': 0.35, 'short_d3': 0.25}

    # 根据 regime_confidence 进一步调整
    if regime_confidence < 0.65:
        weights = {k: v * 0.7 for k, v in weights.items()}  # 降低所有权重

    # 根据过去 20 天的策略表现微调（如果某策略绩效低，降低权重）
    for strategy, perf in strategy_perf_history.items():
        if perf < 0.5:  # 负期望收益
            weights[strategy] *= 0.8

    # 归一化
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    return weights
    ```
  - 返回: {'short_d1': w1, 'short_d2': w2, 'short_d3': w3}（sum=1.0）
- **验收标准**:
  - [ ] 函数实现完整
  - [ ] 返回的权重和为 1.0，每个权重 ∈ [0, 1]
  - [ ] 单元测试（3-4 个 case）

---

#### M5.3: Strategy 子模型联合推理选股模块
- **目标**: 并行运行 3 个 Strategy 子模型，加权融合个股得分
- **新建模块**: `learnEngine/inference/strategy_inference.py`
  - 类: `StrategyInference`
  - 方法:
    - `__init__(models_dict, weights_dict)` 加载 3 个模型和权重
    - `infer(stock_features, regime) -> stock_scores` 返回每只股票的融合得分
  - 逻辑:
    1. 根据当前 regime，选择对应的 StrategyModel（如果 regime=0/BULL，则只用 bull model；其他情况用对应 model）
    2. 或者：并行运行 3 个模型，基于权重融合概率
    3. 对每只股票的 predict_proba 进行加权平均
    4. 返回每只股票的融合得分（概率 ∈ [0, 1]）
- **验收标准**:
  - [ ] 模块可导入
  - [ ] 推理返回正确的股票得分向量
  - [ ] 融合逻辑清晰（基于 regime 或加权平均都可以）

---

#### M5.4: T+1 开盘动态持仓决策模块
- **目标**: T+1 09:45，根据开盘 15 分钟信号做最终决策
- **新建模块**: `learnEngine/inference/morning_decision.py`
  - 函数: `make_morning_decision(candidate_stocks: Dict, market_signal: Dict, regime_confidence: float) -> Dict[str, str]`
  - 输入:
    - candidate_stocks: {'ts_code': stock_score, ...}（来自 M5.3）
    - market_signal: {'open_15min_return': float, 'open_15min_volume_ratio': float, ...}
    - regime_confidence: 昨日的 Regime 置信度
  - 输出: {'ts_code': 'HOLD'|'STOP_PROFIT'|'STOP_LOSS', ...}
  - 规则:
    ```python
    if open_15min_return > 0.01 and open_15min_volume_ratio > 1.2:
        decision = 'HOLD'
    elif open_15min_return < -0.005 or open_15min_volume_ratio < 0.8:
        decision = 'STOP_LOSS'
    else:
        decision = 'HOLD' if regime_confidence > 0.6 else 'STOP_PROFIT'
    ```
- **验收标准**:
  - [ ] 函数实现完整
  - [ ] 单元测试（至少 3 个 case）

---

### **Phase 6: Agent 集成 & 实盘测试**

**任务序号**: `M6.1 ~ M6.3`

#### M6.1: 改造现有 Agent（short_d1, short_d2, short_d3）
- **目标**: 现有 Agent 接收动态权重和 Regime 信息
- **改造范围**:
  - `agent_stats/agents/high_position_stock.py`（short_d1 隔日）
  - 新增或改造 D+2 和 D+3 策略（或复用现有逻辑）
- **改造内容**:
  - `get_signal_stock_pool()` 接收额外参数：`regime_weight=1.0`
  - 返回的信号中添加 `weight` 字段，表示该策略的配置权重
- **验收标准**:
  - [ ] 现有 Agent 可以接收权重参数
  - [ ] 返回信号格式保持向后兼容

---

#### M6.2: 改造 long_stats_engine.py，集成 Regime & Strategy 推理
- **目标**: 在 run_full_flow() 中调用新的推理模块
- **改造点**:
  1. 每日开盘前（或收盘后），调用 RegimeInference
  2. 根据 Regime 调用 StrategyWeightAllocator
  3. 遍历所有 Agent（short_d1/d2/d3），用权重调整返回的信号
  4. 合并所有信号（考虑权重），输出最终的买入池
  5. T+1 09:45，调用 MorningDecision
- **新建文件**: `agent_stats/engine_regime_strategy.py`（或改造现有 long_stats_engine.py）
  - 函数: `run_regime_aware_flow(trade_date, mode='daily')`
- **验收标准**:
  - [ ] 新流程可运行
  - [ ] 买入信号数量和质量不下降
  - [ ] 日志输出 Regime、权重、最终买入池大小

---

#### M6.3: 回测验证与上线
- **目标**: 对比新旧系统，验证改进效果
- **测试计划**:
  1. 用历史数据（2026-01-01 ~ 2026-03-24）回测
  2. 统计指标：
     - 日均买入信号数
     - 日均夏普比、年化收益、最大回撤
     - 与无 Regime 约束的基线对比（收益、回撤、Sharpe）
  3. 对比两个版本的买入池，分析 Regime 的影响
- **验收标准**:
  - [ ] 回测可运行，输出完整报告
  - [ ] Sharpe 比 >= 基线（或持平，且回撤显著降低）
  - [ ] 无 bug，可上线

---

## 📊 数据流与文件对应

```
M1: Regime 标签 & 训练集
  ├─ learnEngine/labels/regime_label.py           (M1.1)
  ├─ learnEngine/dataset.py (改造)                (M1.2)
  ├─ learnEngine/objectives/regime_sample_weight.py (M1.3)
  └─ learnEngine/dataset_regime.py                (M1.4)
       ↓ 输出: regime_X_train.npz, regime_y_train.npy, ...

M2: Regime 模型 & 校准
  ├─ learnEngine/models/regime_model.py           (M2.1)
  ├─ learnEngine/objectives/regime_calibration.py (M2.2)
  ├─ learnEngine/train_regime.py                  (M2.3)
  └─ 输出: regime_model.pkl, regime_history.csv

M3: Strategy 训练集拆分
  ├─ learnEngine/dataset_strategy.py              (M3.1)
  ├─ learnEngine/objectives/strategy_sample_weight.py (M3.2)
  └─ learnEngine/dataset_strategy_final.py        (M3.3)
       ↓ 输出: strategy_*_train.npz, strategy_*_val.npy, ...

M4: Strategy 模型训练
  ├─ learnEngine/models/base_model.py             (M4.1)
  ├─ learnEngine/models/strategy_model.py         (M4.2)
  ├─ learnEngine/train_strategy.py                (M4.2)
  └─ 输出: strategy_bull.pkl, strategy_bear.pkl, strategy_choppy.pkl

M5: 实盘推理
  ├─ learnEngine/inference/regime_inference.py    (M5.1)
  ├─ learnEngine/inference/strategy_weight_allocator.py (M5.2)
  ├─ learnEngine/inference/strategy_inference.py  (M5.3)
  └─ learnEngine/inference/morning_decision.py    (M5.4)

M6: Agent 集成
  ├─ agent_stats/agents/* (改造)                  (M6.1)
  ├─ agent_stats/engine_regime_strategy.py        (M6.2)
  └─ 回测验证                                     (M6.3)
```

---

## ✅ 验收标准总表

| 阶段 | 任务 | 关键输出 | 验收条件 |
|------|------|---------|---------|
| M1.1 | Regime 标签规则 | regime_label.py | 5 个单元测试通过，标签分布合理 |
| M1.2 | dataset.py 改造 | train_dataset_final.csv + regime 列 | CSV 可读，regime 列无 NaN |
| M1.3 | 样本权重设计 | regime_sample_weight.py | 权重函数正确，统计合理 |
| M1.4 | 训练集生成 | dataset_regime.py + train/val data | 数据形状 80/20 分割，无 NaN |
| M2.1 | Regime 模型训练 | regime_model.py + val AUC>65% | 模型可加载，评估指标达标 |
| M2.2 | 温度校准 | ECE < 5% | 校准前后概率对比显示改进 |
| M2.3 | 历史 Regime 推理 | regime_history.csv | 全历史数据推理完毕，置信度分布正常 |
| M3.1 | Regime 分层 | 3 个子集 (bull/bear/choppy) | 子集行数都 > 100 |
| M3.2 | 风险加权设计 | strategy_sample_weight.py | 权重函数正确 |
| M3.3 | 最终训练集 | 3 × (X_train, y_train, weight, ...) | 数据完整，形状无异常 |
| M4.1 | BaseModel 抽象 | base_model.py | 可被 RegimeModel/StrategyModel 继承 |
| M4.2 | Strategy 模型训练 | 3 个 .pkl 模型 + val AUC>55% | 模型可加载，特征重要性输出 |
| M5.1 | Regime 推理模块 | regime_inference.py | 推理一次输出正确，置信度 ∈ [0,1] |
| M5.2 | 权重分配模块 | strategy_weight_allocator.py | 返回权重和=1.0，分布合理 |
| M5.3 | Strategy 融合推理 | strategy_inference.py | 推理返回正确的股票得分 |
| M5.4 | 早盘决策模块 | morning_decision.py | 返回决策字典，decision ∈ {HOLD,STOP} |
| M6.1 | Agent 改造 | 改造后的 Agent 接收权重参数 | Agent 可运行，返回信号完整 |
| M6.2 | 引擎集成 | engine_regime_strategy.py | 新流程可运行，输出买入池 |
| M6.3 | 回测验证 | 完整的回测报告（Sharpe、收益、回撤） | Sharpe >= 基线，回撤降低 |

---

## 🚀 上线清单

- [ ] 所有代码通过 code review
- [ ] 单元测试覆盖 >= 80%
- [ ] 回测验证完毕，指标达标
- [ ] 实盘前 dry run（输出买入信号但不下单）
- [ ] 监控告警配置（Regime 切换、异常信号）
- [ ] 更新 CLAUDE.md，记录 Regime & Strategy 架构
- [ ] 文档完整（API、参数、故障排查）

---

## 📝 进度追踪（每次迭代后更新）

### Iteration 1 (2026-03-24 ~ ?)
- [ ] M1.1 - M1.4: Regime 标签 & 训练集（计划 1 天）
- [ ] M2.1 - M2.3: Regime 模型 & 校准（计划 1 天）
- [ ] M3.1 - M3.3: Strategy 训练集拆分（计划 1 天）
- [ ] M4.1 - M4.2: Strategy 模型训练（计划 1 天）

### Iteration 2
- [ ] M5.1 - M5.4: 实盘推理模块（计划 1.5 天）
- [ ] M6.1 - M6.3: Agent 集成 & 回测（计划 2 天）

### Post-Launch
- [ ] 生产环境监控
- [ ] 根据实盘效果微调权重和规则

---

**当前状态**: 待开始 (准备完毕)
**最后更新**: 2026-03-24
