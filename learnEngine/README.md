# ML 中台重构总览

> 仅保留后续迭代真正需要的架构事实与关键约定。

## 1. 设计目标
- 从策略层统一收集支持 ML 的候选股逻辑。
- 生成全策略共享训练池，而不是为单策略各自拼训练脚本。
- 同一股票同一日允许因不同策略/板块形成多条样本，不能按 `(stock_code, trade_date)` 去重。
- dataset 生成后冻结 split，selector / train 只读取同一批 artifact。
- 训练归档与 runtime 严格分离，runtime 仅允许人工晋升。

## 2. 当前架构

### dataset artifact
```text
learnEngine/datasets/<dataset_run>/
  train_dataset.csv
  split_spec.json
  processed_dates.json
  dataset_manifest.json
  train_config.snapshot.json
  selected_features_<strategy_id>.json
```

### model / runtime artifact
```text
model/<strategy_id>/<version>/
  model.pkl
  model.json
  model_meta.json
  train_config.snapshot.json

strategies/<strategy_id>/<strategy_id>_strategy.py
strategies/<strategy_id>/runtime_model/
  <strategy_id>_V1.pkl
  <strategy_id>_V1.meta.json
```

约定：
- `model/<strategy>/<version>/` 只做训练归档。
- `strategies/<strategy>/runtime_model/` 只放人工晋升后的运行模型。
- runtime 目录必须恰好只有一个 `<strategy_id>_V*.pkl`；多个同时存在时直接报错。
- 训练流程不会自动改写 runtime。

### demo 路径
```text
learnEngine/datasets/acceptance_20260331_153415/
model/sector_heat/acceptance_v1/
strategies/sector_heat/sector_heat_strategy.py
strategies/sector_heat/runtime_model/sector_heat_V1.pkl
```

## 3. 样本语义
- `trade_date = D`
- 特征只使用 D 日收盘后及更早数据
- 标签表示 D+1 / D+2 的结果
- `feature_trade_date` 已删除

### 重要工程约定
项目通常在 **D+1 凌晨** 执行策略，而不是 D 日收盘瞬间执行；原因是 Tushare / 入库链路常常要数小时后才能拿到 D 日完整盘面。

因此：
- `daily_df=get_daily_kline_data(D)` 的显式注入是有意设计
- 某些 `prev_trade_date` / “上一完整交易日盘面”表述，实际语义是“当前触发时刻可稳定获得的最近一个完整收盘日盘面”
- 不要仅凭命名把这条链路改成 `daily_df=None`

## 4. 核心调用链
```text
Strategy.build_training_candidates(trade_date, daily_df)
  -> candidate_df + context
  -> FeatureDataBundle
  -> FeatureEngine.run_single_date()
  -> LabelEngine.generate_single_date(D)
  -> train_dataset.csv
  -> split_spec.json
  -> factor_selector.py
  -> selected_features_<strategy_id>.json
  -> train.py
  -> model/<strategy_id>/<version>/...
```

## 5. 当前已落地的关键点
- 多策略样本汇总已打通，样本唯一键为 `sample_id`
- frozen split spec 已落地，selector / train 读取同一份切分
- `factor_selector.py` / `train.py` 按 `strategy_id` 过滤自己的样本
- `SectorHeatStrategy`、`dataset.py`、`_model_signal_helper.py` 已共用候选池主链路
- runtime 路径统一由 `cfg.get_strategy_runtime_model_path()` 解析

## 6. 已验证基线
- acceptance dataset：`learnEngine/datasets/acceptance_20260331_153415/`
- rows / cols：`302 / 364`
- split：`train=206 / val=96`
- selector Stage12：通过
- selector Optuna10：通过
- train.py：通过
- H3 推理 smoke：通过（`2026-03-27`，成功产出概率信号）

## 7. 未来两阶段开盘架构：仅预留，不启用
当前主链路仍是单阶段训练；本次只在 schema / 文档层为未来实验预留空间。

### 预留 label 名称
- `label_open_regime_stage1`
- `label_open_regime_stage1_bin`
- `label_open_regime_stage2`
- `label_open_regime_stage2_bin`

### 预留 feature 名称
- `feat_pred_open_regime_stage1`
- `feat_pred_open_regime_stage1_prob_low`
- `feat_pred_open_regime_stage1_prob_mid`
- `feat_pred_open_regime_stage1_prob_high`
- `feat_pred_open_regime_stage2`
- `feat_pred_open_regime_stage2_conf`

约定：
- 当前 dataset 仅占位这些列
- 当前 selector / train 一律排除这些列
- 当前不会生成真实值，也不会作为训练目标

### 未来路线
- 阶段 A：用 D 日因子预测 D+1 开盘状态
- 阶段 B：拼接 A 的 OOF / 实盘预测输出，预测开盘买入后的收益质量
- 实盘先跑 A，再按 A 的输出驱动 B

## 8. 当前仍值得保留的补测
1. C2/C3：因子层运行级验证
2. B4：split_spec 独立读写测试
3. J1/J2：固定日期推理稳定性 / 旧模型 pickle 兼容性

## 9. 已知非阻塞项
- `_model_signal_helper.py` runtime 输出可能出现重复股票；当前按用户要求暂不修复。
