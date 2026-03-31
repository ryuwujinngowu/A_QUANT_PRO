# ML 验收最小 TODO

> 只保留仍有价值的验收项。

## 已完成
- 编译检查：`dataset.py / dataset_acceptance.py / factor_selector.py / train.py / bundle_factory.py / strategies/sector_heat/sector_heat_strategy.py / high_low_switch_ml_strategy.py / _model_signal_helper.py`
- acceptance dataset 主链路：通过
- selector Stage12：通过
- selector Optuna10：通过
- train.py：通过
- H3 端到端推理 smoke：通过（历史日期 `2026-03-27` 成功产出概率列表）
- dataset-run / split / selected_features / model archive 路径一致性：通过
- runtime 路径迁移：通过（`strategies/sector_heat/runtime_model/sector_heat_V1.pkl`）
- `feature_trade_date` 删除回归：通过
- `adapt_score` 在 `sector_heat` 训练特征中保留：通过
- `EXCLUDE_COLS` 覆盖 label 列：通过
- future 两阶段预留列已纳入 schema / `EXCLUDE_COLS`，当前不参与训练：通过

## 当前产物基线
- acceptance dataset: `learnEngine/datasets/acceptance_20260331_153415/`
- rows/cols: `302 / 364`
- split: `train=206 / val=96`
- selector result: `selected_features_sector_heat.json`
- trained model: `model/sector_heat/acceptance_v1/`
- runtime model: `strategies/sector_heat/runtime_model/sector_heat_V1.pkl`

## 剩余建议补测
1. **C2/C3 因子层运行级验证**
   - 目标：检查 `is_global` 标记与单日因子计算 smoke
2. **B4 split_spec 独立读写测试**
   - 目标：给基础模块补一条独立回归日志
3. **J1/J2 回归兼容测试**
   - 固定日期推理稳定性
   - 旧模型 pickle 兼容性

## 已知非阻塞项
- `_model_signal_helper.py` runtime 输出可能出现重复股票，当前暂不修复。
