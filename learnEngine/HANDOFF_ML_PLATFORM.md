# ML 中台接手摘要

> 下一次 session 只需要先读这份，再按需看 README / ACCEPTANCE_TODO。

## 当前结论
- ML 中台核心重构已完成。
- 主链路已跑通：`dataset_acceptance.py -> factor_selector.py -> train.py`。
- 样本语义固定为 `trade_date=D`；`feature_trade_date` 已删除。
- dataset / split / selected_features 已绑定同一 `dataset_run` 目录。
- 训练只写 `model/<strategy>/<version>/`，不会自动改写 runtime。
- `SectorHeatStrategy` 已迁移到 `strategies/sector_heat/sector_heat_strategy.py`。
- runtime 模型统一放在 `strategies/sector_heat/runtime_model/sector_heat_V*.pkl`。
- runtime 目录若存在多个版本模型，加载阶段直接报错。

## 重要工程约定
项目通常在 **D+1 凌晨** 执行策略，以等待 Tushare/入库链路把 D 日完整盘面写齐。
因此候选池阶段显式注入 `get_daily_kline_data(D)` 是有意设计，表达“最近一个已完整落库的收盘日盘面”，不要按字面误改。

## demo 路径
- dataset: `learnEngine/datasets/acceptance_20260331_153415/`
- archive model: `model/sector_heat/acceptance_v1/`
- strategy code: `strategies/sector_heat/sector_heat_strategy.py`
- runtime model: `strategies/sector_heat/runtime_model/sector_heat_V1.pkl`

## 已验证通过
- acceptance dataset：`302 / 364`
- split：`train=206 / val=96`
- selector Stage12：通过
- selector Optuna10：通过
- train.py：通过
- H3 推理 smoke：通过（`2026-03-27`）

## 未来两阶段开盘架构
当前只完成 schema reservation：
- 预留 future label / feature 列名
- 已纳入 `EXCLUDE_COLS`
- dataset 可占位这些列
- 当前主链路仍保持单阶段训练，不启用两阶段训练

## 剩余建议补测
1. C2/C3：因子层运行级验证
2. B4：split_spec 独立读写测试日志补记
3. J1/J2：回归兼容测试

## 已知非阻塞项
- `_model_signal_helper.py` runtime 输出可能出现重复股票；用户已明确要求暂不修复。
