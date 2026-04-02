# 训练集生成 — 测试手册 & Bug记录

> 路径：`/opt/A_QUANT_test/A_QUANT_PRO/learnEngine/DATASET_TEST_NOTES.md`
> 最后更新：2026-04-02

---

## 一、运行命令

```bash
# 单日测试（覆盖 DATE_RANGES 配置）
cd /opt/A_QUANT_test/A_QUANT_PRO
nohup python3.8 -m learnEngine.dataset --start 2025-03-10 --end 2025-03-10 > /tmp/dataset_test_0310.log 2>&1 &

# 正式多日运行（DATE_RANGES 已在 dataset.py 中配置好，直接跑）
nohup python3.8 -m learnEngine.dataset > /tmp/dataset_full_run.log 2>&1 &

# 追踪日志
tail -f /tmp/dataset_full_run.log
```

---

## 二、发现并修复的 Bug（按时间顺序）

### Bug 1 — dataset.py Step1 重复打 API（已修复）

**现象**：每次运行都对 limit_list_ths / limit_step / limit_cpt_list / index_daily 调用 API，即使 DB 已有数据。历史回填时每张表 3×30s=90s 超时浪费。

**根因**：Step1 中无条件调用 `data_cleaner.clean_and_insert_*`，没有先检查 DB。

**修复**（`learnEngine/dataset.py`）：
```python
# 每个 insert 前先 SELECT 1 FROM <table> WHERE trade_date=%s LIMIT 1
# 有数据则跳过 API
if not db.query("SELECT 1 FROM limit_list_ths WHERE trade_date=%s LIMIT 1", (date_fmt,)):
    data_cleaner.clean_and_insert_limit_list_ths(...)
```

---

### Bug 2 — OversoldReversal `KeyError: 'pct_chg'`（已修复）

**现象**：OversoldReversalStrategy 在 Step 6（最大单日跌幅过滤）时崩溃。

**根因**：`common_tools.get_kline_day_range` SQL 没有 SELECT `pct_chg` 字段，但策略代码用 `recent10["pct_chg"]` 直接取列（非 `.get()`）。

**修复**（`utils/common_tools.py`）：
```python
# 旧 SQL: SELECT ts_code, trade_date, open, high, low, close, pre_close, volume, amount
# 新 SQL: 增加 pct_chg
SELECT ts_code, trade_date, open, high, low, close, pre_close, pct_chg, volume, amount
```

---

### Bug 3 — `ensure_ths_daily_data` 历史日期每次超时 90s（已修复）

**现象**：训练回填 2023~2025 日期时，每个日期的 4~5 个 lookback date 全部触发 ths_daily API 超时（30s×3次=90s），约 20 个 lookback date/天 = 30min 浪费。

**根因**：ths_daily 最早数据 2025-12-04，但代码对所有日期都尝试 API，没有最早日期下界检查。

**修复**（`utils/common_tools.py`）：新增 `_THS_DAILY_MIN_DATE_CACHE` 进程级缓存，若请求日期 < 表最早数据日期则直接 skip。

---

### Bug 4 — `ensure_ths_hot_data` 稀疏日期每次超时 30s（已修复）

**现象**：ths_hot 有数据但极其稀疏（63个有数据日期），大量日期如 2025-03-04~2025-03-10 在表的时间范围内但 API 对这些日期返回超时（30s），每个训练日 5 个 lookback date × 30s = 2.5min 浪费。509 个训练日 × 2.5min = **~21 小时** 浪费。

**根因**：min_date 检查只能跳过 < 表最早日期的请求。对于在范围内但实际无数据的日期，API 超时，但代码不记录这个失败，下次仍重试。

**修复**（`utils/common_tools.py`）：新增 `_THS_HOT_TRIED_DATES: set` 进程级缓存：
```python
# API 补拉后若 DB 仍无数据，记录该日期
if not db.query(check_sql, (trade_date_dash,)):
    _THS_HOT_TRIED_DATES.add(trade_date_fmt)
# 再次调用时直接 skip
if trade_date_fmt in _THS_HOT_TRIED_DATES:
    return
```
同样逻辑也加到了 `ensure_ths_daily_data`（`_THS_DAILY_TRIED_DATES`）。

---

### Bug 5 — `TrendFollowStrategy.build_training_candidates` 缺少 `sector_name` 列（已修复）

**现象**：运行到 Step 4.5（策略元数据注入）时崩溃：
```
KeyError: "['sector_name'] not in index"
```

**根因**：`trend_follow_strategy.py` 构造 `candidate_df` 时缺少 `sector_name=""` 字段。`sector_heat` / `high_low_switch` / `oversold_reversal` 都正确设置了，但 `trend_follow` 遗漏。

**修复**（`strategies/trend_follow/trend_follow_strategy.py`）：
```python
candidate_df["strategy_id"]   = self.strategy_id
candidate_df["strategy_name"] = self.strategy_name
candidate_df["sector_name"]   = ""   # 新增这行
```

---

## 三、单日测试时序（2025-03-10，修复后预期）

| 阶段 | 耗时（修复后） | 说明 |
|------|--------------|------|
| Step1 宏观数据 | <1s | 有数据则跳过 API |
| Step2 多策略候选池 | ~25s | sector_heat+high_low+trend+oversold |
| Step3 DataBundle加载 | ~3min | 日线60天+分钟线2585对+hp_ext_cache |
| Step4 特征计算 | ~30s | 14个特征模块 |
| ths_hot lookback | 首次:5×30s=2.5min; 再次:0s | tried_dates 缓存生效后无重试 |
| Step5 标签生成 | ~5s | D+1/D+2标签 |
| 合并写盘 | ~2s | CSV追加 |
| **合计** | **~7min（首次）** / **~4min（后续同lookback日期）** | |

---

## 四、DB 覆盖情况

| 表 | 覆盖范围 | 备注 |
|----|---------|------|
| kline_day | 全量 | 每日自动更新 |
| limit_list_ths | 2023+ | ensure_limit_list_ths_data 按需补拉 |
| index_daily | 全量 | Step1 存在则跳过 |
| ths_hot | 2025-01-01起，但极稀疏（63日） | tried_dates缓存跳过空洞日期 |
| ths_daily | 2025-12-04起 | min_date缓存跳过历史 |
| stock_dividend | 按需 | ensure_dividend_data 按股票补拉 |

---

## 五、各策略候选池大小（2025-03-10 测试结果）

| 策略 | 候选池大小 | 说明 |
|------|-----------|------|
| sector_heat | 按板块 (366行) | Top3板块：人工智能/机器人概念/军工 |
| high_low_switch | 264只 | 涨停首/二板66 + 5日涨10-35%241 |
| trend_follow | 84只 | 60日动量→过热过滤→MA5>MA30 |
| oversold_reversal | 25只 | 跌幅榜300→过滤后25 |
| **合计特征行数** | **366（sector_heat为主）** | 多策略union后feature_df |

---

## 六、2025-03-10 单日测试结果（2026-04-02 实测）

**运行结果**：✅ 流程全部跑通，无崩溃
- 耗时：约 3 分钟（分钟线已有 DB 缓存，所以比预期快）
- 输出：485 行 × 426 列
- 标签分布：label1={0:390, 1:95}（正例率 19.6%），label1_3pct={0:309, 1:176}
- 策略分布：sector_heat=570行, high_low=293行, trend_follow=88行, oversold=29行
- 特征列缺失率：正常范围内（<5%），ths_hot/ths_daily相关特征有合理NaN

**已确认正常的警告**：
- `hp_stage_peak_time_rate` / `agent_high_position_intraday_5d_ratio_d0` 单日单值 → 全局因子，同日相同是正常的
- `label_5d_min_dd_30pct` 全0 → 单日样本量少，正常
- Split spec 写出失败 → 单日测试无法切分，正常
- label1 缺失 254 行（已 dropna 过滤） → 这些行被正确丢弃，最终 CSV 中 label1 全部非空

---

## 七、待排查问题（2026-04-02 发现）

### [BUG-6] `stock_up_vol_ratio_d0` / `stock_dn_vol_ratio_d0` / `stock_chase_success_d0` 全为 1.0

**现象**：485 行中这 3 列 nunique=1，全部等于 1.0
**已核实**：
- DataCleaner.get_kline_min_by_stock_date 返回正确分钟数据（000016.SZ raw up_vol_ratio=6.07）
- 手动 winsorize 测试：30 只样本 → 值域 [0.86, 5.88]，正常多样
- 说明 raw 计算正确，但 CSV 中全为 1.0
**待查**：从 individual_feature.calculate() → FeatureEngine 合并 → dataset assembly 的哪个环节出了问题
**注意**：d1~d4 的对应列有正常值域（d1: nunique=4，mean=0.9951）

### [待确认] `open_regime` 相关列 100% NaN（10列）

`feat_pred_open_regime_*` / `label_open_regime_*` 全部为 NaN。
→ 如果这是规划中的未来特征，属正常占位；如果应该有数据，需要排查。

### [已记录，待核实] label1 缺失 254 行的原因

最终 CSV 无影响（dropna 后 485 行全有 label1），但需要确认：
1. 缺失来源：是停牌（D+1 kline 无数据）？还是 stock_close_d0<=0 触发的价格校验过滤？
2. D+1 (2025-03-11) 和 D+2 (2025-03-12) 均为交易日，D+1 有 5358 只股票数据

---

## 八、正式运行前检查清单

- [x] 测试单日 (`--start 2025-03-10 --end 2025-03-10`) 输出 CSV 无崩溃 ✅
- [x] 检查输出 CSV 列数（426列，超预期）✅
- [x] 检查 label1 分布（正例率 19.6%，合理）✅
- [ ] 修复 BUG-6（stock_up/dn_vol_ratio_d0 全为1.0，根因未定位）
- [ ] 确认 open_regime 列是否为预留占位
- [ ] 运行更多抽检日期（建议 2024-11 到 2026-03 区间内各抽2-3天）
- [ ] 单日耗时确认（目标 <10min/日，当前3min含缓存命中）
- [ ] 确认磁盘空间充足（509日 × 预计 2MB/日 ≈ 1GB）

---

## 九、正式运行配置

`learnEngine/dataset.py` 中 `DATE_RANGES`（正式运行时去掉 `--start/--end` 参数使用此配置）：
```python
DATE_RANGES = [
    ("2023-01-01", "2023-09-30"),
    ("2024-11-01", "2026-03-10"),
]
```

正式运行命令：
```bash
cd /opt/A_QUANT_test/A_QUANT_PRO
nohup python3.8 -m learnEngine.dataset > /tmp/dataset_full_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "PID: $!"
```

---

## 十、给下次 Session 的提醒

1. **检查最新 git 提交**：所有修复都在 master 分支，SCP 更新或 git pull 同步
2. **ths_hot 特征 NaN 是正常的**：历史数据 sparse，`_THS_HOT_TRIED_DATES` 缓存生效后不会浪费时间
3. **每天最慢的是 DataBundle 的分钟线加载**（~2.5min），是设计如此，不是 Bug
4. **如果中断重跑**：`dataset.py` 有断点续跑逻辑（`processed_dates.json`），已处理的日期自动跳过
5. **单日测试用 `--start/--end` 参数**，不修改 DATE_RANGES 代码

---

*记录人：Claude Sonnet 4.6（2026-04-02 Session）*
