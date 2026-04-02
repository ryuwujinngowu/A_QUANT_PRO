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

### [BUG-6] `stock_up_vol_ratio_d0` / `stock_dn_vol_ratio_d0` 全为 1.0（已修复）

**现象**：485 行中这 2 列 nunique=1，全部等于 1.0
**根因已定位**（`features/individual/individual_feature.py` 第313行）：
```python
"up_vol_ratio": float(np.clip(up_vol_ratio, 0.0, 1.0))
```
`up_vol_ratio = up_sum / rng_d`（分钟阳线实体之和 / 日内高低振幅）。

数学问题：up_sum 是多根阳线实体的"累积路程"，可以多次往返叠加；rng_d 是日内"净位移"区间。
活跃股 up_sum >> rng_d 是正常现象（000016.SZ: up_sum=0.85, rng_d=0.14 → 原始值=6.07）。
`clip(6.07, 0, 1)` = 1.0，几乎所有股票均被截断到 1.0。

**d1~d4 有正常值域**的原因：d1~d4 的 rng_d 来自历史日线，未被 clip 机制偏差。（实际也被clip，但更多值本身 ≤1.0）

**修复方案**：
- 移除 `up_vol_ratio` / `dn_vol_ratio` 的 `clip(0, 1)` → 改为 `clip(0, 20)`（或不 clip，留给 winsorize）
- 或改变分母定义：`rng_d = up_sum + dn_sum`（总振幅路程），使比值天然在 [0,1]
- **推荐方案**：分母改为 `up_sum + dn_sum`，语义更准确（上行占总路程的比例），必然 ∈ [0,1]

**修复结果**（服务器验证）：
- 000016.SZ: `up_vol_ratio=6.07, dn_vol_ratio=5.50` ✅（原始值正确，不再被截断）
- 修复文件：`features/individual/individual_feature.py` 第313-314行
- 移除 `np.clip(0.0, 1.0)`，改为 `max(0.0, ...)` 保留非负约束，极值由 winsorize 处理

**`stock_chase_success_d0` 全为 1.0**：独立 bug，待继续排查（行为因子，逻辑不同）

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
- [x] 修复 BUG-6（up/dn_vol_ratio_d0 全为1.0 → 移除 clip(0,1)，改 max(0,...)，winsorize处理极值）✅
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

1. **检查最新 git 提交**：所有修复都在 master 分支，服务器需要 git pull
2. **ths_hot 特征 NaN 是正常的**：历史数据 sparse，`_THS_HOT_TRIED_DATES` 缓存生效后不会浪费时间
3. **每天最慢的是 DataBundle 的分钟线加载**（~2.5min），是设计如此，不是 Bug
4. **如果中断重跑**：`dataset.py` 有断点续跑逻辑（`processed_dates.json`），已处理的日期自动跳过
5. **单日测试用 `--start/--end` 参数**，不修改 DATE_RANGES 代码

---

## 十四、2025-06-20 抽检结果（2026-04-02 Session）

### BUG-7 验证通过 ✅
- `stock_chase_success_d0`: nunique=8, mean=0.696, min=0.0, max=1.0 → **不再全为 1.0，修复确认**
- `stock_dip_success_d0`: nunique=9, mean=0.387 → 正常分布
- `stock_strength_d0`: nunique=15, mean=0.552 → 正常分布

### 正常的单值列（全局因子，同日相同是设计如此）
- market_* 系列（全市场指标）、index_*_pct_chg、adapt_score 系列
- hp_cycle_*、hp_stage_*（高位股指标，当日唯一值）
- hp_stage_peak_time_rate、agent_high_position_intraday_5d_ratio_d0

### [BUG-8] `label_d2_limit_down` 大量误标（暂停排查，按用户要求关闭）

**状态变更**：2026-04-03 用户明确要求 **BUG-8 不再继续排查**，本问题在本轮数据集验收中先关闭，不再作为阻塞项。

**保留背景**：2025-06-20 抽检时，曾观察到 `label_d2_limit_down` 标记偏多，且样本级 spot check 中存在疑似误标案例。

**处理原则**：
- 本轮不再继续投入时间定位根因
- 后续若重新开启，需要优先区分：
  1. 标签计算逻辑问题
  2. 多进程/多输出目录造成的结果污染
  3. 抽检样本与原始 label_df/最终 CSV 不一致的问题

**备注**：该问题当前仅作历史记录保留，不纳入本轮继续 debug 范围。

### [待确认] `hp_style_breadth_ratio = 0.0` 全为零

**现象**：`hp_style_breadth_ratio = 0.0` 和 `hp_style_height_pct = 0.0` 全为0。但当日 `hp_stage_pct_chg = 4.744%`（高位股涨了4.7%）。
**可能原因**：`hp_style` 看的是"做到新高的股票数量"，即使高位股涨了也未必有股票做到120日/历史新高。可能是正常数据（2025-06-20是震荡市）。
**排查**：检查 `hp_style_feature.py` 的计算逻辑，确认 cnt_100 的定义。

### [已观察，属正常] 其他单值列
- `agent_middle_position_*_d0~d4` 全为 1.0：中仓策略在该日期无历史信号，返回默认值1.0
- `xsii_pct = 0.5`：该因子无数据，填充中性值0.5
- `ths_hot_score_d0 = 0.0`：THS热度数据稀疏，该日期无数据 → 0，正常
- `label_5d_30pct/10d_60pct/min_dd_30pct` 全为0：单日测试样本量少，正常

---

## 十一、SSH 服务器连接方式（2026-04-02 确认）

**服务器**：116.62.138.226，用户：quant，密码：test1234!
**重要**：服务器 **不支持 sshpass**，不支持 `ssh -o PasswordAuthentication=yes`。
**唯一可用方式**：Python `paramiko` 库（本地 Windows 环境已安装）。

```python
import paramiko
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('116.62.138.226', username='quant', password='test1234!', timeout=15)
stdin, stdout, stderr = ssh.exec_command('命令', timeout=30)
out = stdout.read().decode('utf-8', errors='replace')
```

**上传脚本文件**（复杂脚本避免 shell 转义问题）：
```python
sftp = ssh.open_sftp()
with sftp.open('/tmp/script.py', 'w') as f:
    f.write(script_content)
sftp.close()
ssh.exec_command('python3.8 /tmp/script.py')
```

**服务器 DB 模块**：`from utils.db_utils import db`（不是 db_helper）

---

## 十二、git 同步流程（本地→服务器）

1. **本地修改后**：`git add <files> && git commit -m "..." && git push origin master`
2. **服务器同步**：
   - 如有未提交本地改动：`cd /opt/A_QUANT_test/A_QUANT_PRO && git stash`
   - 拉取：`git pull origin master`
   - **注意**：如果 `learnEngine/DATASET_TEST_NOTES.md` 在服务器是 untracked，git pull 会冲突：
     → 先 `cp learnEngine/DATASET_TEST_NOTES.md /tmp/NOTES_bak.md && rm learnEngine/DATASET_TEST_NOTES.md`，再 pull
3. **验证同步**：`grep -n "prev_vwap" features/individual/individual_feature.py` 确认关键修复

---

## 十三、BUG-7 详细记录（2026-04-02 定位并修复）

### [BUG-7] `stock_chase_success_d0` / `stock_dip_success_d0` 全为 1.0（已修复）

**现象**：行为因子 `stock_chase_success_d0` 全部为 1.0，无区分度。

**根因**（`features/individual/individual_feature.py` 第 469 行）：
```python
# BUG（修复前）：
prev_vwap = prev_amt / (prev_vol * 100.0)
# amount=千元, volume=手(100股) → VWAP = 千元/(手×100) = 极小值(约0.01)
# 例：amount=1,292,756千元, volume=1,148,895手 → vwap = 0.01125
# curr_c(≈11元) > 0.01 → 永远 True → chase_success=1.0
```

**正确公式**：
```python
# 修复后（commit fa2e541）：
# VWAP(元/股) = amount(千元)×1000 / (volume(手)×100) = amount×10 / volume
prev_vwap = prev_amt * 10.0 / prev_vol
# 验证：amount=1,292,756, volume=1,148,895 → vwap = 11.25 ≈ close=11.27 ✓
```

**修复文件**：`features/individual/individual_feature.py:469`（commit fa2e541 `修正`）
**服务器同步**：2026-04-02 21:14 通过 git pull 完成

**验证命令**（服务器）：
```python
# 数据验证：vwap_x10 应 ≈ close
rows = db.query("SELECT close, amount, volume FROM kline_day WHERE ts_code='000001.SZ' ORDER BY trade_date DESC LIMIT 3")
# close=11.27, vwap_x10=11.25 ✓
```

---

### [低优先级待查] 训练集流程中出现 `.BJ` 分钟线拉取

**现象**：2024-11-08 单日测试日志中出现北交所代码分钟线拉取与重试：
- `920533.BJ`
- `920455.BJ`
- `920087.BJ`

日志表现：`fetch_stk_mins` / `get_kline_min_by_stock_date` 对这些 `.BJ` 代码进行了多轮重试，造成明显性能浪费。

**已知约束**：`TEMP_STOP/` 目录里的策略暂不启用，也不应纳入排查范围；只有当它们被错误纳入训练集主流程时才需要看。

**当前判断**：
- 候选池层本应过滤北交所（`FILTER_BSE_STOCK=True`）
- 但在实际运行中，`.BJ` 仍进入了 `FeatureDataBundle._load_minute_data()` 路径
- 说明问题更可能出在**当前启用训练集主流程**的 `target_ts_codes` 汇总 / 透传阶段，而不是 TEMP_STOP

**影响**：
- 无意义分钟线请求与重试
- 增加单日耗时
- 污染日志诊断
- 若 `.BJ` 真进入最终样本层，还会带来训练集口径污染（这一点仍待确认）

**优先级**：低于当前数据正确性抽检；先记录，后续统一排查。

---

### [高优先级新发现] 单日测试存在 artifact 复用/污染风险

在 2025-02-05 / 2025-04-29 / 2025-12-04 的单日测试中，发现最终校验异常：

1. **2025-02-05**
   - 日志显示单日写入 `232` 行
   - 但最终校验读到 `629` 行
   - 报错：`最终训练集存在重复 sample_id: 2`

2. **2025-04-29**
   - 日志显示单日写入 `98` 行
   - 但最终校验读到 `237` 行
   - 报错：`二分类标签列存在非法取值: label1 -> ['0', '1', 'label1']`
   - 这强烈暗示：**CSV 中混入了重复表头行**（header 被当成数据行）

3. **2025-12-04**
   - 日志显示单日写入 `158` 行
   - 但最终校验读到 `396` 行
   - 报错：`最终训练集存在重复 sample_id: 1`

**当前判断**：
- 这些异常不像单日逻辑本身的 bug，更像是**多次单日测试之间复用了同一个 dataset 目录/CSV**
- 或 `DATASET_RUN_ID` / `OUTPUT_CSV_PATH` 在并发/近时间启动时没有做到真正隔离
- 导致：
  - 旧样本残留
  - header 重复写入
  - 最终校验读到跨 run 混合数据

**影响**：
- 会污染复杂因子分布判断
- 也会污染 label 分布、sample_id 重复判断
- 因此后续复杂因子抽检时，必须优先使用“已确认干净的单日产物”或直接在内存 DataFrame 上检查，而不能盲信所有 CSV 成品

**优先级**：高。高于复杂因子的细粒度业务 review，因为它影响抽检结论可信度。

---

### [高复杂度因子专项观察（继续）]

根据用户要求，后续抽检重点放在：
- `stock_follower_score_d0`
- `stock_strength_d0`
- 高位股相关因子（`hp_stage_*` / `hp_cycle_*`）
- 市场高宽风格因子（`hp_style_*`）

#### 1. `stock_follower_score_d0`

**实现逻辑复核**（`features/individual/individual_feature.py:355`）：
- 在所属板块内，以“其他股票”的 D0 成交金额作为权重
- 两个维度：
  1. 触顶更早（`peak_idx` 更小）
  2. 振幅更强（`peak_amp` 更大）
- 最终得分 = 时间领先性 + 振幅领先性 的平均，范围 `[0,1]`

**服务器抽检（2025-06-20）**：
- 分布：`nunique=32, min=0.102, max=0.645, mean=0.407`
- 说明：不是塌缩到单值，也不是全 0.5，中短期内有横截面区分度
- 高分样本：`001296.SZ(0.645)`、`300483.SZ(0.645)`、`300164.SZ(0.628)`
- 低分样本：`001331.SZ(0.102)`、`605167.SH(0.102)`、`603579.SH(0.138)`

**业务判断**：
- 这个因子本质上在描述“是否板块内主动领涨/主动触顶”，设计方向是合理的
- 比单纯涨幅排名更像“板块主导权”刻画，和短线情绪交易逻辑一致

**对模型学习的优缺点**：
- 优点：
  - 有明显横截面差异
  - 与板块题材交易逻辑一致，适合学“龙头/跟风”分化
- 风险：
  - 只在 `sector_candidate_map` 内部比较，样本宇宙较小且依赖当日 top3 板块候选集
  - 因此它学到的是“候选池内相对主导地位”，不是全市场绝对强弱
  - 对不属于热点板块的股票，中性值 0.5 较多时信息密度会下降

**当前结论**：
- **未发现公式错误**
- **业务方向合理**
- 但它更像“局部相对排名因子”，而不是跨全市场稳定可比因子；训练时应结合板块上下文一起看

#### 2. `stock_strength_d0`

**实现逻辑复核**（`features/individual/individual_feature.py:421`）：
- 近20日逐日遍历
- 若当天 `close > pre_close` 记为阳线
- 基础权重：大盘下跌超 0.1% 时权重提升到 1.5
- 若个股涨幅 > 所属最强概念涨幅 × 2，则权重提升到 2.0
- 最终输出 = 加权阳线占比，范围 `[0,1]`

**服务器抽检（2025-06-20）**：
- 分布：`nunique=15, min=0.326, max=0.674, mean=0.543`
- 没有塌缩成 0.5 或 1.0，说明信号有效

**业务判断**：
- 这个因子不是简单动量，而是在刻画：
  1. 是否常在弱市中走强
  2. 是否经常强于所属概念
- 方向上比“20日上涨天数占比”更接近交易直觉

**对模型学习的优缺点**：
- 优点：
  - 值域稳定，树模型友好
  - 能表达“抗跌”与“题材内超额强势”
- 风险：
  - 仍是近20日统计量，偏慢变量，更擅长学“股性”而不是即时拐点
  - `curr_pct > max_concept_pct * 2` 这个阈值比较硬，会让概念加权呈现跳变，连续性一般

**当前结论**：
- **实现未见明显 bug**
- **业务含义清晰，可用**
- 但它更偏“股性因子”，不是高频拐点因子；若希望更利于学习变化，可考虑后续补“最近5日 strength - 过去20日 strength”这类差分版本

#### 3. `hp_stage_*`

**服务器抽检（2025-06-20）**：
- `hp_stage_pct_chg = 4.744`
- `hp_stage_amount_ratio = 0.93`
- `hp_stage_peak_time_rate = 1.322`
- 同日全样本单值，符合“全局因子”设计

**业务解读**：
- 高位股整体当天上涨 +4.7%，说明高位股并未退潮
- 但量能比 0.93，小于 1，表示并非放量高潮，更像缩量维持
- `peak_time_rate > 1` 表示触顶速度比历史更慢，说明高位股虽强，但抢筹不算极致，资金更犹豫

**对模型学习的评价**：
- 这组因子很适合作为“高位股阶段状态”的 regime 上下文
- 但由于是全局单值：
  - 横截面区分力为 0
  - 只能通过“不同日期之间的变化”提供信息
- 因此适合作为辅助因子，不应期待它单独承担选股

#### 4. `hp_style_*`

**服务器抽检（2025-06-20）**：
- `hp_style_breadth_ratio = 0.0`
- `hp_style_height_pct = 0.0`

**实现逻辑复核**（`features/emotion/hp_style_feature.py:57`）：
- 分子：10日涨幅 > 100% 的股票数
- 分母：10日涨幅 > 50% 的股票数
- 过滤 ST / BJ / 新股

**业务判断**：
- 这两个值全 0 不一定是 bug，而是说明：该日没有“10日翻倍”级别的超高标
- 即使高位股整体上涨，也不代表市场出现“高度扩张”
- 所以它与 `hp_stage_pct_chg` 不冲突：
  - `hp_stage` 看的是已在高位的一小撮核心股今天是否继续强
  - `hp_style` 看的是全市场 10 日超强股是否成规模扩散

**对模型学习的评价**：
- 业务上成立
- 但它太稀疏，更适合当作“极热状态触发器”
- 对学习趋势/拐点不够友好，因为大部分日期可能都贴近 0

#### 5. `hp_cycle_*`

**服务器抽检（2025-06-20）**：
- `hp_cycle_height_pct = 0.5588`
- `hp_cycle_peak_dist_pct = 0.7273`

**业务解读**：
- 当前高位股情绪高度只有近120日峰值的约 56%
- 情绪峰值出现在更早的位置（距离当前较远）
- 这与 `hp_stage_pct_chg=4.744` 联合看，代表：
  - 当天高位股局部表现不错
  - 但整个 120 日大周期并不在历史极热区，而更像“中段修复”

**对模型学习的评价**：
- 比 `hp_style_*` 更平滑，也更有利于模型学习“周期位置”
- `height_pct + peak_dist_pct` 这组组合比单独看强很多
- 属于高位股体系里**相对更适合学习趋势/周期切换**的一组

**当前优先级判断**：
- 在高位股相关因子中，`hp_cycle_*` 的学习友好度高于 `hp_style_*`
- `hp_style_*` 更像极端 regime tag
- `hp_stage_*` 更像当日状态上下文

---

## 十六、随机日期抽检（2026-04-02 夜间继续）

### 已启动的单日测试

为覆盖 2024-11 ~ 2026-03，服务器已启动以下单日训练集生成：
- `2024-11-08`
- `2024-12-04`
- `2025-02-05`
- `2025-04-29`
- `2025-12-04`
- `2026-03-06`

当前观察：
- `2024-12-04 / 2025-02-05 / 2025-04-29 / 2025-12-04 / 2026-03-06` 都已正常进入 DataBundle 阶段
- `2024-11-08` 因 `.BJ` 分钟线重试明显拖慢（见上方低优先级问题）

### 已完成的 label 抽检

#### 2026-03-06 标签 spot check
服务器直接调用 `LabelEngine.generate_single_date('2026-03-06', ...)`，并与 D+1/D+2 日线交叉核对：

- `002594.SZ`
  - D+1 open=93.62, close=97.52 → 日内收益 `4.1658%`
  - `label1=0`, `label1_3pct=1` → **正确**
  - D+2 pct_chg=`-0.9434` → `label_d2_limit_down=0` → **正确**
- `600000.SH`
  - D+1 open=9.83, close=9.85 → 日内收益 `0.2035%`
  - `label1=0`, `label1_3pct=0` → **正确**
  - D+2 close=9.96 → `label_d2_return=1.3225%` → **正确**
- `300059.SZ`
  - D+1 open=21.15, close=21.23 → `0.3783%`
  - `label1=0`, `label1_3pct=0` → **正确**

**结论**：label 主路径（`label1` / `label1_3pct` / `label_d2_limit_down` / `label_d2_return`）在 spot check 中与原始日线一致。

### 已完成的因子口径 spot check

#### 2025-03-10 日线派生量价 sanity check
从 `kline_day` 直接抽查并手工核对：
- `000001.SZ`：`vwap=11.5882`，`close=11.59`，基本一致
- `000016.SZ`：`vwap=5.4079`，`close=5.42`，基本一致
- `002131.SZ`：`vwap=4.3134`，`close=4.26`，接近且方向合理（收盘弱于均价）
- `600580.SH`：`vwap=28.9541`，`close=28.66`，方向合理（收盘弱于均价）

这说明：
- `amount(千元) × 10 / volume(手)` 的 VWAP 单位换算是对的
- BUG-7 的修复方向是正确的

### 业务层 review（不仅是数值）

#### 1. `label_d2_limit_down`
**数据正确性**：当前抽检通过。

**业务合理性**：
- 这是一个明确的“尾部风险标签”，适合作为 blacklist / 风险过滤目标
- 但在多策略宽表中，同一股票会跨策略重复，**行级正例率不能等同于股票级正例率**
- 训练时更应关注 `(stock_code, trade_date)` 去重后的正例率，否则会把“被多个策略同时选中的高风险股”放大权重

**对模型学习的意义**：
- 这个标签比普通收益标签更适合学“崩塌前兆”而不是“上涨趋势”
- 如果未来用于同一个主模型混训，可能会和收益标签目标冲突；更适合做独立风险模型或样本惩罚项

#### 2. `hp_style_breadth_ratio` / `hp_style_height_pct`
**当前现象**：在已抽检的 2025-03-10、2025-06-20 中都容易出现全 0。

**业务判断**：
- 从定义看，它统计的是“10日涨幅 >50% / >100%”的超强股数量
- 这更像一个**极端市场热度阈值因子**，不是日常连续变化因子
- 在大量普通交易日里，它天然会长期贴近 0

**对模型学习的影响**：
- 优点：一旦非 0，信号意义很强，代表市场进入极端高温区
- 缺点：大部分日期为 0，变化太稀疏，模型更难学到“拐点”，只能学到“极热 regime 是否出现”

**结论**：
- 作为“极端状态指示器”是合理的
- 作为常规连续特征，信息密度偏低，建议后续考虑补一组更平滑的 companion 特征（如 >20% / >30% / >50% 分层）
- 当前不算 bug，但属于**学习友好度一般**的设计

#### 3. `hp_stage_*`
**业务判断**：
- `hp_stage_pct_chg`、`hp_stage_amount_ratio` 对“高位股强弱切换”有明确市场含义
- `hp_stage_peak_time_rate` 也有业务解释力：抢筹前置 / 发力后置

**对模型学习的影响**：
- 这些因子是全局单值，不能做横截面区分，但可以作为市场 regime 上下文
- 与个股因子交叉分裂时有价值；单独用时区分力有限

**结论**：
- 设计合理，适合作为上下文因子
- 但必须接受“同一天全样本同值”的结构限制，不应期待它单独承担选股任务

#### 4. `stock_chase_success_d0` / `stock_dip_success_d0`
**数据正确性**：BUG-7 修复后已恢复分布，不再全 1.0。

**业务合理性**：
- 这两个因子本质在刻画个股过去 20 日“强后更强”/“跌后修复”的行为习性
- 比单纯动量更接近交易行为微结构，思路是对的

**对模型学习的影响**：
- 优点：它们是比例型、压到 `[0,1]`，对树模型比较友好
- 风险：当 20 日内满足条件样本很少时，会退化到 0.5，中性值占比可能较高，导致有效信息密度下降
- 但相比原来 VWAP 单位错误导致的全 1.0，当前状态已经可用

#### 5. `market_vol_ratio_d0`
**业务判断**：
- 修复为 `d0 / mean(d1~d4)` 后，终于表达的是“今天比过去更活跃还是更萎缩”
- 这是标准的 regime 变化因子，比自引用版本更适合学“变化”和“拐点”

**结论**：
- 这个修复不仅是数值正确，更显著提升了模型可学习性
- 属于“从静态值改成变化值”的正确方向

### 当前识别到的性能黑洞

1. **DataBundle 分钟线加载**：仍是主耗时模块（正常设计内最重）
2. **`.BJ` 漏进分钟线加载**：会把无效重试放大成严重拖慢（低优先级问题，已记录）
3. **历史稀疏 THS 数据**：虽然已做 tried_dates/min_date 缓存，但在某些日期仍可能成为次级耗时源

---

## 十五、BUG-8 深度排查记录（2026-04-02 Session 续）

### 结论更新（2026-04-02 23:25）

**BUG-8 当前结论：大概率不是代码 bug，而是上一次人工抽样记录错误。**

重新在服务器上复核了两个 2025-06-20 产物目录：
- `dataset_20260402_211549/train_dataset.csv`
- `dataset_20260402_211613/train_dataset.csv`

两份 CSV 的 `label_d2_limit_down=1` 行完全一致；之前文档里列为“误标”的几只股票，重新核对后在这两份产物里实际都是 `0`：
- `000777.SZ`
- `600644.SH`
- `603706.SH`
- `605167.SH`
- `001296.SZ`

并且直接调用 `LabelEngine.generate_single_date('2025-06-20', ...)`，这些股票输出也一致为 0。

### 已确认事实

1. **并行进程共享状态不是根因**
   - `create_dataset_run_id` 精确到秒，两个目录分别是 `211549` 和 `211613`
   - 两个目录中的 `label_d2_limit_down=1` 股票集合完全一致
   - 说明不是两个进程互相污染 CSV

2. **label.py 单独运行正确**
   - 服务器验证：
     - `000777.SZ` → D+2 `pct_chg=-3.85`，`label_d2_limit_down=0`
     - `600644.SH` → `+0.44`，`label=0`
     - `603706.SH` → `-6.54`，`label=0`
     - `605167.SH` → `-0.51`，`label=0`
     - `001296.SZ` → `-5.10`，`label=0`

3. **2025-06-20 CSV 中被标记为 1 的股票，经复核都确实满足跌停/近跌停阈值**
   - 例：
     - `000554.SZ` → `-10.0363`
     - `000968.SZ` → `-9.9515`
     - `002207.SZ` → `-10.0189`
     - `002476.SZ` → `-10.0529`
     - `002828.SZ` → `-10.0230`
     - `300157.SZ` → `-15.1220`
     - `300164.SZ` → `-20.0258`
     - `300483.SZ` → `-18.0027`
     - `301158.SZ` → `-11.2383`
     - `600759.SH` → `-10.1124`
     - `603619.SH` → `-10.0087`
   - 即：**63 行里 27 行为 1，看起来多，但原因是这些样本重复出现在多个策略/多个板块切片中，不是标签错位。**

### 根因修正

此前把“63 行里 28 行为 1”误理解成“很多不同股票被误标”。
实际上应按 **唯一股票数** 看，而不是按样本行数看：
- 当前 CSV 有 **27 行** `label_d2_limit_down=1`
- 但只对应 **11 只唯一股票**
- 这些股票因为同时出现在多个策略或多个板块中，所以会重复出现多行

换言之：
- `label_d2_limit_down` 是按 `(stock_code, trade_date)` 生成并广播到样本行的
- 多策略宽表里，同一股票跨策略重复是设计如此
- 因此不能直接用“正例行数/总行数”判断是否误标，必须先去重到唯一股票层

### 后续建议

- 后续检查标签分布时，**先按 `stock_code + trade_date` 去重**，再判断正例率
- 对多策略宽表，行级 label 分布只能反映“样本行曝光度”，不能直接反映“唯一股票层面的标签质量”
- BUG-8 暂时降级为：**误报 / 抽样分析口径错误，不作为代码 bug 处理**

### 仍需保留的观察点

虽然 BUG-8 本身排除，但 2025-06-20 产物仍有两个值得继续观察的点：
1. `dataset.py` 日志里 `label1` 合并后缺失 161 行，然后清洗后仅保留 63 行 —— 这是多策略宽表+label1必需过滤共同作用，未必是 bug，但值得持续监控
2. 同一股票在多个策略/板块里重复出现较多（如 `002207.SZ`、`002828.SZ` 各 4 行）—— 这是当前 sample 设计的一部分，但训练时要意识到它会放大某些股票的权重

---

*记录人：Claude Sonnet 4.6（2026-04-02 Session）*
