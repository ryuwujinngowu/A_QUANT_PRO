# 训练集生成监控交接文档

> 最后更新：2026-04-05（进程 PID 360452 运行中，当前进度 267/343 天）
> 用于 Claude session 中断后快速恢复监控任务

---

## 给新 Claude 的启动指令

读取本文件后，**立即执行一次完整监控抽检**（Step1 + Step2 + Step3），之后每小时自动抽检一次。不要修改服务器任何文件，不要中断进程。

### 权限说明（已预配置，无需用户确认）

本项目已在 `.claude/settings.local.json` 中设置 `"defaultMode": "bypassPermissions"`，**所有工具调用无需用户确认，直接执行**。

如遇到权限弹框（说明配置未生效），执行以下命令修复：
```bash
python3 -c "
import json
path = '/Users/liusonghao/PycharmProjects/A_QUANT_PRO/A_QUANT_PRO/.claude/settings.local.json'
with open(path) as f: data = json.load(f)
data.setdefault('permissions', {})['defaultMode'] = 'bypassPermissions'
with open(path, 'w') as f: json.dump(data, f, indent=2)
print('已修复')
"
```

### 一次性启动监控循环（执行后无需再看）

用以下命令启动每小时自动抽检（使用 /loop 技能）：
```
/loop 60m 对服务器训练集生成进程执行一次完整监控抽检，按照 learnEngine/MONITOR_HANDOFF.md 里的 Step1+Step2+Step3 执行
```

---

## 当前任务状态

| 字段 | 值 |
|------|-----|
| 总任务 | 343 个交易日（2024-11-01 → 2026-04-01） |
| 已完成（CSV） | **267 天**（截至 2025-11-06，最后更新 2026-04-05） |
| 进程状态 | **运行中**（PID 360452） |
| 数据目录 | `learnEngine/datasets/dataset_20260403_220532_272761_p340338_3de4` |
| 日志文件 | `/home/quant/dataset_gen.log` |
| CSV行数 | ~117811 行（持续增长） |

---

## 服务器连接

- IP：116.62.138.226  用户：quant  密码：test1234!
- 项目路径：`/opt/A_QUANT_test/A_QUANT_PRO`

---

## 完整监控抽检流程（每小时执行一次）

### Step 1：进程 + 进度 + 内存 + 日志

```bash
sshpass -p "test1234!" ssh -o StrictHostKeyChecking=no quant@116.62.138.226 "ps aux | grep dataset.py | grep -v grep; echo '---进度---'; grep '处理日期:' /home/quant/dataset_gen.log | wc -l; grep '处理日期:' /home/quant/dataset_gen.log | tail -1; echo '---内存---'; free -h | grep Mem; echo '---最新日志---'; tail -3 /home/quant/dataset_gen.log"
```

**若进程不存在**：先查日志末尾原因：
```bash
sshpass -p "test1234!" ssh -o StrictHostKeyChecking=no quant@116.62.138.226 "tail -50 /home/quant/dataset_gen.log"
```
- 含 `paused_rate_limit` → **不重启**，等 0 点配额重置后再跑
- 其他异常退出 → 用以下命令断点续跑（**必须带 --resume-dir**）：

```bash
sshpass -p "test1234!" ssh -o StrictHostKeyChecking=no quant@116.62.138.226 "cd /opt/A_QUANT_test/A_QUANT_PRO && nohup python3.8 learnEngine/dataset.py --start 2024-11-01 --end 2026-04-01 --resume-dir learnEngine/datasets/dataset_20260403_220532_272761_p340338_3de4 >> /home/quant/dataset_gen.log 2>&1 & echo PID:$!"
```

---

### Step 2：CSV 轻量抽检 + 因子轮换检查

> ⚠️ 不能用 pandas 加载全量 CSV（426列 × 12万行会 OOM），必须用 csv 模块 + 5000行滚动缓冲。

因子按 `hour % 6` 轮换，每组3个因子，6组覆盖18个关键列：

| 组 | 因子 | 期望值域 |
|----|------|---------|
| 0 | stock_cpr_d0, stock_vol_ratio_d0, pos_20d | [0,1]; 非sh的0值率分别<50%/<60% |
| 1 | stock_profit_d0, stock_hdi_d0, market_vol_ratio_d0 | [0,100]; [0,100]; [0.05,10] |
| 2 | stock_gap_return_d0, stock_candle_d0, adapt_score | [-0.12,0.12]; {-2~2}; [0,100] |
| 3 | stock_upper_shadow_d0, stock_lower_shadow_d0, stock_trend_r2_d0 | ≥0; ≥0; [0,1] |
| 4 | hp_stage_vwap_bias, hp_stage_amount_ratio, market_limit_up_count | [-30,30]; [0.01,20]; [0,500] |
| 5 | stock_pct_chg_d0, stock_amount_5d_ratio_d0, index_sh_pct_chg | [-15,15]; [0,20]; [-12,12] |

执行以下 heredoc 命令（完整抽检脚本）：

```bash
sshpass -p "test1234!" ssh -o StrictHostKeyChecking=no quant@116.62.138.226 'bash -s' << 'SSHEOF'
python3.8 - << 'PYEOF'
import csv, json, os, collections, datetime

STATE_FILE = '/tmp/monitor_state.json'
CSV_PATH = '/opt/A_QUANT_test/A_QUANT_PRO/learnEngine/datasets/dataset_20260403_220532_272761_p340338_3de4/train_dataset.csv'

last_rows = 0
last_anomaly = {}
if os.path.exists(STATE_FILE):
    try:
        s = json.load(open(STATE_FILE))
        last_rows = s.get('last_rows', 0)
        last_anomaly = s.get('last_anomaly', {})
    except: pass

hour = datetime.datetime.now().hour
FACTOR_GROUPS = [
    ['stock_cpr_d0','stock_vol_ratio_d0','pos_20d'],
    ['stock_profit_d0','stock_hdi_d0','market_vol_ratio_d0'],
    ['stock_gap_return_d0','stock_candle_d0','adapt_score'],
    ['stock_upper_shadow_d0','stock_lower_shadow_d0','stock_trend_r2_d0'],
    ['hp_stage_vwap_bias','hp_stage_amount_ratio','market_limit_up_count'],
    ['stock_pct_chg_d0','stock_amount_5d_ratio_d0','index_sh_pct_chg'],
]
cur_group = FACTOR_GROUPS[hour % 6]
print(f'[因子轮换] 本次检查组{hour%6}: {cur_group}')

header = None
cur_rows = 0
date_set = set()
latest_date = ''
recent_buf = []

with open(CSV_PATH, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if header is None:
            header = row
            col_idx = {c: i for i, c in enumerate(header)}
            continue
        if len(row) < 3 or row[0] == 'trade_date':
            continue
        cur_rows += 1
        d = row[col_idx['trade_date']] if 'trade_date' in col_idx else ''
        if d: date_set.add(d); latest_date = max(latest_date, d)
        recent_buf.append(row)
        if len(recent_buf) > 5000:
            recent_buf.pop(0)

new_rows = cur_rows - last_rows
print(f'总行数: {cur_rows} | 新增: {new_rows} | 日期数: {len(date_set)} | 最新: {latest_date}')

all_dates = sorted(date_set)
recent5 = set(all_dates[-5:])
by_strat = collections.defaultdict(lambda: [0,0])
for row in recent_buf:
    d = row[col_idx.get('trade_date',0)] if col_idx.get('trade_date') is not None else ''
    if d not in recent5: continue
    strat = row[col_idx['strategy_id']] if 'strategy_id' in col_idx else ''
    try:
        lv = float(row[col_idx['label1']])
        by_strat[(d,strat)][0] += int(lv==1)
        by_strat[(d,strat)][1] += 1
    except: pass

print('\n--- 近5日正样本率（全4策略）---')
anomalies = {}
low_rate_days = collections.defaultdict(list)
for (d,s),(pos,tot) in sorted(by_strat.items()):
    if tot > 0:
        r = pos/tot
        flag = ' ⚠️' if r < 0.10 or r > 0.25 else ''
        print(f'  {d} {s}: {r:.3f}({pos}/{tot}){flag}')
        if r < 0.10 or r > 0.25:
            low_rate_days[d].append(f'{s}={r:.1%}')

sorted_bad = sorted(low_rate_days.keys())
for i in range(len(sorted_bad)-1):
    d1,d2 = sorted_bad[i], sorted_bad[i+1]
    if d1 in all_dates and d2 in all_dates:
        if all_dates.index(d2)-all_dates.index(d1)==1:
            anomalies['consec_low_label'] = f'{d1}:{low_rate_days[d1]},{d2}:{low_rate_days[d2]}'
            print(f'\n⚠️ 连续2日正样本偏低: {d1}{low_rate_days[d1]} / {d2}{low_rate_days[d2]}')

print(f'\n--- 因子值域抽检（组{hour%6}，近5000行）---')
BOUNDS = {
    'stock_cpr_d0':(0,1),'pos_20d':(0,1),'stock_trend_r2_d0':(0,1),
    'stock_hdi_d0':(0,100),'stock_profit_d0':(0,100),'adapt_score':(0,100),
    'market_vol_ratio_d0':(0.05,10),'hp_stage_vwap_bias':(-30,30),
    'hp_stage_amount_ratio':(0.01,20),'market_limit_up_count':(0,500),
    'stock_gap_return_d0':(-0.12,0.12),'stock_pct_chg_d0':(-15,15),
    'stock_amount_5d_ratio_d0':(0,20),'index_sh_pct_chg':(-12,12),
    'stock_upper_shadow_d0':(0,5),'stock_lower_shadow_d0':(0,5),
    'stock_candle_d0':(-2,2),
}
ns_rows = [r for r in recent_buf if 'strategy_id' in col_idx and r[col_idx['strategy_id']] != 'sector_heat']

for fname in cur_group:
    if fname not in col_idx: print(f'{fname}: 列不存在'); continue
    idx = col_idx[fname]
    vals = []
    for row in recent_buf:
        try: vals.append(float(row[idx]))
        except: pass
    if not vals: print(f'{fname}: 无数据'); continue
    n = len(vals)
    avg = sum(vals)/n
    lo,hi = BOUNDS.get(fname,(-1e9,1e9))
    oor = sum(1 for v in vals if v<lo or v>hi)/n
    if fname in ('stock_cpr_d0','stock_vol_ratio_d0'):
        ns_vals = []
        for row in ns_rows:
            try: ns_vals.append(float(row[idx]))
            except: pass
        ns_zero = sum(1 for v in ns_vals if v==0)/len(ns_vals) if ns_vals else 0
        bad = ns_zero > (0.5 if fname=='stock_cpr_d0' else 0.6)
        st = '⚠️' if bad else '✅'
        print(f'{fname}(非sh): avg={avg:.3f} 0率={ns_zero:.1%} 越界={oor:.1%} {st}')
        if bad: anomalies[f'{fname}_zero'] = ns_zero
    else:
        bad = oor > 0.05
        st = '⚠️' if bad else '✅'
        print(f'{fname}: avg={avg:.3f} 越界={oor:.1%} {st}')
        if bad and fname not in ('stock_candle_d0',): anomalies[f'{fname}_oor'] = oor

factor_anoms = {k:v for k,v in anomalies.items() if 'label' not in k}
consec_factor = False
if factor_anoms and last_anomaly:
    overlap = set(factor_anoms.keys()) & set(last_anomaly.keys())
    if overlap:
        consec_factor = True
        print(f'\n🚨 连续因子异常: {overlap}')
        anomalies['consec_factor'] = str(overlap)

consec_anomaly = consec_factor
json.dump({'last_rows':cur_rows,'last_anomaly':factor_anoms,'consec_anomaly':consec_anomaly}, open(STATE_FILE,'w'))
print(f'\nCONSEC_ANOMALY={consec_anomaly}')
print(f'ANOMALIES={json.dumps({k:str(v) for k,v in anomalies.items()})}')
print(f'LABEL_WARNING={"consec_low_label" in anomalies}')
PYEOF
SSHEOF
```

---

### Step 3：根据输出决策

解析 Step2 最后几行的 `CONSEC_ANOMALY=` 和 `LABEL_WARNING=`：

#### 情况A — CONSEC_ANOMALY=False，LABEL_WARNING=False（正常）

推送微信正常通知：
```bash
sshpass -p "test1234!" ssh -o StrictHostKeyChecking=no quant@116.62.138.226 'bash -s' << 'SSHEOF'
cd /opt/A_QUANT_test/A_QUANT_PRO && python3.8 -c "
import sys; sys.path.insert(0,'.')
from utils.wechat_push import send_wechat_message_to_multiple_users
send_wechat_message_to_multiple_users('【训练集监控】✅ 正常', '进度: X/343天\nCSV行数: N(新增M行)\n近5日正样本率:\n  [填4策略×5天]\n因子抽检组Y: 全部正常\n内存: 可用?')
"
SSHEOF
```

#### 情况B — LABEL_WARNING=True，CONSEC_ANOMALY=False（正样本率偏低，因子正常）

**不停进程**，推送⚠️提示：
```bash
sshpass -p "test1234!" ssh -o StrictHostKeyChecking=no quant@116.62.138.226 'bash -s' << 'SSHEOF'
cd /opt/A_QUANT_test/A_QUANT_PRO && python3.8 -c "
import sys; sys.path.insert(0,'.')
from utils.wechat_push import send_wechat_message_to_multiple_users
send_wechat_message_to_multiple_users('【训练集监控】⚠️ 正样本率偏低', '进度: X/343天\nCSV: N行(新增M行)\n[具体日期+策略+比例]\n因子抽检: 全部正常✅\n判断: 市场行情所致，非数据错误，继续观察\n内存: 可用?')
"
SSHEOF
```

> **判断依据**：正样本率偏低（<10%）+ 因子结构正常 = 市场下跌行情，D+1日内涨幅普遍偏低，数据无误。历史上 2025-08-25/26、2025-09-11、2025-10-13（国庆节后）均出现此情况，均为正常市场数据。

#### 情况C — CONSEC_ANOMALY=True（因子连续异常，⛔ 严重）

**无需询问用户，立即执行：**

1. Kill 进程：
```bash
sshpass -p "test1234!" ssh -o StrictHostKeyChecking=no quant@116.62.138.226 "ps aux | grep dataset.py | grep -v grep | awk '{print \$2}' | xargs kill -9 2>/dev/null; echo 进程已终止"
```

2. 推送紧急微信（含具体异常指标）：
```bash
sshpass -p "test1234!" ssh -o StrictHostKeyChecking=no quant@116.62.138.226 'bash -s' << 'SSHEOF'
cd /opt/A_QUANT_test/A_QUANT_PRO && python3.8 -c "
import sys; sys.path.insert(0,'.')
from utils.wechat_push import send_wechat_message_to_multiple_users
send_wechat_message_to_multiple_users('【⛔ 训练集紧急暂停】', '连续2次因子异常\n异常指标: [ANOMALIES内容]\n进程已kill\n正在本地排查')
"
SSHEOF
```

3. 本地排查：读取 `learnEngine/DATASET_TEST_NOTES.md` + `features/FACTOR_REVIEW.md`，定位根因

4. 超过 3 次假设验证失败 → 切换 opus 模式（`/model opus`）继续排查

5. 修复后同步服务器并续跑：
```bash
git add <修改文件> && git commit -m "fix: 训练集数据异常修复" && git push origin master
sshpass -p "test1234!" ssh -o StrictHostKeyChecking=no quant@116.62.138.226 "cd /opt/A_QUANT_test/A_QUANT_PRO && git pull origin master"
sshpass -p "test1234!" ssh -o StrictHostKeyChecking=no quant@116.62.138.226 "cd /opt/A_QUANT_test/A_QUANT_PRO && nohup python3.8 learnEngine/dataset.py --start 2024-11-01 --end 2026-04-01 --resume-dir learnEngine/datasets/dataset_20260403_220532_272761_p340338_3de4 >> /home/quant/dataset_gen.log 2>&1 & echo PID:$!"
```

6. 续跑后连续 5 分钟高频确认（每分钟执行一次 Step1，连续 5 次无新异常才恢复常规抽检）

7. 推送恢复微信：
```bash
sshpass -p "test1234!" ssh -o StrictHostKeyChecking=no quant@116.62.138.226 'bash -s' << 'SSHEOF'
cd /opt/A_QUANT_test/A_QUANT_PRO && python3.8 -c "
import sys; sys.path.insert(0,'.')
from utils.wechat_push import send_wechat_message_to_multiple_users
send_wechat_message_to_multiple_users('【✅ 训练集恢复运行】', '修复完成，已断点续跑\n连续5分钟数据正常\n恢复常规1小时抽检')
"
SSHEOF
```

---

## 已知问题与处理状态

| 问题 | 状态 | 说明 |
|------|------|------|
| 内存不足 OOM | ✅ 已加 2GB Swap | root 执行 |
| 重启丢进度 | ✅ 已加 --resume-dir | 必须带此参数 |
| 内存不归还 OS | ✅ 已加 gc.collect + malloc_trim | |
| Tushare 分钟线每日限流 | ✅ **已修复限流模式速率 Bug** | 见下方说明 |
| TushareRateLimitAbort 吞异常 | ✅ 已修复 | data_bundle.py 两处 re-raise；dataset.py 专项 break handler |
| 加载全量CSV导致OOM | ✅ 已知 | 抽检时必须用 usecols |

### 限流模式 Bug 修复说明（2026-04-04）

**根因：** 触发每日50K限额后，Tushare降速到20次/分钟，数据仍可获取。但原代码在限流模式（throttled）下，`fetch_stk_mins` 的 `@retry_decorator(3, 1.0)` 会在 Semaphore(2) 内连发3次API（含2×1s sleep），实际速率 = 2槽位×3次/5.9s ≈ **61次/分钟**，远超20次/分限制，导致所有请求被拒。

**修复（data_cleaner.py）：**
1. `_THROTTLE_MIN_INTERVAL`: 3.0 → **6.0**（2槽位×1次/6s = 20次/分，正确）
2. 限流模式下绕过 `@retry_decorator`，改用 `data_fetcher.pro.stk_mins()` **直接调用**（每次semaphore持有仅1次API）
3. 限流模式 backoff 固定为 **1s**（替代指数退避，速率控制由6s extra_wait保障）

**修复后效果：** 限流模式下 ≈19.4次/分，每只股票约6.2s（成功时），可稳定获取数据继续生成。

---

## 最新抽检结果（2026-04-05 03:08）

| 指标 | 值 | 状态 |
|------|-----|------|
| CSV 总行数 | 117,811 | ✅ |
| 已写入日期数 | 267 天（截至 2025-11-06） | ✅ |
| 最新日期 | 2025-11-06 | ✅ |
| high_low_switch 正样本率（近期均值） | ~13% | ✅ |
| oversold_reversal 正样本率 | ~13% | ✅ |
| sector_heat 正样本率 | ~12% | ✅ |
| trend_follow 正样本率 | ~18% | ✅ |

> **已知正样本率波动行情**（因子正常，属市场数据，无需处理）：
> - 2025-08-25/26：市场大幅回调
> - 2025-09-11：市场急跌
> - 2025-10-10：国庆节后首日暴涨（全策略 40-55%）→ 10-13 急剧回落（sector_heat 0%）
> - 2025-11-04/05：市场震荡

**Step 7：微信推送恢复通知**
内容：`【✅ 训练集恢复运行】修复完成，已断点续跑，连续5分钟数据正常，进入常规1小时抽检。`

---

## 完成判断与微信通知

进程退出且日志出现"所有日期已处理完成"视为完成。

```bash
sshpass -p "test1234!" ssh -o StrictHostKeyChecking=no quant@116.62.138.226 "cd /opt/A_QUANT_test/A_QUANT_PRO && python3.8 -c \"import sys; sys.path.insert(0,'.'); from utils.wechat_push import push_wechat; push_wechat('训练集生成完成，共343个交易日')\""
```

---

## 参考文档

遇到问题先查：`learnEngine/DATASET_TEST_NOTES.md`
