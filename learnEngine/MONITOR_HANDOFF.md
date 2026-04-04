# 训练集生成监控交接文档

> 最后更新：2026-04-04 15:45（进程 PID 360452 运行中，续跑起点 2025-07-24）
> 用于 Claude session 中断后快速恢复监控任务

---

## 给新 Claude 的启动指令

> 读取本文件后，立即执行一次监控检查，确认进程存活和进度，之后每2小时抽检一次。不要修改服务器任何文件，不要中断进程。

---

## 当前任务状态

| 字段 | 值 |
|------|-----|
| 总任务 | 343 个交易日（2024-11-01 → 2026-04-01） |
| 已完成（CSV） | **177 天**（截至 2025-07-23） |
| 当前进程续跑起点 | **2025-07-24**（剩余166天） |
| 进程状态 | **运行中**（PID 360452，15:33 启动） |
| 当前处理 | 2025-07-24（分钟线加载中，受当日限流影响） |
| 暂停原因 | 无。今日 Tushare 50K 配额已耗尽，0点重置后将满速恢复 |
| 数据目录 | `learnEngine/datasets/dataset_20260403_220532_272761_p340338_3de4` |
| 日志文件 | `/home/quant/dataset_gen.log` |

---

## 服务器连接方式

```bash
sshpass -p "test1234!" ssh -o StrictHostKeyChecking=no quant@116.62.138.226 "远程命令"
```

- IP：116.62.138.226  用户：quant  密码：test1234!
- 项目路径：`/opt/A_QUANT_test/A_QUANT_PRO`

---

## 一键监控检查命令

```bash
sshpass -p "test1234!" ssh -o StrictHostKeyChecking=no quant@116.62.138.226 "ps aux | grep dataset.py | grep -v grep; grep '处理日期:' /home/quant/dataset_gen.log | wc -l; grep '处理日期:' /home/quant/dataset_gen.log | tail -1; grep -c 'ERROR' /home/quant/dataset_gen.log; free -h; tail -5 /home/quant/dataset_gen.log"
```

正常状态：有 `dataset.py` 进程、已处理天数持续增加、ERROR仅为分钟线限流（可接受）。

---

## 重启命令（必须用 --resume-dir，否则从头重跑）

```bash
sshpass -p "test1234!" ssh -o StrictHostKeyChecking=no quant@116.62.138.226 "cd /opt/A_QUANT_test/A_QUANT_PRO && nohup python3.8 learnEngine/dataset.py --start 2024-11-01 --end 2026-04-01 --resume-dir learnEngine/datasets/dataset_20260403_220532_272761_p340338_3de4 >> /home/quant/dataset_gen.log 2>&1 & echo PID:$!"
```

重启后确认进程启动（等 5 秒）：
```bash
sshpass -p "test1234!" ssh -o StrictHostKeyChecking=no quant@116.62.138.226 "sleep 5 && ps aux | grep dataset.py | grep -v grep && tail -3 /home/quant/dataset_gen.log"
```

续跑成功的标志（日志中出现）：
```
INFO - dataset.py:601 - 待处理日期（共 N 个）: ['2025-07-xx', ...]
INFO - dataset.py:612 - 断点续跑 | 固定列数: 426
```

---

## 数据质量抽检（heredoc stdin 方式，避免OOM）

```bash
sshpass -p "test1234!" ssh -o StrictHostKeyChecking=no quant@116.62.138.226 'bash -s' << 'SSHEOF'
python3.8 - << 'PYEOF'
import pandas as pd
df = pd.read_csv('/opt/A_QUANT_test/A_QUANT_PRO/learnEngine/datasets/dataset_20260403_220532_272761_p340338_3de4/train_dataset.csv',
    usecols=['trade_date', 'strategy_id', 'label1'], dtype={'label1':'int8'})
print('总行数:', len(df), '| 日期数:', df['trade_date'].nunique(), '| 最新:', df['trade_date'].max())
print(df.groupby('trade_date').size().tail(5).to_string())
print('label1正样本率:')
print(df.groupby('strategy_id')['label1'].mean().round(3).to_string())
PYEOF
SSHEOF
```

抽检重点：日期连续无跳空、4个策略均有数据、label1正样本率10-25%、NaN占比<5%。

> ⚠️ 注意：不要用 `pd.read_csv(CSV)` 加载全部列，会 OOM（426列，约85K行，内存不足）。必须用 usecols 限定列。

---

## 进程停止时的恢复流程

### 第一步：查原因
```bash
sshpass -p "test1234!" ssh -o StrictHostKeyChecking=no quant@116.62.138.226 "tail -50 /home/quant/dataset_gen.log"
```

### 第二步：判断是否需要等 0 点
- 日志末尾含 `paused_rate_limit` → **等 0 点后重启**（Tushare 每日配额耗尽）
- 其他异常退出 → 查清原因后用 --resume-dir 续跑

### 第三步：用 --resume-dir 续跑（必须带此参数）
见上方重启命令。

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

## 最新抽检结果（2026-04-04 15:02）

| 指标 | 值 | 状态 |
|------|-----|------|
| CSV 总行数 | 85,819 | ✅ |
| 已写入日期数 | 177 天（截至2025-07-23） | ✅ |
| 最新日期 | 2025-07-23 | ✅ |
| high_low_switch 正样本率 | 12.1% | ✅ |
| oversold_reversal 正样本率 | 12.7% | ✅ |
| sector_heat 正样本率 | 12.3% | ✅ |
| trend_follow 正样本率 | 17.8% | ✅ |

---

## 数据质量异常处理规则（强制执行，无需询问用户）

> ⚠️ 训练集数据严谨性优先级高于一切，发现异常必须停止，不允许带病运行。

### 触发条件
连续 **2个交易日** 出现以下任一情况：
- 任意策略正样本率 < 10% 或 > 25%
- `stock_cpr_d0` 某策略 0值率 > 50%（非sector_heat策略）
- `stock_vol_ratio_d0` 某策略 0值率 > 60%（含停牌正常，但批量0不正常）
- `pos_20d` 出现 < 0 或 > 1 的值
- `market_vol_ratio_d0` 出现 > 10 或 < 0.05 的极端值
- CSV 行数异常（某日行数 < 100 或 > 3000）

单日异常：**记录警告，继续监控**。
连续两日异常：**立即执行以下流程，无需询问用户**。

### 异常处理流程

**Step 1：立即 kill 服务器进程**
```bash
sshpass -p "test1234!" ssh -o StrictHostKeyChecking=no quant@116.62.138.226 "ps aux | grep dataset.py | grep -v grep | awk '{print \$2}' | xargs kill -9 2>/dev/null; echo '进程已终止'"
```

**Step 2：微信推送紧急通知**
内容：`【⛔ 训练集紧急暂停】连续2日数据异常，进程已停止，正在排查。异常详情：[具体指标]`

**Step 3：本地排查**
- 读取 `learnEngine/DATASET_TEST_NOTES.md` 了解历史 Bug 背景
- 读取 `features/FACTOR_REVIEW.md` 对照因子口径
- 如果本 Claude 实例无法独立定位根因（超过3次假设验证失败），**立即切换到 opus 模式** (`/model opus`) 联合排查
- 在本地修改代码，本地验证通过

**Step 4：同步到服务器**
```bash
git add <修改文件> && git commit -m "fix: 训练集数据异常修复" && git push origin master
sshpass -p "test1234!" ssh -o StrictHostKeyChecking=no quant@116.62.138.226 "cd /opt/A_QUANT_test/A_QUANT_PRO && git pull origin master"
```

**Step 5：断点续跑**
```bash
sshpass -p "test1234!" ssh -o StrictHostKeyChecking=no quant@116.62.138.226 "cd /opt/A_QUANT_test/A_QUANT_PRO && nohup python3.8 learnEngine/dataset.py --start 2024-11-01 --end 2026-04-01 --resume-dir learnEngine/datasets/dataset_20260403_220532_272761_p340338_3de4 >> /home/quant/dataset_gen.log 2>&1 & echo PID:$!"
```

**Step 6：连续5分钟高频监控**
续跑后每隔 1 分钟抽检一次，连续 5 次均正常后，才恢复常规 1 小时抽检。
期间仍检查：进程存活 + 最新日志无新 ERROR + 数据行数正常增长。

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
