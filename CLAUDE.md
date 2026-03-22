# a_quant 项目记忆文档

> 最后更新：2026-03-21
> 分支：`claude/fix-limit-up-agent-strategy-v2CHN`

---

## 项目概述

量化交易系统，含短线 Agent（T+1 强制平仓）和中长线 Agent（前向扫描卖出）。
主要模块：数据层（Tushare 拉取 → DB）、特征层（MA/MACD/宏观）、Agent 层（选股 + 卖出信号）、统计引擎层（回测/实盘）。

---

## 核心数据原则（MUST FOLLOW）

> **数据准确性排第一**。本项目所有数据最终都是因子，用于模型训练和回测，数据偏差直接影响模型质量。

1. **优先获取真实数据**：能拿到真实数据，就不用近似/估算/降级。宁愿等待重试（API 有限流时 sleep 后重试），也要拿准确数据。
2. **中性值兜底**：确实无法获取（所有重试耗尽、数据源不可用）时，用**中性值**填充，而非偏向性的 True/False。
   - 中性值定义：不对模型产生方向性偏差的值。对 bool 特征通常为 `False` 或 `None`（排除该样本），对数值特征通常为 `0` 或该特征的历史均值。
   - **错误示范**：涨停基因查询失败返回全 `True` → 所有样本都被标记为"有涨停基因"→ 训练数据正向偏差。
   - **正确示范**：失败时返回 `False`（无信号），或排除该样本（`None`），避免引入虚假正样本。
3. **Tushare 无配额限制**：Tushare 接口不存在"配额耗尽"问题，遇到限流时 sleep 后重试即可。**不应以"节省配额"为由降级数据获取链路**。分钟线等关键数据必须走完整的 DB→API→入库 链路（`get_kline_min_by_stock_date`）。
4. **异常处理最小化影响**：捕获异常时记录日志，用中性值填充，继续运行。**不允许**用异常处理掩盖数据缺失、跳过关键计算步骤。
5. **非自动更新数据必须走 DB→API→DB 链路**：不会每日自动更新的数据表，在使用前必须先查库判断数据是否存在，不存在则通过 API 补拉入库后再使用。**不允许**直接查库后假定数据已存在。
   - **自动更新表**（`data/autoUpdating.py` 每个工作日 17:00 定时更新）：`stock_basic`、`stock_st`、`kline_day`、`index_daily`（4个核心指数）—— 可直接查库使用。
   - **按需更新表**（不会自动更新，使用时必须 ensure）：`kline_day_qfq`、`kline_day_hfq`、`stock_dividend`、`limit_list_ths`、`kline_min` —— 必须通过对应的 `ensure_*` 函数确保数据新鲜：
     - `limit_list_ths` → `ensure_limit_list_ths_data(trade_date)`
     - `stock_dividend` → `ensure_dividend_data(ts_code_list)`
     - `kline_day_hfq` / `kline_day_qfq` → `data_cleaner.clean_and_insert_kline_day_hfq()` / `clean_and_insert_kline_day_qfq()`
     - `kline_min` → `data_cleaner.get_kline_min_by_stock_date()`
   - **错误示范**：直接 `SELECT * FROM limit_list_ths WHERE trade_date=X` 后假定有数据 → 表中无当日数据时返回空集 → 判断全部股票"无涨停"→ 训练数据偏差。
   - **正确示范**：先调 `ensure_limit_list_ths_data(trade_date)` 补拉 → 再查库。

---

## 核心架构

### 数据表体系

| 表名 | 内容 | 复权 | 用途 |
|------|------|------|------|
| `kline_day` | 日线原始数据 | 不复权 | 粗筛候选池、日内收益计算 |
| `kline_day_qfq` | 前复权日线 | 前复权（QFQ） | 短线涨幅计算（有未来函数，不用于回测） |
| `kline_day_hfq` | 后复权日线 | 后复权（HFQ） | 长线止损/MA/区间统计（无未来函数，回测核心） |
| `kline_min` | 分钟线 | 不复权 | 短线日内特征 |
| `stock_dividend` | 分红送股数据 | — | 除权除息日判断（替代 JOIN 方案） |
| `limit_list_ths` | 涨跌停池 | — | 市场恐慌判断、连板信号 |
| `index_daily` | 指数日线 | — | 市场情绪/恐慌判断 |
| `agent_daily_profit_stats` | 买入信号记录 | — | 短线/长线买入信号双写 |
| `agent_long_position_stats` | 长线持仓记录 | — | status=0 持仓中 / status=1 已平仓 |

**重要**：`trade_date` 在所有表中均为 `DATE` 类型，MySQL 自动兼容 `YYYYMMDD` 和 `YYYY-MM-DD` 两种格式查询。`data_cleaner._clean_kline_day_data` 写入前统一转为 `YYYY-MM-DD` 格式字符串。

### 关键数据概念

- **QFQ（前复权）**：以最新日为基准回算历史，每次除权后历史价格全变 → **有未来函数，不能用于回测**
- **HFQ（后复权）**：从上市日起向后调整，历史价格永不改变 → **无未来函数，回测/止损基准**
- **未来函数**：用了 QFQ 的历史数据做回测，相当于用了"未来信息"，导致回测失真

---

## 关键文件说明

### 数据层
- **`data/data_fetcher.py`**：Tushare API 封装，`DataFetcher` 类
  - `fetch_kline_day_qfq(adj="hfq")` — 同时支持 HFQ
  - `fetch_dividend()` — 分红送股数据（新增）
  - 每次调用 `time.sleep(1.2)` 限流
- **`data/data_cleaner.py`**：数据清洗+入库
  - `clean_and_insert_kline_day_hfq()` — HFQ 入库
  - `clean_and_insert_kline_day_qfq()` — QFQ 入库（HFQ 复用此方法）
  - `clean_and_insert_dividend()` — 分红数据入库（新增）
  - **关键修复**：`has_data` 检查改为首尾日期覆盖验证（原来 `LIMIT 1` 会因部分入库而跳过整段）

### 工具层
- **`utils/common_tools.py`**：DB 查询封装
  - `get_hfq_kline_range(ts_code_list, start, end)` — HFQ 区间查询
  - `get_kline_day_range(ts_code_list, start, end)` — 原始日线区间
  - `get_ex_div_stocks(trade_date)` — 旧方案：JOIN kline_day+kline_day_qfq，**性能差**
  - `get_ex_div_dates_for_stock(ts_code, start, end)` — **新方案**：从 stock_dividend 表查，O(1)
  - `get_market_limit_counts(trade_date)` — 当日涨停/跌停数量（从 limit_list_ths）
  - `get_index_pct_chg(trade_date)` — 当日指数涨跌幅（从 index_daily）
  - `has_recent_limit_up_batch(ts_code_list, start, end)` — 批量判断是否有涨停记录（从 limit_list_ths），错误时返回 False（中性值）
  - `get_dividend_check_batch(ts_code_list, start, end)` — 批量查 stock_dividend 判断区间内是否有除权（用于跳过非除权股的 HFQ 请求）
  - `ensure_limit_list_ths_data(trade_date)` — 确保 limit_list_ths 有当日数据（无则 API 补拉）
  - `analyze_stock_follower_strength(ts_code, trade_date)` — 跟风票综合判断（日线排名 + 分钟线确认，分钟线走完整 DB→API 链路）

### Agent 统计引擎层
- **`agent_stats/long_stats_engine.py`**：长线引擎
  - `run_full_flow(mode="full"|"daily")` — 主入口
  - `_forward_scan_sell_for_position()` — **核心方法**，已重写为批量预取（循环内零 DB 查询）
  - `_ensure_hfq_data(ts_code, start, end)` — 确保 HFQ 入库，现分两段调用
  - `_calc_period_stats()` — 平仓区间统计，依赖 HFQ 数据完整性
  - `_try_aggregate_long_stats()` — 平仓后检查是否可聚合，回写 agent_daily_profit_stats
- **`agent_stats/engine_shared.py`**：短线/长线共用工具
  - `build_trade_date_context()` — 构建上下文（**注意**：包含 `get_daily_kline_data(pre_date)` 全市场查询，长线扫描不应使用）

### Agent 层
- **`agent_stats/agents/long_breakout_buy.py`**：突破买入策略
  - 两阶段选股：Phase 1 用 kline_day 粗筛 → Phase 2 仅对粗筛股拉 HFQ 精确判断
  - 参数：`BREAKOUT_DAYS=120`（120日新高）、`PRE_GAIN_DAYS=30`、`PRE_GAIN_MAX_PCT=40%`、`LIMIT_COUNT_DAYS=30`、`LIMIT_COUNT_MAX=3`
  - 止损：`STOP_LOSS_PCT=-8%`，有恐慌弹性机制
  - MA：HFQ MA30/MA60 破位卖出
  - `check_sell_signal()` 支持从 `context["_hfq_preloaded"]` 直接内存计算 MA（零 DB）
  - Phase 2 新增跟风过滤（`analyze_stock_follower_strength`），包裹异常处理

### 特征层
- **`features/ma_indicator.py`**：MA 计算
  - `compute_ma_from_hfq_range(preloaded_kline=None)` — 支持预加载 DataFrame 零 DB 查询，无预加载时降级查库
  - `compute_ma_from_qfq_range()` — QFQ 版本
- **`features/macro/market_macro_feature.py`**：市场宏观特征（依赖 `data_bundle.macro_cache`，不能在长线引擎直接用）

---

## 当前已完成的修复和优化

### P0 致命 Bug 修复
1. **HFQ 数据入库跳过 Bug**：`clean_and_insert_kline_day_qfq` 的 `has_data` 检查从 `LIMIT 1`（只要有1行就跳过）改为首尾日期覆盖验证。根因：Phase 2 买入信号已插入 lookback~buy_date 的 HFQ，导致后续持仓期 HFQ 永远跳过入库。
2. **`_ensure_hfq_data` 分段调用**：原来一次调用 `context_start~today`，被 Phase 2 数据误判为已有而跳过。现改为两段：`context_start~buy_date`（MA回看窗口）+ `buy_date~today`（持仓期）。
3. **直接后果修复**：`trading_days=1, period_return=0.00%` 问题（HFQ 只有1行），止损/MA永不触发问题。

### P1 性能优化
4. **前向扫描批量预取**：`_forward_scan_sell_for_position` 重写，扫描前一次性批量拉取 kline_day + kline_day_hfq + 除权日，循环内零 DB 查询（原：300次/股 → 现：5次/股）
5. **长线 context 轻量化**：扫描循环内 context 不含 `pre_close_data`（全市场）和 `st_stock_list`，只保含卖出信号必要字段
6. **stock_dividend 表**：替代 `get_ex_div_stocks` 的两表 JOIN，建表 SQL 在 `sql/create_stock_dividend.sql`
7. **MA 内存计算**：HFQ MA 改为从预取数据内存计算，不再逐日查库

### P2 策略升级
8. **120日突破**：`BREAKOUT_DAYS: 20→120`，避免短期假突破
9. **30日涨幅过滤**：近30日涨幅>40% 的连板/暴涨股排除
10. **市场恐慌弹性止损**：
    - 条件：跌停>20 且上证跌>2%，或跌停>涨停（且>10只）
    - 进入观察期：止损线 -8%→-16%，观察5天
    - 恢复：收盘>恐慌日收盘 → 恢复正常规则
    - 强制退出：浮亏>-20% 或5天到期

### P3 数据聚合
11. **长线平均盈亏聚合**：当某日买入信号的所有持仓全部平仓后，自动回写 `agent_daily_profit_stats`
    - 新增字段：`long_median_return`（中位收益）、`long_max_return`（最高收益）、`long_min_return`（最低收益）、`long_avg_trading_days`（平均持仓天数）、`long_closed_count`（已平仓数）
    - ALTER TABLE SQL：`sql/alter_agent_daily_profit_stats_long_agg.sql`
    - 触发点：`_forward_scan_sell_for_position` 和 `_check_unsettled_sell_signals` 平仓后调用 `_try_aggregate_long_stats`
    - 幂等：每次平仓后检查，全部平仓才写入；重复写入覆盖无副作用
12. **修改9完成：MA 预加载参数化**：`compute_ma_from_hfq_range` 新增 `preloaded_kline: Optional[pd.DataFrame]` 参数
    - 有预加载数据直接内存过滤计算，零 DB 查询
    - 无预加载数据降级走 `get_hfq_kline_range` 查库（兼容旧调用方式）

### P4 跟风过滤 + 涨停回看优化（2026-03-20）
13. **跟风判断**：`analyze_stock_follower_strength()` 新增到 `common_tools.py`
    - 突破买入 Phase 2 新增跟风过滤：日线排名后 70% + 分钟线后排确认 + 近 5 日历史阳线复核
    - 分钟线走完整 DB→API→入库 链路（`data_cleaner.get_kline_min_by_stock_date`），保证数据准确
    - 分钟线真的获取失败（所有重试耗尽）时用中性值：不标记为跟风（`is_follower=False`），不排除股票
    - `long_breakout_buy.py` 中调用包裹 try/except，捕获 `TushareRateLimitAbort` 和通用异常，异常时中性保留（不标记跟风）
14. **涨停回看批量化**：`has_recent_limit_up_batch()` 新增到 `common_tools.py`
    - 一次 DB 查询 `limit_list_ths` 表判断批量股票是否有涨停记录
    - 替代旧的「逐日拉 kline_day + 逐行算涨停价」实现
    - `sector_heat_strategy.py` 和 `learnEngine/dataset.py` 已切换调用
    - 长线 agent 继续使用 `get_stock_limit_counts_batch`（不同用途：统计涨停+跌停总次数）
15. **HFQ 覆盖率判定优化**：`_get_kline_range_coverage` 新增高覆盖率豁免
    - 新增 `_KLINE_HIGH_COVERAGE_RATIO = 0.95`
    - 当覆盖率 > 95% 时，即使首尾日未覆盖也视为完整（停牌导致的边缘缺口，补拉无意义）
    - 解决 98.3%/95.5% 覆盖率仍触发无效 API 补拉的问题
16. **候选池准入门槛提升**：
    - `long_breakout_buy`：`MIN_AVG_AMOUNT_YI = 1.5`（近 20 日平均成交额 ≥ 1.5 亿）
    - `long_ma_trend_tracking`：`MIN_AVG_AMOUNT_YI = 1.2`

### P5 数据不足修复 + 分红跳过 + 涨跌停补拉 + 回踩简化（2026-03-21）
17. **HFQ 覆盖率尾部边缘修复**：`_get_kline_range_coverage` 高覆盖率豁免增加 `end_edge_ok` 检查
    - 原 Bug：覆盖率 > 95% 时即使 `max_d < end_date`（今日数据缺失）也标记为完整 → Phase2 找不到 today_row → "数据不足"
    - 修复：高覆盖率豁免仅当尾部日期已覆盖时生效（首部缺口仍可豁免，因停牌导致）
18. **分红感知 HFQ 跳过**：`get_dividend_check_batch()` 新增到 `common_tools.py`
    - 批量查 `stock_dividend` 表判断区间内是否有除权事件
    - 无除权 → 原始日线与 HFQ 的相对关系（MA/突破/涨幅/量比）完全一致，直接用 kline_day
    - 有除权 / 未查过分红 → 走 HFQ 链路（保守）
    - `long_breakout_buy` 和 `long_ma_trend_tracking` 两个 agent Phase2 已接入
    - 兜底：HFQ 获取失败的股票降级用原始日线
19. **涨跌停数据新鲜度保障**：`ensure_limit_list_ths_data()` 新增到 `common_tools.py`
    - 查询 `limit_list_ths` 前先检查当日数据是否存在，无则 API 补拉
    - `long_breakout_buy` 和 `long_ma_trend_tracking` 已在涨跌停过滤前调用
20. **回踩条件简化**（`long_ma_trend_tracking`）：
    - 删除 `_is_first_pullback_rebound_signal`（趋势形成→首次回踩→首根阳线 序列匹配）
    - 简化为：今日最低价触及 MA10 * 1.02 以内 + 收盘仍在 MA10 * 0.97 以上
    - 过滤率从 ~30/143 大幅降低
21. **`has_recent_limit_up_batch` 错误兜底改中性值**：`conservative_on_error` 默认改 `False`
    - 原默认 `True`（异常返回全涨停）→ 训练数据正向偏差
    - 改为 `False`（异常返回全无涨停）→ 中性值，不引入虚假正样本

### 历史修复（本会话前）
- **日期格式 Bug**：DB 存储 `trade_date` 为 `YYYY-MM-DD`，代码比较用 `YYYYMMDD`，修复：`.astype(str).str.replace("-", "")`
- **HFQ 预取超时**：移除引擎级别全市场 HFQ 预取（4900只×1.2s），改为两阶段按需拉取
- **QFQ 涨幅计算**：`_position_stock_helpers.py` 和 `mid_position_stock.py` 改用 QFQ 计算涨幅，fallback 到原始数据

---

## 跑起来前的检查清单

1. **建表**：执行 `sql/create_stock_dividend.sql`（stock_dividend 表）
2. **加字段**：执行 `sql/alter_agent_daily_profit_stats_long_agg.sql`（长线聚合字段）
3. **重置 agent**：因参数从 20 日改为 120 日，历史信号需重跑
   ```bash
   python agent_stats/run.py --reset-agent long_breakout_buy --from-date 2024-07-01
   ```
4. **dividend 数据入库**：首次运行时 `clean_and_insert_dividend` 按需触发，会有 API 延迟
5. **验证 HFQ 数据**：确认 kline_day_hfq 表有数据（可能首次运行需要较长时间入库）

---

## 数据流全链路（长线 Agent）

```
每日运行（full 模式）
  ↓
get_daily_kline_data(trade_date)  ← 全市场原始日线
  ↓
LongBreakoutBuyAgent.get_signal_stock_pool()
  Phase 1: kline_day 粗筛（~4900→~50只）
  Phase 2: HFQ 精确判断（仅对粗筛股）
    → _ensure_hfq_data(rough_list, lookback~trade_date)
    → get_hfq_kline_range(rough_list, ...)
    → 120日新高 + 量能 + 30日涨幅过滤
  ↓
_process_buy_signal() → 写入 agent_daily_profit_stats + agent_long_position_stats(status=0)
  ↓
_forward_scan_sell_for_position(ts_code, buy_date, today)
  批量预取：
    - _ensure_hfq_data(context_start~buy_date) + _ensure_hfq_data(buy_date~today)
    - get_kline_day_range([ts_code], buy_date, today)
    - get_hfq_kline_range([ts_code], context_start, today)
    - get_ex_div_dates_for_stock(ts_code, buy_date, today)
  循环扫描（零DB）：
    check_sell_signal(position, today_row, context)
      → 超期 / 恐慌弹性止损 / MA破位
  触发卖出 → _calc_period_stats() → close_position(status=1)
              ↓
  _try_aggregate_long_stats(buy_date)
    → 检查该日所有持仓是否全部平仓
    → 全部平仓 → 计算中位/最高/最低收益 + 平均持仓天数
    → 回写 agent_daily_profit_stats（long_median_return 等）
```

---

## 常用命令

```bash
# 全量补全历史
python agent_stats/run.py --mode full --start-date 2024-07-01

# 每日更新
python agent_stats/run.py --mode daily

# 重置某个 agent
python agent_stats/run.py --reset-agent long_breakout_buy --from-date 2024-07-01
```
