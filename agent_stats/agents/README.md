# agent_stats/agents — 策略智能体模块说明

## 模块定位

`agent_stats/agents/` 目录下每个文件对应一个**交易策略智能体（Agent）**。
每个 Agent 负责在指定交易日生成"命中股票池"，引擎（`stats_engine.py`）负责统一回测各 Agent 的日内收益与次日表现，并写入数据库。

---

## 每日运行时序

```
交易日 T-1（前一个交易日）
├─ 市场正常交易，日线/分钟线数据由数据层在收盘后写入 DB
│
凌晨 3:00（交易日 T 当天，开盘前）
├─ cron 触发 agent_stats/run.py
├─ 引擎初始化：处理范围到 T-1（last_trade_date = get_prev_trade_date()）
├─ 对 T-1：
│   Step A  各 Agent 生成 T-1 的命中股票池 → INSERT DB
│   Step B  计算 T-1 信号的日内收益（T-1 收盘价 vs 买入价）
│   Step C  计算 T-2 信号的 D+1 表现（T-1 数据已到）→ UPDATE DB
├─ 推送微信汇报（仅最新一个交易日，历史补全静默）
```

**关键约束**：引擎处理范围上界 = `get_prev_trade_date()`（凌晨运行时当日市场未开盘，
当天日线数据尚未入库，绝不写入当日占位记录，避免次日漏算）。

---

## 历史回看设计

`config.START_DATE`（默认 `"2026-01-01"`）表示**开始跟踪的第一个交易日**。

部分 Agent 需要 START_DATE 之前的历史数据：
- `limit_down_buy`：需要 T-6 收盘价（5日涨幅计算基准）和前 5 日跌停池
- 未来的 MA30 策略：需要 30 个交易日历史均价

引擎在初始化时会向前延伸 **90 个自然日（≈60 交易日）** 加载 `context_trade_dates`，
并通过 `context["trade_dates"]` 传入各 Agent，覆盖绝大多数历史回看需求。
历史日线数据由项目数据层预先入库，Agent 直接查询即可。

---

## Agent 开发规范

### 必须实现

```python
from agent_stats.agent_base import BaseAgent

class MyStrategyAgent(BaseAgent):
    agent_id   = "my_strategy"      # DB 唯一标识，小写下划线
    agent_name = "策略中文名"        # 显示名称
    agent_desc = "策略详细描述..."   # 存入 DB reserve_str_2

    def get_signal_stock_pool(
        self,
        trade_date: str,           # 当日交易日 "YYYY-MM-DD"
        daily_data: pd.DataFrame,  # 全市场日线（kline_day SELECT *）
        context: Dict,             # 见下方 context 说明
    ) -> List[Dict]:
        # 返回命中股票列表，每条包含：
        # {"ts_code": "000001.SZ", "stock_name": "平安银行", "buy_price": 12.34}
        ...
```

### context 字段说明

| 字段 | 类型 | 说明 |
|---|---|---|
| `trade_dates` | `List[str]` | 历史交易日列表（含 START_DATE 前 90 天），YYYY-MM-DD，升序 |
| `st_stock_list` | `List[str]` | 当日 ST / \*ST 股票 ts_code 列表 |
| `pre_close_data` | `DataFrame` | T-1 日全市场日线（含 ts_code, close 等字段） |

### 编码规范

- 排除北交所：`ts_code.split(".")[0].startswith(("83","87","88")) or ts_code.endswith(".BJ")`
- 排除 ST：`ts_code in set(context["st_stock_list"])`
- 涨停价计算：`calc_limit_up_price(ts_code, pre_close)`（在 `utils/common_tools.py`）
- 跌停价计算：`calc_limit_down_price(ts_code, pre_close)`
- 分钟线获取：`data_cleaner.get_kline_min_by_stock_date(ts_code, trade_date)`（自动入库）
- 每只股票只命中一次（用 `dict` 去重）
- **无未来函数**：只允许用 `high`/`low`/`open` 判断日内行为，不允许用 `close` 判断当日收盘

---

## 当前 Agent 列表

### 1. morning_limit_up — 早盘打板选手

**文件**：`morning_limit_up.py`

**逻辑**：
1. 候选池：`high >= 涨停价 × 0.999`，非 ST，非北交所
2. 并发拉取候选股分钟线
3. 一字板（`open >= 涨停价 × 0.999`）：必须曾开板（`min_low < 阈值`）且有回封；回封时间 < 11:30
4. 非一字板：分钟线首次触板时间 < 11:30
5. 买入价 = 涨停价

**适用场景**：早盘情绪强、资金主动打板的标的

---

### 2. afternoon_limit_up — 午盘打板选手

**文件**：`afternoon_limit_up.py`

**逻辑**：同早盘打板，触板/回封时间 ≥ 13:00 归为午盘命中。

**适用场景**：尾盘资金发动、下午涨停的标的；与早盘对比揭示市场情绪节奏

---

### 3. limit_down_buy — 跌停板战法选手

**文件**：`limit_down_buy.py`

**逻辑**：
1. 候选池：`low <= 跌停价 × 1.001`，非 ST，非北交所，`low > 0`
2. **首板过滤**：查 T-5 至 T-1 的 `limit_list_ths` 跌停池，剔除曾出现过跌停的股票
3. **5日涨幅过滤**：`(T-1 收盘) / (T-6 收盘) - 1 > 30%`（只追近期强势股的情绪性回调）
4. 买入价 = 跌停价

**历史数据依赖**：需要 T-6 日收盘价（`get_kline_day_range`）和前 5 日跌停池数据

**适用场景**：强势股遭遇非理性跌停、次日情绪回暖反弹

---

### 4. sector_top_high_open — 昨日最强板块高开选手

**文件**：`sector_top_high_open.py`

**逻辑**：
1. 从 T-1 日 `limit_cpt_list` 取 `rank = 1`（最强）的板块名称
2. 查询该板块所有股票（`get_stocks_in_sector`）
3. 筛选 T 日高开（`open > pre_close`）的非 ST、非北交所股票
4. 买入价 = 当日开盘价（集合竞价后 9:25 可知）

**适用场景**：验证连板主线板块的次日惯性效应

---

## D+1 次日表现字段说明

所有字段均以各股票的**买入价（buy_price）**为基准，使用 **T+1 日**的日线和分钟线计算：

| DB 字段 | 含义 | 公式 |
|---|---|---|
| `next_day_avg_open_premium` | 次日开盘溢价（%） | mean( (T+1开盘 - 买入价) / 买入价 × 100 ) |
| `next_day_avg_close_return` | 次日收盘收益（%） | mean( (T+1收盘 - 买入价) / 买入价 × 100 ) |
| `next_day_avg_max_premium` | 次日最大浮盈（%） | mean( (T+1最高 - 买入价) / 买入价 × 100 ) |
| `next_day_avg_max_drawdown` | 次日最大回撤（%） | mean( (T+1最低 - 买入价) / 买入价 × 100 )；**负值**=曾跌破买入，正值=全天未跌破 |
| `next_day_avg_intraday_profit` | 次日均价收益（%） | mean( (T+1 VWAP - 买入价) / 买入价 × 100 )，VWAP = amount×10/volume（千元/手→元/股） |
| `next_day_avg_red_minute` | 飘红分钟数 | mean( T+1分钟线中收盘≥T日收盘价的分钟数 ) |
| `next_day_avg_profit_minute` | 盈利分钟数 | mean( T+1分钟线中收盘≥买入价的分钟数 ) |

---

## 新增 Agent 流程

1. 在 `agent_stats/agents/` 新建 `.py` 文件
2. 定义类继承 `BaseAgent`，设置 `agent_id` / `agent_name` / `agent_desc`
3. 实现 `get_signal_stock_pool` 方法，返回 `List[Dict]`
4. 引擎**自动发现**，无需修改任何配置文件

---

## 重要文件索引

| 文件 | 职责 |
|---|---|
| `agent_stats/agent_base.py` | Agent 基类，定义接口 |
| `agent_stats/stats_engine.py` | 引擎：历史补全 / 并发处理 / D+1 结账 |
| `agent_stats/agent_db_operator.py` | DB CRUD 封装 |
| `agent_stats/run.py` | CLI 入口（`--start-date`, `--reset-agent`） |
| `agent_stats/wechat_reporter.py` | 微信推送（仅最新日期，历史补全静默） |
| `agent_stats/config.py` | 配置（START_DATE, STATS_TABLE_NAME 等） |
| `utils/common_tools.py` | 公共工具（涨跌停价计算、日线查询等） |
