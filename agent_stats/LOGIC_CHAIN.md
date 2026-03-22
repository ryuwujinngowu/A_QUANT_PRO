# agent_stats 全链路逻辑文档（后复权 HFQ 版）

## 数据复权方式选择

| 复权方式 | 特点 | 问题 |
|---|---|---|
| 前复权 QFQ | 以最新交易日为基准向前调整历史价格 | 每次除权除息后所有历史价格全部重算——**未来函数** |
| 后复权 HFQ | 从上市日往后调整，历史价格永不改变 | 价格绝对值与实际成交价不同（但相对涨跌幅一致） |

**结论**：全链路（买入信号 + 卖出信号 + 区间统计）统一使用后复权（HFQ）价格，
买入价入库仍用原始收盘价（实际成交价），止损/MA 判断用 HFQ 相对涨跌。

---

## 买入信号链路

```
引擎 run_full_flow / daily 模式
│
├─ get_daily_kline_data(trade_date)            # 获取 T 日全市场原始日线
├─ _get_context(trade_date)                     # 构建上下文（ST 列表、交易日、除息股）
├─ _prefetch_hfq_for_candidates(daily_data)     # ★ 预取候选股 HFQ 数据
│   ├─ 筛选候选股（非 ST、非北交所、有收盘价）
│   ├─ 批量 DB 查 kline_day_hfq 已有股票
│   ├─ 仅对缺失股调用 data_cleaner.clean_and_insert_kline_day_hfq()
│   └─ 流程：查库 → Tushare 接口（adj=hfq）→ 入库
│
├─ agent.get_signal_stock_pool(trade_date, daily_data, context)
│   │   # 以 LongBreakoutBuyAgent 为例
│   ├─ 确定回看起始日期（trade_dates 列表）
│   ├─ 筛选候选池（非 ST、非北交所、有收盘价）
│   ├─ get_hfq_kline_range(ts_codes, lookback_start, trade_date)  # 查 DB
│   ├─ 逐股检查突破条件：HFQ 收盘 > 近 N 日 HFQ 最高价
│   ├─ 量能条件：当日成交量 > N 日均量 × 倍数
│   └─ 返回信号列表，buy_price = 原始收盘价（来自 daily_data）
│
├─ _calc_intraday_stats(signal_list, trade_date)  # 日内均收益
├─ signal_db.insert_signal_record(...)            # 写 agent_daily_profit_stats
└─ pos_db.insert_position(...)                    # 写 agent_long_position_stats（status=0）
```

---

## 卖出信号链路

```
引擎 _forward_scan_sell_for_position / _check_unsettled_sell_signals
│
├─ _ensure_hfq_data(ts_code, context_start, today)   # 确保 HFQ 数据已入库
├─ _get_hfq_buy_price(ts_code, buy_date)              # 查 kline_day_hfq 获取买入日 HFQ 收盘
├─ position["hfq_buy_price"] = hfq_buy_price          # 注入持仓记录
│
├─ 逐日扫描（full 模式）或当日检查（daily 模式）
│   └─ agent.check_sell_signal(position, today_row, context)
│       │   # 以 LongBreakoutBuyAgent 为例
│       ├─ 1. 超期检查：trading_days_so_far ≥ MAX_HOLD_DAYS（60 日）
│       ├─ 2. 止损检查（HFQ）：
│       │   └─ (hfq_close_today - hfq_buy_price) / hfq_buy_price ≤ STOP_LOSS_PCT（-8%）
│       └─ 3. MA 破位检查（HFQ）：
│           ├─ compute_ma_from_hfq_range() 计算 HFQ 均线
│           ├─ 除权除息日跳过（context["ex_div_stocks"]）
│           └─ hfq_close_today < MA30 或 MA60 → 卖出
│
└─ 触发卖出 → _calc_period_stats → pos_db.close_position
```

---

## 区间统计链路（_calc_period_stats）

```
平仓时调用
│
├─ get_hfq_kline_range([ts_code], buy_date, sell_date)
│   └─ HFQ 数据为空？→ 降级用 get_kline_day_range（原始价格）并记录警告
│
├─ buy_price  = HFQ 买入日收盘价（range_df.iloc[0]["close"]）
├─ sell_price = HFQ 卖出日收盘价（range_df.iloc[-1]["close"]）
│
├─ 逐日计算：
│   ├─ float_pnl = (close - buy_price) / buy_price × 100
│   ├─ max_high / min_low 跟踪
│   └─ up_days / down_days 统计
│
└─ 输出：
    ├─ period_return        = (sell_price - buy_price) / buy_price × 100
    ├─ max_floating_profit  = (max_high - buy_price) / buy_price × 100
    ├─ max_drawdown         = (min_low - buy_price) / buy_price × 100
    ├─ trading_days / up_days / down_days
    └─ daily_detail[]
```

---

## HFQ 数据获取流程

```
data_cleaner.clean_and_insert_kline_day_hfq(ts_codes, start_date, end_date)
│
├─ 复用 clean_and_insert_kline_day_qfq，参数 adj="hfq"，表名 kline_day_hfq
├─ 逐股检查：SELECT 1 FROM kline_day_hfq WHERE ... LIMIT 1
│   ├─ 有数据 → 跳过
│   └─ 无数据 → fetch_kline_day_qfq(ts_code, adj="hfq") → INSERT kline_day_hfq
└─ 写入成功后，后续 get_hfq_kline_range() 即可查到
```

---

## 关键文件索引

| 文件 | 职责 |
|---|---|
| `long_stats_engine.py` | 长线引擎：买入信号 + 前向扫描卖出 + 区间统计 |
| `agents/long_breakout_buy.py` | 突破买入 agent（HFQ 突破 + HFQ 止损/MA 卖出） |
| `long_agent_base.py` | 长线 agent 基类 |
| `long_position_db_operator.py` | 持仓表 CRUD |
| `engine_shared.py` | 引擎共享工具（交易日查找、日内统计） |
| `features/ma_indicator.py` | MA 计算（含 compute_ma_from_hfq_range） |
| `utils/common_tools.py` | get_hfq_kline_range / get_kline_day_range 等 |
| `data/data_cleaner.py` | clean_and_insert_kline_day_hfq（数据入库） |
