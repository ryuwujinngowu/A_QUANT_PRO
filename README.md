# A-Quant · A 股量化交易系统

多层次量化交易框架，包含 **短线 Agent**（T+1 强制平仓）和 **长线 Agent**（前向扫描卖出）、机器学习模型（XGBoost）、板块轮动、特征工程等核心模块。

> 最后更新：2026-03-28
> 核心特性：数据精度优先、HFQ 复权、全市场特征缓存、全局因子广播、恐慌弹性止损、跟风过滤

---

## 项目架构

```
A_QUANT_PRO/
├── config/                         # 配置层
│   └── config.py                   # 配置加载 + 全局开关
│
├── data/                           # 数据层（Tushare API + 数据清洗）
│   ├── data_fetcher.py             # Tushare 接口封装（日线/分钟线/涨跌停/指数等）
│   ├── data_cleaner.py             # 数据清洗 + 入库（clean_and_insert_* 系列）
│   └── autoUpdating.py             # 定时脚本（每日 17:00 自动更新核心表）
│
├── features/                       # 特征工程层
│   ├── __init__.py                 # FeatureEngine 入口 + 因子注册
│   ├── base_feature.py             # 因子抽象基类
│   ├── feature_registry.py         # 因子注册中心（装饰器模式）
│   ├── data_bundle.py              # 数据容器（单次 IO 预加载 + hp_ext_cache）
│   ├── ma_indicator.py             # MA 计算（HFQ/QFQ + 预加载零 DB）
│   ├── macd_indicator.py           # MACD 计算
│   ├── market_stats.py             # 市场统计
│   │
│   ├── emotion/                    # 情绪因子
│   │   ├── sei_feature.py          # SEI/HDI 情绪指数
│   │   ├── hp_stage_feature.py     # 高位股阶段因子（vwap_bias / pct_chg / amount_ratio）
│   │   └── hp_style_feature.py     # 市场高宽风格（breadth_ratio / height_pct）
│   │
│   ├── macro/                      # 宏观因子
│   │   ├── market_macro_feature.py # 涨跌停 + 连板 + 最强板块 + 指数
│   │   ├── hp_cycle_feature.py     # 高位涨幅周期（120日切片，height_pct / peak_dist_pct）
│   │   └── liquid_stats_feature.py # 液态股广度（60日新高占比 / 高开低走占比）
│   │
│   ├── sector/                     # 板块因子
│   │   ├── sector_heat_feature.py  # 板块热度 + 轮动分
│   │   └── sector_stock_feature.py # 个股板块特征
│   │
│   ├── technical/                  # 技术因子
│   │   ├── ma_position_feature.py  # 均线位置 + 乖离率
│   │   └── stk_factor_feature.py   # MACD/KDJ/RSI/布林带等技术指标（可选启用）
│   │
│   └── utils/
│       └── high_position_utils.py  # 高位股统一计算口径（纯内存，无 IO）
│
├── learnEngine/                    # 机器学习层
│   ├── dataset.py                  # 训练集生成（逐日原子性 + 断点续跑）
│   ├── label.py                    # 标签生成（label1 日内 / label2 隔夜）
│   ├── model.py                    # XGBoost 模型（训练/推理/保存）
│   └── train_config.py             # 训练配置参数
│
├── agent_stats/                    # Agent 统计引擎层（核心买卖逻辑）
│   ├── run.py                      # Agent 运行主入口（--mode full/daily）
│   ├── long_stats_engine.py        # 长线引擎（前向扫描 + 批量预取 + 聚合统计）
│   ├── stats_engine.py             # 短线引擎
│   ├── engine_shared.py            # 短线/长线共用工具
│   ├── long_position_db_operator.py
│   ├── agent_db_operator.py
│   ├── long_agent_base.py
│   ├── config.py                   # Agent 参数配置
│   ├── wechat_reporter.py          # 微信报告推送
│   └── agents/
│       ├── long_breakout_buy.py    # 突破买入（120日新高 + 量能 + 跟风过滤）
│       ├── long_ma_trend_tracking.py  # MA 趋势追踪（MA10 回踩）
│       ├── long_high_low_switch.py    # 高低切换
│       ├── long_sector_rank3to5_open_buy.py  # 板块排名买入
│       ├── model_open_buy.py       # 开盘模型买入（短线）
│       ├── limit_down_buy.py       # 跌停买入
│       ├── hot_sector_dip_buy.py   # 热点板块跌买
│       ├── high_position_stock.py  # 高位股筛选
│       ├── mid_position_stock.py   # 中位股筛选
│       └── sector_top_high_open.py # 板块头部高开
│
├── strategies/                     # 策略层（回测驱动）
│   ├── base_strategy.py
│   ├── sector_heat_strategy.py     # 板块热度 ML 策略（核心回测策略）
│   ├── model_dip_strategy.py
│   ├── model_open_strategy.py
│   ├── model_surge_strategy.py
│   ├── long_high_low_switch_strategy.py
│   └── multi_limit_up_strategy.py
│
├── backtest/                       # 回测引擎
│   ├── engine.py                   # 逐 K 线模拟交易
│   ├── account.py                  # 模拟账户（持仓/资金/订单）
│   └── metrics.py                  # 绩效指标（夏普/最大回撤/胜率）
│
├── position_tracker/               # 持仓追踪
│   ├── tracker.py
│   ├── models.py
│   └── rules.py
│
├── utils/                          # 工具层
│   ├── common_tools.py             # 通用函数（DB 查询 / ensure_* / 高位股辅助）
│   ├── db_utils.py                 # 数据库封装（连接池 / query / batch_insert）
│   ├── log_utils.py                # 日志管理
│   ├── wechat_push.py              # 微信推送
│   └── xgb_compat.py              # XGBoost 版本兼容
│
├── model/                          # 模型存储
│   └── sector_heat_xgb_latest.pkl  # 最新生产模型
│
├── sql/                            # SQL 脚本
│   ├── create_stock_dividend.sql
│   ├── create_agent_long_position_stats.sql
│   ├── alter_agent_daily_profit_stats_long_agg.sql
│   └── add_kline_day_covering_index.sql  # kline_day 覆盖索引（性能优化）
│
├── charts/                         # 图表 + 回测报告
│   ├── chart_core.py
│   ├── agent_chart.py
│   ├── backtest_chart.py
│   └── reports/                    # 回测 HTML 报告（自动生成）
│
├── a_quant_dashboard/              # 监控仪表板（Flask）
│   └── app.py
│
├── main.py                         # 回测主入口
├── train.py                        # 模型训练入口
├── CLAUDE.md                       # 项目记忆文档（数据原则 + 架构说明 + 修复列表）
├── ROADMAP.md                      # 开发路线图
└── FACTOR_IMPROVEMENT_PLAN.md      # 因子优化方向
```

---

## 核心数据库表

### 日线数据

| 表名 | 复权方式 | 用途 |
|------|--------|------|
| `kline_day` | 不复权 | 粗筛候选池、日内收益、全市场特征计算 |
| `kline_day_qfq` | 前复权（QFQ） | 短线涨幅（含未来函数，不用于回测） |
| `kline_day_hfq` | 后复权（HFQ） | 长线回测、止损、MA（无未来函数 ⭐） |
| `kline_min` | 不复权 | 1min 分钟线，SEI/情绪/跟风判断 |

### 行情 & 基础

| 表名 | 说明 |
|------|------|
| `limit_list_ths` | 涨跌停池（limit_type: 1=涨停/2=跌停/3=炸板） |
| `index_daily` | 指数日线（上证/深证/创业板） |
| `stock_basic` | 股票基础信息（ts_code/name/list_date） |
| `stock_st` | ST 股标记 |
| `stock_dividend` | 分红送股（除权日判断，替代 JOIN 方案） |
| `trade_calendar` | 交易日历 |

### Agent 统计表

| 表名 | 说明 |
|------|------|
| `agent_daily_profit_stats` | 买入信号记录 + 长线聚合回写（avg/median/max/min 收益） |
| `agent_long_position_stats` | 长线持仓（status=0 持仓中 / status=1 已平仓） |

### 数据单位速查

| 字段 | 单位 |
|------|------|
| `kline_day.amount` | 千元（VWAP = `amount × 10 / vol`） |
| `kline_day.vol` | 手（1 手 = 100 股） |
| `kline_min.amount` | 元 |
| `trade_date` | DATE，MySQL 自动兼容 `YYYYMMDD` ↔ `YYYY-MM-DD` |

---

## 特征层设计

### 因子类型

**个股因子**（含 `stock_code`，逐股计算）：`sei`、`market_macro`、`sector_heat`、`sector_stock`、`ma_position`、`stk_factor_pro`

**全局因子**（无 `stock_code`，通过 `trade_date` LEFT JOIN 广播到所有个股）：`hp_stage`、`hp_style`、`hp_cycle`、`liquid_stats`

### hp_ext_cache 两轮 IO 架构

`FeatureDataBundle` 在 `_load_hp_ext_cache()` 中集中完成全市场数据加载，避免各因子重复拉全市场日线：

```
Round 1（并发，ThreadPoolExecutor）:
  ├─ 全市场 D0 / D-10 / D-21 日线（不复权，约5200只）
  ├─ ST 集合、上市日期映射
  ├─ 12 个切片 SQL（hp_cycle，每切片取前1%均涨幅）
  └─ 液态股广度统计 SQL（60日新高/低占比 + 高开低走占比）

Round 2（顺序）:
  └─ 高位股基础池近5日日线（内存计算识别基础池后，批量拉 ~50只×5天）
```

### 新增因子（2026-03-28，P7）

| 因子名 | 类型 | 输出列 | 说明 |
|--------|------|--------|------|
| `hp_stage` | 全局 | vwap_bias / pct_chg / amount_ratio | 高位股阶段（MA5乖离率/涨跌幅/量能倍数） |
| `hp_style` | 全局 | breadth_ratio / height_pct | 市场高宽风格（高位股广度/高度） |
| `hp_cycle` | 全局 | height_pct / peak_dist_pct | 120日高位股涨幅周期（当前强度/峰值距离） |
| `liquid_stats` | 全局 | breakout_ratio / holf_ratio | 液态股广度（60日新高占比/高开低走占比） |

---

## 快速开始

### 环境初始化

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量（config/.env）
TUSHARE_TOKEN=your_token_here
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=password
DB_NAME=quant_pro
```

### 数据初始化（首次运行）

```bash
# 建表（按顺序执行）
mysql -u root -p quant_pro < sql/create_stock_dividend.sql
mysql -u root -p quant_pro < sql/create_agent_long_position_stats.sql
mysql -u root -p quant_pro < sql/alter_agent_daily_profit_stats_long_agg.sql

# 可选：添加 kline_day 覆盖索引（提升全市场特征 SQL 性能约 7x）
mysql -u root -p quant_pro < sql/add_kline_day_covering_index.sql

# 同步历史数据
python data/autoUpdating.py
```

### 模型训练

```bash
# 生成训练集
python learnEngine/dataset.py

# 训练 XGBoost 模型
python train.py

# 运行回测
python main.py
```

### Agent 统计运行

```bash
# 全量补全历史（首次）
python agent_stats/run.py --mode full --start-date 2024-07-01

# 每日增量更新
python agent_stats/run.py --mode daily

# 重置某个 Agent（参数变更后重跑）
python agent_stats/run.py --reset-agent long_breakout_buy --from-date 2024-07-01
```

---

## 核心开发规范

### 数据准确性原则

> 数据准确性排第一。所有数据最终都是因子，偏差直接影响模型质量。

1. **优先真实数据**：宁愿 sleep 重试 API，也不降级用近似值。Tushare 无配额限制。
2. **中性值兜底**：无法获取时用中性值（不对模型产生方向性偏差），而非 `True`/`False` 偏向值。
3. **按需更新表必须先 ensure**：

| 表 | 使用前调用 |
|------|------|
| `limit_list_ths` | `ensure_limit_list_ths_data(trade_date)` |
| `stock_dividend` | `ensure_dividend_data(ts_code_list)` |
| `kline_day_hfq/qfq` | `data_cleaner.clean_and_insert_kline_day_hfq()` |
| `kline_min` | `data_cleaner.get_kline_min_by_stock_date()` |

### 复权选择

| 场景 | 使用 |
|------|------|
| 长线回测、止损、MA | **HFQ**（无未来函数 ⭐） |
| 短线涨幅计算（非回测） | **QFQ** |
| 粗筛候选池 | `kline_day`（不复权，性能优先） |

### 新增因子

```python
# 1. 新建文件，继承 BaseFeature，加装饰器
@feature_registry.register("my_factor")
class MyFactor(BaseFeature):
    def calculate(self, data_bundle) -> tuple:
        ...  # 返回 (df, {})，禁止在 calculate() 内发起 DB 查询

# 2. features/__init__.py 添加 import（导入即注册）
from features.technical.my_factor import MyFactor

# 3. 更新 learnEngine/dataset.py 的 FACTOR_VERSION（触发重新生成训练集）
```

**全局因子约定**：因子 `calculate()` 内不含 `stock_code`，FeatureEngine 通过 `trade_date` LEFT JOIN 广播。所有 IO 集中在 `FeatureDataBundle` 的 `_load_*_cache()` 方法中完成。

---

## 数据流全景

```
每日运行流程
    ↓
[Phase 1] 数据同步（autoUpdating.py @ 17:00）
  ├─ kline_day（全市场不复权）
  ├─ kline_day_hfq/qfq（按需补拉）
  ├─ limit_list_ths（涨跌停）
  └─ index_daily（指数）

[Phase 2] Agent 运行（agent_stats/run.py）
  ├─ LongBreakoutBuyAgent.get_signal_stock_pool()
  │  ├─ Phase 1: kline_day 粗筛  (~5200 → ~50只)
  │  └─ Phase 2: HFQ 精确判断   (120日新高 + 量能 + 跟风过滤)
  │
  ├─ _process_buy_signal() → INSERT agent_daily_profit_stats + agent_long_position_stats
  │
  └─ _forward_scan_sell_for_position()
     ├─ 批量预取：kline_day + kline_day_hfq + 除权日（扫描前一次性完成）
     ├─ 循环扫描（零 DB）：超期 / 恐慌弹性止损 / MA 破位
     └─ 触发卖出 → _try_aggregate_long_stats() 聚合回写

[Phase 3] 特征计算（FeatureEngine）
  ├─ FeatureDataBundle 构建
  │  ├─ Round 1 并发：全市场 D0/D10/D21 + ST + 上市日期 + 切片 SQL + 液态股 SQL
  │  └─ Round 2：高位股基础池近5日日线
  ├─ 个股因子计算（sei / sector_heat / ma_position / ...）
  └─ 全局因子计算（hp_stage / hp_style / hp_cycle / liquid_stats）→ trade_date JOIN 广播

[Phase 4] 模型推理 & 回测
  ├─ learnEngine/dataset.py → 训练集生成
  ├─ train.py → XGBoost 训练
  └─ main.py → 回测执行
```

---

## 故障排查

| 问题 | 原因 | 解决 |
|------|------|------|
| HFQ 数据不足（trading_days=1） | HFQ 首尾日期覆盖缺口 | `_ensure_hfq_data()` 分两段调用，检查覆盖率 |
| 涨停判断全为 False | `limit_list_ths` 未同步 | `ensure_limit_list_ths_data(trade_date)` |
| 分钟线无数据 | 未走完整 DB→API 链路 | `data_cleaner.get_kline_min_by_stock_date()` |
| hp_ext_cache 为空 | `_load_hp_ext_cache()` 异常 | 检查 `stock_st` 表是否存在，查看日志 |
| 全局因子列全为中性值 | hp_ext_cache 未能加载 | 确认 `kline_day` 有当日数据 |
| 因子计算失败 | FACTOR_VERSION 不一致 | 检查 `dataset.py` 版本，重跑生成训练集 |

---

## 项目记忆文档

**`CLAUDE.md`** — 详细架构说明、核心数据原则、P0~P7 全部修复/新增列表、完整数据流链路、SQL 建表说明。首次接触代码库强烈建议先读。
