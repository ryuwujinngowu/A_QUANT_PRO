# A-Quant · A 股量化交易系统

多层次量化交易框架，包含 **短线 Agent**（T+1 强制平仓）和 **长线 Agent**（前向扫描卖出）、机器学习模型（XGBoost）、板块轮动等策略。

> 📌 **最后更新**：2026-03-22
> 📌 **核心特性**：数据精度优先、HFQ 复权、分钟线链路完整、恐慌弹性止损、跟风过滤

---

## 项目架构

```
A_QUANT_PRO/
├── config/                       # 配置层
│   ├── config.py                 # 配置加载 + 全局开关
│   └── __init__.py
│
├── data/                         # 数据层（Tushare API + 数据清洗）
│   ├── data_fetcher.py           # Tushare 接口封装（日线/分钟线/涨跌停/指数等）
│   ├── data_cleaner.py           # 数据清洗 + 入库（clean_and_insert_* 系列）
│   └── autoUpdating.py           # 定时脚本（每日 17:00 自动更新核心表）
│
├── data_realtime/                # 实时数据层
│   ├── realtime_KlineDayFetcher.py # 盘中实时行情拉取
│   ├── auction_open_reporter.py   # 竞价开盘数据
│   └── ...
│
├── features/                     # 特征工程层（详见 features/README.md）
│   ├── __init__.py               # FeatureEngine 入口 + 因子注册
│   ├── base_feature.py           # 因子抽象基类
│   ├── feature_registry.py       # 因子注册中心（装饰器模式）
│   ├── data_bundle.py            # 数据容器（单次 IO 预加载）
│   ├── ma_indicator.py           # MA 计算（支持 HFQ/QFQ + 预加载）
│   ├── macd_indicator.py         # MACD 计算
│   ├── market_stats.py           # 市场统计
│   ├── emotion/                  # 情绪引擎
│   │   └── sei_feature.py        # SEI/HDI 情绪指标
│   ├── sector/                   # 板块分析
│   │   ├── sector_heat_feature.py    # 板块热度 + 轮动分
│   │   └── sector_stock_feature.py   # 个股板块特征
│   ├── technical/                # 技术面
│   │   └── ma_position_feature.py    # 均线 + 乖离率 + 价格位置
│   └── macro/                    # 宏观面
│       └── market_macro_feature.py   # 涨跌停 + 连板 + 最强板块 + 指数
│
├── learnEngine/                  # 机器学习层（详见 learnEngine/README.md）
│   ├── dataset.py                # 训练集生成（逐日原子性 + 断点续跑）
│   ├── label.py                  # 标签生成（日内盈利 / 隔夜延续）
│   ├── model.py                  # XGBoost 模型（训练/推理/保存）
│   └── factor_ic.py              # 因子 IC 分析工具（ICIR 评估）
│
├── agent_stats/                  # Agent 统计引擎层（核心买卖逻辑）
│   ├── run.py                    # Agent 运行主入口（--mode full/daily）
│   ├── long_stats_engine.py      # 长线引擎（前向扫描卖出 + 聚合统计）
│   ├── engine_shared.py          # 短线/长线共用工具
│   ├── long_position_db_operator.py # 长线持仓 DB 操作
│   ├── agent_db_operator.py      # 买入信号 DB 操作
│   ├── long_agent_base.py        # 长线 Agent 基类
│   ├── agents/                   # Agent 实现
│   │   ├── long_breakout_buy.py      # 突破买入（120日新高 + 量能 + 30日涨幅过滤）
│   │   ├── long_ma_trend_tracking.py # MA 趋势追踪（MA10 回踩 + 突破）
│   │   ├── limit_down_buy.py         # 跌停买入
│   │   ├── high_position_stock.py    # 高位持仓
│   │   ├── mid_position_stock.py     # 中位持仓
│   │   ├── sector_top_high_open.py   # 板块头部高开
│   │   └── ...
│   ├── config.py                 # Agent 参数配置
│   └── temp_stop/                # 暂停策略目录
│
├── strategies/                   # 策略层（ML + 规则驱动）
│   ├── base_strategy.py          # 策略抽象基类
│   ├── sector_heat_strategy.py   # 板块热度 ML 策略（核心策略）
│   ├── LimitUpPullback_Strategy.py # 涨停回踩策略
│   ├── overSold.py               # 超卖反弹策略
│   └── ...
│
├── backtest/                     # 回测引擎
│   ├── engine.py                 # 逐 K 线模拟交易
│   ├── account.py                # 模拟账户（持仓/资金/订单）
│   ├── metrics.py                # 绩效指标（夏普/最大回撤/胜率）
│   └── utils_forBackTest.py      # 回测工具函数
│
├── runner/                       # 策略运行器
│   └── sector_heat_runner.py     # 板块热度策略运行器
│
├── utils/                        # 工具层
│   ├── common_tools.py           # 通用函数（DB 查询/数据处理/工具函数）
│   ├── db_utils.py               # 数据库封装（query/batch_insert）
│   ├── log_utils.py              # 日志管理
│   └── wechat_push.py            # 微信推送通知
│
├── main.py                       # 总入口（回测/实盘驱动）
├── train.py                      # 模型训练入口
├── CLAUDE.md                     # 项目内存文档（核心数据原则 + 修复列表）
├── requirements.txt              # Python 依赖
└── ...
```

---

## 核心数据库表

### 日线数据

| 表名 | 复权方式 | 用途 | 说明 |
|------|--------|------|------|
| `kline_day` | 不复权 | 粗筛、日内收益 | 原始 OHLCV + amount（千元）+ pre_close |
| `kline_day_qfq` | 前复权（QFQ） | 短线涨幅（含未来函数） | 前向基准回算，历史价格会变动 |
| `kline_day_hfq` | 后复权（HFQ） | 长线回测、止损、MA | 从上市日向后调整，历史价格不变 ⭐️ |

### 分钟线 & 行情

| 表名 | 说明 |
|------|------|
| `kline_min` | 1min 分钟线（不复权），SEI/HDI/跟风判断 |
| `limit_list_ths` | 涨跌停池（limit_type: 1=涨停/2=跌停/3=炸板） |
| `limit_list_step` | 连板天梯（各股连板天数统计） |
| `limit_list_cpt` | 最强板块排行（连板家数/涨停家数） |

### 索引 & 基础

| 表名 | 说明 |
|------|------|
| `index_daily` | 指数日线（上证/深证/创业板） |
| `stock_basic` | 股票基础信息（ts_code/name/list_date/delist_date） |
| `stock_st` | ST 股标记 |
| `stock_dividend` | 分红送股数据（除权判断） |
| `trade_calendar` | 交易日历 |

### Agent 统计表

| 表名 | 说明 |
|------|------|
| `agent_daily_profit_stats` | 买入信号记录（买入日期/数量/平均价/回写聚合结果） |
| `agent_long_position_stats` | 长线持仓记录（status=0 持仓中 / status=1 已平仓） |

### 数据单位速查

| 字段 | 单位 | 备注 |
|------|------|------|
| `kline_day.amount` | 千元 | VWAP 计算：`amount × 10 / vol` |
| `kline_day.vol` | 手 | 1 手 = 100 股 |
| `kline_min.volume` | 股（个） | 分钟线成交数量 |
| `kline_min.amount` | 元 | 分钟线成交金额 |
| `trade_date` | DATE | 格式自动转换 `YYYYMMDD` ↔ `YYYY-MM-DD` |

---

## 快速开始

### 环境初始化

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境（config/.env 或 ~/.bashrc）
export TUSHARE_TOKEN="your_token_here"
export MYSQL_USER="root"
export MYSQL_PASSWORD="password"
export MYSQL_DB="quant_pro"
export MYSQL_HOST="localhost"
```

### 数据初始化

```bash
# 3. 初始化 DB 表（首次运行）
# - 手动执行 SQL 建表脚本（参考 CLAUDE.md 中数据表体系）
# - 或在 data/data_cleaner.py 中找到建表初始化代码

# 4. 同步历史数据（可选，耗时较长）
python data/autoUpdating.py

# 5. 确保核心表已有数据（stock_basic, trade_calendar, kline_day 等）
```

### 模型训练 & 回测

```bash
# 6. 生成训练集
python learnEngine/dataset.py

# 7. 训练 XGBoost 模型
python train.py

# 8. 分析因子有效性
python learnEngine/factor_ic.py

# 9. 运行回测（基于 sector_heat_strategy）
python main.py
```

### Agent 统计运行

```bash
# 10. Agent 全量补全历史（首次）
python agent_stats/run.py --mode full --start-date 2024-07-01

# 11. 每日增量更新
python agent_stats/run.py --mode daily

# 12. 重置某个 Agent（参数变更时）
python agent_stats/run.py --reset-agent long_breakout_buy --from-date 2024-07-01
```

---

## 核心开发规范

### 数据准确性原则（MUST FOLLOW）

> 📌 **数据准确性排第一**。所有数据最终都是因子，用于模型训练和回测，数据偏差直接影响模型质量。

1. **优先获取真实数据**
   - 能拿到真实数据，就不用近似/估算/降级
   - 宁愿等待重试（API 限流时 sleep 后重试），也要拿准确数据
   - Tushare **无配额限制**，遇到限流 sleep 后重试即可

2. **中性值兜底**
   - 无法获取（所有重试耗尽、数据源不可用）时，用**中性值**填充
   - 中性值定义：不对模型产生方向性偏差（bool 用 `False`/`None`，数值用 `0` 或均值）
   - ❌ 错误：涨停查询失败返回全 `True` → 训练数据正向偏差
   - ✅ 正确：失败时返回 `False`（无信号），避免虚假正样本

3. **关键数据必须走完整链路**（DB→API→入库）
   - **自动更新表**（每日 17:00 自动更新）：`stock_basic`、`stock_st`、`kline_day`、`index_daily` → 可直接查库
   - **按需更新表**（不自动更新）：`kline_min`、`kline_day_hfq`、`stock_dividend`、`limit_list_ths` → 必须通过 `ensure_*` 函数确保新鲜：
     ```python
     ensure_limit_list_ths_data(trade_date)        # 涨停数据
     ensure_dividend_data(ts_code_list)            # 分红数据
     data_cleaner.get_kline_min_by_stock_date()    # 分钟线
     ```

4. **异常处理最小化影响**
   - 捕获异常时记录日志，用中性值填充，继续运行
   - ❌ 不允许：用异常处理掩盖数据缺失、跳过关键计算

### 复权选择

| 场景 | 使用 | 原因 |
|------|------|------|
| 长线回测、止损、MA | **HFQ**（后复权） | 无未来函数，历史价格不变 ⭐️ |
| 短线涨幅计算（非回测） | **QFQ**（前复权） | 最新价格基准 |
| 粗筛候选池 | `kline_day`（不复权） | 性能优先，精筛再用 HFQ |

### 新增因子

详见 `features/README.md`。核心 3 步：

1. 在 `features/<子目录>/` 新建文件，继承 `BaseFeature`，加装饰器
   ```python
   from features.base_feature import BaseFeature
   from features.feature_registry import feature_registry

   @feature_registry.register("my_factor_name")
   class MyFactor(BaseFeature):
       def compute(self, data_bundle):
           # 实现计算逻辑
           pass
   ```

2. 在 `features/__init__.py` 添加 import（导入即注册）
   ```python
   from features.technical.my_factor import MyFactor
   ```

3. 更新 `learnEngine/dataset.py` 的 `FACTOR_VERSION`（触发重新生成训练集）

### 修改因子逻辑

1. 修改对应因子文件
2. 更新 `FACTOR_VERSION`（版本号递增）
3. 重跑 `dataset.py` → 旧版本训练集自动失效 → 新版本自动生成

### Agent 开发

见 `agent_stats/README.md`（若无则参考 CLAUDE.md Agent 统计引擎部分）

- 长线 Agent 基于 `LongAgentBase` 继承，需实现 `get_signal_stock_pool()` 和 `check_sell_signal()`
- 所有 Agent 均自动注册到统计引擎，支持 `--reset-agent` 重置
- 关键优化：前向扫描批量预取、HFQ 分段确保、分红感知跳过

### 常见陷阱

| 陷阱 | 表现 | 解决 |
|------|------|------|
| QFQ 用于回测 | 回测虚高，无未来函数验证 | 改用 HFQ |
| 直接查库假定数据存在 | `limit_list_ths` 为空 → 判断全无涨停 | 调 `ensure_limit_list_ths_data()` 先补拉 |
| 异常处理返回偏向值 | 异常返回全 `True` → 训练数据偏差 | 改为中性值 `False` |
| 跳过分钟线完整链路 | 分钟线缺数据 → 情绪指标失效 | 用 `data_cleaner.get_kline_min_by_stock_date()` |

---

## 关键文件导读

### 数据层

| 文件 | 关键功能 | 说明 |
|------|--------|------|
| `data/data_fetcher.py` | `DataFetcher` 类 | Tushare API 封装，每次调用 sleep(1.2) 限流 |
| `data/data_cleaner.py` | `clean_and_insert_*` | 数据清洗入库，HFQ 数据首尾日期覆盖验证 |
| `utils/common_tools.py` | `get_hfq_kline_range()` 等 | DB 查询工具 + `ensure_*` 函数 |

### 特征层

| 文件 | 关键功能 | 说明 |
|------|--------|------|
| `features/__init__.py` | `FeatureEngine` | 因子注册触发 + 全局缓存 |
| `features/data_bundle.py` | `FeatureDataBundle` | 单次 IO 预加载数据容器 |
| `features/ma_indicator.py` | `compute_ma_from_hfq_range()` | MA 计算，支持预加载零 DB 查询 |

### Agent 引擎

| 文件 | 关键功能 | 说明 |
|------|--------|------|
| `agent_stats/run.py` | 主入口 | `--mode full/daily` 控制模式 |
| `agent_stats/long_stats_engine.py` | 长线引擎 | 前向扫描卖出、聚合统计、止损规则 |
| `agent_stats/agents/long_breakout_buy.py` | 突破买入 | 120 日新高 + 量能 + 30 日涨幅过滤 |
| `agent_stats/agents/long_ma_trend_tracking.py` | MA 趋势追踪 | MA10 回踩 + 突破 |

### 模型层

| 文件 | 关键功能 | 说明 |
|------|--------|------|
| `learnEngine/dataset.py` | 训练集生成 | 逐日原子性 + 断点续跑 + FACTOR_VERSION 驱动 |
| `learnEngine/label.py` | 标签生成 | label1 日内盈利 / label2 隔夜延续 |
| `train.py` | 模型训练 | 从 `learnEngine/model.py` 调用 XGBoost |

---

## 数据流全景

```
每日运行流程（full/daily 模式）
    ↓
├─ [Phase 1] 数据自动同步（autoUpdating.py @ 17:00）
│  ├─ get_daily_kline_data(trade_date)           → kline_day
│  ├─ fetch & clean kline_day_hfq                → kline_day_hfq（Agent 依赖）
│  ├─ fetch & clean kline_day_qfq                → kline_day_qfq
│  └─ fetch market_macro（涨跌停/连板/指数）     → limit_list_ths, index_daily
│
├─ [Phase 2] Agent 运行（agent_stats/run.py）
│  ├─ LongBreakoutBuyAgent.get_signal_stock_pool()
│  │  ├─ Phase 1: kline_day 粗筛  (~4900 → ~50)
│  │  └─ Phase 2: HFQ 精确判断    (120日新高 + 量能 + 跟风过滤)
│  │
│  ├─ _process_buy_signal()
│  │  ├─ INSERT agent_daily_profit_stats（买入记录）
│  │  └─ INSERT agent_long_position_stats (status=0 持仓中)
│  │
│  └─ _forward_scan_sell_for_position()         ← 核心卖出逻辑
│     ├─ 批量预取：kline_day + kline_day_hfq + 除权日
│     ├─ 循环扫描（零 DB）：check_sell_signal()
│     │  └─ 超期 / 恐慌弹性止损 / MA 破位
│     └─ 触发卖出 → _calc_period_stats() → UPDATE status=1
│        └─ _try_aggregate_long_stats() 聚合回写
│
├─ [Phase 3] 特征计算 + 模型推理（dataset.py / train.py）
│  ├─ FeatureEngine.compute()                    → 逐日因子计算
│  ├─ Model.predict()                           → XGBoost 推理
│  └─ 决策信号输出
│
└─ [Phase 4] 回测 / 实盘执行
   ├─ backtest/engine.py                        → 历史回测
   └─ runner/sector_heat_runner.py               → 实盘执行
```

---

## 项目记忆文档

📖 **详细说明见 `CLAUDE.md`**，内容包括：

- ✅ **核心数据原则**：6 大数据准确性规则
- ✅ **P0~P5 修复列表**：HFQ 入库 Bug、前向扫描优化、恐慌弹性止损等
- ✅ **全链路数据流**：从 API 拉取到决策输出的完整链路
- ✅ **常用命令**：Agent 重置、模式切换、日期范围控制
- ✅ **跑起来前的检查清单**：初始化步骤、数据补拉、验证清单

**首次运行强烈建议先读一遍 CLAUDE.md！**

---

## 常用命令速查

```bash
# Agent 相关
python agent_stats/run.py --mode full --start-date 2024-07-01   # 全量补全
python agent_stats/run.py --mode daily                            # 每日增量
python agent_stats/run.py --reset-agent long_breakout_buy \
    --from-date 2024-07-01                                        # 重置 Agent

# 数据相关
python data/autoUpdating.py                      # 手动数据同步
python learnEngine/dataset.py                    # 生成训练集
python learnEngine/factor_ic.py                  # 因子有效性分析

# 模型相关
python train.py                                  # 训练 XGBoost
python main.py                                   # 回测驱动
```

---

## 故障排查

| 问题 | 原因 | 解决 |
|------|------|------|
| `agent_daily_profit_stats` 为空 | Agent 未运行或无信号 | 检查 `run.py --mode daily` 日志 |
| HFQ 数据不足（trading_days=1） | HFQ 未完整入库 | 调 `ensure_hfq_data()`，检查覆盖率 |
| 涨停判断全为 False | `limit_list_ths` 未同步 | 调 `ensure_limit_list_ths_data()` |
| 分钟线无数据 | 未走完整 DB→API 链路 | 用 `data_cleaner.get_kline_min_by_stock_date()` |
| 因子计算失败 | FACTOR_VERSION 过期 | 检查 `dataset.py` 版本，重跑生成 |

---

## 联系 & 反馈

- 📧 开发记录：见 CLAUDE.md 最后更新日期
- 🐛 Bug 追踪：检查 CLAUDE.md P0~P5 修复列表
- 📚 代码规范：参考各子模块的 README（features/README.md 等）