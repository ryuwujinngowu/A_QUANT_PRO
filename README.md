# A-Quant · A 股量化交易系统

基于机器学习（XGBoost）+ 板块轮动的 A 股短线量化策略框架。

---

## 项目架构

```
a_quant/
├── main.py                     # 项目总入口（驱动回测 / 实盘监控）
├── train.py                    # 模型训练入口（在根目录直接运行）
├── requirements.txt            # Python 依赖清单
│
├── config/                     # 配置层
│   ├── .env                    # 环境变量（Tushare Token / DB 连接，不提交 git）
│   └── config.py               # 配置加载 + 全局开关（FILTER_BSE_STOCK 等）
│
├── data/                       # 数据层
│   ├── data_fetcher.py         # Tushare 接口封装（日线 / 分钟线 / 涨跌停 / 指数 等）
│   ├── data_cleaner.py         # 数据清洗 + 入库（clean_and_insert_* 系列）
│   └── autoUpdating.py         # 每日自动更新脚本（定时拉取 kline_day + 宏观数据）
│
├── data_realtime/              # 实时数据层
│   └── data_realtime_fetcher.py  # 盘中实时行情拉取
│
├── features/                   # 特征工程层  ← 详见 features/README.md
│   ├── __init__.py             # FeatureEngine 入口 + 因子注册触发点
│   ├── base_feature.py         # 因子抽象基类 BaseFeature
│   ├── feature_registry.py     # 因子注册中心（单例 + 装饰器）
│   ├── data_bundle.py          # 数据容器 FeatureDataBundle（单次 IO 预加载）
│   ├── emotion/
│   │   └── sei_feature.py      # SEI/HDI 情绪引擎（内部工具，不单独注册）
│   ├── sector/
│   │   ├── sector_heat_feature.py    # 板块热度 + 轮动分（全局因子）
│   │   └── sector_stock_feature.py   # 板块个股全量特征（个股因子）
│   ├── technical/
│   │   └── ma_position_feature.py    # 均线 + 乖离率 + 价格位置（个股因子）
│   └── macro/
│       └── market_macro_feature.py   # 涨跌停 + 连板 + 最强板块 + 指数（全局因子）
│
├── learnEngine/                # 机器学习层  ← 详见 learnEngine/README.md
│   ├── dataset.py              # 训练集生成（逐日原子性 + 断点续跑）
│   ├── label.py                # 标签生成（label1 日内盈利 / label2 隔夜延续）
│   ├── model.py                # XGBoost 模型类（训练 / 推理 / 保存）
│   └── factor_ic.py            # 因子 IC 分析工具（ICIR / 有效性评估）
│
├── strategies/                 # 策略层
│   ├── base_strategy.py        # 策略抽象基类
│   ├── sector_heat_strategy.py # 板块热度 ML 策略（核心实盘策略）
│   ├── limit_up_strategy.py    # 涨停板策略
│   └── multi_limit_up_strategy.py # 连板策略
│
├── backtest/                   # 回测层
│   ├── engine.py               # 回测引擎（逐 K 线模拟）
│   ├── account.py              # 模拟账户（持仓 / 资金 / 订单）
│   └── metrics.py              # 绩效指标（夏普 / 最大回撤 / 胜率）
│
└── utils/                      # 工具层
    ├── common_tools.py         # 通用函数（交易日查询 / 日线批量拉取 / 涨停价计算等）
    ├── db_utils.py             # 数据库封装（query / batch_insert_df）
    ├── log_utils.py            # 日志管理
    └── wechat_push.py          # 微信消息推送（autoUpdating 完成时通知）
```

---

## 核心数据库表

| 表名 | 说明 |
|------|------|
| `kline_day` | 日线（不复权），OHLCV + amount（千元）+ pre_close |
| `kline_day_qfq` | 日线（前复权），供 MA 计算使用 |
| `kline_min` | 分钟线（1min），供 SEI/HDI 和时间因子计算 |
| `limit_list_ths` | 涨跌停池 / 炸板池（limit_type: 涨停池/跌停池/炸板池） |
| `limit_list_step` | 连板天梯（各股连板天数） |
| `limit_list_cpt` | 最强板块（连板家数 / 涨停家数） |
| `kline_index_daily` | 指数日线（上证 / 深证 / 创业板） |
| `trade_calendar` | 交易日历 |

> **单位注意**：`kline_day.amount` 单位为**千元**；`kline_day.vol` 单位为**手**（100 股/手）。
> 计算日 VWAP 时：`VWAP (元/股) = amount × 10 / vol`

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 config/.env（填入 Tushare Token 和 MySQL 连接信息）

# 3. 初始化数据库表结构（参考 data/structure.md）

# 4. 生成训练集（修改 dataset.py 内 START_DATE/END_DATE）
python learnEngine/dataset.py

# 5. 训练模型
python train.py

# 6. 分析因子有效性
python learnEngine/factor_ic.py

# 7. 运行回测 / 实盘
python main.py
```

---

## 重要开发规范

### 新增因子
详见 `features/README.md` → 扩展指南。核心 3 步：
1. 在 `features/<子目录>/` 新建文件，继承 `BaseFeature`，加 `@feature_registry.register("名字")` 装饰器
2. 在 `features/__init__.py` 添加 import（导入即注册）
3. 更新 `learnEngine/dataset.py` 的 `FACTOR_VERSION`

### 修改因子计算逻辑
只需改对应因子文件 → 更新 `FACTOR_VERSION` → 重跑 `dataset.py`（旧数据自动失效）

### 数据单位（高频易错）
- `kline_day.amount`: 千元（Tushare 标准）
- `kline_day.vol`: 手
- `kline_min.volume`: 股（个）
- `kline_min.amount`: 元
