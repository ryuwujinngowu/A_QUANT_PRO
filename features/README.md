# 特征层（features/）

## 架构概览

```
features/
├── __init__.py               # FeatureEngine 入口，注册并调度所有因子
├── base_feature.py           # 因子抽象基类（BaseFeature）
├── feature_registry.py       # 因子注册中心（单例模式，装饰器注册）
├── data_bundle.py            # 数据容器（FeatureDataBundle，一次 IO 预加载）
│
├── emotion/
│   └── sei_feature.py        # SEI/HDI 情绪因子（内部工具，不单独注册）
├── sector/
│   ├── sector_heat_feature.py    # 板块热度 + 轮动分（全局因子）
│   └── sector_stock_feature.py   # 板块个股全量特征（个股因子）
├── technical/
│   └── ma_position_feature.py    # 均线 + 位置因子（个股因子）
└── macro/
    └── market_macro_feature.py   # 市场宏观因子（全局因子）
```

## 数据流

```
dataset.py
  ↓ select_top3_hot_sectors(date) → top3_sectors + adapt_score
  ↓ 构建候选池 sector_candidate_map（过滤 ST/涨停封板/低流动性）
  ↓
FeatureDataBundle(trade_date, target_ts_codes, ...)
  ├─ _load_trade_dates()     → lookback_dates_5d / 20d
  ├─ _load_daily_data()      → daily_grouped dict（O(1) 查找，20 日不复权）
  ├─ _load_qfq_data()        → qfq_daily_grouped dict（20 日前复权，MA 专用）
  ├─ _load_macro_data()      → macro_cache（涨跌停/连板/板块/指数/5日历史趋势）
  └─ _load_minute_data()     → minute_cache（候选股近 5 日分钟线）
  ↓
FeatureEngine.run_single_date(data_bundle)
  ├─ sector_heat.calculate()      → 全局因子（adapt_score + 板块赚亏效应）
  ├─ sector_stock.calculate()     → 个股因子（120+ 列）
  ├─ ma_position.calculate()      → 个股因子（11 列）
  └─ market_macro.calculate()     → 全局因子（18 列）
  ↓
  个股级 inner join → 全局级 left join → feature_df
```

---

## 已注册因子一览

### 1. sector_heat — 板块热度因子（全局级）

**文件**: `sector/sector_heat_feature.py`
**输出类型**: 全局（无 stock_code，left join 广播到所有个股）

| 因子名 | 维度 | 计算逻辑 |
|--------|------|----------|
| `sector{1,2,3}_d{0-4}_profit` | 30 列 | 各板块各日上涨股的 SEI 均值（赚钱效应强度） |
| `sector{1,2,3}_d{0-4}_loss` | 同上 | 各板块各日下跌股的 100-SEI 均值（亏钱效应强度） |
| `adapt_score` | 1 列 | 板块轮动速度分 0-100（高=轮动快） |

**adapt_score 计算公式**:
```
overlap_rate   = 相邻日 Top5 板块加权重合度（Top2 双倍权重）
HHI            = Herfindahl-Hirschman 集中度指数（25 席位）
base_score     = (1 - overlap_rate) × 0.8 + (1 - HHI) / (1 - 0.04) × 0.2
adapt_score    = base_score × 100 × 主线修正系数
```

**Top3 板块选取规则**:
1. 规则 1：选 2 个核心题材（出现 ≥3 日 + 曾进 Top2，按加权得分排序）
2. 规则 2：选 1 个预期差题材（排名斜率 < 0，即上升趋势）
3. 兜底：不足 3 个时按总分补齐

---

### 2. sector_stock — 板块个股特征（个股级）

**文件**: `sector/sector_stock_feature.py`
**输出类型**: 个股级（含 stock_code），d0~d4 共 5 日各生成一组

#### 原始行情（5×5=25 列）

| 因子名 | 计算逻辑 |
|--------|----------|
| `stock_open/high/low/close_{d}` | 当日 OHLC（4 列×5 日=20 列；可通过 `train.py` 中 `EXCLUDE_PATTERNS` 过滤） |
| `stock_pct_chg_{d}` | 当日涨跌幅 |

#### 情绪合成分（3×5=15 列）

| 因子名 | 取值 | 计算逻辑 |
|--------|------|----------|
| `stock_profit_{d}` | 0-100 | 上涨日 = SEI；下跌/平盘日 = 0 |
| `stock_loss_{d}` | 0-100 | 下跌日 = 100-SEI；上涨/平盘日 = 0 |
| `stock_hdi_{d}` | 0-100 | HDI 持股难度指数（越高越煎熬） |

#### 方向/价格原子因子（3×5=15 列）

| 因子名 | 计算逻辑 |
|--------|----------|
| `stock_gap_return_{d}` | 开盘缺口率 = (open - pre_close) / pre_close。正=高开，负=低开 |
| `stock_candle_{d}` | K 线结构：2=真阳, 1=假阳, -1=假阴, -2=真阴 |
| `stock_cpr_{d}` | 收盘位置比 = (close - low) / (high - low)。0=收于最低，1=收于最高 |

#### 波动/持仓结构原子因子（5×5=25 列）

| 因子名 | 计算逻辑 |
|--------|----------|
| `stock_max_dd_{d}` | 日内最大回撤（分钟线累计高点到当前价的最大跌幅） |
| `stock_upper_shadow_{d}` | 上影线比率 = (high - max(open, close)) / pre_close |
| `stock_lower_shadow_{d}` | 下影线比率 = (min(open, close) - low) / pre_close |
| `stock_trend_r2_{d}` | 分钟线趋势 R²：0=震荡，1=单边趋势 |
| `stock_vwap_dev_{d}` | VWAP 偏离度 = mean(|close_i - vwap_i| / vwap_i) |

#### 涨跌停行为原子因子（3×5=15 列）

| 因子名 | 计算逻辑 |
|--------|----------|
| `stock_seal_times_{d}` | 涨停封板次数 |
| `stock_break_times_{d}` | 涨停开板次数（开板越多情绪越弱） |
| `stock_lift_times_{d}` | 跌停翘板次数 |

#### 时间持续类因子（4×5=20 列）

> 数据来源：当日分钟线（1min bars）+ 昨日日线（pre_close / 昨日 VWAP）

| 因子名 | 取值 | 计算逻辑 |
|--------|------|----------|
| `stock_red_time_ratio_{d}` | [0,1] | 分钟线中 close > pre_close（昨收）的比例 |
| `stock_float_profit_time_ratio_{d}` | [0,1] | 分钟线中 close > 昨日 VWAP 的比例（浮盈时间） |
| `stock_red_session_pm_ratio_{d}` | [-1,1] | 红盘分钟中处于午盘（13:00-15:00）的比例；-1=全天无红盘（最弱），0=早盘主导，0.5=均衡，1=午盘主导（最强）；无分钟线=0.5（中性） |
| `stock_float_session_pm_ratio_{d}` | [-1,1] | 浮盈分钟中处于午盘的比例；编码含义同上 |

> **昨日 VWAP 计算**：`kline_day.amount（千元）× 1000 / (kline_day.volume（手）× 100)` = `amount × 10 / volume`（元/股）
> 当昨日 VWAP 无法计算时（新股/D-1 停牌），回退使用 pre_close 作为参考价。

#### 量能因子（2×5=10 列）

| 因子名 | 计算逻辑 |
|--------|----------|
| `stock_vol_ratio_{d}` | 量比 = 当日成交量 / 近 5 日均量。放量>1，缩量<1，停牌=0 |
| `stock_amount_5d_ratio_{d}` | 成交额量级比 = 当日成交额 / 近 5 日均额。消除跨股绝对额差异，放量>1，缩量<1，停牌=0 |

#### 板块均值 + 排名（2×5+1=11 列）

| 因子名 | 计算逻辑 |
|--------|----------|
| `sector_avg_profit_{d}` | 该板块当日上涨股的 SEI 均值 |
| `sector_avg_loss_{d}` | 该板块当日下跌股的 100-SEI 均值 |
| `stock_sector_20d_rank` | 个股在所属板块内的 20 日涨幅排名 |

**小计**: ~141 列（stock_amount 替换为 stock_amount_5d_ratio）

---

### 3. ma_position — 均线+位置因子（个股级）

**文件**: `technical/ma_position_feature.py`
**输出类型**: 个股级，D 日截面（无 d0-d4 后缀）
**数据来源**: `kline_day_qfq`（前复权）优先，单日缺失时降级用 `kline_day`（不复权）

| 因子名 | 计算逻辑 |
|--------|----------|
| `ma5` / `ma10` / `ma13` | 均线原始价格（保留 CSV；`train.py` `EXCLUDE_PATTERNS` 默认排除，绝对价格跨股无可比性） |
| `bias5` / `bias10` / `bias13` | 乖离率 = (close - MA) / MA × 100 |
| `ma5_slope` | MA5 动能 = (MA5_D - MA5_{D-1}) / MA5_{D-1} |
| `ma_align` | 均线排列评分：2=完美多头, 1=弱多, 0=中性, -1=弱空, -2=完美空头 |
| `pos_20d` | 20 日价格位置 = (close - low_20d) / (high_20d - low_20d)，∈ [0,1] |
| `pos_5d` | 5 日价格位置，短期超买/超卖 |
| `from_high_20d` | 距 20 日最高点跌幅 = (high_20d - close) / high_20d |

**小计**: 11 列

---

### 4. market_macro — 市场宏观因子（全局级）

**文件**: `macro/market_macro_feature.py`
**输出类型**: 全局（无 stock_code，left join 广播）

| 因子名 | 计算逻辑 |
|--------|----------|
| `market_limit_up_count` | D 日涨停池股票数量 |
| `market_limit_down_count` | D 日跌停池股票数量 |
| `market_max_consec_num` | D 日最高连板数（市场高度） |
| `market_consec_2plus_count` | D 日 2 板及以上股票数量（连板梯队宽度） |
| `market_top_cpt_up_nums` | D 日最强板块涨停家数 |
| `market_top_cpt_cons_nums` | D 日最强板块连板家数 |
| `index_sh_pct_chg` | D 日上证指数涨跌幅 |
| `index_sz_pct_chg` | D 日深证成指涨跌幅 |
| `index_cyb_pct_chg` | D 日创业板指涨跌幅 |
| `market_vol_ratio_d{0-4}` | 全市场成交量比率 = vol_di / mean(vol_d0..d4)，窗口内归一化，>1=放量，<1=缩量，无数据=1.0 |
| `market_limit_up_rate` | 涨停参与率 = 涨停数 / 5200（全市场近似总股数），衡量赚钱效应广度 |
| `market_limit_up_5d_trend` | 涨停热度趋势 = D 日涨停数 / 近 4 日均值，clip [0.1, 10.0]；历史数据缺失时=1.0 |
| `market_consec_5d_trend` | 连板高度趋势 = D 日最高连板数 / 近 4 日均值，clip [0.1, 10.0]；历史数据缺失时=1.0 |

**小计**: 18 列

---

## 内部工具：SEI / HDI 情绪引擎

**文件**: `emotion/sei_feature.py`
**性质**: 不注册到 FeatureRegistry，由 `sector_stock_feature` 内部调用

### HDI（持股难度指数，0-100）

```
hdi_raw = 0.20 × 日内振幅
        + 0.30 × 日内最大回撤
        + 0.20 × 冲高回落距离
        + 0.10 × 涨跌幅绝对值
        + 0.10 × VWAP 穿越次数（归一化）
        + 0.10 × 量价背离率
        + K 线结构修正（-0.04 ~ +0.05）

hdi_score = clip(hdi_raw × 100, 0, 100)
```

### SEI（情绪强度指数，0-100）

```
base       = 50 + 50 × tanh(10 × ret_intra)
gap_adj    = gap_return × 20
trend_adj  = (trend_r2 - 0.5) × 15
vwap_adj   = -vwap_deviation × 20
candle_adj = {真阳: +8, 假阴: +3, 假阳: -3, 真阴: -8}
limit_adj  = ±封板/开板次数加减分

sei = clip(base + gap_adj + trend_adj + vwap_adj + candle_adj + limit_adj, 0, 100)
```

### 无分钟线回退

当分钟线不可用时，`calc_daily_atomic()` 从日线 OHLC 计算可得因子（gap_return / candle_type / cpr / 影线），不可计算的维度（max_dd / trend_r2 / vwap_dev）填充保守中性值。

---

## 扩展指南

新增因子只需 3 步：

```python
# 1. 创建文件 features/your_category/your_feature.py
from features.base_feature import BaseFeature
from features.feature_registry import feature_registry

@feature_registry.register("your_feature_name")  # 唯一标识
class YourFeature(BaseFeature):
    def calculate(self, data_bundle) -> tuple:
        # data_bundle 已预加载所有数据，禁止在此发起 IO
        feature_df = ...  # 含 stock_code + trade_date 列（个股级）
                          # 或仅含 trade_date 列（全局级，会 left join 广播）
        return feature_df, {}

# 2. 在 features/__init__.py 中添加 import
from features.your_category.your_feature import YourFeature  # noqa: F401

# 3. 更新 dataset.py 的 FACTOR_VERSION（触发训练集重跑）
```
