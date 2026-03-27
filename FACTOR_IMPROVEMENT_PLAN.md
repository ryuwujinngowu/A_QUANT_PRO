# 因子优化与新增方案（2026-03-26）

## 优化点（修复现有 bug）

### 优化点 1：candle_type 分类
- **所属特征**：`features/emotion/sei_feature.py`
- **设计初衷**：将 K 线分为 4 类（真阳/假阳/假阴/真阴），用于情绪合成
- **现有计算逻辑**：
  ```python
  is_yang = close > open
  is_up   = close > pre_close
  # 无法处理平盘 (close==pre_close) 和十字星 (close==open)
  # 平盘会被误归为 -2（真阴）
  ```
- **建议修改方式**：
  ```python
  # 在判断前增加平盘检查
  if abs(close - pre_close) < 1e-6:
      return 0  # 平盘（中性）
  if abs(close - open) < 1e-6:
      return 0   # 十字星（中性）
  # 再进行 is_yang / is_up 判断
  ```
- **原因**：close==pre_close 时应返回 0（平盘），目前错误返回 -2（真阴），造成中性行情被标记为强空头信号，误导情绪指数

---

### 优化点 2：SEI 调整系数过大
- **所属特征**：`features/emotion/sei_feature.py`
- **设计初衷**：开盘缺口（gap）是主力意图的独立信号，应对 SEI 有显著调整
- **现有计算逻辑**：
  ```python
  gap_coeff = 20
  gap_adj = gap_return × 20
  # gap_return ∈ [-0.1, +0.1] → gap_adj ∈ [-2, +2]
  # 加到 base[50±50] 后，clip(0, 100) → 极端值被截断
  # 后果：高开/低开股的 SEI 全部堆积在两端
  ```
- **建议修改方式**：
  ```python
  gap_coeff = 5  # 或 6~8
  # 这样 gap_adj ∈ [-0.5, +0.5]，相对于 base 的影响 ±1%，合理
  ```
- **原因**：当前 gap_coeff=20 导致调整幅度为基础分的 ±4 倍，多个利好/利空叠加时全部被 clip 截断，SEI 失去区分度

---

### 优化点 3：market_vol_ratio_d0 自引用
- **所属特征**：`features/macro/market_macro_feature.py`
- **设计初衷**：衡量全市场当日成交量相对近期的放缩倍数
- **现有计算逻辑**：
  ```python
  all_vols = [vol_d0, vol_d1, vol_d2, vol_d3, vol_d4]
  avg_vol = mean(all_vols)
  market_vol_ratio_d0 = vol_d0 / avg_vol
  # 问题：d0 既在分子，又参与分母均值计算 → 自我套利
  ```
- **建议修改方式**：
  ```python
  # 方案：d0 用历史均值作分母
  hist_vols = [vol_d1, vol_d2, vol_d3, vol_d4]
  avg_vol_hist = mean(hist_vols)
  market_vol_ratio_d0 = vol_d0 / avg_vol_hist
  # d1~d4 保持原逻辑（用全部 5 日均值）
  ```
- **原因**：当日极端成交量（特别大或特别小）会被均值"套利"，无法准确反映相对历史的放缩倍数；该 bug 导致市场成交情绪因子失真

---

### 优化点 4：adapt_score HHI 固定基数
- **所属特征**：`features/sector/sector_heat_feature.py`
- **设计初衷**：用 HHI 集中度指数衡量 5 日内板块的分散程度，反映轮动快慢
- **现有计算逻辑**：
  ```python
  total_seats = 25  # 硬编码：假设每天 5 个板块 × 5 天
  hhi = sum((count / total_seats)² for count in sector_appear_count.values())
  # 问题：实际板块数可变，total_seats 不准确
  ```
- **建议修改方式**：
  ```python
  total_appear = sum(sector_appear_count.values())  # 动态计算
  if total_appear > 0:
      hhi = sum((count / total_appear)² for count in sector_appear_count.values())
  else:
      hhi = 0.04  # 默认等权
  ```
- **原因**：市场只有 3~4 个活跃板块时，total_seats=25 的假设会导致 HHI 严重偏低，adapt_score 被错误地标记为"轮动快"

---

### 优化点 5：fillna(0) 中性值语义混乱
- **所属特征**：所有需要中性值的因子（特别是 `pos_20d`, `bias*`）
- **设计初衷**：对缺失特征进行统一补填，保证训练集完整
- **现有计算逻辑**：
  ```python
  df = df.fillna(0)
  # 问题：pos_20d 的中性值应是 0.5（区间中点），不是 0（最低点）
  # fillna(0) 后，缺失的 pos_20d 被扭曲为"处在 20 日最低价"→ 极度看空
  ```
- **建议修改方式**：
  ```python
  neutral_values = {
      "pos_20d": 0.5, "pos_5d": 0.5,      # 位置中性值
      "bias5": 0.0, "bias10": 0.0,        # 乖离率中性值
      "cpr": 0.5,                          # 收盘位置中性值
      # ... 其他因子显式定义
  }
  df = df.fillna(neutral_values)
  ```
- **原因**：不同类型的因子有不同的中性值定义，统一 fillna(0) 会扭曲特征语义，造成模型学到虚假规律

---

### 优化点 6：label1 阈值文档-代码不符
- **所属特征**：`learnEngine/label.py`
- **设计初衷**：定义正样本：T+1 日内收益 ≥ 5%
- **现有计算逻辑**：
  ```python
  # 代码 line 95:
  label1 = 1 if d1_intra_return >= 0.03 else 0  # 实际用 3%
  # 文档注释 line 10:
  # label1: (D+1 close - D+1 open) / D+1 open >= 5% → 1
  ```
- **建议修改方式**：
  - 要么改代码：`>= 0.05`（5%）
  - 要么改文档：注释改为 `>= 3%`
  - **建议：统一为 5%**（策略目标更清晰）
- **原因**：训练标签阈值决定了正负样本分布，代码与文档不符会导致回测与实盘结果不一致

---

### 优化点 7：涯停基因过滤窗口过短
- **所属特征**：`learnEngine/dataset.py` 候选池筛选
- **设计初衷**：筛选出有涨停潜力的股票（近期活跃度指标）
- **现有计算逻辑**：
  ```python
  # line 434:
  limit_up_map = _check_stock_has_limit_up(candidates, date, day_count=10)
  keep = [ts for ts, has in limit_up_map.items() if has]
  # 问题：近 10 日无涨停的股全部过滤
  # 但沉默期后爆发的优质股（曾涨停过）被系统性丢弃
  ```
- **建议修改方式**：
  ```python
  # 方案 A：延长回看窗口
  day_count=30  # 改为 30 日或 60 日

  # 方案 B：分级策略（更精细）
  limit_up_strength = _check_stock_limit_up_strength(...)
  # 返回 {ts: strength} 其中 strength ∈ [0, 3]
  # 0=无基因, 1=历史有, 2=30日内有, 3=10日内有
  keep = [ts for ts, strength in limit_up_strength.items() if strength > 0]
  ```
- **原因**：10 日窗口会导致训练集严重偏向"最近活跃的涨停股"，无法学到"沉默期后爆发"的规律，泛化能力弱

---

## 新增因子（按优先级）

### 新增批次 A：时序聚合因子（立即可做，数据已有）

#### 新增点 1：momentum_5d（5 日累计动量）
- **因子维度**：时序动量
- **设计初衷**：捕捉 5 日连续涨幅的累积强度，替代冗余的 d0~d4 逐列
- **计算逻辑**：
  ```python
  momentum_5d = sum(pct_chg_d0~d4) * 100  # 单位：%
  # 取值范围：约 [-50, +50]（极端情况）
  ```
- **新增理由**：
  - 解决 d0~d4 高度相关导致的 ~100+ 列冗余
  - 提供聚合后的时序信号，更高效
  - 与 adapter 相关性更明显（轮动快 → 股价变动快）

---

#### 新增点 2：acceleration_2d（加速度）
- **因子维度**：时序加速度
- **设计初衷**：衡量最近 2 日动能是否在加强/减弱
- **计算逻辑**：
  ```python
  acceleration_2d = pct_chg_d0 - pct_chg_d1
  # 正数：加速上升，负数：动能衰减
  ```
- **新增理由**：
  - 捕捉"反弹加强"或"跌幅扩大"的瞬间信号
  - 对短线日内趋势敏感
  - 补充 momentum_5d 缺失的"瞬时加速"维度

---

#### 新增点 3：volatility_5d（5 日波动率）
- **因子维度**：时序波动性
- **设计初衷**：衡量股票在 5 日内的价格波动剧烈程度
- **计算逻辑**：
  ```python
  volatility_5d = std(pct_chg_d0~d4)
  # 或用极差：(max_pct_chg - min_pct_chg) / 5
  ```
- **新增理由**：
  - 区分"稳定上涨"vs"震荡上涨"→ 风险评估
  - 高波动率 + 正动量 = 高风险高收益
  - 现有特征缺乏对"震荡程度"的整体衡量

---

### 新增批次 B：筹码分布扩展因子（已有数据，需补充 MA 周期）

#### 新增点 4：bias20 / bias60（MA20/MA60 乖离率）
- **因子维度**：筹码分布位置
- **设计初衷**：衡量股价相对不同周期均线的位置，反映浮盈/套牢盘压力
- **计算逻辑**：
  ```python
  bias20 = (close - MA20) / MA20 * 100
  bias60 = (close - MA60) / MA60 * 100
  ```
- **新增理由**：
  - 现有因子只有 bias5/10/13，缺乏中长期均线参考
  - bias20 > 10% 表示浮盈盘众多（抛压强）
  - bias20 < -10% 表示套牢盘众多（支撑强）
  - 极其重要的筹码面信号，直接影响短线反弹空间

---

#### 新增点 5：pos_52w（52 周位置）
- **因子维度**：筹码分布位置
- **设计初衷**：衡量股价在年度高低点的相对位置，反映长期超买/超卖
- **计算逻辑**：
  ```python
  # 从 kline_day_hfq 统计近 252 个交易日的高低点
  pos_52w = (close - low_52w) / (high_52w - low_52w)
  # 取值 [0, 1]：0=年度低点，1=年度高点
  ```
- **新增理由**：
  - 现有 pos_20d 只看近 20 日，缺乏长期透视
  - pos_52w 接近 1 的股票容易回调（获利了结）
  - pos_52w 接近 0 的股票容易反弹（底部信号）
  - 对周期性选股极其重要

---

### 新增批次 C：换手率与流动性因子（已有数据）

#### 新增点 6：turnover_rate（日换手率）
- **因子维度**：换手率
- **设计初衷**：衡量当日股票流动性活跃程度，反映参与度
- **计算逻辑**：
  ```python
  # 从日线数据推导
  turnover_rate = volume / (float_shares * 100) * 100  # %
  # volume 单位：手，float_shares：流通股数（万）
  ```
- **新增理由**：
  - 区分"低换手高涨幅"（短线强）vs"高换手高涨幅"（不可持续）
  - 涨停板必然伴随超低换手，跌停板必然伴随超低换手 → 识别封板强度
  - 现有 vol_ratio 只反映量比，不反映绝对换手率

---

#### 新增点 7：turnover_5d_accel（换手加速度）
- **因子维度**：换手率加速度
- **设计初衷**：衡量近日参与度是否在增强
- **计算逻辑**：
  ```python
  # turnover_5d_avg = mean(turnover_rate_d0~d4)
  turnover_5d_accel = turnover_rate_d0 / turnover_5d_avg
  # 类似 vol_ratio，但更稳定
  ```
- **新增理由**：
  - 捕捉"涨幅放量"的经典买入信号
  - 换手加速 + 正动量 = 强势启动
  - 相比 vol_ratio，turnover 避免了跨股的成交额绝对值差异

---

### 新增批次 D：板块内多维排名（现有只有 20 日涨幅 1 个）

#### 新增点 8：stock_sector_pct_rank_d0（当日涨幅排名）
- **因子维度**：板块内相对强弱
- **设计初衷**：衡量股票在板块内当日的涨幅排名，识别板块内的领跑者
- **计算逻辑**：
  ```python
  # 对板块内所有股按 pct_chg_d0 排序
  stock_sector_pct_rank_d0 = rank(pct_chg_d0) / total_sector_count
  # 范围 [0, 1]：0=垫底，1=领跑
  ```
- **新增理由**：
  - 板块内最强的股比次强的更容易涨停 → 高优先级选股信号
  - 当日涨幅排名更敏感，反应日内主力动向
  - 与 adapt_score（轮动快）结合 → "轮动快的板块中的领跑股" = 高确定性

---

#### 新增点 9：stock_sector_vol_rank_d0（成交额排名）
- **因子维度**：板块内活跃度排名
- **设计初衷**：衡量股票在板块内的成交热度排名
- **计算逻辑**：
  ```python
  # 对板块内所有股按成交额排序
  stock_sector_vol_rank_d0 = rank(amount) / total_sector_count
  # 范围 [0, 1]：0=冷淡，1=最热
  ```
- **新增理由**：
  - 区分"涨幅高但无量"（风险，可能停牌或游资砸板）vs"涨幅高且有量"（确定性）
  - 龙虎榜前 20 的股票几乎都是板块内的成交额 Top 5
  - 弥补现有特征对"参与度排名"的缺失

---

#### 新增点 10：stock_sector_momentum_rank（5 日动量排名）
- **因子维度**：板块内中期趋势排名
- **设计初衷**：衡量股票在板块内的 5 日累计收益排名
- **计算逻辑**：
  ```python
  # 对板块内所有股按 momentum_5d 排序
  stock_sector_momentum_rank = rank(momentum_5d) / total_sector_count
  ```
- **新增理由**：
  - 识别"板块内的长期强者"
  - 与 stock_sector_pct_rank_d0 比较 → 区分"短期炒作"vs"趋势强股"
  - 例：momentum_rank 低但 pct_rank 高 = 急速补涨，高风险

---

### 新增批次 E：市值规模与流动性深度（已有数据）

#### 新增点 11：market_cap_rank（板块内市值排名）
- **因子维度**：市值规模
- **设计初衷**：区分大盘股/中盘股/小盘股，控制风险
- **计算逻辑**：
  ```python
  market_cap = close * float_shares  # 流通市值
  stock_sector_cap_rank = rank(market_cap) / total_sector_count
  ```
- **新增理由**：
  - 小盘股容易暴涨暴跌，风险管理必需
  - 某些日子小盘股容易涨停，某些日子大盘股领涨 → 需要分层
  - 现有特征缺乏对股票规模的控制

---

#### 新增点 12：depth_ratio（成交深度）
- **因子维度**：流动性深度
- **设计初衲**：衡量成交额相对市值的比例，反映日内流动性
- **计算逻辑**：
  ```python
  depth_ratio = amount_d0 / market_cap * 100  # %
  # 高 depth = 成交量活跃
  ```
- **新增理由**：
  - 识别"小盘股容易被拉升"的陷阱
  - depth_ratio > 5% 表示成交异常活跃，可能有游资
  - 结合 pos_52w 使用 → 极端组合识别

---

## 总体优化方案优先级

| 优先级 | 任务 | 工作量 | 预期收益 |
|--------|------|--------|--------|
| 🔴 P0 | 修复 market_vol_ratio_d0 自引用 | 10 分钟 | 高 |
| 🔴 P0 | 修复 label1 阈值（5% vs 3%） | 5 分钟 | 高 |
| 🟠 P1 | 修复 candle_type 平盘分类 | 15 分钟 | 中 |
| 🟠 P1 | 新增 momentum_5d / acceleration_2d / volatility_5d | 45 分钟 | 高 |
| 🟠 P1 | 新增 bias20 / bias60 / pos_52w | 60 分钟 | 高 |
| 🟡 P2 | 修复 SEI gap_coeff 过大 | 5 分钟 | 中 |
| 🟡 P2 | 修复 adapt_score HHI 固定值 | 10 分钟 | 中 |
| 🟡 P2 | 新增板块内多维排名（4 个因子） | 90 分钟 | 高 |
| 🟡 P2 | 新增 turnover_rate / turnover_accel | 60 分钟 | 中 |
| 🟢 P3 | 修复 fillna(0) 中性值语义 | 30 分钟 | 低 |
| 🟢 P3 | 修复涨停基因过滤窗口 | 20 分钟 | 中 |
| 🟢 P3 | 新增 market_cap_rank / depth_ratio | 40 分钟 | 低 |

