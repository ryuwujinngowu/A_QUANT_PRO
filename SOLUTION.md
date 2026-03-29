# 训练集生成完整解决方案

## 问题确认
✅ 宏观数据库表：数据正常（已入库）
✅ Fetcher：正常返回数据
❌ 训练集CSV：**不存在** → 这是数据缺失的真正原因！

## 根本原因
`learnEngine/datasets/train_dataset_latest.csv` 尚未生成。在这个CSV不存在的情况下，即使宏观数据已入库，也无法用于训练。

## 立即修复步骤

### 步骤 1：验证前置条件
```bash
# 检查是否有足够的特征和标签数据
python -c "
from data.data_fetcher import data_fetcher
from utils.common_tools import get_trade_dates

# 检查最近一日是否有数据
dates = get_trade_dates('2026-03-10', '2026-03-11')
for date in dates:
    df = data_fetcher.fetch_kline_day(date)
    print(f'{date}: {len(df)} 支股票')
"
```

### 步骤 2：运行数据集生成
```bash
# 清除已处理日期记录（强制重新生成）
rm -f processed_dates.json

# 生成训练集（这会花费一段时间，取决于日期范围）
python learnEngine/dataset.py
```

**预期输出**：
```
[INFO] 待处理日期（共 327 个）: ['2024-11-02', ..., '2026-03-11']
[INFO] 处理日期: 2024-11-02
...
[INFO] 所有日期已处理完成！
[INFO] 【最终校验】有效行数: XXXXX
```

### 步骤 3：验证生成的CSV
```bash
python -c "
import pandas as pd
import os

csv_path = 'learnEngine/datasets/train_dataset_latest.csv'
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print(f'✓ CSV已生成 | 行数: {len(df)} | 列数: {len(df.columns)}')

    # 检查关键列
    required = ['stock_code', 'trade_date', 'label1', 'label2', 'label_raw_return']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f'✗ 缺失列: {missing}')
    else:
        print(f'✓ 所有关键列齐全')
        print(f'  - 样本数据完整率 (label1): {df[\"label1\"].notna().sum() / len(df) * 100:.1f}%')
        print(f'  - 样本数据完整率 (label_raw_return): {df[\"label_raw_return\"].notna().sum() / len(df) * 100:.1f}%')
else:
    print('✗ CSV未生成，请检查 dataset.py 是否有错误')
" 2>&1 | grep -v "INFO\|WARNING"
```

## 如果 dataset.py 出错

### 排查宏观数据是否真正写入
```bash
python -c "
from utils.db_utils import db

tables = ['limit_list_ths', 'limit_step', 'limit_cpt_list', 'stock_risk_warning', 'index_daily']
for table in tables:
    result = db.query(f'SELECT COUNT(*) as cnt FROM {table}')
    if result:
        count = result[0].get('cnt', 0) if isinstance(result[0], dict) else 0
        print(f'{table}: {count} 行')
"
```

### 强制刷新宏观表（如需要）
```bash
python -c "
from data.data_cleaner import data_cleaner
from utils.common_tools import get_trade_dates
from datetime import datetime, timedelta

# 只补拉最近3个交易日的宏观数据
recent_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
dates = get_trade_dates(recent_date, datetime.now().strftime('%Y-%m-%d'))

for date in dates:
    date_fmt = date.replace('-', '')
    print(f'补拉 {date_fmt}...')
    data_cleaner.insert_stock_st(trade_date=date_fmt)
    data_cleaner.clean_and_insert_limit_list_ths(trade_date=date_fmt, limit_type='涨停池')
    data_cleaner.clean_and_insert_limit_list_ths(trade_date=date_fmt, limit_type='跌停池')
    data_cleaner.clean_and_insert_limit_step(trade_date=date_fmt)
    data_cleaner.clean_and_insert_limit_cpt_list(trade_date=date_fmt)
    data_cleaner.clean_and_insert_index_daily(trade_date=date_fmt)
"
```

## 训练流程（CSV生成后）

### 生成模型
```bash
python train.py
```

**预期输出**：
```
[INFO] 特征列数: XXX | 正样本率: XX.X%
[INFO] 时间序列切分 | 训练集: XXXX 行 | 验证集: XXX 行
[INFO] AUC: 0.XXXX | Precision: 0.XXXX | Recall: 0.XXXX
[INFO] 训练完成！版本化模型: model/sector_heat_xgb_v4.0_loss_severity_raw_return.pkl
```

## 常见错误排查

| 错误 | 原因 | 解决方案 |
|------|------|--------|
| `ModuleNotFoundError: stk_factor_feature` | 特征模块不完整 | ✓ 已修复（try/except包装） |
| `DataFrame 为空` | 特征计算失败 | 检查 FeatureEngine.run_single_date 是否返回数据 |
| `标签为空` | 标签计算失败 | 检查 LabelEngine.generate_single_date 是否返回数据 |
| `CSV列不匹配` | 特征列在运行中变更 | 删除 processed_dates.json 重新生成 |

