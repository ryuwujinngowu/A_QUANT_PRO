# A_QUANT 策略监控台

量化交易策略数据可视化看板，与主项目 `A_QUANT_PRO` 完全解耦，独立部署。

**线上地址**：部署后访问 `http://<server-ip>:8000`

---

## 数据源

直连 `a_quant` 数据库，只读两张表：

| 表 | 用途 |
|----|------|
| `agent_daily_profit_stats` | 短线每日信号记录 + 长线批次聚合统计 |
| `agent_long_position_stats` | 长线每笔持仓完整生命周期 |

---

## 视图说明

### 全局概览
- KPI 卡片：总信号天数、平均胜率、最优策略、长线持仓中数量
- 所有短线策略累计收益多线图（ECharts line）
- 短线策略平均日收益横向对比柱状图

### 短线策略（8个）
| 策略 ID | 名称 | 逻辑简介 |
|---------|------|---------|
| `hot_sector_dip_buy` | 热板块低吸 | 热门板块早盘低吸反弹 |
| `model_dip_buy` | 模型低吸 | XGBoost 信号 + 恐慌低吸 |
| `model_open_buy` | 模型开盘 | XGBoost 信号 + 次日开盘买 |
| `model_surge_buy` | 模型拉升 | XGBoost 信号 + 早盘拉升跟进 |
| `high_position_stock` | 高位股 | 近 20 日涨幅前 1% |
| `mid_position_stock` | 中位股 | 近 20 日涨幅中间层 |
| `limit_down_buy` | 跌停反包 | 5 日涨幅>30% 后首次跌停 |
| `sector_top_high_open` | 板块龙头高开 | 昨日热板块第一 + 今日高开 |

每个策略展示：日收益柱状图、累计收益曲线、收益分布、溢价/回撤对比、开盘溢价散点、每日明细表。

### 长线策略（3个）
| 策略 ID | 名称 | 最长持有 |
|---------|------|---------|
| `long_breakout_buy` | 120日突破 | 60 天 |
| `long_ma_trend_tracking` | MA多头趋势 | 90 天 |
| `long_high_low_switch` | 高低切换 | 10 天 |

每个策略展示：收益/天数散点、收益分布、风险收益散点、权益曲线、持仓明细表（含每日净值弹窗）。

---

## 本地开发

```bash
cd E:\PyCharm\a_quant_dashboard

# 安装依赖（首次）
pip install -r requirements.txt

# 配置数据库
cp .env.example .env
# 编辑 .env 填入 DB 凭证

# 启动
python app.py
# 访问 http://localhost:8000
```

---

## 部署到服务器

```bash
# 增量同步（不覆盖 .env）
rsync -avz --exclude='.env' a_quant_dashboard/ <user>@<server-ip>:/opt/a_quant_dashboard/

# 重启服务
ssh <user>@<server-ip> 'systemctl restart a_quant_dashboard'
```

详细部署步骤见 [deploy.md](deploy.md)。

---

## 运维

```bash
# 查看服务状态
ssh <user>@<server-ip> 'systemctl status a_quant_dashboard'

# 实时日志
ssh <user>@<server-ip> 'journalctl -u a_quant_dashboard -f'

# 重启
ssh <user>@<server-ip> 'systemctl restart a_quant_dashboard'
```

---

## 文件结构

```
a_quant_dashboard/
├── app.py              # FastAPI 后端（路由 + DB 查询）
├── requirements.txt
├── .env.example        # 环境变量模板
├── .env                # 实际凭证（勿提交 git）
├── README.md           # 本文件
├── deploy.md           # 完整部署手册
└── static/
    └── index.html      # 单页前端（SPA）
```

---

## 迭代扩展

新增图表或视图的步骤：
1. **后端**：在 `app.py` 添加新 `@app.get("/api/...")` 端点
2. **前端**：在 `index.html` 的 JS 中添加对应 `render*` 函数 + ECharts option
3. **导航**：若新增策略，在 `AGENTS` 字典（`app.py`）和 `buildNav()`（`index.html`）中同步添加

阿里云安全组如需开放新端口，在控制台手动添加入方向规则（服务器本地无防火墙）。
