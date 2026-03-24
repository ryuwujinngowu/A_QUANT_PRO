"""
绘图核心函数（共用层）
=====================
回测层 + Agent 层共用的基础图表生成函数。
所有函数返回 plotly Figure 对象，由上层组装为最终 HTML。
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================
# 1. 资金/净值曲线（回测用 total_asset，agent 用累计收益率）
# ============================================================

def fig_equity_curve(dates, values, title="资金曲线", y_label="总资产(元)"):
    """
    绘制资金曲线 / 累计收益曲线。

    :param dates:   日期序列（list / Series）
    :param values:  金额序列（list / Series）
    :param title:   图表标题
    :param y_label: Y 轴标签
    :return: plotly Figure
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=values,
        mode="lines",
        name=y_label,
        line=dict(color="#1f77b4", width=2),
        fill="tozeroy",
        fillcolor="rgba(31,119,180,0.08)",
    ))
    fig.update_layout(
        title=title, xaxis_title="日期", yaxis_title=y_label,
        hovermode="x unified",
        template="plotly_white",
    )
    return fig


# ============================================================
# 2. 逐日回撤曲线
# ============================================================

def fig_drawdown(dates, values, title="最大回撤"):
    """
    根据资金/净值序列计算逐日回撤并绘图。

    :param dates:  日期序列
    :param values: 资金/净值序列
    :return: plotly Figure
    """
    s = pd.Series(values, dtype=float)
    cummax = s.cummax()
    drawdown = (s - cummax) / cummax * 100  # 百分比

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(dates), y=drawdown.tolist(),
        mode="lines",
        name="回撤(%)",
        line=dict(color="#d62728", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(214,39,40,0.12)",
    ))
    fig.update_layout(
        title=title, xaxis_title="日期", yaxis_title="回撤(%)",
        hovermode="x unified",
        template="plotly_white",
    )
    return fig


# ============================================================
# 3. 每笔交易盈亏柱状图（正=绿，负=红）
# ============================================================

def fig_trade_pnl_bar(trade_dates, pnl_values, labels=None, title="每笔交易盈亏"):
    """
    绘制每笔交易的盈亏柱状图。

    :param trade_dates: 交易日期
    :param pnl_values:  盈亏金额
    :param labels:      悬浮标签（如股票代码）
    :param title:       图表标题
    :return: plotly Figure
    """
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in pnl_values]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(len(pnl_values))),
        y=pnl_values,
        marker_color=colors,
        text=labels,
        hovertemplate="%{text}<br>日期: %{customdata}<br>盈亏: %{y:.2f}元<extra></extra>",
        customdata=trade_dates,
    ))
    fig.update_layout(
        title=title, xaxis_title="交易序号", yaxis_title="盈亏(元)",
        template="plotly_white",
    )
    return fig


# ============================================================
# 4. 胜率 / 收益分布直方图
# ============================================================

def fig_return_distribution(returns, title="收益分布", x_label="收益率(%)"):
    """
    绘制收益率分布直方图。

    :param returns: 收益率序列（百分比）
    :param title:   图表标题
    :param x_label: X 轴标签
    :return: plotly Figure
    """
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=40,
        marker_color="#636efa",
        opacity=0.75,
    ))
    fig.update_layout(
        title=title, xaxis_title=x_label, yaxis_title="频次",
        template="plotly_white",
    )
    return fig


# ============================================================
# 5. KPI 摘要卡片（嵌入 HTML 的表格区域）
# ============================================================

def build_kpi_html(kv_pairs: dict, title="核心指标") -> str:
    """
    生成 KPI 摘要的 HTML 片段（表格样式）。

    :param kv_pairs: {指标名: 值}
    :param title:    标题
    :return: HTML 字符串
    """
    rows = ""
    for k, v in kv_pairs.items():
        # 收益率/夏普等含负值时标红
        color = ""
        if isinstance(v, (int, float)):
            color = ' style="color:#d62728"' if v < 0 else ' style="color:#2ca02c"'
            v = f"{v:.2f}" if isinstance(v, float) else str(v)
        rows += f"<tr><td style='padding:6px 16px;font-weight:600'>{k}</td><td{color} style='padding:6px 16px'>{v}</td></tr>\n"

    return f"""
    <div style="margin:20px 0;">
        <h3 style="font-family:sans-serif;color:#333">{title}</h3>
        <table style="border-collapse:collapse;font-family:sans-serif;font-size:14px;">
            {rows}
        </table>
    </div>
    """


# ============================================================
# 6. 多图组装为单个 HTML 文件
# ============================================================

def assemble_html(figures: list, kpi_html: str = "", title="A-QUANT 报告") -> str:
    """
    将多个 plotly Figure + KPI HTML 拼装成完整的 HTML 页面。

    :param figures:  plotly Figure 对象列表
    :param kpi_html: KPI 摘要 HTML 片段
    :param title:    页面标题
    :return: 完整 HTML 字符串
    """
    import plotly.io as pio

    charts_html = ""
    for fig in figures:
        charts_html += pio.to_html(fig, full_html=False, include_plotlyjs=False)
        charts_html += "<hr style='border:none;border-top:1px solid #eee;margin:10px 0'>\n"

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
               max-width: 1200px; margin: 0 auto; padding: 20px; background: #fafafa; }}
        h1 {{ color: #333; border-bottom: 2px solid #1f77b4; padding-bottom: 10px; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    {kpi_html}
    {charts_html}
    <footer style="text-align:center;color:#999;font-size:12px;margin-top:40px;">
        Generated by A-QUANT Charting Engine
    </footer>
</body>
</html>"""
