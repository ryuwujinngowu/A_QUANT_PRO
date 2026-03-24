"""
绘图层 (charts)
================
提供回测 + Agent 统计的交互式可视化（plotly → HTML）。

对外接口：
    - generate_backtest_report(result_dict)  → 回测结束后调用，生成 HTML
    - generate_agent_report(...)             → Agent 统计数据可视化
"""
from charts.backtest_chart import generate_backtest_report
from charts.agent_chart import generate_agent_report
