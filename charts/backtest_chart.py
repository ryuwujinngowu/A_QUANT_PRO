"""
回测报告图表
============
从 engine.run() 返回的 result_dict 中提取数据，生成交互式 HTML 报告。

数据颗粒度：
    - net_value_df: 每交易日一行（trade_date / total_asset / 当日收益率(%) / 累计收益率(%)）
    - trade_df:     每笔买卖一行（trade_date / ts_code / direction / price / volume / 卖出净盈亏）
"""
import os
from datetime import datetime

import pandas as pd

from charts.chart_core import (
    fig_equity_curve,
    fig_drawdown,
    fig_trade_pnl_bar,
    fig_return_distribution,
    build_kpi_html,
    assemble_html,
)
from utils.log_utils import logger

# 报告默认输出目录
_REPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "charts", "reports")


def generate_backtest_report(result: dict, output_dir: str = None) -> str:
    """
    根据回测结果生成交互式 HTML 报告。

    :param result:     engine.run() 返回的字典，必须含 net_value_df / trade_df
    :param output_dir: 输出目录（默认 charts/reports/）
    :return: HTML 文件路径
    """
    net_value_df: pd.DataFrame = result.get("net_value_df")
    trade_df: pd.DataFrame = result.get("trade_df")

    if net_value_df is None or net_value_df.empty:
        logger.warning("[Chart] net_value_df 为空，跳过图表生成")
        return ""

    # ── 输出路径 ──
    out_dir = output_dir or _REPORT_DIR
    os.makedirs(out_dir, exist_ok=True)

    strategy_name = result.get("策略名称", "未知策略")
    start_date = result.get("回测开始日期", "")
    end_date = result.get("回测结束日期", "")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"backtest_{ts}.html"
    filepath = os.path.join(out_dir, filename)

    # ── KPI 摘要 ──
    kpi = {}
    kpi_keys = [
        ("策略名称", str), ("回测开始日期", str), ("回测结束日期", str),
        ("初始本金(元)", float), ("最终资产(元)", float),
        ("总收益率(%)", float), ("年化收益率(%)", float),
        ("年化波动率(%)", float), ("最大回撤(%)", float),
        ("夏普比率", float), ("交易胜率(%)", float), ("总交易次数", int),
    ]
    for key, typ in kpi_keys:
        val = result.get(key)
        if val is not None:
            try:
                kpi[key] = typ(val)
            except (ValueError, TypeError):
                kpi[key] = val
    kpi_html = build_kpi_html(kpi, title="回测核心指标")

    # ── 图表列表 ──
    figures = []

    # 1. 资金曲线
    dates = net_value_df["trade_date"].tolist()
    assets = net_value_df["total_asset"].tolist()
    figures.append(fig_equity_curve(
        dates, assets,
        title=f"资金曲线 — {strategy_name}（{start_date} ~ {end_date}）",
    ))

    # 2. 逐日回撤
    figures.append(fig_drawdown(
        dates, assets,
        title="逐日回撤",
    ))

    # 3. 每日收益率曲线
    if "当日收益率(%)" in net_value_df.columns:
        daily_returns = net_value_df["当日收益率(%)"].tolist()
        figures.append(fig_equity_curve(
            dates, daily_returns,
            title="每日收益率",
            y_label="收益率(%)",
        ))

    # 4. 累计收益率曲线
    if "累计收益率(%)" in net_value_df.columns:
        cum_returns = net_value_df["累计收益率(%)"].tolist()
        figures.append(fig_equity_curve(
            dates, cum_returns,
            title="累计收益率",
            y_label="累计收益率(%)",
        ))

    # 5. 每笔交易盈亏柱状图（仅卖出记录有盈亏）
    if trade_df is not None and not trade_df.empty and "卖出净盈亏" in trade_df.columns:
        sell_df = trade_df[trade_df["direction"] == "卖出"].copy()
        if not sell_df.empty:
            figures.append(fig_trade_pnl_bar(
                trade_dates=sell_df["trade_date"].tolist(),
                pnl_values=sell_df["卖出净盈亏"].tolist(),
                labels=sell_df["ts_code"].tolist(),
                title="每笔卖出盈亏",
            ))

    # 6. 每日收益率分布
    if "当日收益率(%)" in net_value_df.columns:
        figures.append(fig_return_distribution(
            net_value_df["当日收益率(%)"].dropna().tolist(),
            title="每日收益率分布",
        ))

    # ── 组装 HTML ──
    page_title = f"回测报告 — {strategy_name}"
    html_content = assemble_html(figures, kpi_html=kpi_html, title=page_title)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"[Chart] 回测报告已生成: {filepath}")
    return filepath
