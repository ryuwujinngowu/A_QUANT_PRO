"""
Agent 统计报告图表
==================
从 agent_daily_profit_stats / agent_long_position_stats 两张表读取数据，
生成交互式 HTML 报告。

数据颗粒度：
    - agent_daily_profit_stats: 每 agent 每交易日一行（短线 T+1 信号的次日收益汇总）
    - agent_long_position_stats: 每 agent 每笔交易一行（买入→卖出完整记录）
"""
import os
from datetime import datetime
from typing import Optional

import pandas as pd

from charts.chart_core import (
    fig_equity_curve,
    fig_drawdown,
    fig_trade_pnl_bar,
    fig_return_distribution,
    build_kpi_html,
    assemble_html,
)
from utils.db_utils import db
from utils.log_utils import logger

# 报告默认输出目录
_REPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "charts", "reports")


# ============================================================
# 数据读取（封装 SQL，不依赖 db_operator 避免循环导入）
# ============================================================

def _query_short_agent_daily(agent_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    读取短线 agent 每日统计（agent_daily_profit_stats）。

    返回 DataFrame 列：trade_date, agent_name,
        next_day_avg_close_return, next_day_avg_open_premium,
        next_day_avg_max_premium, next_day_avg_max_drawdown
    """
    sql = """
        SELECT trade_date, agent_name,
               next_day_avg_close_return,
               next_day_avg_open_premium,
               next_day_avg_max_premium,
               next_day_avg_max_drawdown
        FROM agent_daily_profit_stats
        WHERE agent_id = %s
          AND trade_date BETWEEN %s AND %s
          AND reserve_str_1 IS NULL
        ORDER BY trade_date
    """
    rows = db.query(sql, params=(agent_id, start_date, end_date)) or []
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # trade_date 可能是 datetime.date，统一转 str
    df["trade_date"] = df["trade_date"].astype(str)
    return df


def _query_long_agent_positions(agent_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    读取长线 agent 持仓记录（agent_long_position_stats）。

    返回 DataFrame 列：ts_code, stock_name, buy_date, buy_price,
        sell_date, sell_price, status, period_return, trading_days,
        max_drawdown, max_floating_profit
    """
    sql = """
        SELECT ts_code, stock_name, buy_date, buy_price,
               sell_date, sell_price, status,
               period_return, trading_days,
               max_drawdown, max_floating_profit
        FROM agent_long_position_stats
        WHERE agent_id = %s
          AND buy_date BETWEEN %s AND %s
        ORDER BY buy_date
    """
    rows = db.query(sql, params=(agent_id, start_date, end_date)) or []
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["buy_date"] = df["buy_date"].astype(str)
    if "sell_date" in df.columns:
        df["sell_date"] = df["sell_date"].astype(str)
    return df


def _query_all_agent_ids() -> list:
    """返回所有有记录的 agent_id 列表"""
    sql = "SELECT DISTINCT agent_id, agent_name FROM agent_daily_profit_stats ORDER BY agent_id"
    rows = db.query(sql) or []
    return rows


# ============================================================
# 短线 Agent 报告
# ============================================================

def _generate_short_agent_report(agent_id: str, agent_name: str,
                                  df: pd.DataFrame, out_dir: str) -> str:
    """为单个短线 agent 生成报告 HTML"""
    figures = []
    dates = df["trade_date"].tolist()

    # ── 累计收益率曲线（从每日 avg_close_return 累乘）──
    daily_ret = df["next_day_avg_close_return"].fillna(0).astype(float)
    cum_ret = ((1 + daily_ret / 100).cumprod() - 1) * 100  # 百分比
    figures.append(fig_equity_curve(
        dates, cum_ret.tolist(),
        title=f"累计收益率 — {agent_name}",
        y_label="累计收益率(%)",
    ))

    # ── 逐日回撤（基于累计净值）──
    nav = (1 + daily_ret / 100).cumprod()
    figures.append(fig_drawdown(
        dates, nav.tolist(),
        title=f"逐日回撤 — {agent_name}",
    ))

    # ── 每日收盘收益率柱状图 ──
    figures.append(fig_trade_pnl_bar(
        trade_dates=dates,
        pnl_values=daily_ret.tolist(),
        labels=dates,
        title=f"每日平均收盘收益(%) — {agent_name}",
    ))

    # ── 收益分布 ──
    figures.append(fig_return_distribution(
        daily_ret.tolist(),
        title=f"每日收益分布 — {agent_name}",
        x_label="次日平均收盘收益(%)",
    ))

    # ── KPI ──
    total_days = len(df)
    win_days = (daily_ret > 0).sum()
    total_cum_ret = cum_ret.iloc[-1] if len(cum_ret) > 0 else 0
    max_dd = ((nav / nav.cummax()) - 1).min() * 100 if len(nav) > 0 else 0
    avg_ret = daily_ret.mean()
    # 简单夏普：avg / std * sqrt(252)
    std_ret = daily_ret.std()
    sharpe = (avg_ret / std_ret * (252 ** 0.5)) if std_ret > 0 else 0

    kpi = {
        "Agent": agent_name,
        "Agent ID": agent_id,
        "统计天数": total_days,
        "盈利天数": int(win_days),
        "日胜率(%)": round(win_days / total_days * 100, 2) if total_days > 0 else 0,
        "累计收益率(%)": round(float(total_cum_ret), 2),
        "最大回撤(%)": round(float(max_dd), 2),
        "日均收益(%)": round(float(avg_ret), 4),
        "夏普比率": round(float(sharpe), 2),
    }
    kpi_html = build_kpi_html(kpi, title=f"短线 Agent 核心指标")

    # ── 组装 ──
    html = assemble_html(figures, kpi_html=kpi_html, title=f"Agent 报告 — {agent_name}")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(out_dir, f"agent_short_{agent_id}_{ts}.html")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)
    return filepath


# ============================================================
# 长线 Agent 报告
# ============================================================

def _generate_long_agent_report(agent_id: str, agent_name: str,
                                 df: pd.DataFrame, out_dir: str) -> str:
    """为单个长线 agent 生成报告 HTML"""
    figures = []

    # ── 已平仓交易盈亏 ──
    closed = df[df["status"] == 1].copy()
    if not closed.empty and "period_return" in closed.columns:
        closed["period_return"] = closed["period_return"].fillna(0).astype(float)

        # 每笔交易盈亏柱状图
        figures.append(fig_trade_pnl_bar(
            trade_dates=closed["buy_date"].tolist(),
            pnl_values=closed["period_return"].tolist(),
            labels=[f"{r['ts_code']}({r.get('stock_name', '')})" for _, r in closed.iterrows()],
            title=f"每笔交易收益率(%) — {agent_name}",
        ))

        # 收益分布
        figures.append(fig_return_distribution(
            closed["period_return"].tolist(),
            title=f"交易收益分布 — {agent_name}",
            x_label="单笔收益率(%)",
        ))

        # 按买入日累计收益曲线
        sorted_closed = closed.sort_values("sell_date")
        cum_ret = ((1 + sorted_closed["period_return"] / 100).cumprod() - 1) * 100
        figures.append(fig_equity_curve(
            sorted_closed["sell_date"].tolist(), cum_ret.tolist(),
            title=f"累计收益率（按平仓日） — {agent_name}",
            y_label="累计收益率(%)",
        ))

    # ── 持仓天数分布 ──
    if not closed.empty and "trading_days" in closed.columns:
        figures.append(fig_return_distribution(
            closed["trading_days"].fillna(0).astype(float).tolist(),
            title=f"持仓天数分布 — {agent_name}",
            x_label="持仓天数",
        ))

    # ── KPI ──
    total_trades = len(closed)
    open_count = (df["status"] == 0).sum()
    win_trades = (closed["period_return"] > 0).sum() if not closed.empty else 0
    avg_ret = closed["period_return"].mean() if not closed.empty else 0
    max_ret = closed["period_return"].max() if not closed.empty else 0
    min_ret = closed["period_return"].min() if not closed.empty else 0
    avg_days = closed["trading_days"].mean() if not closed.empty and "trading_days" in closed.columns else 0

    kpi = {
        "Agent": agent_name,
        "Agent ID": agent_id,
        "已平仓交易数": int(total_trades),
        "未平仓数": int(open_count),
        "胜率(%)": round(win_trades / total_trades * 100, 2) if total_trades > 0 else 0,
        "平均收益(%)": round(float(avg_ret), 2),
        "最高收益(%)": round(float(max_ret), 2),
        "最低收益(%)": round(float(min_ret), 2),
        "平均持仓天数": round(float(avg_days), 1),
    }
    kpi_html = build_kpi_html(kpi, title="长线 Agent 核心指标")

    # ── 组装 ──
    html = assemble_html(figures, kpi_html=kpi_html, title=f"Agent 报告 — {agent_name}")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(out_dir, f"agent_long_{agent_id}_{ts}.html")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)
    return filepath


# ============================================================
# 对外入口
# ============================================================

def generate_agent_report(
    agent_id: str,
    start_date: str,
    end_date: str,
    agent_type: str = "auto",
    output_dir: str = None,
) -> str:
    """
    为指定 agent 生成可视化报告。

    :param agent_id:   agent 标识
    :param start_date: 起始日期 YYYY-MM-DD
    :param end_date:   截止日期 YYYY-MM-DD
    :param agent_type: "short" / "long" / "auto"（自动检测）
    :param output_dir: 输出目录
    :return: 生成的 HTML 文件路径
    """
    out_dir = output_dir or _REPORT_DIR
    os.makedirs(out_dir, exist_ok=True)

    # 自动检测：先查短线表，再查长线表
    if agent_type == "auto":
        short_df = _query_short_agent_daily(agent_id, start_date, end_date)
        long_df = _query_long_agent_positions(agent_id, start_date, end_date)
        # 哪边有数据就出哪边的报告（两边都有则都出）
        paths = []
        agent_name = agent_id
        if not short_df.empty:
            agent_name = short_df["agent_name"].iloc[0] if "agent_name" in short_df.columns else agent_id
            p = _generate_short_agent_report(agent_id, agent_name, short_df, out_dir)
            paths.append(p)
            logger.info(f"[Chart] 短线 Agent 报告已生成: {p}")
        if not long_df.empty:
            p = _generate_long_agent_report(agent_id, agent_name, long_df, out_dir)
            paths.append(p)
            logger.info(f"[Chart] 长线 Agent 报告已生成: {p}")
        if not paths:
            logger.warning(f"[Chart] agent_id={agent_id} 在 {start_date}~{end_date} 无数据")
            return ""
        return paths[0]  # 返回第一个生成的路径

    elif agent_type == "short":
        df = _query_short_agent_daily(agent_id, start_date, end_date)
        if df.empty:
            logger.warning(f"[Chart] agent_id={agent_id} 短线数据为空")
            return ""
        name = df["agent_name"].iloc[0] if "agent_name" in df.columns else agent_id
        p = _generate_short_agent_report(agent_id, name, df, out_dir)
        logger.info(f"[Chart] 短线 Agent 报告已生成: {p}")
        return p

    elif agent_type == "long":
        df = _query_long_agent_positions(agent_id, start_date, end_date)
        if df.empty:
            logger.warning(f"[Chart] agent_id={agent_id} 长线数据为空")
            return ""
        # 长线表无 agent_name，从短线表尝试获取
        name_rows = db.query(
            "SELECT agent_name FROM agent_daily_profit_stats WHERE agent_id = %s LIMIT 1",
            params=(agent_id,),
        )
        name = name_rows[0]["agent_name"] if name_rows else agent_id
        p = _generate_long_agent_report(agent_id, name, df, out_dir)
        logger.info(f"[Chart] 长线 Agent 报告已生成: {p}")
        return p

    else:
        logger.error(f"[Chart] 不支持的 agent_type: {agent_type}")
        return ""


def generate_all_agents_report(start_date: str, end_date: str, output_dir: str = None) -> list:
    """
    为所有有数据的 agent 批量生成报告。

    :return: 生成的 HTML 文件路径列表
    """
    agents = _query_all_agent_ids()
    if not agents:
        logger.warning("[Chart] 无 agent 记录")
        return []

    paths = []
    for row in agents:
        aid = row["agent_id"]
        p = generate_agent_report(aid, start_date, end_date, output_dir=output_dir)
        if p:
            paths.append(p)
    return paths
