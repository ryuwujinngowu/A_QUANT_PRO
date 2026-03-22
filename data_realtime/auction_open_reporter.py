"""
集合竞价开盘报告
================
触发时机：每个交易日 9:25:10（集合竞价结束后）
使用方式：
  python data_realtime/auction_open_reporter.py
  python data_realtime/auction_open_reporter.py --date 20260317

功能流程：
  1. 从 DB 取所有 agent 最新日期的 signal_stock_detail（含 intraday_close_price / intraday_return）
  2. 调用 stk_auction 批量获取今日集合竞价均价
  3. 以昨日收盘价为基准，计算每只股今日开盘涨幅
  4. 每个 agent 输出：
       今日开盘：平均开盘涨幅 / 开盘最高 top3 / 开盘最低 top3
       截止昨收：平均收益 / 最高收益 top3 / 最大亏损 top3
  5. 微信推送
"""

import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import tushare as ts

from utils.db_utils import db          # 自动加载 config/.env
from utils.log_utils import logger
from utils.wechat_push import send_wechat_message_to_multiple_users
from agent_stats.agent_db_operator import AgentStatsDBOperator


# ─────────────────────────────────────────────────────────────────────────────
# Tushare 初始化
# ─────────────────────────────────────────────────────────────────────────────
API_REQUEST_INTERVAL = 1  # Tushare接口限流间隔（秒），统一管理
_TS_TOKEN_DEFAULT =  os.getenv("TS_TOKEN_DEFAULT")
_TUSHARE_API_URL  = "http://tushare.xyz"
DEFAULT_PAGE_LIMIT = 8000  # 分钟线接口分页大小（适配Tushare接口限制）
# ===================== 通用常量配置（统一管理，提升可维护性） =====================

def _init_pro():
    token = os.getenv("TS_TOKEN", _TS_TOKEN_DEFAULT)
    ts.set_token(token)
    pro = ts.pro_api()
    pro._DataApi__http_url = _TUSHARE_API_URL
    return pro

# ─────────────────────────────────────────────────────────────────────────────
# 辅助：批量查询股票名称
# ─────────────────────────────────────────────────────────────────────────────
def _get_stock_names(ts_codes: List[str]) -> Dict[str, str]:
    """从 stock_basic 批量查询名称，返回 {ts_code: name}"""
    if not ts_codes:
        return {}
    placeholders = ",".join(["%s"] * len(ts_codes))
    sql = f"SELECT ts_code, name FROM stock_basic WHERE ts_code IN ({placeholders})"
    try:
        rows = db.query(sql, params=tuple(ts_codes)) or []
        return {r["ts_code"]: r["name"] for r in rows}
    except Exception as e:
        logger.error(f"[集合竞价报告] 查询股票名称失败：{e}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# 辅助：集合竞价价格获取
# ─────────────────────────────────────────────────────────────────────────────
def _get_auction_prices(pro, ts_codes: List[str], trade_date: str) -> Dict[str, float]:
    """
    调用 stk_auction 获取今日集合竞价均价，返回 {ts_code: price}
    :param trade_date: YYYYMMDD 格式
    """
    if not ts_codes:
        return {}

    # stk_auction 支持逗号分隔多个代码；单次最大返回 8000 行，正常够用
    ts_code_str = ",".join(ts_codes)
    try:
        df = pro.stk_auction(
            ts_code=ts_code_str,
            trade_date=trade_date,
            fields="ts_code,price"
        )
        if df is None or df.empty:
            logger.warning(f"[集合竞价报告] stk_auction 返回空数据，trade_date={trade_date}")
            return {}
        return {row["ts_code"]: float(row["price"]) for _, row in df.iterrows() if row["price"]}
    except Exception as e:
        logger.error(f"[集合竞价报告] stk_auction 调用失败：{e}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# 辅助：格式化涨跌幅（自动加+/-符号）
# ─────────────────────────────────────────────────────────────────────────────
def _fmt(val: float) -> str:
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.2f}%"


# ─────────────────────────────────────────────────────────────────────────────
# 辅助：构建单个 agent 的报告文本块
# ─────────────────────────────────────────────────────────────────────────────
def _build_agent_block(
    agent_name: str,
    prev_date: str,
    intraday_avg_return: float,
    stock_list: List[Dict],
    name_map: Dict[str, str],
    auction_prices: Dict[str, float],
) -> str:
    """返回单个 agent 的文本块（不含尾部换行）"""

    lines = [f"▶ {agent_name}（昨日 {prev_date}，共 {len(stock_list)} 只）"]

    if not stock_list:
        lines.append("  昨日信号池为空")
        return "\n".join(lines)

    # ── 为每只股计算今日开盘涨幅 & 记录昨日收益 ──
    enriched = []
    for s in stock_list:
        ts_code      = s.get("ts_code", "")
        close_price  = s.get("intraday_close_price") or 0.0
        intraday_ret = float(s.get("intraday_return") or 0.0)   # 昨日各股收益%
        stock_name   = name_map.get(ts_code) or s.get("stock_name") or ts_code
        auction_p    = auction_prices.get(ts_code)

        if auction_p and close_price and close_price > 0:
            open_ret = (auction_p - close_price) / close_price * 100
        else:
            open_ret = None  # 无竞价数据（停牌/未返回）

        enriched.append({
            "ts_code":      ts_code,
            "stock_name":   stock_name,
            "close_price":  close_price,
            "auction_price": auction_p,
            "open_ret":     open_ret,
            "intraday_ret": intraday_ret,
        })

    valid = [e for e in enriched if e["open_ret"] is not None]

    # ════════════ 今日开盘部分 ════════════
    lines.append("  ┌─ 今日开盘")

    if not valid:
        lines.append("  │暂无竞价数据（可能停牌或接口未返回）")
    else:
        avg_open = sum(e["open_ret"] for e in valid) / len(valid)
        lines.append(f"  │平均开盘涨幅：{_fmt(avg_open)}"
                     f"  （{len(valid)}/{len(stock_list)} 只有竞价数据）")

        sorted_desc = sorted(valid, key=lambda x: x["open_ret"], reverse=True)

        # 开盘最高 top3
        top3 = sorted_desc[:3]
        lines.append("  │开盘最高 Top3：")
        for rank, e in enumerate(top3, 1):
            # price_info = (f"竞价{e['auction_price']:.2f} / 昨收{e['close_price']:.2f}"
            #               if e["auction_price"] else "")
            lines.append(f"  │{rank}. {e['stock_name']}({e['ts_code']})  "
                         f"{_fmt(e['open_ret'])} ")

        # 开盘最低 top3（仅当股票数 > 3，避免与 top3 完全重叠）
        if len(sorted_desc) > 3:
            bot3 = sorted_desc[-3:][::-1]   # 最低在前
            lines.append("  │开盘最低 Top3：")
            for rank, e in enumerate(bot3, 1):
                # price_info = (f"竞价{e['auction_price']:.2f} / 昨收{e['close_price']:.2f}"
                #               if e["auction_price"] else "")
                lines.append(f"  │{rank}. {e['stock_name']}({e['ts_code']})  "
                             f"{_fmt(e['open_ret'])} ")

    # ════════════ 昨日收盘部分 ════════════
    lines.append("  └─ 截止昨日收盘")
    lines.append(f"     平均收益：{_fmt(intraday_avg_return)}")

    if enriched:
        sorted_intraday = sorted(enriched, key=lambda x: x["intraday_ret"], reverse=True)

        # 最高收益 top3
        top3_y = sorted_intraday[:3]
        lines.append("     最高收益 Top3：")
        for rank, e in enumerate(top3_y, 1):
            lines.append(f"       {rank}. {e['stock_name']}({e['ts_code']})  "
                         f"{_fmt(e['intraday_ret'])}")

        # 最大亏损 top3（仅当股票数 > 3）
        if len(sorted_intraday) > 3:
            bot3_y = sorted_intraday[-3:][::-1]
            lines.append("     最大亏损 Top3：")
            for rank, e in enumerate(bot3_y, 1):
                lines.append(f"       {rank}. {e['stock_name']}({e['ts_code']})  "
                             f"{_fmt(e['intraday_ret'])}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────
def run_auction_report(trade_date: Optional[str] = None) -> None:
    """
    集合竞价报告主入口
    :param trade_date: 今日交易日，YYYYMMDD 格式；为 None 时取当日
    """
    time.sleep(10)
    today_str = trade_date or datetime.now().strftime("%Y%m%d")
    today_display = f"{today_str[:4]}-{today_str[4:6]}-{today_str[6:]}"
    logger.info(f"[集合竞价报告] 开始，交易日：{today_display}")

    pro   = _init_pro()
    db_op = AgentStatsDBOperator()

    # ── 1. 取所有 agent 最新有效日期 ──────────────────────────────────
    agent_last_dates = db_op.get_all_agents_last_dates()
    if not agent_last_dates:
        logger.warning("[集合竞价报告] DB 中无 agent 有效数据，跳过推送")
        return

    # ── 2. 按日期分组，批量拉取该日期所有 agent 的 signal + 摘要 ────────
    # agent_info: {agent_id: {agent_name, prev_date, intraday_avg_return, stock_list}}
    agent_info: Dict[str, Dict] = {}
    dates_seen = set(agent_last_dates.values())

    for prev_date in dates_seen:
        # 该日期下的统计摘要（含 intraday_avg_return）
        summary_map = {r["agent_id"]: r for r in db_op.get_latest_stats(prev_date)}

        EXCLUDED_AGENTS = {"afternoon_limit_up", "morning_limit_up"}

        for agent_id, agent_date in agent_last_dates.items():
            # ── 新增：剔除指定的 agent_id ──
            if agent_id in EXCLUDED_AGENTS:
                continue

            if agent_date != prev_date:
                continue
            summary = summary_map.get(agent_id, {})
            stock_list = db_op.get_signal_detail(agent_id, prev_date)
            agent_info[agent_id] = {
                "agent_name": summary.get("agent_name", agent_id),
                "prev_date": prev_date,
                "intraday_avg_return": float(summary.get("intraday_avg_return") or 0),
                "stock_list": stock_list,
            }

    # ── 3. 汇总所有股票代码，批量请求 ──────────────────────────────────
    all_ts_codes: List[str] = list({
        s["ts_code"]
        for info in agent_info.values()
        for s in info["stock_list"]
        if s.get("ts_code")
    })

    name_map       = _get_stock_names(all_ts_codes)
    auction_prices = _get_auction_prices(pro, all_ts_codes, today_str)
    logger.info(f"[集合竞价报告] 股票 {len(all_ts_codes)} 只，"
                f"获取竞价数据 {len(auction_prices)} 只")

    # ── 4. 逐 agent 构建推送内容 ───────────────────────────────────────
    sections = [f"【集合竞价开盘报告】{today_display} 09:25", "=" * 35]

    # 按昨日平均收益降序排列（好的 agent 排前面）
    sorted_agents = sorted(
        agent_info.items(),
        key=lambda x: x[1]["intraday_avg_return"],
        reverse=True
    )

    for agent_id, info in sorted_agents:
        block = _build_agent_block(
            agent_name          = info["agent_name"],
            prev_date           = info["prev_date"],
            intraday_avg_return = info["intraday_avg_return"],
            stock_list          = info["stock_list"],
            name_map            = name_map,
            auction_prices      = auction_prices,
        )
        sections.append(block)
        sections.append("-" * 35)

    sections.append("=" * 30)
    content = "\n".join(sections)
    title   = f"集合竞价报告 {today_display}"

    logger.info(f"[集合竞价报告] 推送内容 {len(content)} 字符，开始推送")
    send_wechat_message_to_multiple_users(title, content)
    logger.info("[集合竞价报告] 推送完成")


# ─────────────────────────────────────────────────────────────────────────────
# 命令行入口
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="集合竞价开盘报告推送")
    parser.add_argument(
        "--date", type=str, default=None,
        help="指定交易日期，格式 YYYYMMDD（默认：当日）"
    )
    args = parser.parse_args()
    run_auction_report(trade_date=args.date)
