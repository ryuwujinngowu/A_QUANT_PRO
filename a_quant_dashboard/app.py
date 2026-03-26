"""
A_QUANT PRO 策略监控台 - FastAPI 后端
与主项目完全解耦，独立读取数据库
"""
import os
import json
import math
from datetime import date, timedelta
from typing import Optional
from contextlib import contextmanager

import pymysql
import pymysql.cursors
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

app = FastAPI(title="A_QUANT Dashboard", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── DB 配置 ────────────────────────────────────────────────────────────────────
_db_host = os.getenv("DB_HOST")
_db_port = os.getenv("DB_PORT", "3306")
_db_user = os.getenv("DB_USER")
_db_pass = os.getenv("DB_PASS")
_db_name = os.getenv("DB_NAME", "a_quant")

if not all([_db_host, _db_user, _db_pass]):
    raise RuntimeError("缺少数据库配置，请检查 .env 文件（DB_HOST / DB_USER / DB_PASS）")

DB_CONFIG = {
    "host":    _db_host,
    "port":    int(_db_port),
    "user":    _db_user,
    "password": _db_pass,
    "database": _db_name,
    "charset":  "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor,
    "connect_timeout": 10,
}


@contextmanager
def get_db():
    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            yield cur
    finally:
        conn.close()


def query(sql: str, params=None):
    with get_db() as cur:
        cur.execute(sql, params or [])
        return cur.fetchall()


def _default_range():
    end = date.today().strftime("%Y-%m-%d")
    start = (date.today() - timedelta(days=180)).strftime("%Y-%m-%d")
    return start, end


def _to_float(v, default=None):
    if v is None:
        return default
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 2)
    except Exception:
        return default


def _parse_json(v):
    if v is None:
        return None
    if isinstance(v, (dict, list)):
        return v
    try:
        return json.loads(v)
    except Exception:
        return None


def get_stock_names(ts_codes: list) -> dict:
    """从 stock_basic 批量获取股票名称 {ts_code: name}"""
    unique = list(set(str(t) for t in ts_codes if t))
    if not unique:
        return {}
    try:
        ph = ','.join(['%s'] * len(unique))
        rows = query(f"SELECT ts_code, name FROM stock_basic WHERE ts_code IN ({ph})", unique)
        return {r['ts_code']: r['name'] for r in rows}
    except Exception:
        return {}


def _enrich_detail_names(detail, name_map: dict) -> None:
    """就地填充 signal/next_day stock_detail 里 stock_list 的 stock_name"""
    if not isinstance(detail, dict):
        return
    for s in detail.get('stock_list', []):
        ts = s.get('ts_code') or s.get('code')
        if ts and not s.get('stock_name') and name_map.get(ts):
            s['stock_name'] = name_map[ts]


# ── Agent 元数据 ───────────────────────────────────────────────────────────────
AGENTS = {
    "short": [
        {"id": "hot_sector_dip_buy",   "name": "热板块低吸",   "desc": "热门板块早盘低吸反弹，09:30-10:30 低于开盘3%买入"},
        {"id": "model_dip_buy",        "name": "模型低吸",     "desc": "XGBoost 信号 + 恐慌低吸，D-1 选股 D 日低吸"},
        {"id": "model_open_buy",       "name": "模型开盘",     "desc": "XGBoost 信号 + 次日开盘价买入，关注隔夜溢价"},
        {"id": "model_surge_buy",      "name": "模型拉升",     "desc": "XGBoost 信号 + 早盘 5 分钟涨幅 >4% 跟进"},
        {"id": "high_position_stock",  "name": "高位股",       "desc": "近 20 日涨幅前 1% 强势股池，动量最强"},
        {"id": "mid_position_stock",   "name": "中位股",       "desc": "近 20 日涨幅 50%~1% 中间层，次强动量"},
        {"id": "limit_down_buy",       "name": "跌停反包",     "desc": "5 日涨幅 >30% 后首次跌停，反弹博弈"},
        {"id": "sector_top_high_open", "name": "板块龙头高开", "desc": "昨日热板块第一名 + 今日高开跟进"},
    ],
    "long": [
        {"id": "long_breakout_buy",       "name": "120日突破",  "desc": "HFQ 收盘突破 120 日新高 + 量能放大 1.5 倍，持有最长 60 天"},
        {"id": "long_ma_trend_tracking",  "name": "MA多头趋势", "desc": "MA5>10>20>60 多头排列 + 正向斜率 + 放量，持有最长 90 天"},
        {"id": "long_high_low_switch",    "name": "高低切换",   "desc": "高位钝化后转连板博弈，持有最长 10 天"},
    ],
}

# ── Overview ───────────────────────────────────────────────────────────────────
@app.get("/api/agents")
def get_agents():
    return AGENTS


@app.get("/api/overview")
def get_overview(start_date: Optional[str] = None, end_date: Optional[str] = None):
    if not end_date:
        _, end_date = _default_range()
    if not start_date:
        start_date, _ = _default_range()

    # Short-line 汇总
    short_rows = query("""
        SELECT agent_id,
               COUNT(*) AS total_days,
               AVG(next_day_avg_close_return) AS avg_return,
               SUM(CASE WHEN next_day_avg_close_return > 0 THEN 1 ELSE 0 END) AS win_days,
               MAX(next_day_avg_close_return) AS max_return,
               MIN(next_day_avg_close_return) AS min_return,
               AVG(next_day_avg_max_premium) AS avg_max_premium,
               AVG(next_day_avg_max_drawdown) AS avg_max_drawdown
        FROM agent_daily_profit_stats
        WHERE trade_date BETWEEN %s AND %s
          AND next_day_avg_close_return IS NOT NULL
          AND (reserve_str_1 IS NULL OR reserve_str_1 NOT LIKE '[ERR]%%')
          AND agent_id NOT LIKE 'long_%%'
        GROUP BY agent_id
    """, [start_date, end_date])

    short_summary = []
    for r in short_rows:
        td = r["total_days"] or 0
        r["win_rate"] = round((r["win_days"] or 0) / td * 100, 1) if td > 0 else 0
        for f in ["avg_return", "max_return", "min_return", "avg_max_premium", "avg_max_drawdown"]:
            r[f] = _to_float(r[f], 0)
        short_summary.append(r)

    # Long-line 汇总
    long_rows = query("""
        SELECT agent_id,
               COUNT(*) AS total_positions,
               SUM(CASE WHEN status = 0 THEN 1 ELSE 0 END) AS open_positions,
               SUM(CASE WHEN status = 1 THEN 1 ELSE 0 END) AS closed_positions,
               AVG(CASE WHEN status = 1 THEN period_return END) AS avg_return,
               SUM(CASE WHEN status = 1 AND period_return > 0 THEN 1 ELSE 0 END) AS win_positions,
               AVG(CASE WHEN status = 1 THEN trading_days END) AS avg_days
        FROM agent_long_position_stats
        WHERE buy_date BETWEEN %s AND %s
        GROUP BY agent_id
    """, [start_date, end_date])

    long_summary = []
    for r in long_rows:
        closed = r["closed_positions"] or 0
        win = r["win_positions"] or 0
        r["win_rate"] = round(win / closed * 100, 1) if closed > 0 else 0
        r["avg_return"] = _to_float(r["avg_return"], 0)
        r["avg_days"] = _to_float(r["avg_days"], 0)
        long_summary.append(r)

    # 各 agent 每日收益时间序列（用于 Overview 多线图）— 扩展包含日内收益和 win/loss 计数
    trend_rows = query("""
        SELECT agent_id, trade_date, next_day_avg_close_return, intraday_avg_return, next_day_stock_detail
        FROM agent_daily_profit_stats
        WHERE trade_date BETWEEN %s AND %s
          AND (reserve_str_1 IS NULL OR reserve_str_1 NOT LIKE '[ERR]%%')
          AND agent_id NOT LIKE 'long_%%'
        ORDER BY trade_date
    """, [start_date, end_date])

    trend_by_agent = {}
    intraday_by_agent = {}
    win_loss_by_agent = {}  # {agent_id: {date_str: {total, win, loss}}}

    for r in trend_rows:
        aid = r["agent_id"]
        date_str = str(r["trade_date"])

        if aid not in trend_by_agent:
            trend_by_agent[aid] = []
            intraday_by_agent[aid] = []
            win_loss_by_agent[aid] = {}

        t1_ret = _to_float(r["next_day_avg_close_return"])
        if t1_ret is not None:
            trend_by_agent[aid].append({"date": date_str, "return": t1_ret})

        id_ret = _to_float(r["intraday_avg_return"])
        if id_ret is not None:
            intraday_by_agent[aid].append({"date": date_str, "return": id_ret})

        detail = _parse_json(r.get("next_day_stock_detail"))
        if isinstance(detail, dict):
            sl = detail.get("stock_list", [])
            if sl:
                total = len(sl)
                win = sum(1 for s in sl if (s.get("close_return") or s.get("pct_chg") or 0) > 0)
                win_loss_by_agent[aid][date_str] = {"total": total, "win": win, "loss": total - win}

    # 长线策略月度盈亏均值（用于概览页图表）
    long_perf_rows = query("""
        SELECT agent_id,
               DATE_FORMAT(buy_date, '%%Y-%%m') AS ym,
               AVG(CASE WHEN period_return > 0 THEN period_return END) AS avg_win,
               AVG(CASE WHEN period_return < 0 THEN period_return END) AS avg_loss,
               SUM(CASE WHEN period_return > 0 THEN 1 ELSE 0 END) AS win_cnt,
               SUM(CASE WHEN period_return < 0 THEN 1 ELSE 0 END) AS loss_cnt
        FROM agent_long_position_stats
        WHERE status = 1 AND buy_date BETWEEN %s AND %s
        GROUP BY agent_id, DATE_FORMAT(buy_date, '%%Y-%%m')
        ORDER BY agent_id, ym
    """, [start_date, end_date])

    long_perf: dict = {}
    for r in long_perf_rows:
        aid = r["agent_id"]
        if aid not in long_perf:
            long_perf[aid] = []
        long_perf[aid].append({
            "ym":      r["ym"],
            "avg_win":  _to_float(r["avg_win"]),
            "avg_loss": _to_float(r["avg_loss"]),
            "win_cnt":  int(r["win_cnt"] or 0),
            "loss_cnt": int(r["loss_cnt"] or 0),
        })

    return {
        "short":          short_summary,
        "long":           long_summary,
        "trend":          trend_by_agent,
        "intraday_trend": intraday_by_agent,
        "win_loss":       win_loss_by_agent,
        "long_perf":      long_perf,
        "period":         {"start": start_date, "end": end_date},
    }


# ── Short-line ─────────────────────────────────────────────────────────────────
@app.get("/api/short/latest")
def get_short_latest(agent_id: str = Query(...)):
    """返回该 agent 最近一条有效信号记录（含持仓明细）"""
    rows = query("""
        SELECT trade_date, signal_stock_detail, next_day_stock_detail,
               next_day_avg_close_return, intraday_avg_return
        FROM agent_daily_profit_stats
        WHERE agent_id = %s
          AND (reserve_str_1 IS NULL OR reserve_str_1 NOT LIKE '[ERR]%%')
        ORDER BY trade_date DESC LIMIT 1
    """, [agent_id])
    if not rows:
        return None
    r = rows[0]
    r["trade_date"] = str(r["trade_date"])
    r["signal_stock_detail"] = _parse_json(r.get("signal_stock_detail"))
    r["next_day_stock_detail"] = _parse_json(r.get("next_day_stock_detail"))
    r["next_day_avg_close_return"] = _to_float(r.get("next_day_avg_close_return"))
    r["intraday_avg_return"] = _to_float(r.get("intraday_avg_return"))
    # 补充股票名称
    all_ts = set()
    for d in [r["signal_stock_detail"], r["next_day_stock_detail"]]:
        if isinstance(d, dict):
            for s in d.get("stock_list", []):
                ts = s.get("ts_code") or s.get("code")
                if ts:
                    all_ts.add(ts)
    name_map = get_stock_names(list(all_ts))
    for d in [r["signal_stock_detail"], r["next_day_stock_detail"]]:
        _enrich_detail_names(d, name_map)
    return r


@app.get("/api/short/daily")
def get_short_daily(
    agent_id: str = Query(...),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    if not end_date:
        _, end_date = _default_range()
    if not start_date:
        start_date, _ = _default_range()

    rows = query("""
        SELECT trade_date, agent_id, agent_name,
               intraday_avg_return,
               next_day_avg_open_premium,
               next_day_avg_close_return,
               next_day_avg_max_premium,
               next_day_avg_max_drawdown,
               next_day_avg_red_minute,
               next_day_avg_profit_minute,
               next_day_avg_intraday_profit,
               signal_stock_detail,
               next_day_stock_detail,
               reserve_str_1,
               reserve_str_2
        FROM agent_daily_profit_stats
        WHERE agent_id = %s
          AND trade_date BETWEEN %s AND %s
        ORDER BY trade_date
    """, [agent_id, start_date, end_date])

    result = []
    for r in rows:
        r["trade_date"] = str(r["trade_date"])
        for f in ["intraday_avg_return", "next_day_avg_open_premium",
                  "next_day_avg_close_return", "next_day_avg_max_premium",
                  "next_day_avg_max_drawdown", "next_day_avg_intraday_profit"]:
            r[f] = _to_float(r[f])
        # 解析 JSON，仅取 stock_list 数量
        detail = _parse_json(r.get("signal_stock_detail"))
        r["signal_count"] = len(detail.get("stock_list", [])) if isinstance(detail, dict) else 0
        r["signal_stock_detail"] = detail
        r["next_day_stock_detail"] = _parse_json(r.get("next_day_stock_detail"))
        r["has_error"] = bool(r.get("reserve_str_1") and str(r["reserve_str_1"]).startswith("[ERR]"))
        # 保留 reserve_str_2（策略描述），不做额外处理
        result.append(r)

    # 批量补充股票名称
    all_ts: set = set()
    for r in result:
        for key in ["signal_stock_detail", "next_day_stock_detail"]:
            d = r.get(key)
            if isinstance(d, dict):
                for s in d.get("stock_list", []):
                    ts = s.get("ts_code") or s.get("code")
                    if ts:
                        all_ts.add(ts)
    if all_ts:
        name_map = get_stock_names(list(all_ts))
        for r in result:
            for key in ["signal_stock_detail", "next_day_stock_detail"]:
                _enrich_detail_names(r.get(key), name_map)

    return result


# ── Long-line ──────────────────────────────────────────────────────────────────
@app.get("/api/long/positions")
def get_long_positions(
    agent_id: str = Query(...),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    status: Optional[int] = None,
):
    if not end_date:
        _, end_date = (date.today() - timedelta(days=0)).strftime("%Y-%m-%d"), date.today().strftime("%Y-%m-%d")
    if not start_date:
        start_date = (date.today() - timedelta(days=365)).strftime("%Y-%m-%d")

    sql = """
        SELECT id, agent_id, agent_name, ts_code, stock_name,
               buy_date, buy_price, sell_date, sell_price,
               period_return, trading_days, up_days, down_days,
               max_drawdown, max_floating_profit, status
        FROM agent_long_position_stats
        WHERE agent_id = %s AND buy_date BETWEEN %s AND %s
    """
    params: list = [agent_id, start_date, end_date]
    if status is not None:
        sql += " AND status = %s"
        params.append(status)
    sql += " ORDER BY buy_date DESC LIMIT 1000"

    rows = query(sql, params)
    # 补充缺失股票名称
    missing_ts = [r["ts_code"] for r in rows if not r.get("stock_name")]
    if missing_ts:
        name_map = get_stock_names(missing_ts)
        for r in rows:
            if not r.get("stock_name"):
                r["stock_name"] = name_map.get(r["ts_code"], "")
    for r in rows:
        r["buy_date"] = str(r["buy_date"])
        r["sell_date"] = str(r["sell_date"]) if r["sell_date"] else None
        for f in ["buy_price", "sell_price", "period_return", "max_drawdown", "max_floating_profit"]:
            r[f] = _to_float(r[f])

    # 为持仓中（status=0）股票计算当前收益
    open_ps = [r for r in rows if r["status"] == 0]
    if open_ps:
        ts_codes = list({p["ts_code"] for p in open_ps})
        ph = ','.join(['%s'] * len(ts_codes))
        try:
            latest_rows = query(f"""
                SELECT k.ts_code, k.close, k.trade_date
                FROM kline_day k
                INNER JOIN (
                    SELECT ts_code, MAX(trade_date) AS max_date
                    FROM kline_day WHERE ts_code IN ({ph})
                    GROUP BY ts_code
                ) m ON k.ts_code = m.ts_code AND k.trade_date = m.max_date
            """, ts_codes)
            price_map = {r2["ts_code"]: r2 for r2 in latest_rows}
        except Exception:
            price_map = {}

        today = date.today()
        for p in open_ps:
            lp = price_map.get(p["ts_code"])
            if lp and p.get("buy_price"):
                try:
                    cr = (float(lp["close"]) / float(p["buy_price"]) - 1) * 100
                    p["current_return"] = round(cr, 2)
                    p["latest_price_date"] = str(lp["trade_date"])
                except Exception:
                    p["current_return"] = None
            else:
                p["current_return"] = None
            try:
                buy_d = date.fromisoformat(p["buy_date"])
                p["days_held"] = (today - buy_d).days
            except Exception:
                p["days_held"] = None

    return rows


@app.get("/api/long/position/{position_id}")
def get_long_position_detail(position_id: int):
    rows = query("""
        SELECT id, agent_id, ts_code, stock_name,
               buy_date, buy_price, sell_date, sell_price,
               period_return, trading_days, up_days, down_days,
               max_drawdown, max_floating_profit, status, daily_detail
        FROM agent_long_position_stats WHERE id = %s
    """, [position_id])
    if not rows:
        raise HTTPException(status_code=404, detail="Position not found")
    r = rows[0]
    if not r.get("stock_name"):
        nm = get_stock_names([r["ts_code"]])
        r["stock_name"] = nm.get(r["ts_code"], "")
    r["buy_date"] = str(r["buy_date"])
    r["sell_date"] = str(r["sell_date"]) if r["sell_date"] else None
    for f in ["buy_price", "sell_price", "period_return", "max_drawdown", "max_floating_profit"]:
        r[f] = _to_float(r[f])
    r["daily_detail"] = _parse_json(r.get("daily_detail")) or []
    return r


@app.get("/api/long/summary")
def get_long_summary(
    agent_id: str = Query(...),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    """区间内汇总统计（不受 LIMIT 限制，用于卡片数据）"""
    if not end_date:
        _, end_date = _default_range()
    if not start_date:
        start_date = (date.today() - timedelta(days=365)).strftime("%Y-%m-%d")

    rows = query("""
        SELECT
            COUNT(*) AS total,
            SUM(status=0) AS open_cnt,
            SUM(status=1) AS closed_cnt,
            SUM(status=1 AND period_return > 0) AS win_cnt,
            AVG(CASE WHEN status=1 THEN period_return END) AS avg_return,
            AVG(CASE WHEN status=1 THEN trading_days END) AS avg_days
        FROM agent_long_position_stats
        WHERE agent_id = %s AND buy_date BETWEEN %s AND %s
    """, [agent_id, start_date, end_date])

    r = rows[0]
    closed = int(r["closed_cnt"] or 0)
    win = int(r["win_cnt"] or 0)

    # 止损统计：period_return <= -7% 视为触发止损
    _STOP_THRESH = -7.0
    stop_rows = query("""
        SELECT
            SUM(CASE WHEN status=1 AND period_return <= %s THEN 1 ELSE 0 END) AS stop_cnt,
            AVG(CASE WHEN status=1 AND period_return <= %s THEN trading_days END) AS avg_stop_days
        FROM agent_long_position_stats
        WHERE agent_id = %s AND buy_date BETWEEN %s AND %s
    """, [_STOP_THRESH, _STOP_THRESH, agent_id, start_date, end_date])
    sr = stop_rows[0]
    stop_cnt = int(sr["stop_cnt"] or 0)

    return {
        "total":           int(r["total"] or 0),
        "open":            int(r["open_cnt"] or 0),
        "closed":          closed,
        "win":             win,
        "win_rate":        round(win / closed * 100, 1) if closed > 0 else 0,
        "avg_return":      _to_float(r["avg_return"], 0),
        "avg_days":        _to_float(r["avg_days"], 0),
        "stop_loss_count": stop_cnt,
        "stop_loss_rate":  round(stop_cnt / closed * 100, 1) if closed > 0 else 0,
        "avg_stop_days":   _to_float(sr["avg_stop_days"]),
    }


@app.get("/api/long/aggregate")
def get_long_aggregate(
    agent_id: str = Query(...),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    """从 agent_daily_profit_stats 的聚合字段获取已平仓批次的统计"""
    if not end_date:
        _, end_date = _default_range()
    if not start_date:
        start_date = (date.today() - timedelta(days=365)).strftime("%Y-%m-%d")

    rows = query("""
        SELECT trade_date,
               long_median_return, long_max_return, long_min_return,
               long_avg_trading_days, long_closed_count
        FROM agent_daily_profit_stats
        WHERE agent_id = %s
          AND trade_date BETWEEN %s AND %s
          AND long_closed_count IS NOT NULL
          AND long_closed_count > 0
        ORDER BY trade_date
    """, [agent_id, start_date, end_date])

    for r in rows:
        r["trade_date"] = str(r["trade_date"])
        for f in ["long_median_return", "long_max_return", "long_min_return", "long_avg_trading_days"]:
            r[f] = _to_float(r[f])
    return rows


@app.get("/api/long/stop_stats")
def get_long_stop_stats(
    agent_id: str = Query(...),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    """月度止损率 + 平均止损天数 + 盈亏均值（用于详情页图表）"""
    if not start_date:
        start_date = (date.today() - timedelta(days=365)).strftime("%Y-%m-%d")
    if not end_date:
        end_date = date.today().strftime("%Y-%m-%d")

    _STOP_THRESH = -7.0
    rows = query("""
        SELECT
            DATE_FORMAT(buy_date, '%%Y-%%m') AS ym,
            SUM(CASE WHEN status=1 THEN 1 ELSE 0 END) AS closed,
            SUM(CASE WHEN status=1 AND period_return <= %s THEN 1 ELSE 0 END) AS stop_cnt,
            AVG(CASE WHEN status=1 AND period_return <= %s THEN trading_days END) AS avg_stop_days,
            AVG(CASE WHEN status=1 AND period_return > 0 THEN period_return END) AS avg_win,
            AVG(CASE WHEN status=1 AND period_return < 0 THEN period_return END) AS avg_loss
        FROM agent_long_position_stats
        WHERE agent_id = %s AND buy_date BETWEEN %s AND %s
        GROUP BY DATE_FORMAT(buy_date, '%%Y-%%m')
        ORDER BY ym
    """, [_STOP_THRESH, _STOP_THRESH, agent_id, start_date, end_date])

    result = []
    for r in rows:
        closed = int(r["closed"] or 0)
        stop = int(r["stop_cnt"] or 0)
        result.append({
            "ym":            r["ym"],
            "stop_rate":     round(stop / closed * 100, 1) if closed > 0 else 0,
            "avg_stop_days": _to_float(r["avg_stop_days"]),
            "stop_cnt":      stop,
            "closed":        closed,
            "avg_win":       _to_float(r["avg_win"]),
            "avg_loss":      _to_float(r["avg_loss"]),
        })
    return result


# ── 静态文件（放最后，避免覆盖 /api 路由）─────────────────────────────────────
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=False,
    )
