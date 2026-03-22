#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据增量更新脚本（crontab 触发版）
====================================
部署路径  : /home/a_quant
Python    : python3.8

每日由 crontab 触发一次，拉取最新市场数据并推送结果到微信。
无重试逻辑 — 若数据不完整，手动重跑本脚本即可。

─── crontab 配置（每工作日下午 17:00）────────────────────────────────────
  0 17 * * 1-5 cd /home/a_quant && python3.8 data/autoUpdating.py >> logs/autoUpdating.log 2>&1

─── 手动触发 ──────────────────────────────────────────────────────────────
  cd /home/a_quant && python3.8 data/autoUpdating.py

─── 更新内容 ──────────────────────────────────────────────────────────────
  1. stock_basic   : 全量更新股票基础信息
  2. stock_st      : 增量更新 ST 风险警示数据
  3. kline_day     : 增量更新 A 股日线行情
  4. index_daily   : 增量更新核心指数日线（沪深300等）

─── 断点续跑 ──────────────────────────────────────────────────────────────
  每次运行完成后将当日日期写入 last_update_record，
  下次运行时从下一个交易日开始增量拉取，不重复处理历史数据。
"""

import datetime
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.data_cleaner import DataCleaner
from data.data_fetcher import data_fetcher
from utils.common_tools import (
    calc_incremental_date_range,
    read_last_update_record,
    write_update_record,
)
from utils.db_utils import db
from utils.log_utils import logger
from utils.wechat_push import send_wechat_message_to_multiple_users

cleaner = DataCleaner()


# ── 各表更新函数 ────────────────────────────────────────────────────────────

def update_stock_basic() -> int:
    """全量更新 stock_basic，返回影响行数"""
    logger.info("── stock_basic 全量更新 ──")
    try:
        affected = cleaner.clean_and_insert_stockbase(table_name="stock_basic") or 0
        logger.info(f"stock_basic 完成，影响 {affected} 行")
        return affected
    except Exception as e:
        logger.error(f"stock_basic 更新失败：{e}", exc_info=True)
        return 0


def update_stock_st(start_date: str, end_date: str) -> int:
    """增量更新 ST 数据，返回入库行数"""
    if not start_date or not end_date:
        return 0
    logger.info(f"── stock_st 增量更新（{start_date} ~ {end_date}）──")
    try:
        affected = cleaner.insert_stock_st(start_date=start_date, end_date=end_date) or 0
        logger.info(f"stock_st 完成，入库 {affected} 行")
        return affected
    except Exception as e:
        logger.error(f"stock_st 更新失败：{e}", exc_info=True)
        return 0


def update_kline_day(date_list: list) -> tuple:
    """
    增量更新日线数据，返回 (total_rows, per_date_dict)
    per_date_dict: {YYYYMMDD: row_count}
    """
    if not date_list:
        logger.info("── kline_day：无增量日期，跳过 ──")
        return 0, {}

    logger.info(f"── kline_day 增量更新（{len(date_list)} 个交易日）──")
    stock_codes = db.get_all_a_stock_codes()
    if not stock_codes:
        logger.error("无股票代码，跳过 kline_day 更新")
        return 0, {}

    BATCH_SIZE     = 600
    total          = 0
    per_date: dict = {}

    for trade_date in date_list:
        day_rows = 0
        batches  = [stock_codes[i:i + BATCH_SIZE] for i in range(0, len(stock_codes), BATCH_SIZE)]
        for batch in batches:
            try:
                raw_df   = data_fetcher.fetch_kline_day(",".join(batch), trade_date, trade_date)
                if raw_df.empty:
                    continue
                clean_df = cleaner._clean_kline_day_data(raw_df)
                if not clean_df.empty:
                    day_rows += db.batch_insert_df(clean_df, "kline_day", ignore_duplicate=True)
            except Exception as e:
                logger.error(f"kline_day {trade_date} 批次失败：{e}")
        per_date[trade_date] = day_rows
        logger.info(f"kline_day {trade_date}：入库 {day_rows} 行")
        total += day_rows

    logger.info(f"kline_day 累计入库 {total} 行")
    return total, per_date


def update_index_daily(start_date: str) -> int:
    """增量更新核心指数日线，返回入库行数"""
    logger.info("── index_daily 增量更新 ──")
    start_fmt = start_date.replace("-", "")
    end_fmt   = datetime.datetime.now().strftime("%Y%m%d")
    indexes   = ["000001.SH", "399001.SZ", "399006.SZ", "399107.SZ"]
    total     = 0
    try:
        for code in indexes:
            rows = cleaner.clean_and_insert_index_daily(
                ts_code=code, start_date=start_fmt, end_date=end_fmt
            ) or 0
            total += rows
            logger.info(f"  {code}：{rows} 行")
        logger.info(f"index_daily 累计入库 {total} 行")
        return total
    except Exception as e:
        logger.error(f"index_daily 更新失败：{e}", exc_info=True)
        return 0


# ── 推送格式化 ──────────────────────────────────────────────────────────────

def _build_push(
    last_date:  str,
    today:      str,
    inc_dates:  list,
    rows:       dict,       # {table: count}
    per_date:   dict,       # {YYYYMMDD: count}
) -> str:
    total = sum(rows.values())
    lines = [
        f"📅 上次记录：{last_date}  →  本次更新至：{today}",
        f"📆 增量天数：{len(inc_dates)} 天",
        f"",
        f"📊 各表入库行数：",
        f"  stock_basic : {rows.get('stock_basic', 0):>7,} 行",
        f"  stock_st    : {rows.get('stock_st',    0):>7,} 行",
        f"  kline_day   : {rows.get('kline_day',   0):>7,} 行",
        f"  index_daily : {rows.get('index_daily', 0):>7,} 行",
        f"  ──────────────────────────",
        f"  合计        : {total:>7,} 行",
    ]
    if per_date and len(per_date) > 1:
        lines += ["", "📋 kline 逐日明细："]
        for d, cnt in per_date.items():
            lines.append(f"  {d}: {cnt:>6,} 行")
    return "\n".join(lines)


# ── 主函数 ──────────────────────────────────────────────────────────────────

def main():
    today      = datetime.datetime.now().strftime("%Y-%m-%d")
    last_record = read_last_update_record()
    last_date   = last_record["last_update_date"]

    logger.info(f"===== 数据增量更新启动 | 上次记录：{last_date} | 本次：{today} =====")

    inc_dates = calc_incremental_date_range(last_date, today)
    logger.info(f"增量交易日：{inc_dates}")

    rows = {}
    rows["stock_basic"] = update_stock_basic()
    rows["stock_st"]    = update_stock_st(last_date, today)
    rows["kline_day"], per_date = update_kline_day(inc_dates)
    rows["index_daily"] = update_index_daily(last_date)

    # 无论数据是否完整，均将记录推进至今日
    # 如有缺漏，手动再次运行本脚本即可（本次运行后 last_date 仍是 today，
    # 下次手动运行 inc_dates 为空，相当于幂等重跑 — 若需重拉历史请手动修改记录文件）
    write_update_record(today, list(rows.keys()))

    push_msg = _build_push(last_date, today, inc_dates, rows, per_date)
    logger.info(f"\n{push_msg}")

    try:
        send_wechat_message_to_multiple_users(f"【数据更新】{today}", push_msg)
    except Exception as e:
        logger.error(f"微信推送失败：{e}")

    logger.info("===== 数据增量更新完成 =====")


if __name__ == "__main__":
    main()
