"""
DB 操作封装
===========
所有 agent_daily_profit_stats 表的读写收口于此，引擎不直接执行 SQL。
字段约定：
  reserve_str_1 — 错误信息占位（非空说明该记录计算时出现异常）
  reserve_str_2 — 策略描述（agent_desc）
"""
import json
from typing import Dict, List, Optional, Set
from utils.db_utils import db
from utils.log_utils import logger
from agent_stats.config import STATS_TABLE_NAME


def _first(rows):
    """从 db.query() 结果列表中安全取第一行，无结果返回 None。"""
    if rows and isinstance(rows, list) and len(rows) > 0:
        return rows[0]
    return None


class AgentStatsDBOperator:
    def __init__(self):
        self.t = STATS_TABLE_NAME

    # ------------------------------------------------------------------ #
    # 查询方法
    # ------------------------------------------------------------------ #

    def get_all_agents_last_dates(self) -> Dict[str, str]:
        """
        一次查询返回所有 agent 在 DB 中最后一条「有效」记录的日期。
        有效 = reserve_str_1 IS NULL 或不以 [ERR] 开头（含 [MIN_FAIL] 等部分失败记录）。

        ⚠ 刻意排除纯错误记录（[ERR]）：
          若某 agent 历史数据全部为 [ERR]（如因代码 bug 导致每日失败），
          MAX(trade_date) 会指向最新的错误日期，引擎误认为该 agent 已追上最新
          而不再重跑任何日期。排除 [ERR] 后，引擎会把错误日期重新纳入
          dates_missing，下次运行时通过 ON DUPLICATE KEY UPDATE 自动修复。

        {agent_id: "YYYY-MM-DD"} 或 {} (无有效记录时对应 agent_id 不在字典里)
        """
        sql = f"""
            SELECT agent_id, MAX(trade_date) AS max_date
            FROM {self.t}
            WHERE reserve_str_1 IS NULL
               OR reserve_str_1 NOT LIKE %s
            GROUP BY agent_id
        """
        try:
            rows = db.query(sql, params=('[ERR]%',)) or []
            return {
                r["agent_id"]: r["max_date"].strftime("%Y-%m-%d")
                for r in rows if r["max_date"]
            }
        except Exception as e:
            logger.error(f"get_all_agents_last_dates 失败：{e}")
            return {}

    def get_agents_err_dates(self) -> Dict[str, int]:
        """
        返回各 agent 的 [ERR] 记录条数（用于断点日志展示诊断信息）。
        {agent_id: err_count}
        """
        sql = f"""
            SELECT agent_id, COUNT(*) AS err_count
            FROM {self.t}
            WHERE reserve_str_1 LIKE %s
            GROUP BY agent_id
        """
        try:
            rows = db.query(sql, params=('[ERR]%',)) or []
            return {r["agent_id"]: int(r["err_count"]) for r in rows}
        except Exception as e:
            logger.error(f"get_agents_err_dates 失败：{e}")
            return {}

    def get_agents_closed_dates(self) -> Dict[str, Set[str]]:
        """
        返回各 agent 中已完成 D+1 结账的日期集合（next_day_avg_close_return IS NOT NULL）。
        {agent_id: {"2024-10-01", "2024-10-02", ...}}
        """
        sql = f"""
            SELECT agent_id, trade_date
            FROM {self.t}
            WHERE next_day_avg_close_return IS NOT NULL
        """
        try:
            rows = db.query(sql) or []
            result: Dict[str, Set[str]] = {}
            for r in rows:
                aid  = r["agent_id"]
                date = r["trade_date"].strftime("%Y-%m-%d")
                result.setdefault(aid, set()).add(date)
            return result
        except Exception as e:
            logger.error(f"get_agents_closed_dates 失败：{e}")
            return {}

    def get_agent_recorded_dates(self, agent_id: str) -> List[str]:
        """返回某 agent 所有已入库的交易日列表（升序，含 ERR 记录）"""
        sql = f"SELECT trade_date FROM {self.t} WHERE agent_id = %s ORDER BY trade_date ASC"
        try:
            rows = db.query(sql, params=(agent_id,)) or []
            return [r["trade_date"].strftime("%Y-%m-%d") for r in rows]
        except Exception as e:
            logger.error(f"[{agent_id}] get_agent_recorded_dates 失败：{e}")
            return []

    def get_agent_err_date_list(self, agent_id: str) -> List[str]:
        """返回某 agent 所有 [ERR] 记录的交易日列表（升序）"""
        sql = f"""
            SELECT trade_date FROM {self.t}
            WHERE agent_id = %s AND reserve_str_1 LIKE %s
            ORDER BY trade_date ASC
        """
        try:
            rows = db.query(sql, params=(agent_id, '[ERR]%')) or []
            return [r["trade_date"].strftime("%Y-%m-%d") for r in rows]
        except Exception as e:
            logger.error(f"[{agent_id}] get_agent_err_date_list 失败：{e}")
            return []

    def get_signal_detail(self, agent_id: str, trade_date: str) -> List[Dict]:
        """读取指定 agent + 日期的 signal_stock_detail（stock_list 字段）"""
        sql = f"""
            SELECT signal_stock_detail FROM {self.t}
            WHERE agent_id = %s AND trade_date = %s
        """
        try:
            row = _first(db.query(sql, params=(agent_id, trade_date)))
            if not row or not row.get("signal_stock_detail"):
                return []
            detail = row["signal_stock_detail"]
            if isinstance(detail, str):
                detail = json.loads(detail)
            return detail.get("stock_list", [])
        except Exception as e:
            logger.error(f"[{agent_id}][{trade_date}] get_signal_detail 失败：{e}")
            return []

    def get_unclosed_records(self, trade_date: str) -> List[Dict]:
        """查询指定日期待结账记录（仅作外部调用，引擎内部不再使用）"""
        sql = f"""
            SELECT agent_id, trade_date, signal_stock_detail
            FROM {self.t}
            WHERE trade_date = %s AND next_day_avg_close_return IS NULL
        """
        try:
            rows = db.query(sql, params=(trade_date,)) or []
            for r in rows:
                if isinstance(r.get("signal_stock_detail"), str):
                    r["signal_stock_detail"] = json.loads(r["signal_stock_detail"])
            return rows
        except Exception as e:
            logger.error(f"get_unclosed_records 失败：{e}")
            return []

    def get_last_processed_date(self, agent_id: str) -> Optional[str]:
        """兼容旧调用（run.py wechat_reporter）"""
        sql = f"SELECT MAX(trade_date) AS max_date FROM {self.t} WHERE agent_id = %s"
        try:
            row = _first(db.query(sql, params=(agent_id,)))
            return row["max_date"].strftime("%Y-%m-%d") if row and row["max_date"] else None
        except Exception as e:
            logger.error(f"[{agent_id}] get_last_processed_date 失败：{e}")
            return None

    def get_latest_stats(self, trade_date: str) -> List[Dict]:
        """获取指定日期所有 agent 的统计摘要（用于微信推送）"""
        sql = f"""
            SELECT agent_id, agent_name, trade_date,
                   intraday_avg_return,
                   next_day_avg_open_premium, next_day_avg_close_return,
                   next_day_avg_max_premium, next_day_avg_max_drawdown,
                   next_day_avg_red_minute, next_day_avg_profit_minute,
                   reserve_str_1
            FROM {self.t}
            WHERE trade_date = %s
            ORDER BY intraday_avg_return DESC
        """
        try:
            return db.query(sql, params=(trade_date,)) or []
        except Exception as e:
            logger.error(f"get_latest_stats 失败：{e}")
            return []

    # ------------------------------------------------------------------ #
    # 写入方法
    # ------------------------------------------------------------------ #

    def insert_signal_record(self, record: Dict) -> bool:
        """
        幂等插入选股信号记录（ON DUPLICATE KEY UPDATE）。
        已存在时仅更新 intraday/detail 字段，不覆盖 next_day 字段。
        reserve_str_2 存储策略描述（agent_desc），每次写入覆盖以保持最新。
        """
        sql = f"""
            INSERT INTO {self.t} (
                agent_id, agent_name, trade_date,
                intraday_avg_return, signal_stock_detail,
                reserve_str_2, create_time, update_time
            ) VALUES (%s, %s, %s, %s, %s, %s, NOW(), NOW())
            ON DUPLICATE KEY UPDATE
                agent_name           = VALUES(agent_name),
                intraday_avg_return  = VALUES(intraday_avg_return),
                signal_stock_detail  = VALUES(signal_stock_detail),
                reserve_str_2        = VALUES(reserve_str_2),
                reserve_str_1        = NULL,
                update_time          = NOW()
        """
        try:
            db.execute(sql, (
                record["agent_id"],
                record["agent_name"],
                record["trade_date"],
                record["intraday_avg_return"],
                json.dumps(record["signal_stock_detail"], ensure_ascii=False),
                record.get("agent_desc", ""),
            ))
            return True
        except Exception as e:
            logger.error(f"[{record['agent_id']}][{record['trade_date']}] insert_signal_record 失败：{e}")
            return False

    def update_next_day_stats(self, agent_id: str, trade_date: str, stats: Dict) -> bool:
        """更新 D+1 隔日表现字段（结账操作）"""
        sql = f"""
            UPDATE {self.t} SET
                next_day_avg_open_premium   = %s,
                next_day_avg_close_return   = %s,
                next_day_avg_red_minute     = %s,
                next_day_avg_profit_minute  = %s,
                next_day_avg_intraday_profit= %s,
                next_day_avg_max_premium    = %s,
                next_day_avg_max_drawdown   = %s,
                next_day_stock_detail       = %s,
                update_time                 = NOW()
            WHERE agent_id = %s AND trade_date = %s
        """
        try:
            db.execute(sql, (
                stats["next_day_avg_open_premium"],
                stats["next_day_avg_close_return"],
                stats["next_day_avg_red_minute"],
                stats["next_day_avg_profit_minute"],
                stats["next_day_avg_intraday_profit"],
                stats["next_day_avg_max_premium"],
                stats["next_day_avg_max_drawdown"],
                json.dumps(stats["next_day_stock_detail"], ensure_ascii=False),
                agent_id, trade_date,
            ))
            return True
        except Exception as e:
            logger.error(f"[{agent_id}][{trade_date}] update_next_day_stats 失败：{e}")
            return False

    def insert_error_record(
        self, agent_id: str, agent_name: str, trade_date: str, error_msg: str
    ) -> None:
        """
        当 agent 某日运算失败时，插入一条 error 占位记录。
        reserve_str_1 存储错误摘要，后续可根据此字段排查问题。
        幂等：若当日已有正常记录，仅更新 reserve_str_1 不覆盖数据。
        """
        err = f"[ERR]{error_msg[:230]}"   # 列宽 256
        sql = f"""
            INSERT INTO {self.t} (
                agent_id, agent_name, trade_date,
                intraday_avg_return, signal_stock_detail,
                reserve_str_1, create_time, update_time
            ) VALUES (%s, %s, %s, 0, %s, %s, NOW(), NOW())
            ON DUPLICATE KEY UPDATE
                reserve_str_1 = VALUES(reserve_str_1),
                update_time   = NOW()
        """
        try:
            db.execute(sql, (
                agent_id, agent_name, trade_date,
                json.dumps({"stock_list": []}, ensure_ascii=False),
                err,
            ))
        except Exception as e:
            logger.error(f"[{agent_id}][{trade_date}] insert_error_record 失败：{e}")

    def mark_error(self, agent_id: str, trade_date: str, error_msg: str) -> None:
        """标记已有记录出现错误（仅更新 reserve_str_1，不影响其他字段）"""
        sql = f"""
            UPDATE {self.t} SET
                reserve_str_1 = %s,
                update_time   = NOW()
            WHERE agent_id = %s AND trade_date = %s
        """
        try:
            db.execute(sql, (f"[ERR]{error_msg[:230]}", agent_id, trade_date))
        except Exception as e:
            logger.error(f"[{agent_id}][{trade_date}] mark_error 失败：{e}")

    def mark_min_fetch_incomplete(
        self, agent_id: str, trade_date: str, warn_msg: str
    ) -> None:
        """
        当 agent 某日的信号生成阶段有分钟线永久获取失败时，将告警写入 reserve_str_1。
        前缀 [MIN_FAIL] 与错误前缀 [ERR] 区分：
          - [MIN_FAIL]：信号已入库但数据不完整（部分候选因分钟线缺失被跳过）
          - [ERR]     ：信号生成完全失败
        后续可通过 get_min_fail_records() 查找这类记录并触发重算。
        """
        sql = f"""
            UPDATE {self.t} SET
                reserve_str_1 = %s,
                update_time   = NOW()
            WHERE agent_id = %s AND trade_date = %s
        """
        try:
            db.execute(sql, (f"[MIN_FAIL]{warn_msg[:220]}", agent_id, trade_date))
        except Exception as e:
            logger.error(f"[{agent_id}][{trade_date}] mark_min_fetch_incomplete 失败：{e}")

    def get_min_fail_records(self) -> List[Dict]:
        """
        查找所有分钟线聚合告警记录（reserve_str_1 LIKE '[MIN_FAIL]%'）。
        用于 --repair-incomplete 流程：删除这些记录，由 run_full_flow 重新处理。
        """
        sql = f"""
            SELECT agent_id, trade_date
            FROM {self.t}
            WHERE reserve_str_1 LIKE %s
            ORDER BY agent_id, trade_date
        """
        try:
            rows = db.query(sql, params=('[MIN_FAIL]%',)) or []
            return [
                {
                    "agent_id":   r["agent_id"],
                    "trade_date": r["trade_date"].strftime("%Y-%m-%d"),
                }
                for r in rows
            ]
        except Exception as e:
            logger.error(f"get_min_fail_records 失败：{e}")
            return []

    def delete_record(self, agent_id: str, trade_date: str) -> bool:
        """
        精确删除单条记录（--repair-incomplete 专用）。
        删除后，run_full_flow 的断点续跑逻辑会将该日重新纳入 dates_missing。
        """
        sql = f"DELETE FROM {self.t} WHERE agent_id = %s AND trade_date = %s"
        try:
            db.execute(sql, (agent_id, trade_date))
            return True
        except Exception as e:
            logger.error(f"[{agent_id}][{trade_date}] delete_record 失败：{e}")
            return False

    def delete_records_from(self, agent_id: str, from_date: str) -> int:
        """
        删除 agent 从 from_date 起（含）的所有记录。
        仅由手动重置流程（--reset-agent）调用，不自动触发。
        """
        sql = f"DELETE FROM {self.t} WHERE agent_id = %s AND trade_date >= %s"
        try:
            affected = db.execute(sql, (agent_id, from_date))
            logger.info(f"[{agent_id}] 重置：删除 {from_date} 起 {affected} 条记录")
            return affected
        except Exception as e:
            logger.error(f"[{agent_id}] delete_records_from 失败：{e}")
            return 0

    def get_empty_pool_unsettled_records(self) -> List[Dict]:
        """
        查找所有"信号池为空但 D+1 字段尚未显式结账"的记录。
        条件：signal_stock_detail 含 {"stock_list": []}，且 next_day_avg_close_return IS NULL，
              且 reserve_str_1 为空（非错误记录）。
        用于 --repair-zeros 修复历史上因旧逻辑导致的漏结账空池记录。
        """
        sql = f"""
            SELECT agent_id, trade_date
            FROM {self.t}
            WHERE next_day_avg_close_return IS NULL
              AND (reserve_str_1 IS NULL OR reserve_str_1 NOT LIKE %s)
              AND signal_stock_detail LIKE %s
            ORDER BY agent_id, trade_date
        """
        try:
            rows = db.query(sql, params=('[ERR]%', '%"stock_list": []%')) or []
            return [
                {
                    "agent_id":   r["agent_id"],
                    "trade_date": r["trade_date"].strftime("%Y-%m-%d"),
                }
                for r in rows
            ]
        except Exception as e:
            logger.error(f"get_empty_pool_unsettled_records 失败：{e}")
            return []

    # ------------------------------------------------------------------ #
    # 长线聚合：平仓后回写平均盈亏
    # ------------------------------------------------------------------ #

    def update_long_agg_stats(
        self, agent_id: str, trade_date: str, agg: Dict
    ) -> bool:
        """
        回写长线持仓聚合统计到 agent_daily_profit_stats。
        当 trade_date 当日所有买入信号对应的长线持仓全部平仓后调用。

        agg 必须包含：
          long_median_return, long_max_return, long_min_return,
          long_avg_trading_days, long_closed_count
        """
        sql = f"""
            UPDATE {self.t} SET
                long_median_return    = %s,
                long_max_return       = %s,
                long_min_return       = %s,
                long_avg_trading_days = %s,
                long_closed_count     = %s,
                update_time           = NOW()
            WHERE agent_id = %s AND trade_date = %s
        """
        try:
            db.execute(sql, (
                agg["long_median_return"],
                agg["long_max_return"],
                agg["long_min_return"],
                agg["long_avg_trading_days"],
                agg["long_closed_count"],
                agent_id, trade_date,
            ))
            return True
        except Exception as e:
            logger.error(
                f"[{agent_id}][{trade_date}] update_long_agg_stats 失败：{e}"
            )
            return False

    def check_date_data_exists(self, trade_date: str) -> bool:
        """检查日线数据是否已入库（kline_day.trade_date 为 YYYYMMDD 格式）"""
        date_fmt = trade_date.replace("-", "")
        sql = "SELECT COUNT(1) AS cnt FROM kline_day WHERE trade_date = %s LIMIT 1"
        try:
            row = _first(db.query(sql, params=(date_fmt,)))
            return bool(row and row["cnt"] > 0)
        except Exception as e:
            logger.error(f"check_date_data_exists 失败：{e}")
            return False
