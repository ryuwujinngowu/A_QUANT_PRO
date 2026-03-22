"""
中长线持仓 DB 操作封装
======================
对应表：agent_long_position_stats
每行代表一个完整交易（买入 → [持仓] → 卖出），粒度为单只股票。

状态流转
--------
  status=0（持仓中）  →  平仓触发  →  status=1（已平仓）

关键设计
--------
- 幂等写入：insert_position 使用 ON DUPLICATE KEY UPDATE，断点续跑安全
- 精确平仓：close_position 仅更新 status=0 的行（防重复平仓）
- 查询聚焦：所有"未平仓"查询只返回 status=0 的行，不受历史已平仓数据影响
"""

import json
from typing import Dict, List, Optional

from utils.db_utils import db
from utils.log_utils import logger

LONG_POS_TABLE = "agent_long_position_stats"


class LongPositionDBOperator:

    def __init__(self):
        self.t = LONG_POS_TABLE

    # ------------------------------------------------------------------ #
    # 查询
    # ------------------------------------------------------------------ #

    def get_open_positions(self, agent_id: str) -> List[Dict]:
        """获取指定 agent 全部未平仓持仓（按买入日升序）"""
        sql = f"""
            SELECT * FROM {self.t}
            WHERE agent_id = %s AND status = 0
            ORDER BY buy_date
        """
        rows = db.query(sql, params=(agent_id,)) or []
        return [self._parse_row(r) for r in rows]

    def get_open_positions_before_date(self, agent_id: str, trade_date: str) -> List[Dict]:
        """获取 buy_date < trade_date 的全部未平仓持仓（用于当日卖出信号检查）"""
        sql = f"""
            SELECT * FROM {self.t}
            WHERE agent_id = %s AND status = 0 AND buy_date < %s
            ORDER BY buy_date
        """
        rows = db.query(sql, params=(agent_id, trade_date)) or []
        return [self._parse_row(r) for r in rows]

    def get_all_agents_earliest_open_buy_date(self) -> Dict[str, str]:
        """
        返回各 agent 最早的未平仓持仓买入日期 {agent_id: earliest_buy_date}。
        用于确定中长线引擎需要回溯的起始日期。
        """
        sql = f"""
            SELECT agent_id, MIN(buy_date) AS min_buy_date
            FROM {self.t}
            WHERE status = 0
            GROUP BY agent_id
        """
        rows = db.query(sql) or []
        result = {}
        for r in rows:
            if r.get("min_buy_date"):
                dt = r["min_buy_date"]
                result[r["agent_id"]] = dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)
        return result

    def get_positions_by_buy_date(self, agent_id: str, buy_date: str) -> List[Dict]:
        """获取指定 agent 在 buy_date 买入的所有持仓（含已平仓和未平仓）"""
        sql = f"""
            SELECT * FROM {self.t}
            WHERE agent_id = %s AND buy_date = %s
            ORDER BY ts_code
        """
        rows = db.query(sql, params=(agent_id, buy_date)) or []
        return [self._parse_row(r) for r in rows]

    def position_exists(self, agent_id: str, ts_code: str, buy_date: str) -> bool:
        """检查指定持仓记录是否已存在（含已平仓）"""
        sql = f"SELECT 1 FROM {self.t} WHERE agent_id = %s AND ts_code = %s AND buy_date = %s LIMIT 1"
        rows = db.query(sql, params=(agent_id, ts_code, buy_date))
        return bool(rows)

    # ------------------------------------------------------------------ #
    # 写入
    # ------------------------------------------------------------------ #

    def insert_position(self, position: Dict) -> bool:
        """
        幂等开仓（ON DUPLICATE KEY UPDATE）。
        重复插入时只更新 agent_name / stock_name / buy_price，不影响已有平仓数据。

        position 必须包含：agent_id, agent_name, ts_code, buy_date, buy_price
        可选：stock_name（默认空字符串）
        """
        sql = f"""
            INSERT INTO {self.t} (
                agent_id, agent_name, ts_code, stock_name,
                buy_date, buy_price, status, create_time, update_time
            ) VALUES (%s, %s, %s, %s, %s, %s, 0, NOW(), NOW())
            ON DUPLICATE KEY UPDATE
                agent_name  = VALUES(agent_name),
                stock_name  = VALUES(stock_name),
                buy_price   = VALUES(buy_price),
                update_time = NOW()
        """
        try:
            db.execute(sql, (
                position["agent_id"],
                position["agent_name"],
                position["ts_code"],
                position.get("stock_name", ""),
                position["buy_date"],
                position["buy_price"],
            ))
            return True
        except Exception as e:
            logger.error(f"[long_pos] insert_position 失败 | "
                         f"{position.get('agent_id')} {position.get('ts_code')} {position.get('buy_date')} | {e}")
            return False

    def close_position(
        self,
        agent_id: str,
        ts_code: str,
        buy_date: str,
        stats: Dict,
    ) -> bool:
        """
        平仓：将 status 从 0 → 1，同时写入卖出价及区间统计字段。

        stats 必须包含：
          sell_date, sell_price, period_return, trading_days,
          up_days, down_days, max_drawdown, max_floating_profit
        可选：daily_detail（List[Dict]，逐日明细）
        """
        sql = f"""
            UPDATE {self.t} SET
                status               = 1,
                sell_date            = %s,
                sell_price           = %s,
                period_return        = %s,
                trading_days         = %s,
                up_days              = %s,
                down_days            = %s,
                max_drawdown         = %s,
                max_floating_profit  = %s,
                daily_detail         = %s,
                update_time          = NOW()
            WHERE agent_id = %s AND ts_code = %s AND buy_date = %s AND status = 0
        """
        try:
            db.execute(sql, (
                stats["sell_date"],
                stats["sell_price"],
                stats["period_return"],
                stats["trading_days"],
                stats["up_days"],
                stats["down_days"],
                stats["max_drawdown"],
                stats["max_floating_profit"],
                json.dumps(stats.get("daily_detail", []), ensure_ascii=False),
                agent_id, ts_code, buy_date,
            ))
            return True
        except Exception as e:
            logger.error(f"[long_pos] close_position 失败 | {agent_id} {ts_code} {buy_date} | {e}")
            return False

    def mark_error(self, agent_id: str, ts_code: str, buy_date: str, error_msg: str) -> None:
        """将错误信息写入 reserve_str_1（不影响 status 和价格字段）"""
        sql = f"""
            UPDATE {self.t} SET
                reserve_str_1 = %s,
                update_time   = NOW()
            WHERE agent_id = %s AND ts_code = %s AND buy_date = %s
        """
        try:
            db.execute(sql, (f"[ERR]{error_msg[:230]}", agent_id, ts_code, buy_date))
        except Exception as e:
            logger.error(f"[long_pos] mark_error 失败 | {agent_id} {ts_code} {buy_date} | {e}")

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_row(r: Dict) -> Dict:
        """将数据库原始行转换为标准字典（日期格式化 / JSON 解析）"""
        r = dict(r)
        for date_field in ("buy_date", "sell_date"):
            if r.get(date_field) and hasattr(r[date_field], "strftime"):
                r[date_field] = r[date_field].strftime("%Y-%m-%d")
        if r.get("daily_detail") and isinstance(r["daily_detail"], str):
            try:
                r["daily_detail"] = json.loads(r["daily_detail"])
            except Exception:
                r["daily_detail"] = []
        return r
