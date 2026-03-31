"""
Agent 收益跟踪因子
=================
输出列（d0~d4，全局因子，无 stock_code）：

  agent_high_position_intraday_5d_ratio_d{0-4}
  agent_middle_position_intraday_5d_ratio_d{0-4}
  agent_limit_down_intraday_5d_ratio_d{0-4}
  agent_hot_sector_dip_intraday_5d_ratio_d{0-4}

定义：
  某 agent 在某日的因子值 = 当日 intraday_avg_return / 近5日均值（含当日）
  > 1 代表该策略当日强于近期常态
  < 1 代表该策略当日弱于近期常态

数据来源：
  agent_daily_profit_stats.intraday_avg_return

注意：
  - 仅使用 D 日及之前数据，无未来函数
  - 若近5日均值接近 0，则返回 1.0 中性值，避免比值爆炸
"""
import pandas as pd
import numpy as np

from features.base_feature import BaseFeature
from features.feature_registry import feature_registry
from utils.db_utils import db
from utils.log_utils import logger

_AGENT_MAP = {
    "high_position_stock": "agent_high_position_intraday_5d_ratio",
    "middle_position_stock": "agent_middle_position_intraday_5d_ratio",
    "limit_down_buy": "agent_limit_down_intraday_5d_ratio",
    "hot_sector_dip_buy": "agent_hot_sector_dip_intraday_5d_ratio",
}

_NEUTRAL = {
    f"{prefix}_d{i}": 1.0
    for prefix in _AGENT_MAP.values()
    for i in range(5)
}


@feature_registry.register("agent_return_stats")
class AgentReturnStatsFeature(BaseFeature):
    """Agent 日内收益5日相对强弱因子"""

    feature_name = "agent_return_stats"
    factor_columns = list(_NEUTRAL.keys())

    def calculate(self, data_bundle) -> tuple:
        trade_date = data_bundle.trade_date
        lookback_5d = getattr(data_bundle, "lookback_dates_5d", [])
        row = {"trade_date": trade_date, **_NEUTRAL}

        if not lookback_5d:
            return pd.DataFrame([row]), {}

        date_params = tuple(lookback_5d)
        placeholders = ", ".join(["%s"] * len(date_params))
        agent_params = tuple(_AGENT_MAP.keys())
        agent_ph = ", ".join(["%s"] * len(agent_params))
        sql = f"""
            SELECT agent_id, trade_date, intraday_avg_return
            FROM agent_daily_profit_stats
            WHERE trade_date IN ({placeholders})
              AND agent_id IN ({agent_ph})
        """
        try:
            df = db.query(sql, params=date_params + agent_params, return_df=True)
        except Exception as e:
            logger.warning(f"[agent_return_stats] 查询失败：{e}")
            return pd.DataFrame([row]), {}

        if df is None or df.empty:
            return pd.DataFrame([row]), {}

        df = df.copy()
        df["trade_date"] = df["trade_date"].astype(str).str.replace("-", "", regex=False)
        df["trade_date"] = df["trade_date"].str.slice(0, 4) + "-" + df["trade_date"].str.slice(4, 6) + "-" + df["trade_date"].str.slice(6, 8)

        for agent_id, prefix in _AGENT_MAP.items():
            sub = df[df["agent_id"] == agent_id]
            ret_map = {
                r["trade_date"]: float(r.get("intraday_avg_return", 0.0) or 0.0)
                for _, r in sub.iterrows()
            }
            vals = [ret_map.get(d, 0.0) for d in lookback_5d]
            avg5 = float(np.mean(vals)) if vals else 0.0
            for di in range(5):
                idx = len(lookback_5d) - 1 - di
                d = lookback_5d[idx]
                v = ret_map.get(d, 0.0)
                row[f"{prefix}_d{di}"] = round(v / avg5, 4) if abs(avg5) > 1e-6 else 1.0

        return pd.DataFrame([row]), {}
