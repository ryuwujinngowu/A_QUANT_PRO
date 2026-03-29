"""
个股资金面因子
==============
数据来源：moneyflow_combined 表（THS + DC 双源融合）

输出列（d0~d4，共10列）：

  mf_lg_net_rate_dX : 大单净流入占比%（THS+DC均值；单源时降级单源；均无时为0）
                       正值=大资金净买入，负值=大资金净卖出
                       注：大单净流出信号噪声较高（可能伴随散户补仓），
                           正向信号（净流入）更可信，交由树模型学习非线性关系

  mf_sm_net_rate_dX : 散户净流入占比%（THS+DC均值；单源时降级单源；均无时为0）
                       正值=散户追涨，负值=散户撤离
                       与大单形成对比：大单进+散户进=主力拉升；大单进+散户出=主力吃货

无未来函数：所有 dX 特征仅用 D0 及之前日期数据。
"""

from typing import List, Dict, Optional
import numpy as np
import pandas as pd

from features.base_feature import BaseFeature
from features.feature_registry import feature_registry
from features.data_bundle import FeatureDataBundle
from utils.common_tools import ensure_moneyflow_data
from utils.db_utils import db
from utils.log_utils import logger

_DAY_TAGS = ["d0", "d1", "d2", "d3", "d4"]

_NEUTRAL: Dict[str, float] = {
    **{f"mf_lg_net_rate_{t}": 0.0 for t in _DAY_TAGS},
    **{f"mf_sm_net_rate_{t}": 0.0 for t in _DAY_TAGS},
}


def _avg_not_none(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """两值均值；单值时返回单值；均 None 时返回 None"""
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return (a + b) / 2.0


@feature_registry.register("moneyflow")
class MoneyflowFeature(BaseFeature):
    """个股资金面因子（大单净占比 + 散户净占比，d0~d4，共10列）"""

    feature_name = "moneyflow"

    factor_columns = list(_NEUTRAL.keys())

    def calculate(self, data_bundle: FeatureDataBundle) -> tuple:
        trade_date   = data_bundle.trade_date
        dates_5d: List[str] = getattr(data_bundle, "lookback_dates_5d", [])
        target_codes: List[str] = getattr(data_bundle, "target_ts_codes", [])

        if len(dates_5d) < 5 or not target_codes:
            logger.warning(f"[moneyflow] {trade_date} 数据不足，返回空")
            return pd.DataFrame(), {}

        # ── 确保 d0~d4 每日均有资金流向数据 ──────────────────────────────
        for d in dates_5d:
            try:
                ensure_moneyflow_data(d)
            except Exception as e:
                logger.warning(f"[moneyflow] ensure_moneyflow_data({d}) 失败：{e}")

        # ── 批量查询 5 日全部目标股 ───────────────────────────────────────
        date_params = [d if "-" in d else f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in dates_5d]
        placeholders_d = ", ".join(["%s"] * len(date_params))
        placeholders_c = ", ".join(["%s"] * len(target_codes))
        sql = (
            f"SELECT ts_code, trade_date, "
            f"ths_lg_net_rate, ths_sm_net_rate, "
            f"dc_lg_net_rate,  dc_sm_net_rate "
            f"FROM moneyflow_combined "
            f"WHERE trade_date IN ({placeholders_d}) "
            f"AND ts_code IN ({placeholders_c})"
        )
        try:
            rows = db.query(sql, tuple(date_params) + tuple(target_codes)) or []
        except Exception as e:
            logger.error(f"[moneyflow] 批量查询失败：{e}")
            rows = []

        # ── 构建 {(ts_code, date_dash): row} 映射 ────────────────────────
        row_map: Dict[tuple, dict] = {}
        for r in rows:
            td = str(r.get("trade_date", ""))
            if "-" not in td:
                td = f"{td[:4]}-{td[4:6]}-{td[6:]}"
            row_map[(r["ts_code"], td)] = r

        # dates_5d 升序 [d4..d0]，reversed 后 col_dates[0]=d0
        col_dates: List[str] = list(reversed(dates_5d))

        records = []
        for code in target_codes:
            rec: dict = {"stock_code": code, "trade_date": trade_date}

            for i, d in enumerate(col_dates):
                tag = f"d{i}"
                r   = row_map.get((code, d))

                if r is None:
                    rec[f"mf_lg_net_rate_{tag}"] = 0.0
                    rec[f"mf_sm_net_rate_{tag}"] = 0.0
                    continue

                def _f(key):
                    v = r.get(key)
                    return float(v) if v is not None else None

                lg = _avg_not_none(_f("ths_lg_net_rate"), _f("dc_lg_net_rate"))
                sm = _avg_not_none(_f("ths_sm_net_rate"), _f("dc_sm_net_rate"))

                rec[f"mf_lg_net_rate_{tag}"] = round(lg if lg is not None else 0.0, 4)
                rec[f"mf_sm_net_rate_{tag}"] = round(sm if sm is not None else 0.0, 4)

            records.append(rec)

        if not records:
            return pd.DataFrame(), {}

        result_df = pd.DataFrame(records)
        logger.debug(
            f"[moneyflow] {trade_date} 完成 | 股票数:{len(records)} "
            f"| d0大单均值:{np.mean([r['mf_lg_net_rate_d0'] for r in records]):.2f}%"
        )
        return result_df, {}
