"""
同花顺热股榜因子
================
输出列（个股级，含 stock_code）：

  ths_hot_score_d0 : D0（当日）在热股榜中的排名得分
  ths_hot_score_d1 : D-1
  ths_hot_score_d2 : D-2
  ths_hot_score_d3 : D-3
  ths_hot_score_d4 : D-4

得分规则（只关注前10名）：
  排名 1 → 10，排名 2 → 9，…，排名 10 → 1，未入前10 → 0
  归一化到 [0, 1]：score / 10

数据来源：ths_hot 表（DB→API→DB 链路，ensure_ths_hot_data 保障新鲜度）
无未来函数：每日数据对应当日收盘后（is_new=Y 22:30/is_new=N 盘后快照），
           D0~D4 均为历史数据，不含任何 T+1 信息。
"""

from typing import List, Dict
import pandas as pd

from features.base_feature import BaseFeature
from features.feature_registry import feature_registry
from utils.common_tools import ensure_ths_hot_data
from utils.db_utils import db
from utils.log_utils import logger

_TOP_N = 10          # 只关注前 N 名
_MAX_SCORE = 10.0    # 排名 1 的原始得分


def _rank_to_score(rank: int) -> float:
    """排名转归一化得分：rank 1→1.0，rank 10→0.1，未入榜→0.0"""
    if rank is None or rank > _TOP_N:
        return 0.0
    return round((_MAX_SCORE - rank + 1) / _MAX_SCORE, 1)


@feature_registry.register("ths_hot")
class THSHotFeature(BaseFeature):
    """同花顺热股榜因子（d0~d4 排名归一化得分，个股级）"""

    feature_name = "ths_hot"

    factor_columns = [f"ths_hot_score_d{i}" for i in range(5)]

    def calculate(self, data_bundle) -> tuple:
        trade_date  = data_bundle.trade_date
        dates_5d: List[str] = getattr(data_bundle, "lookback_dates_5d", [])

        if len(dates_5d) < 5:
            logger.warning(f"[ths_hot] {trade_date} lookback_dates_5d 不足5天，返回空")
            return pd.DataFrame(), {}

        # d0=dates_5d[-1]（最新）… d4=dates_5d[0]（最旧）
        target_codes: List[str] = getattr(data_bundle, "target_ts_codes", [])

        # ── 确保 D0~D4 每日均有热榜数据 ──────────────────────────────────────
        for d in dates_5d:
            try:
                ensure_ths_hot_data(d)
            except Exception as e:
                logger.warning(f"[ths_hot] ensure_ths_hot_data({d}) 失败：{e}")

        # ── 一次性批量查询 5 日前10名热股 ────────────────────────────────────
        date_strs = [d.replace("-", "") for d in dates_5d]
        # DB trade_date 存 YYYY-MM-DD，查询用 dash 格式
        date_params = [f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in date_strs]
        placeholders = ", ".join(["%s"] * len(date_params))
        sql = (
            f"SELECT ts_code, trade_date, `rank` FROM ths_hot "
            f"WHERE trade_date IN ({placeholders}) AND `rank` <= {_TOP_N} "
            f"ORDER BY trade_date, `rank`"
        )
        try:
            rows = db.query(sql, tuple(date_params)) or []
        except Exception as e:
            logger.error(f"[ths_hot] 查询热榜失败：{e}")
            rows = []

        # ── 构建 {(ts_code, trade_date_dash): rank} 映射 ────────────────────
        rank_map: Dict[tuple, int] = {}
        for r in rows:
            td = str(r.get("trade_date", "")).replace("-", "")
            td_dash = f"{td[:4]}-{td[4:6]}-{td[6:]}" if len(td) == 8 else str(r["trade_date"])
            rank_map[(r["ts_code"], td_dash)] = int(r["rank"])

        # ── 构建结果 DataFrame ───────────────────────────────────────────────
        # dates_5d 升序：[d4, d3, d2, d1, d0]，对应列名 d4→d0
        col_dates = list(reversed(dates_5d))  # [d0, d1, d2, d3, d4]

        records = []
        for code in target_codes:
            row: dict = {"stock_code": code, "trade_date": trade_date}
            for i, d in enumerate(col_dates):
                score = _rank_to_score(rank_map.get((code, d)))
                row[f"ths_hot_score_d{i}"] = score
            records.append(row)

        if not records:
            return pd.DataFrame(), {}

        result_df = pd.DataFrame(records)
        logger.debug(
            f"[ths_hot] {trade_date} 计算完成 | 股票数:{len(records)} "
            f"| 上榜(d0): {sum(1 for r in records if r['ths_hot_score_d0'] > 0)} 只"
        )
        return result_df, {}
