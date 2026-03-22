"""
跌停板战法（LimitDownBuyAgent）
=================================
策略逻辑
--------
模拟当日首次跌停且 5 日大涨后回调的股票买入，买入价为跌停价。

"跌停板战法" 核心逻辑
---------------------
强势股在急跌时被强制卖盘打到跌停，次日往往因情绪回暖出现反弹。
本策略筛选「首板跌停 + 近 5 日已大涨 30%+」的标的，验证该规律统计效力。

命中条件：
  1. 当日 low <= 跌停价 * 1.001（日内曾触板，避免价格精度误差）
  2. 非 ST / *ST 股票
  3. 排除北交所
  4. 首板跌停：近 5 个交易日（T-5 ~ T-1）内均未出现跌停
     （从 limit_list_ths 跌停池核查，避免连跌板情绪崩溃股）
  5. 5 日涨幅过滤：(T-1 收盘) / (T-6 收盘) - 1 > 30%
     （仅追近期强势股的情绪性回调，不追弱势连跌）

买入价：跌停价（由前收盘价根据板块规则计算）

注意事项
--------
- 跌停板战法通常适用于情绪高涨期的个股性下跌，不适合系统性下跌（大盘连跌）。
- 使用日内 low 判断是否触板，不依赖 limit_list_ths（避免历史数据缺失问题）。
- open_times 不做过滤（全量跟踪，用数据说话）。
"""
from typing import List, Dict

import pandas as pd

from agent_stats.agent_base import BaseAgent
from utils.common_tools import (
    get_limit_list_ths,
    calc_limit_down_price,
    get_kline_day_range,
)
from utils.log_utils import logger

LIMIT_DOWN_TOL = 1.001    # 跌停触板容差：low <= 跌停价 * 1.001
GAIN_THRESHOLD = 0.30     # 5 日涨幅门槛（30%）
LOOKBACK_DAYS  = 5        # 首板检查/涨幅计算的回溯交易日数


class LimitDownBuyAgent(BaseAgent):
    agent_id   = "limit_down_buy"
    agent_name = "跌停板战法选手"
    agent_desc = (
        "跌停板战法：非ST/非北交所；当日 low 触及跌停价；"
        "近5个交易日内无跌停（首板）；T-6至T-1涨幅 > 30%（强势股回调）；买入价为跌停价。"
    )

    def get_signal_stock_pool(
        self,
        trade_date: str,
        daily_data: pd.DataFrame,
        context: Dict,
    ) -> List[Dict]:
        """
        返回当日符合首板跌停 + 5日大涨条件的标的列表，买入价为跌停价。
        """
        st_set = set(context.get("st_stock_list", []))
        trade_dates: List[str] = context.get("trade_dates", [])

        # ── 获取 T 日在历史交易日列表中的位置 ────────────────────────────
        if trade_date not in trade_dates:
            logger.warning(f"[{self.agent_id}][{trade_date}] trade_date 不在 trade_dates 中，无法回溯")
            return []

        idx = trade_dates.index(trade_date)
        # 需要 T-1~T-5（首板）+ T-6（涨幅基准），共需 idx >= 6
        if idx <= LOOKBACK_DAYS:
            logger.info(f"[{self.agent_id}][{trade_date}] 历史数据不足 {LOOKBACK_DAYS + 1} 日，跳过")
            return []

        # T-1 ~ T-5（用于首板检查）
        prev_5_dates: List[str] = trade_dates[idx - LOOKBACK_DAYS: idx]
        # T-6（用于 5 日涨幅计算基准）
        t6_date: str = trade_dates[idx - LOOKBACK_DAYS - 1]

        # ── 构建前收价映射 ────────────────────────────────────────────────
        pre_close_map: Dict[str, float] = {}
        pre_data = context.get("pre_close_data", pd.DataFrame())
        if not pre_data.empty and "ts_code" in pre_data.columns and "close" in pre_data.columns:
            pre_close_map = dict(zip(pre_data["ts_code"], pre_data["close"]))
        if "pre_close" in daily_data.columns:
            for _, row in daily_data.iterrows():
                if row["ts_code"] not in pre_close_map:
                    pre_close_map[row["ts_code"]] = row["pre_close"]

        # ── Step 1: 候选池 — 当日 low 触及跌停 + 非ST + 非北交所 ─────────
        candidates: List[Dict] = []
        for _, row in daily_data.iterrows():
            ts_code = row["ts_code"]
            if ts_code in st_set:
                continue
            code_prefix = ts_code.split(".")[0]
            if code_prefix.startswith(("83", "87", "88")) or ts_code.endswith(".BJ"):
                continue
            pre_close = pre_close_map.get(ts_code, 0.0)
            if pre_close <= 0:
                continue
            limit_down_price = calc_limit_down_price(ts_code, pre_close)
            if limit_down_price <= 0:
                continue
            low = float(row.get("low", 0.0) if hasattr(row, "get") else 0.0)
            if low <= 0:
                continue   # 数据缺失
            if low > limit_down_price * LIMIT_DOWN_TOL:
                continue   # 当日未触及跌停
            candidates.append({
                "ts_code":          ts_code,
                "stock_name":       str(row.get("name", "") if hasattr(row, "get") else ""),
                "buy_price":        limit_down_price,
                "pre_close":        pre_close,
            })

        if not candidates:
            logger.info(f"[{self.agent_id}][{trade_date}] 当日无触板标的，无信号")
            return []

        # ── Step 2: 首板过滤 — 近 5 日均无跌停记录 ──────────────────────
        # 从跌停池批量查出近5日出现过跌停的股票集合
        hist_limit_down_set = set()
        for d in prev_5_dates:
            try:
                ldf = get_limit_list_ths(d, limit_type="跌停池")
                if ldf is not None and not ldf.empty and "ts_code" in ldf.columns:
                    hist_limit_down_set.update(ldf["ts_code"].tolist())
            except Exception as e:
                logger.warning(f"[{self.agent_id}] 查询 {d} 跌停池失败：{e}")

        candidates = [c for c in candidates if c["ts_code"] not in hist_limit_down_set]

        if not candidates:
            logger.info(f"[{self.agent_id}][{trade_date}] 首板过滤后无剩余标的")
            return []

        # ── Step 3: 5 日涨幅过滤 — (T-1 close) / (T-6 close) - 1 > 30% ─
        cand_codes = [c["ts_code"] for c in candidates]
        t6_df = get_kline_day_range(cand_codes, t6_date, t6_date)
        t6_close_map: Dict[str, float] = {}
        if not t6_df.empty and "ts_code" in t6_df.columns and "close" in t6_df.columns:
            t6_close_map = dict(zip(t6_df["ts_code"], t6_df["close"]))

        result = []
        for c in candidates:
            ts_code  = c["ts_code"]
            t1_close = c["pre_close"]   # T 日前收 = T-1 收盘
            t6_close = t6_close_map.get(ts_code, 0.0)
            if t6_close <= 0:
                logger.debug(f"[{self.agent_id}][{trade_date}][{ts_code}] 无 T-6 收盘价，跳过")
                continue
            gain_5d = t1_close / t6_close - 1
            if gain_5d <= GAIN_THRESHOLD:
                logger.debug(
                    f"[{self.agent_id}][{trade_date}][{ts_code}] "
                    f"5日涨幅 {gain_5d:.1%} ≤ {GAIN_THRESHOLD:.0%}，跳过"
                )
                continue
            result.append(c)

        logger.info(
            f"[{self.agent_id}][{trade_date}] 跌停板命中 {len(result)} 只"
            f"（首板+5日涨幅>30%过滤后）"
        )
        return [{"ts_code": c["ts_code"], "stock_name": c["stock_name"], "buy_price": c["buy_price"]}
                for c in result]
