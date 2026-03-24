"""
模型信号恐慌低吸买入（ModelDipBuyAgent）
=========================================
策略逻辑（已对齐引擎：D-1日选股，D日买入）
--------
跟踪 SectorHeatStrategy 模型输出的买入信号，D 日恐慌下跌时买入：

1. D-1 日（trade_date）：调用模型完整选股流程，获取信号股列表
2. D 日（trade_date+1）09:30-10:30 监测分钟线：
   - 若任意 bar 的 low ≤ D 日开盘价 × (1 - DIP_PCT)（跌破开盘 3%）
   → 触发低吸信号，以 open × (1 - DIP_PCT) 为模拟买入价

buy_price = D 日开盘价 × (1 - DIP_PCT)（恐慌坑位价）

设计意图
--------
参考 hot_sector_dip_buy.py 的恐慌低吸逻辑。区别在于：
  - hot_sector_dip_buy：候选池来自板块 5 日涨幅排名
  - model_dip_buy：候选池来自 XGBoost 模型信号（sector_heat_strategy）
用于衡量模型信号股在次日出现恐慌回调时低吸的胜率和赔率。
"""
from typing import List, Dict

import pandas as pd

from agent_stats.agent_base import BaseAgent
from agent_stats.agents._model_signal_helper import get_model_signal_stocks
from data.data_cleaner import data_cleaner, TushareRateLimitAbort
from utils.common_tools import get_daily_kline_data, calc_limit_up_price
from utils.log_utils import logger

# ── 策略参数（与 hot_sector_dip_buy 保持一致）──────────────────────────────
DIP_PCT      = 0.03     # 触发低吸的开盘跌幅阈值（3%）
WINDOW_START = "09:30"  # 低吸监测窗口开始
WINDOW_END   = "10:30"  # 低吸监测窗口结束（含）


class ModelDipBuyAgent(BaseAgent):
    agent_id   = "model_dip_buy"
    agent_name = "模型信号恐慌低吸买入"
    agent_desc = (
        "跟踪 SectorHeatStrategy 模型信号，D-1 日生成信号，D 日 09:30-10:30 内"
        "若价格触及开盘价 -3% 则模拟低吸买入。"
        "参考 hot_sector_dip_buy 逻辑，候选池改为模型信号。"
    )

    def get_signal_stock_pool(
        self,
        trade_date: str,
        daily_data: pd.DataFrame,
        context: Dict,
    ) -> List[Dict]:
        # ── 日期格式 ─────────────────────────────────────────────────────────
        # trade_date = D-1 日（信号生成日）
        if len(trade_date) == 8 and trade_date.isdigit():
            trade_date_dash = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}"
        else:
            trade_date_dash = trade_date

        # ── 获取 D-1 日模型信号 ─────────────────────────────────────────────
        signals = get_model_signal_stocks(trade_date_dash, daily_data, caller_agent_id=self.agent_id)
        if not signals:
            return []

        # ── 获取 D 日（D-1 的下一个交易日）────────────────────────────────────
        trade_dates = context.get("trade_dates", [])
        next_date = _get_next_trade_date(trade_dates, trade_date_dash)
        if not next_date:
            logger.warning(f"[{self.agent_id}][{trade_date}] 无法获取 D 日（D-1+1）交易日，跳过")
            return []
        next_date_8 = next_date.replace("-", "")

        # ── 获取 D 日日线（取开盘价 / 前收价）────────────────────────────────
        ts_codes = [s["ts_code"] for s in signals]
        next_daily = get_daily_kline_data(next_date, ts_code_list=ts_codes)
        if next_daily.empty:
            logger.warning(f"[{self.agent_id}][{trade_date}] D 日({next_date}) 日线数据为空")
            return []

        open_map      = {}
        pre_close_map = {}
        name_map      = {s["ts_code"]: s["stock_name"] for s in signals}

        for _, row in next_daily.iterrows():
            ts = row["ts_code"]
            open_p = float(row.get("open", 0) or 0)
            if open_p <= 0:
                continue
            open_map[ts] = open_p
            pre_close_map[ts] = float(row.get("pre_close", 0) or 0)

        # ── 过滤一字板（D 日开盘即涨停封死，无法低吸）────────────────────────
        filtered_ts = []
        for ts in ts_codes:
            if ts not in open_map:
                continue
            open_p = open_map[ts]
            pre_close = pre_close_map.get(ts, 0)
            if pre_close > 0:
                limit_up = calc_limit_up_price(ts, pre_close)
                low_p = 0
                row_data = next_daily[next_daily["ts_code"] == ts]
                if not row_data.empty:
                    low_p = float(row_data.iloc[0].get("low", 0) or 0)
                if (
                    limit_up > 0
                    and abs(open_p - limit_up) < 0.015
                    and abs(low_p - limit_up) < 0.015
                ):
                    logger.debug(f"[{self.agent_id}][{trade_date}][{ts}] D 日一字板，跳过")
                    continue
            filtered_ts.append(ts)

        if not filtered_ts:
            logger.info(f"[{self.agent_id}][{trade_date}] D 日一字板过滤后为空")
            return []

        # ── 逐股检测 D 日恐慌低吸信号 ────────────────────────────────────────
        result = []
        for ts in filtered_ts:
            open_price = open_map[ts]
            dip_price  = round(open_price * (1 - DIP_PCT), 2)

            # 拉取 D 日分钟线
            try:
                min_df = data_cleaner.get_kline_min_by_stock_date(ts, next_date_8)
            except TushareRateLimitAbort:
                raise
            except Exception as e:
                logger.warning(f"[{self.agent_id}][{trade_date}][{ts}] D 日分钟线获取失败: {e}")
                self._minute_fetch_failures.append(ts)
                continue

            if min_df is None or min_df.empty:
                continue

            # 截取 09:30-10:30 窗口
            try:
                min_df = min_df.copy()
                min_df["_hm"] = pd.to_datetime(min_df["trade_time"]).dt.strftime("%H:%M")
                window = min_df[
                    (min_df["_hm"] >= WINDOW_START) & (min_df["_hm"] <= WINDOW_END)
                ]
            except Exception:
                continue

            if window.empty:
                continue

            window_low = float(window["low"].min())

            if window_low <= dip_price:
                logger.info(
                    f"[{self.agent_id}][{trade_date}][{ts}] {name_map.get(ts, '')} "
                    f"触发低吸: D日 open={open_price:.2f} "
                    f"dip_price={dip_price:.2f} window_low={window_low:.2f}"
                )
                result.append({
                    "ts_code":    ts,
                    "stock_name": name_map.get(ts, ""),
                    "buy_price":  dip_price,
                })
            else:
                logger.debug(
                    f"[{self.agent_id}][{trade_date}][{ts}] "
                    f"未触发: D日 open={open_price:.2f} dip_target={dip_price:.2f} "
                    f"window_low={window_low:.2f}"
                )

        logger.info(
            f"[{self.agent_id}][{trade_date}] D日({next_date}) 恐慌低吸 {len(result)} 只 "
            f"（信号={len(signals)} 只，候选={len(filtered_ts)} 只）: "
            + " | ".join(f"{s['ts_code']}(dip={s['buy_price']:.2f})" for s in result)
        )
        return result


def _get_next_trade_date(trade_dates: List[str], trade_date: str) -> str:
    """从交易日列表中找到 trade_date 的下一个交易日"""
    try:
        idx = trade_dates.index(trade_date)
        if idx + 1 < len(trade_dates):
            return trade_dates[idx + 1]
    except ValueError:
        pass
    return ""
