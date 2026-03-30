"""
实时行情数据获取层 (data_realtime/realtime_fetcher.py)
=======================================================
设计原则：
  1. 与 data/ 层完全解耦 — 不依赖 DB、不入库、不导入 data_cleaner
  2. 速度优先 — 最小化延迟，直接 API 请求，懒初始化连接
  3. Schema 兼容 — 返回与 kline_day / kline_min 表完全兼容的 DataFrame，
     策略可直接把 fetch_kline_day() 的结果作为 daily_df 传入

用法示例：
  from data_realtime.realtime_fetcher import RealtimeFetcher

  fetcher = RealtimeFetcher()

  # 全市场实时日线（kline_day 兼容）
  daily_df = fetcher.fetch_kline_day()

  # 指定股票 5 分钟线（kline_min 兼容）
  min_df = fetcher.fetch_kline_min(["600000.SH", "000001.SZ"], freq="5MIN")
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
from datetime import datetime
from typing import List, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import tushare as ts
import tushare.pro.client as _ts_client
from dotenv import load_dotenv

from utils.log_utils import logger

# 加载 config/.env（realtime 层不依赖 db_utils，需手动加载 Token）
_ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", ".env")
load_dotenv(_ENV_PATH)

# tushare.xyz 是自定义代理服务器，应直连，不需要再走 VPN 代理。
# MangoVPN 会把 HTTP_PROXY=socks5://127.0.0.1:51081 注入环境变量，
# 叠两层代理会导致 rt_k 等接口超时。将 tushare.xyz 加入 NO_PROXY 强制直连。
_no_proxy = os.environ.get("NO_PROXY", "")
if "tushare.xyz" not in _no_proxy:
    os.environ["NO_PROXY"] = f"tushare.xyz,{_no_proxy}".strip(",")
    os.environ["no_proxy"] = os.environ["NO_PROXY"]

# ─────────────────────────────────────────────────────────────────────────────
# 连接配置（与 data/data_fetcher.py 保持一致）
# ─────────────────────────────────────────────────────────────────────────────
_TS_TOKEN_DEFAULT = os.getenv("TS_TOKEN_DEFAULT")
_TUSHARE_API_URL = "http://tushare.xyz"

# rt_min 单次最大 1000 行；保守估算各 freq 每日最多 bar 数，确保不超限
# 14:55 时各频率 bar 数约：1MIN≈225, 5MIN≈46, 15MIN≈16, 30MIN≈9, 60MIN≈5
_MIN_BATCH_SIZES = {
    "1MIN": 4,
    "5MIN": 20,
    "15MIN": 60,
    "30MIN": 100,
    "60MIN": 100,
}

# rt_k 全市场通配符（主板+创业板+科创板；BJ 单独追加以便可选）
_FULL_MARKET_PATTERN = "3*.SZ,6*.SH,0*.SZ"
_FULL_MARKET_WITH_BJ = "3*.SZ,6*.SH,0*.SZ,9*.BJ"


# ─────────────────────────────────────────────────────────────────────────────
# 内部工具
# ─────────────────────────────────────────────────────────────────────────────
def _init_pro():
    """初始化 Tushare Pro API（设置自定义 URL）"""
    token = os.getenv("TS_TOKEN", _TS_TOKEN_DEFAULT)
    ts.set_token(token)
    pro = ts.pro_api()
    pro._DataApi__http_url = _TUSHARE_API_URL
    _ts_client.DataApi._DataApi__http_url = _TUSHARE_API_URL
    return pro


def _today_str() -> str:
    """当前日期，YYYY-MM-DD 格式（与 kline_day.trade_date 格式一致）"""
    return datetime.now().strftime("%Y-%m-%d")


def _chunks(lst: list, n: int):
    """将列表按 n 切分成子列表"""
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


# ─────────────────────────────────────────────────────────────────────────────
# 核心类
# ─────────────────────────────────────────────────────────────────────────────
class RealtimeFetcher:
    """
    实时行情数据获取器

    特性：
      - 懒初始化：首次请求时才建立 Tushare 连接
      - 无 DB 依赖：纯内存操作
      - Schema 兼容：输出可直接用作策略的 daily_df / kline_min_df
    """

    def __init__(self):
        self._pro = None  # 懒初始化

    @property
    def pro(self):
        if self._pro is None:
            self._pro = _init_pro()
            logger.info("[RealtimeFetcher] Tushare 连接初始化成功")
        return self._pro

    # ─────────────────────────────── 实时日线 ────────────────────────────────

    def fetch_kline_day(
        self,
        ts_code_pattern: Optional[str] = None,
        include_bj: bool = False,
    ) -> pd.DataFrame:
        """
        获取全市场（或指定模式）实时日线，返回 kline_day 表兼容 DataFrame。

        Args:
            ts_code_pattern: Tushare rt_k 通配符，None 时使用全市场默认值
            include_bj: 是否包含北交所（默认 False）

        Returns:
            DataFrame，列名与 kline_day 表一致：
            ts_code, trade_date, open, high, low, close, pre_close,
            change1, pct_chg, volume, amount(千元)
            其余 DB 字段（turnover_rate/swing/limit_up/limit_down）填 0
        """
        if ts_code_pattern is None:
            ts_code_pattern = _FULL_MARKET_WITH_BJ if include_bj else _FULL_MARKET_PATTERN

        logger.info(f"[RealtimeFetcher] 开始拉取实时日线，pattern={ts_code_pattern}")
        t0 = time.time()

        # rt_k 限流：最多50次/分钟；遇到限流时最多重试 3 次，间隔 65s
        raw = None
        for attempt in range(3):
            try:
                raw = self.pro.rt_k(ts_code=ts_code_pattern)
                break
            except Exception as e:
                err = str(e)
                if "每分钟最多访问" in err or "rate" in err.lower():
                    wait = 65
                    logger.warning(f"[RealtimeFetcher] rt_k 限流，{wait}s 后重试 (attempt {attempt+1}/3)")
                    time.sleep(wait)
                else:
                    logger.error(f"[RealtimeFetcher] rt_k 调用失败: {e}")
                    return pd.DataFrame()

        if raw is None or raw.empty:
            logger.warning("[RealtimeFetcher] rt_k 返回空数据（非交易时段或无权限）")
            return pd.DataFrame()

        df = self._normalize_kline_day(raw)
        logger.info(
            f"[RealtimeFetcher] 实时日线获取完成 {len(df)} 只，耗时 {time.time()-t0:.2f}s"
        )
        return df

    def fetch_kline_day_codes(self, ts_codes: List[str]) -> pd.DataFrame:
        """
        获取指定股票列表的实时日线（用于持仓跟踪等小批量场景）

        Args:
            ts_codes: 股票代码列表，如 ["600000.SH", "000001.SZ"]

        Returns:
            同 fetch_kline_day()，只含请求的股票
        """
        if not ts_codes:
            return pd.DataFrame()

        pattern = ",".join(ts_codes)
        return self.fetch_kline_day(ts_code_pattern=pattern)

    # ─────────────────────────────── 实时分钟线 ──────────────────────────────

    def fetch_kline_min(
        self,
        ts_codes: List[str],
        freq: str = "5MIN",
    ) -> pd.DataFrame:
        """
        获取指定股票实时分钟线，自动分批（避免超 1000 行限制）。

        Args:
            ts_codes: 股票代码列表
            freq: 1MIN / 5MIN / 15MIN / 30MIN / 60MIN（大写）

        Returns:
            DataFrame，列名与 kline_min 表兼容：
            ts_code, trade_time, trade_date, open, close, high, low, volume, amount
        """
        if not ts_codes:
            return pd.DataFrame()

        freq = freq.upper()
        if freq not in _MIN_BATCH_SIZES:
            logger.error(f"[RealtimeFetcher] 不支持的 freq={freq}，有效值: {list(_MIN_BATCH_SIZES)}")
            return pd.DataFrame()

        batch_size = _MIN_BATCH_SIZES[freq]
        all_parts: List[pd.DataFrame] = []

        logger.info(
            f"[RealtimeFetcher] 开始拉取实时分钟线 freq={freq}，"
            f"共 {len(ts_codes)} 只，批大小={batch_size}"
        )
        t0 = time.time()

        for batch in _chunks(ts_codes, batch_size):
            batch_str = ",".join(batch)
            try:
                raw = self.pro.rt_min(ts_code=batch_str, freq=freq)
                if raw is not None and not raw.empty:
                    all_parts.append(raw)
            except Exception as e:
                logger.warning(f"[RealtimeFetcher] rt_min 批次失败 {batch}: {e}")
            # 轻微限流间隔（实时接口比历史接口宽松，但避免被封）
            time.sleep(0.1)

        if not all_parts:
            logger.warning("[RealtimeFetcher] rt_min 全部批次均返回空")
            return pd.DataFrame()

        combined = pd.concat(all_parts, ignore_index=True)
        df = self._normalize_kline_min(combined)
        logger.info(
            f"[RealtimeFetcher] 实时分钟线获取完成 {len(df)} 行，耗时 {time.time()-t0:.2f}s"
        )
        return df

    # ──────────────────── 实时分钟线（全量当日 bars，rt_min_daily）────────────

    def fetch_kline_min_daily_batch(
        self,
        ts_codes: List[str],
        freq: str = "5MIN",
        max_workers: int = 5,
    ) -> Dict[str, pd.DataFrame]:
        """
        批量获取指定股票当日开盘以来所有分钟 bar（rt_min_daily 接口）。
        rt_min_daily 每次只支持单只股票，自动并发请求。

        Args:
            ts_codes   : 股票代码列表（候选股）
            freq       : 1MIN / 5MIN / 15MIN / 30MIN / 60MIN
            max_workers: 并发线程数（建议 3-5，避免触发限流）

        Returns:
            {ts_code: DataFrame}，DataFrame 列名与 kline_min 表兼容：
            ts_code, trade_time, trade_date, open, close, high, low, volume, amount
        """
        if not ts_codes:
            return {}

        freq = freq.upper()
        logger.info(
            f"[RealtimeFetcher] 开始拉取 rt_min_daily freq={freq}，"
            f"共 {len(ts_codes)} 只，并发={max_workers}"
        )
        t0 = time.time()
        result: Dict[str, pd.DataFrame] = {}

        def _fetch_one(ts_code: str):
            try:
                raw = self.pro.rt_min_daily(ts_code=ts_code, freq=freq)
                if raw is None or raw.empty:
                    return ts_code, pd.DataFrame()
                df = self._normalize_kline_min_daily(raw, ts_code)
                return ts_code, df
            except Exception as e:
                logger.warning(f"[RealtimeFetcher] rt_min_daily {ts_code} 失败: {e}")
                return ts_code, pd.DataFrame()

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_fetch_one, code): code for code in ts_codes}
            for fut in as_completed(futures):
                ts_code, df = fut.result()
                if not df.empty:
                    result[ts_code] = df

        success = len(result)
        logger.info(
            f"[RealtimeFetcher] rt_min_daily 完成 {success}/{len(ts_codes)} 只，"
            f"耗时 {time.time()-t0:.2f}s"
        )
        return result

    # ─────────────────────────────── 格式化方法 ──────────────────────────────

    @staticmethod
    def _normalize_kline_day(raw: pd.DataFrame) -> pd.DataFrame:
        """
        将 rt_k 原始返回转换为 kline_day 兼容格式。

        rt_k 字段：ts_code, name, pre_close, high, open, low, close,
                   vol(股), amount(元), num, trade_time, ...

        kline_day 字段：ts_code, trade_date, open, high, low, close,
                        pre_close, change1, pct_chg, volume(股), amount(千元), ...
        """
        df = raw.copy()

        # 1. 重命名
        df.rename(columns={"vol": "volume"}, inplace=True)

        # 2. 计算衍生字段
        pre = pd.to_numeric(df["pre_close"], errors="coerce").fillna(0)
        close = pd.to_numeric(df["close"], errors="coerce").fillna(0)
        df["change1"] = (close - pre).round(4)
        df["pct_chg"] = ((close - pre) / pre.replace(0, float("nan")) * 100).round(4)

        # 3. amount: rt_k 单位为元 → kline_day 单位为千元（与历史接口一致）
        if "amount" in df.columns:
            df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0) / 1000.0

        # 4. trade_date：取当日日期（rt_k 无独立日期字段）
        df["trade_date"] = _today_str()

        # 5. 数值字段类型
        for col in ["open", "high", "low", "close", "pre_close"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float)
        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")

        # 6. 填充 kline_day 其他字段（策略兼容，设为 0）
        for col in ["turnover_rate", "swing", "limit_up", "limit_down"]:
            df[col] = 0.0

        # 7. 保留 kline_day 核心列（保持列顺序）
        keep = [
            "ts_code", "trade_date", "open", "high", "low", "close",
            "pre_close", "change1", "pct_chg", "volume", "amount",
            "turnover_rate", "swing", "limit_up", "limit_down",
        ]
        existing = [c for c in keep if c in df.columns]
        return df[existing].reset_index(drop=True)

    @staticmethod
    def _normalize_kline_min(raw: pd.DataFrame) -> pd.DataFrame:
        """
        将 rt_min 原始返回转换为 kline_min 兼容格式。

        rt_min 字段：ts_code, time, open, close, high, low, vol, amount

        kline_min 字段：ts_code, trade_time, trade_date, open, close,
                        high, low, volume, amount
        """
        df = raw.copy()

        # 1. 重命名
        rename_map = {}
        if "vol" in df.columns:
            rename_map["vol"] = "volume"
        if "time" in df.columns:
            rename_map["time"] = "trade_time"
        if rename_map:
            df.rename(columns=rename_map, inplace=True)

        # 2. trade_time → datetime；提取 trade_date
        if "trade_time" in df.columns:
            df["trade_time"] = pd.to_datetime(df["trade_time"], errors="coerce")
            df["trade_date"] = df["trade_time"].dt.strftime("%Y-%m-%d")
            df.dropna(subset=["trade_time"], inplace=True)

        # 3. 数值字段
        for col in ["open", "close", "high", "low", "amount"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float)
        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")

        # 4. 去重
        if "trade_time" in df.columns:
            df.drop_duplicates(subset=["ts_code", "trade_time"], keep="last", inplace=True)

        keep = ["ts_code", "trade_time", "trade_date", "open", "close", "high", "low", "volume", "amount"]
        existing = [c for c in keep if c in df.columns]
        return df[existing].sort_values(["ts_code", "trade_time"]).reset_index(drop=True)

    @staticmethod
    def _normalize_kline_min_daily(raw: pd.DataFrame, ts_code: str) -> pd.DataFrame:
        """
        将 rt_min_daily 原始返回转换为 kline_min 兼容格式。

        rt_min_daily 字段：code, freq, time, open, close, high, low, vol, amount
        注意：字段名为 code（非 ts_code），且按单只股票返回

        kline_min 字段：ts_code, trade_time, trade_date, open, close, high, low, volume, amount
        """
        df = raw.copy()

        # 1. 重命名（code→ts_code / vol→volume / time→trade_time）
        rename_map = {}
        if "code" in df.columns:
            rename_map["code"] = "ts_code"
        elif "ts_code" not in df.columns:
            df["ts_code"] = ts_code          # 兜底：直接写入
        if "vol" in df.columns:
            rename_map["vol"] = "volume"
        if "time" in df.columns:
            rename_map["time"] = "trade_time"
        if rename_map:
            df.rename(columns=rename_map, inplace=True)

        # 若 ts_code 列值为 code 格式（无交易所后缀），直接用传入的 ts_code 覆盖
        if "ts_code" in df.columns and not str(df["ts_code"].iloc[0]).endswith((".SH", ".SZ", ".BJ")):
            df["ts_code"] = ts_code

        # 2. trade_time → datetime；提取 trade_date
        if "trade_time" in df.columns:
            df["trade_time"] = pd.to_datetime(df["trade_time"], errors="coerce")
            df["trade_date"] = df["trade_time"].dt.strftime("%Y-%m-%d")
            df.dropna(subset=["trade_time"], inplace=True)

        # 3. 数值字段
        for col in ["open", "close", "high", "low", "amount"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float)
        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")

        # 4. 去重 + 排序
        if "trade_time" in df.columns:
            df.drop_duplicates(subset=["ts_code", "trade_time"], keep="last", inplace=True)

        keep = ["ts_code", "trade_time", "trade_date", "open", "close", "high", "low", "volume", "amount"]
        existing = [c for c in keep if c in df.columns]
        return df[existing].sort_values("trade_time").reset_index(drop=True)
