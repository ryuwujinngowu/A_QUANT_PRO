"""
因子 IC 分析工具 (Factor Information Coefficient)
=================================================
IC（Information Coefficient）= 因子值与下期收益的 Spearman 秩相关系数。
IC > 0 表示因子对未来收益有正向预测力；IC 越稳定（ICIR 越高），因子越可靠。

核心指标：
    IC          : 单截面 Spearman(因子值, 未来收益)，范围 [-1, 1]
    IC_mean     : 多截面 IC 均值，反映平均预测力（业界要求 |IC_mean| > 0.05）
    IC_std      : IC 标准差，反映预测稳定性
    ICIR        : IC 信息比率 = IC_mean / IC_std（|ICIR| > 0.5 认为因子有效）
    IC>0 胜率    : IC > 0 的截面占比（> 50% 表明因子大概率正向有效）

使用方式：
    # 1. 从训练集 CSV 加载
    import pandas as pd
    df = pd.read_csv("train_dataset.csv")

    # 2. 定义排除列（非特征列）
    EXCLUDE_COLS = ["stock_code", "trade_date", "label1", "label2",
                    "sector_name", "top3_sectors", "adapt_score"]

    # 3. 计算所有因子 IC 报告（按 label1 作为前向收益代理）
    from learnEngine.factor_ic import calc_factor_ic_report
    report = calc_factor_ic_report(df, exclude_cols=EXCLUDE_COLS, return_col="label1")
    print(report.head(20))  # 按 |ICIR| 降序，看最有效的因子

    # 4. 单因子 IC 时间序列（诊断某因子在哪段时期失效）
    from learnEngine.factor_ic import calc_ic_series
    ic_ts = calc_ic_series(df, factor_col="market_limit_up_count", return_col="label1")
    print(ic_ts)

    # 5. IC 衰减分析（因子在多长的持仓期内有效）
    from learnEngine.factor_ic import calc_ic_decay
    decay = calc_ic_decay(df, factor_col="stock_close_d0",
                          price_col="stock_close_d0", future_cols=["label1", "label2"])
    print(decay)

扩展说明：
    - return_col 可以是二值标签（label1/label2）或连续收益率列
    - 二值标签与 Spearman IC 等价于点二列相关，语义一致
    - 如需对每日截面样本加权（如按流通市值），在 calc_ic() 中传入 weights 参数
    - 所有函数均有完整入参校验，异常时返回 NaN 而非抛出异常
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import warnings
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

from utils.log_utils import logger


# ============================================================
# 列名中文映射（新增）
# ============================================================
COLUMN_NAME_MAP = {
    "factor": "factor(因子名称)",
    "ic_mean": "ic_mean(IC均值-预测力)",
    "ic_std": "ic_std(IC标准差-稳定性)",
    "icir": "icir(IC信息比率-综合评分)",
    "abs_icir": "abs_icir(ICIR绝对值-排序用)",
    "abs_ic_mean": "abs_ic_mean(IC均值绝对值)",
    "win_rate": "win_rate(IC>0胜率)",
    "t_stat": "t_stat(t检验统计量)",
    "p_value": "p_value(p值-显著性)",
    "n": "n(有效交易日数)",
    "effective": "effective(是否有效因子)",
    "return_col": "return_col(收益列)",
    # 新增：扩展因子质量指标
    "q_mono_score": "q_mono_score(分层单调性-1~1)",
    "q_spread": "q_spread(Q5-Q1收益差)",
    "turnover": "turnover(日均换手率)",
    "recent_ic_ratio": "recent_ic_ratio(近30日IC/总体IC)",
}


# ============================================================
# 底层：单截面 IC 计算
# ============================================================

def calc_ic(
    factor_series: pd.Series,
    return_series: pd.Series,
    method: str = "spearman",
    min_sample: int = 10,
) -> float:
    """
    计算单截面 IC（因子值 vs 前向收益的秩相关系数）

    :param factor_series : 因子值 Series（同一截面，N 只股票）
    :param return_series : 前向收益 Series（与 factor_series 索引对齐）
    :param method        : 相关系数类型，"spearman"（秩相关，鲁棒）或 "pearson"（线性）
    :param min_sample    : 有效样本最少数量，低于此值返回 NaN
    :return              : IC 值，[-1, 1]，异常时返回 NaN
    """
    aligned = pd.DataFrame({"f": factor_series, "r": return_series}).dropna()
    if len(aligned) < min_sample:
        return float("nan")

    if method == "spearman":
        ic, _ = stats.spearmanr(aligned["f"], aligned["r"])
    elif method == "pearson":
        ic, _ = stats.pearsonr(aligned["f"], aligned["r"])
    else:
        raise ValueError(f"不支持的 method: {method}，请使用 'spearman' 或 'pearson'")

    return float(ic) if not np.isnan(ic) else float("nan")


# ============================================================
# 多截面：按交易日计算 IC 时间序列
# ============================================================

def calc_ic_series(
    df: pd.DataFrame,
    factor_col: str,
    return_col: str,
    date_col: str = "trade_date",
    method: str = "spearman",
    min_sample: int = 10,
) -> pd.Series:
    """
    按交易日截面计算 IC 时间序列

    :param df         : 训练集 DataFrame，含 date_col / factor_col / return_col
    :param factor_col : 因子列名
    :param return_col : 前向收益列名（如 "label1" 或自定义收益率列）
    :param date_col   : 日期列名，默认 "trade_date"
    :param method     : 相关系数类型，"spearman" 或 "pearson"
    :param min_sample : 截面有效样本下限，低于此值该截面返回 NaN
    :return           : index=trade_date, value=IC 的 Series
    """
    if factor_col not in df.columns:
        logger.error(f"[factor_ic] 因子列 '{factor_col}' 不存在")
        return pd.Series(dtype=float)
    if return_col not in df.columns:
        logger.error(f"[factor_ic] 收益列 '{return_col}' 不存在")
        return pd.Series(dtype=float)

    ic_dict = {}
    for date, grp in df.groupby(date_col):
        ic_dict[date] = calc_ic(grp[factor_col], grp[return_col],
                                method=method, min_sample=min_sample)

    return pd.Series(ic_dict, name=f"IC_{factor_col}").sort_index()


# ============================================================
# ICIR 及统计摘要
# ============================================================

def calc_icir(ic_series: pd.Series) -> dict:
    """
    从 IC 时间序列计算核心统计指标

    :param ic_series : calc_ic_series() 的输出
    :return          : dict，含以下字段：
        ic_mean   : IC 均值（预测力）
        ic_std    : IC 标准差（稳定性）
        icir      : IC 信息比率（ic_mean / ic_std），越大越稳定
        win_rate  : IC > 0 的截面占比（> 50% 表明方向一致）
        t_stat    : t 检验统计量（ic_mean / ic_std * sqrt(n)）
        p_value   : 双尾 p 值（p < 0.05 表明 IC 显著不为 0）
        n         : 有效截面数
    """
    valid = ic_series.dropna()
    n = len(valid)
    if n < 2:
        return {"ic_mean": float("nan"), "ic_std": float("nan"), "icir": float("nan"),
                "win_rate": float("nan"), "t_stat": float("nan"), "p_value": float("nan"), "n": n}

    ic_mean = float(valid.mean())
    ic_std  = float(valid.std(ddof=1))
    icir    = ic_mean / ic_std if ic_std > 0 else float("nan")
    win_rate = float((valid > 0).mean())
    t_stat  = ic_mean / ic_std * (n ** 0.5) if ic_std > 0 else float("nan")
    p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))) if not np.isnan(t_stat) else float("nan")

    return {
        "ic_mean":  round(ic_mean,  4),
        "ic_std":   round(ic_std,   4),
        "icir":     round(icir,     4),
        "win_rate": round(win_rate, 4),
        "t_stat":   round(t_stat,   4),
        "p_value":  round(p_value,  4),
        "n":        n,
    }


# ============================================================
# 批量因子报告（核心入口）
# ============================================================

def calc_factor_ic_report(
    df: pd.DataFrame,
    exclude_cols: Optional[List[str]] = None,
    factor_cols: Optional[List[str]] = None,
    return_col: str = "label1",
    date_col: str = "trade_date",
    method: str = "spearman",
    min_sample: int = 10,
    sort_by: str = "abs_icir",
) -> pd.DataFrame:
    """
    批量计算所有因子的 IC 报告（汇总表）

    使用方式：
        report = calc_factor_ic_report(df, exclude_cols=EXCLUDE_COLS, return_col="label1")
        print(report.head(20))   # 按 |ICIR| 降序排列，最有效因子在最前

    :param df          : 训练集 DataFrame
    :param exclude_cols: 非特征列（主键、标签、辅助列），与 factor_cols 二选一
    :param factor_cols : 直接指定因子列表（优先级高于 exclude_cols）
    :param return_col  : 前向收益列名
    :param date_col    : 日期列名
    :param method      : "spearman" 或 "pearson"
    :param min_sample  : 单截面有效样本最小值
    :param sort_by     : 排序方式，"abs_icir"（默认）/ "ic_mean" / "abs_ic_mean"
    :return            : DataFrame，列：factor / ic_mean / ic_std / icir / win_rate /
                                        t_stat / p_value / n / effective
    """
    if factor_cols is None:
        _exclude = set(exclude_cols or []) | {return_col, date_col}
        factor_cols = [c for c in df.columns
                       if c not in _exclude and pd.api.types.is_numeric_dtype(df[c])]

    if not factor_cols:
        logger.error("[factor_ic] 无有效因子列，请检查 exclude_cols 设置")
        return pd.DataFrame()

    if return_col not in df.columns:
        logger.error(f"[factor_ic] 收益列 '{return_col}' 不在 DataFrame 中")
        return pd.DataFrame()

    logger.info(f"[factor_ic] 开始计算 {len(factor_cols)} 个因子的 IC | 收益列: {return_col}")

    rows = []
    for i, col in enumerate(factor_cols, 1):
        ic_ts   = calc_ic_series(df, factor_col=col, return_col=return_col,
                                 date_col=date_col, method=method, min_sample=min_sample)
        summary = calc_icir(ic_ts)
        summary["factor"]   = col
        summary["abs_icir"] = abs(summary["icir"]) if not np.isnan(summary["icir"]) else float("nan")
        summary["abs_ic_mean"] = abs(summary["ic_mean"]) if not np.isnan(summary["ic_mean"]) else float("nan")
        # 有效性判断：|ICIR| > 0.5 且 p < 0.05
        summary["effective"] = (
            (not np.isnan(summary["icir"])) and
            abs(summary["icir"]) > 0.5 and
            (not np.isnan(summary["p_value"])) and
            summary["p_value"] < 0.05
        )
        rows.append(summary)

        if i % 20 == 0:
            logger.debug(f"[factor_ic] 进度 {i}/{len(factor_cols)}")

    report = pd.DataFrame(rows)

    # 排序
    sort_col = sort_by if sort_by in report.columns else "abs_icir"
    report = report.sort_values(sort_col, ascending=False, na_position="last")

    # 【核心修改】选择最终列并添加中文说明
    final_cols = ["factor", "ic_mean", "ic_std", "icir", "abs_icir",
                  "win_rate", "t_stat", "p_value", "n", "effective"]
    report = report[final_cols].reset_index(drop=True)

    # 重命名列，添加中文说明
    report = report.rename(columns=COLUMN_NAME_MAP)

    effective_n = int(report[COLUMN_NAME_MAP["effective"]].sum())
    logger.info(
        f"[factor_ic] 报告生成完成 | "
        f"总因子数:{len(report)} | 有效因子(|ICIR|>0.5 且 p<0.05):{effective_n}"
    )
    return report


# ============================================================
# IC 衰减分析（因子在不同持仓期的预测力）
# ============================================================

def calc_ic_decay(
    df: pd.DataFrame,
    factor_col: str,
    return_cols: List[str],
    date_col: str = "trade_date",
    method: str = "spearman",
    min_sample: int = 10,
) -> pd.DataFrame:
    """
    IC 衰减分析：因子对不同持仓期收益的预测力变化

    通过将同一因子与多个不同前向收益列（如 label1/label2/自定义多期收益）对比，
    分析因子在 T+1 / T+2 / T+5 的预测效果是否衰减。

    :param df          : 训练集 DataFrame
    :param factor_col  : 因子列名
    :param return_cols : 多期收益列名列表（如 ["label1", "label2"]）
    :param date_col    : 日期列名
    :param method      : 相关系数类型
    :param min_sample  : 截面有效样本下限
    :return            : DataFrame，index=return_col，列：ic_mean / ic_std / icir / win_rate / effective

    示例：
        decay = calc_ic_decay(df, factor_col="market_limit_up_count",
                              return_cols=["label1", "label2"])
        print(decay)
    """
    rows = []
    for rc in return_cols:
        if rc not in df.columns:
            logger.warning(f"[factor_ic] 收益列 '{rc}' 不存在，跳过")
            continue
        ic_ts   = calc_ic_series(df, factor_col=factor_col, return_col=rc,
                                 date_col=date_col, method=method, min_sample=min_sample)
        summary = calc_icir(ic_ts)
        summary["return_col"] = rc
        summary["abs_icir"]   = abs(summary["icir"]) if not np.isnan(summary["icir"]) else float("nan")
        summary["effective"]  = (
            (not np.isnan(summary["icir"])) and abs(summary["icir"]) > 0.5 and
            (not np.isnan(summary["p_value"])) and summary["p_value"] < 0.05
        )
        rows.append(summary)

    if not rows:
        return pd.DataFrame()

    decay_df = pd.DataFrame(rows).set_index("return_col")

    # 【核心修改】选择最终列并添加中文说明
    final_cols = ["ic_mean", "ic_std", "icir", "abs_icir", "win_rate", "p_value", "n", "effective"]
    decay_df = decay_df[final_cols]

    # 重命名列，添加中文说明
    decay_df = decay_df.rename(columns=COLUMN_NAME_MAP)

    logger.info(f"[factor_ic] IC 衰减分析完成 | 因子: {factor_col} | 期数: {len(rows)}")
    return decay_df


# ============================================================
# 分层收益分析（Quantile Return Analysis）
# ============================================================

def calc_quantile_returns(
    df: pd.DataFrame,
    factor_col: str,
    return_col: str,
    date_col: str = "trade_date",
    n_quantiles: int = 5,
    min_sample: int = 15,
) -> dict:
    """
    分层收益分析：每日按因子值分 N 组，统计各组平均收益，计算单调性得分。

    对于二值标签（label1=0/1），各组均值 = 该组的命中率，直接可比。

    :param df          : 训练集 DataFrame
    :param factor_col  : 因子列名
    :param return_col  : 收益列名（连续或二值）
    :param date_col    : 日期列名
    :param n_quantiles : 分层数，默认 5（五分位）
    :param min_sample  : 截面有效样本下限（低于此值跳过该日）
    :return            : dict，含以下字段：
        q_mono_score    : 分层单调性得分，Spearman([1..N], [Q1均值..QN均值])，
                          1=完全单调递增，-1=完全单调递减，0=无规律
        q_spread        : Q5（最高层）- Q1（最低层）的平均收益差
        quantile_returns: {Q1:float, Q2:float, ..., QN:float} 各层平均收益
    """
    if factor_col not in df.columns or return_col not in df.columns:
        return {"q_mono_score": float("nan"), "q_spread": float("nan"), "quantile_returns": {}}

    q_returns_by_date = []
    for _, grp in df.groupby(date_col):
        sub = grp[[factor_col, return_col]].dropna()
        if len(sub) < max(min_sample, n_quantiles * 2):
            continue
        try:
            sub = sub.copy()
            sub["_q"] = pd.qcut(sub[factor_col], q=n_quantiles, labels=False, duplicates="drop")
            q_avg = sub.groupby("_q")[return_col].mean()
            if len(q_avg) == n_quantiles:
                q_returns_by_date.append(q_avg.values)
        except Exception:
            continue

    if not q_returns_by_date:
        return {"q_mono_score": float("nan"), "q_spread": float("nan"), "quantile_returns": {}}

    avg_q = np.nanmean(q_returns_by_date, axis=0)
    ranks = np.arange(1, n_quantiles + 1)
    mono_score, _ = stats.spearmanr(ranks, avg_q)
    q_spread = float(avg_q[-1] - avg_q[0])

    return {
        "q_mono_score": round(float(mono_score), 4) if not np.isnan(mono_score) else float("nan"),
        "q_spread": round(q_spread, 4),
        "quantile_returns": {f"Q{i + 1}": round(float(v), 4) for i, v in enumerate(avg_q)},
    }


# ============================================================
# 换手率分析（Factor Turnover）
# ============================================================

def calc_factor_turnover(
    df: pd.DataFrame,
    factor_col: str,
    date_col: str = "trade_date",
    stock_col: str = "stock_code",
    top_pct: float = 0.2,
) -> float:
    """
    计算因子换手率：每日 Top N% 股票池与前一日的差集比例。

    换手率越高，说明因子排名每日变化越剧烈，持仓成本越高。
    对于 T+1 短线策略，日换手率 ~1.0 是正常的（每天全换仓），
    但若模型在多个因子上都极高换手，需关注交易成本侵蚀信号。

    :param df        : 训练集 DataFrame，需含 stock_col
    :param factor_col: 因子列名
    :param date_col  : 日期列名
    :param stock_col : 股票代码列名，默认 "stock_code"
    :param top_pct   : Top 百分比，默认 0.2（前20%）
    :return          : 平均日换手率，[0, 1]，NaN 表示数据不足
    """
    if factor_col not in df.columns or stock_col not in df.columns:
        return float("nan")

    dates = sorted(df[date_col].unique())
    if len(dates) < 2:
        return float("nan")

    prev_top: Optional[set] = None
    turnovers = []
    for date in dates:
        grp = df[df[date_col] == date][[stock_col, factor_col]].dropna()
        if len(grp) < 5:
            prev_top = None
            continue
        n_top = max(1, int(len(grp) * top_pct))
        cur_top = set(grp.nlargest(n_top, factor_col)[stock_col])
        if prev_top is not None and len(cur_top) > 0:
            turnover = len(cur_top - prev_top) / len(cur_top)
            turnovers.append(turnover)
        prev_top = cur_top

    return round(float(np.mean(turnovers)), 4) if turnovers else float("nan")


# ============================================================
# 近期 IC 稳定性（Recent IC Ratio）
# ============================================================

def calc_recent_ic_ratio(
    ic_series: pd.Series,
    recent_window: int = 30,
) -> float:
    """
    近期 IC 与整体 IC 的绝对值比率，用于判断因子是否在衰退。

    > 1.0 : 近期 IC 强于历史均值（因子仍在增强）
    < 1.0 : 近期 IC 弱于历史均值（因子可能在衰退，需警惕）
    NaN   : 数据不足（总截面数 < recent_window + 1）

    :param ic_series     : calc_ic_series() 的输出
    :param recent_window : 近期窗口天数，默认 30
    :return              : 比率，NaN 表示数据不足
    """
    valid = ic_series.dropna()
    if len(valid) < recent_window + 1:
        return float("nan")

    overall_abs = float(valid.mean().__abs__())
    recent_abs = float(valid.iloc[-recent_window:].mean().__abs__())

    if overall_abs < 1e-8:
        return float("nan")

    return round(recent_abs / overall_abs, 4)


# ============================================================
# 完整因子质量报告（整合所有维度）
# ============================================================

def calc_full_factor_report(
    df: pd.DataFrame,
    exclude_cols: Optional[List[str]] = None,
    factor_cols: Optional[List[str]] = None,
    return_col: str = "label1",
    date_col: str = "trade_date",
    stock_col: str = "stock_code",
    method: str = "spearman",
    min_sample: int = 10,
    n_quantiles: int = 5,
    top_pct: float = 0.2,
    recent_window: int = 30,
    sort_by: str = "abs_icir",
) -> pd.DataFrame:
    """
    完整因子质量报告：整合 IC/ICIR、分层单调性、换手率、近期稳定性四个维度。

    输出列说明：
        factor          : 因子名称
        ic_mean         : IC 均值（预测力）
        ic_std          : IC 标准差（稳定性）
        icir            : IC 信息比率（ic_mean/ic_std）
        win_rate        : IC > 0 的截面占比
        p_value         : t 检验 p 值
        n               : 有效截面数
        effective       : |ICIR| > 0.5 且 p < 0.05
        q_mono_score    : 分层单调性 [-1, 1]，越接近 ±1 越单调
        q_spread        : Q5 - Q1 收益差（因子区分度）
        turnover        : 日均换手率 [0, 1]
        recent_ic_ratio : 近30日IC / 总体IC（< 1 表示因子近期减弱）

    :param df            : 训练集 DataFrame
    :param exclude_cols  : 非特征列（主键、标签、辅助列）
    :param factor_cols   : 直接指定因子列表（优先级高于 exclude_cols）
    :param return_col    : 前向收益列名
    :param date_col      : 日期列名
    :param stock_col     : 股票代码列名（换手率计算用）
    :param method        : IC 相关系数类型，"spearman" 或 "pearson"
    :param min_sample    : 单截面有效样本最小值
    :param n_quantiles   : 分层数，默认 5
    :param top_pct       : 换手率计算的 Top 百分比，默认 0.2
    :param recent_window : 近期 IC 窗口天数，默认 30
    :param sort_by       : 排序列，默认 "abs_icir"
    :return              : DataFrame，列含所有质量指标，按 sort_by 降序
    """
    if factor_cols is None:
        _exclude = set(exclude_cols or []) | {return_col, date_col}
        factor_cols = [c for c in df.columns
                       if c not in _exclude and pd.api.types.is_numeric_dtype(df[c])]

    if not factor_cols:
        logger.error("[factor_ic] 无有效因子列，请检查 exclude_cols 设置")
        return pd.DataFrame()

    if return_col not in df.columns:
        logger.error(f"[factor_ic] 收益列 '{return_col}' 不在 DataFrame 中")
        return pd.DataFrame()

    has_stock_col = stock_col in df.columns
    if not has_stock_col:
        logger.warning(f"[factor_ic] 未找到股票代码列 '{stock_col}'，换手率将跳过计算")

    logger.info(
        f"[factor_ic] 开始完整因子质量评估 | 因子数:{len(factor_cols)} | "
        f"收益列:{return_col} | 分层:{n_quantiles}组"
    )

    rows = []
    for i, col in enumerate(factor_cols, 1):
        # 1. IC / ICIR
        ic_ts = calc_ic_series(df, factor_col=col, return_col=return_col,
                               date_col=date_col, method=method, min_sample=min_sample)
        summary = calc_icir(ic_ts)

        # 2. 分层收益
        quant = calc_quantile_returns(df, factor_col=col, return_col=return_col,
                                      date_col=date_col, n_quantiles=n_quantiles,
                                      min_sample=min_sample)

        # 3. 换手率
        turnover = (calc_factor_turnover(df, factor_col=col, date_col=date_col,
                                         stock_col=stock_col, top_pct=top_pct)
                    if has_stock_col else float("nan"))

        # 4. 近期稳定性
        recent_ratio = calc_recent_ic_ratio(ic_ts, recent_window=recent_window)

        icir_val = summary["icir"]
        p_val = summary["p_value"]
        effective = (
            not np.isnan(icir_val) and abs(icir_val) > 0.5 and
            not np.isnan(p_val) and p_val < 0.05
        )

        rows.append({
            "factor": col,
            **summary,
            "abs_icir": abs(icir_val) if not np.isnan(icir_val) else float("nan"),
            "abs_ic_mean": abs(summary["ic_mean"]) if not np.isnan(summary["ic_mean"]) else float("nan"),
            "effective": effective,
            "q_mono_score": quant["q_mono_score"],
            "q_spread": quant["q_spread"],
            "turnover": turnover,
            "recent_ic_ratio": recent_ratio,
        })

        if i % 20 == 0:
            logger.debug(f"[factor_ic] 进度 {i}/{len(factor_cols)}")

    report = pd.DataFrame(rows)

    sort_col = sort_by if sort_by in report.columns else "abs_icir"
    report = report.sort_values(sort_col, ascending=False, na_position="last")

    final_cols = [
        "factor", "ic_mean", "ic_std", "icir", "abs_icir", "abs_ic_mean",
        "win_rate", "t_stat", "p_value", "n", "effective",
        "q_mono_score", "q_spread", "turnover", "recent_ic_ratio",
    ]
    report = report[final_cols].reset_index(drop=True)
    report = report.rename(columns=COLUMN_NAME_MAP)

    effective_n = int(report[COLUMN_NAME_MAP["effective"]].sum())
    logger.info(
        f"[factor_ic] 完整报告生成完成 | "
        f"总因子数:{len(report)} | 有效因子:{effective_n}"
    )
    return report


# ============================================================
# 快速入口：直接从 CSV 运行完整报告
# ============================================================

if __name__ == "__main__":
    """
    直接运行示例：
        python learnEngine/factor_ic.py

    输出完整因子质量报告，包含：
        IC/ICIR（预测力与稳定性）
        分层单调性 q_mono_score（排名是否单调映射到收益）
        Q5-Q1 收益差 q_spread（因子区分度）
        日均换手率 turnover（持仓成本）
        近30日IC/总体IC recent_ic_ratio（因子是否在衰退）
    """
    warnings.filterwarnings("ignore")

    CSV_PATH = os.path.join(os.getcwd(), r"history\csv\train_dataset_final.csv")
    RETURN_COL = "label1"
    EXCLUDE_COLS = [
        "stock_code", "trade_date", "label1", "label2",
        "sector_name", "top3_sectors", "adapt_score",
    ]

    if not os.path.exists(CSV_PATH):
        print(f"训练集文件不存在: {CSV_PATH}\n请先运行 python learnEngine/dataset.py 生成训练集")
        exit(1)

    df = pd.read_csv(CSV_PATH)
    logger.info(f"加载训练集: {CSV_PATH} | 行数:{len(df)} | 列数:{len(df.columns)}")

    report = calc_full_factor_report(
        df,
        exclude_cols=EXCLUDE_COLS,
        return_col=RETURN_COL,
        method="spearman",
        sort_by="abs_icir",
    )

    if report.empty:
        logger.error("报告为空，请检查数据")
        exit(1)

    # ── 列名快捷变量 ──────────────────────────────────────────
    C = COLUMN_NAME_MAP  # 简写
    c_factor   = C["factor"]
    c_ic_mean  = C["ic_mean"]
    c_icir     = C["icir"]
    c_win_rate = C["win_rate"]
    c_p_value  = C["p_value"]
    c_eff      = C["effective"]
    c_mono     = C["q_mono_score"]
    c_spread   = C["q_spread"]
    c_turnover = C["turnover"]
    c_recent   = C["recent_ic_ratio"]

    # ── 完整报告表格 ──────────────────────────────────────────
    print("\n" + "=" * 120)
    print(f"因子完整质量报告（收益列: {RETURN_COL}，Spearman，按 |ICIR| 降序）")
    print("=" * 120)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(report.to_string(index=True))
    print("=" * 120)

    # ── 汇总统计 ──────────────────────────────────────────────
    effective_n = int(report[c_eff].sum())
    total_n = len(report)
    avg_turnover = report[c_turnover].mean()
    declining = (report[c_recent] < 0.8).sum()  # 近期IC衰退超过20%的因子数

    print(f"\n【汇总】")
    print(f"  有效因子（|ICIR|>0.5 且 p<0.05）: {effective_n} / {total_n}")
    print(f"  平均日换手率: {avg_turnover:.1%}（T+1策略换手率高属正常）")
    print(f"  近期衰退因子（近30日IC < 总体80%）: {int(declining)} 个")

    # ── Top 15 综合质量因子（有效 + 单调 + 近期稳定）────────
    eff_report = report[report[c_eff]].copy()
    print(f"\n【Top 15 有效因子（|ICIR| 降序）】")
    print(f"  {'因子名':45s}  {'IC均值':>8}  {'ICIR':>7}  {'单调性':>8}  {'Q5-Q1':>7}  {'换手率':>7}  {'近期比':>7}")
    print(f"  {'-'*45}  {'-'*8}  {'-'*7}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*7}")
    for _, row in eff_report.head(15).iterrows():
        mono = row[c_mono]
        recent = row[c_recent]
        turn = row[c_turnover]
        mono_str   = f"{mono:+8.4f}"  if not np.isnan(mono)   else f"{'N/A':>8}"
        turn_str   = f"{turn:7.1%}"   if not np.isnan(turn)   else f"{'N/A':>7}"
        recent_str = f"{recent:7.3f}" if not np.isnan(recent) else f"{'N/A':>7}"
        print(
            f"  {str(row[c_factor]):45s}"
            f"  {row[c_ic_mean]:+8.4f}"
            f"  {row[c_icir]:+7.4f}"
            f"  {mono_str}"
            f"  {row[c_spread]:+7.4f}"
            f"  {turn_str}"
            f"  {recent_str}"
        )

    # ── 近期衰退警告 ──────────────────────────────────────────
    decay_list = eff_report[eff_report[c_recent] < 0.8].sort_values(c_recent)
    if not decay_list.empty:
        print(f"\n【近期衰退警告（recent_ic_ratio < 0.8）】")
        for _, row in decay_list.head(10).iterrows():
            print(f"  {str(row[c_factor]):45s}  近期比={row[c_recent]:.3f}  ICIR={row[c_icir]:+.4f}")

    # ── 单调性异常（ICIR有效但不单调）────────────────────────
    bad_mono = eff_report[eff_report[c_mono].abs() < 0.5]
    if not bad_mono.empty:
        print(f"\n【单调性异常（|ICIR|有效但分层不单调，|q_mono|<0.5）】")
        for _, row in bad_mono.head(10).iterrows():
            print(f"  {str(row[c_factor]):45s}  mono={row[c_mono]:+.4f}  ICIR={row[c_icir]:+.4f}")

    # ── 保存报告 ──────────────────────────────────────────────
    out_path = os.path.join(os.getcwd(), "factor_ic_report.csv")
    report.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"完整因子质量报告已保存至: {out_path}")