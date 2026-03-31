"""
板块热度因子计算
因子命名规则：sector{板块ID}_d{时间跨度}_{指标名}
- 板块ID：1/2/3（每日前3活跃板块）
- 时间跨度：d4=4天前、d3=3天前、d2=2天前、d1=1天前、d0=D日
- 指标名：profit=涨幅>7%个股数（赚钱效应）、loss=跌幅>7%个股数（亏钱效应）
全局因子：
- adapt_score：板块轮动速度分，0-100，越高轮动越快
- adapt_score_norm：adapt_score / 100，归一化到 0-1
- adapt_score_slow：低轮动环境强度 = 1 - adapt_score_norm
- adapt_score_fast：高轮动环境强度 = adapt_score_norm
- adapt_score_mid：中等轮动强度（钟形），用于帮助 XGBoost 切出中间 regime
"""
import re
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
import pandas as pd
from features.base_feature import BaseFeature
from features.feature_registry import feature_registry
from utils.common_tools import (
    get_trade_dates, getStockRank_fortraining, getTagRank_daily, sort_by_recent_gain
)
from utils.log_utils import logger


# 固定配置
TIME_WEIGHT_MAP = {0: 1.0, -1: 0.8, -2: 0.6, -3: 0.4, -4: 0.2}
TOTAL_DAYS = 5
OVERLAP_WEIGHT = 0.8
HHI_WEIGHT = 0.2
MAIN_LINE_RULES = [
    {"level": "absolute", "appear": 4, "top2": 2, "coeff": 0.1},
    {"level": "strong", "appear": 3, "top2": 2, "coeff": 0.3},
    {"level": "weak", "appear": 3, "top3": 1, "coeff": 0.6},
]


@feature_registry.register("sector_heat")
class SectorHeatFeature(BaseFeature):
    """板块热度特征类，负责Top3板块筛选、轮动分计算"""
    feature_name = "sector_heat"
    # 预生成30个板块级因子列名
    factor_columns = [
        f"sector{sector_id}_d{day_offset}_{indicator}"
        for sector_id in range(1, 4)
        for day_offset in range(5)
        for indicator in ["profit", "loss"]
    ] + ["adapt_score", "adapt_score_norm", "adapt_score_slow", "adapt_score_fast", "adapt_score_mid"]

    def __init__(self):
        super().__init__()
        self.top_sector_num = 3
        self.lookback_days = 5

    def select_top3_hot_sectors(self, trade_date: str) -> dict:
        """
        板块热度计算主入口（兼容原有策略调用，完全保留原有逻辑）
        :param trade_date: D日日期，格式yyyy-mm-dd
        :return: {
            "top3_sectors": List[str] 最终选中的3个板块,
            "adapt_score": int 0-100分，板块轮动速度分
        }
        """
        # 基础格式校验
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', trade_date):
            logger.error(f"[板块热度] 日期格式错误：{trade_date}，要求yyyy-mm-dd")
            return {"top3_sectors": [], "adapt_score": 0}

        # 获取5个连续交易日
        try:
            d_date = datetime.strptime(trade_date, "%Y-%m-%d")
            start_date = (d_date - timedelta(days=20)).strftime("%Y-%m-%d")
            all_trade_dates = get_trade_dates(start_date, trade_date)
            trade_dates = all_trade_dates[-TOTAL_DAYS:]
            if len(trade_dates) != TOTAL_DAYS:
                logger.error(f"[板块热度] 获取交易日失败，仅拿到{len(trade_dates)}个，要求5个")
                return {"top3_sectors": [], "adapt_score": 0}
        except Exception as e:
            logger.error(f"[板块热度] 获取交易日异常：{str(e)}")
            return {"top3_sectors": [], "adapt_score": 0}

        # 逐天生成板块榜单
        daily_board_data = []
        all_daily_sectors = []
        daily_rank_maps = []
        for idx, day in enumerate(trade_dates):
            distance = idx - 4
            try:
                stock_df = getStockRank_fortraining(day)
                if stock_df.empty or "ts_code" not in stock_df.columns:
                    logger.warning(f"[板块热度] {day} 无符合条件的股票，跳过")
                    continue
                ts_list = stock_df["ts_code"].dropna().unique().tolist()

                tag_df = getTagRank_daily(ts_list)
                if tag_df.empty or "concept_name" not in tag_df.columns:
                    logger.warning(f"[板块热度] {day} 无板块数据，跳过")
                    continue
                tag_df = tag_df.head(5).reset_index(drop=True)

                daily_board = [{"rank": i + 1, "name": str(row["concept_name"]).strip()} for i, row in tag_df.iterrows()]
                daily_board_data.append({"distance": distance, "board": daily_board})
                all_daily_sectors.append(set([item["name"] for item in daily_board]))
                daily_rank_maps.append({item["name"]: item["rank"] for item in daily_board})
            except Exception as e:
                logger.warning(f"[板块热度] {day} 数据处理失败：{str(e)}，跳过")
                continue

        # 最低有效数据校验
        if len(daily_board_data) < 3:
            logger.error(f"[板块热度] 有效交易日不足3个，无法计算")
            return {"top3_sectors": [], "adapt_score": 0}

        # 板块热度统计
        sector_stats = {}
        for daily_data in daily_board_data:
            distance = daily_data["distance"]
            time_weight = TIME_WEIGHT_MAP[distance]
            board = daily_data["board"]
            unique_sectors = set()

            for sector in board:
                name, rank = sector["name"], sector["rank"]
                if name in unique_sectors:
                    continue
                unique_sectors.add(name)

                if name not in sector_stats:
                    sector_stats[name] = {"appear_count": 0, "has_top3": False, "total_score": 0.0}
                sector_stats[name]["appear_count"] += 1
                sector_stats[name]["has_top3"] |= (rank <= 3)
                sector_stats[name]["total_score"] += (6 - rank) * time_weight

        if not sector_stats:
            logger.error(f"[板块热度] 无有效板块统计数据")
            return {"top3_sectors": [], "adapt_score": 0}

        # 规则1：选取2个核心题材
        rule1_candidate_1 = {}
        for name, stat in sector_stats.items():
            has_top2 = any(rank <= 2 for daily_map in daily_rank_maps for n, rank in daily_map.items() if n == name)
            if stat["appear_count"] >= 3 and has_top2:
                rule1_candidate_1[name] = stat
        sorted_rule1_1 = sorted(rule1_candidate_1.items(), key=lambda x: (x[1]["total_score"], x[1]["appear_count"]), reverse=True)
        rule1_selected = [item[0] for item in sorted_rule1_1[:2]]

        # 不足2个，放宽规则
        if len(rule1_selected) < 2:
            need = 2 - len(rule1_selected)
            rule1_candidate_2 = {
                name: stat for name, stat in sector_stats.items()
                if name not in rule1_selected and stat["appear_count"] >= 3 and stat["has_top3"]
            }
            sorted_rule1_2 = sorted(rule1_candidate_2.items(), key=lambda x: (x[1]["total_score"], x[1]["appear_count"]), reverse=True)
            rule1_selected += [item[0] for item in sorted_rule1_2[:need]]

        # 兜底补全
        if len(rule1_selected) < 2:
            need = 2 - len(rule1_selected)
            rule1_candidate_3 = {
                name: stat for name, stat in sector_stats.items()
                if name not in rule1_selected and stat["has_top3"]
            }
            sorted_rule1_3 = sorted(rule1_candidate_3.items(), key=lambda x: (x[1]["total_score"], x[1]["appear_count"]), reverse=True)
            rule1_selected += [item[0] for item in sorted_rule1_3[:need]]

        # 规则2：选取1个上升趋势预期差题材
        sector_trend_data = {}
        for sector_name in sector_stats.keys():
            if sector_name in rule1_selected:
                continue
            rank_series = [daily_map.get(sector_name, 6) for daily_map in daily_rank_maps]
            n = len(rank_series)
            x = list(range(n))
            y = rank_series
            x_mean, y_mean = sum(x) / n, sum(y) / n
            cov = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
            var_x = sum((x[i] - x_mean) ** 2 for i in range(n))
            slope = cov / var_x if var_x != 0 else 0
            avg_rank = sum(y) / n
            sector_trend_data[sector_name] = {"slope": slope, "avg_rank": avg_rank}

        uptrend_candidates = {name: data for name, data in sector_trend_data.items() if data["slope"] < 0}
        sorted_uptrend = sorted(uptrend_candidates.items(), key=lambda x: x[1]["avg_rank"])
        rule2_selected = [item[0] for item in sorted_uptrend[:1]]

        # 兜底补全
        if not rule2_selected:
            remaining_candidates = {
                name: stat for name, stat in sector_stats.items()
                if name not in rule1_selected
            }
            sorted_remaining = sorted(remaining_candidates.items(), key=lambda x: (x[1]["total_score"], x[1]["appear_count"]), reverse=True)
            rule2_selected = [item[0] for item in sorted_remaining[:1]]

        # 最终Top3合并+兜底
        final_top3 = rule1_selected + rule2_selected
        if len(final_top3) < 3:
            need = 3 - len(final_top3)
            all_sorted = sorted(sector_stats.items(), key=lambda x: (x[1]["total_score"], x[1]["appear_count"]), reverse=True)
            for item in all_sorted:
                if item[0] not in final_top3 and need > 0:
                    final_top3.append(item[0])
                    need -= 1
        final_top3 = final_top3[:3]

        # 轮动分计算
        adapt_score = 0
        if len(all_daily_sectors) == 5 and len(daily_rank_maps) == 5:
            try:
                # 相邻日重合率
                overlap_rates = []
                for i in range(4):
                    set1, set2 = all_daily_sectors[i], all_daily_sectors[i + 1]
                    map1, map2 = daily_rank_maps[i], daily_rank_maps[i + 1]
                    weighted_score = 0
                    for sector in set1 & set2:
                        rank1, rank2 = map1[sector], map2[sector]
                        if rank1 <= 2 and rank2 <= 2:
                            weighted_score += 2
                        else:
                            weighted_score += 1
                    overlap_rate = weighted_score / 7
                    overlap_rates.append(overlap_rate)
                avg_overlap_rate = sum(overlap_rates) / len(overlap_rates)
                overlap_rotate_coeff = 1 - avg_overlap_rate

                # HHI集中度
                sector_appear_count = defaultdict(int)
                for daily_set in all_daily_sectors:
                    for sector in daily_set:
                        sector_appear_count[sector] += 1
                total_seats = 25
                hhi = sum((count / total_seats) ** 2 for count in sector_appear_count.values())
                hhi_rotate_coeff = (1 - hhi) / (1 - 0.04)

                # 基础轮动分
                base_rotate_coeff = overlap_rotate_coeff * OVERLAP_WEIGHT + hhi_rotate_coeff * HHI_WEIGHT
                base_score = round(base_rotate_coeff * 100)

                # 主线强度修正
                sector_detail_stats = defaultdict(lambda: {"appear": 0, "top2": 0, "top3": 0})
                for daily_map in daily_rank_maps:
                    for sector, rank in daily_map.items():
                        sector_detail_stats[sector]["appear"] += 1
                        if rank <= 2:
                            sector_detail_stats[sector]["top2"] += 1
                        if rank <= 3:
                            sector_detail_stats[sector]["top3"] += 1

                final_coeff = 1.0
                for rule in MAIN_LINE_RULES:
                    for sector, stat in sector_detail_stats.items():
                        match = True
                        if "appear" in rule and stat["appear"] < rule["appear"]:
                            match = False
                        if "top2" in rule and stat["top2"] < rule["top2"]:
                            match = False
                        if "top3" in rule and stat["top3"] < rule["top3"]:
                            match = False
                        if match:
                            final_coeff = min(final_coeff, rule["coeff"])
                            break

                adapt_score = round(base_score * final_coeff)
                adapt_score = max(0, min(100, adapt_score))
            except Exception as e:
                logger.warning(f"[板块热度] 轮动分计算失败：{str(e)}，置为0")
                adapt_score = 0

        logger.info(f"[板块热度] 计算完成 | 基准日：{trade_date} | 轮动分：{adapt_score} | 最终TOP3：{final_top3}")
        return {"top3_sectors": final_top3, "adapt_score": adapt_score}

    def calculate(self, data_bundle, **kwargs) -> tuple:
        """
        统一板块热度因子计算入口
        注意：top3_sectors 和 adapt_score 由 dataset.py 调用 select_top3_hot_sectors() 后
             透传到 FeatureDataBundle，这里直接读取，不重复计算。
        """
        trade_date   = data_bundle.trade_date
        top3_sectors = data_bundle.top3_sectors
        adapt_score  = float(data_bundle.adapt_score or 0.0)
        adapt_score  = max(0.0, min(100.0, adapt_score))
        adapt_norm   = round(adapt_score / 100.0, 4)
        adapt_slow   = round(1.0 - adapt_norm, 4)
        adapt_fast   = adapt_norm
        adapt_mid    = round(max(0.0, 1.0 - abs(adapt_norm - 0.5) / 0.5), 4)

        feature_df = pd.DataFrame([{
            "trade_date":   trade_date,
            "adapt_score":  adapt_score,
            "adapt_score_norm": adapt_norm,
            "adapt_score_slow": adapt_slow,
            "adapt_score_fast": adapt_fast,
            "adapt_score_mid": adapt_mid,
            "top3_sectors": ",".join(top3_sectors),
        }])
        factor_dict = {
            "top3_sectors": top3_sectors,
            "adapt_score":  adapt_score,
            "adapt_score_norm": adapt_norm,
            "adapt_score_slow": adapt_slow,
            "adapt_score_fast": adapt_fast,
            "adapt_score_mid": adapt_mid,
        }
        logger.info(
            f"[板块热度] {trade_date} adapt_score={adapt_score} Top3={top3_sectors}"
        )
        return feature_df, factor_dict