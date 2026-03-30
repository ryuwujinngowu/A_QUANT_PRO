"""
learnEngine/split_spec.py
==========================
全局 Frozen Split Spec 管理模块。

设计原则：
  - 由 dataset.py 生成训练集后「一次性」写出 split spec
  - factor_selector.py / train.py 只读取，不本地重算
  - Split 按日期边界冻结，而非按行数比例切分
    → 相同的 val_ratio 在不同运行之间产生完全一致的 train/val 边界

公开接口：
  write_split_spec(csv_path, val_ratio, output_path) -> dict
  load_split_spec(spec_path)                         -> dict
  apply_split_spec(df, spec)                         -> (train_df, val_df)
  split_spec_is_valid(spec_path, csv_path)           -> bool
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd


# ─── 公开接口 ────────────────────────────────────────────────────────────────

def write_split_spec(
    csv_path: str,
    val_ratio: float,
    output_path: Optional[str] = None,
) -> dict:
    """
    读取训练集 CSV，按日期边界冻结 train/val split，写出 spec JSON。

    Split 策略：
      - 对 trade_date 去重排序，取全部唯一日期的前 (1-val_ratio) 作为 train
      - val_end_date 为数据集最后一个日期
      - 边界严格按「日期」冻结，与行数无关

    :param csv_path:    训练集 CSV 路径
    :param val_ratio:   验证集比例（如 0.2 表示后 20% 的时间段）
    :param output_path: spec 输出路径（默认同 csv_path 目录下的 split_spec.json）
    :return: spec dict（已写入磁盘）
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"训练集不存在: {csv_path}")

    if output_path is None:
        output_path = os.path.join(os.path.dirname(csv_path), "split_spec.json")

    # 只读 trade_date 列，避免加载全量宽表
    dates_df = pd.read_csv(csv_path, usecols=["trade_date"], dtype=str)
    unique_dates = sorted(dates_df["trade_date"].dropna().unique().tolist())

    if len(unique_dates) < 2:
        raise ValueError(f"训练集日期数量不足（{len(unique_dates)} 个），无法切分")

    n = len(unique_dates)
    split_idx = int(n * (1.0 - val_ratio))
    split_idx = max(1, min(split_idx, n - 1))  # 保证 train/val 都非空

    train_start_date = unique_dates[0]
    train_end_date   = unique_dates[split_idx - 1]
    val_start_date   = unique_dates[split_idx]
    val_end_date     = unique_dates[-1]

    # 统计行数（按 csv 全量，不区分 strategy）
    full_df    = pd.read_csv(csv_path, usecols=["trade_date"], dtype=str)
    train_rows = int((full_df["trade_date"] <= train_end_date).sum())
    val_rows   = int((full_df["trade_date"] >= val_start_date).sum())

    spec = {
        "version":          "1.0",
        "created_at":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_csv":      os.path.abspath(csv_path),
        "val_ratio":        val_ratio,
        "n_unique_dates":   n,
        "train_start_date": train_start_date,
        "train_end_date":   train_end_date,
        "val_start_date":   val_start_date,
        "val_end_date":     val_end_date,
        "train_rows":       train_rows,
        "val_rows":         val_rows,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=2)

    return spec


def load_split_spec(spec_path: str) -> dict:
    """
    加载 split spec JSON。

    :raises FileNotFoundError: spec 文件不存在
    :raises ValueError:        JSON 格式不合法
    """
    if not os.path.exists(spec_path):
        raise FileNotFoundError(f"split spec 不存在: {spec_path}")

    with open(spec_path, encoding="utf-8") as f:
        spec = json.load(f)

    _validate_spec(spec)
    return spec


def apply_split_spec(
    df: pd.DataFrame,
    spec: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    根据 spec 将 df 切分为 (train_df, val_df)。

    切分规则：
      - train: trade_date <= spec["train_end_date"]
      - val:   trade_date >= spec["val_start_date"]

    :param df:   含 trade_date 列的 DataFrame
    :param spec: 由 load_split_spec / write_split_spec 返回的 dict
    :return:     (train_df, val_df)，均不 reset_index
    """
    _validate_spec(spec)
    dates = df["trade_date"].astype(str)
    train_df = df[dates <= spec["train_end_date"]].copy()
    val_df   = df[dates >= spec["val_start_date"]].copy()
    return train_df, val_df


def split_spec_is_valid(spec_path: str, csv_path: str) -> bool:
    """
    快速检查 spec 是否存在且与当前 CSV 匹配（仅校验 dataset_csv 路径）。

    :return: True = spec 可用；False = 需重新生成
    """
    if not os.path.exists(spec_path):
        return False
    try:
        spec = load_split_spec(spec_path)
        return os.path.abspath(spec.get("dataset_csv", "")) == os.path.abspath(csv_path)
    except Exception:
        return False


# ─── 私有辅助 ─────────────────────────────────────────────────────────────────

def _validate_spec(spec: dict) -> None:
    required = ["train_end_date", "val_start_date", "val_ratio", "dataset_csv"]
    missing  = [k for k in required if k not in spec]
    if missing:
        raise ValueError(f"split spec 缺少必要字段: {missing}")
