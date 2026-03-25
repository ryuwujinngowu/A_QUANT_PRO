"""
utils/xgb_compat.py
===================
XGBoost 跨版本兼容工具。

问题背景：
  模型用新版 XGBoost（2.x）训练，运行环境是旧版（1.x，Python 3.8）。
  旧版 sklearn wrapper 的 get_params() 会 getattr(model, param) 遍历 __init__ 签名，
  但新版模型对象没有 use_label_encoder / gpu_id 等已废弃属性 → AttributeError。

解决方案：
  直接调用底层 Booster.predict()，完全绕开 sklearn wrapper 的 get_params() 链。
"""
import numpy as np
import xgboost as xgb


def safe_predict_proba(model, X) -> np.ndarray:
    """
    版本安全的 predict_proba，返回 shape (n, 2) 的概率矩阵（列0=负类，列1=正类）。

    直接使用 model.get_booster().predict()，跳过 sklearn wrapper，
    避免新版模型在旧版运行时出现 AttributeError（use_label_encoder / gpu_id 等）。

    用法（替换 model.predict_proba(X)[:, 1]）：
        from utils.xgb_compat import safe_predict_proba
        probs = safe_predict_proba(model, X)[:, 1]
    """
    booster = model.get_booster()
    dmatrix = xgb.DMatrix(X)
    raw = booster.predict(dmatrix)           # shape (n,)，已是正类概率（binary:logistic）
    return np.column_stack([1 - raw, raw])   # shape (n, 2)
