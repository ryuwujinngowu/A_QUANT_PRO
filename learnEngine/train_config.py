"""
learnEngine/train_config.py
===========================
统一配置入口 — 所有可调参数集中在此处。
train.py 和 factor_selector.py 均从此处读取，避免参数分散在多个文件。

修改参数只需改此文件，无需碰 train.py / factor_selector.py 的逻辑代码。
"""
import os

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录

# ═══════════════════════════════════════════════
# 路径
# ═══════════════════════════════════════════════
TRAIN_CSV_PATH = os.path.join(
    _BASE_DIR, "learnEngine", "datasets", "train_dataset_final.csv"
)
SELECTED_FEATURES_PATH = os.path.join(
    _BASE_DIR, "learnEngine", "search_results", "selected_features.json"
)
MODEL_DIR = os.path.join(_BASE_DIR, "model")

# ═══════════════════════════════════════════════
# 训练基础配置
# ═══════════════════════════════════════════════
MODEL_VERSION = "v5.2_auc_first"
TARGET_LABEL  = "label1"   # "label1"（次日日内涨3%）或 "label2"（隔夜高开）
VAL_RATIO     = 0.2        # 验证集比例（时序尾部切分，不随机打乱）

# 非特征列（固定排除，与 dataset.py 列结构绑定）
EXCLUDE_COLS = [
    "stock_code", "trade_date",
    "label1", "label2",
    "label_raw_return",  # 仅用于历史分析，不作训练特征
    "sector_name", "top3_sectors",
]

# ═══════════════════════════════════════════════
# Stage 1: IC 过滤阈值
# 三条件「同时」满足才剔除（宽松策略）
# 低 ICIR 因子可能在 XGBoost 组合中提供互补信息，只剔除三项都差的
# ═══════════════════════════════════════════════
IC_MIN_ICIR     = 0.15   # |ICIR| 低于此值（放宽：给 Optuna 更多候选因子）
IC_MIN_WIN_RATE = 0.48   # 胜率（IC>0 的日期占比）低于此值
IC_MAX_PVALUE   = 0.20   # t 检验 p 值高于此值（不显著）

# ═══════════════════════════════════════════════
# Stage 2: 相关性去重
# ═══════════════════════════════════════════════
CORR_MAX = 0.75   # |Pearson 相关| > 此值时视为冗余，保留 |ICIR| 更高的那个

# ═══════════════════════════════════════════════
# Stage 3: Optuna 搜索
# ═══════════════════════════════════════════════
N_TRIALS            = 400    # 搜索轮数（快速验证用 50，正式用 400）
TIMEOUT             = 7200   # 最长运行秒数
TOP_K_PCT           = 0.30   # Precision@K 的 K：取概率最高的前 30% 预测
MIN_TOP_K_POSITIVES = 10     # top-K 中正样本至少此数量，否则惩罚（防空信号过拟合）
MIN_FEATURES        = 5      # 试验中最少选入因子数，低于此值返回 0 分
N_CV_FOLDS          = 1      # 单折（CV在金融时序上因市场行情差异大效果不稳定，暂用单折）

# XGBoost 超参搜索范围（整数对用 int range，连续对用 float range）
XGB_MAX_DEPTH_RANGE        = (3, 7)
XGB_LR_RANGE               = (0.02, 0.20)
XGB_N_EST_RANGE            = (100, 1000)
XGB_SUBSAMPLE_RANGE        = (0.5, 1.0)
XGB_COLSAMPLE_RANGE        = (0.4, 1.0)
XGB_MIN_CHILD_WEIGHT_RANGE = (1, 20)
XGB_GAMMA_RANGE            = (0.0, 0.5)
XGB_REG_ALPHA_RANGE        = (0.0, 2.0)
XGB_REG_LAMBDA_RANGE       = (0.5, 3.0)
# scale_pos_weight 不再由 Optuna 搜索，改为训练时按数据自然正负比自动计算（neg/pos）
# 这样模型有足够的"开仓勇气"，同时不人为偏置
