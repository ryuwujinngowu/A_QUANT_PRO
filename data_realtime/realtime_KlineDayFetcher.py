# data/data_api.py
import requests
import  os
import pandas as pd
from typing import Optional, Union
from datetime import datetime
from utils.log_utils import logger
import tushare as ts


API_REQUEST_INTERVAL = 1  # Tushare接口限流间隔（秒），统一管理
TS_TOKEN_DEFAULT =  os.getenv("TS_TOKEN_DEFAULT")
TUSHARE_API_URL = "http://tushare.xyz"  # Tushare接口地址，统一配置

def initialize_tushare():
    """初始化Tushare Pro API，读取环境变量Token，失败则抛出异常"""
    # 获取Token（优先环境变量，兜底默认值）
    TS_TOKEN = os.getenv("TS_TOKEN", TS_TOKEN_DEFAULT)
    try:
        ts.set_token(TS_TOKEN)
        pro = ts.pro_api()
        pro._DataApi__http_url = TUSHARE_API_URL  # 自定义接口地址
        logger.info("Tushare Pro API 初始化成功")
        return pro
    except Exception as e:
        logger.critical(f"Tushare Pro API 初始化失败：{str(e)}")
        raise RuntimeError(f"Tushare初始化失败：{str(e)}")


class RealTimeDataHandler:
    """极简实时数据处理器：仅做接口请求、涨幅计算、筛选、内存存储"""

    def __init__(self):
        self.realtime_df = None  # 内存存储全市场实时日线数据（核心：直接读取）
        self.pro = initialize_tushare()

    def get_realtime_data_and_filter(self):
        """
        核心方法：
        1. 调用rt_k获取全市场实时日线（存内存）
        2. 计算涨幅：(现价-昨收)/昨收 * 100
        3. 筛选涨幅≥6%的个股，打印ts_code
        """
        # ==================== 1. 极速请求全市场数据（按你的接口示例） ====================
        # 直接请求全市场（按你给的通配符组合，无分批/缓存冗余，满足速度要求）
        self.realtime_df = self.pro.rt_k(ts_code='3*.SZ,6*.SH,0*.SZ')  # 数据直接存内存属性

        # ==================== 2. 立即计算涨幅（无冗余，速度优先） ====================
        # 避免除零错误（极简处理：pre_close为0时涨幅设为0）
        self.realtime_df['pct_chg'] = (self.realtime_df['close'] - self.realtime_df['pre_close']) / \
                                      self.realtime_df['pre_close'].replace(0, 1) * 100

        # ==================== 3. 筛选涨幅≥6%的个股，打印ts_code ====================
        filter_df = self.realtime_df[self.realtime_df['pct_chg'] >= 6.0]
        # 打印符合条件的ts_code（按要求仅输出这个字段）
        print("涨幅≥6%的个股代码：")
        for ts_code in filter_df['ts_code'].tolist():
            print(ts_code)

        # 可选：返回筛选结果（方便后续计算调用，直接读内存）
        return filter_df


# ==================== 调用示例（极速执行，全程内存操作） ====================
if __name__ == "__main__":
    handler = RealTimeDataHandler()
    # 执行核心逻辑：请求数据→计算涨幅→筛选→打印
    filter_result = handler.get_realtime_data_and_filter()

    # 后续计算直接读取内存中的数据（速度无损耗）
    print(f"\n内存中全市场数据量：{len(handler.realtime_df)}")
    print(f"内存中涨幅≥6%的个股数：{len(filter_result)}")