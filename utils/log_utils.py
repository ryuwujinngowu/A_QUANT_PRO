import logging
import os
import time
from logging.handlers import RotatingFileHandler
from typing import Optional

from dotenv import load_dotenv


# ====================== 极简：动态获取项目根目录 ======================
def get_project_root() -> str:
    """获取项目根目录（兼容移植）"""
    current_file_path = os.path.abspath(__file__)
    utils_dir = os.path.dirname(current_file_path)
    project_root = os.path.dirname(utils_dir)
    return project_root


# 加载配置文件
PROJECT_ROOT = get_project_root()
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", ".env")
load_dotenv(CONFIG_PATH)


class LogUtils:
    """
    高性能日志工具类（单例模式）
    核心特性：
    1. 日志包含【文件名:行号】，精准定位代码位置
    2. 日志文件名带日期（如 quant_20260214.log），按天分割
    3. 控制台+文件双输出，文件自动轮转（避免过大）
    4. 极简逻辑，无多余开销，兼顾性能与定位需求
    """
    _instance = None
    _logger: Optional[logging.Logger] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_logger()
        return cls._instance

    def _get_log_file_path(self) -> str:
        """生成带日期的日志文件路径（核心修改）"""
        # 读取基础文件名（如 logs/quant.log）
        log_file_rel_path = os.getenv("LOG_FILE_PATH", "logs/quant.log")
        # 拆分路径、文件名、后缀
        log_dir_rel, log_file = os.path.split(log_file_rel_path)
        log_name, log_ext = os.path.splitext(log_file)
        # 生成日期后缀（格式：20260214）
        date_suffix = time.strftime("%Y%m%d")
        # 拼接带日期的文件名（如 quant_20260214.log）
        new_log_file = f"{log_name}_{date_suffix}{log_ext}"
        # 转换为项目根目录下的绝对路径
        log_dir_abs = os.path.join(PROJECT_ROOT, log_dir_rel)
        log_file_path = os.path.join(log_dir_abs, new_log_file)
        return log_file_path

    def _init_logger(self):
        """初始化日志器（极简逻辑+带时间文件名）"""
        # 1. 基础配置（仅保留核心参数）
        log_name = os.getenv("LOG_NAME", "a_quant")
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        # 核心：获取带日期的日志文件路径
        log_file_path = self._get_log_file_path()
        log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # 10MB
        log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))

        # ========== 核心：极简日志格式（仅保留行号+文件名） ==========
        log_format = os.getenv(
            "LOG_FORMAT",
            "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
        )
        log_date_format = os.getenv("LOG_DATE_FORMAT", "%Y-%m-%d %H:%M:%S")

        # 2. 日志等级转换（极简映射，性能最优）
        level_mapping = {"DEBUG":10, "INFO":20, "WARNING":30, "ERROR":40, "CRITICAL":50}
        log_level = level_mapping.get(log_level, 20)

        # 3. 创建日志目录（仅一次）
        log_dir = os.path.dirname(log_file_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        # 4. 初始化日志器（避免重复处理器，极简逻辑）
        self._logger = logging.getLogger(log_name)
        self._logger.setLevel(log_level)
        self._logger.handlers.clear()  # 清除重复处理器
        self._logger.propagate = False  # 防止向上传播（减少开销）

        # 5. 格式器（极简，无多余处理）
        formatter = logging.Formatter(fmt=log_format, datefmt=log_date_format)

        # 6. 控制台处理器（仅必要配置）
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)

        # 7. 文件处理器（带日期文件名+自动轮转）
        file_handler = RotatingFileHandler(
            filename=log_file_path,
            maxBytes=log_max_bytes,
            backupCount=log_backup_count,
            encoding="utf-8"  # 仅保留中文支持（必需）
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)

        # 仅打印一次初始化信息（减少日志输出）
        # self._logger.info(f"日志器初始化完成 | 带时间日志文件：{log_file_path} | 等级：{log_level}")

    @property
    def logger(self) -> logging.Logger:
        """获取日志器实例（对外暴露，极简接口）"""
        return self._logger


# 全局单例日志器（极简使用，无多余封装）
logger = LogUtils().logger

if __name__ == "__main__":
    """测试带时间文件名的日志（验证行号+文件名）"""
    print(f"项目根目录：{PROJECT_ROOT}")
    print(f"带时间日志文件路径：{LogUtils()._get_log_file_path()}")

    # 测试行号打印（核心验证）

    logger.debug("DEBUG级日志（调试信息，含行号）")  # 行号精准指向这一行
    logger.info("INFO级日志（普通信息）")
    logger.warning("WARNING级日志（警告信息）")
    logger.error("ERROR级日志（错误信息）")
    logger.critical("CRITICAL级日志（严重错误）")