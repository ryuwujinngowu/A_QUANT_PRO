"""
agent_stats 模块初始化文件
核心作用：自动处理项目路径，保证服务器上任意部署路径都能正确导入
"""
import sys
import os

# 自动获取模块根目录、项目根目录的绝对路径
MODULE_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(MODULE_ROOT)  # 对应AQuant主根目录

# 将项目根目录加入系统路径，优先级最高，解决所有导入问题
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 切换工作目录到项目根目录，和本地开发环境完全一致
os.chdir(PROJECT_ROOT)