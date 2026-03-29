import json
import os

import requests
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# 导入项目工具模块
from utils.log_utils import logger


# ====================== 加载配置 ======================
def load_config():
    """加载配置文件"""
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", ".env")
    if not os.path.exists(env_path):
        raise FileNotFoundError(f"配置文件不存在：{env_path}")
    load_dotenv(dotenv_path=env_path)

    # 读取配置并验证
    config = {
        "api_key": os.getenv("DOUBAO_API_KEY"),
        "api_url": os.getenv("DOUBAO_API_URL"),
        "model": os.getenv("DOUBAO_MODEL", "doubao-1-5-pro-32k-250115"),
        "retry_times": int(os.getenv("DOUBAO_RETRY_TIMES", 3)),
        "retry_delay": int(os.getenv("DOUBAO_RETRY_DELAY", 1)),
        "token_stat": os.getenv("DOUBAO_TOKEN_STAT", "True").lower() == "true",
        "system_prompt": os.getenv("DOUBAO_SYSTEM_PROMPT", "你是人工智能助手."),
        # Token统计文件路径（默认放在data目录下）
        "token_usage_file": os.path.join(os.path.dirname(__file__), "token_usage.json")
    }

    # 必选配置校验
    if not config["api_key"]:
        raise ValueError("DOUBAO_API_KEY 未配置，请检查config/.env文件")
    return config


# 初始化配置
CONFIG = load_config()


# ====================== 重试机制 ======================
def get_retry_decorator():
    """获取重试装饰器"""
    return retry(
        stop=stop_after_attempt(CONFIG["retry_times"]),
        wait=wait_exponential(multiplier=CONFIG["retry_delay"], max=5),
        retry=retry_if_exception_type(
            (
                requests.exceptions.RequestException,
                requests.exceptions.HTTPError,
                json.JSONDecodeError
            )
        ),
        before_sleep=lambda retry_state: logger.warning(
            f"接口调用失败，即将重试（第{retry_state.attempt_number}次）：{retry_state.outcome.exception()}"
        ),
        retry_error_callback=lambda retry_state: {
            "error": f"重试{CONFIG['retry_times']}次后仍失败",
            "detail": str(retry_state.outcome.exception())
        }
    )


# ====================== 核心API调用函数 ======================
@get_retry_decorator()
def call_doubao_api(prompt: str) -> dict:
    """
    调用豆包API核心函数（适配纯文本返回，移除强制JSON解析）
    :param prompt: 输入提示词
    :return: 结构化响应（含raw_text/token_usage/error）
    """
    headers = {
        "Authorization": f"Bearer {CONFIG['api_key']}",
        "Content-Type": "application/json; charset=utf-8"
    }

    data = {
        "model": CONFIG["model"],
        "messages": [
            {"role": "system", "content": CONFIG["system_prompt"]},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "top_p": 0.9,
        "max_tokens": 1024,
        "stream": False
    }

    try:
        response = requests.post(
            url=CONFIG["api_url"],
            headers=headers,
            json=data,
            timeout=10
        )
        response.raise_for_status()
        resp_json = response.json()

        result = {
            "raw_text": "",  # 替换原data字段，存储纯文本响应
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "error": None
        }

        # 提取大模型返回的纯文本（不再强制JSON解析）
        if "choices" in resp_json and len(resp_json["choices"]) > 0:
            result["raw_text"] = resp_json["choices"][0]["message"]["content"].strip()
            logger.info(f"API返回纯文本：\n{result['raw_text']}")  # 日志打印原始文本，方便调试

        # Token统计（保留原有逻辑）
        if CONFIG["token_stat"] and "usage" in resp_json:
            prompt_tokens = resp_json["usage"].get("prompt_tokens", 0)
            completion_tokens = resp_json["usage"].get("completion_tokens", 0)
            total_tokens = resp_json["usage"].get("total_tokens", 0)

            result["token_usage"] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }

            logger.info(f"本次API调用Token消耗：输入{prompt_tokens} | 输出{completion_tokens} | 总计{total_tokens}")

        return result

    except requests.exceptions.HTTPError as e:
        error_detail = f"HTTP错误 {response.status_code}：{response.text}"
        logger.error(error_detail)
        if response.status_code == 401:
            error_detail += "（请检查API_KEY是否正确）"
        elif response.status_code == 429:
            error_detail += "（接口限流，请降低调用频率）"
        return {"raw_text": "", "token_usage": {}, "error": error_detail}
    except requests.exceptions.Timeout:
        logger.error("接口调用超时")
        return {"raw_text": "", "token_usage": {}, "error": "接口调用超时"}
    except Exception as e:
        logger.error(f"接口调用异常：{str(e)}")
        return {"raw_text": "", "token_usage": {}, "error": f"接口调用异常：{str(e)}"}


# ====================== 业务封装函数 ======================
def get_stock_tags(stock_list: list, trade_date: str) -> dict:
    """
    个股批量打标签（精简Prompt+本地组装JSON，降低Token消耗）
    :param stock_list: 股票列表 [{"ts_code": "002456.SZ", "name": "瑞芯微"}, ...]
    :param trade_date: 交易日期（如2026-02-13）
    :return: 结构化标签结果（本地组装的标准JSON）
    """
    # 步骤1：提取股票代码+名称，生成极简股票列表
    stock_codes = [f"{item['ts_code']}|{item['name']}" for item in stock_list]
    # 步骤2：精简Prompt（仅保留核心指令）
    prompt = f"""
你是A股短线题材分析师，仅需按以下规则输出文字，无任何多余内容、换行、标点（除|和数字序号）：
1. 分析日期：{trade_date}
2. 分析股票（按顺序）：{stock_codes}
3. 输出规则：
   - 每只股票占1行，开头标序号（1/2/3...），无其他前缀；
   - 内容格式：序号 核心概念、市场记忆|上涨逻辑|市场地位|事件催化
   - 字段要求：核心概念≤8字，上涨逻辑≤30字，市场地位（龙一/龙二/跟风/首板），事件催化≤18字；
   - 无内容填：无
4. 示例输出：
1 算力|AI服务器订单超预期|龙一|网传小作文利好
2 地产|政策放松预期|跟风|无
注意,你的回答中如果涉及同一个意思,就需要用相同文字描述,如有两只股票你检索后发现上涨逻辑都是雅下水电站开工催化,那么这两个股的逻辑说明都需要相同的文字说明.
    """

    # 步骤3：调用API获取极简文本响应（注意：现在返回的是raw_text字段）
    api_response = call_doubao_api(prompt)
    if api_response.get("error") or not api_response.get("raw_text"):
        # 透传错误，补充data字段保持返回格式统一
        api_response["data"] = []
        return api_response

    # 步骤4：本地解析极简文本，组装标准JSON
    try:
        raw_text = api_response["raw_text"]
        result_list = []

        # 按行解析大模型返回的内容
        for line in raw_text.split("\n"):
            line = line.strip()
            if not line or not line[0].isdigit():
                logger.warning(f"跳过无效行：{line}")
                continue  # 跳过空行/非序号行

            # 拆分序号和内容（兼容多空格分隔，比如"1  算力|AI订单"）
            parts = line.split(" ", 1)
            if len(parts) < 2:
                logger.warning(f"行格式错误，无内容部分：{line}")
                continue
            seq_part, content_part = parts
            try:
                seq = int(seq_part) - 1  # 转成列表索引（从0开始）
            except ValueError:
                logger.warning(f"序号非数字：{seq_part}，行内容：{line}")
                continue

            # 检查序号是否在股票列表范围内
            if seq < 0 or seq >= len(stock_list):
                logger.warning(f"序号{seq + 1}超出股票列表范围（共{len(stock_list)}只），跳过")
                continue

            # 拆分字段（核心概念|上涨逻辑|市场地位|事件催化）
            fields = content_part.split("|")
            # 补全缺失字段（避免大模型漏返回）
            core_concept = fields[0].strip() if len(fields) >= 1 else "无"
            rise_logic = fields[1].strip() if len(fields) >= 2 else "无"
            market_rank = fields[2].strip() if len(fields) >= 3 else "无"
            event_catalyst = fields[3].strip() if len(fields) >= 4 else "无"

            # 本地组装标准JSON结构
            result_list.append({
                "ts_code": stock_list[seq]["ts_code"],
                "name": stock_list[seq]["name"],
                "core_concept": core_concept,
                "rise_logic": rise_logic,
                "market_rank": market_rank,
                "event_catalyst": event_catalyst
            })

        # 组装最终返回结果（保持data字段，兼容原有调用逻辑）
        api_response["data"] = result_list
        # 保留raw_text方便调试
        logger.info(f"本地组装JSON完成，共解析{len(result_list)}只股票数据（原始文本行数：{len(raw_text.split(' |'))}）")
        return api_response

    except Exception as e:
        logger.error(f"本地组装JSON失败：{str(e)}，原始文本：{api_response['raw_text']}")
        api_response["error"] = f"本地解析组装JSON失败：{str(e)}"
        api_response["data"] = []
        return api_response

