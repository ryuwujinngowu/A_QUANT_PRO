import requests
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv

_ENV_PATH = os.path.join(os.path.dirname(__file__), "..", "config", ".env")
load_dotenv(dotenv_path=_ENV_PATH)


def _load_push_tokens():
    """从 .env 中读取 PUSH_TOKENS（逗号分隔），未配置时返回空列表。"""
    raw = os.getenv("PUSH_TOKENS", "")
    return [t.strip() for t in raw.split(",") if t.strip()]


def send_wechat_message_to_multiple_users(title, content, tokens=None):
    """
    通过PushPlus向多个微信用户发送消息（官方合规方式）
    :param title: 消息标题（必填）
    :param content: 消息内容
    :param tokens: 接收人的PushPlus Token列表，如["token1", "token2"]；
                   不传时自动从 config/.env 的 PUSH_TOKENS 读取
    :return: 整体推送结果（True=全部成功/False=至少一个失败）
    """
    if tokens is None:
        tokens = _load_push_tokens()

    # PushPlus的核心API地址
    url = "http://www.pushplus.plus/send"
    # 记录每个用户的推送结果
    push_results = []

    if not tokens:
        print("❌ 未配置任何用户Token！")
        return False

    for idx, token in enumerate(tokens, 1):
        if not token:
            print(f"❌ 第{idx}个用户的Token为空，跳过推送")
            push_results.append(False)
            continue

        # 构造单用户推送参数（符合官方文档要求）
        data = {
            "token": token,  # 单个用户的Token（官方仅支持单个）
            "title": title,
            "content": content,
            "template": "txt"  # 消息格式：txt(纯文本)/markdown(富文本)
        }

        try:
            # 发送POST请求（单用户）
            response = requests.post(
                url=url,
                data=json.dumps(data),
                headers={"Content-Type": "application/json"},
                timeout=10  # 增加超时控制，避免卡壳
            )
            # 解析响应结果
            result = response.json()
            if result["code"] == 200:
                print(f"✅ 第{idx}个用户消息推送成功！")
                push_results.append(True)
            else:
                print(f"❌ 第{idx}个用户推送失败：{result['msg']}（Token：{token[:8]}...）")
                push_results.append(False)
        except Exception as e:
            print(f"❌ 第{idx}个用户推送过程出错：{str(e)}（Token：{token[:8]}...）")
            push_results.append(False)

    # 整体结果：全部成功才返回True，否则False
    return all(push_results)


# 主程序：模拟Python程序输出并推送
if __name__ == "__main__":
    # 模拟Python程序的输出内容
    program_output = """
[量化Agent] 2026-03-16 收益汇总 📊 策略跟踪统计 | 2026-03-16
====================================
▶ 跌停板战法选手
   日内均收: +3.36%
   次日开盘: N/A  |  收盘: N/A
   最高溢价: N/A  |  最大亏损: N/A
▶ 高位股跟踪选手
   日内均收: +1.23%
   次日开盘: N/A  |  收盘: N/A
   最高溢价: N/A  |  最大亏损: N/A
▶ 午盘打板选手
   日内均收: -0.08%
   次日开盘: N/A  |  收盘: N/A
   最高溢价: N/A  |  最大亏损: N/A
▶ 昨日top1板块高开无脑跟选手
   日内均收: -0.76%
   次日开盘: N/A  |  收盘: N/A
   最高溢价: N/A  |  最大亏损: N/A
▶ 中位股跟踪选手
   日内均收: -0.96%
   次日开盘: N/A  |  收盘: N/A
   最高溢价: N/A  |  最大亏损: N/A
▶ 早盘打板选手
   日内均收: -1.83%
   次日开盘: N/A  |  收盘: N/A
   最高溢价: N/A  |  最大亏损: N/A
====================================
⚠ 以上为模拟跟踪数据，仅供参考。
    """

    # 配置两个接收人的PushPlus Token（替换成你实际的有效Token！）


    # 调用多用户推送函数
    total_result = send_wechat_message_to_multiple_users(
        title="Python程序运行报告",
        content=program_output
    )

    # 最终结果提示
    if total_result:
        print("🎉 所有用户都推送成功！")
    else:
        print("⚠️ 部分用户推送失败，请检查Token是否正确！")