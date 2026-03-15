"""
DeepSeek Chatbot 示例代码
《人工智能导论》HW02作业
功能：实现用户文本输入 → 调用DeepSeek大模型 → 获取并打印模型回复的完整对话流程
兼容：DeepSeek官方API、火山引擎DeepSeek R1 API
"""
import os
from openai import OpenAI


class DeepSeekChatbot:
    """DeepSeek 对话机器人核心类"""

    def __init__(self, api_key: str = None, base_url: str = "https://api.deepseek.com"):
        """
        初始化Chatbot
        :param api_key: DeepSeek API Key，优先从环境变量 DEEPSEEK_API_KEY 读取
        :param base_url: API 接口地址，默认DeepSeek官方地址，火山引擎需替换为对应地址
        """
        # 读取API Key，优先使用环境变量，避免硬编码密钥泄露
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError(
                "请配置DeepSeek API Key！\n"
                "方式1：设置系统环境变量 DEEPSEEK_API_KEY\n"
                "方式2：初始化时传入 api_key 参数"
            )

        # 初始化OpenAI兼容客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url
        )

        # 默认对话配置，可根据需求调整
        self.default_model = "deepseek-chat"
        self.default_temperature = 0.7
        self.default_max_tokens = 2048

    def send_message(self, user_input: str, stream: bool = True) -> str:
        """
        发送用户消息到DeepSeek模型，获取并返回回复
        :param user_input: 用户输入的文本问题
        :param stream: 是否开启流式输出，默认开启，模拟实时对话效果
        :return: 模型的完整回复内容
        """
        if not user_input.strip():
            return "请输入有效的问题内容！"

        # 构造对话消息
        messages = [
            {"role": "system", "content": "你是一个专业的人工智能助手，回答准确、简洁、有条理"},
            {"role": "user", "content": user_input}
        ]

        print(f"\n👤 用户：{user_input}")
        print(f"\n🤖 DeepSeek：", end="", flush=True)

        # 流式输出模式
        if stream:
            full_response = ""
            response = self.client.chat.completions.create(
                model=self.default_model,
                messages=messages,
                temperature=self.default_temperature,
                max_tokens=self.default_max_tokens,
                stream=True
            )

            for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    print(content, end="", flush=True)
            print("\n")
            return full_response

        # 非流式输出模式
        else:
            response = self.client.chat.completions.create(
                model=self.default_model,
                messages=messages,
                temperature=self.default_temperature,
                max_tokens=self.default_max_tokens,
                stream=False
            )
            full_response = response.choices[0].message.content
            print(full_response, "\n")
            return full_response


# 主程序入口
if __name__ == "__main__":
    print("=" * 50)
    print("DeepSeek Chatbot 启动成功！")
    print("输入 exit 或 quit 退出对话")
    print("=" * 50)

    # 初始化Chatbot
    # 火山引擎用户请替换base_url为火山方舟提供的接口地址，并传入对应的API Key
    try:
        chatbot = DeepSeekChatbot()
    except ValueError as e:
        print(f"初始化失败：{e}")
        exit(1)

    # 对话循环
    while True:
        # 获取用户输入
        user_input = input("\n请输入你的问题：").strip()

        # 退出指令
        if user_input.lower() in ["exit", "quit", "退出"]:
            print("感谢使用，对话结束！")
            break

        # 发送消息并获取回复
        chatbot.send_message(user_input)