from rag_pipeline_5__arun import SmartRecruitAgent
import asyncio


# 运行示例
async def main():
    agent = SmartRecruitAgent()

    # 从返回的字典中获取 'response' 字段进行打印
    response1 = await agent.arun("你好")
    print("Agent: " + response1.get("response", "无法获取响应"))

    print("---")

    response2 = await agent.arun("你能做什么？")
    print("Agent: " + response2.get("response", "无法获取响应"))

    print("---")

    response3 = await agent.arun("我需要一个有三年经验的Python工程师，熟悉微服务架构。")
    print("Agent: " + response3.get("response", "无法获取响应"))

    print("---")

    response4 = await agent.arun("这个职位还需要熟悉机器学习。")
    print("Agent: " + response4.get("response", "无法获取响应"))


if __name__ == "__main__":
    asyncio.run(main())