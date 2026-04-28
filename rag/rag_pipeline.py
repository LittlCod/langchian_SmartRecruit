# rag/rag_pipeline.py
import asyncio
import json
from typing import List, Dict, Any, Optional

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger

from config import config
from rag.chain import get_rag_chain

# --- 1. 初始化核心组件 ---
llm = ChatOpenAI(
    model_name="qwen-plus",
    openai_api_key=config.DASHSCOPE_API_KEY,
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.0,
)



# --- 3. 主Agent逻辑 ---
class SmartRecruitAgent:
    pass




#  验证代码 ---
if __name__ == '__main__':
    async def main1():
        pass


    async def main2():
        pass



    asyncio.run(main1())
