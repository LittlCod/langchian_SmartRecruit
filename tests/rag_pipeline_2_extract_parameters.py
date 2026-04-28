# rag/rag_pipeline.py
import asyncio
import json
from typing import List, Dict, Any, Optional

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger

from config import config

# --- 1. 初始化核心组件 ---
llm = ChatOpenAI(
    model_name="qwen-plus",
    openai_api_key=config.DASHSCOPE_API_KEY,
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.0,
)

# --- 2. 意图识别、参数提取与安全护栏 ---

INTENT_PROMPT = ChatPromptTemplate.from_template(
    """
你是一个精准的用户意图分类机器人。请分析用户的最新输入，并结合历史对话，将其意图分类为以下几种之一：
'recruitment', 'refinement_or_correction', 'follow_up_question', 'general_job_inquiry', 'chit_chat', 'meta_inquiry'。

- 'recruitment': 用户首次提出或提出一个全新的招聘需求。
- 'refinement_or_correction': 用户对上一个招聘需求进行修改、补充或纠正。
- 'follow_up_question': 用户针对上一次返回的候选人或职位信息进行追问。
- 'general_job_inquiry': 用户提出一个与具体招聘需求无关，但与职位、技能、行业知识等相关的通用性问题。
- 'chit_chat': 与招聘无关的闲聊、问候或反馈。
- 'meta_inquiry': 询问关于你自身能力或身份的问题。

如果完全无法判断，或用户的要求超出了招聘助手的能力范围，请分类为 'fallback'。

---
对话历史:
{chat_history}
---
用户的最新输入: "{input}"
---

请严格按照JSON格式输出，只包含意图分类:
{{"intent": "..."}}
"""
)

# 定义意图识别链
intent_chain = INTENT_PROMPT | llm
async def recognize_intent(query: str, chat_history: List[BaseMessage]) -> str:
    # 函数整体注释：异步识别用户意图，基于查询和历史，使用 INTENT_PROMPT 和 LLM 分类。
    # 将历史转为字符串，调用链解析 JSON，返回意图或默认 'fallback'。
    history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history])  # 构建历史字符串：循环遍历 chat_history，格式化为 "type: content"，用换行连接，便于提示输入。
    try:  # 尝试调用意图链：捕获异常，确保函数不崩溃。
        response = await intent_chain.ainvoke({"input": query, "chat_history": history_str})  # 异步调用意图链：输入 query 和历史，获取 LLM 响应。
        logger.debug(f"LLM原始意图响应: {response.content}")  # 调试日志：记录原始响应内容。
        cleaned_content = response.content.strip().removeprefix("```json").removesuffix("```").strip()  # 清理响应：去除前后空格、JSON 标记，确保纯 JSON 字符串。
        result = json.loads(cleaned_content)  # 解析 JSON：转为字典。
        intent = result.get("intent", "fallback")  # 获取意图：从字典取 "intent"，默认 "fallback"。
        logger.info(f"意图识别成功: '{query}' -> '{intent}'")  # 记录日志：显示识别结果。
        return intent  # 返回意图字符串。
    except Exception as e:  # 捕获异常：如 JSON 解析失败或链调用错误。
        logger.error(f"意图识别失败: {e}. 默认为 'fallback'.")  # 记录错误日志。
        return "fallback"  # 返回默认意图。




PARAMETER_EXTRACTION_PROMPT = ChatPromptTemplate.from_template(
    """
你是一个顶级的HR助理机器人。请从用户的招聘需求中，提取结构化的筛选条件。

**严格遵守以下规则:**
1.  **提取字段**: `count` (数量), `gender` (性别), `age_min` (最小年龄), `age_max` (最大年龄), `experience_min` (最少工作经验), `experience_max` (最多工作经验)。
2.  **JSON格式**: 必须严格按照JSON格式输出。如果某个字段未提及，则其值应为 `null`。
3.  **逻辑推断**: 
    - **count**: 如果未提及，默认为 3。
    - **gender**: 只能是 "男", "女", 或 `null`。
    - **age**: "30岁左右" 可推断为 `age_min: 28, age_max: 32`。"不超过40岁" 为 `age_max: 40`。
    - **experience**: "5年以上" 为 `experience_min: 5`。"3到5年" 为 `experience_min: 3, experience_max: 5`。
4.  **数值类型**: 所有年龄和经验字段必须是整数。

**用户需求:**
---
{input}
---

**输出JSON:**
"""
)

parameter_extraction_chain = PARAMETER_EXTRACTION_PROMPT | llm
async def extract_parameters(query: str) -> Dict[str, Any]:
    """提取结构化招聘参数"""
    # 函数整体注释：异步提取招聘参数，使用 PARAMETER_EXTRACTION_PROMPT 和 LLM 解析查询为 JSON 字典。
    # 如果失败，返回默认参数字典。
    try:  # 尝试调用参数链：捕获异常，确保返回默认值。
        response = await parameter_extraction_chain.ainvoke({"input": query})  # 异步调用参数链：输入 query，获取 LLM 响应。
        cleaned_content = response.content.strip().removeprefix("```json").removesuffix("```").strip()  # 清理响应：去除空格和 JSON 标记。
        params = json.loads(cleaned_content)  # 解析 JSON：转为字典。
        logger.info(f"从查询 '{query}' 中提取到参数: {params}")  # 记录日志：显示提取结果。
        return params  # 返回参数字典。
    except Exception as e:  # 捕获异常：如解析失败。
        logger.error(f"参数提取失败: {e}. 使用默认值。")  # 记录错误日志。
        return {"count": 3, "gender": None, "age_min": None, "age_max": None, "experience_min": None, "experience_max": None}  # 返回默认字典：count=3，其他 null。

# 验证 extract_parameters 函数
import asyncio
test_query = "帮我找一个有5年以上经验的算法工程师"
async def params_test():
    params = await extract_parameters(test_query)
    assert isinstance(params, dict) and "count" in params, "参数提取失败"
    logger.info(f"参数: {params}")

asyncio.run(params_test())