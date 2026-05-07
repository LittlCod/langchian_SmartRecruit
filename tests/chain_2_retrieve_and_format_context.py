# rag/chain.py
import asyncio
import os
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from loguru import logger
from config import config
from utils.vector_store import VectorStore

# 初始化LLM和检索器
llm = ChatOpenAI(
    model_name="qwen-plus",
    openai_api_key=config.DASHSCOPE_API_KEY,
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.1
)
retriever = VectorStore()

def _format_docs(docs: List[Document]) -> str:
    # 函数整体注释：格式化检索到的文档列表为字符串上下文，用于 LLM 输入。
    # 如果文档为空，返回默认消息；否则，逐个文档提取元数据和内容，格式化为可读字符串。
    if not docs:  # 检查文档列表是否为空：如果为空，返回提示消息，避免空上下文。
        return "未在简历库中找到相关信息。"
    formatted_docs = []  # 初始化格式化文档列表：用于存储每个文档的字符串表示。
    for doc in docs:  # 循环遍历每个文档：提取并格式化信息。
        doc_hash = doc.metadata.get("doc_hash", doc.metadata.get("hash", "N/A"))  # 获取文档哈希：优先取 "doc_hash"，否则取 "hash"，默认 "N/A"。
        doc_str = (  # 构建单个文档字符串：包括来源文件、哈希和内容，使用换行分隔。
            f"简历来源文件: {os.path.basename(doc.metadata.get('file_path', 'N/A'))}\n"
            f"简历哈希值: {doc_hash}\n"
            f"内容: {doc.page_content}"
        )
        formatted_docs.append(doc_str)  # 添加到列表：收集所有格式化字符串。
    return "\n\n---\n\n".join(formatted_docs)  # 返回合并字符串：用分隔线 "---" 连接每个文档，便于阅读。



REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是一位专业的招聘助理..."), # 内容不变，为简洁省略
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])

async def retrieve_and_format_context(input_dict: dict, config: RunnableConfig) -> str:
    # 函数整体注释：异步检索和格式化上下文，支持查询改写、参数过滤和异常处理。
    # 步骤：改写查询 → 解析参数 → 异步检索 → 格式化结果，返回上下文字符串。
    # 1. 改写查询  # 步骤注释：使用 REWRITE_PROMPT 和 LLM 改写原始输入，融入历史聊天，确保查询精确。
    rewritten_question = await (REWRITE_PROMPT | llm | StrOutputParser()).ainvoke(
        {"input": input_dict["input"], "chat_history": input_dict["chat_history"]},
        config=config
    )
    logger.info(f"原始查询: '{input_dict['input']}' -> 改写后查询: '{rewritten_question}'")  # 记录日志：显示改写前后对比，便于调试。

    # 2. [升级] 解析参数并构建Filter表达式  # 步骤注释：从 input_dict 获取 params，用于检索过滤，为什么升级？因为支持动态过滤如经验年限。
    params = input_dict.get("params", {})  # 获取参数字典：默认空字典，避免 KeyError。

    # 3. [升级] 异步执行高级混合检索  # 步骤注释：调用检索器的异步方法，传入改写查询和 params，进行混合搜索。
    try:  # 尝试执行检索：捕获异常，确保链不崩溃。
        retrieved_docs = await retriever.aget_relevant_documents(
            query=rewritten_question,
            params=params
        )
    except Exception as e:  # 捕获检索异常：记录错误，返回提示消息。
        logger.error(f"检索器执行失败: {e}", exc_info=True)  # 记录错误日志：包含栈追踪。
        return "检索简历时发生内部错误，请稍后再试。"  # 返回用户友好错误消息。

    # 4. 格式化  # 步骤注释：调用 _format_docs 将文档列表转为字符串上下文。
    context = _format_docs(retrieved_docs)  # 格式化检索文档：生成可读字符串。
    logger.debug(f"为LLM准备的上下文: \n{context}")  # 调试日志：显示准备好的上下文。
    return context  # 返回上下文字符串：供后续链使用。


# 验证 retrieve_and_format_context 函数（异步，需要 asyncio）
import asyncio
from langchain_core.runnables import RunnableConfig
test_input_dict = {"input": "测试查询", "chat_history": [], "params": {"count": 1}}
test_config = RunnableConfig()
async def test_retrieve():
    context = await retrieve_and_format_context(test_input_dict, test_config)
    assert isinstance(context, str) and len(context) > 0, "上下文为空"
    print("上下文:", context)

try:
    asyncio.run(test_retrieve())
except Exception as e:
    print("测试失败:", e)