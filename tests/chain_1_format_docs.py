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

# retriever = VectorStore()

def _format_docs(docs: List[Document]) -> str:
    # 函数整体注释：格式化检索到的文档列表为字符串上下文，用于 LLM 输入。
    # 如果文档为空，返回默认消息；否则，逐个文档提取元数据和内容，格式化为可读字符串。
    if not docs:  # 检查文档列表是否为空：如果为空，返回提示消息，避免空上下文。
        return "未在简历库中找到相关信息。"
    formatted_docs = []  # 初始化格式化文档列表：用于存储每个文档的字符串表示。
    for doc in docs:  # 循环遍历每个文档：提取并格式化信息。
        print("==========Document==============")
        print(doc)
        doc_hash = doc.metadata.get("doc_hash", doc.metadata.get("hash", "N/A"))  # 获取文档哈希：优先取 "doc_hash"，否则取 "hash"，默认 "N/A"。
        doc_str = (  # 构建单个文档字符串：包括来源文件、哈希和内容，使用换行分隔。
            f"简历来源文件: {os.path.basename(doc.metadata.get('file_path', 'N/A'))}\n"
            f"简历哈希值: {doc_hash}\n"
            f"内容: {doc.page_content}"
        )

        # 使用括号
        doc_str = (
            f"简历来源文件: {os.path.basename(doc.metadata.get('file_path', 'N/A'))}\n"
            f"简历哈希值: {doc_hash}\n"
            f"内容: {doc.page_content}"
        )

        print(doc_str)
        print(type(doc_str))
        formatted_docs.append(doc_str)  # 添加到列表：收集所有格式化字符串。
    return "\n\n---\n\n".join(formatted_docs)  # 返回合并字符串：用分隔线 "---" 连接每个文档，便于阅读。


# 验证 _format_docs 函数
from langchain_core.documents import Document
test_docs = [Document(page_content="测试内容1 具有大模型开发经历，在多个业务场景均有丰富的落地经验", metadata={"file_path": "test1.pdf", "doc_hash": "hash1"}),
             Document(page_content="测试内容2 具有机器学习开发经验，在多银行金融风控等场景，均有交付经验", metadata={"file_path": "test2.pdf", "doc_hash": "hash2"})]
formatted = _format_docs(test_docs)
assert "测试内容1" in formatted and "---" in formatted, "格式化失败"
print("格式化结果:", formatted)
# 空列表测试
empty_formatted = _format_docs([])
assert empty_formatted == "未在简历库中找到相关信息。", "空列表处理失败"



