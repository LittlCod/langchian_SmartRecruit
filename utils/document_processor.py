# document_processor.py
import os
import re
import hashlib
from typing import List, Optional, Dict, Any
from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader, PyPDFLoader, Docx2txtLoader, \
    UnstructuredPowerPointLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownTextSplitter
from datetime import datetime
from openai import OpenAI
import base64
from loguru import logger
from config import config
import json

# --- 日志配置 ---
logger.add(os.path.join(config.LOG_DIR, "document_processor.log"), rotation="10 MB", encoding="utf-8")

# --- LLM 客户端初始化 ---
# 为结构化数据解析创建一个独立的客户端
parser_client = OpenAI(
    api_key=config.DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)







# --- [新增] 验证代码 ---
if __name__ == "__main__":
    """验证文档加载、解析和切分功能"""
    logger.info("="*50)
    logger.info("开始独立验证 document_processor.py 模块...")
    
    # 选择一个测试文件
    test_dir = config.LOCAL_RESUME_DIR
    test_file_name = "李明AI大模型产品经理简历.pdf" # 你可以换成任何一个存在的文件名
    test_file_path = os.path.join(test_dir, test_file_name)
    print(test_file_path)
    if not os.path.exists(test_file_path):
        logger.error(f"测试文件不存在，请确保 '{test_file_path}' 存在后再运行验证。")
    else:
        try:
           pass
        except Exception as e:
            logger.critical(f"document_processor.py 模块验证失败: {e}", exc_info=True)
            print(f"\n[FAILURE] document_processor.py module validation failed. Check logs at ")
