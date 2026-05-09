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

def compute_file_hash(file_path: str) -> str:
    """
        计算文件的 MD5 哈希值，用于简历去重。

        原理：同一份文件（内容完全相同）的 MD5 值一定相同，
        不同文件的 MD5 值几乎一定不同（碰撞概率极低）。
        因此可以用 MD5 来判断"这份简历我们是不是已经处理过了"。

        Args:
            file_path: 文件路径

        Returns:
            32位十六进制 MD5 字符串，例如 "d41d8cd98f00b204e9800998ecf8427e"

        Raises:
            FileNotFoundError: 文件不存在时抛出
    """
    hasher = hashlib.md5()
    try:
        # 以二进制模式读取，避免编码问题
        with open(file_path, "rb") as f:
            # 每次读 4KB，避免大文件一次性占满内存
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        logger.error(f"计算文件hash失败: {file_path}, 错误: {str(e)}")
        raise

def extract_text_from_image(image_path: str, client: OpenAI) -> str:
    """
    使用多模态大模型从图片中提取简历文本。

    本项目使用阿里云 qwen-omni-turbo 模型，它是一个视觉语言模型（VLM），
    能够理解图片中的文字和布局。

    与传统 OCR（如 Tesseract）的区别：
    - OCR 只能识别文字，不理解语义
    - VLM 不仅能识别文字，还能理解布局（如"这段是工作经历"）

    Args:
        image_path: 图片文件路径（.jpg / .png）
        client: OpenAI 兼容客户端（已配置 DashScope base_url）

    Returns:
        提取的纯文本字符串

    Raises:
        FileNotFoundError: 图片文件不存在
        Exception: API 调用失败
    """
    # 1. 将图片转为 base64 编码
    #    OpenAI Vision API 要求图片以 base64 格式嵌入请求
    with open(image_path, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode("utf-8")

    # 2. 构造多模态请求
    #    消息体是一个列表，包含文本和图片两种类型的 content
    response = client.chat.completions.create(
        model="qwen-omni-turbo",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_base64}"
                        },
                    },
                    {
                        "type": "text",
                        "text": "提取图片中的简历文本信息，包括个人信息、教育背景、工作经历等。输出纯文本。",
                    },
                ],
            }
        ],
        stream=False,  # 不使用流式输出，直接返回完整结果
    )

    content = response.choices[0].message.content
    logger.info(f"图片提取文本成功: {image_path}, 内容长度: {len(content)}")
    return content

# --- 文件格式 → 加载器映射表 ---
# None 表示不使用 LangChain Loader，走特殊处理（图片用 VLM）
document_loaders = {
    ".txt": TextLoader,
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".ppt": UnstructuredPowerPointLoader,
    ".pptx": UnstructuredPowerPointLoader,
    ".jpg": None,  # 图片：使用 extract_text_from_image()
    ".png": None,
    ".md": UnstructuredMarkdownLoader,
}

def load_and_hash_document(file_path: str, client: OpenAI) -> tuple[str, str]:
    """
    加载文件内容并计算 MD5 哈希。

    根据文件扩展名自动选择对应的加载器：
    - 图片（.jpg/.png）：调用 qwen-omni-turbo 多模态提取
    - .txt：尝试 utf-8 → gbk → latin1 三种编码（兼容 Windows GBK 文件）
    - 其他格式：使用对应的 LangChain Loader

    Args:
        file_path: 文件路径
        client: OpenAI 兼容客户端（图片提取时需要）

    Returns:
        (content, doc_hash) 元组
        - content: 提取的纯文本
        - doc_hash: 文件 MD5 哈希（32位十六进制）

    Raises:
        ValueError: 不支持的文件格式
        UnicodeDecodeError: txt 文件所有编码都失败
        Exception: 加载过程中的其他错误
    """
    logger.info(f"开始加载并哈希文件：{file_path}")

    # 得到扩展名
    file_extension = os.path.splitext(file_path)[1].lower()

    # 检查格式是否支持
    if file_extension not in document_loaders:
        raise ValueError(f"不支持的格式类型：{file_extension}")

    content = ""

    try:
        # 图片走多模态
        if file_extension in [".jpg", ".png"]:
            content = extract_text_from_image(file_path, client)
        else:
            loader_class = document_loaders[file_extension]
            # 3. TXT 文件需要尝试多种编码（兼容中文环境）
            if file_extension == ".txt":
                encodings = ["utf-8", "gbk", "latin1"]
                for enc in encodings:
                    try:
                        loader = loader_class(file_path, encoding=enc)
                        content = loader.load()[0].page_content
                        break  # 成功就用这个编码
                    except UnicodeDecodeError:
                        continue  # 当前编码失败，试下一个
                else:
                    # 所有编码都失败
                    raise UnicodeDecodeError(
                        f"无法以支持的编码加载文件: {file_path}",
                        b"", 0, 0, "尝试所有编码失败"
                    )
            else:
                # 4. 其他格式直接加载
                loader = loader_class(file_path)
                content = loader.load()[0].page_content
        # 计算hash
        doc_hash = compute_file_hash(file_path)
        logger.info(f"文件加载并哈希成功: {file_path}, hash: {doc_hash}")
        return content, doc_hash

    except Exception as e:
        logger.error(f"加载或哈希文件失败：{file_path}，错误：{str(e)}")
        raise e


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
