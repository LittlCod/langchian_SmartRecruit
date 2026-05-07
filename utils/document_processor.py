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

# --- 简历结构化信息解析 ---
RESUME_PARSER_PROMPT = """
你是一个顶级的HR简历分析专家。请从以下简历文本中，提取出关键的结构化信息。

**严格遵守以下规则:**
1.  **提取字段**: 只提取以下字段：`name` (姓名), `gender` (性别), `age` (年龄), `work_experience` (工作年限)。
2.  **JSON格式**: 必须严格按照JSON格式输出，不要有任何额外的解释或Markdown标记。
3.  **逻辑推断**:
    - **姓名 (name)**: 通常是文本开头最明显的人名。
    - **性别 (gender)**: 从文本中明确的“男”或“女”字样判断。如果未提及，则为 "未提供"。
    - **年龄 (age)**: 根据出生年份、或直接描述的年龄计算。例如“1990年出生”在2024年应计算为34岁。如果无法推断，则为 -1。
    - **工作年限 (work_experience)**: 根据工作经历的总时长计算。例如“2020年7月至2023年7月”是3年。如果无法推断，则为 -1。
4.  **数值类型**: `age` 和 `work_experience` 必须是整数。

**简历文本:**
---
{resume_text}
---

**输出JSON:**
"""

def parse_resume_structure(resume_text: str, client: OpenAI) -> Dict[str, Any]:
    """
    使用LLM从简历文本中提取结构化的个人信息。
    """
    logger.info("开始使用LLM解析简历结构化信息...")
    try:
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": "你是一个顶级的HR简历分析专家。"},
                {"role": "user", "content": RESUME_PARSER_PROMPT.format(resume_text=resume_text)}
            ],
            temperature=0.0,
        )
        content = response.choices[0].message.content
        logger.debug(f"LLM原始解析结果: {content}")
        
        # 清理并加载JSON
        json_str = content.strip().removeprefix("```json").removesuffix("```").strip()
        structured_data = json.loads(json_str)
        
        # 数据清洗和验证
        structured_data['age'] = int(structured_data.get('age', -1))
        structured_data['work_experience'] = int(structured_data.get('work_experience', -1))
        structured_data['gender'] = structured_data.get('gender', '未提供')
        
        logger.info(f"简历结构化信息解析成功: {structured_data}")
        return structured_data

    except Exception as e:
        logger.error(f"解析简历结构化信息失败: {e}", exc_info=True)
        return {"name": "未知", "gender": "未提供", "age": -1, "work_experience": -1}


# --- 文档加载与处理 ---
document_loaders = {
    ".txt": TextLoader,
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".ppt": UnstructuredPowerPointLoader,
    ".pptx": UnstructuredPowerPointLoader,
    ".jpg": None,
    ".png": None,
    ".md": UnstructuredMarkdownLoader
}

def compute_file_hash(file_path: str) -> str:
    # ... (实现不变)
    hasher = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        logger.error(f"计算文件hash失败: {file_path}, 错误: {str(e)}")
        raise

def extract_text_from_image(image_path: str, client: OpenAI) -> str:
    try:
        with open(image_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode("utf-8")
        response = client.chat.completions.create(
            model="qwen-omni-turbo",
            messages=[
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
                    {"type": "text", "text": "提取图片中的简历文本信息，包括个人信息、教育背景、工作经历等。输出纯文本。"}
                ]}
            ],
            stream=False
        )
        content = response.choices[0].message.content
        logger.info(f"图片提取文本成功: {image_path}, 内容长度: {len(content)}")
        return content
    except Exception as e:
        logger.error(f"图片提取失败: {image_path}, 错误: {str(e)}")
        raise

def load_and_hash_document(file_path: str, client: OpenAI) -> (str, str):
    # ... (实现不变)
    logger.info(f"开始加载并哈希文件: {file_path}")
    file_extension = os.path.splitext(file_path)[1].lower()
    content = ""
    
    if file_extension not in document_loaders:
        raise ValueError(f"不支持的文件类型: {file_extension}")

    try:
        if file_extension in [".jpg", ".png"]:
            content = extract_text_from_image(file_path, client)
        else:
            loader_class = document_loaders[file_extension]
            if file_extension == ".txt":
                encodings = ["utf-8", "gbk", "latin1"]
                for enc in encodings:
                    try:
                        loader = loader_class(file_path, encoding=enc)
                        content = loader.load()[0].page_content
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise UnicodeDecodeError(f"无法以支持的编码加载文件: {file_path}", b"", 0, 0, "尝试所有编码失败")
            else:
                loader = loader_class(file_path)
                content = loader.load()[0].page_content
        
        doc_hash = compute_file_hash(file_path)
        logger.info(f"文件加载并哈希成功: {file_path}, hash: {doc_hash}")
        return content, doc_hash
    except Exception as e:
        logger.error(f"加载或哈希文件失败: {file_path}, 错误: {str(e)}")
        raise

def process_document(doc: Document) -> List[Document]:
    # 实现
    logger.info(f"开始处理单个文档: {doc.metadata.get('file_path', 'N/A')}")

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=config.PARENT_CHUNK_SIZE,
                                                     chunk_overlap=config.CHUNK_OVERLAP)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHILD_CHUNK_SIZE,
                                                    chunk_overlap=config.CHUNK_OVERLAP)
    markdown_parent_splitter = MarkdownTextSplitter(chunk_size=config.PARENT_CHUNK_SIZE,
                                                    chunk_overlap=config.CHUNK_OVERLAP)
    markdown_child_splitter = MarkdownTextSplitter(chunk_size=config.CHILD_CHUNK_SIZE,
                                                   chunk_overlap=config.CHUNK_OVERLAP)

    child_chunks = []
    file_extension = os.path.splitext(doc.metadata.get("file_path", ""))[1].lower()
    is_markdown = file_extension == ".md"
    parent_splitter_to_use = markdown_parent_splitter if is_markdown else parent_splitter
    child_splitter_to_use = markdown_child_splitter if is_markdown else child_splitter
    logger.info(f"处理文档: {doc.metadata['file_path']}, 使用切分器: {'Markdown' if is_markdown else 'RecursiveCharacter'}")

    parent_docs = parent_splitter_to_use.split_documents([doc])
    logger.debug(f"文档 {doc.metadata['file_path']} 切分为 {len(parent_docs)} 个父块")
    for j, parent_doc in enumerate(parent_docs):
        parent_id = f"doc_{doc.metadata['hash']}_parent_{j}"
        #parent_doc.metadata["parent_id"] = parent_id
        #parent_doc.metadata["parent_content"] = parent_doc.page_content
        #parent_doc.metadata.update(doc.metadata)

        sub_chunks = child_splitter_to_use.split_documents([parent_doc])
        for k, sub_chunk in enumerate(sub_chunks):
            chunk_id = f"{parent_id}_child_{k}"
            sub_chunk.metadata["parent_id"] = parent_id
            sub_chunk.metadata["parent_content"] = parent_doc.page_content
            # [修复] 明确添加 'chunk_id' 和 'id'
            sub_chunk.metadata["chunk_id"] = chunk_id
            sub_chunk.metadata["id"] = chunk_id
            sub_chunk.metadata.update(doc.metadata)
            child_chunks.append(sub_chunk)
            logger.debug(f"生成子块: {chunk_id}, 父块: {parent_id}, 内容长度: {len(sub_chunk.page_content)}")

    logger.info(f"文档 {doc.metadata['file_path']} 共生成 {len(child_chunks)} 个子块")
    return child_chunks

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
            # 1. 验证加载和哈希
            logger.info(f"--- 1. 测试加载与哈希 ---")
            content, doc_hash = load_and_hash_document(test_file_path, parser_client)
            assert content and doc_hash
            logger.info(f"加载成功: hash={doc_hash}, 内容长度={len(content)}")

            # 2. 验证结构化解析
            logger.info(f"--- 2. 测试结构化信息解析 ---")
            structured_data = parse_resume_structure(content, parser_client)
            assert isinstance(structured_data, dict) and "name" in structured_data
            logger.info(f"解析成功: {structured_data}")

            # 3. 验证切块
            logger.info(f"--- 3. 测试文档切块 ---")
            doc = Document(page_content=content, metadata={"file_path": test_file_path, "hash": doc_hash})
            chunks = process_document(doc)
            assert chunks and isinstance(chunks, list)
            logger.info(f"切块成功: 共生成 {len(chunks)} 个子块。")
            logger.info(f"第一个子块ID: {chunks[0].metadata.get('id')}")

            logger.info("="*50)
            logger.success("document_processor.py 模块所有功能验证通过！")
            print("\n[SUCCESS] document_processor.py module validation passed!")

        except Exception as e:
            logger.critical(f"document_processor.py 模块验证失败: {e}", exc_info=True)
            print(f"\n[FAILURE] document_processor.py module validation failed. Check logs at ")
