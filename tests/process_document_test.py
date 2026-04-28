# document_processor.py
import os
import re
import hashlib
from typing import List, Optional, Dict, Any
from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader, PyPDFLoader, Docx2txtLoader, \
    UnstructuredPowerPointLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import RecursiveCharacterTextSplitter, MarkdownTextSplitter
from datetime import datetime
from openai import OpenAI
import base64
from loguru import logger
from config import config
import json
from load_and_hash_document_test import load_and_hash_document


def process_document(doc: Document) -> List[Document]:
    # 函数整体注释：将输入的Document对象切分成父块和子块，支持Markdown和普通文本格式。
    # 使用递归字符切分器或Markdown专用切分器，根据配置的块大小和重叠进行分块。
    # 为每个子块添加元数据，包括ID、父块内容等，便于后续检索。
    # 返回子文档列表；日志记录切分过程。
    logger.info(f"开始处理单个文档: {doc.metadata.get('file_path', 'N/A')}")  # 记录日志：开始处理文档，显示文件路径（如果不存在则显示'N/A'）。
    print("开始处理单个文档Document:============== ")
    print(doc)

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=200,  # 创建父块切分器：使用RecursiveCharacterTextSplitter，设置父块大小（从config获取）。
                                                     chunk_overlap=100)  # 设置块重叠大小（从config获取），用于保持上下文连续性。

    child_splitter = RecursiveCharacterTextSplitter(chunk_size=100,  # 创建子块切分器：类似父块，但块大小更小（从config获取）。
                                                    chunk_overlap=50)  # 设置子块重叠大小（从config获取）。

    markdown_parent_splitter = MarkdownTextSplitter(chunk_size=200,  # 创建Markdown专用父块切分器：使用MarkdownTextSplitter，适合处理MD文件结构（如标题、列表）。
                                                    chunk_overlap=100)  # 设置重叠大小。

    markdown_child_splitter = MarkdownTextSplitter(chunk_size=100,  # 创建Markdown专用子块切分器：类似，但块大小更小。
                                                   chunk_overlap=50)  # 设置重叠大小。


    child_chunks = []  # 初始化空列表，用于存储所有生成的子块Document对象。


    file_extension = os.path.splitext(doc.metadata.get("file_path", ""))[1].lower()  # 从metadata中提取文件路径，获取文件扩展名（小写），如".md"或".txt"。
    print("文件路径 doc.metadata.get('file_path', "")获取:  \n", doc.metadata.get("file_path", ""))
    print("文件路径切分 os.path.splitext(doc.metadata.get('file_path', "")):\n", os.path.splitext(doc.metadata.get("file_path", "")))
    print("文件路径切分 os.path.splitext(doc.metadata.get('file_path', ""))[1]: \n", os.path.splitext(doc.metadata.get("file_path", ""))[1])
    print("文件路径切分 os.path.splitext(doc.metadata.get('file_path', ""))[1].lower(): \n", os.path.splitext(doc.metadata.get("file_path", ""))[1].lower())


    is_markdown = file_extension == ".md"  # 判断是否为Markdown文件：如果扩展名为".md"，则为True，否则False。

    print("判断文件类型为是否为Markdown 文件,是则采用markdown_parent_splitter切分器，否则采用parent_splitter切分器（RecursiveCharacterTextSplitter） ")
    parent_splitter_to_use = markdown_parent_splitter if is_markdown else parent_splitter  # 根据是否Markdown选择父块切分器：如果是MD，用Markdown切分器；否则用普通递归字符切分器。

    child_splitter_to_use = markdown_child_splitter if is_markdown else child_splitter  # 根据是否Markdown选择子块切分器：类似选择逻辑。

    logger.info(f"处理文档: {doc.metadata['file_path']}, 使用切分器: {'Markdown' if is_markdown else 'RecursiveCharacter'}")  # 记录日志：显示处理的文档路径和使用的切分器类型。


    parent_docs = parent_splitter_to_use.split_documents([doc])  # 使用选择的父块切分器，将输入doc切分成父块列表（输入是[doc]，因为split_documents期望文档列表）。


    logger.debug(f"文档 {doc.metadata['file_path']} 切分为 {len(parent_docs)} 个父块")  # 调试日志：记录切分后的父块数量。


    for j, parent_doc in enumerate(parent_docs):  # 循环遍历每个父块：j是索引，从0开始；parent_doc是当前父块Document。

        parent_id = f"doc_{doc.metadata['hash']}_parent_{j}"  # 生成父块唯一ID：格式如"doc_哈希值_parent_0"，使用原始doc的hash和索引j。

        parent_doc.metadata["parent_id"] = parent_id  # 在父块的metadata中添加"parent_id"键，值为生成的ID。

        parent_doc.metadata["parent_content"] = parent_doc.page_content  # 在父块的metadata中添加"parent_content"键，值为父块自身的文本内容（用于子块引用）。

        parent_doc.metadata.update(doc.metadata)  # 更新父块的metadata：合并原始doc的metadata（如file_path、hash）。

        sub_chunks = child_splitter_to_use.split_documents([parent_doc])  # 使用选择的子块切分器，将当前父块切分成子块列表。

        for k, sub_chunk in enumerate(sub_chunks):  # 循环遍历每个子块：k是索引，从0开始；sub_chunk是当前子块Document。

            chunk_id = f"{parent_id}_child_{k}"  # 生成子块唯一ID：格式如"doc_哈希值_parent_0_child_0"，基于父ID和k。

            sub_chunk.metadata["parent_id"] = parent_id  # 在子块的metadata中添加"parent_id"键，值为父块ID，用于追溯。

            sub_chunk.metadata["parent_content"] = parent_doc.page_content  # 在子块的metadata中添加"parent_content"键，值为父块完整文本，提供上下文。

            # [修复] 明确添加 'chunk_id' 和 'id'  # 注释：修复部分，确保添加chunk_id和id键。

            sub_chunk.metadata["chunk_id"] = chunk_id  # 在子块的metadata中添加"chunk_id"键，值为生成的子块ID。

            sub_chunk.metadata["id"] = chunk_id  # 在子块的metadata中添加"id"键，值为相同的子块ID（可能用于检索系统的主键）。

            sub_chunk.metadata.update(doc.metadata)  # 更新子块的metadata：合并原始doc的metadata（如file_path、hash）。

            child_chunks.append(sub_chunk)  # 将当前子块添加到总列表child_chunks中。

            logger.debug(f"生成子块: {chunk_id}, 父块: {parent_id}, 内容长度: {len(sub_chunk.page_content)}")  # 调试日志：记录生成的子块ID、父块ID和子块文本长度。

    logger.info(f"文档 {doc.metadata['file_path']} 共生成 {len(child_chunks)} 个子块")  # 记录日志：显示总生成的子块数量。

    return child_chunks  # 返回子块列表：每个元素是Document对象，带有page_content和扩展的metadata。


# 验证process_document函数（基于脚本__main__中的部分）
# 验证load_and_hash_document函数（基于脚本__main__中的部分）
test_file_path = os.path.join(r"D:\LLM_Codes\Chapter3_RAG\SmartRecruit\data\resume\林青霞_test.txt")
# 初始化客户端
client = OpenAI(api_key="sk-b052cdd2b23249f5bbe1949928a2600a", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
content, doc_hash = load_and_hash_document(test_file_path, client)
doc = Document(page_content=content, metadata={"file_path": test_file_path, "hash": doc_hash})
try:
    chunks = process_document(doc)
    assert chunks and isinstance(chunks, list)
    logger.info(f"切块成功: 共生成 {len(chunks)} 个子块。")
    for i,chunk in enumerate(chunks):
        assert chunk and isinstance(chunk, Document)

        logger.info(f"第{i}个子块ID: {chunk.metadata.get('id')}")
except Exception as e:
    logger.error(f"文档切分失败: {e}")