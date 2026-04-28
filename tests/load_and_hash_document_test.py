from extract_text_from_image_test import extract_text_from_image
from compute_file_hash_test import compute_file_hash
import os
from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader, PyPDFLoader, Docx2txtLoader, \
    UnstructuredPowerPointLoader
from openai import OpenAI
from loguru import logger
# --- 文档加载与处理 ---
document_loaders = {
    ".txt": TextLoader, ".pdf": PyPDFLoader, ".docx": Docx2txtLoader,
    ".ppt": UnstructuredPowerPointLoader, ".pptx": UnstructuredPowerPointLoader,
    ".jpg": None, ".png": None, ".md": UnstructuredMarkdownLoader
}

# 初始化客户端
client = OpenAI(api_key="sk-b052cdd2b23249f5bbe1949928a2600a", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
def load_and_hash_document(file_path: str, client: OpenAI) -> (str, str):
    # 加载文档内容并计算哈希，支持TXT、PDF、DOCX、PPTX、MD、JPG、PNG等格式。
    # 对于图像文件，调用extract_text_from_image提取文本；对于TXT文件，尝试多种编码以避免解码错误。
    # 返回文档内容和哈希值；如果格式不支持或加载失败，抛出异常。
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


# 替换为实际图像路径
# # 验证load_and_hash_document函数（基于脚本__main__中的部分）
# test_file_path = os.path.join(r"D:\LLM_Codes\Chapter3_RAG\SmartRecruit\data\resume\林青霞.txt")
# try:
#     content, doc_hash = load_and_hash_document(test_file_path, client)
#     assert content and doc_hash
#     logger.info(f"加载成功: hash={doc_hash}, 内容长度={len(content)}")
# except Exception as e:
#     logger.error(f"加载与哈希失败: {e}")