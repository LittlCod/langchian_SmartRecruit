
# vector_store.py
import os
import asyncio
from typing import List, Dict, Any
from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from loguru import logger
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from sentence_transformers import CrossEncoder
from elasticsearch import Elasticsearch, NotFoundError

from config import config
from milvus_model.hybrid import BGEM3EmbeddingFunction

# --- 日志与组件初始化 ---
logger.add(os.path.join(config.LOG_DIR, "vector_store.log"), rotation="10 MB", encoding="utf-8")

class VectorStore:
    def __init__(self):
        self._initialize_components()

    def _initialize_components(self):
        # 函数整体注释：初始化所有组件，包括Milvus客户端、嵌入函数、重排模型、MongoDB客户端和Elasticsearch客户端。
        # 加载模型路径，确保组件可用；如果失败，记录日志并抛出异常。
        logger.info("开始初始化向量存储及检索组件...")  # 记录日志：开始初始化过程。
        try:
            self.client = MilvusClient(
                uri=f"http://{config.MILVUS_HOST}:{config.MILVUS_PORT}")  # 初始化Milvus客户端：使用配置的URI连接Milvus服务器。

            embedding_model_path = os.path.join(config.MODEL_PATH, config.EMBEDDING_MODEL)  # 构建嵌入模型路径：从配置中获取模型目录和名称。
            print("embedding_model_path======================")  # 打印嵌入模型路径：用于调试。
            print(embedding_model_path)  # 打印实际路径值。
            self.embedding_function = BGEM3EmbeddingFunction(model_name=embedding_model_path, device='cpu',
                                                             use_fp16=False)  # 初始化BGEM3嵌入函数：加载模型，指定设备为CPU，不使用FP16以确保兼容性。
            self.dense_dim = self.embedding_function.dim["dense"]  # 获取稠密向量维度：从嵌入函数中提取，用于后续schema定义。

            reranker_model_path = os.path.join(config.MODEL_PATH, config.RERANKER_MODEL)  # 构建重排模型路径：从配置中获取。
            print("reranker_model_path======================")  # 打印重排模型路径：用于调试。
            self.reranker = CrossEncoder(reranker_model_path)  # 初始化CrossEncoder重排模型：加载指定路径的模型，用于结果重排序。

            MONGO_DB="test_db_rag"
            mongo_uri = f"mongodb://{config.MONGO_USER}:{config.MONGO_PASSWORD}@{config.MONGO_HOST}:{config.MONGO_PORT}/{config.MONGO_DB}?authSource=admin"  # 构建MongoDB URI：包含用户名、密码、主机、端口和数据库。
            self.mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)  # 初始化MongoDB客户端：使用URI连接，设置超时为5秒。
            self.mongo_client.admin.command("ping")  # 测试MongoDB连接：发送ping命令验证连接可用性。
            self.mongo_db = self.mongo_client[MONGO_DB]  # 获取指定数据库实例。
            self.mongo_collection = self.mongo_db["resumes"]  # 获取集合实例：用于存储简历文档。

            self.es_client = Elasticsearch(config.ES_HOST)  # 初始化Elasticsearch客户端：使用配置的主机地址连接。
            if not self.es_client.indices.exists(index=config.ES_INDEX_NAME):  # 检查索引是否存在：如果不存在，则创建。
                self.es_client.indices.create(index=config.ES_INDEX_NAME)  # 创建Elasticsearch索引：使用配置的索引名称。

            self._create_or_load_collection()  # 调用私有方法：创建或加载Milvus集合。
            logger.info("向量存储及检索组件初始化完成")  # 记录日志：初始化完成。
        except Exception as e:  # 捕获所有异常：包括连接失败或模型加载错误。
            logger.critical(f"组件初始化失败: {e}", exc_info=True)  # 记录 critical 日志：包含异常信息和栈追踪。
            raise  # 重新抛出异常：允许上层处理。

    def _create_or_load_collection(self):
        # 函数整体注释：创建或加载Milvus集合，定义schema、字段和索引参数。
        # 如果集合不存在，则创建；否则直接加载，确保集合可用。
        collection_name = "test_collect_rag"  # 获取集合名称：从配置中读取。
        if not self.client.has_collection(collection_name):  # 检查集合是否存在：如果不存在，则创建。
            schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)  # 创建schema：禁用自动ID，启用动态字段。
            schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=100)  # 添加主键字段：id，VARCHAR类型，长度100。
            schema.add_field("dense_vector", DataType.FLOAT_VECTOR,
                             dim=self.dense_dim)  # 添加稠密向量字段：FLOAT_VECTOR，维度从嵌入函数获取。
            schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)  # 添加稀疏向量字段：SPARSE_FLOAT_VECTOR，用于混合搜索。
            schema.add_field("doc_hash", DataType.VARCHAR, max_length=32)  # 添加文档哈希字段：VARCHAR，长度32，用于唯一标识。
            schema.add_field("text", DataType.VARCHAR, max_length=65535)  # 添加文本字段：VARCHAR，最大长度65535，用于存储块内容。
            schema.add_field("gender", DataType.VARCHAR, max_length=10)  # 添加性别字段：VARCHAR，长度10，用于过滤。
            schema.add_field("age", DataType.INT64)  # 添加年龄字段：INT64，用于范围过滤。
            schema.add_field("work_experience", DataType.INT64)  # 添加工作年限字段：INT64，用于范围过滤。

            index_params = self.client.prepare_index_params()  # 准备索引参数：用于定义向量索引。
            index_params.add_index(field_name="dense_vector", index_name="dense_index", index_type="IVF_FLAT",
                                   metric_type="IP", params={"nlist": 128})  # 添加稠密向量索引：IVF_FLAT类型，IP度量，nlist=128。
            index_params.add_index(field_name="sparse_vector", index_name="sparse_index",
                                   index_type="SPARSE_INVERTED_INDEX", metric_type="IP", params={
                    "drop_ratio_build": 0.2})  # 添加稀疏向量索引：SPARSE_INVERTED_INDEX类型，IP度量，drop_ratio_build=0.2。

            self.client.create_collection(collection_name=collection_name, schema=schema,
                                          index_params=index_params)  # 创建集合：使用schema和index_params。
        self.client.load_collection(collection_name)  # 加载集合：确保集合在内存中可用，用于查询。

    def store_resume(self, doc: Document, chunks: List[Document], structured_data: Dict[str, Any]) -> bool:
        # 函数整体注释：存储简历文档到Milvus、Elasticsearch和MongoDB。
        # 先检查是否重复（基于doc_hash），然后生成嵌入，插入数据；如果成功返回True，否则False或抛出异常。
        doc_hash = doc.metadata["hash"]  # 从doc元数据中提取哈希值：用于唯一标识和重复检查。
        if self.mongo_collection.find_one({"doc_hash": doc_hash}):  # 检查MongoDB中是否已存在：如果存在，返回False避免重复存储。
            return False
        if not chunks:  # 检查chunks是否为空：如果为空，返回False。
            return False

        texts = [chunk.page_content for chunk in chunks]  # 提取所有块的文本内容：用于生成嵌入。
        embeddings = self.embedding_function.encode_documents(texts)  # 生成嵌入：使用BGEM3函数编码文档，返回稠密和稀疏向量。

        data_to_insert = []  # 初始化插入数据列表：用于Milvus插入。
        for idx, chunk in enumerate(chunks):  # 循环遍历每个块：idx是索引，chunk是Document。
            sparse_indices = embeddings["sparse"].indices[embeddings["sparse"].indptr[idx]:embeddings["sparse"].indptr[idx + 1]]  # 提取当前块的稀疏向量索引：从CSR矩阵中切片。
            sparse_data = embeddings["sparse"].data[embeddings["sparse"].indptr[idx]:embeddings["sparse"].indptr[idx + 1]]  # 提取当前块的稀疏向量数据：从CSR矩阵中切片。
            sparse_vector = {int(k): float(v) for k, v in zip(sparse_indices, sparse_data)}  # 构建稀疏向量字典：键为整数索引，值为浮点数据。

            chunk_data = {  # 构建单个块的数据字典：包含ID、文本、向量、哈希和结构化数据。
                "id": chunk.metadata["id"],
                "text": chunk.page_content,
                "dense_vector": embeddings["dense"][idx].tolist(),
                "sparse_vector": sparse_vector,
                "doc_hash": doc_hash,
                **structured_data
            }
            data_to_insert.append(chunk_data)  # 添加到插入列表。

        try:  # 尝试插入数据：到Milvus、ES和MongoDB。
            collection_name = "test_collect_rag"
            ES_INDEX_NAME="test_index_rag"
            self.client.insert(collection_name, data_to_insert)  # 插入到Milvus：使用集合名称和数据列表。
            for chunk in chunks:  # 循环遍历每个块：插入到Elasticsearch。
                es_doc = {"content": chunk.page_content, "metadata": chunk.metadata}  # 构建ES文档：包含内容和元数据。
                self.es_client.index(index="test_index_rag", id=chunk.metadata["id"],
                                     document=es_doc)  # 索引到ES：使用块ID作为文档ID。
            mongo_doc = {"doc_hash": doc_hash, "content": doc.page_content, "metadata": doc.metadata,
                         "structured_data": structured_data}  # 构建MongoDB文档：包含哈希、完整内容、元数据和结构化数据。
            self.mongo_collection.insert_one(mongo_doc)  # 插入到MongoDB：单个文档。
            logger.info(f"成功存储简历到Milvus, ES和MongoDB, hash: {doc_hash}")  # 记录日志：存储成功。
            return True  # 返回True：表示存储成功。
        except Exception as e:  # 捕获异常：包括插入失败。
            logger.error(f"存储简历失败: {e}", exc_info=True)  # 记录错误日志：包含异常信息。
            raise  # 重新抛出异常# 。


    def get_metadata_by_hash(self, doc_hash: str) -> dict:
        # 函数整体注释：根据文档哈希从MongoDB获取元数据。
        # 如果找到，返回字典（去除_id）；否则返回空字典。
        result = self.mongo_collection.find_one({"doc_hash": doc_hash})  # 查询MongoDB：使用doc_hash过滤，find_one返回单个文档。
        if result:  # 如果结果存在：处理结果。
            result.pop('_id', None)  # 移除MongoDB自动生成的_id字段：确保返回纯数据。
        return result or {}  # 返回结果字典或空字典。


    def get_full_resume(self, doc_hash: str) -> str:
        """从MongoDB获取完整简历文本"""
        # 函数整体注释：根据哈希从MongoDB获取完整简历内容。
        # 如果找到，返回content字符串；否则返回空字符串或抛出异常。
        try:  # 尝试查询：捕获异常。
            result = self.mongo_collection.find_one({"doc_hash": doc_hash})  # 查询MongoDB：使用doc_hash过滤。
            if result:  # 如果结果存在：提取content。
                logger.info(f"成功获取完整简历: hash {doc_hash}")  # 记录日志：获取成功。
                return result["content"]  # 返回完整内容字符串。
            logger.warning(f"未找到完整简历: hash {doc_hash}")  # 记录警告：未找到。
            return ""  # 返回空字符串。
        except Exception as e:  # 捕获异常：如查询失败。
            logger.error(f"获取完整简历失败: hash {doc_hash}, 错误: {str(e)}")  # 记录错误日志。
            raise  # 重新抛出异常。


# 验证get_full_resume函数
vector_store = VectorStore()
test_hash = "test_hash"  # 假设已存储
try:
    content = vector_store.get_full_resume(test_hash)
    print(f"完整简历内容: {content}")
    assert isinstance(content, str), "返回不是字符串"
    if content:
        logger.info(f"完整简历获取成功: 长度 {len(content)}")
    else:
        logger.warning("未找到完整简历")
except Exception as e:
    logger.error(f"完整简历获取失败: {e}")