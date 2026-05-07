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
        self.client = MilvusClient(uri=f"http://{config.MILVUS_HOST}:{config.MILVUS_PORT}")
        self.dense_dim=1000
        self.collection_name="test_collect_rag"
        pass
    def _create_or_load_collection(self):
        # 函数整体注释：创建或加载Milvus集合，定义schema、字段和索引参数。
        # 如果集合不存在，则创建；否则直接加载，确保集合可用。
        collection_name = self.collection_name  # 获取集合名称：从配置中读取。
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

# 函数验证代码：
# 验证_create_or_load_collection函数（在VectorStore初始化后测试）
vector_store = VectorStore()  # 确保初始化
try:
    # 手动调用以验证（实际在初始化中调用）
    vector_store._create_or_load_collection()
    assert vector_store.client.has_collection(config.MILVUS_COLLECTION_NAME), "集合创建或加载失败"
    logger.info("集合创建/加载验证通过！")
except Exception as e:
    logger.error(f"集合创建/加载失败: {e}")