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

            mongo_uri = f"mongodb://{config.MONGO_USER}:{config.MONGO_PASSWORD}@{config.MONGO_HOST}:{config.MONGO_PORT}/{config.MONGO_DB}?authSource=admin"  # 构建MongoDB URI：包含用户名、密码、主机、端口和数据库。
            self.mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)  # 初始化MongoDB客户端：使用URI连接，设置超时为5秒。
            self.mongo_client.admin.command("ping")  # 测试MongoDB连接：发送ping命令验证连接可用性。
            self.mongo_db = self.mongo_client[config.MONGO_DB]  # 获取指定数据库实例。
            self.mongo_collection = self.mongo_db["resumes"]  # 获取集合实例：用于存储简历文档。

            self.es_client = Elasticsearch(config.ES_HOST)  # 初始化Elasticsearch客户端：使用配置的主机地址连接。
            if not self.es_client.indices.exists(index=config.ES_INDEX_NAME):  # 检查索引是否存在：如果不存在，则创建。
                self.es_client.indices.create(index=config.ES_INDEX_NAME)  # 创建Elasticsearch索引：使用配置的索引名称。

            # self._create_or_load_collection()  # 调用私有方法：创建或加载Milvus集合。
            logger.info("向量存储及检索组件初始化完成")  # 记录日志：初始化完成。
        except Exception as e:  # 捕获所有异常：包括连接失败或模型加载错误。
            logger.critical(f"组件初始化失败: {e}", exc_info=True)  # 记录 critical 日志：包含异常信息和栈追踪。
            raise  # 重新抛出异常：允许上层处理。

# 函数验证代码：
# 验证_initialize_components函数（模拟独立测试）
try:
    vector_store = VectorStore()  # 隐式调用_initialize_components
    assert vector_store.client is not None, "Milvus客户端初始化失败"
    assert vector_store.embedding_function is not None, "嵌入函数初始化失败"
    assert vector_store.reranker is not None, "重排模型初始化失败"
    assert vector_store.mongo_client is not None, "MongoDB客户端初始化失败"
    assert vector_store.es_client is not None, "Elasticsearch客户端初始化失败"
    logger.info("组件初始化验证通过！")
except Exception as e:
    logger.error(f"组件初始化失败: {e}")