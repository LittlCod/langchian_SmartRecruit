# vector_store.py
import os
import asyncio
from typing import List, Dict, Any

from modelscope.pipelines.multi_modal.diffusers_wrapped.devices import device
from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker
from langchain_core.documents import Document
from loguru import logger
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from sentence_transformers import CrossEncoder
from elasticsearch import Elasticsearch, NotFoundError

from config import config
from milvus_model.hybrid import BGEM3EmbeddingFunction
from tests.vector_3_store_resume_csr import sparse_indices

# --- 步骤 1: 日志与组件初始化 ---
# 1.1 配置日志记录器，指定日志文件路径、最大大小和编码
logger.add(os.path.join(config.LOG_DIR, "vector_store.log"), rotation="10 MB", encoding="utf-8")


# 定义VectorStore类，用于管理向量存储、检索和相关组件
class VectorStore:
    def __init__(self):
        # 调用组件初始化方法,设置实例属性
        self._initialize_components()

    def _initialize_components(self):
        # 记录初始化开始的日志
        logger.info("开始初始化向量存储及检索组件...")
        try:
            # 初始化Milvus客户端,连接到指定的Milvus服务地址
            self.client = MilvusClient(uri=f"http://{config.MILVUS_HOST}:{config.MILVUS_PORT}")

            # 初始化 embedding model
            embedding_model_path = os.path.join(config.MODEL_PATH, config.EMBEDDING_MODEL)
            self.embedding_function = BGEM3EmbeddingFunction(model_name=embedding_model_path, device="cpu", use_fp16=False)
            # 获取嵌入函数的稠密向量维度
            self.dense_dim = self.embedding_function.dim['dense']

            # 初始化 reranker model
            reranker_model_path = os.path.join(config.MODEL_PATH, config.RERANKER_MODEL)
            self.reranker = CrossEncoder(reranker_model_path)

            # 初始化 mongo
            self.mongo_client = MongoClient(
                host=config.MONGO_HOST,
                port=config.MONGO_PORT,
                username=config.MONGO_USER,
                password=config.MONGO_PASSWORD,
                authSource=config.MONGO_USER
            )
            self.mongo_db = self.mongo_client[config.MONGO_DB]
            self.mongo_collection = self.mongo_db[config.MONGO_COLLECTION]

            # 初始化 es
            self.es_client = Elasticsearch(config.ES_HOST)
            # 检查Elasticsearch索引是否存在,若不存在则创建
            if not self.es_client.indices.exists(index=config.ES_INDEX_NAME):
                self.es_client.indices.create(index=config.ES_INDEX_NAME)

            # 调用方法创建或加载Milvus集合
            self._create_or_load_collection()

            # 记录初始化成功的日志
            logger.info("向量存储及检索组件初始化完成")

        except Exception as e:
            # 2.19 捕获异常并记录初始化失败的日志
            logger.critical(f"组件初始化失败: {e}", exc_info=True)
            # 2.20 抛出异常,终止初始化
            raise

    # 创建或加载 Milvus 集合
    def _create_or_load_collection(self):
        # 3.1 获取集合名称
        collection_name = config.MILVUS_COLLECTION_NAME
        # 3.2 检查集合是否存在
        if not self.client.has_collection(collection_name):
            # 3.3 创建Milvus集合的schema,禁用自动ID,启用动态字段
            schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
            # 3.4 添加ID字段,主键,VARCHAR类型,最大长度100
            schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=100)
            # 3.5 添加稠密向量字段,FLOAT_VECTOR类型,维度由dense_dim指定
            schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=self.dense_dim)
            # 3.6 添加稀疏向量字段,SPARSE_FLOAT_VECTOR类型
            schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
            # 3.7 添加文档哈希字段,VARCHAR类型,最大长度32
            schema.add_field("doc_hash", DataType.VARCHAR, max_length=32)
            # 3.8 添加文本字段,VARCHAR类型,最大长度65535
            schema.add_field("text", DataType.VARCHAR, max_length=65535)
            # 3.9 添加性别字段,VARCHAR类型,最大长度10
            schema.add_field("gender", DataType.VARCHAR, max_length=10)
            # 3.10 添加年龄字段,INT64类型
            schema.add_field("age", DataType.INT64)
            # 3.11 添加工作经验字段,INT64类型
            schema.add_field("work_experience", DataType.INT64)

            # 3.12 创建索引参数对象
            index_params = self.client.prepare_index_params()
            # 3.13 为稠密向量字段添加索引,类型为IVF_FLAT(一种倒排索引,将向量空间划分为多个聚类区域(Voronoi 单元),检索时只搜索与查询向量最近的几个聚类,避免全量扫描),
            # 距离度量为IP(Inner Product内积,用于衡量两个向量的相似度)当向量都已归一化时,内积等价于余弦相似度,数值越大表示越相似
            # nlist:IVF 索引中聚类中心的数量(也就是将向量空间划分成多少个簇),128 是一个常见默认值
            index_params.add_index(field_name="dense_vector", index_name="dense_index", index_type="IVF_FLAT",
                                   metric_type="IP", params={"nlist": 128})
            # 3.14 为稀疏向量字段添加索引,类型为SPARSE_INVERTED_INDEX(专为高维稀疏向量(如词袋、BM25、BGE-M3 生成的稀疏向量)设计的倒排索引结构)
            # drop_ratio_build:构建稀疏索引时丢弃后 20% 的低权重维度 - 减少索引体积,提升检索速度
            index_params.add_index(field_name="sparse_vector", index_name="sparse_index",
                                   index_type="SPARSE_INVERTED_INDEX", metric_type="IP",
                                   params={"drop_ratio_build": 0.2})

            # 3.15 创建Milvus集合,使用定义的schema和索引参数
            self.client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)

        self.client.load_collection(collection_name)

    # 将简历数据分别写入三个数据库(Milvus + Elasticsearch + MongoDB)
    def store_resume(self, doc: Document, chunks: List[Document], structured_data: Dict[str, Any]) -> bool:
        """
        将简历数据分别写入三个数据库(Milvus + Elasticsearch + MongoDB)。

        Args:
            doc: 完整简历 Document 对象,doc.page_content 是全文,doc.metadata["hash"] 是 MD5 去重标识
            chunks: 切分后的子块列表,每个子块有 page_content(文本)、metadata["id"](唯一标识)、metadata["parent_content"](父块文本)
            structured_data: 结构化提取的四个字段(name/gender/age/work_experience),作为 Milvus 标量字段存入

        Returns:
            bool: 存储成功返回 True,简历已存在或 chunks 为空返回 False
        """
        # 使用hash进行去重判断
        doc_hash = doc.metadata["hash"]  # 文件的hash
        if self.mongo_collection.find_one({"doc_hash": doc_hash}):
            logger.info("文件已存在")
            return False
        if not chunks:
            logger.info("子块为空")
            return False

        # 向量化
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.embedding_function.encode_documents(texts)

        # 向 milvus 中插入数据
        data_to_insert = []
        for idx, chunk in enumerate(chunks):
            # 获取稀疏向量的索引和数据
            sparse_indices = embeddings["sparse"].indices[
                embeddings["sparse"].indptr[idx] : embeddings["sparse"].indptr[idx + 1]
            ]
            sparse_data = embeddings["sparse"].data[
                embeddings["sparse"].indptr[idx] : embeddings["sparse"].indptr[idx + 1]
            ]
            # 构造稀疏向量字典
            sparse_vector = {int(k): float(v) for k, v in zip(sparse_indices, sparse_data)}

            # 构建存入 milvus 的数据
            chunk_data = {
                "id": chunk.metadata["id"],
                "text": chunk.page_content,
                "dense_vector": embeddings["dense"][idx].tolist(),
                "sparse_vector": sparse_vector,
                "doc_hash": doc_hash,
                "parent_content": chunk.metadata["parent_content"],
                **structured_data
            }
            data_to_insert.append(chunk_data)

        # 插入数据
        try:
            # 向 milvus 插入
            self.client.insert(config.MILVUS_COLLECTION_NAME, data_to_insert)

            # 向 es 插入
            for chunk in chunks:
                es_doc = {
                    "content": chunk.page_content,
                    "metadata": chunk.metadata
                }
                self.es_client.index(index=config.ES_INDEX_NAME, id=chunk.id, document=es_doc)

            # 向 mongo 插入
            mongo_doc = {
                "doc_hash": doc_hash,
                "content": doc.page_content,
                "metadata": doc.metadata,
                "structured_data": structured_data
            }
            self.mongo_collection.insert_one(mongo_doc)

            logger.info(f"成功存储简历到Milvus, ES和MongoDB, hash: {doc_hash}")
            return True

        except Exception as e:
            logger.error(f"存入简历失败：{e}", exc_info=True)
            raise e

    # 从MongoDB查询指定哈希值的文档
    def get_metadata_by_hash(self, doc_hash: str) -> dict:
        # 5.1 从MongoDB查询指定哈希值的文档
        result = self.mongo_collection.find_one({"doc_hash": doc_hash})
        # 5.2 如果查询到结果,移除MongoDB自动生成的_id字段
        if result:
            result.pop('_id', None)
        # 5.3 返回查询结果,若无结果返回空字典
        return result or {}

    # 根据哈希从MongoDB获取完整简历文本
    def get_full_resume(self, doc_hash: str) -> str:
        # 6.1 从MongoDB获取完整简历文本
        """从MongoDB获取完整简历文本"""
        try:
            # 6.2 从MongoDB查询指定哈希值的文档
            result = self.mongo_collection.find_one({"doc_hash": doc_hash})
            # 6.3 如果查询到结果,记录成功日志并返回内容
            if result:
                logger.info(f"成功获取完整简历: hash {doc_hash}")
                return result["content"]
            # 6.4 如果未找到,记录警告日志并返回空字符串
            logger.warning(f"未找到完整简历: hash {doc_hash}")
            return ""
        except Exception as e:
            # 6.5 捕获异常,记录错误日志并抛出
            logger.error(f"获取完整简历失败: hash {doc_hash}, 错误: {str(e)}")
            raise

    # 执行混合检索+重排
    def hybrid_search_with_rerank(self, query: str, params: Dict[str, Any]) -> List[Document]:
        """
        执行混合检索 + 重排序:
        1. Milvus 混合搜索(稠密 0.7 + 稀疏 0.3),支持元数据精确过滤
        2. Elasticsearch BM25 关键词搜索,补充召回
        3. 两路结果合并去重
        4. BGE-Reranker 精排打分
        5. 按 doc_hash 去重,从 MongoDB 取完整简历返回

        Args:
            query: 用户查询文本,如 "需要AI大模型产品经理"
            params: 检索参数字典,支持以下字段:
                - count (int): 返回结果数量
                - gender (str): 性别过滤,"未提供" 表示不过滤
                - age_min / age_max (int): 年龄范围
                - experience_min / experience_max (int): 工作经验范围

        Returns:
            List[Document]: 去重后的完整简历列表,按 rerank_score 降序排列
        """
        # 设置初始检索数量k为count的x倍
        k = params.get('count', 1) * 1
        # 设置最终返回数量m为count
        m = params.get('count', 1)

        # ------------ Milvus 查询 ------------
        # Milvus 过滤条件列表
        filter_conditions = []
        # 性别
        if params.get('gender') and params['gender'] != '未提供':
            filter_conditions.append(f"gender == '{params['gender']}'")
        # 最小和最大年龄
        if params.get('age_min') is not None:
            filter_conditions.append(f"age >= {params['age_min']}")
        if params.get('age_max') is not None:
            filter_conditions.append(f"age <= {params['age_max']}")
        # 最小和最大工作经验年限
        if params.get('experience_min') is not None:
            filter_conditions.append(f"work_experience >= {params['experience_min']}")
        if params.get('experience_max') is not None:
            filter_conditions.append(f"work_experience <= {params['experience_max']}")
        # 最终拼接
        filter_expr = " and ".join(filter_conditions)

        # 混合检索开始
        logger.info(f"开始混合检索: query='{query}', m={m}, filter='{filter_expr}'")

        # 得到查询向量
        query_embedding = self.embedding_function.encode_queries([query])

        # 查询的稠密向量
        dense_vector = query_embedding["dense"][0]
        # 查询的稀疏向量
        sparse_vector = { int(k): float(v) for k, v in zip(
            query_embedding["sparse"].indices, query_embedding["sparse"].data
        )}

        # 构建稠密向量查询请求
        dense_req = AnnSearchRequest(
            data=[dense_vector],
            anns_field="dense_vector",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=k,
            expr=filter_expr
        )
        # 构建稀疏向量查询请求
        sparse_req = AnnSearchRequest(
            data=[sparse_vector],
            anns_field="sparse_vector",
            param={"metric_type": "IP"},
            limit=k,
            expr=filter_expr
        )

        # milvus 双路查询
        milvus_results = self.client.hybrid_search(
            collection_name=config.MILVUS_COLLECTION_NAME,
            reqs=[dense_req, sparse_req],
            ranker=WeightedRanker(0.7, 0.3),
            limit=k,
            filter=filter_expr,
            output_fields=['id', "work_experience", "age", "name", "doc_hash", "text", "parent_content", "gender"]
        )
        milvus_results = milvus_results[0]  # 只有一组ranker
        logger.info(f"Milvus召回了 {len(milvus_results)} 个结果。")
        # ------------ Milvus 查询结束 ------------

        # ------------ es 查询 ------------
        es_body = {
            "query": {
                "match": {"content": query}
            },
            "size": k,
        }
        es_results = self.es_client.search(index=config.ES_INDEX_NAME, body=es_body)["hits"]["hits"]
        logger.info(f"ES召回了 {len(es_results)} 个结果。")
        # ------------ es 查询结束 ------------

        # ------------ 去重+reranker ------------
        # 合并Milvus搜索结果到字典
        all_hits = {hit['id']: hit['entity'] for hit in milvus_results}
        # 合并Elasticsearch结果,补充未在Milvus中找到的文档
        for hit in es_results:
            if hit['_id'] not in all_hits:
                mongo_data = self.get_metadata_by_hash(hit['_source'].get('metadata', {}).get('hash'))
                if mongo_data:
                    all_hits[hit['_id']] = {**hit['_source']['metadata'], **mongo_data.get('structured_data', {})}
        # 如果没有检索到任何结果,返回空列表
        if not all_hits: return []

        # 构建 reranker 需要的 pairs
        pairs = [[query, entity.get('text', entity.get('parent_content', ''))] for entity in all_hits.values()]
        # 调用 reranker 模型
        scores = self.reranker.predict(pairs)

        # 构建新的文档列表
        docs_with_scores = [
            Document(page_content=entity.get("text", ""), metadata={**entity, "rerank_score": score})
                for entity, score in zip(all_hits.values(), scores)
        ]
        # 排序
        docs_with_scores.sort(key=lambda x: x.metadata["rerank_score"], reverse=True)

        # 最终根据 hash 去重文档
        final_docs = []
        parent_hashes = set()
        # 遍历排序后的文档,去重并获取完整简历
        for doc in docs_with_scores:
            doc_hash = doc.metadata.get('doc_hash')
            if doc_hash and doc_hash not in parent_hashes: # 判断是否在set中，去重
                # 根据哈希值获取MongoDB中的完整文档
                mongo_doc = self.get_metadata_by_hash(doc_hash)
                if mongo_doc:
                    # 合并元数据和重排得分
                    final_metadata = mongo_doc.get('metadata', {})
                    final_metadata['rerank_score'] = doc.metadata.get('rerank_score')
                    final_metadata['doc_hash'] = doc_hash

                    # 构造最终文档对象
                    final_docs.append(Document(
                        page_content=mongo_doc.get('content', ''),
                        metadata=final_metadata
                    ))
                    # 将哈希值添加到去重集合
                    parent_hashes.add(doc_hash)
            # 如果达到返回数量限制,终止循环
            if len(final_docs) >= m: break

        # 记录重排和去重后的结果数量
        logger.info(f"重排和去重后,返回 {len(final_docs)} 份独立简历。")
        # 返回最终文档列表
        return final_docs

    # 异步包装
    async def aget_relevant_documents(self, query: str, params: Dict[str, Any]) -> List[Document]:
        # 8.1 将同步的混合检索方法包装为异步调用
        return await asyncio.to_thread(self.hybrid_search_with_rerank, query, params)


if __name__ == '__main__':
    """
    独立验证VectorStore的核心功能，使用一份真实的简历文件进行端到端测试。
    """
    # 开始验证核心功能
    logger.info("=" * 50)
    logger.info("开始独立验证 vector_store.py 模块...")
    # step1:验证【构造函数与初始化】【创建或加载 Milvus 集合】:实例化VectorStore
    vector_store = VectorStore()

    # 验证集合字段结构
    print(f"集合名称: {config.MILVUS_COLLECTION_NAME}")
    fields_info = vector_store.client.describe_collection(config.MILVUS_COLLECTION_NAME)['fields']
    for field in fields_info:
        print(f"  字段: {field['name']}, 类型: {field['type']}")

    # step1 验证到此结束
    # 如果没抛异常,说明 Milvus/MongoDB/ES 三个数据库已连接,集合已加载

    # step2:验证【存储简历】
    try:
        from utils.document_processor import (
            load_and_hash_document,
            parse_resume_structure,
            process_document,
            parser_client
        )
        from langchain_core.documents import Document

        test_file_path = os.path.join(config.LOCAL_RESUME_DIR, "李明AI大模型产品经理简历.pdf")
        if not os.path.exists(test_file_path):
            raise FileNotFoundError(f"测试文件不存在: {test_file_path}")

        # 加载文档、提取结构化信息、切块
        content, doc_hash = load_and_hash_document(test_file_path, parser_client)
        structured_data = parse_resume_structure(content, parser_client)
        doc = Document(page_content=content,
                       metadata={"file_path": test_file_path, "hash": doc_hash, **structured_data})
        chunks = process_document(doc)
        print(f"切块数量: {len(chunks)}, 结构化信息: {structured_data}")

        # 清理旧数据(确保可重复执行)
        vector_store.mongo_collection.delete_one({"doc_hash": doc_hash})
        vector_store.client.delete(collection_name=config.MILVUS_COLLECTION_NAME, filter=f"doc_hash == '{doc_hash}'")

        # 执行存储
        success = vector_store.store_resume(doc, chunks, structured_data)
        assert success, "存储简历失败!"
        print("store_resume 验证通过!")
    except Exception as e:
        logger.error(f"简历存储测试失败！错误信息: {e}")

    # step3: 验证【混合检索 + 重排】
    query = "需要AI大模型产品经理"
    params = {
        "count": 1,
        "gender": "未提供",
        "experience_min": 3,
    }
    results = vector_store.hybrid_search_with_rerank(query, params)
    assert len(results) > 0, "检索结果为空!"
    first = results[0]
    print(f"检索到 {len(results)} 个结果")
    print(f"内容: {first.page_content[:200]}...")
    print(f"重排分数: {first.metadata.get('rerank_score', 'N/A')}")
    print(f"元数据: {first.metadata}")
    print("hybrid_search_with_rerank 验证通过!")
    print("hybrid_search_with_rerank 函数验证通过！")