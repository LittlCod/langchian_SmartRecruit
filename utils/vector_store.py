# vector_store.py
import os
import asyncio
from typing import List, Dict, Any
from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker
from langchain_core.documents import Document
from loguru import logger
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from sentence_transformers import CrossEncoder
from elasticsearch import Elasticsearch, NotFoundError

from config import config
from milvus_model.hybrid import BGEM3EmbeddingFunction

# --- 步骤 1: 日志与组件初始化 ---
# 1.1 配置日志记录器，指定日志文件路径、最大大小和编码
logger.add(os.path.join(config.LOG_DIR, "vector_store.log"), rotation="10 MB", encoding="utf-8")


# 定义VectorStore类，用于管理向量存储、检索和相关组件
class VectorStore:
    def __init__(self):
        # 1.2 调用组件初始化方法，设置实例属性
        self._initialize_components()

    def _initialize_components(self):
        # 步骤 2: 初始化所有必要组件
        # 2.1 记录初始化开始的日志
        logger.info("开始初始化向量存储及检索组件...")
        try:
            # 2.2 初始化Milvus客户端，连接到指定的Milvus服务地址
            self.client = MilvusClient(uri=f"http://{config.MILVUS_HOST}:{config.MILVUS_PORT}")

            # 2.3 构造嵌入模型的路径
            embedding_model_path = os.path.join(config.MODEL_PATH, config.EMBEDDING_MODEL)
            # 2.4 打印嵌入模型路径以便调试
            print(f"embedding_model_path=={embedding_model_path}")
            # 2.5 初始化BGEM3嵌入函数，指定模型路径、设备和浮点精度
            self.embedding_function = BGEM3EmbeddingFunction(model_name=embedding_model_path, device='cpu',
                                                             use_fp16=False)
            # 2.6 获取嵌入函数的稠密向量维度
            self.dense_dim = self.embedding_function.dim["dense"]

            # 2.7 构造重排模型的路径
            reranker_model_path = os.path.join(config.MODEL_PATH, config.RERANKER_MODEL)
            # 2.8 打印重排模型路径以便调试
            print(f"reranker_model_path=={reranker_model_path}")
            # 2.9 初始化CrossEncoder重排模型
            self.reranker = CrossEncoder(reranker_model_path)

            # 2.10 构造MongoDB连接URI
            mongo_uri = f"mongodb://{config.MONGO_USER}:{config.MONGO_PASSWORD}@{config.MONGO_HOST}:{config.MONGO_PORT}/{config.MONGO_DB}?authSource=admin"
            # 2.11 初始化MongoDB客户端，设置连接超时时间
            self.mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
            # 2.12 测试MongoDB连接是否有效
            self.mongo_client.admin.command("ping")
            # 2.13 获取MongoDB数据库实例
            self.mongo_db = self.mongo_client[config.MONGO_DB]
            # 2.14 获取MongoDB简历集合
            self.mongo_collection = self.mongo_db["resumes"]

            # 2.15 初始化Elasticsearch客户端
            self.es_client = Elasticsearch(config.ES_HOST)
            # 2.16 检查Elasticsearch索引是否存在，若不存在则创建
            if not self.es_client.indices.exists(index=config.ES_INDEX_NAME):
                self.es_client.indices.create(index=config.ES_INDEX_NAME)

            # 2.17 调用方法创建或加载Milvus集合
            self._create_or_load_collection()
            # 2.18 记录初始化成功的日志
            logger.info("向量存储及检索组件初始化完成")
        except Exception as e:
            # 2.19 捕获异常并记录初始化失败的日志
            logger.critical(f"组件初始化失败: {e}", exc_info=True)
            # 2.20 抛出异常，终止初始化
            raise

    def _create_or_load_collection(self):
        # 步骤 3: 创建或加载Milvus集合
        # 3.1 获取集合名称
        collection_name = config.MILVUS_COLLECTION_NAME
        # 3.2 检查集合是否存在
        if not self.client.has_collection(collection_name):
            # 3.3 创建Milvus集合的schema，禁用自动ID，启用动态字段
            schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
            # 3.4 添加ID字段，主键，VARCHAR类型，最大长度100
            schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=100)
            # 3.5 添加稠密向量字段，FLOAT_VECTOR类型，维度由dense_dim指定
            schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=self.dense_dim)
            # 3.6 添加稀疏向量字段，SPARSE_FLOAT_VECTOR类型
            schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
            # 3.7 添加文档哈希字段，VARCHAR类型，最大长度32
            schema.add_field("doc_hash", DataType.VARCHAR, max_length=32)
            # 3.8 添加文本字段，VARCHAR类型，最大长度65535
            schema.add_field("text", DataType.VARCHAR, max_length=65535)
            # 3.9 添加性别字段，VARCHAR类型，最大长度10
            schema.add_field("gender", DataType.VARCHAR, max_length=10)
            # 3.10 添加年龄字段，INT64类型
            schema.add_field("age", DataType.INT64)
            # 3.11 添加工作经验字段，INT64类型
            schema.add_field("work_experience", DataType.INT64)

            # 3.12 创建索引参数对象
            index_params = self.client.prepare_index_params()
            # 3.13 为稠密向量字段添加索引，类型为IVF_FLAT，距离度量为IP
            index_params.add_index(field_name="dense_vector", index_name="dense_index", index_type="IVF_FLAT",
                                   metric_type="IP", params={"nlist": 128}) #
            # 3.14 为稀疏向量字段添加索引，类型为SPARSE_INVERTED_INDEX，距离度量为IP
            index_params.add_index(field_name="sparse_vector", index_name="sparse_index",
                                   index_type="SPARSE_INVERTED_INDEX", metric_type="IP",
                                   params={"drop_ratio_build": 0.2})

            # 3.15 创建Milvus集合，使用定义的schema和索引参数
            self.client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)
        # 3.16 加载集合到内存
        self.client.load_collection(collection_name)

    def store_resume(self, doc: Document, chunks: List[Document], structured_data: Dict[str, Any]) -> bool:
        # 步骤 4: 存储简历数据
        # 4.1 获取文档的哈希值
        doc_hash = doc.metadata["hash"]
        # 4.2 检查MongoDB中是否已存在该哈希值的文档
        if self.mongo_collection.find_one({"doc_hash": doc_hash}):
            # 4.3 若存在，返回False表示存储失败
            return False
        # 4.4 检查输入的文档块是否为空
        if not chunks:
            # 4.5 若为空，返回False表示存储失败
            return False

        # 4.6 提取所有文档块的文本内容
        texts = [chunk.page_content for chunk in chunks]
        # 4.7 使用嵌入函数为文本生成嵌入向量
        embeddings = self.embedding_function.encode_documents(texts)

        # 4.8 初始化用于插入Milvus的数据列表
        data_to_insert = []
        # 4.9 遍历每个文档块，构造插入数据
        for idx, chunk in enumerate(chunks):
            # 4.10 获取稀疏向量的索引和数据
            sparse_indices = embeddings["sparse"].indices[
                             embeddings["sparse"].indptr[idx]:embeddings["sparse"].indptr[idx + 1]]
            sparse_data = embeddings["sparse"].data[
                          embeddings["sparse"].indptr[idx]:embeddings["sparse"].indptr[idx + 1]]
            # 4.11 构造稀疏向量字典
            sparse_vector = {int(k): float(v) for k, v in zip(sparse_indices, sparse_data)}

            # 4.12 构造单个文档块的数据，包括ID、文本、向量等
            chunk_data = {
                "id": chunk.metadata["id"],
                "text": chunk.page_content,
                "dense_vector": embeddings["dense"][idx].tolist(),
                "sparse_vector": sparse_vector,
                "doc_hash": doc_hash,
                **structured_data,
                "parent_content": chunk.metadata.get("parent_content", "")
            }
            # 4.13 将数据添加到插入列表
            data_to_insert.append(chunk_data)

        try:
            # 4.14 将数据插入Milvus集合
            self.client.insert(config.MILVUS_COLLECTION_NAME, data_to_insert)
            # 4.15 遍历文档块，将每个块索引到Elasticsearch
            for chunk in chunks:
                es_doc = {"content": chunk.page_content, "metadata": chunk.metadata}
                self.es_client.index(index=config.ES_INDEX_NAME, id=chunk.metadata["id"], document=es_doc)
            # 4.16 构造MongoDB文档，包括哈希、内容、元数据和结构化数据
            mongo_doc = {"doc_hash": doc_hash, "content": doc.page_content, "metadata": doc.metadata,
                         "structured_data": structured_data}
            # 4.17 插入MongoDB文档
            self.mongo_collection.insert_one(mongo_doc)
            # 4.18 记录存储成功的日志
            logger.info(f"成功存储简历到Milvus, ES和MongoDB, hash: {doc_hash}")
            # 4.19 返回True表示存储成功
            return True
        except Exception as e:
            # 4.20 捕获异常并记录存储失败的日志
            logger.error(f"存储简历失败: {e}", exc_info=True)
            # 4.21 抛出异常，终止存储操作
            raise

    def get_metadata_by_hash(self, doc_hash: str) -> dict:
        # 步骤 5: 根据哈希值获取元数据
        # 5.1 从MongoDB查询指定哈希值的文档
        result = self.mongo_collection.find_one({"doc_hash": doc_hash})
        # 5.2 如果查询到结果，移除MongoDB自动生成的_id字段
        if result:
            result.pop('_id', None)
        # 5.3 返回查询结果，若无结果返回空字典
        return result or {}

    def get_full_resume(self, doc_hash: str) -> str:
        # 步骤 6: 从MongoDB获取完整简历文本
        # 6.1 记录获取简历的日志
        """从MongoDB获取完整简历文本"""
        try:
            # 6.2 从MongoDB查询指定哈希值的文档
            result = self.mongo_collection.find_one({"doc_hash": doc_hash})
            # 6.3 如果查询到结果，记录成功日志并返回内容
            if result:
                logger.info(f"成功获取完整简历: hash {doc_hash}")
                return result["content"]
            # 6.4 如果未找到，记录警告日志并返回空字符串
            logger.warning(f"未找到完整简历: hash {doc_hash}")
            return ""
        except Exception as e:
            # 6.5 捕获异常，记录错误日志并抛出
            logger.error(f"获取完整简历失败: hash {doc_hash}, 错误: {str(e)}")
            raise

    def hybrid_search_with_rerank(self, query: str, params: Dict[str, Any]) -> List[Document]:
        # 步骤 7: 执行多路召回（milvus混合检索+es）并重排
        # 7.1 设置初始检索数量k为count的3倍
        # k = params.get('count', 3) * 3
        k = params.get('count', 1) * 1
        # 7.2 设置最终返回数量m为count
        # m = params.get('count', 3)
        m = params.get('count', 1)

        # 7.3 构造过滤条件列表
        filter_conditions = []
        # 7.4 添加性别过滤条件（若非“未提供”）
        if params.get('gender') and params['gender'] != '未提供':
            filter_conditions.append(f"gender == '{params['gender']}'")
        # 7.5 添加最小年龄过滤条件
        if params.get('age_min') is not None:
            filter_conditions.append(f"age >= {params['age_min']}")
        # 7.6 添加最大年龄过滤条件
        if params.get('age_max') is not None:
            filter_conditions.append(f"age <= {params['age_max']}")
        # 7.7 添加最小工作经验过滤条件
        if params.get('experience_min') is not None:
            filter_conditions.append(f"work_experience >= {params['experience_min']}")
        # 7.8 添加最大工作经验过滤条件
        if params.get('experience_max') is not None:
            filter_conditions.append(f"work_experience <= {params['experience_max']}")
        # 7.9 将过滤条件拼接为字符串
        filter_expr = " and ".join(filter_conditions)

        # 7.10 记录混合检索开始的日志
        logger.info(f"开始混合检索: query='{query}', m={m}, filter='{filter_expr}'")

        # 7.11 使用嵌入函数为查询生成嵌入向量
        query_embeddings = self.embedding_function.encode_queries([query])
        # 7.12 获取稠密向量
        dense_vector = query_embeddings["dense"][0].tolist()
        # 7.13 构造稀疏向量字典
        sparse_vector = {int(idx): float(val) for idx, val in
                         zip(query_embeddings["sparse"].indices, query_embeddings["sparse"].data)}

        # 7.14 创建稠密向量搜索请求
        dense_req = AnnSearchRequest(data=[dense_vector], anns_field="dense_vector",
                                     param={"metric_type": "IP", "params": {"nprobe": 10}}, limit=k)
        # 7.15 创建稀疏向量搜索请求
        sparse_req = AnnSearchRequest(data=[sparse_vector], anns_field="sparse_vector", param={"metric_type": "IP"},
                                      limit=k)

        # 7.16 执行Milvus混合搜索，结合稠密和稀疏向量
        milvus_results = self.client.hybrid_search(
            collection_name=config.MILVUS_COLLECTION_NAME,
            reqs=[dense_req, sparse_req],
            ranker=WeightedRanker(0.7, 0.3),
            # limit=k, filter=filter_expr, output_fields=["*"]
            limit=k,
            filter=filter_expr,
            output_fields=['id',"work_experience", "age", "name", "doc_hash", "text", "parent_content","gender"]
        )
        # 7.17 记录Milvus召回结果数量
        logger.info(f"Milvus召回了 {len(milvus_results)} 个结果。")
        # logger.info(f"Milvus结果: {milvus_results}")
        milvus_results=milvus_results[0]
        # 7.18 在Elasticsearch中执行文本搜索
        es_results = \
        self.es_client.search(index=config.ES_INDEX_NAME, body={"query": {"match": {"content": query}}, "size": k})[
            "hits"]["hits"]
        # 7.19 记录Elasticsearch召回结果数量
        logger.info(f"ES召回了 {len(es_results)} 个结果。")
        # logger.info(f"ES结果: {es_results}")
        print("====================es_results==========================")
        print(es_results)

        # 7.20 合并Milvus搜索结果到字典
        all_hits = {hit['id']: hit['entity'] for hit in milvus_results}
        print("==============Milvus的======milvus_results==========================")
        print(all_hits)
        # 7.21 合并Elasticsearch结果，补充未在Milvus中找到的文档
        for hit in es_results:
            if hit['_id'] not in all_hits:
                mongo_data = self.get_metadata_by_hash(hit['_source'].get('metadata', {}).get('hash'))
                if mongo_data:
                    all_hits[hit['_id']] = {**hit['_source']['metadata'], **mongo_data.get('structured_data', {})}

        print("==============ES与Milvus合并后======all_hits==========================")
        print(all_hits)
        # 7.22 如果没有检索到任何结果，返回空列表
        if not all_hits: return []

        # 7.23 用父块文本 构造查询与文档内容的配对，用于重排
        # 使用 entity.get('parent_content', ...) 来获取父块的文本
        # 如果parent_content不存在，则回退使用子块自身的text，以增强代码的健壮性  用父块文本
        # pairs = [[query, entity.get('parent_content', entity.get('text', ''))] for entity in all_hits.values()]
        pairs = [[query, entity.get('text', entity.get('parent_content', ''))] for entity in all_hits.values()]
        print("===============用于重排序的====句子pairs==========================")
        print(pairs)
        # 7.24 使用重排模型预测文档相关性得分
        scores = self.reranker.predict(pairs)
        print("================reranker.predict(pairs)得到====scores==========================")
        print(scores)

        # 7.25 构造带得分的文档列表
        docs_with_scores = [Document(page_content=entity.get('text', ''), metadata={**entity, 'rerank_score': score})
                            for entity, score in zip(all_hits.values(), scores)]
        # 7.26 按重排得分降序排序
        docs_with_scores.sort(key=lambda x: x.metadata['rerank_score'], reverse=True)
        print("===============docs_with_scores======按重排得分降序排序=============")
        print(docs_with_scores)

        # 7.27 初始化最终文档列表和去重集合
        final_docs = []
        parent_hashes = set()
        # 7.28 遍历排序后的文档，去重并获取完整简历
        for doc in docs_with_scores:
            doc_hash = doc.metadata.get('doc_hash')
            if doc_hash and doc_hash not in parent_hashes:
                # 7.29 根据哈希值获取MongoDB中的完整文档
                mongo_doc = self.get_metadata_by_hash(doc_hash)
                if mongo_doc:
                    # 7.30 合并元数据和重排得分
                    final_metadata = mongo_doc.get('metadata', {})
                    final_metadata['rerank_score'] = doc.metadata.get('rerank_score')
                    final_metadata['doc_hash'] = doc_hash  # 确保doc_hash在顶层

                    # 7.31 构造最终文档对象
                    final_docs.append(Document(
                        page_content=mongo_doc.get('content', ''),
                        metadata=final_metadata
                    ))
                    # 7.32 将哈希值添加到去重集合
                    parent_hashes.add(doc_hash)
            # 7.33 如果达到返回数量限制，终止循环
            if len(final_docs) >= m: break

        # 7.34 记录重排和去重后的结果数量
        logger.info(f"重排和去重后，返回 {len(final_docs)} 份独立简历。")
        # 7.35 返回最终文档列表
        return final_docs

    async def aget_relevant_documents(self, query: str, params: Dict[str, Any]) -> List[Document]:
        # 步骤 8: 异步获取相关文档
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

    # step2:验证【存储简历】
    try:
        from utils.document_processor import (
            load_and_hash_document,
            parse_resume_structure,
            process_document,
            parser_client
        )

        test_file_path = os.path.join(config.LOCAL_RESUME_DIR, "李明AI大模型产品经理简历.pdf")
        if not os.path.exists(test_file_path):
            raise FileNotFoundError(f"关键测试文件不存在: {test_file_path}")

        print(f"--- 使用真实简历进行端到端测试: {os.path.basename(test_file_path)} ---")
        # 加载文档内容和获取文档 hash
        content, doc_hash = load_and_hash_document(test_file_path, parser_client)
        structured_data = parse_resume_structure(content, parser_client)
        doc = Document(
            page_content=content,
            metadata={
                "file_path": test_file_path,
                "hash":
                    doc_hash, **structured_data
            }
        )
        chunks = process_document(doc)
        logger.success("简历加载、解析、切块成功！")
        print(f"    - 提取的结构化信息: {structured_data}")

        # 测试存储 (确保可重复执行)
        logger.info("--> 测试简历存储 (清理旧数据后)...")
        # 为了可重复测试，先尝试删除旧数据
        vector_store.mongo_collection.delete_one({"doc_hash": doc_hash})
        # milvus 使用关键字参数 filter 调用 delete 方法
        vector_store.client.delete(collection_name=config.MILVUS_COLLECTION_NAME, filter=f"doc_hash == '{doc_hash}'")
        # ES清理比较复杂，这里我们假设ID是唯一的，重复索引会覆盖

        success = vector_store.store_resume(doc, chunks, structured_data)
        assert success, "存储简历失败！"
        logger.success("简历存储功能验证通过！")
    except Exception as e:
        logger.error(f"简历存储测试失败！错误信息: {e}")

    # step3 验证 【混合检索 + 重排】
    # 定义测试查询和参数
    query = "需要AI大模型产品经理"  # 示例查询，根据简历内容调整
    params: Dict[str, any] = {
        "count": 1,  # 返回前1个结果
        "gender": "未提供",  # 性别过滤
        "experience_min": 3,  # 最小工作经验
    }
    # 调用混合检索与重排函数
    results: List[Document] = vector_store.hybrid_search_with_rerank(query, params)
    # 验证检索结果是否非空
    assert len(results) > 0, "检索结果为空，无法验证"
    print(f"检索到 {len(results)} 个结果。")
    # 打印第一个结果的详细信息
    if results:
        first_doc = results[0]
        print("第一个检索结果:")
        print(f"内容: {first_doc.page_content[:200]}...")  # 打印前200字符
        print(f"元数据: {first_doc.metadata}")
        print(f"重排分数: {first_doc.metadata.get('rerank_score', 'N/A')}")

    # 验证结果是否符合过滤条件
    for doc in results:
        structured_data = doc.metadata.get('structured_data', {})
        assert structured_data.get('gender') == params.get('gender') or params.get('gender') == '未提供', \
            "性别过滤不匹配"
    print("hybrid_search_with_rerank 函数验证通过！")