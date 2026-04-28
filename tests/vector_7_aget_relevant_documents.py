
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
            self.mongo_db = self.mongo_client[config.MONGO_DB]  # 获取指定数据库实例。
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
        # collection_name = "test_collect_rag"  # 获取集合名称：从配置中读取。
        if not self.client.has_collection(config.MILVUS_COLLECTION_NAME):  # 检查集合是否存在：如果不存在，则创建。
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

            self.client.create_collection(collection_name=config.MILVUS_COLLECTION_NAME, schema=schema,
                                          index_params=index_params)  # 创建集合：使用schema和index_params。
        self.client.load_collection(config.MILVUS_COLLECTION_NAME)  # 加载集合：确保集合在内存中可用，用于查询。

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

    def hybrid_search_with_rerank(self, query: str, params: Dict[str, Any]) -> List[Document]:
        # 函数整体注释：执行混合搜索（Milvus向量 + ES全文），结合重排和去重，返回独立简历文档列表。
        # 支持过滤（如性别、年龄、工作年限），使用WeightedRanker融合结果，重排后去重基于doc_hash。
        # 步骤清晰描述：
        # 步骤1: 计算召回参数：k 为初始召回数（最终返回 m 的 3 倍，用于缓冲），m 为最终返回数。
        k = params.get('count', 3) * 3  # 计算初始召回数量：最终返回数的3倍，用于重排前缓冲，确保有足够候选。
        m = params.get('count', 3)  # 获取最终返回数量：默认3，从 params 中取或用默认值。

        # 步骤2: 构建 Milvus 过滤表达式：基于 params 中的性别、年龄、工作经验范围，使用 AND 连接。
        filter_conditions = []  # 初始化过滤条件列表：用于Milvus表达式，存储字符串条件。
        if params.get('gender') and params['gender'] != '未提供':  # 添加性别过滤：如果指定且非默认，则添加 equality 条件。
            filter_conditions.append(f"gender == '{params['gender']}'")
        if params.get('age_min') is not None:  # 添加最小年龄过滤：如果指定，则添加 >= 条件。
            filter_conditions.append(f"age >= {params['age_min']}")
        if params.get('age_max') is not None:  # 添加最大年龄过滤：如果指定，则添加 <= 条件。
            filter_conditions.append(f"age <= {params['age_max']}")
        if params.get('experience_min') is not None:  # 添加最小工作经验过滤：如果指定，则添加 >= 条件。
            filter_conditions.append(f"work_experience >= {params['experience_min']}")
        if params.get('experience_max') is not None:  # 添加最大工作经验过滤：如果指定，则添加 <= 条件。
            filter_conditions.append(f"work_experience <= {params['experience_max']}")
        filter_expr = " and ".join(filter_conditions)  # 构建过滤表达式：用 " and " 连接所有条件，形成 Milvus 可用的字符串表达式。

        logger.info(f"开始混合检索: query='{query}', m={m}, filter='{filter_expr}'")  # 记录日志：检索开始，显示查询、返回数和过滤表达式，便于调试。

        # 步骤3: 生成查询嵌入：使用 BGEM3 编码 query，提取稠密和稀疏向量。
        query_embeddings = self.embedding_function.encode_queries(
            [query])  # 生成查询嵌入：使用BGEM3编码查询（输入列表 [query]），返回包含 "dense" 和 "sparse" 的字典。
        dense_vector = query_embeddings["dense"][0].tolist()  # 提取稠密向量：从字典取第一个稠密数组，转换为 Python 列表（Milvus 要求的格式）。
        sparse_vector = {int(idx): float(val) for idx, val in zip(query_embeddings["sparse"].indices, query_embeddings[
            "sparse"].data)}  # 构建稀疏向量字典：从 CSR 矩阵的 indices 和 data 配对，转换为 {维度: 值} 格式（Milvus 稀疏向量要求）。

        # 步骤4: 执行 Milvus 混合搜索：分别创建稠密和稀疏请求，加权融合结果，应用过滤，召回 k 个。
        dense_req = AnnSearchRequest(data=[dense_vector], anns_field="dense_vector",
                                     param={"metric_type": "IP", "params": {"nprobe": 10}},
                                     limit=k)  # 创建稠密搜索请求：指定数据、字段、度量类型（IP 内积）、搜索参数（nprobe=10）和限量 k。
        sparse_req = AnnSearchRequest(data=[sparse_vector], anns_field="sparse_vector", param={"metric_type": "IP"},
                                      limit=k)  # 创建稀疏搜索请求：类似稠密，但无额外 params（如 nprobe），限量 k。

        milvus_results = self.client.hybrid_search(  # 执行Milvus混合搜索：融合稠密和稀疏结果。
            collection_name=config.MILVUS_COLLECTION_NAME,  # 指定集合名称：从 config 获取。
            reqs=[dense_req, sparse_req],  # 搜索请求列表：包含稠密和稀疏请求。
            ranker=WeightedRanker(0.7, 0.3),  # 使用加权排序器：稠密权重0.7，稀疏0.3，融合两者的排名。
            limit=k, filter=filter_expr, output_fields=["*"]  # 设置限量 k、应用过滤表达式、输出所有字段。
        )[0]  # 获取第一个输出（混合结果列表）：hybrid_search 返回列表，取 [0] 为融合后的结果。
        logger.info(f"Milvus召回了 {len(milvus_results)} 个结果。")  # 记录日志：Milvus结果数量，便于监控召回效果。

        # 步骤5: 执行 ES 全文搜索：匹配 query，召回 k 个结果。
        es_results = \
        self.es_client.search(index=config.ES_INDEX_NAME, body={"query": {"match": {"content": query}}, "size": k})[
            "hits"]["hits"]  # 执行ES全文搜索：使用 match 查询匹配 content 字段，限量 k，返回 hits 列表。
        logger.info(f"ES召回了 {len(es_results)} 个结果。")  # 记录日志：ES结果数量。

        # 步骤6: 合并 Milvus 和 ES 结果：用字典存储，避免重复 ID，从 MongoDB 补充元数据。
        all_hits = {hit['id']: hit['entity'] for hit in
                    milvus_results}  # 构建所有命中字典：以 ID 为键，Milvus 实体（entity）为值，用于后续合并和去重。
        for hit in es_results:  # 合并ES结果：循环遍历 ES hits，如果 ID 不在 all_hits 中，则添加。
            if hit['_id'] not in all_hits:  # 检查ID是否存在：避免重复。
                mongo_data = self.get_metadata_by_hash(hit['_source'].get('metadata', {}).get(
                    'hash'))  # 从MongoDB获取元数据：基于 ES hit 中的 hash 值调用 get_metadata_by_hash。
                if mongo_data:  # 如果数据存在：合并 ES 元数据和 MongoDB 结构化数据到 all_hits。
                    all_hits[hit['_id']] = {**hit['_source']['metadata'], **mongo_data.get('structured_data',
                                                                                           {})}  # 合并字典：ES metadata + Mongo structured_data。

        if not all_hits: return []  # 如果无命中（all_hits 为空）：直接返回空列表。

        # 步骤7: 重排结果：构建查询-文本对，使用 CrossEncoder 预测分数，排序文档。
        pairs = [[query, entity.get('text', '')] for entity in
                 all_hits.values()]  # 构建重排对：列表中每个元素是 [query, 文本]，用于 CrossEncoder 输入，从 all_hits 值中取 text（默认空字符串）。
        scores = self.reranker.predict(pairs)  # 执行重排：使用CrossEncoder预测分数，返回分数数组（每个 pair 一个分数）。

        docs_with_scores = [Document(page_content=entity.get('text', ''), metadata={**entity, 'rerank_score': score})
                            for entity, score in zip(all_hits.values(),
                                                     scores)]  # 构建带分数的Document列表：每个 Document 使用 entity text 作为内容，metadata 合并 entity 和 rerank_score。
        docs_with_scores.sort(key=lambda x: x.metadata['rerank_score'], reverse=True)  # 排序：按 rerank_score 降序排序，确保最高分先。
        # print("==================================")  # 打印分隔线：调试用途，实际可注释。
        # print(docs_with_scores)  # 打印排序后文档：调试用途，实际可注释。

        # 步骤8: 去重并构建最终文档：基于 doc_hash 去重，从 MongoDB 获取完整内容，合并元数据，返回前 m 个。
        final_docs = []  # 初始化最终文档列表：存储去重后的完整 Document。
        parent_hashes = set()  # 初始化哈希集合：用于去重，存储已处理的 doc_hash。
        for doc in docs_with_scores:  # 循环遍历排序文档：去重并构建最终列表。
            doc_hash = doc.metadata.get('doc_hash')  # 获取doc_hash：从当前 doc metadata 中取。
            if doc_hash and doc_hash not in parent_hashes:  # 如果哈希存在且未重复：处理该文档。
                mongo_doc = self.get_metadata_by_hash(
                    doc_hash)  # 从MongoDB获取完整文档：基于 doc_hash 调用 get_metadata_by_hash，返回字典。
                if mongo_doc:  # 如果存在：构建最终Document。
                    # 核心修复：将mongo_doc中的元数据与rerank_score平铺合并  # 注释：修复合并逻辑，确保所有键在同一层级。
                    final_metadata = mongo_doc.get('metadata', {})  # 获取元数据：从 mongo_doc 中取 metadata 字典。
                    final_metadata['rerank_score'] = doc.metadata.get(
                        'rerank_score')  # 添加重排分数：从当前 doc metadata 复制 rerank_score。
                    final_metadata['doc_hash'] = doc_hash  # 确保doc_hash在顶层：显式添加 doc_hash 到 final_metadata。

                    final_docs.append(Document(  # 创建Document：使用完整内容和合并元数据，添加到 final_docs。
                        page_content=mongo_doc.get('content', ''),
                        metadata=final_metadata
                    ))
                    parent_hashes.add(doc_hash)  # 添加到去重集合：标记该 hash 已处理，避免重复简历。
            if len(final_docs) >= m: break  # 如果达到 m：停止循环，只返回前 m 个。

        logger.info(f"重排和去重后，返回 {len(final_docs)} 份独立简历。")  # 记录日志：最终返回数量，便于追踪结果。
        return final_docs  # 返回最终文档列表：每个是完整简历的 Document，按分数排序并去重。


    async def aget_relevant_documents(self, query: str, params: Dict[str, Any]) -> List[Document]:
        # 函数整体注释：异步版本的检索方法，包装同步hybrid_search_with_rerank以支持异步调用。
        # 使用asyncio.to_thread在线程中运行同步函数，避免阻塞。
        return await asyncio.to_thread(self.hybrid_search_with_rerank, query, params)  # 异步执行：将同步函数转换为线程任务，等待返回结果。


# 验证aget_relevant_documents函数（使用asyncio运行）
# import asyncio
# vector_store = VectorStore()
# test_query = "熟悉AI大模型的产品经理"
# test_params = {"count": 3, "gender": "未提供", "experience_min": 3}
# async def test_async():
#     results = await vector_store.aget_relevant_documents(test_query, test_params)
#     assert len(results) > 0, "异步检索结果为空"
#     logger.info("异步检索验证通过！")
#
# try:
#     asyncio.run(test_async())
# except Exception as e:
#     logger.error(f"异步检索失败: {e}")


import asyncio
from loguru import logger

# 假设 VectorStore 已定义
vector_store = VectorStore()

test_query1 = "熟悉AI大模型的产品经理"  # 第一条查询
test_params1 = {"count": 3, "gender": "未提供", "experience_min": 3}

test_query2 = "资深软件工程师"  # 第二条查询（不同内容，模拟并发）
test_params2 = {"count": 2, "gender": "男", "experience_min": 5}


async def test_async():
    # 并发启动两条查询（用 asyncio.gather）
    task1 = vector_store.aget_relevant_documents(test_query1, test_params1)
    task2 = vector_store.aget_relevant_documents(test_query2, test_params2)

    # await 同时等待两条结果（并发运行，不阻塞）
    results1, results2 = await asyncio.gather(task1, task2)

    # 验证
    assert len(results1) > 0, "查询1 结果为空"
    assert len(results2) > 0, "查询2 结果为空"
    logger.info(f"异步并发测试通过！查询1 结果数: {len(results1)}, 查询2 结果数: {len(results2)}")


try:
    asyncio.run(test_async())
except Exception as e:
    logger.error(f"异步检索失败: {e}")