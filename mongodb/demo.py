"""
MongoDB 前置知识学习 - CRUD 操作演示

独立可运行脚本，使用独立数据库 prerequisite_demo，不污染项目数据。
连接参数与 SmartRecruit 项目一致（admin/123456）。

运行前确保：
1. MongoDB 服务已启动（docker compose up -d mongodb）
2.pip install pymongo
"""
from langchain_classic.chains.sql_database import query

from sqlalchemy.ext.asyncio import result

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from bson import ObjectId


# ==================== 全局配置 ====================
# 连接参数与 vector_store.py 一致
MONGO_HOST = "localhost"
MONGO_PORT = 27017
MONGO_USER = "admin"
MONGO_PASSWORD = "123456"
MONGO_AUTH_SOURCE = "admin"
DEMO_DB = "prerequisite_demo"  # 独立数据库，不污染项目数据

# 测试数据：简历相关示例
SAMPLE_RESUMES = [
    {
        "name": "张三",
        "gender": "男",
        "age": 28,
        "work_experience": 5,
        "skills": ["Python", "MongoDB", "FastAPI"],
        "education": "本科",
        "expected_salary": "25K-35K",
    },
    {
        "name": "李四",
        "gender": "女",
        "age": 25,
        "work_experience": 3,
        "skills": ["Java", "MySQL", "Spring Boot"],
        "education": "硕士",
        "expected_salary": "20K-30K",
    },
    {
        "name": "王五",
        "gender": "男",
        "age": 32,
        "work_experience": 8,
        "skills": ["Python", "Elasticsearch", "LangChain"],
        "education": "博士",
        "expected_salary": "35K-50K",
    },
    {
        "name": "赵六",
        "gender": "女",
        "age": 24,
        "work_experience": 2,
        "skills": ["Go", "Redis", "Docker"],
        "education": "本科",
        "expected_salary": "18K-25K",
    },
    {
        "name": "孙七",
        "gender": "男",
        "age": 29,
        "work_experience": 6,
        "skills": ["Python", "Milvus", "NLP"],
        "education": "硕士",
        "expected_salary": "30K-40K",
    },
]


def connect_and_ping():
    """连接 MongoDB 并执行健康检查（ping 命令）。

    这是所有操作的第一步：建立连接并确认数据库可达。
    在 SmartRecruit 的 vector_store.py 中，_initialize_components 阶段
    会执行类似的连接和初始化操作。

    Returns:
        MongoClient: 已验证连接的客户端实例。
        None: 连接失败时返回 None。
    """
    try:
        # 创建 MongoClient 实例
        # 参数说明：
        #   host: MongoDB 地址
        #   port: 端口号，默认 27017
        #   username/password: 认证凭据
        #   authSource: 认证数据库（用户信息存储在 admin 库中）
        client = MongoClient(
            host=MONGO_HOST,
            port=MONGO_PORT,
            username=MONGO_USER,
            password=MONGO_PASSWORD,
            authSource=MONGO_AUTH_SOURCE
        )

        # 执行 ping 命令验证连接
        # ping 是 MongoDB 内置命令，成功返回 {"ok": 1.0}
        result = client.admin.command('ping')
        print(f"[连接成功] ping 结果: {result}")

        # 获取数据库实例（此时并未真正创建数据库，插入数据时才会创建）
        db = client[DEMO_DB]
        print(f"[数据库] 已获取数据库: {DEMO_DB}")
        print(f"[数据库] 内容: {db}")

        # 列出已有集合（首次运行时为空）
        collections = db.list_collection_names()
        print(f"[集合列表] 当前集合: {collections if collections else '（空）'}")

        return client
    except ConnectionFailure as e:
        print(f"[连接失败] 无法连接到 MongoDB: {e}")
        return None


def insert_one(client):
    """插入单条文档到 resumes 集合。

    insert_one 是 MongoDB 最基本的写入操作。
    在 SmartRecruit 中，store_resume 函数在去重检查通过后，
    使用 insert_one 将完整简历写入 MongoDB。

    Args:
        client: MongoClient 实例。

    Returns:
        InsertOneResult: 包含 inserted_id（自动生成的 _id）。
    """
    db = client[DEMO_DB]
    collection = db.resumes  # 集合名为 resumes（简历）

    # 构造一条简历文档
    # MongoDB 的文档就是 Python 字典，字段名是字符串，值可以是各种类型
    doc = SAMPLE_RESUMES[0]  # 张三的简历

    # 插入文档
    # MongoDB 会自动为每条文档生成唯一的 _id（ObjectId 类型）
    result = db.resumes.insert_one(doc)

    print(f"[insert_one] 插入成功")
    print(f"  自动生成的 _id: {result.inserted_id}")
    print(f"  插入的文档: {doc}")

    return result


def insert_many(client):
    """批量插入多条文档。

    insert_many 一次性插入多条文档，比循环调用 insert_one 高效得多。
    在 SmartRecruit 中，批量导入简历时会用到类似操作。

    Args:
        client: MongoClient 实例。

    Returns:
        InsertManyResult: 包含 inserted_ids 列表。
    """
    db = client[DEMO_DB]
    collection = db.resumes

    # 批量插入剩余的简历数据（跳过已插入的张三）
    docs = SAMPLE_RESUMES[1:]

    # ordered=False 表示即使某条文档插入失败，也继续插入其余文档
    result = db.resumes.insert_many(docs, ordered=False)

    print(f"[insert_many] 批量插入成功，共插入 {len(result.inserted_ids)} 条")
    for i, oid in enumerate(result.inserted_ids):
        print(f"  [{i + 1}] _id: {oid}")

    return result


def find_one(client):
    """按条件查询单条文档。

    find_one 返回匹配查询条件的第一条文档（字典），未找到返回 None。
    在 SmartRecruit 中，多处使用 find_one：
    - store_resume 中按 doc_hash 做去重检查
    - get_metadata_by_hash 按 doc_hash 获取元数据
    - get_full_resume 按 doc_hash 获取完整简历

    Args:
        client: MongoClient 实例。

    Returns:
        dict: 查询到的文档，未找到时返回 None。
    """
    db = client[DEMO_DB]
    collection = db.resumes

    # 按姓名查询
    # 查询条件是字典：{"字段名": "值"}
    query = {"name": "张三"}
    result = collection.find_one(query)

    print(f"[find_one] 查询条件: {query}")
    if result:
        print(f"  查询结果: name={result['name']}, age={result['age']}, skills={result['skills']}")
        print(f"  _id: {result['_id']}")
    else:
        print("  未找到匹配文档")

    # 按 ObjectId 查询（MongoDB 中 _id 是主键）
    if result:
        query_by_id = {"_id": result["_id"]}
        result_by_id = db.resumes.find_one(query_by_id)
        print(f"\n[find_one] 按 _id 查询: {result_by_id['name']}")

    return result


def find(client):
    """条件查询多条文档，返回游标。

    find 返回一个 Cursor 对象（游标），可以迭代获取所有匹配文档。
    在 SmartRecruit 中，虽然主要用 find_one，但 find 适用于
    批量检索、统计分析等场景。

    Args:
        client: MongoClient 实例。

    Returns:
        list: 查询结果文档列表。
    """
    db = client[DEMO_DB]
    collection = db.resumes

    # 查询所有 gender=="男" 的简历
    query = {"gender": "男"}
    cursor = collection.find(query)

    # 将游标转换为列表（注意：大数据量时应迭代处理，不要一次性转列表）
    results = []
    for doc in cursor:
        results.append(doc)

    print(f"[find] 查询条件: {query}")
    print(f"  匹配文档数: {len(results)}")
    for i, doc in enumerate(results):
        print(f"  [{i + 1}] {doc['name']}, {doc['age']}岁, 技能: {', '.join(doc['skills'])}")

    # 查询所有文档（空查询条件）
    print(f"\n[find] 查询全部文档:")
    all_docs = list(collection.find({}))
    print(f"  总文档数: {len(all_docs)}")

    return results


def count_documents(client):
    """统计集合中的文档数量。

    count_documents 接受查询条件，返回匹配的文档数。
    在 SmartRecruit 中用于判断简历库规模、去重统计等。

    Args:
        client: MongoClient 实例。

    Returns:
        int: 匹配查询条件的文档数量。
    """
    db = client[DEMO_DB]
    collection = db.resumes

    # 统计全部文档数（空条件）
    total = collection.count_documents({})
    print(f"[count_documents] 总文档数: {total}")

    # 按条件统计：男性简历数
    male_count = collection.count_documents({"gender": "男"})
    print(f"[count_documents] 男性简历数: {male_count}")

    # 按条件统计：工作年限 >= 5 的简历数
    # $gte 是 MongoDB 比较操作符，表示 "大于等于"
    experienced_count = collection.count_documents({"work_experience": {"$gte": 5}})
    print(f"[count_documents] 工作年限>=5: {experienced_count}")

    # 按条件统计：包含 Python 技能的简历数
    # skills 是数组字段，直接用值查询会匹配数组中包含该值的文档
    python_count = collection.count_documents({"skills": {"$in": ["Python"]}})
    print(f"[count_documents] 会Python的: {python_count}")

    return total


def update_one(client):
    """按条件更新单条文档的字段。

    update_one 使用 $set 操作符更新指定字段，不影响其他字段。
    在 SmartRecruit 中，vector_store.py 目前没有显式的 update 操作，
    但在实际应用中，更新简历状态（如"已查看"、"已面试"）是常见需求。

    Args:
        client: MongoClient 实例。

    Returns:
        UpdateResult: 包含 matched_count 和 modified_count。
    """
    db = client[DEMO_DB]
    collection = db.resumes

    # 更新操作：将张三的年龄从 28 改为 29
    query = {"name": "张三"}
    # $set：只更新指定字段，其他字段保持不变
    # 如果不用 $set，整个文档会被替换
    update = {"$set": {"age": 29}}

    result = collection.update_one(query, update)

    print(f"[update_one] 更新条件: {query}")
    print(f"  匹配文档数: {result.matched_count}")
    print(f"  实际修改数: {result.modified_count}")

    # 验证更新结果
    updated_doc = collection.find_one(query)
    print(f"  更新后: age={updated_doc['age']}, salary={updated_doc['expected_salary']}")

    return result


def delete_one(client):
    """按条件删除单条文档。

    delete_one 删除匹配条件的第一条文档。
    在 SmartRecruit 中，vector_store.py 有 delete_collection 操作，
    但单条删除在简历管理场景中同样常见（如撤回错误录入）。

    Args:
        client: MongoClient 实例。

    Returns:
        DeleteResult: 包含 deleted_count。
    """
    db = client[DEMO_DB]
    collection = db.resumes

    # 先插入一条专门用来删除的测试数据
    test_doc = {
        "name": "测试用户",
        "gender": "男",
        "age": 30,
        "work_experience": 5,
        "skills": ["Testing"],
        "education": "本科",
    }
    collection.insert_one(test_doc)
    print(f"[delete_one] 已插入测试文档: {test_doc['name']}")

    # 按条件删除
    query = {"name": "测试用户"}
    result = collection.delete_one(query)

    print(f"[delete_one] 删除条件: {query}")
    print(f"  删除文档数: {result.deleted_count}")

    # 验证删除
    check = collection.find_one(query)
    print(f"  删除后查询: {'已删除' if check is None else '仍存在'}")

    return result


def delete_many(client):
    """按条件批量删除文档。

    delete_many 删除所有匹配条件的文档。
    在 SmartRecruit 中，重置向量库时会用到类似操作。

    Args:
        client: MongoClient 实例。

    Returns:
        DeleteResult: 包含 deleted_count。
    """
    db = client[DEMO_DB]
    collection = db.resumes

    # 查看删除前的文档数
    before_count = collection.count_documents({})
    print(f"[delete_many] 删除前文档数: {before_count}")

    # 删除所有工作年限 < 3 的简历
    result = collection.delete_many({"work_experience": {"$lt": 3}})

    print(f"[delete_many] 删除条件: work_experience < 3")
    print(f"  删除文档数: {result.deleted_count}")

    # 查看删除后的文档数
    after_count = collection.count_documents({})
    print(f"  删除后文档数: {after_count}")

    # 清理：删除本 demo 创建的所有测试数据，保持数据库干净
    print(f"\n[清理] 删除剩余测试数据...")
    cleanup = collection.delete_many({})
    print(f"  清理了 {cleanup.deleted_count} 条文档")

    # 删除 demo 数据库（确保不留下痕迹）
    client.drop_database(DEMO_DB)
    print(f"[清理] 已删除数据库 {DEMO_DB}")

    return result


if __name__ == "__main__":
    print("=" * 60)
    print("MongoDB 前置知识学习 - CRUD 操作演示")
    print("=" * 60)

    # 第 1 步：连接 + 健康检查
    print("\n--- 1. 连接 + 健康检查 ---")
    client = connect_and_ping()
    if client is None:
        print("无法连接 MongoDB，请检查服务是否启动。")
        exit(1)

    # 第 2 步：插入单条文档
    print("\n--- 2. insert_one ---")
    insert_one(client)

    # 第 3 步：批量插入
    print("\n--- 3. insert_many ---")
    insert_many(client)

    # 第 4 步：查询单条
    print("\n--- 4. find_one ---")
    find_one(client)

    # 第 5 步：条件查询多条
    print("\n--- 5. find ---")
    find(client)

    # 第 6 步：统计文档数量
    print("\n--- 6. count_documents ---")
    count_documents(client)

    # 第 7 步：更新单条
    print("\n--- 7. update_one ---")
    update_one(client)

    # 第 8 步：删除单条
    print("\n--- 8. delete_one ---")
    delete_one(client)

    # 第 9 步：批量删除
    print("\n--- 9. delete_many ---")
    delete_many(client)

    # 关闭连接
    client.close()
    print("\n[完成] 连接已关闭，演示结束。")
