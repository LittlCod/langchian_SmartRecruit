# MongoDB 前置知识学习

## 一、核心问题：SmartRecruit 为什么需要 MongoDB？

SmartRecruit 系统处理的核心对象是**简历**。一份简历包含姓名、性别、年龄、技能列表、工作经历等多层次、非结构化的数据。MongoDB 在系统中承担三个关键角色：

| 角色 | 说明 | vector_store.py 对应位置 |
|---|---|---|
| 存完整简历 | 简历文本原文存储，支持按 hash 快速取回 | `get_full_resume` |
| 存结构化元数据 | 文件名、doc_hash、页数、分段数等与检索无关但管理需要的信息 | `get_metadata_by_hash` |
| 去重判断 | 新简历入库前，按 doc_hash 检查是否已存在 | `store_resume` 中的去重检查 |

为什么不用 MySQL？简历的 skills 字段是一个数组（`["Python", "MongoDB", "FastAPI"]`），work_experience 可能嵌套多个公司记录。关系型数据库需要拆表 + JOIN，而 MongoDB 直接存储嵌套文档，天然匹配简历的数据结构。

```mermaid
graph LR
    A[简历文件] --> B[pdf_processor<br/>解析提取]
    B --> C[MongoDB<br/>存储原文+元数据]
    B --> D[Milvus<br/>存储向量]
    B --> E[Elasticsearch<br/>BM25索引]
    C -->|按hash取回| F[搜索结果展示]
    D -->|向量检索| F
    E -->|BM25检索| F
```

## 二、前置知识

### 2.1 文档数据库 vs 关系型数据库

| 维度 | MongoDB（文档数据库） | MySQL（关系型数据库） |
|---|---|---|
| 数据模型 | 文档（JSON/BSON） | 表（行+列） |
| Schema | 灵活，同一集合内文档结构可以不同 | 固定，建表时定义列 |
| 数组/嵌套 | 原生支持 | 需要 JSON 列或拆表 |
| JOIN | 有限支持（$lookup） | 核心能力 |
| 适合场景 | 数据结构多变、嵌套多、读多写少 | 数据结构固定、事务强一致 |

### 2.2 核心概念

| MongoDB 概念 | 类比 MySQL | 说明 |
|---|---|---|
| Database（数据库） | Database | 一组相关集合的容器 |
| Collection（集合） | Table | 存储文档的容器，无需预定义结构 |
| Document（文档） | Row | 一条数据记录，格式为 BSON |
| Field（字段） | Column | 文档中的一个键值对 |
| _id | PRIMARY KEY | 每个文档的唯一标识，自动生成 ObjectId |

### 2.3 BSON 是什么

BSON（Binary JSON）是 MongoDB 的存储格式，在 JSON 基础上增加了：

- **更多数据类型**：Date、ObjectId、Binary、Regex 等
- **效率更高**：二进制编码，解析速度比 JSON 快
- **支持嵌入**：文档中可以嵌套文档和数组

在 Python 中，pymongo 自动处理 Python 字典与 BSON 的转换，开发者无需手动编码。

```mermaid
graph TD
    A[Python 字典] -->|pymongo 自动转换| B[BSON 文档]
    B -->|存入| C[MongoDB 集合]
    C -->|读取| B
    B -->|pymongo 自动转换| A
```

## 三、CRUD 操作详解

> 以下代码均来自 `demo.py`，每个操作封装为独立函数。

### 3.1 连接 + 健康检查（connect_and_ping）

```python
from pymongo import MongoClient

client = MongoClient(
    host="localhost",
    port=27017,
    username="admin",
    password="123456",
    authSource="admin",       # 用户凭据存储在 admin 数据库
)

# ping 命令验证连接是否成功
result = client.admin.command("ping")
# 成功时返回: {"ok": 1.0}

# 获取数据库实例（不会立即创建，插入数据时才创建）
db = client["prerequisite_demo"]
```

输出示例：
```
[连接成功] ping 结果: {'ok': 1.0}
[数据库] 已获取数据库: prerequisite_demo
[集合列表] 当前集合: （空）
```

### 3.2 插入单条文档（insert_one）

```python
doc = {
    "name": "张三",
    "gender": "男",
    "age": 28,
    "work_experience": 5,
    "skills": ["Python", "MongoDB", "FastAPI"],
    "education": "本科",
    "expected_salary": "25K-35K",
}
result = db.resumes.insert_one(doc)
print(result.inserted_id)  # ObjectId("6629a1b3...")
```

要点：
- `insert_one` 返回 `InsertOneResult`，其中 `inserted_id` 是自动生成的 ObjectId
- 如果文档中没有 `_id` 字段，MongoDB 自动生成一个 ObjectId
- 集合 `resumes` 在第一次插入时自动创建，无需提前建表

### 3.3 批量插入（insert_many）

```python
docs = [
    {"name": "李四", "gender": "女", "age": 25, ...},
    {"name": "王五", "gender": "男", "age": 32, ...},
]
result = db.resumes.insert_many(docs, ordered=False)
print(result.inserted_ids)  # [ObjectId("..."), ObjectId("...")]
```

要点：
- `ordered=False` 表示某条失败不影响其余文档的插入
- 比循环调用 `insert_one` 效率高得多（一次网络往返）

### 3.4 按条件查单条（find_one）

```python
# 按字段查询
doc = db.resumes.find_one({"name": "张三"})

# 按 _id 查询
from bson import ObjectId
doc = db.resumes.find_one({"_id": ObjectId("6629a1b3...")})
```

要点：
- `find_one` 返回字典或 `None`（未找到时）
- 查询条件是字典，`{"字段": "值"}` 表示等值匹配
- 按 `_id` 查询是最快的（主键索引）

### 3.5 条件查询多条（find）

```python
# 查询所有男性简历
cursor = db.resumes.find({"gender": "男"})
for doc in cursor:
    print(doc["name"], doc["skills"])

# 查询全部文档
all_docs = list(db.resumes.find({}))
```

要点：
- `find` 返回 Cursor 游标，惰性加载，可以迭代但只能遍历一次
- 大数据量时应逐条迭代，不要 `list()` 一次性加载到内存
- 空字典 `{}` 表示无过滤条件，查询全部

### 3.6 统计文档数量（count_documents）

```python
# 全部文档数
total = db.resumes.count_documents({})

# 条件统计
male_count = db.resumes.count_documents({"gender": "男"})

# 比较操作符：$gte（大于等于）、$lt（小于）等
experienced = db.resumes.count_documents({"work_experience": {"$gte": 5}})
```

常用比较操作符：

| 操作符 | 含义 | 示例 |
|---|---|---|
| `$eq` | 等于（默认） | `{"age": 28}` |
| `$ne` | 不等于 | `{"age": {"$ne": 28}}` |
| `$gt` / `$gte` | 大于 / 大于等于 | `{"age": {"$gte": 25}}` |
| `$lt` / `$lte` | 小于 / 小于等于 | `{"age": {"$lt": 30}}` |
| `$in` | 在列表中 | `{"skills": {"$in": ["Python", "Go"]}}` |

### 3.7 按条件更新（update_one）

```python
result = db.resumes.update_one(
    {"name": "张三"},                    # 查询条件
    {"$set": {"age": 29, "expected_salary": "30K-40K"}}  # 更新操作
)
print(result.matched_count)    # 匹配了几条
print(result.modified_count)   # 实际修改了几条
```

要点：
- **必须用 `$set`**，否则整个文档会被替换为 `{"age": 29, ...}`
- `matched_count` ≠ `modified_count`：如果新值和旧值相同，匹配但不修改
- 其他更新操作符：`$inc`（增减数值）、`$push`（向数组添加元素）、`$unset`（删除字段）

### 3.8 删除单条（delete_one）

```python
result = db.resumes.delete_one({"name": "测试用户"})
print(result.deleted_count)  # 1 或 0
```

### 3.9 批量删除（delete_many）

```python
# 删除所有工作年限小于 3 年的简历
result = db.resumes.delete_many({"work_experience": {"$lt": 3}})

# 删除全部文档（慎用！）
result = db.resumes.delete_many({})

# 删除整个数据库
client.drop_database("prerequisite_demo")
```

## 四、与 vector_store.py 的关联

`demo.py` 中的操作与 SmartRecruit 源码 `vector_store.py` 的对应关系：

| demo.py 操作 | vector_store.py 中的使用 | 步骤编号 |
|---|---|---|
| `connect_and_ping` | `_initialize_components` 中创建 MongoClient | 2.10-2.14 |
| `insert_one` | `store_resume` 中写入完整简历文档 | 4.17 |
| `find_one`（去重检查） | `store_resume` 中按 doc_hash 检查是否已存在 | 4.3 |
| `find_one`（取元数据） | `get_metadata_by_hash` 获取文件名、页数等 | 5.1 |
| `find_one`（取完整简历） | `get_full_resume` 获取简历原文 | 6.2 |
| `count_documents` | 判断集合是否为空、统计简历数量 | 初始化阶段 |
| `update_one` | 源码中未直接使用，但更新简历状态时需要 | — |
| `delete_one` | 源码中未直接使用，撤回错误录入时需要 | — |
| `delete_many` | `delete_collection` 清空集合 | 7.2 |

```mermaid
graph TD
    subgraph store_resume 流程
        A1[connect_and_ping<br/>连接MongoDB] --> A2[find_one<br/>按hash去重检查]
        A2 -->|不存在| A3[insert_one<br/>写入完整简历]
        A2 -->|已存在| A4[跳过<br/>返回已存在提示]
    end
    subgraph 查询流程
        B1[find_one<br/>get_metadata_by_hash] --> B2[返回元数据]
        B3[find_one<br/>get_full_resume] --> B4[返回简历原文]
    end
    subgraph 管理流程
        C1[delete_many<br/>delete_collection] --> C2[清空集合]
    end
```

## 五、常见问题

### 5.1 连接失败排查

| 错误信息 | 可能原因 | 解决方法 |
|---|---|---|
| `ServerSelectionTimeoutError` | MongoDB 服务未启动 | `docker compose up -d mongodb` |
| `AuthenticationFailed` | 用户名或密码错误 | 检查 `docker-compose.yml` 中的 `MONGO_INITDB_ROOT_PASSWORD` |
| `ConnectionRefused` | 端口不对或防火墙 | 检查 `27017` 端口是否暴露 |

验证连接的快速命令：
```bash
# 在宿主机上测试
mongosh "mongodb://admin:123456@localhost:27017"
```

### 5.2 `_id` 字段处理

- MongoDB 自动生成的 `_id` 是 `ObjectId` 类型，不是字符串
- 按 `_id` 查询时需要用 `ObjectId("...")` 包装，不能直接传字符串
- `insert_one` 后通过 `result.inserted_id` 获取自动生成的 `_id`
- 如果想用自定义 `_id`，可以在文档中手动指定（如用 `doc_hash` 作为 `_id`）

### 5.3 中文编码

- MongoDB / BSON 原生支持 UTF-8，中文存储和查询无需特殊处理
- pymongo 自动处理 Python 字符串与 BSON 的编码转换
- 如果遇到乱码，检查的是 Python 文件本身的编码（确保是 UTF-8）

### 5.4 数组字段查询

```python
# skills 是数组 ["Python", "MongoDB"]
# 以下查询会匹配数组中包含 "Python" 的文档
db.resumes.find({"skills": "Python"})

# 精确匹配整个数组（顺序和元素都一致）
db.resumes.find({"skills": ["Python", "MongoDB"]})

# 匹配包含多个元素的文档（顺序无关）
db.resumes.find({"skills": {"$all": ["Python", "MongoDB"]}})
```

## 六、运行验证

### 前置条件

1. MongoDB 服务已启动（通过 Docker Compose）
2. 安装 Python 依赖：`pip install pymongo`

### 运行

```bash
cd 2026-04-22-smartrecruit/mongodb
python demo.py
```

### 预期输出

脚本会依次执行 9 个操作，每个操作打印执行结果，最后自动清理测试数据并删除 `prerequisite_demo` 数据库。正常情况下：

```
============================================================
MongoDB 前置知识学习 - CRUD 操作演示
============================================================

--- 1. 连接 + 健康检查 ---
[连接成功] ping 结果: {'ok': 1.0}
[数据库] 已获取数据库: prerequisite_demo
[集合列表] 当前集合: （空）

--- 2. insert_one ---
[insert_one] 插入成功
  自动生成的 _id: ObjectId("...")

...（中间操作省略）...

--- 9. delete_many ---
[delete_many] 删除前文档数: 5
[delete_many] 删除条件: work_experience < 3
  删除文档数: 2
[清理] 删除剩余测试数据...
[清理] 已删除数据库 prerequisite_demo

[完成] 连接已关闭，演示结束。
```

脚本结束后 `prerequisite_demo` 数据库会被完全删除，不会留下任何测试数据。
