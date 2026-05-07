import numpy as np
from scipy.sparse import csr_matrix

# 模拟一个小稀疏矩阵：2 行（2 个 chunks），10 维度（小一点，便于理解）
# 设置非零元素：
# 行 0：维度 1=0.5, 维度 3=0.7
# 行 1：维度 2=0.8, 维度 5=0.9, 维度 7=1.0

# data：所有非零值，按行顺序
data = np.array([0.5, 0.7, 0.8, 0.9, 1.0])

# indices：对应 data 的列索引（维度）
indices = np.array([1, 3, 2, 5, 7])

# indptr：行指针，[0, 2, 5] 表示：
# 行0: 0到2（2个非零）
# 行1: 2到5（3个非零）
indptr = np.array([0, 2, 5])

# 创建 CSR 矩阵
sparse_matrix = csr_matrix((data, indices, indptr), shape=(2, 10))
print("稀疏矩阵（CSR 格式）：")
print(sparse_matrix)

# 打印矩阵整体（稠密形式，便于查看）
print("稀疏矩阵（稠密表示）：")
print(sparse_matrix.toarray())

# 打印 CSR 组件
print("\nCSR 数据组织：")
print("data (所有非零值):", data)
print("indices (对应维度/列索引):", indices)
print("indptr (行指针):", indptr)

# 模拟 chunks 列表（假设 2 个 chunks，对应 2 行）
# 这里 chunks 只用于 enumerate 的 idx，实际代码中 chunk 是 Document 对象，但模拟时无需真实对象
chunks = ["模拟 chunk 0", "模拟 chunk 1"]  # 简单列表，长度匹配行数

# 模拟代码循环：针对每个 idx, chunk in enumerate(chunks)
for idx, chunk in enumerate(chunks):  # 循环遍历每个块：idx是索引，chunk是Document（这里模拟为字符串）。
    sparse_indices = sparse_matrix.indices[sparse_matrix.indptr[idx]:sparse_matrix.indptr[idx + 1]]  # 提取当前块的稀疏向量索引：从CSR矩阵中切片，使用 indptr[idx] 到 indptr[idx+1] 作为范围，获取对应 indices 的子数组（维度列表）。
    sparse_data = sparse_matrix.data[sparse_matrix.indptr[idx]:sparse_matrix.indptr[idx + 1]]  # 提取当前块的稀疏向量数据：从CSR矩阵中切片，使用相同范围，获取对应 data 的子数组（非零值列表）。
    sparse_vector = {int(k): float(v) for k, v in zip(sparse_indices, sparse_data)}  # 构建稀疏向量字典：键为整数索引（维度，从 sparse_indices），值为浮点数据（从 sparse_data），使用 zip 配对并转换为 dict 格式（Milvus 插入要求的格式）。

    print(f"\n针对 idx={idx} (第 {idx} 行/chunk):")
    print("切片范围: 从", sparse_matrix.indptr[idx], "到", sparse_matrix.indptr[idx + 1])
    print("sparse_indices (维度):", sparse_indices)
    print("sparse_data (值):", sparse_data)
    print("sparse_vector (字典):", sparse_vector)