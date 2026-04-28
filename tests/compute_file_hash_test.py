#函数验证代码
from loguru import logger
import hashlib

def compute_file_hash(file_path: str) -> str:
    # 初始化MD5哈希对象
    hasher = hashlib.md5()
    try:
        # 以二进制模式打开文件
        with open(file_path, "rb") as f:
            # 分块读取文件内容，避免内存溢出
            for chunk in iter(lambda: f.read(4096), b""):
                # 更新哈希值
                hasher.update(chunk)
        # 返回十六进制哈希字符串
        return hasher.hexdigest()
    except Exception as e:
        # 记录错误日志
        logger.error(f"计算文件哈希失败: {file_path}, 错误: {e}", exc_info=True)
        # 抛出异常
        raise



# # 测试文件路径（需替换为实际文件）
test_file_path = r"D:\LLM_Codes\Chapter3_RAG\SmartRecruit\data\resume\王宝强.jpg"  # 请确保文件存在

try:
    hash_value = compute_file_hash(test_file_path)
    assert len(hash_value) == 32, "哈希长度不正确"
    logger.info(f"哈希计算成功: {hash_value}")
    print(f"验证通过: 哈希 = {hash_value}")
except Exception as e:
    logger.error(f"验证失败: {e}")
    print("验证失败")