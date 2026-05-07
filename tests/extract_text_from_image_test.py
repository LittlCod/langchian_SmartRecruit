from openai import OpenAI
from loguru import logger
import base64

# 初始化客户端
client = OpenAI(api_key="sk-b052cdd2b23249f5bbe1949928a2600a", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")


# 注意：此函数在提供的原始脚本中位于truncated部分，未完整显示。以下基于逻辑假设的实现（使用base64和OpenAI Vision）。
def extract_text_from_image(image_path: str, client: OpenAI) -> str:
    # 以二进制读取图像文件
    with open(image_path, "rb") as image_file:
        # 将图像转换为base64编码
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    # 构造OpenAI Vision提示
    prompt = "提取图像中的所有文本，包括简历内容。输出纯文本。"

    try:
        # 调用OpenAI API进行OCR
        response = client.chat.completions.create(
            model="qwen-omni-turbo",  # 使用Vision模型
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ],
            temperature=0.0,
        )
        # 提取响应文本
        extracted_text = response.choices[0].message.content
        logger.info(f"图像文本提取成功: 长度 {len(extracted_text)}")
        return extracted_text
    except Exception as e:
        logger.error(f"图像文本提取失败: {image_path}, 错误: {e}", exc_info=True)
        return ""

# # 测试图像路径（需替换为实际文件）
# test_image_path = r"D:\LLM_Codes\Chapter3_RAG\SmartRecruit\data\resume\王宝强.jpg"  # 请确保文件存在
#
# try:
#     text = extract_text_from_image(test_image_path, client)
#     assert text, "提取文本为空"
#     logger.info(f"提取文本: {text[:100]}...")  # 打印前100字符
#     print("验证通过")
# except Exception as e:
#     logger.error(f"验证失败: {e}")
#     print("验证失败")