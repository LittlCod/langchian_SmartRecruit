
from openai import OpenAI
from loguru import logger
import json
from typing import Dict, Any


# 简历结构化信息解析 ---
RESUME_PARSER_PROMPT = """
你是一个顶级的HR简历分析专家。请从以下简历文本中，提取出关键的结构化信息。

**严格遵守以下规则:**
1.  **提取字段**: 只提取以下字段：`name` (姓名), `gender` (性别), `age` (年龄), `work_experience` (工作年限)。
2.  **JSON格式**: 必须严格按照JSON格式输出，不要有任何额外的解释或Markdown标记。
3.  **逻辑推断**:
    - **姓名 (name)**: 通常是文本开头最明显的人名。
    - **性别 (gender)**: 从文本中明确的“男”或“女”字样判断。如果未提及，则为 "未提供"。
    - **年龄 (age)**: 根据出生年份、或直接描述的年龄计算。例如“1990年出生”在2024年应计算为34岁。如果无法推断，则为 -1。
    - **工作年限 (work_experience)**: 根据工作经历的总时长计算。例如“2020年7月至2023年7月”是3年。如果无法推断，则为 -1。
4.  **数值类型**: `age` 和 `work_experience` 必须是整数。

**简历文本:**
---
{resume_text}
---

**输出JSON:**
"""


def parse_resume_structure(resume_text: str, client: OpenAI) -> Dict[str, Any]:
    """
    使用LLM从简历文本中提取结构化的个人信息。
    """
    logger.info("开始使用LLM解析简历结构化信息...")
    try:
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": "你是一个顶级的HR简历分析专家。"},
                {"role": "user", "content": RESUME_PARSER_PROMPT.format(resume_text=resume_text)}
            ],
            temperature=0.0,
        )
        content = response.choices[0].message.content
        logger.debug(f"LLM原始解析结果: {content}")

        # 清理并加载JSON
        json_str = content.strip().removeprefix("```json").removesuffix("```").strip()
        structured_data = json.loads(json_str)

        # 数据清洗和验证
        structured_data['age'] = int(structured_data.get('age', -1))
        structured_data['work_experience'] = int(structured_data.get('work_experience', -1))
        structured_data['gender'] = structured_data.get('gender', '未提供')

        logger.info(f"简历结构化信息解析成功: {structured_data}")
        return structured_data

    except Exception as e:
        logger.error(f"解析简历结构化信息失败: {e}", exc_info=True)
        return {"name": "未知", "gender": "未提供", "age": -1, "work_experience": -1}

# 初始化客户端
client = OpenAI(api_key="sk-b052cdd2b23249f5bbe1949928a2600a", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
# 验证parse_resume_structure函数（基于脚本__main__中的部分）
test_resume_text = "李明 23岁 具有5年工作经验 毕业于北京大学...."  # 从加载函数获取或手动提供
try:
    structured_data = parse_resume_structure(test_resume_text, client)
    assert isinstance(structured_data, dict) and "name" in structured_data
    logger.info(f"解析成功: {structured_data}")
except Exception as e:
    logger.error(f"结构化解析失败: {e}")
