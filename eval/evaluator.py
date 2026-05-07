# evaluator.py
# 一个用于评估RAG系统性能的脚本，包括健全性检查和端到端评估。

# --- 步骤 1: 导入所需库与模块 ---
# 1.1 导入标准库，用于基本的文件操作、JSON处理和异步编程。
import json
import os
import asyncio
from typing import List, Dict, Any, Tuple

# 1.2 导入第三方库，包括评估框架和模型调用模块。
from datasets import Dataset  # 用于创建Ragas评估所需的数据集格式
from langchain_openai import ChatOpenAI  # 用于调用LLM
from loguru import logger  # 高性能日志库
from ragas import evaluate  # Ragas评估框架的核心函数
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)  # 导入评估指标
from ragas.evaluation import EvaluationResult  # 评估结果的类型提示
from langchain_core.documents import Document  # LangChain的文档对象
from langchain_core.runnables import RunnableConfig  # LangChain链的配置
from langchain_core.output_parsers import StrOutputParser  # 用于从LLM输出中解析字符串

# 1.3 导入项目内部模块，复用RAG系统的核心组件。
from config import config
from milvus_model.hybrid import BGEM3EmbeddingFunction
from rag.chain import llm, retriever, REWRITE_PROMPT, ANSWER_PROMPT, _format_docs


# --- 步骤 2: 定义评估专用组件与全局配置 ---
# 2.1 定义 Ragas 嵌入模型适配器，以兼容Ragas和我们自己的BGE-M3模型。
class RagasBgeM3EmbeddingsAdapter:
    """Ragas框架的嵌入模型适配器，将BGEM3EmbeddingFunction封装为Ragas可识别的接口。"""

    def __init__(self, *args, **kwargs):
        # 2.1.1 初始化项目自用的BGEM3嵌入模型。加载模型并初始化。
        self.bge_function = BGEM3EmbeddingFunction(*args, **kwargs)

    def embed_query(self, text: str) -> List[float]:
        """实现 Ragas 的查询嵌入接口，将单个文本转为稠密向量。"""
        # 调用BGEM3的编码函数，并提取"dense"向量的第一个元素。
        return self.bge_function.encode_queries([text])["dense"][0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """实现 Ragas 的文档嵌入接口，将文本列表转为向量列表。"""
        # 调用BGEM3的文档编码函数，并提取"dense"向量列表。
        return self.bge_function.encode_documents(texts)["dense"]


# 2.2 配置日志记录器，将日志输出到文件，方便调试和追溯。
logger.add(os.path.join(config.LOG_DIR, "evaluator.log"), rotation="10 MB", encoding="utf-8")

# 2.3 初始化Ragas评估专用的LLM实例，保证评估过程的独立性。
# 使用Qwen-plus模型作为评估LLM，它将用于判断答案的忠实度和相关性等。
ragas_llm = ChatOpenAI(
    model="qwen-plus",
    api_key=config.DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 2.4 初始化Ragas评估专用的嵌入模型实例，用于计算语义相似度。
# 使用上面定义的适配器，将本地的BGE-M3模型封装起来。
ragas_embeddings = RagasBgeM3EmbeddingsAdapter(
    model_name=os.path.join(config.MODEL_PATH, config.EMBEDDING_MODEL),
    device="cpu",
    use_fp16=False,
)


# --- 步骤 3: 封装RAG管道调用逻辑 ---
async def get_rag_response_for_evaluation(
        query: str, params: Dict[str, Any]
) -> Tuple[str, List[Document]]:
    """
    专为评估设计的RAG管道调用函数。
    此函数模拟完整的RAG流程，并返回评估所需的最终答案和检索到的原始文档。

    Args:
        query (str): 用户的原始查询。
        params (Dict[str, Any]): 检索器所需的额外参数，如检索数量等。

    Returns:
        Tuple[str, List[Document]]:
        - final_answer (str): RAG管道生成的最终答案字符串。
        - retrieved_docs (List[Document]): 从数据库中检索出的原始文档列表。
    """
    # 3.1 使用重写提示词和LLM，对原始查询进行改写，以提高检索效果。
    rewritten_question = await (REWRITE_PROMPT | llm | StrOutputParser()).ainvoke(
        {"input": query, "chat_history": []},
        config=RunnableConfig()
    )
    logger.info(f"评估 - 原始查询: '{query}' -> 改写后查询: '{rewritten_question}'")

    # 3.2 调用检索器，根据改写后的问题获取相关文档。
    try:
        retrieved_docs = await retriever.aget_relevant_documents(
            query=rewritten_question,
            params=params
        )
        logger.info(f"评估 - 检索到 {len(retrieved_docs)} 个文档块。")
    except Exception as e:
        logger.error(f"评估 - 检索器执行失败: {e}", exc_info=True)
        return "[]", []  # 检索失败时返回空结果，避免程序中断。

    # 3.3 对检索到的文档进行去重和格式化，作为LLM的输入上下文。
    unique_contexts = []
    seen_content = set()
    for doc in retrieved_docs:
        # 通过检查文档内容是否已存在来去重。
        if doc.page_content not in seen_content:
            unique_contexts.append(doc.page_content)
            seen_content.add(doc.page_content)
    logger.info(f"评估 - 原始检索到 {len(retrieved_docs)} 个上下文块，去重后剩余 {len(unique_contexts)} 个。")
    # 将去重后的文档内容合并成一个字符串。
    context_str = "\n".join(unique_contexts)

    # 3.4 使用答案生成提示词和LLM，结合上下文和原始问题生成最终答案。
    final_answer = await (ANSWER_PROMPT | llm | StrOutputParser()).ainvoke(
        {"input": query, "context": context_str},
        config=RunnableConfig()
    )
    logger.info(f"评估 - RAG管道生成原始答案: {final_answer}")

    # 3.5 返回最终答案和检索到的文档，这是Ragas评估所必需的两个关键要素。
    return final_answer, retrieved_docs


# --- 步骤 4: 定义核心评估函数 ---
def run_evaluation(dataset: Dataset) -> EvaluationResult:
    """
    封装了Ragas评估的通用函数，负责执行评估并返回结果。
    """
    try:
        # 4.1 调用Ragas的evaluate函数，传入数据集、评估指标和LLM/嵌入模型。
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )
        return result
    except Exception as e:
        logger.error(f"Ragas评估过程中出错: {e}")
        # 如果评估失败，抛出异常以中断程序。
        raise


# --- 步骤 5: 评估流程健全性检查 (Sanity Check) ---
def run_evaluator_sanity_check():
    """
    Part 1: 验证评估流程本身是否可靠。
    此函数构造一个完美的中文问答场景，期望评估指标达到理想值。
    """
    print("\n--- Part 1: 评估流程健全性检查 ---")

    # 5.1 构造一个独立的、完美的中文测试样本。
    question = "谁是姚明？"  # 用户问题
    # `ground_truth`用于与LLM的答案进行比较，评估答案相关性。
    ground_truth = "姚明是一名来自中国的篮球运动员，曾在NBA休斯顿火箭队效力。"
    # `contexts`是模拟的理想检索结果，用于评估忠实度和上下文指标。
    contexts = ["姚明是一名来自中国的篮球运动员，他身高2米26，司职中锋，曾在NBA休斯顿火箭队效力。"]
    # `answer`是基于上下文生成的完美答案，用于评估答案相关性。
    perfect_answer = "姚明是来自中国的篮球运动员，曾在休斯顿火箭队打球。"

    # 5.2 准备Ragas评估所需的Dataset格式。
    dataset = Dataset.from_dict({
        "question": [question],
        "answer": [perfect_answer],
        "contexts": [contexts],
        "ground_truth": [ground_truth],
    })

    print(f"健全性检查 - 输入问题: {question}")

    # 5.3 执行评估并打印结果。
    result = run_evaluation(dataset)
    print("\n健全性检查 - 评估结果:")
    print(result)

    # 5.4 断言检查，验证核心指标是否达到预期。
    # 由于是单行评估，Ragas返回的指标是包含单个元素的列表，需要使用索引[0]来获取值。
    if result['faithfulness'][0] < 1.0:
        print("[警告] 健全性检查未通过！Faithfulness 应为 1.0。")
    else:
        print("[成功] 健全性检查通过！评估工具链工作正常。")
    print("--- 健全性检查结束 ---")


# --- 步骤 6: RAG系统端到端评估 ---
async def run_end_to_end_evaluation():
    """
    Part 2: 评估RAG系统的真实性能。
    此函数调用完整的RAG管道，并使用Ragas评估其生成结果。
    """
    print("\n--- Part 2: RAG系统端到端评估 ---")

    # 6.1 定义一个真实的测试查询和对应的“黄金标准”答案。
    test_query = "需要一位熟悉AI大模型的产品经理"
    # `ground_truth`是人工定义的理想答案，用于评估答案相关性。
    test_ground_truth = "该候选人拥有AI大模型相关的产品管理经验，具备大语言模型（LLM）、NLP等技术理解能力，与岗位需求高度匹配，完全符合用人需求。熟悉AI模型开发流程、应用场景设计及产品化落地。他具备需求分析、产品规划、跨团队协作和商业化方案设计能力"
    # 定义检索器参数。
    test_params = {"count": 1}

    # 6.2 调用为评估封装的RAG管道函数，获取LLM生成的答案和检索到的文档。
    generated_json, retrieved_docs = await get_rag_response_for_evaluation(test_query, test_params)

    # 6.3 解析RAG返回的JSON，提取自然语言答案。
    natural_language_answer = ""
    try:
        # 清理可能存在的Markdown代码块标记。
        clean_json_str = generated_json.strip().removeprefix("```json").removesuffix("```").strip()
        # 将清理后的字符串解析为JSON对象。
        response_data = json.loads(clean_json_str)
        # 从JSON中提取推荐理由作为评估答案。
        if response_data and isinstance(response_data, list):
            natural_language_answer = response_data[0].get("reason", "")
    except (json.JSONDecodeError, IndexError, AttributeError) as e:
        # 捕获JSON解析错误，并记录警告。
        logger.warning(f"无法从RAG响应中解析推荐理由: {generated_json}, 错误: {e}")
        # 保持答案为空，评估结果将反映出解析失败的问题。
        natural_language_answer = ""

    # 6.4 将检索到的Document对象列表转换为Ragas所需的字符串列表。
    string_contexts = [doc.page_content for doc in retrieved_docs]

    # 6.5 准备Ragas评估所需的数据集。
    dataset = Dataset.from_dict({
        "question": [test_query],
        "answer": [natural_language_answer],
        "contexts": [string_contexts],
        "ground_truth": [test_ground_truth],
    })

    print(f"端到端评估 - 输入问题: {test_query}")
    print(f"端到端评估 - RAG生成的推荐理由: {natural_language_answer or '未能成功解析'}")

    # 6.6 执行评估并打印结果。
    result = run_evaluation(dataset)
    print("\n端到端评估 - 评估结果:")
    print(result)
    print("--- 端到端评估结束 ---")


# --- 步骤 7: 主程序入口 ---
async def main():
    """
    主异步函数，用于统一管理和执行所有评估任务。
    """
    # 7.1 首先运行健全性检查，这是一个同步任务。
    run_evaluator_sanity_check()
    # 7.2 接着运行端到端评估，这是一个异步任务，使用await调用。
    await run_end_to_end_evaluation()


if __name__ == "__main__":
    # 7.3 使用asyncio.run()启动事件循环，执行主异步函数。
    asyncio.run(main())