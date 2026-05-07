"""
本地模型下载工具

将 BGE-M3 和 BGE-Reranker-Base 从 HuggingFace 下载到项目的 models/ 目录。
下载后的目录结构与项目代码中的加载路径一致。

用法：
    cd 项目根目录
    python utils/model_download.py
"""

import os
from huggingface_hub import snapshot_download

# 模型目录（项目根目录/models）
MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models"
)

MODELS = [
    {
        "repo_id": "BAAI/bge-m3",
        "local_name": "bge-m3",
        "description": "BGE-M3 嵌入模型（~2.1GB，稠密+稀疏向量）",
    },
    {
        "repo_id": "BAAI/bge-reranker-base",
        "local_name": "bge-reranker-base",
        "description": "BGE-Reranker-Base 重排模型（~1.0GB）",
    },
]


def download_model(repo_id: str, local_name: str) -> str:
    """下载单个模型到 models/<local_name>/ 目录"""
    target_dir = os.path.join(MODELS_DIR, local_name)

    # 如果目录已存在且有内容，跳过
    if os.path.isdir(target_dir) and os.listdir(target_dir):
        print(f"  [跳过] {local_name} 已存在: {target_dir}")
        return target_dir

    os.makedirs(target_dir, exist_ok=True)

    print(f"  [下载] {repo_id} -> {target_dir}")
    model_dir = snapshot_download(
        repo_id=repo_id,
        local_dir=target_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"  [完成] {local_name}")
    return model_dir


if __name__ == "__main__":
    print(f"模型保存目录: {MODELS_DIR}\n")

    for model in MODELS:
        print(f"--- {model['description']} ---")
        download_model(model["repo_id"], model["local_name"])
        print()

    print("全部模型下载完成！")
