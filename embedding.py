"""
embedding.py —— 向量化与存储模块
功能：将 load.py 切分好的文本块通过 Embeddings 模型转换为向量，并存入 ChromaDB 向量数据库。
"""

import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# ChromaDB 持久化存储目录
VECTOR_STORE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")


def get_embeddings_model() -> OpenAIEmbeddings:
    """
    获取 Embeddings 模型实例。

    通过环境变量配置自定义的 API base（如 DeepSeek / 通义千问兼容接口）。
    所需环境变量：
        EMBEDDING_API_KEY     —— 你的 API Key（若不设置，回退使用 LLM_API_KEY）
        EMBEDDING_API_BASE    —— Embeddings 接口地址
        EMBEDDING_MODEL_NAME  —— Embeddings 模型名称（默认 text-embedding-ada-002）
    """
    api_key = os.getenv("EMBEDDING_API_KEY", os.getenv("LLM_API_KEY", ""))
    api_base = os.getenv("EMBEDDING_API_BASE", os.getenv("LLM_API_BASE", ""))
    model_name = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-ada-002")

    if not api_key:
        raise ValueError("请设置环境变量 EMBEDDING_API_KEY 或 LLM_API_KEY")

    kwargs = {
        "openai_api_key": api_key,
        "model": model_name,
    }
    if api_base:
        kwargs["openai_api_base"] = api_base

    return OpenAIEmbeddings(**kwargs)


def create_vector_store(chunks: list[str], persist_dir: str = VECTOR_STORE_DIR) -> Chroma:
    """
    将文本块列表向量化并存入 ChromaDB，返回可直接用于检索的向量数据库对象。

    参数:
        chunks:      文本块列表（由 load.py 产出）
        persist_dir: ChromaDB 持久化目录路径

    返回:
        Chroma 向量数据库对象（已持久化到磁盘）
    """
    if not chunks:
        raise ValueError("文本块列表为空，无法创建向量存储")

    embeddings = get_embeddings_model()
    print(f"[embedding] 正在将 {len(chunks)} 个文本块写入向量数据库...")
    print(f"[embedding] Embeddings 模型: {embeddings.model}")

    # Chroma.from_texts 会自动逐条调用 Embeddings API 并将结果持久化
    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    vector_store.persist()
    print(f"[embedding] 向量数据库已保存至: {persist_dir}")
    return vector_store


def load_vector_store(persist_dir: str = VECTOR_STORE_DIR) -> Chroma:
    """
    从磁盘加载已有的向量数据库，用于后续检索。

    参数:
        persist_dir: ChromaDB 持久化目录路径

    返回:
        Chroma 向量数据库对象（已加载）
    """
    if not os.path.exists(persist_dir):
        raise FileNotFoundError(
            f"向量数据库目录不存在: {persist_dir}\n请先运行 create_vector_store() 创建数据库。"
        )

    embeddings = get_embeddings_model()
    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )
    print(f"[embedding] 已从 {persist_dir} 加载向量数据库")
    return vector_store


# ============================================================
# 如果直接运行此文件，可快速测试 Embeddings + 存储流程
# ============================================================
if __name__ == "__main__":
    import sys
    from load import load_pdf_and_chunk

    if len(sys.argv) < 2:
        print("用法: python embedding.py <pdf文件路径>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    chunks = load_pdf_and_chunk(pdf_path)
    vector_store = create_vector_store(chunks)
    print(f"\n向量数据库条目数: {vector_store._collection.count()}")
    print("Embeddings 存储完成！")
