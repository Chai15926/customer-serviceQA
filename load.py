"""
load.py —— 文档处理模块
功能：读取本地 PDF 简历文件，提取全部文本，并按合适大小切分成文本块（chunks）。
"""

import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    读取本地 PDF 文件，提取全部文本内容。

    参数:
        pdf_path: PDF 文件的本地路径

    返回:
        提取出的完整文本字符串
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"找不到 PDF 文件: {pdf_path}")

    reader = PdfReader(pdf_path)
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)

    full_text = "\n".join(text_parts)
    if not full_text.strip():
        raise ValueError("PDF 文件中未能提取到任何文本内容，请检查文件是否为扫描件或图片型 PDF。")

    return full_text


def split_text_into_chunks(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[str]:
    """
    将长文本按语义边界切分成多个文本块，便于后续向量检索。

    参数:
        text:       待切分的完整文本
        chunk_size: 每个文本块的最大字符数（默认 500）
        chunk_overlap: 相邻块之间的重叠字符数（默认 50），用于保持语义连续性

    返回:
        切分后的文本块列表
    """
    # 中文文本对换行、句号、问号、感叹号等敏感 ——
    # RecursiveCharacterTextSplitter 会按 "\n\n" → "\n" → "。" → " " 的顺序递归切分
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )
    chunks = splitter.split_text(text)
    return chunks


def load_pdf_and_chunk(pdf_path: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """
    一站式函数：读取 PDF → 提取文本 → 切分成 chunks。

    参数:
        pdf_path:       PDF 文件路径
        chunk_size:     每个文本块的最大字符数
        chunk_overlap:  相邻块之间的重叠字符数

    返回:
        切分后的文本块列表
    """
    print(f"[load] 正在读取 PDF: {pdf_path}")
    full_text = extract_text_from_pdf(pdf_path)
    print(f"[load] 提取文本完成，共 {len(full_text)} 个字符")

    chunks = split_text_into_chunks(full_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"[load] 文本切分完成，共 {len(chunks)} 个文本块")
    return chunks


# ============================================================
# 如果直接运行此文件，可快速测试 PDF 加载效果
# ============================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python load.py <pdf文件路径>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    chunks = load_pdf_and_chunk(pdf_path)
    print(f"\n共生成 {len(chunks)} 个文本块:\n")
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk #{i + 1} ({len(chunk)} 字符) ---")
        print(chunk[:200])
        print()
