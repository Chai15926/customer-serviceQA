"""
app.py —— 主逻辑与 Gradio 网页界面
功能：串联文档加载、向量检索与大模型回答，提供对话式问答界面。

使用方式：
    python app.py                      # 使用已有的向量数据库
    python app.py --pdf 产品手册.pdf        # 从指定 PDF 构建向量数据库后启动
    python app.py --pdf 产品手册.pdf --rebuild  # 强制重建向量数据库

环境变量（按需设置，不硬编码）：
    LLM_API_KEY         —— 大模型 API Key
    LLM_API_BASE        —— 大模型 API 地址（如 https://api.deepseek.com/v1）
    LLM_MODEL_NAME      —— 大模型名称（如 deepseek-chat、qwen-plus）
    EMBEDDING_API_KEY   —— Embeddings API Key（不设置则回退到 LLM_API_KEY）
    EMBEDDING_API_BASE  —— Embeddings API 地址
    EMBEDDING_MODEL_NAME—— Embeddings 模型名称
"""
from dotenv import load_dotenv
load_dotenv()
import os
import argparse
import gradio as gr
from openai import OpenAI

from load import load_pdf_and_chunk
from embedding import create_vector_store, load_vector_store, VECTOR_STORE_DIR


# ============================================================
# 大模型调用
# ============================================================
def get_llm_client() -> OpenAI:
    """获取 OpenAI 兼容客户端（可用于 DeepSeek、通义千问等兼容接口）"""
    api_key = os.getenv("LLM_API_KEY", "")
    api_base = os.getenv("LLM_API_BASE", "")
    if not api_key:
        raise ValueError("请设置环境变量 LLM_API_KEY")
    kwargs = {"api_key": api_key}
    if api_base:
        kwargs["base_url"] = api_base
    return OpenAI(**kwargs)


def build_prompt(context: str, question: str) -> str:
    """
    客服助手 Prompt：根据产品手册回答用户问题。
    """
    return (
        "你是一位专业、耐心的客服助手。请严格根据以下产品手册的内容回答用户的问题。\n"
        "要求：\n"
        "1. 回答要准确、简洁、易懂。\n"
        "2. 如果用户的问题在手册中没有提及，请礼貌地告知用户「暂时没有找到相关信息，建议联系人工客服」。\n"
        "3. 不要编造任何内容，不要提供手册以外的信息。\n\n"
        f"【产品手册内容】\n{context}\n\n"
        f"【用户问题】\n{question}\n\n"
        "【客服回答】\n"
    )


def query_llm(prompt: str) -> str:
    """调用大模型生成回答"""
    client = get_llm_client()
    model = os.getenv("LLM_MODEL_NAME", "deepseek-chat")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,   # 降低随机性，让回答更忠于手册内容
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


# ============================================================
# 核心问答逻辑（Gradio ChatInterface 回调函数）
# ============================================================
class CustomerServiceQA:
    """智能客服系统，封装文档加载、检索与回答的完整流程"""

    def __init__(self):
        self.vector_store = None

    def init_or_load_db(self, pdf_path: str | None = None, rebuild: bool = False):
        """
        初始化向量数据库：
        - 若指定了 pdf_path 且数据库不存在 / 要求重建 → 重新构建
        - 否则从磁盘加载已有数据库
        """
        if pdf_path and (rebuild or not os.path.exists(VECTOR_STORE_DIR)):
            print(f"[app] 正在从 PDF 构建向量数据库: {pdf_path}")
            chunks = load_pdf_and_chunk(pdf_path)
            self.vector_store = create_vector_store(chunks)
        else:
            print(f"[app] 加载已有向量数据库: {VECTOR_STORE_DIR}")
            self.vector_store = load_vector_store()

    def answer(
        self, message: str, history: list[list[str]]
    ) -> str:
        """
        Gradio ChatInterface 的回调：接收用户消息，返回 AI 回答。

        参数:
            message: 用户当前输入的消息文本
            history: 对话历史，格式为 [[user_msg1, bot_msg1], [user_msg2, bot_msg2], ...]

        返回:
            AI 回答文本
        """
        if self.vector_store is None:
            return "⚠ 系统未初始化，请先上传 产品手册。"

        # 1. 在向量数据库中检索与问题最相关的 k 个文本块
        k = int(os.getenv("RETRIEVAL_K", "4"))
        docs = self.vector_store.similarity_search(message, k=k)

        # 2. 将检索到的文本块拼接为上下文
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])

        # 3. 组装 Prompt 并调用大模型
        prompt = build_prompt(context, message)

        try:
            answer = query_llm(prompt)
            return answer
        except Exception as e:
            return f"❌ 调用大模型时发生错误: {e}"


# ============================================================
# Gradio 界面构建
# ============================================================
def create_ui(qa_system: CustomerServiceQA) -> gr.Blocks:
    with gr.Blocks(title="💬 智能客服助手") as demo:
        gr.Markdown(
            "# 💬 智能客服助手\n"
            "上传一份产品手册或帮助文档（PDF），然后针对产品功能、使用方法、售后政策等提问。"
        )

        with gr.Row():
            with gr.Column(scale=1):
                pdf_file = gr.File(
                    label="上传 PDF 产品手册",
                    file_types=[".pdf"],
                )
                init_btn = gr.Button("🔧 初始化知识库", variant="primary")
                init_status = gr.Textbox(label="初始化状态", interactive=False, lines=3)

            with gr.Column(scale=2):
                chat = gr.ChatInterface(
                    fn=qa_system.answer,
                    chatbot=gr.Chatbot(height=500),
                    textbox=gr.Textbox(
                        placeholder="请输入你的问题，例如：如何退货？产品支持哪些支付方式？",
                        container=False,
                        scale=7,
                    ),
                    title="产品咨询",
                    description="在下方输入问题，客服助手将根据手册内容作答。",
                    examples=[
                        "如何申请售后服务？",
                        "这款产品的主要功能有哪些？",
                        "支持哪些支付方式？",
                        "退货流程是怎样的？",
                        "保修期是多长时间？",
                    ],
                )
        # ... 其余部分保持不变

        def on_init_click(pdf_file_obj):
            """处理「初始化/重建」按钮点击事件"""
            if pdf_file_obj is None:
                try:
                    qa_system.init_or_load_db(pdf_path=None, rebuild=False)
                    return "✅ 已从磁盘加载已有的向量数据库。\n现在可以在右侧对话框中提问了。"
                except FileNotFoundError:
                    return "⚠ 本地没有向量数据库，且未上传 PDF 文件。\n请先上传 PDF 再点击初始化。"
                except Exception as e:
                    return f"❌ 初始化失败: {e}"

            try:
                pdf_path = pdf_file_obj.name
                qa_system.init_or_load_db(pdf_path=pdf_path, rebuild=True)
                return f"✅ 已成功处理 PDF 并构建向量数据库。\n文件名: {os.path.basename(pdf_path)}\n现在可以在右侧对话框中提问了。"
            except Exception as e:
                return f"❌ 初始化失败: {e}"

        init_btn.click(fn=on_init_click, inputs=[pdf_file], outputs=[init_status])

    return demo


# ============================================================
# 程序入口
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="智能客服助手")
    parser.add_argument(
        "--pdf",
        type=str,
        default=None,
        help="启动时加载的 PDF 产品手册文件路径",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="强制重新构建向量数据库（即使已有本地向量库）",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Gradio 服务端口（默认 7860）",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="是否生成公网分享链接",
    )
    args = parser.parse_args()

    # 启动时初始化向量数据库
    qa = CustomerServiceQA()

    if args.pdf or os.path.exists(VECTOR_STORE_DIR):
        qa.init_or_load_db(pdf_path=args.pdf, rebuild=args.rebuild)
        print("[app] 初始化完成，正在启动 Gradio 界面...")
    else:
        print("[app] 未指定 PDF 且本地无向量数据库，请在网页界面上传 PDF 后点击初始化按钮。")

    # 启动 Gradio 界面
    demo = create_ui(qa)
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        inbrowser=True,
    )
