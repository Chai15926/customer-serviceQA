# AI 岗位学习指南 —— 基于 RAG 智能客服项目

## 项目概述

本项目是一个典型的 **RAG（检索增强生成）智能客服问答系统**。作为 AI 初学者，通过学习和修改这个项目，你可以掌握当前 AI 行业最核心的实战技能。

---

## 一、学到的核心技术点

### 1. RAG 架构（当前大模型落地最主流的技术方向）

整个项目的数据流是：**文档 → 分块 → 向量化 → 检索 → 大模型回答**，这是目前企业级 AI 应用的标准架构。理解这个完整链路对你面试非常有用。

### 2. 文档处理与文本分块（Document Chunking）

- 学习如何读取 PDF 文件（使用 `pypdf` 库）
- 掌握 **文本分块策略**：chunk_size、chunk_overlap、按语义边界切分
- 理解中文文本需要特殊处理分隔符（句号、问号、感叹号等）

### 3. Embedding 与向量检索

- 学习如何将文本转为向量（OpenAI Embeddings API）
- 了解 **ChromaDB 向量数据库**的创建、持久化和加载
- 理解 similarity_search 是如何通过余弦相似度找到最相关的文本块

### 4. LLM API 调用与大模型对接

- 学习调用 OpenAI 兼容接口（DeepSeek、通义千问等都兼容）
- 学习如何使用环境变量管理 API Key（安全最佳实践）
- 理解 temperature、max_tokens 等关键参数的作用

### 5. Prompt Engineering（提示词工程）

展示了如何编写 **结构化 Prompt**，约束模型只基于提供的上下文回答，不编造信息。包括角色设定、回答要求、上下文引用等技巧。

### 6. LangChain 生态的使用

- Text Splitter：智能文本分割
- Embeddings：文本向量化
- VectorStore：向量存储与检索

掌握这些能让你快速上手更多框架。

### 7. Gradio 快速搭建 AI 应用界面

用极少的代码搭建了带文件上传和对话界面的 Web 应用，展示 AI 项目能力的实用技能。

---

## 二、简历中可以写的亮点

| 技术关键词 | 对应能力 |
|---|---|
| RAG 架构 | 文档检索 + 大模型生成完整链路 |
| LangChain / ChromaDB | 向量数据库与检索框架 |
| PDF 解析 + 文本分块 | 非结构化数据处理能力 |
| Prompt Engineering | 提示词设计与优化 |
| API 集成 | 对接主流大模型服务 |
| Gradio | AI 应用原型开发 |

---

## 三、后续可以深入的方向

如果你想让这个项目更有竞争力，可以尝试：

1. **替换为更先进的 chunking 策略**：如语义分块、元数据过滤
2. **接入本地模型**：如 Ollama + Qwen，减少对外部 API 的依赖
3. **增加多轮对话记忆功能**：让 AI 能记住上下文
4. **添加评价机制**：评估回答质量，持续优化效果
5. **引入 FAISS / Milvus 等工业级向量数据库**：提升大规模检索性能
6. **微调 Embedding 模型**：针对特定领域优化向量质量

---

## 四、项目结构说明

```
customer-serviceQA/
├── app.py          # 主逻辑：Gradio 界面 + 问答流程
├── load.py         # 文档处理：PDF 读取 + 文本分块
├── embedding.py    # 向量化：Embedding API + ChromaDB 存储
├── requirements.txt # 依赖包列表
├── .env            # 环境变量配置（API Key 等）
└── chroma_db/      # 向量数据库（自动生成）
```

---

## 五、运行方式

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量（编辑 .env 文件）
LLM_API_KEY=your_api_key
LLM_API_BASE=https://api.deepseek.com/v1
LLM_MODEL_NAME=deepseek-chat

# 启动应用
python app.py --pdf 产品手册.pdf

# 或先上传 PDF 再点击网页中的"初始化知识库"按钮
python app.py
```

---

*本文档由 Claude 生成，助你快速入门 AI 开发并备战 AI 相关岗位。*
