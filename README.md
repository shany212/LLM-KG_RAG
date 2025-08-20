# LLM-KG_RAG
# 智能医疗知识图谱问答系统 (灵枢)

## 📖 项目简介

本项目是一个先进的**智能医疗问答系统**，其核心是**知识图谱驱动的检索增强生成（Knowledge Graph-Driven RAG）**架构。系统能够理解用户的自然语言提问，通过高精度的命名实体识别（NER）和意图识别，将问题转化为对医疗知识图谱（Neo4j）的精确查询，再将检索到的结构化知识与大语言模型（ChatGLM）相结合，生成精准、可靠且可解释的医疗健康答案。

本项目旨在解决通用大模型在专业领域容易产生“知识幻觉”的核心痛病，为用户提供值得信赖的医疗信息服务。

## ✨ 主要特性

* **高精度实体与意图识别**: 采用先进的BERT和text2vec-base-chinese模型，准确识别用户问题中的医疗实体（疾病、症状等）和真实意图。
* **知识图谱驱动**: 所有答案的核心依据均来自结构化的医疗知识图谱（Neo4j），确保了信息的精准性和可追溯性。
* **多跳推理能力**: 利用图数据库的特性，能够回答需要多步推理的复杂关联问题。
* **大模型增强生成**: 结合本地化的ChatGLM2-6B大语言模型，将图谱返回的结构化知识，转化为流畅、完整、人性化的自然语言回答。
* **模块化架构**: 系统采用Flask框架，各功能模块（KG、NER、Intent、LLM）高度解耦，易于维护和扩展。

## 🏛️ 系统架构

本项目遵循一个清晰的“理解 -> 检索 -> 生成”流程：

```
用户提问 (自然语言)
     │
     ▼
┌───────────────────────┐
│  1. 自然语言理解模块  │
│  (NER & Intent)       │
├───────────────────────┤
│ 实体: [高血压]        │
│ 意图: query_symptom   │
└───────────────────────┘
     │
     ▼
┌───────────────────────┐
│  2. 知识图谱查询模块  │
│  (KG Module)          │
├───────────────────────┤
│ 生成Cypher查询语句    │
└───────────────────────┘
     │
     ▼
┌───────────────────────┐
│  Neo4j 知识图谱数据库 │
└───────────────────────┘
     │ (返回结构化事实)
     ▼
┌───────────────────────┐
│  3. 大语言模型生成模块│
│  (LLM Module)         │
├───────────────────────┤
│   构建Prompt          │
│   (原始问题 + 事实)   │
└───────────────────────┘
     │
     ▼
最终答案 (自然语言)
```

## 🛠️ 技术栈

* **后端框架**: Flask
* **数据库**: Neo4j
* **核心库**: LangChain, Transformers, Sentence-Transformers, PyTorch, Neo4j-Driver
* **大语言模型 (LLM)**: ChatGLM2-6B (INT4量化版)
* **命名实体识别 (NER) 模型**: `UebPW/bert-base-chinese-cmeee-ner`
* **意图识别模型**: `shibing624/text2vec-base-chinese`

## 🚀 部署与运行指南

请严格按照以下步骤进行环境配置和部署。

### 1. 先决条件

* Python 3.10+
* Git 和 Git LFS
* Conda 或 venv
* 已启动的 Neo4j 数据库服务 (推荐使用 Neo4j Desktop)

### 2. 克隆项目

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

### 3. 配置环境并安装依赖

建议使用虚拟环境以避免包冲突。

```bash
# 创建并激活虚拟环境 (以venv为例)
python -m venv .venv
source .venv/bin/activate 

# 安装所有必需的Python库
pip install -r requirements.txt
```

**`requirements.txt` 文件内容:**
```
accelerate==1.10.0
annotated-types==0.7.0
anyio==4.10.0
blinker==1.9.0
certifi==2025.8.3
charset-normalizer==3.4.3
click==8.2.1
cpm-kernels==1.0.11
filelock==3.19.1
Flask==3.1.1
flask-cors==6.0.1
fsspec==2025.7.0
greenlet==3.2.4
h11==0.16.0
hf-xet==1.1.8
httpcore==1.0.9
httpx==0.28.1
huggingface-hub==0.34.4
idna==3.10
itsdangerous==2.2.0
jieba==0.42.1
Jinja2==3.1.6
joblib==1.5.1
jsonpatch==1.33
jsonpointer==3.0.0
langchain==0.3.27
langchain-core==0.3.74
langchain-text-splitters==0.3.9
langsmith==0.4.14
MarkupSafe==3.0.2
modelscope==1.29.0
mpmath==1.3.0
neo4j==5.28.2
networkx==3.5
numpy==2.3.2
nvidia-cublas-cu12==12.8.4.1
nvidia-cuda-cupti-cu12==12.8.90
nvidia-cuda-nvrtc-cu12==12.8.93
nvidia-cuda-runtime-cu12==12.8.90
nvidia-cudnn-cu12==9.10.2.21
nvidia-cufft-cu12==11.3.3.83
nvidia-cufile-cu12==1.13.1.3
nvidia-curand-cu12==10.3.9.90
nvidia-cusolver-cu12==11.7.3.90
nvidia-cusparse-cu12==12.5.8.93
nvidia-cusparselt-cu12==0.7.1
nvidia-nccl-cu12==2.27.3
nvidia-nvjitlink-cu12==12.8.93
nvidia-nvtx-cu12==12.8.90
orjson==3.11.2
packaging==25.0
pandas==2.3.1
pillow==11.3.0
psutil==7.0.0
pydantic==2.11.7
pydantic_core==2.33.2
python-dateutil==2.9.0.post0
pytz==2025.2
PyYAML==6.0.2
regex==2025.7.33
requests==2.32.5
requests-toolbelt==1.0.0
safetensors==0.6.2
scikit-learn==1.7.1
scipy==1.16.1
sentence-transformers==5.1.0
sentencepiece==0.2.1
setuptools==80.9.0
six==1.17.0
sniffio==1.3.1
SQLAlchemy==2.0.43
sympy==1.14.0
tenacity==9.1.2
threadpoolctl==3.6.0
tokenizers==0.19.1
torch==2.8.0
tqdm==4.67.1
transformers==4.40.2
triton==3.4.0
typing-inspection==0.4.1
typing_extensions==4.14.1
tzdata==2025.2
urllib3==2.5.0
Werkzeug==3.1.3
zstandard==0.24.0

```

### 4. 下载预训练模型

在项目根目录下创建`models`文件夹，并下载所有必需的模型。

* **ChatGLM2-6B-INT4**:
    * 从ModelScope或Hugging Face下载，并将其存放到：`./models/modelscope/ZhipuAI/chatglm2-6b-int4/`

* **NER模型**:
    * 从Hugging Face下载 `https://huggingface.co/iioSnail/bert-base-chinese-medical-ner` 模型。
    * 将其存放到：`./models/bert-base-chinese-medical-ner/`

* **意图识别模型**:
    * 从Hugging Face下载 `shibing624/text2vec-base-chinese` 模型。
    * 将其存放到：`./models/text2vec-base-chinese/`

> **提示**: 您可以使用我们项目中的 `download_model.py` 脚本（需自行编写或使用之前版本）来自动完成下载和放置。或者使用魔法进入huggingface直接下载。

### 5. 构建知识图谱

1.  **准备数据**: 将您的医疗数据文件（例如 `medical.json`）放置到项目根目录。
2.  **配置连接**: 打开 `dataset_importer.py` (知识图谱导入脚本)，确保其中的Neo4j连接信息（URI, 用户名, 密码）正确无误。
3.  **运行导入脚本**:
    ```bash
    python importer.py 
    ```
    等待脚本执行完成，您的Neo4j数据库中就会充满医疗知识。

### 6. 配置并启动应用

1.  **修改配置**: 打开`config.py`文件，找到`NEO4J_PASSWORD`，将其修改为您自己的Neo4j数据库密码。同时，请再次确认所有模型路径配置正确。
2.  **启动后端服务**:
    ```bash
    python app.py
    ```
    看到Flask启动信息后，表示服务已成功运行在 `5000` 端口。

### 7. API接口调用

服务启动后，您可以通过POST请求与系统进行交互。

**请求示例 (使用curl):**

```bash
curl -X POST [http://127.0.0.1:5000/api/chat](http://127.0.0.1:5000/api/chat) \
     -H "Content-Type: application/json" \
     -d '{"query": "高血压有什么症状？"}'
```

**成功响应示例:**
```json
{
  "entities": [
    {
      "name": "高血压",
      "type": "Disease"
    }
  ],
  "final_answer": "根据知识图谱中的信息，高血压的常见症状包括头晕、头痛、颈项板紧、疲劳和心悸等。不过，很多高血压患者在早期可能没有明显症状，建议您定期测量血压。",
  "intent": "query_symptom",
  "kg_context": "头晕、头痛、颈项板紧、疲劳、心悸",
  "query": "高血压有什么症状？"
}
```

## 📁 项目结构

```
.
├── app.py                  # Flask应用入口
├── config.py               # 全局配置文件
├── main_handler.py         # 核心业务逻辑处理器
├── dataset_importer.py             # KG数据导入脚本
├── requirements.txt        # Python依赖列表
├── README.md               # 本文档
├── data/
│   └── data.json           # 处理数据
│   └── medical.json        # 原始数据(数据来自https://github.com/baiyang2464/chatbot-base-on-Knowledge-Graph/tree/master)
├── models/                 # 存放所有预训练模型
│   ├── chatglm2-6b-int4/
│   ├── UebPW/bert-base-chinese-cmeee-ner/
│   └── text2vec-base-chinese/
└── modules/                # 核心功能模块
    ├── kg_module.py        # 知识图谱查询模块
    ├── llm_module.py       # 大语言模型生成模块 (LangChain)
    ├── medical_ner_module.py   # 医疗命名实体识别模块
    ├── medical_intent_module.py# 医疗意图识别模块
    └── ner_intent_module.py    # NER与意图识别的组合模块
```

## 🌟 未来展望

* **模型微调**: 在自有数据上对NER模型进行微调、并实现命名实体识别后的自动分类，以提升特定场景的准确率。
* **增加实体类型**: 通过微调或更强大的LLM辅助识别，增加对“食物”、“手术”等更多实体类型的支持。
* **对话历史管理**: 集成对话记忆功能，支持多轮对话。
* **前端界面**: 开发一个用户友好的前端界面，提升交互体验。
---
