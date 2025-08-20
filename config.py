# config.py
import os
import torch

class Config:
    # --- Neo4j 知识图谱配置 ---
    NEO4J_URI = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
    NEO4J_USER = os.environ.get('NEO4J_USER', 'neo4j')
    NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD', '123456') # 请修改为你的密码

  
    # ChatGLM 模型本地路径
    # 请确保您的glm模型实际存放在 './models/chatglm2-6b-int4'
    CHATGLM_PATH = os.path.join('./models/modelscope/ZhipuAI/', 'chatglm2-6b-int4')

    # NER 模型本地路径
    # 根据您的描述，它现在位于 './models/bert-base-chinese-medical-ner'
    NER_MODEL_NAME = os.path.join('./models', 'bert-base-chinese-medical-ner')
    
    # 意图识别模型本地路径
    # 根据您的描述，它现在位于 './models/text2vec-base-chinese'
    INTENT_MODEL_NAME = os.path.join('./models', 'text2vec-base-chinese')

    # --- LangChain配置 ---
    LANGCHAIN_CONFIG = {
        'verbose': True,
        'temperature': 0.8,
        'max_tokens': 512,
        'top_p': 0.9
    }
    
    # --- LLM 生成参数 ---
    GENERATION_CONFIG = {
        'max_length': 2048,
        'max_new_tokens': 512,  # 添加新token数量限制
        'temperature': 0.8,
        'top_p': 0.9,
        'do_sample': True,
        'repetition_penalty': 1.1
    }

    # --- 设备配置 ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"