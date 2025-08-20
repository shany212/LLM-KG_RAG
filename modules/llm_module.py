# /opt/tkxt/modules/llm_module.py

import logging
import os
import torch
from transformers import AutoTokenizer, AutoModel
# 新增导入
from langchain.llms.base import LLM
from pydantic import PrivateAttr

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 模型路徑配置 (直接在這裡定義，方便測試) ---
CHATGLM_PATH = '/opt/tkxt/models/modelscope/ZhipuAI/chatglm2-6b-int4'

# --- LLM 生成參數 ---
GENERATION_CONFIG = {
    'max_length': 2048,
    'temperature': 0.8,
    'top_p': 0.9,
    'do_sample': True,  
}

# --- 設備配置 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model_and_tokenizer():
    """
    加載本地的ChatGLM模型和分詞器。
    """
    logger.info(f"正在從本地路徑加載模型: {CHATGLM_PATH}")
    
    if not os.path.isdir(CHATGLM_PATH):
        error_msg = f"模型路徑不存在: '{CHATGLM_PATH}'。請檢查路徑是否正確。"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    try:
        tokenizer = AutoTokenizer.from_pretrained(CHATGLM_PATH, trust_remote_code=True)
        model = AutoModel.from_pretrained(CHATGLM_PATH, trust_remote_code=True).half().to(DEVICE)
        model = model.eval()
        
        logger.info("✅ 模型和分詞器加載成功！")
        return model, tokenizer

    except Exception as e:
        logger.error(f"加載模型失敗: {e}", exc_info=True)
        return None, None

def generate_answer(model, tokenizer, query, context="", gen_kwargs=None):
    """
    使用加載好的模型和分詞器生成回答。
    """
    prompt = f"""你是一名专业医学助手。请严格遵循以下规则作答：
- 若[知识库信息]中包含与问题直接相关的内容，必须严格依据其内容回答，不要编造或引入未提供的事实。
- 若知识库信息不足或不相关，请明确说明“我未在知识库中找到足够的信息”，必要时仅给出谨慎的通用性建议。
- 回答应准确、简洁、可执行，可按要点分条说明。
- 回答结尾必须追加：我只是一个语言模型，具体情况请您咨询医生，提供的方案仅供参考。

[知识库信息]
{context}

[用戶提问]
{query}

[你的回答]
"""
    logger.info(f"構建的最終Prompt:\n{prompt}")

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        # 合并默认与调用时的生成参数
        final_cfg = {**GENERATION_CONFIG}
        if gen_kwargs:
            final_cfg.update(gen_kwargs)
        # 若设置了采样相关参数而未显式指定 do_sample，则默认开启采样
        if ('temperature' in final_cfg or 'top_p' in final_cfg) and 'do_sample' not in final_cfg:
            final_cfg['do_sample'] = True

        response_ids = model.generate(**inputs, **final_cfg)
        input_length = inputs.input_ids.shape[1]
        response_text = tokenizer.decode(response_ids[0][input_length:], skip_special_tokens=True)
        
        return response_text.strip()

    except Exception as e:
        logger.error(f"生成答案時發生錯誤: {e}", exc_info=True)
        return "抱歉，生成答案時遇到了技術問題。"

class ChatGLMForLangChain(LLM):
    _model = PrivateAttr(default=None)
    _tokenizer = PrivateAttr(default=None)
    _device = PrivateAttr(default="cpu")
    _gen_config = PrivateAttr(default_factory=dict)

    def __init__(self, model, tokenizer, device: str = "cpu", gen_config: dict = None):
        super().__init__()
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._gen_config = gen_config or {}

        if getattr(self._tokenizer, "pad_token", None) is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    @property
    def _llm_type(self) -> str:
        return "chatglm-local"

    def _call(self, prompt: str, stop=None, run_manager=None, **kwargs) -> str:
        """
        直接复用你已经验证可用的生成路径：generate_answer(model, tokenizer, ...)，
        只是把 prompt 当成 query 放入，避免再次走远程的 chat/generate 特殊逻辑。
        """
        try:
            # 将 LangChain 传入参数合并为生成配置
            local_cfg = {
                "temperature": kwargs.get("temperature", self._gen_config.get("temperature", 0.8)),
                "top_p": kwargs.get("top_p", self._gen_config.get("top_p", 0.9)),
                "max_length": kwargs.get("max_length", self._gen_config.get("max_length", 2048)),
                # 默认开启采样，避免告警
                "do_sample": kwargs.get("do_sample", self._gen_config.get("do_sample", True)),
            }

            text = generate_answer(self._model, self._tokenizer, query=prompt, context="", gen_kwargs=local_cfg)
            return text or "抱歉，我无法生成回答。"
        except Exception as e:
            logging.error(f"LangChain包装层生成失败: {e}", exc_info=True)
            return "抱歉，由于技术问题，我现在无法提供回答。"

# 新增：提供与主流程兼容的 LLMModule（供 main_handler 引用）
class LLMModule:
    def __init__(self):
        # 仍然沿用你已经验证可用的加载方式
        self.model, self.tokenizer = load_model_and_tokenizer()
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("ChatGLM 模型/分词器加载失败")
        self.device = DEVICE
        self.gen_config = GENERATION_CONFIG.copy()

        # 暴露一个 LangChain LLM 实例，方便需要链式使用的场景
        self.llm = ChatGLMForLangChain(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            gen_config=self.gen_config
        )

    def generate_answer(self, query: str, kg_results: str) -> str:
        """
        供主流程调用：复用你函数版的生成，确保与独立测试一致。
        """
        return generate_answer(self.model, self.tokenizer, query=query, context=kg_results)

    def get_model_info(self):
        return {
            "framework": "LangChain-adapter",
            "model_path": CHATGLM_PATH,
            "device": self.device,
            "llm_type": "chatglm-local",
        }