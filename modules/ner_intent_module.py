# modules/ner_intent_module.py
import logging
# 确保导入的是我们新编写的、有实际功能的模块
from .medical_ner_module import MedicalNERModule
from .medical_intent_module import MedicalIntentModule

logger = logging.getLogger(__name__)

class NERIntentModule:
    def __init__(self):
        try:
            # 初始化我们新编写的模块，它们会自动下载并加载模型
            self.ner_model = MedicalNERModule()
            self.intent_model = MedicalIntentModule()
            # 关键修复：日志信息已更改，表示加载了正确的模块
            logger.info("NER与意图识别模块初始化成功 (Hugging Face模型)。")
        except Exception as e:
            logger.error(f"NER与意图识别模块初始化失败: {e}")
            raise

    def analyze_query(self, query: str):
        try:
            entities = self.ner_model.extract_entities(query)
            intent = self.intent_model.recognize_intent(query)
            
            return {
                "intent": intent,
                "entities": entities
            }
        except Exception as e:
            logger.error(f"查询分析失败: {e}")
            return {
                "intent": "unknown_error",
                "entities": []
            }