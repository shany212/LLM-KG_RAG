# main_handler.py
import logging
from modules.ner_intent_module import NERIntentModule
from modules.kg_module import KnowledgeGraphModule
from modules.llm_module import LLMModule

logger = logging.getLogger(__name__)

class MainHandler:
    def __init__(self):
        logger.info("正在初始化所有模块...")
        self.ner_intent_module = NERIntentModule()
        self.kg_module = KnowledgeGraphModule()
        self.llm_module = LLMModule()
        logger.info("所有模块初始化完成。")

    def process_query(self, query):
        """
        处理用户查询的完整流程。
        """
        # 1. NER 和 意图识别
        logger.info("步骤 1: 进行NER和意图识别...")
        analysis = self.ner_intent_module.analyze_query(query)
        intent = analysis['intent']
        entities = analysis['entities']
        logger.info(f"识别结果 - 意图: {intent}, 实体: {entities}")

        # 改进的处理逻辑
        if intent == "unknown_intent":
            logger.warning("意图不明确，将直接使用LLM进行通用回答。")
            kg_context = "无法确定用户的具体意图，建议提供通用的医疗建议。"
        elif intent == "find_disease_by_symptom" and not entities:
            # 特殊处理：根据症状查疾病但没有识别到实体
            logger.info("识别到症状查询意图，但未提取到具体症状实体，尝试从文本中提取关键词。")
            symptom_keywords = self._extract_symptom_keywords(query)
            if symptom_keywords:
                entities = [{'name': kw, 'type': 'Symptom'} for kw in symptom_keywords]
                logger.info(f"通过关键词提取到症状: {entities}")
                kg_context = self.kg_module.query_graph(intent, entities)
            else:
                kg_context = "未能识别到具体症状，建议详细描述症状或咨询专业医生。"
        elif intent in ["query_food_avoid", "query_food_recommend", "query_department"]:
            # 特殊处理：需要疾病实体的查询
            logger.info(f"识别到需要疾病实体的查询意图: {intent}")
            
            # 尝试从文本中提取疾病实体
            if not any(e['type'] == 'Disease' for e in entities):
                disease_keywords = self._extract_disease_keywords(query)
                if disease_keywords:
                    disease_entities = [{'name': kw, 'type': 'Disease'} for kw in disease_keywords]
                    entities.extend(disease_entities)
                    logger.info(f"通过关键词提取到疾病: {disease_entities}")
            
            if any(e['type'] == 'Disease' for e in entities):
                # 有疾病实体，查询知识图谱
                kg_context = self.kg_module.query_graph(intent, entities)
            else:
                # 没有疾病实体，提供一般性建议
                if intent == "query_food_avoid":
                    kg_context = "建议咨询医生了解具体的饮食禁忌，一般建议避免辛辣、油腻、生冷食物。"
                elif intent == "query_food_recommend":
                    kg_context = "建议咨询医生了解具体的饮食建议，一般建议多吃新鲜蔬菜水果，保持营养均衡。"
                elif intent == "query_department":
                    kg_context = "建议先到内科进行初步检查，医生会根据具体症状推荐合适的专科。"
        elif intent in ["query_drug", "query_symptom", "query_check", "query_cure_way"] and not entities:
            # 其他需要疾病实体的查询
            logger.info(f"识别到查询意图: {intent}，但未提取到疾病实体，尝试提取疾病关键词。")
            disease_keywords = self._extract_disease_keywords(query)
            if disease_keywords:
                entities = [{'name': kw, 'type': 'Disease'} for kw in disease_keywords]
                logger.info(f"通过关键词提取到疾病: {entities}")
                kg_context = self.kg_module.query_graph(intent, entities)
            else:
                kg_context = f"虽然识别到意图为'{intent}'，但未能提取到具体的疾病实体，建议明确指出疾病名称。"
        elif not entities:
            logger.warning("未识别到实体，将提供一般性建议。")
            kg_context = f"虽然识别到意图为'{intent}'，但未能提取到具体的医疗实体，建议提供更具体的信息。"
        else:
            # 2. 知识图谱查询
            logger.info("步骤 2: 查询知识图谱...")
            kg_context = self.kg_module.query_graph(intent, entities)
            logger.info(f"知识图谱返回内容: {kg_context}")
        
        # 3. LLM生成最终答案
        logger.info("步骤 3: LLM生成最终答案...")
        final_answer = self.llm_module.generate_answer(query, kg_context)
        logger.info(f"LLM生成的最终答案: {final_answer}")

        return {
            "query": query,
            "intent": intent,
            "entities": entities,
            "kg_context": kg_context,
            "final_answer": final_answer
        }
    
    def _extract_symptom_keywords(self, text):
        """从文本中提取症状关键词"""
        import re
        
        # 常见症状关键词
        symptom_keywords = [
            '头疼', '头痛', '头晕', '发烧', '发热', '咳嗽', '乏力', '恶心', '呕吐',
            '腹痛', '胸痛', '心慌', '气短', '失眠', '疼', '痛', '晕', '热', '咳'
        ]
        
        found_symptoms = []
        for keyword in symptom_keywords:
            if keyword in text:
                found_symptoms.append(keyword)
        
        return found_symptoms
    
    def _extract_disease_keywords(self, text):
        """从文本中提取疾病关键词"""
        disease_keywords = [
            '痛风', '高血压', '糖尿病', '心脏病', '胃病', '肝病', '肾病', '肺炎', 
            '支气管炎', '感冒', '发烧', '癌症', '肿瘤', '结石', '炎症', '感染', 
            '过敏', '贫血', '失眠症', '抑郁症', '焦虑症', '关节炎', '冠心病'
        ]
        
        found_diseases = []
        for keyword in disease_keywords:
            if keyword in text:
                found_diseases.append(keyword)
        
        return found_diseases