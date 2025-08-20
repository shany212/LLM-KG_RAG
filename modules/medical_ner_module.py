import logging
import torch
from transformers import AutoModelForTokenClassification, BertTokenizerFast
from config import Config

logger = logging.getLogger(__name__)

class MedicalNerModel:
    """来自iioSnail/bert-base-chinese-medical-ner项目的工具类"""
    
    @staticmethod
    def format_outputs(sentences, outputs):
        """
        将模型输出转换为实体列表
        outputs: tensor格式的输出，其中 1=B, 2=I, 3=E, 4=O
        """
        results = []
        
        for sentence_idx, sentence in enumerate(sentences):
            entities = []
            output = outputs[sentence_idx].tolist()
            
            i = 0
            while i < len(output) and i < len(sentence):
                if output[i] == 1:  # B标签，开始新实体
                    start = i
                    end = i + 1
                    
                    # 查找对应的I和E标签
                    j = i + 1
                    while j < len(output) and j < len(sentence):
                        if output[j] == 2:  # I标签，继续实体
                            end = j + 1
                            j += 1
                        elif output[j] == 3:  # E标签，结束实体
                            end = j + 1
                            break
                        else:  # 其他标签，结束当前实体
                            break
                    
                    # 提取实体文本
                    word = sentence[start:end]
                    if word.strip():  # 确保实体不为空
                        entities.append({
                            'start': start,
                            'end': end,
                            'word': word
                        })
                    
                    i = end
                else:
                    i += 1
            
            results.append(entities)
        
        return results

class MedicalNERModule:
    def __init__(self):
        self.model_name = Config.NER_MODEL_NAME
        self.device = Config.DEVICE
        
        try:
            logger.info(f"加载医疗专用NER模型: {self.model_name}")
            
            # 使用BertTokenizerFast（按照官方示例）
            self.tokenizer = BertTokenizerFast.from_pretrained(self.model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)
            
            # 移动模型到指定设备
            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.to(self.device)
            
            self.model.eval()
            
            # 获取标签映射（BIES格式）
            self.id2label = {0: 'PAD', 1: 'B', 2: 'I', 3: 'E', 4: 'O'}
            self.label2id = {'PAD': 0, 'B': 1, 'I': 2, 'E': 3, 'O': 4}
            
            logger.info("✅ 医疗专用NER模型加载成功")
            logger.info(f"标注格式: BIES (1=B, 2=I, 3=E, 4=O)")
            
            # 测试模型
            test_result = self._test_model()
            logger.info(f"模型测试结果: {test_result}")
            
        except Exception as e:
            logger.error(f"医疗NER模型加载失败: {e}")
            raise

    def _test_model(self):
        """测试模型功能"""
        try:
            test_sentences = ["风寒了吃什么药", "感冒的病因"]
            entities = self.extract_entities_batch(test_sentences)
            return entities
        except Exception as e:
            logger.warning(f"模型测试失败: {e}")
            return []

    def extract_entities(self, text: str):
        """提取单个文本的医疗实体"""
        logger.info(f"正在对文本进行医疗NER: '{text}'")
        
        if not text.strip():
            return []
        
        try:
            # 调用批处理方法
            batch_results = self.extract_entities_batch([text])
            entities = batch_results[0] if batch_results else []
            
            # 转换为Neo4j实体格式并分类
            neo4j_entities = []
            for entity in entities:
                word = entity['word'].strip()
                if word and len(word) >= 2:  # 过滤太短的实体
                    # 使用简单规则判断医疗实体类型
                    entity_type = self._classify_medical_entity(word)
                    
                    if entity_type:
                        neo4j_entities.append({
                            'name': word,
                            'type': entity_type,
                            'start': entity['start'],
                            'end': entity['end']
                        })
            
            # 去重
            unique_entities = self._deduplicate_entities(neo4j_entities)
            
            logger.info(f"最终提取到的实体: {unique_entities}")
            return unique_entities
            
        except Exception as e:
            logger.error(f"医疗实体提取失败: {e}")
            return []

    def extract_entities_batch(self, sentences):
        """批量提取医疗实体"""
        try:
            if not sentences:
                return []
            
            # 使用tokenizer处理输入（按照官方示例，不添加特殊token）
            inputs = self.tokenizer(
                sentences, 
                return_tensors="pt", 
                padding=True, 
                add_special_tokens=False,
                truncation=True,
                max_length=512
            )
            
            # 移动到设备
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 模型推理
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 获取预测结果
                predictions = outputs.logits.argmax(-1) * inputs['attention_mask']
            
            # 移回CPU进行后处理
            predictions = predictions.cpu()
            
            # 使用格式化函数转换输出
            entities_list = MedicalNerModel.format_outputs(sentences, predictions)
            
            return entities_list
            
        except Exception as e:
            logger.error(f"批量实体提取失败: {e}")
            return [[] for _ in sentences]

    def _classify_medical_entity(self, word):
        """简单规则分类医疗实体到Neo4j类型"""
        # Neo4j实体类型: Check, Department, Disease, Drug, Food, Symptom
        
        # 疾病相关关键词（最常见的）
        if any(keyword in word for keyword in [
            '病', '症', '炎', '癌', '瘤', '血压', '糖尿', '痛风', '感冒', '发烧',
            '肺炎', '胃炎', '肝炎', '肾炎', '心脏病', '脑梗', '中风', '骨折',
            '钙化', '结石', '囊肿', '增生', '硬化', '萎缩', '狭窄', '梗阻'
        ]) or word.endswith(('病', '症', '炎', '癌')):
            return 'Disease'
        
        # 症状相关关键词
        elif any(keyword in word for keyword in [
            '疼', '痛', '酸', '胀', '麻', '痒', '热', '冷', '晕', '乏力',
            '恶心', '呕吐', '腹泻', '便秘', '咳嗽', '气短', '心慌', '失眠',
            '头晕', '头痛', '胸闷', '腹胀', '食欲不振', '体重下降'
        ]) or any(word.endswith(suffix) for suffix in ['疼', '痛', '胀', '酸', '麻']):
            return 'Symptom'
        
        # 药物相关关键词  
        elif any(keyword in word for keyword in [
            '针', '片', '胶囊', '颗粒', '丸', '散', '膏', '液', '素', '林', '霉素',
            '阿司匹林', '布洛芬', '青霉素', '胰岛素', '玻尿酸', '肉毒素',
            '药', '剂', '制剂', '注射', '滴眼', '滴鼻', '口服', '外用'
        ]) or word.endswith(('片', '胶囊', '颗粒', '丸', '散', '膏', '液', '针')):
            return 'Drug'
        
        # 检查相关关键词
        elif any(keyword in word for keyword in [
            'CT', 'MRI', 'X光', 'B超', '彩超', '心电图', '脑电图', '肌电图',
            '血常规', '尿常规', '肝功', '肾功', '血糖', '血脂', '血压',
            '检查', '检验', '化验', '筛查', '监测', '测定', '分析'
        ]) or word.endswith(('检查', '检验', '化验')):
            return 'Check'
        
        # 科室相关关键词
        elif any(keyword in word for keyword in [
            '科', '内科', '外科', '儿科', '妇科', '骨科', '眼科', '耳鼻喉',
            '皮肤科', '神经科', '心内科', '消化科', '呼吸科', '肿瘤科',
            '急诊', '门诊', '病房', '诊室'
        ]) or word.endswith('科'):
            return 'Department'
        
        # 食物相关关键词
        elif any(keyword in word for keyword in [
            '食物', '食品', '饮食', '营养', '蛋白', '维生素', '钙', '铁', '锌',
            '水果', '蔬菜', '肉类', '海鲜', '豆类', '坚果', '奶制品', '主食',
            '米', '面', '肉', '鱼', '虾', '蟹', '奶', '蛋'
        ]):
            return 'Food'
        
        # 默认分类：根据长度和常见模式
        elif len(word) >= 3:
            # 较长的医疗实体通常是疾病名
            return 'Disease'
        elif len(word) == 2:
            # 较短的实体可能是症状
            return 'Symptom'
        
        # 无法分类的不保留
        return None

    def _deduplicate_entities(self, entities):
        """去重实体"""
        seen = {}
        unique_entities = []
        
        for entity in entities:
            name = entity['name']
            entity_type = entity['type']
            # 以名称作为去重key（忽略大小写）
            key = name.lower()
            
            if key not in seen:
                seen[key] = True
                unique_entities.append({
                    'name': name,
                    'type': entity_type
                })
        
        return unique_entities

    def get_model_info(self):
        """获取模型信息"""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'labels': list(self.id2label.values()),
            'num_labels': len(self.id2label),
            'model_type': 'Medical_BERT_NER',
            'supported_neo4j_entities': ['Check', 'Department', 'Disease', 'Drug', 'Food', 'Symptom'],
            'annotation_scheme': 'BIES',
            'description': '医疗领域中文命名实体识别专用模型 + 简单规则分类'
        }