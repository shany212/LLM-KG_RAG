# modules/medical_intent_module.py
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from config import Config

logger = logging.getLogger(__name__)

class MedicalIntentModule:
    def __init__(self):
        self.model_name = Config.INTENT_MODEL_NAME
        self.device = Config.DEVICE
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"医疗意图识别模型 ({self.model_name}) 加载成功。")
        except Exception as e:
            logger.error(f"加载意图识别模型失败: {e}")
            raise
        
        # 扩展意图模板，包含科室咨询
        self.intent_templates = {
            "query_symptom": [
            "这个病有什么症状", "得了这个病会有什么表现", "症状是什么",
            "有哪些症状", "会出现哪些症状", "临床表现有哪些", "常见症状有哪些",
            "典型症状是什么", "早期症状有哪些", "初期表现是什么", "晚期症状有哪些",
            "有什么体征", "症状表现有哪些", "会不会发烧", "会不会咳嗽",
            "有没有皮疹", "有没有头疼", "会不会腹痛", "会不会恶心呕吐",
            "有什么不适", "主要表现", "症状都有什么", "症状严重吗", "常见表现"
            ],
            "query_drug": [
            "这个病吃什么药", "推荐用药有哪些", "用什么药治疗", "药物治疗",
            "吃点什么药", "用药建议", "推荐药品", "有什么药可以吃",
            "有没有特效药", "需要吃消炎药吗", "要不要用抗生素", "有无非处方药",
            "中成药可以吗", "中药能治吗", "要不要打针", "是否需要输液",
            "外用药有哪些", "口服药有哪些", "儿童用药怎么选", "孕妇能吃什么药",
            "哺乳期能用什么药", "老年人用药建议", "用药需要注意什么", "吃多久"
            ],
            "query_check": [
            "需要做什么检查", "怎么查这个病", "检查什么", "要做哪些检验",
            "需要做哪些检查", "如何确诊", "确诊要做什么检查", "要不要做血常规",
            "要不要做CT", "要不要做核磁共振", "需不需要B超", "彩超要做吗",
            "需不需要验血", "尿检要做吗", "便检需要吗", "需要做哪些化验",
            "影像学检查有哪些", "检查项目有哪些", "检查流程是什么", "要不要做X光",
            "心电图要不要做", "需要做皮试吗", "需要做核酸吗", "检查前要注意什么"
            ],
            "query_prevent": [
            "怎么预防这个病", "如何避免得这个病", "预防方法",
            "如何预防", "怎么避免", "日常如何预防", "生活中怎么防范",
            "平时要注意什么", "有没有疫苗可以预防", "需要打疫苗吗",
            "日常防护怎么做", "怎样降低复发", "如何减少发作", "怎么增强抵抗力",
            "饮食上如何预防", "作息如何调整", "运动能预防吗", "要不要隔离",
            "居家如何预防", "工作中如何预防", "孩子如何预防"
            ],
            "query_cause": [
            "这个病是什么原因引起的", "为什么会得这个病", "病因",
            "发病原因是什么", "成因是什么", "诱因有哪些", "是什么引起的",
            "是细菌还是病毒", "属于遗传性吗", "会不会遗传",
            "免疫异常会导致吗", "压力大能导致吗", "饮食会不会引起",
            "受凉上火会导致吗", "环境因素有哪些", "传染导致的吗",
            "内分泌失调会不会引起", "有哪些高危因素", "危险因素是什么"
            ],
            "query_cure_way": [
            "这个病怎么治", "有哪些治疗方法", "治疗方案",
            "怎么治疗", "治疗方式有哪些", "有没有推荐的治疗",
            "需要手术吗", "能不能保守治疗", "是否需要住院",
            "吃药能好吗", "理疗可以吗", "中医治疗行不行",
            "康复治疗怎么做", "要不要打针", "是否需要输液",
            "是否可以自愈", "需要多久治疗", "联合治疗可以吗",
            "有没有最新疗法", "物理治疗有哪些", "随访复诊要怎么安排"
            ],
            "query_desc": [
            "介绍一下这个病", "这是什么病", "病情描述",
            "科普一下这个病", "详细讲讲这个病", "这是个什么情况",
            "定义是什么", "概述一下", "基本情况", "常见吗",
            "属于什么类型的疾病", "严重吗", "是急性还是慢性",
            "传染吗", "需要注意什么", "简单介绍一下", "总体情况如何"
            ],
            "query_department": [
            "应该挂什么科", "看什么科", "去哪个科室", "挂号科室",
            "这个病看什么科", "应该去什么科", "挂什么科室",
            "该挂哪个科", "挂哪个号", "去什么科挂号",
            "属于哪个科室", "找哪个科的医生", "内科还是外科",
            "去急诊还是门诊", "需要挂儿科吗", "妇科还是产科",
            "皮肤科可以看吗", "耳鼻喉科合适吗", "哪个专科更合适",
            "应该看哪个门诊", "在哪个科就诊", "哪个科负责"
            ],
            "find_disease_by_symptom": [
            "我头疼怎么办", "最近头疼和发烧是怎么回事", "我有头痛症状可能是什么病",
            "我最近有点头疼", "头疼可能是什么病", "我头痛", "感觉头疼",
            "我不舒服", "身体不适", "有症状", "感觉不好",
            "我咳嗽怎么办", "咳嗽有痰怎么办", "干咳是怎么回事",
            "发烧了怎么办", "低烧不退怎么办", "发冷发热怎么回事",
            "肚子疼怎么办", "腹泻怎么回事", "拉肚子怎么办", "便秘怎么办",
            "恶心想吐是怎么回事", "胸口疼怎么办", "胸闷气短怎么办",
            "嗓子疼怎么办", "咽喉肿痛怎么办", "鼻塞流鼻涕怎么办",
            "起皮疹怎么回事", "身上出红点怎么办", "头晕目眩怎么办",
            "腰疼怎么办", "关节痛怎么办", "牙疼怎么办",
            "眼睛疼怎么办", "耳朵疼怎么办", "手脚麻木怎么办",
            "夜里盗汗怎么回事", "心慌心悸怎么办", "睡不着怎么办",
            "小便疼怎么办", "尿频尿急怎么办", "小便有血怎么办",
            "大便带血怎么办", "全身乏力怎么办", "浑身酸痛怎么办"
            ],
            "query_food_avoid": [
            "这个病不能吃什么", "有什么食物禁忌", "什么食物不能吃",
            "忌口什么", "饮食禁忌有哪些", "不能吃的食物",
            "什么东西不能吃", "有什么忌口的", "饮食上要注意什么",
            "哪些食物要避免", "不适合吃什么", "忌食什么",
            "能不能喝酒", "咖啡能喝吗", "能吃辣吗", "海鲜能吃吗",
            "牛奶能不能喝", "哪些水果不要吃", "高脂肪食物能吃吗",
            "辛辣刺激要不要忌", "烟酒要不要戒", "油腻要不要少吃",
            "生冷能不能吃", "甜食需要少吃吗", "有哪些发物需要避开",
            "碳酸饮料能不能喝", "哪些调料需要避免", "夜宵能不能吃"
            ],
            "query_food_recommend": [
            "这个病适合吃什么", "推荐吃什么食物", "吃什么好", "什么食物好",
            "适合的食物有哪些", "饮食建议", "吃什么有助于康复",
            "什么食物对病情有好处", "营养建议", "食疗方法",
            "吃什么能改善", "有益的食物", "推荐的饮食", "推荐的食物",
            "有什么推荐的食物", "食物推荐", "推荐食物",
            "吃清淡点吗", "多吃什么比较好", "适合吃哪些水果",
            "喝什么粥比较好", "有哪些汤品推荐", "高蛋白食物可以吗",
            "多补充哪些维生素", "要不要多喝水", "哪些蔬菜更合适",
            "主食怎么选择", "奶制品可以吃吗", "有没有简单食谱",
            "康复期饮食怎么搭配", "早餐有什么建议"
            ]
        }
        
        # 对模板进行编码，预先计算好向量
        self.template_embeddings = {}
        for intent, templates in self.intent_templates.items():
            self.template_embeddings[intent] = self.model.encode(templates, convert_to_tensor=True)

    def recognize_intent(self, text: str):
        """
        使用语义相似度识别用户意图。
        """
        logger.info(f"正在对文本进行意图识别: '{text}'")
        if not text.strip():
            return "unknown_intent"
            
        try:
            query_embedding = self.model.encode(text, convert_to_tensor=True)
            
            best_match_intent = "unknown_intent"
            max_similarity = -1.0
            
            for intent, template_embs in self.template_embeddings.items():
                # 计算余弦相似度
                similarities = cosine_similarity(
                    query_embedding.cpu().numpy().reshape(1, -1),
                    template_embs.cpu().numpy()
                )
                avg_similarity = np.mean(similarities)
                
                logger.info(f"意图 '{intent}' 的相似度: {avg_similarity:.3f}")
                
                if avg_similarity > max_similarity:
                    max_similarity = avg_similarity
                    best_match_intent = intent
            
            # 降低阈值，提高召回率
            logger.info(f"最高相似度: {max_similarity:.3f}, 匹配意图: {best_match_intent}")
            
            if max_similarity < 0.3:  # 从0.5降低到0.3
                return "unknown_intent"

            return best_match_intent
        except Exception as e:
            logger.error(f"意图识别失败: {e}")
            return "unknown_error"