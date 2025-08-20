# modules/kg_module.py
import logging
from neo4j import GraphDatabase
from config import Config

logger = logging.getLogger(__name__)

class KnowledgeGraphModule:
    def __init__(self):
        try:
            self._driver = GraphDatabase.driver(
                Config.NEO4J_URI, 
                auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD)
            )
            logger.info("成功连接到Neo4j数据库。")
        except Exception as e:
            logger.error(f"连接Neo4j失败: {e}")
            self._driver = None

    def close(self):
        if self._driver is not None:
            self._driver.close()

    def query_graph(self, intent, entities):
        """
        根据意图和实体，查询知识图谱。
        :param intent: 意图 (e.g., 'query_symptom')
        :param entities: 实体列表 (e.g., [{'type': 'Disease', 'name': '高血压'}])
        :return: 查询结果的格式化字符串。
        """
        if not self._driver:
            return "数据库连接失败。"
        
        if not entities:
            return "未能识别出有效的实体，无法进行知识查询。"

        # 提取核心实体
        # 简化处理，默认使用第一个识别出的实体
        primary_entity_name = entities[0]['name']
        
        # 构建Cypher查询
        cypher_query, params = self._build_cypher_query(intent, primary_entity_name, entities)

        if not cypher_query:
            return f"无法处理意图 '{intent}'。"

        logger.info(f"执行Cypher查询: {cypher_query} with params {params}")
        
        try:
            with self._driver.session() as session:
                results = session.run(cypher_query, params)
                return self._format_results(intent, results)
        except Exception as e:
            logger.error(f"知识图谱查询失败: {e}")
            return "知识图谱查询时发生错误。"

    def _build_cypher_query(self, intent, primary_entity_name, all_entities):
        """根据意图构建Cypher查询语句和参数"""
        query = ""
        params = {'name': primary_entity_name}

        if intent == 'query_symptom':
            query = "MATCH (d:Disease {name: $name})-[:HAS_SYMPTOM]->(s:Symptom) RETURN s.name as result"
        elif intent == 'query_drug':
            query = "MATCH (d:Disease {name: $name})-[:RECOMMENDS_DRUG]->(dr:Drug) RETURN dr.name as result"
        elif intent == 'query_check':
            query = "MATCH (d:Disease {name: $name})-[:NEEDS_CHECK]->(c:Check) RETURN c.name as result"
        elif intent == 'query_prevent':
            query = "MATCH (d:Disease {name: $name}) RETURN d.prevent as result"
        elif intent == 'query_cause':
            query = "MATCH (d:Disease {name: $name}) RETURN d.cause as result"
        elif intent == 'query_cure_way':
            query = "MATCH (d:Disease {name: $name}) RETURN d.cure_way as result"
        elif intent == 'query_desc':
            query = "MATCH (d:Disease {name: $name}) RETURN d.desc as result"
        elif intent == 'query_food_avoid':
            # 不能吃的食物
            query = "MATCH (d:Disease {name: $name})-[:AVOIDS_EAT]->(f:Food) RETURN f.name as result"
        elif intent == 'query_food_recommend':
            # 推荐吃的食物
            query = "MATCH (d:Disease {name: $name})-[:RECOMMENDS_EAT]->(f:Food) RETURN f.name as result"
        elif intent == 'query_department':
            # 挂号科室
            query = "MATCH (d:Disease {name: $name})-[:BELONGS_TO_DEPT]->(dep:Department) RETURN dep.name as result"
        elif intent == 'query_complication':
            # 并发症（并发的疾病）
            query = "MATCH (d:Disease {name: $name})-[:HAS_COMPLICATION]->(c:Disease) RETURN c.name as result"
        elif intent == 'find_disease_by_symptom':
            # 根据症状查疾病
            symptom_names = [e['name'] for e in all_entities if e.get('type') == 'Symptom']
            if not symptom_names:
                return None, None

            match_clauses = []
            for i, name in enumerate(symptom_names):
                key = f"symptom_{i}"
                match_clauses.append(f"(d:Disease)-[:HAS_SYMPTOM]->(:Symptom {{name: ${key}}})")
                params[key] = name

            query = f"MATCH {', '.join(match_clauses)} RETURN DISTINCT d.name as result"

        return query, params

    def _format_results(self, intent, results):
        """格式化查询结果为自然语言字符串"""
        records = [record['result'] for record in results if record['result']]
        
        if not records:
            return "在知识图谱中未找到相关信息。"

        if intent in ['query_prevent', 'query_cause', 'query_desc']:
            return records[0]
        elif intent == 'query_cure_way':
            return "、".join(records[0]) if isinstance(records[0], list) else records[0]
        elif intent == 'query_food_avoid':
            return f"忌食：{' 、'.join(set(records))}"
        elif intent == 'query_food_recommend':
            return f"推荐食物：{' 、'.join(set(records))}"
        elif intent == 'query_department':
            return f"建议挂号科室：{' 、'.join(set(records))}"
        elif intent == 'query_symptom':
            return f"主要症状：{' 、'.join(set(records))}"
        elif intent == 'query_drug':
            return f"推荐药物：{' 、'.join(set(records))}"
        elif intent == 'query_check':
            return f"建议检查：{' 、'.join(set(records))}"
        elif intent == 'find_disease_by_symptom':
            return f"可能的疾病：{' 、'.join(set(records))}"
        
        return "、".join(set(records))