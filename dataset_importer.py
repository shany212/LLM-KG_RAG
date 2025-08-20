import json
import os
from neo4j import GraphDatabase

class MedicalGraphImporter:
    """
    一个用于将医疗JSON数据导入Neo4j知识图谱的封装类。
    """

    def __init__(self, uri, user, password):
        """
        初始化导入器并连接到Neo4j数据库。

        """
        self._uri = uri
        self._user = user
        self._password = password
        self._driver = None
        try:
            self._driver = GraphDatabase.driver(self._uri, auth=(self._user, self._password))
            print("成功连接到Neo4j数据库。")
        except Exception as e:
            print(f"数据库连接失败: {e}")

    def close(self):
        """关闭数据库连接。"""
        if self._driver is not None:
            self._driver.close()
            print("数据库连接已关闭。")

    def _execute_query(self, query, parameters=None):
        """
        一个私有的辅助函数，用于执行Cypher查询。
        :param query: 要执行的Cypher查询语句。
        :param parameters: 查询的参数。
        """
        if self._driver is None:
            print("数据库未连接，无法执行查询。")
            return
            
        with self._driver.session() as session:
            session.run(query, parameters)

    def create_constraints(self):
        """
        为所有实体类型创建唯一性约束，确保节点不重复并提高查询效率。
        """
        print("正在创建唯一性约束...")
        constraints = [
            "CREATE CONSTRAINT ON (d:Disease) ASSERT d.name IS UNIQUE;",
            "CREATE CONSTRAINT ON (s:Symptom) ASSERT s.name IS UNIQUE;",
            "CREATE CONSTRAINT ON (d:Drug) ASSERT d.name IS UNIQUE;",
            "CREATE CONSTRAINT ON (c:Check) ASSERT c.name IS UNIQUE;",
            "CREATE CONSTRAINT ON (f:Food) ASSERT f.name IS UNIQUE;",
            "CREATE CONSTRAINT ON (d:Department) ASSERT d.name IS UNIQUE;",
        ]
        for constraint in constraints:
            try:
                self._execute_query(constraint)
            except Exception as e:
                # 约束可能已存在，可以忽略此错误
                if "An equivalent constraint already exists" in str(e):
                    pass
                else:
                    print(f"创建约束时出错: {e}")
        print("约束创建完成。")

    def import_data(self, json_file_path):
        """
        主函数，用于读取JSONL文件并导入数据。
        :param json_file_path: JSONL文件的路径（每行一个JSON对象）。
        """
        # 检查文件是否存在
        if not os.path.exists(json_file_path):
            print(f"错误: 找不到文件 '{json_file_path}'")
            print(f"当前工作目录: {os.getcwd()}")
            print("请确认文件路径是否正确，或将文件放在正确的位置。")
            return False
            
        print(f"开始从 {json_file_path} 导入JSONL数据...")
        
        try:
            data = []
            with open(json_file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:  # 跳过空行
                        continue
                    try:
                        json_obj = json.loads(line)
                        data.append(json_obj)
                    except json.JSONDecodeError as e:
                        print(f"第 {line_num} 行JSON格式错误: {e}")
                        continue
        except Exception as e:
            print(f"读取文件时发生错误: {e}")
            return False

        total_diseases = len(data)
        print(f"成功读取 {total_diseases} 条疾病数据")
        
        for i, disease_data in enumerate(data):
            disease_name = disease_data.get('name')
            if not disease_name:
                print(f"第 {i+1} 条数据缺少疾病名称，跳过")
                continue

            print(f"正在处理: {disease_name} ({i+1}/{total_diseases})")

            try:
                # 1. 创建疾病节点并设置其属性
                disease_props = {k: v for k, v in disease_data.items() if isinstance(v, (str, int, float))}
                query = "MERGE (d:Disease {name: $name}) SET d += $props"
                self._execute_query(query, parameters={'name': disease_name, 'props': disease_props})

                # 2. 创建并关联相关节点
                self._create_relationships(disease_name, 'symptom', 'Symptom', 'HAS_SYMPTOM', disease_data)
                self._create_relationships(disease_name, 'acompany', 'Disease', 'HAS_COMPLICATION', disease_data)
                self._create_relationships(disease_name, 'common_drug', 'Drug', 'RECOMMENDS_DRUG', disease_data)
                self._create_relationships(disease_name, 'recommand_drug', 'Drug', 'RECOMMENDS_DRUG', disease_data)
                self._create_relationships(disease_name, 'check', 'Check', 'NEEDS_CHECK', disease_data)
                self._create_relationships(disease_name, 'cure_department', 'Department', 'BELONGS_TO_DEPT', disease_data)
                self._create_relationships(disease_name, 'do_eat', 'Food', 'RECOMMENDS_EAT', disease_data)
                self._create_relationships(disease_name, 'recommand_eat', 'Food', 'RECOMMENDS_EAT', disease_data)
                self._create_relationships(disease_name, 'not_eat', 'Food', 'AVOIDS_EAT', disease_data)
            except Exception as e:
                print(f"处理疾病 '{disease_name}' 时发生错误: {e}")
                continue
                
        print("数据导入完成！")
        return True

    def _create_relationships(self, disease_name, json_key, node_label, rel_type, data):
        """
        一个通用的私有辅助函数，用于创建节点和关系。
        :param disease_name: 疾病名称 (源节点)
        :param json_key: JSON中的键 (例如 'symptom')
        :param node_label: 目标节点的标签 (例如 'Symptom')
        :param rel_type: 关系类型 (例如 'HAS_SYMPTOM')
        :param data: 单条疾病的JSON数据字典
        """
        items = data.get(json_key, [])
        if not items:
            return

        for item_name in items:
            query = f"""
            MATCH (d:Disease {{name: $disease_name}})
            MERGE (n:{node_label} {{name: $item_name}})
            MERGE (d)-[:{rel_type}]->(n)
            """
            self._execute_query(query, parameters={'disease_name': disease_name, 'item_name': item_name})


if __name__ == '__main__':
    NEO4J_URI = "bolt://localhost:7687" 
    NEO4J_USER = "neo4j"                 
    NEO4J_PASSWORD = "123456"   
    JSON_FILE_PATH = "./data/data.json"  # 使用绝对路径

    # --- 开始执行导入 ---
    importer = MedicalGraphImporter(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    if importer._driver:
        # 1. 创建约束 (首次运行时执行，后续可注释掉)
        importer.create_constraints()
        
        # 2. 导入数据
        success = importer.import_data(JSON_FILE_PATH)
        if not success:
            print("数据导入失败，请检查文件路径和格式。")
        
        # 3. 关闭连接
        importer.close()
    else:
        print("无法连接到数据库，程序退出。")