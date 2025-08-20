# app.py
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from main_handler import MainHandler

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)


try:
    handler = MainHandler()
except Exception as e:
    logging.error(f"应用初始化失败: {e}", exc_info=True)
    handler = None

@app.route('/api/chat', methods=['POST'])
def chat():
    if not handler:
        return jsonify({"error": "服务未成功初始化，请检查日志。"}), 500

    data = request.json
    query = data.get('query')

    if not query:
        return jsonify({"error": "请求中缺少 'query' 参数。"}), 400

    try:
        result = handler.process_query(query)
        return jsonify(result)
    except Exception as e:
        logging.error(f"处理查询 '{query}' 时发生错误: {e}", exc_info=True)
        return jsonify({"error": "处理您的请求时发生内部错误。"}), 500

if __name__ == '__main__':
    # 请不要在生产环境中使用Flask自带的服务器
    app.run(host='0.0.0.0', port=5000, debug=False)