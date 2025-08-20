# download_model.py
import os
import logging
from config import Config

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_model_from_modelscope(model_id, local_dir):
    """从ModelScope下载模型"""
    try:
        from modelscope.hub.snapshot_download import snapshot_download
        logger.info(f"开始从 ModelScope 下载: {model_id}")
        logger.info(f"目标路径: {local_dir}")
        # 创建目标目录
        os.makedirs(local_dir, exist_ok=True)
        
        snapshot_download(
            model_id=model_id,
            cache_dir=local_dir, # 直接下载到指定目录
            local_dir=local_dir # 确保文件在最终目录
        )
        logger.info(f"✅ 模型成功下载并存放于: {local_dir}")
        return True
    except ImportError:
        logger.error("检测到 'modelscope' 库未安装。请先执行: pip install modelscope")
        return False
    except Exception as e:
        logger.error(f"从 ModelScope 下载模型 {model_id} 失败: {e}")
        return False

def download_model_from_huggingface(model_id, local_dir):
    """从Hugging Face Hub下载整个模型仓库"""
    try:
        from huggingface_hub import snapshot_download
        logger.info(f"开始从 Hugging Face 下载: {model_id}")
        logger.info(f"目标路径: {local_dir}")
        # 创建目标目录
        os.makedirs(local_dir, exist_ok=True)

        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # 建议禁用符号链接，直接复制文件
            # 如果您的网络不稳定，可以取消下面的注释
            # resume_download=True 
        )
        logger.info(f"✅ 模型成功下载并存放于: {local_dir}")
        return True
    except ImportError:
        logger.error("检测到 'huggingface_hub' 库未安装。请先执行: pip install huggingface-hub")
        return False
    except Exception as e:
        logger.error(f"从 Hugging Face 下载模型 {model_id} 失败: {e}")
        return False

if __name__ == "__main__":
    print("=" * 80)
    print("🚀 开始执行医疗大模型项目依赖模型下载脚本 🚀")
    print("=" * 80)

    # 确保所有目标目录存在
    os.makedirs(os.path.dirname(Config.CHATGLM_PATH), exist_ok=True)
    os.makedirs(Config.NER_MODEL_NAME, exist_ok=True)
    os.makedirs(Config.INTENT_MODEL_NAME, exist_ok=True)

    all_success = True

    # --- 1. 下载ChatGLM模型 ---
    print("\n--- 任务1: 下载 ChatGLM LLM ---")
    chatglm_model_id_ms = "ZhipuAI/chatglm2-6b-int4" # ModelScope上的ID
    chatglm_local_path = Config.CHATGLM_PATH
    if os.path.exists(os.path.join(chatglm_local_path, 'config.json')):
        logger.info(f"✅ ChatGLM模型已存在于 {chatglm_local_path}，跳过下载。")
    else:
        if not download_model_from_modelscope(chatglm_model_id_ms, chatglm_local_path):
            all_success = False

    # --- 2. 下载医疗NER模型 ---
    print("\n--- 任务2: 下载医疗命名实体识别 (NER) 模型 ---")
    ner_model_id_hf = "iioSnail/bert-base-chinese-medical-ner" # HuggingFace上的ID
    ner_local_path = Config.NER_MODEL_NAME
    if os.path.exists(os.path.join(ner_local_path, 'config.json')):
        logger.info(f"✅ NER模型已存在于 {ner_local_path}，跳过下载。")
    else:
        if not download_model_from_huggingface(ner_model_id_hf, ner_local_path):
            all_success = False
    
    # --- 3. 下载意图识别模型 ---
    print("\n--- 任务3: 下载医疗意图识别模型 ---")
    intent_model_id_hf = "shibing624/text2vec-base-chinese" # HuggingFace上的ID
    intent_local_path = Config.INTENT_MODEL_NAME
    if os.path.exists(os.path.join(intent_local_path, 'config.json')):
        logger.info(f"✅ 意图识别模型已存在于 {intent_local_path}，跳过下载。")
    else:
        if not download_model_from_huggingface(intent_model_id_hf, intent_local_path):
            all_success = False

    print("=" * 80)
    if all_success:
        print("🎉 恭喜！所有模型均已准备就绪。现在您可以运行主程序 app.py 了。")
    else:
        print("❌ 部分模型下载失败，请检查上面的错误日志。")
    print("=" * 80)