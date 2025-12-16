import os
import shutil
from modelscope.hub.snapshot_download import snapshot_download

# 1. 设置保存目录 (对应 config.py 中的 ../models/)
BASE_DIR = "models"
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)

print(f"准备下载模型至: {os.path.abspath(BASE_DIR)}\n")

# 2. 定义 config.py 对应的模型映射
# 键名是 config.py 中 path 的文件夹名
# 值是 ModelScope 上可用的对应模型 ID
model_map = {
    # 对应 config 中的: '../models/chinese_roberta_wwm_ext_pytorch'
    "chinese_roberta_wwm_ext_pytorch": "dienstag/chinese-roberta-wwm-ext",

    # 对应 config 中的: '../models/chinese_wwm_ext_pytorch'
    # 注意：这是 BERT-wwm 版本，不是 RoBERTa
    "chinese_wwm_ext_pytorch": "dienstag/chinese-bert-wwm-ext",

    # 对应 config 中的: '../models/ERNIE'
    # 注意：这是 pytorch 版本的 ERNIE 1.0
    "ERNIE": "nghuyong/ernie-1.0"
}

# 3. 开始下载循环
for folder_name, model_id in model_map.items():
    target_path = os.path.join(BASE_DIR, folder_name)
    
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"正在处理: {folder_name}")
    print(f"对应模型 ID: {model_id}")

    # 如果文件夹已存在且不为空，询问是否跳过
    if os.path.exists(target_path) and os.listdir(target_path):
        print(f"Warning: 目标文件夹 {target_path} 已存在且不为空。")
        print("跳过下载。如果需要重新下载，请手动删除该文件夹。")
        continue

    try:
        # 尝试从 ModelScope 下载
        print(f"正在从 ModelScope 下载...")
        # cache_dir=None 会下载到默认缓存，我们稍后移动它
        cache_path = snapshot_download(model_id)
        
        # 移动文件到指定目录
        print(f"正在安装到: {target_path} ...")
        
        # 如果目标存在（空文件夹），先删除以便 copytree 运行
        if os.path.exists(target_path):
            shutil.rmtree(target_path)
            
        shutil.copytree(cache_path, target_path)
        print(f"✅ 成功! 模型已就绪: {folder_name}")

    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print(f"提示: 请检查网络，或者该模型 ID 在 ModelScope 暂时不可用。")

print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("下载任务结束。请检查 models/ 文件夹下的内容。")