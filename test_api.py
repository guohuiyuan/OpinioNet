from openai import OpenAI
import json
import sys

# ================= 配置区域 (与 llm_predict2.py 保持一致) =================
VLLM_API_BASE = "http://10.249.42.129:8000/v1"
VLLM_API_KEY = "apikey"
MODEL_NAME = "qwen3-32b"  # 注意这里是你脚本里写的模型名

# 简化的 System Prompt，用于快速验证业务逻辑
DEFAULT_SYSTEM_PROMPT = """你是一个专业的中文电商化妆品评论观点四元组抽取助手。
任务：给定一条化妆品电商评论文本，抽取其中所有的观点四元组（AspectTerm, OpinionTerm, Category, Polarity），即 ACOS 四元组。
严格遵守：
1. 输出唯一 JSON：{"quadruples":[{"aspect":"...","opinion":"...","category":"...","polarity":"..."}, ...]}
2. 若没有观点，输出 {"quadruples":[]}
3. 只输出 JSON，不输出其它任何文本。
"""

def main():
    # 1. 初始化客户端
    print(f"正在连接到: {VLLM_API_BASE}")
    try:
        client = OpenAI(api_key=VLLM_API_KEY, base_url=VLLM_API_BASE)
    except Exception as e:
        print(f"客户端初始化失败: {e}")
        return

    # 2. (可选) 检查模型列表，确认服务端的实际模型 ID
    print("\n[Step 1] 检查模型列表...")
    try:
        models = client.models.list()
        model_ids = [m.id for m in models.data]
        print(f"服务端可用模型: {model_ids}")
        
        if MODEL_NAME not in model_ids:
            print(f"⚠️ 警告: 配置的模型名 '{MODEL_NAME}' 不在列表中，可能会报错 (除非使用了别名)。")
    except Exception as e:
        print(f"获取模型列表失败 (不影响后续调用): {e}")

    # 3. 发送测试请求
    test_review = "这瓶精华液保湿效果不错，就是瓶子太难看了。"
    print(f"\n[Step 2] 测试推理...")
    print(f"输入文本: {test_review}")
    print(f"使用模型: {MODEL_NAME}")

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": f"请抽取 ACOS 四元组。\n评论文本: {test_review}\n严格只输出一个 JSON 对象。"}
            ],
            temperature=0.0,
            max_tokens=512,
            response_format={"type": "json_object"} # 测试 vLLM 是否支持 json模式
        )
        
        content = response.choices[0].message.content
        print("-" * 40)
        print("原始返回内容:")
        print(content)
        print("-" * 40)

        # 尝试解析 JSON
        data = json.loads(content)
        print("JSON 解析成功:")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"\n❌ 推理请求失败: {e}")
        # 如果是 404 错误，通常意味着模型名不对
        if "NotFoundError" in str(e):
            print("提示: 请检查 MODEL_NAME 是否与 [Step 1] 中打印的 ID 完全一致（包括路径）。")

if __name__ == "__main__":
    main()