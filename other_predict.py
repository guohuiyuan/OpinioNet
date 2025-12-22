# -*- coding: gbk -*- 

import asyncio
import os
import pandas as pd
import json
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

# --- 1. 配置 ---
API_URL = "http://localhost:8000/v1"
API_KEY = "EMPTY" 
# 请确保这里是你启动 vLLM 时用的模型路径
MODEL_NAME = "/new_disk/jhd/SFT/data/ckpt/Qwen3-32B-ABSA-Merged"

TEST_DATA_PATH = "/new_disk/jhd/SFT/code/Test_reviews.csv"
OUTPUT_CSV_PATH = "/new_disk/jhd/SFT/code/Result.csv"

# 并发数：A100 单卡建议 50-80
CONCURRENCY_LIMIT = 80

# --- 2. System Prompt ---
SYSTEM_PROMPT = """你是一个专业的电商评论观点挖掘专家。请从给定的评论中抽取所有“用户观点四元组”。

四元组定义：(AspectTerm, OpinionTerm, Category, Polarity)
1. AspectTerm (属性词): 商品的具体特征（如“屏幕”、“快递”）。如果未出现具体词，用 "_" 表示。
2. OpinionTerm (观点词): 用户对属性的评价词（如“清晰”、“很快”）。必须保留原文。
3. Category (属性种类): 必须属于以下类别之一：['包装', '成分', '尺寸', '服务', '功效', '价格', '气味', '使用体验', '物流', '新鲜度', '真伪', '整体', '其他']。
4. Polarity (情感极性): 仅限 ['正面', '负面', '中性']。

输出格式要求：
请严格输出一个 JSON 对象，格式如下：
{"quadruples": [{"aspect": "...", "opinion": "...", "category": "...", "polarity": "..."}, ...]}
如果没有观点，输出 {"quadruples": []}
"""

async def fetch_prediction(client, row_id, text, semaphore):
    """发送单条请求"""
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text}
                ],
                temperature=0.0, # 贪婪采样，最稳
                max_tokens=512,
                extra_body={"stop": ["<|im_end|>"]}
            )
            return row_id, response.choices[0].message.content
        except Exception as e:
            print(f"\n[Error] ID {row_id} 请求失败: {e}")
            return row_id, None

def parse_output(output_text):
    """解析 JSON"""
    if not output_text: return []
    try:
        start = output_text.find('{')
        end = output_text.rfind('}')
        if start != -1 and end != -1:
            json_str = output_text[start:end+1]
            data = json.loads(json_str)
            return data.get("quadruples", [])
    except:
        pass
    return []

async def main():
    # 1. 读取数据
    if not os.path.exists(TEST_DATA_PATH):
        print(f"❌ 错误：找不到文件 {TEST_DATA_PATH}")
        return
    
    print("正在读取 CSV...")
    df = pd.read_csv(TEST_DATA_PATH)
    df.columns = df.columns.str.strip()
    
    id_col = next((c for c in df.columns if c.lower() == 'id'), 'id')
    review_col = next((c for c in df.columns if 'review' in c.lower()), 'Reviews')
    
    print(f"✅ 加载 {len(df)} 条数据，准备推理...")

    # 2. 初始化客户端
    client = AsyncOpenAI(api_key=API_KEY, base_url=API_URL)
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    # 3. 生成任务
    tasks = []
    for _, row in df.iterrows():
        tasks.append(fetch_prediction(client, str(row[id_col]), str(row[review_col]), semaphore))

    # 4. 并发执行
    results_raw = await tqdm_asyncio.gather(*tasks)

    # 5. 解析结果
    final_rows = []
    for r_id, r_text in results_raw:
        quadruples = parse_output(r_text)
        if not quadruples:
            # 如果为空，根据截图样式，应该是 ID, _, _, _, _
            final_rows.append([r_id, "_", "_", "_", "_"])
        else:
            for q in quadruples:
                # 【修正点】 截图显示的顺序：ID, Aspect, Opinion, Category, Polarity
                # 第4列是 Category (如"气味")，第5列是 Polarity (如"正面")
                final_rows.append([
                    r_id,
                    q.get("aspect", "_"),
                    q.get("opinion", "_"),
                    q.get("category", "_"), #  <-- 这里改了，Category 放在第4列
                    q.get("polarity", "_")  #  <-- 这里改了，Polarity 放在最后
                ])

    # 6. 保存 Result.csv
    print(f"💾 正在保存结果到 {OUTPUT_CSV_PATH} ...")
    # DataFrame 列名顺序也对应调整，方便检查
    res_df = pd.DataFrame(final_rows, columns=["ID", "AspectTerms", "OpinionTerms", "Categories", "Polarities"])
    
    # 排序 ID
    try:
        res_df['sort_id'] = pd.to_numeric(res_df['ID'], errors='coerce')
        res_df = res_df.sort_values('sort_id').drop('sort_id', axis=1)
    except:
        res_df = res_df.sort_values('ID')

    # 无表头，UTF-8
    res_df.to_csv(OUTPUT_CSV_PATH, index=False, header=False, encoding='utf-8')
    print("🎉 全部完成！顺序已修正为：ID, Aspect, Opinion, Category, Polarity")

if __name__ == "__main__":
    asyncio.run(main())