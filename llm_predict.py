import pandas as pd
import asyncio
import json
import re
import sys
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
import os
import argparse

# ================= 配置区域 =================
# 请用您的 vLLM 服务地址替换下面的占位符
VLLM_API_BASE = "http://10.249.42.129:8863/v1"
VLLM_API_KEY = "apikey"
MODEL_NAME = "qwen3_8b"

CONCURRENCY_LIMIT = 1


def get_prompt(text, categories):
    return f"""你是一个专业的评论分析师。
你的任务是从给定的评论文本中，识别出所有的方面（AspectTerms）、观点（OpinionTerms）、类别（Categories）和情感极性（Polarities）。

你需要遵循以下规则：
1.  **输出格式**: 必须严格按照 JSON 格式输出，返回一个对象列表。每个对象包含四个字段: "AspectTerms", "OpinionTerms", "Categories", "Polarities"。**注意：所有字段的值必须是字符串，不能是列表。**
2.  **内容提取**:
    * `AspectTerms`: 评论中提到的具体方面。如果评论没有明确提到方面（是隐式的），请填写"_"。
    * `OpinionTerms`: 表达观点的词语。
    * `Categories`: 从以下列表中选择最合适的类别: {', '.join(categories)}。
    * `Polarities`: 情感极性，必须是 "正面"、"负面" 或 "中性" 之一。
3.  **颗粒度**: 如果一句话包含多个观点（例如“屏幕好但电池差”），请将其拆分为两个独立的对象分别列出。
4.  **无观点**: 如果评论中没有表达任何观点，请返回一个空列表 `[]`。

**评论文本**:
"{text}"

**输出 (JSON格式)**:
"""


def parse_llm_output(response_text):
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        match = re.search(r"```json\n(.*?)```", response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                return []
        return []

def clean_item(item):
    """
    清洗模型输出，使其符合标准答案的格式
    """
    # 1. 清洗 AspectTerms
    aspect = item.get("AspectTerms", "_")
    if isinstance(aspect, list):
        aspect = " ".join([str(x) for x in aspect]) if aspect else "_"
    # 将 "整体" 或 "None" 映射为标准答案中的 "_"
    if str(aspect).strip() in ["整体", "None", "null", "N/A"]:
        aspect = "_"
    item["AspectTerms"] = str(aspect).strip()

    # 2. 清洗 OpinionTerms
    opinion = item.get("OpinionTerms", "_")
    if isinstance(opinion, list):
        opinion = " ".join([str(x) for x in opinion]) if opinion else "_"
    if str(opinion).strip() in ["None", "null", "N/A", "[]"]:
        opinion = "_"
    item["OpinionTerms"] = str(opinion).strip()

    # 3. 清洗 Categories
    cat = item.get("Categories", "整体")
    if isinstance(cat, list):
        cat = ",".join([str(x) for x in cat]) if cat else "整体"
    item["Categories"] = str(cat).strip()

    # 4. 清洗 Polarities
    pol = item.get("Polarities", "中性")
    if isinstance(pol, list):
        pol = ",".join([str(x) for x in pol]) if pol else "中性"
    item["Polarities"] = str(pol).strip()

    return item

async def get_extraction(client, text, categories, semaphore, row_id, progress_file):
    async with semaphore:
        try:
            prompt = get_prompt(text, categories)
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1024,
            )

            response_text = response.choices[0].message.content
            parsed_output = parse_llm_output(response_text)

            # 增强健壮性处理
            if isinstance(parsed_output, dict):
                parsed_output = [parsed_output]
            if not isinstance(parsed_output, list):
                parsed_output = []

            results = []
            for item in parsed_output:
                if isinstance(item, dict):
                    # 在这里进行清洗
                    clean_dict = clean_item(item)
                    clean_dict["id"] = row_id
                    results.append(clean_dict)

            # 如果没有提取到任何结果，生成一条默认的中性记录
            if not results:
                results = [
                    {
                        "id": row_id,
                        "AspectTerms": "_",
                        "OpinionTerms": "_",
                        "Categories": "整体",
                        "Polarities": "中性",
                    }
                ]

            # 将结果追加到JSONL文件
            with open(progress_file, "a", encoding="utf-8") as f:
                for item in results:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            return results

        except Exception as e:
            print(f"Error processing text ID {row_id}: {text[:30]}... Error: {e}")
            results = [
                {
                    "id": row_id,
                    "AspectTerms": "_",
                    "OpinionTerms": "_",
                    "Categories": "整体",
                    "Polarities": "中性",
                }
            ]
            with open(progress_file, "a", encoding="utf-8") as f:
                for item in results:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            return results


async def main(args):
    if VLLM_API_BASE == "YOUR_VLLM_API_BASE" or VLLM_API_KEY == "YOUR_VLLM_API_KEY":
        print("错误：请在脚本中设置 VLLM_API_BASE 和 VLLM_API_KEY")
        # return 

    if args.mode == "train":
        input_file = "./data/TRAIN/Train_reviews.csv"
        output_file = "./data/TRAIN/llm_generated_labels.csv"
        progress_file = "./data/TRAIN/llm_generated_labels.jsonl"
    else:
        input_file = "./data/TEST/Test_reviews.csv"
        output_file = "./data/TEST/llm_generated_results.csv"
        progress_file = "./data/TEST/llm_generated_results.jsonl"

    if not os.path.exists(input_file):
        print(f"错误：找不到输入文件 {input_file}")
        return

    print("正在加载数据...")
    df_reviews = pd.read_csv(input_file)

    processed_ids = set()
    if os.path.exists(progress_file):
        print(f"找到进度文件 {progress_file}，将从中恢复进度...")
        with open(progress_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_ids.add(data["id"])
                except json.JSONDecodeError:
                    continue
        print(f"已处理 {len(processed_ids)} 条记录。")

    unique_categories = [
        "整体",
        "价格",
        "气味",
        "包装",
        "功效",
        "使用体验",
        "物流",
        "服务",
        "成分",
        "其他",
    ]

    client = AsyncOpenAI(api_key=VLLM_API_KEY, base_url=VLLM_API_BASE)
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    tasks = []
    text_col = "Reviews" if "Reviews" in df_reviews.columns else df_reviews.columns[1]
    id_col = df_reviews.columns[0]

    unprocessed_df = df_reviews[~df_reviews[id_col].isin(processed_ids)]

    if unprocessed_df.empty:
        print("所有评论都已处理完毕。")
    else:
        print(
            f"开始使用大模型进行预测... (模式: {args.mode}, 剩余: {len(unprocessed_df)}/{len(df_reviews)})"
        )
        for _, row in unprocessed_df.iterrows():
            text = str(row[text_col])
            row_id = row[id_col]
            tasks.append(
                get_extraction(
                    client, text, unique_categories, semaphore, row_id, progress_file
                )
            )

        await tqdm.gather(*tasks)

    # === 生成最终CSV ===
    print(f"正在从 {progress_file} 生成最终的CSV文件并进行格式清洗...")
    all_results = []
    with open(progress_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                line = line.strip()
                if not line: continue
                item = json.loads(line)
                item = clean_item(item)
                all_results.append(item)
            except json.JSONDecodeError:
                continue

    # ID 对齐与去重逻辑
    id_to_results = {}
    for item in all_results:
        rid = str(item.get("id", ""))
        if rid not in id_to_results:
            id_to_results[rid] = []
        id_to_results[rid].append(item)
    
    final_rows = []
    for _, row in df_reviews.iterrows():
        original_id = row[id_col]
        str_id = str(original_id)
        
        if str_id in id_to_results:
            seen_content = set()
            for res in id_to_results[str_id]:
                if res.get("OpinionTerms") == "_":
                    pass
                
                content_tuple = (
                    res.get("AspectTerms"),
                    res.get("OpinionTerms"),
                    res.get("Categories"),
                    res.get("Polarities")
                )
                if content_tuple not in seen_content:
                    res["id"] = original_id 
                    final_rows.append(res)
                    seen_content.add(content_tuple)
        else:
            final_rows.append({
                "id": original_id,
                "AspectTerms": "_",
                "OpinionTerms": "_",
                "Categories": "整体",
                "Polarities": "中性"
            })

    df_sub = pd.DataFrame(final_rows)
    
    if not df_sub.empty:
        cols_to_keep = ["id", "AspectTerms", "OpinionTerms", "Categories", "Polarities"]
        for col in cols_to_keep:
            if col not in df_sub.columns:
                df_sub[col] = "_"
        df_sub = df_sub[cols_to_keep]

        # === 新增排序逻辑 ===
        # 尝试创建一个临时列，将ID转换为数字，以便正确排序 (1, 2, 10 而不是 1, 10, 2)
        df_sub["_temp_sort_id"] = pd.to_numeric(df_sub["id"], errors="coerce")
        
        # 如果至少有一个ID能转为数字，则优先按数字ID排序，否则按字符串排序
        if df_sub["_temp_sort_id"].notna().sum() > 0:
            df_sub = df_sub.sort_values(by=["_temp_sort_id"])
        else:
            df_sub = df_sub.sort_values(by=["id"])
            
        # 删除临时列
        df_sub = df_sub.drop(columns=["_temp_sort_id"])
        # ==================

    df_sub.to_csv(output_file, index=False)
    print(f"完成！已生成: {output_file} (总行数: {len(df_sub)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="test",
        choices=["train", "test"],
        help="预测模式：为训练集或测试集生成结果",
    )
    args = parser.parse_args()

    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main(args))