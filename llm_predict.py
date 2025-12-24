import pandas as pd
import asyncio
import json
import re
import sys
import os
import argparse
from typing import List, Tuple, Set, Literal
from dataclasses import dataclass, asdict
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
from pydantic import BaseModel, ValidationError, constr
from collections import Counter # 新增导入

# ================= 配置区域 =================

# vLLM 服务配置
VLLM_API_BASE = "http://10.249.42.129:8000/v1"
VLLM_API_KEY = "apikey"
# MODEL_NAME = "qwen3-8b"
MODEL_NAME = 'qwen3-14b'

# 并发控制
CONCURRENCY_LIMIT = 30  # 根据你的显存情况调整，vLLM通常可以承载较高并发

# 逻辑约束 (来自 predict_acos.py)
CATEGORIES = ["包装","成分","尺寸","服务","功效","价格","气味","使用体验","物流","新鲜度","真伪","整体","其他"]
POLARITIES = ["正面","中性","负面"]

DEFAULT_SYSTEM_PROMPT = """你是一个专业的电商评论观点挖掘专家。请从给定的评论中抽取所有"用户观点四元组"。

四元组定义：(AspectTerm, OpinionTerm, Category, Polarity)
1. AspectTerm (属性词): 商品的具体特征（如"屏幕"、"快递"）。如果未出现具体词，用 "_" 表示。
2. OpinionTerm (观点词): 用户对属性的评价词（如"清晰"、"很快"）。必须保留原文。
3. Category (属性种类): 必须属于以下类别之一：['包装', '成分', '尺寸', '服务', '功效', '价格', '气味', '使用体验', '物流', '新鲜度', '真伪', '整体', '其他']。
4. Polarity (情感极性): 仅限 ['正面', '负面', '中性']。

输出格式要求：
请严格输出一个 JSON 对象，格式如下：
{"quadruples": [{"aspect": "...", "opinion": "...", "category": "...", "polarity": "..."}, ...]}
如果没有观点，输出 {"quadruples": []}
"""

# ================= Pydantic 模型 (用于严格校验) =================

class Quadruple(BaseModel):
    aspect: constr(strip_whitespace=True)
    opinion: constr(strip_whitespace=True, min_length=1)
    category: Literal["包装","成分","尺寸","服务","功效","价格","气味","使用体验","物流","新鲜度","真伪","整体","其他"]
    polarity: Literal["正面","中性","负面"]

class QuadrupleList(BaseModel):
    quadruples: List[Quadruple]

# ================= 核心逻辑函数 =================

def extract_json_block(text: str) -> str:
    """提取 JSON 文本块"""
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    m = re.search(r'\{.*\}', text, flags=re.DOTALL)
    if m:
        return m.group(0)
    return text

def validate_and_repair(json_str: str, original_text: str) -> List[dict]:
    """
    逻辑核心：解析、校验、去重、排序
    返回的是字典列表，方便序列化
    """
    try:
        obj = json.loads(json_str)
    except Exception:
        return []

    if not isinstance(obj, dict):
        return []
    
    items = obj.get("quadruples", [])
    if not isinstance(items, list):
        return []

    cleaned = []
    seen: Set[Tuple[str,str,str,str]] = set()

    for it in items:
        if not isinstance(it, dict):
            continue
        
        aspect = (it.get("aspect") or "").strip()
        if aspect == "" or aspect.lower() in ["none", "null", "n/a"]:
            aspect = "_"
        
        opinion = (it.get("opinion") or "").strip()
        category = (it.get("category") or "").strip()
        polarity = (it.get("polarity") or "").strip()

        # 基本逻辑过滤
        if not opinion or opinion.lower() in ["none", "null", "_", ""]:
            continue
        if category not in CATEGORIES:
            # 尝试简单修复，如果失败则丢弃
            if category == "_": category = "整体" 
            else: continue
        if polarity not in POLARITIES:
            if polarity == "_": polarity = "中性"
            else: continue

        # 去重 Key
        key = (aspect, opinion, category, polarity)
        if key in seen:
            continue
        seen.add(key)

        cleaned.append({
            "aspect": aspect,
            "opinion": opinion,
            "category": category,
            "polarity": polarity
        })

    # 排序辅助函数
    def first_pos(opinion: str) -> int:
        idx = original_text.find(opinion)
        return idx if idx >= 0 else 10**9

    def aspect_pos(aspect: str) -> int:
        if aspect == "_":
            return 10**9
        idx = original_text.find(aspect)
        return idx if idx >= 0 else 10**9 - 1

    # 排序：先按 opinion 出现位置，再按 aspect 出现位置
    cleaned.sort(key=lambda x: (first_pos(x["opinion"]), aspect_pos(x["aspect"])))

    try:
        # 使用 Pydantic 进行最终校验
        qlist = QuadrupleList(quadruples=[Quadruple(**c) for c in cleaned])
        # 转回 dict 用于 JSON 序列化
        return [q.model_dump() for q in qlist.quadruples]
    except ValidationError:
        return []

# ================= 异步任务 =================

def vote_for_best_quadruples(candidates: List[List[dict]]) -> List[dict]:
    """
    自洽性投票：在多个候选列表中选出最频繁出现的那个列表。
    """
    if not candidates:
        return []
    
    # 将 list of dict 转换为可哈希的 tuple of tuples
    # 这样才能放入 Counter 进行计数
    hashable_candidates = []
    for quad_list in candidates:
        # 对列表内的四元组进行排序，确保顺序不影响投票
        # (因为 validate_and_repair 已经排过序了，这里只需转元组)
        temp_tuple = tuple(
            (q['aspect'], q['opinion'], q['category'], q['polarity']) 
            for q in quad_list
        )
        hashable_candidates.append(temp_tuple)
    
    # 统计出现频率
    counts = Counter(hashable_candidates)
    # 获取出现次数最多的结果
    most_common_tuple = counts.most_common(1)[0][0]
    
    # 还原回 list of dict
    final_quads = [
        {"aspect": q[0], "opinion": q[1], "category": q[2], "polarity": q[3]}
        for q in most_common_tuple
    ]
    return final_quads

# ================= 异步任务：自洽性版本 =================

async def get_extraction(client, text, semaphore, row_id, progress_file):
    async with semaphore:
        user_content = f"请抽取 ACOS 四元组。\n评论文本: {text}\n严格只输出一个 JSON 对象。"
        
        try:
            # 修改 1: 调整采样参数
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.7,    # 必须 > 0 才有随机性
                n=5,                # 采样次数，建议 3-10 之间
                max_tokens=1024,    # n 越大，总 tokens 消耗越多，注意增加
                response_format={"type": "json_object"}
            )

            # 修改 2: 处理所有候选结果
            all_candidates = []
            for choice in response.choices:
                raw_content = choice.message.content
                json_candidate = extract_json_block(raw_content)
                # 使用你原有的 validate_and_repair 逻辑清洗每一路结果
                quadruples = validate_and_repair(json_candidate, text)
                all_candidates.append(quadruples)
            
            # 修改 3: 投票选出最终四元组
            final_quadruples = vote_for_best_quadruples(all_candidates)

            result_obj = {
                "id": row_id,
                "review": text,
                "quadruples": final_quadruples
            }

            # 实时写入
            with open(progress_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result_obj, ensure_ascii=False) + "\n")

            return result_obj

        except Exception as e:
            print(f"Error processing ID {row_id}: {e}")
            fallback = {"id": row_id, "review": text, "quadruples": []}
            with open(progress_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(fallback, ensure_ascii=False) + "\n")
            return fallback

# ================= 主程序 =================

async def main(args):
    # 路径配置
    if args.mode == "train":
        input_file = "./data/SPLIT/train_reviews.csv"
        output_file = "./data/TRAIN/Result.csv"
        progress_file = "./data/TRAIN/intermediate_results.jsonl"
    if args.mode == "val":
        input_file = "./data/SPLIT/val_reviews.csv"
        output_file = "./data/VALID/Result.csv"
        progress_file = "./data/VALID/intermediate_results.jsonl"
    else:
        input_file = "./data/TEST/Test_reviews.csv"
        output_file = "./data/TEST/Result.csv"
        progress_file = "./data/TEST/intermediate_results.jsonl"

    if not os.path.exists(input_file):
        print(f"错误：找不到输入文件 {input_file}")
        return

    print("正在加载数据...")
    # 读取 CSV
    df_reviews = pd.read_csv(input_file)
    
    # 自动识别列名
    text_col = "Reviews" if "Reviews" in df_reviews.columns else df_reviews.columns[1]
    id_col = "id" if "id" in df_reviews.columns else df_reviews.columns[0]

    # 恢复进度
    processed_ids = set()
    if os.path.exists(progress_file):
        print(f"从 {progress_file} 恢复进度...")
        with open(progress_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_ids.add(str(data["id"]))
                except json.JSONDecodeError:
                    continue
        print(f"已跳过 {len(processed_ids)} 条记录。")

    # 初始化客户端
    client = AsyncOpenAI(api_key=VLLM_API_KEY, base_url=VLLM_API_BASE)
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    # 准备任务
    tasks = []
    unprocessed_df = df_reviews[~df_reviews[id_col].astype(str).isin(processed_ids)]

    if unprocessed_df.empty:
        print("所有评论都已处理完毕。")
    else:
        print(f"开始任务... 剩余: {len(unprocessed_df)}")
        for _, row in unprocessed_df.iterrows():
            text = str(row[text_col])
            row_id = str(row[id_col])
            tasks.append(
                get_extraction(client, text, semaphore, row_id, progress_file)
            )
        
        # 执行并发
        await tqdm.gather(*tasks)

    # ================= 生成提交文件 (逻辑对齐 predict_acos.py) =================
    print(f"生成最终提交文件: {output_file}")
    
    # 读取所有结果
    id_to_preds = {}
    with open(progress_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                line = line.strip()
                if not line: continue
                obj = json.loads(line)
                id_to_preds[str(obj["id"])] = obj["quadruples"]
            except:
                continue

    csv_lines = []
    # 遍历原始 DataFrame 保证顺序
    for _, row in df_reviews.iterrows():
        rid = str(row[id_col])
        quads = id_to_preds.get(rid, [])
        
        if not quads:
            # 无观点时的占位符逻辑: id, _, _, _, _
            csv_lines.append(f"{rid},_,_,_,_")
        else:
            for q in quads:
                # 确保 Pydantic 字段顺序
                line = f"{rid},{q['aspect']},{q['opinion']},{q['category']},{q['polarity']}"
                csv_lines.append(line)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(csv_lines))

    print(f"完成！提交文件行数: {len(csv_lines)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="val", choices=["train", "val", "test"])
    args = parser.parse_args()

    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main(args))
