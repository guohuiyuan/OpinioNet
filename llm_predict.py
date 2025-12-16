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

# ================= 配置区域 =================

# vLLM 服务配置
VLLM_API_BASE = "http://10.249.42.129:8863/v1"
VLLM_API_KEY = "apikey"
MODEL_NAME = "qwen3_8b"

# 并发控制
CONCURRENCY_LIMIT = 10  # 根据你的显存情况调整，vLLM通常可以承载较高并发

# 逻辑约束 (来自 predict_acos.py)
CATEGORIES = ["包装","成分","尺寸","服务","功效","价格","气味","使用体验","物流","新鲜度","真伪","整体","其他"]
POLARITIES = ["正面","中性","负面"]

DEFAULT_SYSTEM_PROMPT = """你是一个专业的中文电商化妆品评论观点四元组抽取助手。
任务：给定一条化妆品电商评论文本，抽取其中所有的观点四元组（AspectTerm, OpinionTerm, Category, Polarity），即 ACOS 四元组。

严格遵守：
1. 输出唯一 JSON：{"quadruples":[{"aspect":"...","opinion":"...","category":"...","polarity":"..."}, ...]}
2. quadruples 为数组；若没有观点，输出 {"quadruples":[]}
3. aspect：若无显式属性词用 "_"；原文中出现的需与原文一致；不添加空格。
4. opinion：必须是原文中连续片段；保持原字符；无观点不输出该条。
5. category 取值必须在 {包装, 成分, 尺寸, 服务, 功效, 价格, 气味, 使用体验, 物流, 新鲜度, 真伪, 整体, 其他}
6. polarity 取值必须在 {正面, 中性, 负面}
7. 不得出现与原文无关或臆造的词；不得输出解释文字；不得添加多余字段。
8. 去重：同一 (aspect, opinion, category, polarity) 只保留一个。
9. 排序：按 opinion 在原文首次出现位置升序；若相同再按 aspect（"_" 视为 +∞）。
10. 只输出 JSON，不输出其它任何文本。
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

async def get_extraction(client, text, semaphore, row_id, progress_file):
    async with semaphore:
        # 构造用户输入
        user_content = f"请抽取 ACOS 四元组。\n评论ID: {row_id}\n评论文本: {text}\n严格只输出一个 JSON 对象。"
        
        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.0,
                max_tokens=512, # ACOS 提取不需要太长
                response_format={"type": "json_object"} # 如果 vLLM 支持 json_object 最好，不支持也不影响
            )

            raw_content = response.choices[0].message.content
            json_candidate = extract_json_block(raw_content)
            
            # 使用移植过来的核心逻辑进行处理
            quadruples = validate_and_repair(json_candidate, text)

            # 构造结果对象
            result_obj = {
                "id": row_id,
                "review": text, # 必须保存原文以便后续排序/核对
                "quadruples": quadruples
            }

            # 实时写入 JSONL
            with open(progress_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result_obj, ensure_ascii=False) + "\n")

            return result_obj

        except Exception as e:
            print(f"Error processing ID {row_id}: {e}")
            # 错误时写入空结果，防止中断
            fallback = {"id": row_id, "review": text, "quadruples": []}
            with open(progress_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(fallback, ensure_ascii=False) + "\n")
            return fallback

# ================= 主程序 =================

async def main(args):
    # 路径配置
    if args.mode == "train":
        input_file = "./data/TRAIN/Train_reviews.csv"
        output_file = "./data/TRAIN/Result.csv"
        progress_file = "./data/TRAIN/intermediate_results.jsonl"
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
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    args = parser.parse_args()

    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main(args))