#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据转换脚本：ABSA CSV -> JSONL (微调格式)

功能：
将 ABSA 四元组标注数据 (Train_reviews.csv + Train_labels.csv)
转换为 SiliconFlow / OpenAI 微调所需的 .jsonl 格式。
每行一个 JSON，包含 messages: [system, user, assistant]

默认用法（只要文件在 ./data/TRAIN/ 下）：
    python convert_absa_to_jsonl.py

自定义路径用法：
    python convert_absa_to_jsonl.py \
      --reviews ./my_data/reviews.csv \
      --labels ./my_data/labels.csv \
      --output ./my_data/finetune_data.jsonl
"""

import argparse
import csv
import json
import sys
import os
from collections import defaultdict
from pathlib import Path

# ================= 配置区域 =================

# 默认文件路径（根据您的目录结构设定）
DEFAULT_REVIEWS_PATH = "./data/TRAIN/Train_reviews.csv"
DEFAULT_LABELS_PATH  = "./data/TRAIN/Train_labels.csv"
DEFAULT_OUTPUT_PATH  = "./data/TRAIN/train.jsonl"

DEFAULT_SYSTEM_PROMPT = """你是一个专业的中文电商化妆品评论观点四元组抽取助手。
任务：给定一条化妆品电商评论文本，抽取其中所有的观点四元组（AspectTerm, OpinionTerm, Category, Polarity），即 ACOS 四元组。

请严格遵守以下规范：
1. 四元组定义
  - aspect: 属性词（若评论中未出现显式属性词，用 "_" 标记）
  - opinion: 观点情感词（必须与原文中的字面内容完全一致，不得增删或形态改写；如果无观点则不输出该条四元组）
  - category: 必须从以下 13 个预定义类别中选择（逐字匹配）：
   {包装, 成分, 尺寸, 服务, 功效, 价格, 气味, 使用体验, 物流, 新鲜度, 真伪, 整体, 其他}
  - polarity: 情感极性，取值仅限：{正面, 中性, 负面}
2. 当一条评论包含多条独立观点时，输出多个四元组；不得合并。
3. aspect = "_" 仅表示“隐式属性”而非未知；隐式属性仍需给出正确的 category。
4. opinion 必须是评论原文中连续的片段，保持原始顺序与字符。
5. 不要产生重复四元组；同一 (aspect, opinion, category, polarity) 只保留一份。
6. 不要臆造评论中不存在的观点；如果确实没有可抽取的观点，返回空数组。
7. 输出格式：严格输出一个 JSON 对象字符串：
  {"quadruples":[{"aspect":"...","opinion":"...","category":"...","polarity":"..."}, ...]}
  - quadruples 的值为一个列表
  - 列表为空则输出 {"quadruples":[]}
  - 键名必须是 aspect / opinion / category / polarity（全部小写）
  - 字符串使用原始中文，不转义除非 JSON 需要
8. 不添加任何额外解释、注释、换行提示或自然语言说明。仅输出 JSON。
9. 保持输出 determinism：同一输入始终输出相同顺序的四元组。排序规则：按 opinion 在原文出现的起始位置升序；如 opinion 相同则按 aspect 起始位置；若 aspect 为 "_" 视其开始位置为 +∞。
10. 若同一个 opinion 对不同 category 的确合理，可分别输出；但请避免不必要的多类别解释。"""

# ================= 核心逻辑 =================

def read_reviews(path, encoding="utf-8-sig"):
    """读取评论文本 CSV: id, Reviews"""
    reviews = {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到评论文件: {path}")

    with open(path, "r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row.get("id")
            if rid is None:
                continue
            rid = rid.strip()
            # 兼容列名 Reviews 或 Review
            txt = row.get("Reviews") or row.get("Review") or ""
            reviews[rid] = txt.strip()
    return reviews


def read_labels(path, encoding="utf-8-sig"):
    """读取标签 CSV，返回 id -> list of dict"""
    grouped = defaultdict(list)
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到标签文件: {path}")

    with open(path, "r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        required = {"id", "AspectTerms", "OpinionTerms", "Categories", "Polarities"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            # 尝试兼容单数形式
            pass 

        for row in reader:
            rid = row.get("id", "").strip()
            aspect = row.get("AspectTerms", "").strip()
            opinion = row.get("OpinionTerms", "").strip()
            cat = row.get("Categories", "").strip()
            pol = row.get("Polarities", "").strip()

            # 读取位置信息（如果有，用于排序优化）
            try:
                o_str = row.get("O_start", "").strip()
                o_start = int(o_str) if o_str else None
            except:
                o_start = None
            
            try:
                a_str = row.get("A_start", "").strip()
                a_start = int(a_str) if a_str else None
            except:
                a_start = None

            grouped[rid].append({
                "aspect": aspect if aspect else "_",
                "opinion": opinion if opinion else "",
                "category": cat,
                "polarity": pol,
                "o_start": o_start,
                "a_start": a_start
            })
    return grouped


def normalize_and_sort(labels_for_one):
    """去重并按原文顺序排序"""
    seen = set()
    cleaned = []
    
    for item in labels_for_one:
        aspect = item["aspect"] if item["aspect"] else "_"
        opinion = item["opinion"]
        category = item["category"]
        polarity = item["polarity"]

        # 过滤无效数据：没有 opinion 则不构造该条
        if not opinion or opinion == "_":
            continue

        key = (aspect, opinion, category, polarity)
        if key in seen:
            continue
        seen.add(key)
        
        cleaned.append({
            "aspect": aspect,
            "opinion": opinion,
            "category": category,
            "polarity": polarity,
            "o_start": item["o_start"],
            "a_start": item["a_start"]
        })

    # 排序函数
    def sort_key(x):
        o_start = x["o_start"]
        a_start = x["a_start"]
        
        # 如果没有位置信息(CSV里没给)，则放到最后
        if o_start is None:
            o_start = float("inf")
        
        if a_start is None:
            # aspect 为 _ 时，视为无穷大
            if x["aspect"] == "_":
                a_start = float("inf")
            else:
                # 有 aspect 但没位置，暂且排在 _ 之前
                a_start = float("inf") - 1
        
        return (o_start, a_start)

    cleaned.sort(key=sort_key)

    # 移除辅助排序的字段，只保留训练需要的字段
    final_list = [{
        "aspect": c["aspect"],
        "opinion": c["opinion"],
        "category": c["category"],
        "polarity": c["polarity"]
    } for c in cleaned]
    
    return final_list


def build_jsonl(reviews, labels_grouped, system_prompt, output_path, add_missing_ids=False, encoding="utf-8"):
    all_ids = list(reviews.keys())

    # 检查是否有 label 存在但 review 缺失的情况（通常不应发生）
    if add_missing_ids:
        more = [i for i in labels_grouped.keys() if i not in reviews]
        if more:
            print(f"[WARN] 发现 {len(more)} 个 ID 在标签中有但在评论中缺失，将添加占位符。", file=sys.stderr)
            for mid in more:
                reviews[mid] = ""  # 占位
                all_ids.append(mid)

    # 排序 ID 以保证输出顺序固定
    try:
        all_ids.sort(key=lambda x: int(x))
    except:
        all_ids.sort()

    count = 0
    with open(output_path, "w", encoding=encoding) as out_f:
        for rid in all_ids:
            review_text = reviews[rid]
            label_items = labels_grouped.get(rid, [])
            
            # 转换为标准 JSON 结构
            quadruples = normalize_and_sort(label_items) if label_items else []
            assistant_obj = {"quadruples": quadruples}

            # 构造 User Prompt
            user_content = f"请从下面这条评论中抽取所有 ACOS 观点四元组。\n评论ID: {rid}\n评论文本: {review_text}"

            # 构造 ChatML Message
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
                # 使用 ensure_ascii=False 保证中文可读，compact separators 节省 token
                {"role": "assistant", "content": json.dumps(assistant_obj, ensure_ascii=False, separators=(',', ':'))}
            ]

            line_obj = {"messages": messages}
            out_f.write(json.dumps(line_obj, ensure_ascii=False) + "\n")
            count += 1
            
    print(f"成功处理 {count} 条数据。")


def main():
    parser = argparse.ArgumentParser(description="转换 ABSA 数据为微调 JSONL 格式")
    
    # 设置默认值为之前对话中明确的文件结构
    parser.add_argument("--reviews", default=DEFAULT_REVIEWS_PATH, help="Train_reviews.csv 路径")
    parser.add_argument("--labels", default=DEFAULT_LABELS_PATH, help="Train_labels.csv 路径")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="输出 .jsonl 路径")
    
    parser.add_argument("--system-prompt-file", help="外部 system prompt 文件路径（可选）")
    parser.add_argument("--encoding", default="utf-8-sig", help="输入 CSV 编码")
    parser.add_argument("--add_missing_ids", action="store_true", help="是否补全标签中有但评论中缺失的ID")
    
    args = parser.parse_args()

    # 1. 确定 System Prompt
    if args.system_prompt_file:
        try:
            system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8").strip()
        except Exception as e:
            print(f"读取 Prompt 文件失败: {e}")
            sys.exit(1)
    else:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    # 2. 读取数据
    print(f"读取评论: {args.reviews}")
    try:
        reviews = read_reviews(args.reviews, encoding=args.encoding)
    except Exception as e:
        print(f"错误: {e}")
        return

    print(f"读取标签: {args.labels}")
    try:
        labels_grouped = read_labels(args.labels, encoding=args.encoding)
    except Exception as e:
        print(f"错误: {e}")
        return

    if not reviews:
        print("错误: 未读取到任何评论数据。", file=sys.stderr)
        sys.exit(1)

    # 3. 生成 JSONL
    print(f"正在生成: {args.output}")
    build_jsonl(
        reviews=reviews,
        labels_grouped=labels_grouped,
        system_prompt=system_prompt,
        output_path=args.output,
        add_missing_ids=args.add_missing_ids,
        encoding="utf-8"
    )
    print(f"完成！文件已保存至: {args.output}")


if __name__ == "__main__":
    main()