import pandas as pd
import argparse
import os
import json
import re

def f1_score(p, g, s):
    pr = s / p if p > 0 else 0
    rc = s / g if g > 0 else 0
    f1 = 2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0
    return f1, pr, rc

def load_data(file_path, is_prediction=False):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件未找到: {file_path}")

    if is_prediction:
        # 对应 Result.csv 格式: rid,aspect,opinion,category,polarity
        df = pd.read_csv(
            file_path, 
            header=None, 
            names=["id", "AspectTerms", "OpinionTerms", "Categories", "Polarities"],
            dtype=str
        )
    else:
        df = pd.read_csv(file_path, dtype=str)
        rename_map = {
            "AspectTerm": "AspectTerms", "OpinionTerm": "OpinionTerms", 
            "Category": "Categories", "Polarity": "Polarities",
            "ReviewID": "id", "ID": "id", "text": "review", "sentence": "review"
        }
        df = df.rename(columns=rename_map)

    df = df.fillna("_")
    df["id"] = df["id"].astype(str).str.strip()
    return df

def get_valid_quadruples(group_df):
    quad_set = set()
    if group_df is None or group_df.empty:
        return quad_set
    for _, row in group_df.iterrows():
        aspect, opinion = str(row.get("AspectTerms", "_")).strip(), str(row.get("OpinionTerms", "_")).strip()
        category, polarity = str(row.get("Categories", "_")).strip(), str(row.get("Polarities", "_")).strip()
        if opinion == "_" or opinion == "": continue
        quad_set.add((aspect, opinion, category, polarity))
    return quad_set

def evaluate_results(gt_df, pred_df, id_to_text):
    p, g, s = 0, 0, 0
    error_records = []
    
    gt_groups = {str(k): v for k, v in gt_df.groupby("id")}
    pred_groups = {str(k): v for k, v in pred_df.groupby("id")}
    
    # 将所有 ID 按顺序排列
    # 尝试按数字排序，如果不行则按字符串排序
    all_ids = list(set(gt_groups.keys()) | set(pred_groups.keys()))
    all_ids.sort(key=lambda x: int(x) if x.isdigit() else x)

    for review_id in all_ids:
        gt_group, pred_group = gt_groups.get(review_id), pred_groups.get(review_id)
        
        # 关联原始评论文本
        review_text = id_to_text.get(review_id, "")
        if not review_text and gt_group is not None and "review" in gt_group.columns:
            review_text = str(gt_group["review"].iloc[0])

        gt_set, pred_set = get_valid_quadruples(gt_group), get_valid_quadruples(pred_group)
        p, g = p + len(pred_set), g + len(gt_set)
        s += len(gt_set.intersection(pred_set))

        if gt_set != pred_set:
            error_records.append({
                "id": review_id,
                "review": review_text,
                "missing": [list(item) for item in sorted(list(gt_set - pred_set))],
                "extra": [list(item) for item in sorted(list(pred_set - gt_set))],
                "ground_truth": [list(item) for item in sorted(list(gt_set))],
                "predictions": [list(item) for item in sorted(list(pred_set))]
            })
    return f1_score(p, g, s), p, g, s, error_records

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ACOS 评测工具")
    parser.add_argument("--gt", type=str, default="./data/SPLIT/val_labels.csv", help="标签文件")
    parser.add_argument("--pred", type=str, default="./data/VALID/Result.csv", help="预测结果")
    parser.add_argument("--text_file", type=str, default="./data/SPLIT/val_reviews.csv", help="评论文本")
    parser.add_argument("--error_file", type=str, default="error_results.json", help="错误结果路径")
    args = parser.parse_args()

    id_map = {}
    if os.path.exists(args.text_file):
        tdf = pd.read_csv(args.text_file, dtype=str)
        # 兼容 llm_predict.py 中的列名逻辑
        t_col = "Reviews" if "Reviews" in tdf.columns else tdf.columns[1]
        i_col = "id" if "id" in tdf.columns else tdf.columns[0]
        id_map = dict(zip(tdf[i_col].astype(str).str.strip(), tdf[t_col]))

    try:
        gt_df, pred_df = load_data(args.gt), load_data(args.pred, True)
        (f1, pr, rc), p, g, s, errors = evaluate_results(gt_df, pred_df, id_map)

        print("-" * 30)
        print(f"评估结果:")
        print(f"  Precision: {pr:.4f}")
        print(f"  Recall:    {rc:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print("-" * 30)

        # 序列化并压缩内层数组
        full_json = json.dumps(errors, ensure_ascii=False, indent=4)
        compact_json = re.sub(r'\[\s+((?:"[^"]*",?\s*)+)\s+\]', 
                             lambda m: "[" + ", ".join([s.strip() for s in m.group(1).split(",")]) + "]", 
                             full_json)
        
        with open(args.error_file, "w", encoding="utf-8") as f:
            f.write(compact_json)
        
        print(f"已按顺序将 {len(errors)} 条错误导出至: {args.error_file}")

    except Exception as e:
        import traceback
        traceback.print_exc()