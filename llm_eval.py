import pandas as pd
import argparse
import os

def f1_score(p, g, s):
    pr = s / p if p > 0 else 0
    rc = s / g if g > 0 else 0
    f1 = 2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0
    return f1, pr, rc

def load_data(file_path, is_prediction=False):
    """
    加载数据并统一格式
    is_prediction: 如果是预测文件，通常没有表头，需要手动赋予列名
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件未找到: {file_path}")

    if is_prediction:
        # 预测结果通常无表头，按标准顺序指定
        df = pd.read_csv(
            file_path, 
            header=None, 
            names=["id", "AspectTerms", "OpinionTerms", "Categories", "Polarities"],
            dtype=str
        )
    else:
        # 真值文件通常有表头
        df = pd.read_csv(file_path, dtype=str)
        # 兼容可能的列名差异（单数/复数）
        rename_map = {
            "AspectTerm": "AspectTerms",
            "OpinionTerm": "OpinionTerms", 
            "Category": "Categories",
            "Polarity": "Polarities",
            "ReviewID": "id",
            "ID": "id"
        }
        df = df.rename(columns=rename_map)

    # 填充空值为 "_"
    df = df.fillna("_")
    
    # 确保 ID 是字符串类型，防止 int 和 str 无法匹配
    df["id"] = df["id"].astype(str).str.strip()
    
    return df

def get_valid_quadruples(group_df):
    """
    从 DataFrame 分组中提取有效的四元组集合。
    过滤掉 OpinionTerms 为 "_" 的行（即无观点的占位行）。
    """
    quad_set = set()
    if group_df is None or group_df.empty:
        return quad_set

    for _, row in group_df.iterrows():
        aspect = str(row.get("AspectTerms", "_")).strip()
        opinion = str(row.get("OpinionTerms", "_")).strip()
        category = str(row.get("Categories", "_")).strip()
        polarity = str(row.get("Polarities", "_")).strip()

        # 核心逻辑：如果 Opinion 是 "_"，则视为无效观点，不参与 F1 计算
        # (即：模型预测“无观点”且真值也是“无观点”时，既不加分也不扣分，直接忽略)
        if opinion == "_" or opinion == "":
            continue

        quad_set.add((aspect, opinion, category, polarity))
    
    return quad_set

def evaluate_results(gt_df, pred_df):
    p, g, s = 0, 0, 0

    # 按 ID 分组
    gt_groups = {str(k): v for k, v in gt_df.groupby("id")}
    pred_groups = {str(k): v for k, v in pred_df.groupby("id")}

    # 获取所有涉及的 ID
    all_ids = set(gt_groups.keys()) | set(pred_groups.keys())

    for review_id in all_ids:
        gt_group = gt_groups.get(review_id)
        pred_group = pred_groups.get(review_id)

        # 获取清洗后的四元组集合
        gt_set = get_valid_quadruples(gt_group)
        pred_set = get_valid_quadruples(pred_group)

        # 统计数量
        # p (Predict): 预测出的有效观点总数
        p += len(pred_set)
        # g (Ground Truth): 真实的有效观点总数
        g += len(gt_set)
        # s (Success): 预测正确的数量（交集）
        s += len(gt_set.intersection(pred_set))

    f1, pr, rc = f1_score(p, g, s)
    return f1, pr, rc, p, g, s

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ACOS 评测脚本")
    parser.add_argument("--gt", type=str, default="./data/TRAIN/Train_labels.csv", help="真实标签文件路径 (带表头)")
    parser.add_argument("--pred", type=str, default="./data/TRAIN/Result.csv", help="预测结果文件路径 (无表头)")
    args = parser.parse_args()

    print(f"正在评估...")
    print(f"  真实文件: {args.gt}")
    print(f"  预测文件: {args.pred}")

    try:
        # 加载数据
        gt_df = load_data(args.gt, is_prediction=False)
        pred_df = load_data(args.pred, is_prediction=True) # 这里设置为 True，处理无表头的情况

        # 运行评估
        f1, pr, rc, p, g, s = evaluate_results(gt_df, pred_df)

        print("-" * 30)
        print("评估结果 (Micro-F1):")
        print(f"  Precision (精确率): {pr:.4f}  (Correct: {s} / Predicted: {p})")
        print(f"  Recall    (召回率): {rc:.4f}  (Correct: {s} / Gold: {g})")
        print(f"  F1 Score  (F1分数): {f1:.4f}")
        print("-" * 30)

    except Exception as e:
        print(f"发生错误: {e}")
        # 打印更详细的错误堆栈以便调试
        import traceback
        traceback.print_exc()