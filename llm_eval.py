import pandas as pd


def f1_score(p, g, s):
    pr = s / p if p > 0 else 0
    rc = s / g if g > 0 else 0
    f1 = 2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0
    return f1, pr, rc


def evaluate_results(gt_df, pred_df):
    p, g, s = 0, 0, 0

    # 将数据帧按id分组
    gt_groups = {name: group for name, group in gt_df.groupby("id")}
    pred_groups = {name: group for name, group in pred_df.groupby("id")}

    # 遍历所有唯一的id
    all_ids = set(gt_groups.keys()) | set(pred_groups.keys())

    for review_id in all_ids:
        gt_group = gt_groups.get(review_id)
        pred_group = pred_groups.get(review_id)

        # 将DataFrame转换为元组集合以便于比较
        gt_set = set()
        if gt_group is not None:
            for _, row in gt_group.iterrows():
                gt_set.add(
                    (
                        str(row["AspectTerms"]),
                        str(row["OpinionTerms"]),
                        str(row["Categories"]),
                        str(row["Polarities"]),
                    )
                )

        pred_set = set()
        if pred_group is not None:
            for _, row in pred_group.iterrows():
                pred_set.add(
                    (
                        str(row["AspectTerms"]),
                        str(row["OpinionTerms"]),
                        str(row["Categories"]),
                        str(row["Polarities"]),
                    )
                )

        p += len(pred_set)
        g += len(gt_set)
        s += len(gt_set.intersection(pred_set))

    f1, pr, rc = f1_score(p, g, s)
    return f1, pr, rc


if __name__ == "__main__":
    # 加载真实标签和预测结果
    try:
        gt_df = pd.read_csv("./data/TRAIN/Train_labels.csv")
        pred_df = pd.read_csv("./data/TRAIN/llm_generated_labels.csv")
    except FileNotFoundError as e:
        print(
            f"错误: {e.filename} 文件未找到。请先运行 llm_predict.py --mode train 生成预测文件。"
        )
        exit()

    # 评估结果
    f1, pr, rc = evaluate_results(gt_df, pred_df)

    print("评估结果:")
    print(f"  F1 分数: {f1:.4f}")
    print(f"  精确率: {pr:.4f}")
    print(f"  召回率: {rc:.4f}")
