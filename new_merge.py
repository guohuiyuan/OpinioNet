import pandas as pd
import os
import re
import ast
from collections import Counter

def extract_weight_from_filename(filename):
    """
    从文件名中提取权重
    例如：qwen3_8b_full_Result7963.csv -> 0.7963
    """
    match = re.search(r'Result(\d+\.?\d*)', filename)
    if match:
        weight_str = match.group(1)
        if '.' not in weight_str:
            weight = float(weight_str) / 10000
        else:
            weight = float(weight_str)
        return weight
    return 0.0

def clean_element(text):
    """
    清洗LLM输出的列表型字符串
    例如: "['屏幕']" -> "屏幕"
    """
    text = str(text).strip()
    if text.startswith('[') and text.endswith(']'):
        try:
            items = ast.literal_eval(text)
            if isinstance(items, list) and len(items) > 0:
                return str(items[0]).strip()
        except:
            text = re.sub(r"[\[\]'\" ]", "", text)
    return text.replace("'", "").replace('"', "").strip()

def load_data_to_dict(filepath):
    """加载并清洗数据为 {id: [quadruple, ...]}"""
    try:
        df = pd.read_csv(filepath, header=None, 
                         names=['id', 'Aspect', 'Opinion', 'Category', 'Polarity'])
        res = {}
        for _, row in df.iterrows():
            idx = row['id']
            # 将每一项清洗后存入四元组
            quad = (
                clean_element(row['Aspect']),
                clean_element(row['Opinion']),
                clean_element(row['Category']),
                clean_element(row['Polarity'])
            )
            if idx not in res:
                res[idx] = []
            res[idx].append(quad)
        return res
    except Exception as e:
        print(f"读取 {filepath} 出错: {e}")
        return {}

def main():
    directory = "data/TEST"
    all_files_info = []
    
    # 1. 扫描目录下所有模型结果
    for filename in os.listdir(directory):
        if filename.endswith('.csv') and 'Result' in filename and 'submit' not in filename:
            weight = extract_weight_from_filename(filename)
            all_files_info.append({
                'name': filename,
                'path': os.path.join(directory, filename),
                'score': weight
            })
    
    # 按分数从高到低排序，锁定最高分模型作为兜底模型
    all_files_info.sort(key=lambda x: x['score'], reverse=True)
    if not all_files_info:
        print("未找到结果文件。")
        return

    best_model_info = all_files_info[0]
    print(f"确定最高分模型（用于兜底）: {best_model_info['name']} (Score: {best_model_info['score']})")

    # 2. 选取参与投票的前 N 个模型（建议选 3-5 个）
    voter_files = all_files_info[:5]
    print(f"参与投票的模型数: {len(voter_files)}")
    
    # 加载所有数据
    model_data_list = [load_data_to_dict(f['path']) for f in voter_files]
    best_model_data = model_data_list[0] # 因为已排序，第一个就是最好的

    # 获取所有评论 ID
    all_ids = set()
    for data in model_data_list:
        all_ids.update(data.keys())
    
    final_results = []
    fallback_count = 0

    # 3. 执行合并逻辑
    for idx in sorted(all_ids):
        # 收集该 ID 在所有模型中的预测
        all_quads_for_id = []
        for data in model_data_list:
            all_quads_for_id.extend(data.get(idx, []))
        
        # 统计频次
        counts = Counter(all_quads_for_id)
        
        # 策略 1: 2 票及以上通过
        passed_quads = [q for q, count in counts.items() if count >= 2]
        
        # 策略 2: 如果投票后结果为空，使用最高分模型结果兜底
        if not passed_quads:
            passed_quads = best_model_data.get(idx, [])
            if passed_quads:
                fallback_count += 1
        
        for q in passed_quads:
            final_results.append([idx, q[0], q[1], q[2], q[3]])

    # 4. 保存结果
    df_final = pd.DataFrame(final_results)
    output_path = "data/TEST/submit_weighted_optimized.csv"
    df_final.to_csv(output_path, index=False, header=False)
    
    print("-" * 30)
    print(f"合并完成！共处理 {len(all_ids)} 条 ID")
    print(f"触发兜底逻辑的 ID 数: {fallback_count}")
    print(f"保存至: {output_path}")

if __name__ == "__main__":
    main()