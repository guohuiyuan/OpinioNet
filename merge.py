import pandas as pd
df_1 = pd.read_csv("data/TEST/qwen3_8b_full_Result.csv",header=None,names=['id', 'AspectTerms', 'OpinionTerms', 'Categories', 'Polarities'])
df_2 = pd.read_csv("data/TEST/qwen3_4b_full_Result.csv",header=None,names=['id', 'AspectTerms', 'OpinionTerms', 'Categories', 'Polarities'])
df_3 = pd.read_csv("data/TEST/qwen3_8b_lora_Result.csv",header=None,names=['id', 'AspectTerms', 'OpinionTerms', 'Categories', 'Polarities'])
candidate_list = []
idx_dict = {}
for index, row in df_1.iterrows():
    candidate_list.append(row.tolist())
    if row['id'] not in idx_dict:
        idx_dict[row['id']] = [row.tolist()]
    else:
        idx_dict[row['id']].append(row.tolist())
for index, row in df_2.iterrows():
    candidate_list.append(row.tolist())
for index, row in df_3.iterrows():
    candidate_list.append(row.tolist())
df_can = pd.DataFrame(candidate_list, columns=['id', 'AspectTerms', 'OpinionTerms', 'Categories', 'Polarities'])
df_can = df_can[df_can.duplicated(keep=False)]
df_can = df_can.drop_duplicates()
print(df_can.shape)
id_list = list(set(df_can['id'].tolist()))
result = []
for i in range(1, 2238):
    if i not in id_list:
        result += idx_dict[i]
df_add = pd.DataFrame(result, columns=['id', 'AspectTerms', 'OpinionTerms', 'Categories', 'Polarities'])
df = pd.concat([df_can, df_add])
sorted_df = df.sort_values(by='id')
sorted_df.to_csv("data/TEST/submit.csv", index=False, header=False)