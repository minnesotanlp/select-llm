import json
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
import torch
import random

def set_seed(random_state):
    deterministic = True
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)

model = SentenceTransformer('all-MiniLM-L6-v2')

json_path = '/corpora/InstructTune/ait_single/active-instruction-tuning/datasets/data/dolly/data.json'
prompts_path = '/corpora/InstructTune/ait_single/active-instruction-tuning/datasets/data/dolly/prompts.json'
indices_path = '/corpora/InstructTune/ait_single/active-instruction-tuning/datasets/sampled/dolly/llm_search/coreset/1000/sampled_llm_search_1000.parquet.gzip'
clusters_path = '/corpora/InstructTune/ait_single/active-instruction-tuning/datasets/samples/dolly_diverse_kmeans_14_1000.npy'

with open(prompts_path, 'r') as f:
    dolly_prompts = json.load(f)

info_pd = pd.read_csv(json_path)

dolly_format_prompts = []
for i in range(len(info_pd)):
    fsent = f"\###Instruction: {info_pd['instruction'][i]}"
    fsent += f"\n\n###Input: {info_pd['input'][i]}"
    fsent += f"\n\n#Response:"
    dolly_format_prompts.append(fsent)

instructions = info_pd['instruction'].tolist()
outputs = info_pd['output'].tolist()
inputs = info_pd['input'].tolist()

# indices_file = "/corpora/InstructTune/ait_single/active-instruction-tuning/datasets/samples/1208/diverse_kmeans_complex_1k.npy" #1k
indices_file = "/corpora/InstructTune/ait_single/active-instruction-tuning/datasets/samples/1203/diverse_kmeans_complex_3k.npy" #3k

indices_df = pd.read_parquet(indices_path)

# Perform an inner join to find matching rows based on the 'instruction' column
matched_df = pd.merge(info_pd, indices_df, on=['instruction', 'output'], how='inner')

# Extract the indices of the matching rows
matched_indices = matched_df.index.tolist()
coreset_indices = matched_indices

indices_loaded = np.load(indices_file)
indices = indices_loaded.flatten()

sampled_indices = random.sample(list(indices), 100)
sampled_coreset_indices = random.sample(list(coreset_indices), 100)

selected_sampled = np.zeros(len(dolly_prompts), dtype=bool)
selected_sampled[sampled_indices] = True

selected_coreset = np.zeros(len(dolly_prompts), dtype=bool)
selected_coreset[sampled_coreset_indices] = True

non_selected = ~(selected_sampled | selected_coreset)

# Convert the NumPy boolean arrays to lists for BooleanFilter
selected_sampled_list = selected_sampled.tolist()
selected_coreset_list = selected_coreset.tolist()
non_selected_list = non_selected.tolist()

indnum = 1
#Selected clusters
clusters_np = np.load(clusters_path)
selected_cluster_1 = (clusters_np[indnum]).tolist()
selected_cluster_1_array = np.zeros(len(dolly_prompts), dtype=bool)
selected_cluster_1_array[selected_cluster_1] = True
selected_cluster_1_list = selected_cluster_1_array.tolist()

final_1 = indices[indnum*3:indnum*3+3]
chosen_indices_list = final_1.tolist()
final_1_array = np.zeros(len(dolly_prompts), dtype=bool)
final_1_array[final_1] = True
final_1_list = final_1_array.tolist()

instructions_cluster = []
chosen_instructions = []
print("Cluster Instructions are as follows:")
for i in selected_cluster_1:
    instructions_cluster.append(instructions[i])
    print(f'index {i}:\n{instructions[i]}\n')
print('\n Selected Instructions out of these:')
for j in chosen_indices_list:
    chosen_instructions.append(instructions[j])
    print(f'{instructions[j]}')

