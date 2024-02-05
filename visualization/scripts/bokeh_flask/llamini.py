'''Load the Lamini dataset and sample 1000 points and store them in jsonl files'''

from datasets import load_dataset
import json
import random

dataset = load_dataset("MBZUAI/LaMini-instruction")

data = [x for x in dataset['train']]
sampled_dataset = random.sample(data, 1000)

with open('../lamini/lamini_1000.jsonl', 'w') as f:
    for line in sampled_dataset:
        json.dump(line, f)
        f.write("\n")

with open('../lamini/lamini_prompt_1000.jsonl', 'w') as f:
    for line in sampled_dataset:
        json.dump({'prompt': line['instruction']}, f)
        f.write("\n")