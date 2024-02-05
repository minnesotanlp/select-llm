import json
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
import panel as pn
import torch

class MetricLength():
    def __init__(self):
        with open('../../datasets/alpaca/alpaca_prompts_1000.jsonl', 'r') as f:
            self.alpaca_prompts = [json.loads(line)['prompt'] for line in f]

        with open('../../datasets/alpaca/alpaca_data_1000.jsonl', 'r') as f:
            self.alpaca_data = [json.loads(line) for line in f]

        with open('../../datasets/alpaca/alpaca_cleaned_new.jsonl', 'r') as f:
            self.alpaca_predicted = [line.strip() for line in f]

        # Load Lamini data
        with open('../../datasets/lamini/lamini_prompt_1000.jsonl', 'r') as f:
            self.lamini_prompts = [json.loads(line)['prompt'] for line in f]

        with open('../../datasets/lamini/lamini_1000.jsonl', 'r') as f:
            self.lamini_data = [json.loads(line) for line in f]

        with open('../../datasets/lamini/lamini_cleaned_new.jsonl', 'r') as f:
            self.lamini_predicted = [line.strip() for line in f]

    def get_alpaca_length(self):
        alpaca_input=[item['input'] for item in self.alpaca_data]
        alpaca_output=[item['output'] for item in self.alpaca_data]
        alpaca_instruction=[item['instruction'] for item in self.alpaca_data]

        concat_input = [x + y for x,y in zip(alpaca_input, alpaca_instruction)]
        
        alpaca_metric_length = np.array([len(inp) for inp in concat_input], dtype=np.float64)
        alpaca_metric_length /= np.max(alpaca_metric_length)

        return alpaca_metric_length

    def get_lamini_length(self):

        lamini_instruction=[item['instruction'] for item in self.lamini_data]
        lamini_output=[item['response'] for item in self.lamini_data]
        
        lamini_metric_length = np.array([len(inp) for inp in lamini_instruction], dtype=np.float64)
        mean = np.mean(lamini_metric_length)
        lamini_metric_length /= np.max(lamini_metric_length)

        return list(lamini_metric_length)
     
    def meta_vector(self, perp_scores, length, rouge):
        all_measurements = []
        all_measurements.append(torch.tensor(perp_scores).unsqueeze(0))
        all_measurements.append(torch.tensor(length).unsqueeze(0))
        all_measurements.append(torch.tensor(rouge).unsqueeze(0))
        return torch.cat(all_measurements, dim=0).t()

if __name__=='__main__':
    metrics = MetricLength()
    alpaca_length = metrics.get_alpaca_length()
    lamini_length = metrics.get_lamini_length()
    # print(lamini_length)
