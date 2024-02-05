import json
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
import torch
from rouge import Rouge
from transformers import AutoTokenizer, AutoModelForCausalLM 
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import GenerationConfig 
from torch.utils.data import DataLoader
import torch.nn.functional as F
import random
import os
from pathlib import Path
from evaluate import load

FILE_PATH = '/corpora/InstructTune/ait_single/active-instruction-tuning/datasets/scores/'
MODEL_PATH = '/corpora/llama/llama_2_hf/llama-2-7b/7bweights'

class InfoMetrics():
    def __init__(self, data_set):
        self.data_type = data_set

    def get_length(self, data):

        save_file = f'infoverse_{self.data_type}_lengths.json'
        save_dir = Path(FILE_PATH)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = os.path.join(save_dir, save_file) 
        
        if not os.path.exists(save_path):
            input=data['input'].tolist()
            output=data['output'].tolist()
            instruction=data['instruction'].tolist()

            concat_input = [x + y for x,y in zip(input, instruction)]
            
            metric_length = np.array([len(inp) for inp in concat_input], dtype=np.float64)
            metric_length /= np.max(metric_length)

            metric_length = metric_length.tolist()
            with open(save_path, 'w') as f:
                json.dump(metric_length, f)
        else:
            with open(save_path, 'r') as f:
                metric_length = json.load(f)

        return metric_length

    def get_rouge(self, pred_sentences, gt_sentences):

        save_file = f'infoverse_{self.data_type}_rouge.json'
        save_dir = Path(FILE_PATH)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = os.path.join(save_dir, save_file) 

        if not os.path.exists(save_path):
            rouge = Rouge()
            rouge_scores = []
            rougel_scores = []
            for p, g in zip(pred_sentences, gt_sentences):
                if p and g and isinstance(p, str) and isinstance(g, str):  # If the prediction is not empty
                    try:
                        scores = rouge.get_scores(p, g)
                        rouge_scores.append(scores)
                        rougel_scores.append(scores[0]['rouge-l']['f'])
                    except:
                        rouge_scores.append({
                        'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                        'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                        'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}
                    })
                        rougel_scores.append(0)
                else:  # If the prediction is empty
                    rouge_scores.append({
                        'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                        'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                        'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}
                    })
                    rougel_scores.append(0)

            rougel_scores_np = np.array(rougel_scores)
            rougel_mean = np.mean(rougel_scores_np)
            with open(save_path, 'w') as f:
                json.dump(rougel_scores, f)
        else:
            with open(save_path, 'r') as f:
                rougel_scores = json.load(f)

        return rougel_scores

    #Added Batch processing to perp calculation
    def get_perp(self, llama_sentences):
        save_file = f'infoverse_{self.data_type}_perps_new.json'
        save_dir = Path(FILE_PATH)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = os.path.join(save_dir, save_file) 
        print('saving/loading perp from:', save_path)
        
        #Chech values of dummy sentences just to be sure
        # filtered_sentences = [x for x in llama_sentences if len(x.split())==0]
        # for i,sent in enumerate(filtered_sentences):
        #     print('sent is:',type(filtered_sentences))
        #     sent = 'dummy string'
        #     perplexity = load("perplexity", module_type="metric")
        #     results = perplexity.compute(predictions=[sent], model_id='meta-llama/Llama-2-7b-hf')
        #     perplexities = results['mean_perplexity']
        #     print(perplexities)
        # print(llama_sentences)

        for i,sent in enumerate(llama_sentences):
            # print(sent)
            if len(sent.split()) == 0:
                llama_sentences[i] = 'dummy string'
        if not os.path.exists(save_path):
            perplexity = load("perplexity", module_type="metric")
            results = perplexity.compute(predictions=llama_sentences, model_id='meta-llama/Llama-2-7b-hf')
            perplexities = [float(x) for x in results['perplexities']]
            with open(save_path, 'w') as f:
                json.dump(perplexities, f)
        else:
            with open(save_path, 'r') as f:
                perplexities = json.load(f)
            print('Loaded Perps')

        return perplexities
     
    def meta_vector(self, perp_scores, length, rouge):
        all_measurements = []
        all_measurements.append(torch.tensor(perp_scores).unsqueeze(0))
        all_measurements.append(torch.tensor(length).unsqueeze(0))
        all_measurements.append(torch.tensor(rouge).unsqueeze(0))
        return torch.cat(all_measurements, dim=0).t()

if __name__=='__main__':
    metrics = InfoMetrics()
    length = metrics.get_length()
    # print(lamini_length)