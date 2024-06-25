import json
import argparse
import numpy as np
from pathlib import Path
from rouge import Rouge
from evaluate import load
from sentence_transformers import SentenceTransformer, util

class Evaluations():
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.output_dir = Path(__file__).parent.parent.joinpath('llama_2_hf', 'Instructllama-2-7b')
        self.model_path = Path(__file__).parent.parent.joinpath('llama-2-7b', '7bweights')
        
    def perplexity(self, pred_sentences):
        perplexity = load("perplexity", module_type="metric")
        results = perplexity.compute(predictions=pred_sentences, model_id='meta-llama/Llama-2-7b-hf')
        
        return round(results["mean_perplexity"], 2)

    def cossims(self, gt_sentences, pred_sentences):
        embeddings_gt = self.model.encode(gt_sentences)
        embeddings_pred = self.model.encode(pred_sentences)

        cos_sim = util.cos_sim(embeddings_gt, embeddings_pred)

        # Add all pairs to a list with their cosine similarity score
        all_sentence_combinations = []

        for i in range(len(cos_sim)):
            all_sentence_combinations.append(round(cos_sim[i][i].item(), 4))
        
        cossim_mean = np.mean(np.array(all_sentence_combinations))
        cossim_std = np.std(np.array(all_sentence_combinations))

        return all_sentence_combinations, cossim_mean, cossim_std

    def rouge(self, pred_sentences, gt_sentences):
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
        rougel_std = np.std(rougel_scores_np)

        return rouge_scores, rougel_scores, rougel_mean, rougel_std

if __name__=='__main__':
    evaluation = Evaluations()

    parser = argparse.ArgumentParser(description="Process a jsonl file.")
    parser.add_argument('-d','--data_set', default='all', help="Dataset to use for evaluation.")
    parser.add_argument('-s','--sample_type', help="sampling type")
    parser.add_argument('-f','--ftype', help="specific sampling type")
    parser.add_argument('-n','--n_instances', help="no. of instances")
    parser.add_argument('-l','--local_model', type=str, help="Model for SelectLLM. Options: [gpt3.5, mixtral]")
    parser.add_argument('-fm','--finetune_model', type=str, help="Model to finetune: [llama, mistral]")
    args = parser.parse_args()

    data_set = args.data_set
    sample_type = args.sample_type
    n_instances = args.n_instances
    local_selection_model = args.local_model
    finetune_model = args.finetune_model
    preds_dir = Path(__file__).parent.parent.joinpath('datasets', 'test', data_set)
    if sample_type == 'infoverse':
        preds_rs1 = preds_dir.joinpath(f'{finetune_model}_{sample_type}_{args.ftype}_{n_instances}_formatted_det_rs_2021.json')
        preds_rs2 = preds_dir.joinpath(f'{finetune_model}_{sample_type}_{args.ftype}_{n_instances}_formatted_det_rs_2022.json')
        preds_rs3 = preds_dir.joinpath(f'{finetune_model}_{sample_type}_{args.ftype}_{n_instances}_formatted_det_rs_2023.json')
    elif sample_type == 'selectllm':
        preds_rs1 = preds_dir.joinpath(f'{finetune_model}_{sample_type}_{args.ftype}_{local_selection_model}_{n_instances}_formatted_det_rs_2021.json')
        preds_rs2 = preds_dir.joinpath(f'{finetune_model}_{sample_type}_{args.ftype}_{local_selection_model}_{n_instances}_formatted_det_rs_2022.json')
        preds_rs3 = preds_dir.joinpath(f'{finetune_model}_{sample_type}_{args.ftype}_{local_selection_model}_{n_instances}_formatted_det_rs_2023.json')
    else:
        preds_rs1 = preds_dir.joinpath(f'{finetune_model}_{sample_type}_{n_instances}_formatted_det_rs_2021.json')
        preds_rs2 = preds_dir.joinpath(f'{finetune_model}_{sample_type}_{n_instances}_formatted_det_rs_2022.json')
        preds_rs3 = preds_dir.joinpath(f'{finetune_model}_{sample_type}_{n_instances}_formatted_det_rs_2023.json')

    gts_path = Path(__file__).parent.parent.joinpath('datasets', 'data', data_set, 'test_gts.json')
    with open(preds_rs1, 'r') as f:
        preds1 = json.load(f)
    with open(preds_rs2, 'r') as f:
        preds2 = json.load(f)
    with open(preds_rs3, 'r') as f:
        preds3 = json.load(f)
    with open(gts_path, 'r') as g: 
        gts = json.load(g)

    filename = preds_rs1.name
    
    print('#'*210 + '\n')
    
    _, _, camel_rouge_l_mean1, camel_rouge_l_std1 = evaluation.rouge(preds1, gts)
    _, camel_cossims_mean1, camel_cossims_std1 = evaluation.cossims(gts, preds1)
    print(f'For {filename}_{data_set}:\n\n rouge1 mean:{camel_rouge_l_mean1}, std:{camel_rouge_l_std1} cos1 mean:{camel_cossims_mean1}, std:{camel_cossims_std1}\n')

    _, _, camel_rouge_l_mean2, camel_rouge_l_std2 = evaluation.rouge(preds2, gts)
    _, camel_cossims_mean2, camel_cossims_std2 = evaluation.cossims(gts, preds2)

    _, _, camel_rouge_l_mean3, camel_rouge_l_std3 = evaluation.rouge(preds3, gts)
    _, camel_cossims_mean3, camel_cossims_std3 = evaluation.cossims(gts, preds3)

    camel_rouge_l_mean = (camel_rouge_l_mean1 + camel_rouge_l_mean2 + camel_rouge_l_mean3)/3.0
    camel_rouge_l_std = (camel_rouge_l_std1 + camel_rouge_l_std2 + camel_rouge_l_std3)/3.0
    camel_cossims_mean = (camel_cossims_mean1 + camel_cossims_mean2 + camel_cossims_mean3)/3.0
    camel_cossims_std = (camel_cossims_std1 + camel_cossims_std2 + camel_cossims_std3)/3.0

    print(f'For {filename}_{data_set}:\n\n rouge1 mean:{camel_rouge_l_mean1}, std:{camel_rouge_l_std1} cos1 mean:{camel_cossims_mean1}, std:{camel_cossims_std1} \n rouge2 mean:{camel_rouge_l_mean2}, std:{camel_rouge_l_std2} cos2 mean:{camel_cossims_mean2}, std:{camel_cossims_std2} \n rouge3 mean:{camel_rouge_l_mean3}, std:{camel_rouge_l_std3} cos3 mean:{camel_cossims_mean3}, std:{camel_cossims_std3} \n\n rouge mean:{camel_rouge_l_mean}, std:{camel_rouge_l_std} cossims mean:{camel_cossims_mean}, std:{camel_cossims_std}\n\n')
