import json
import argparse
import numpy as np
from pathlib import Path
from rouge import Rouge
from evaluate import load
from sentence_transformers import SentenceTransformer, util

class Preprocess():
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

        return all_sentence_combinations, cossim_mean

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

        return rouge_scores, rougel_scores, rougel_mean



if __name__=='__main__':
    pre_process = Preprocess()

    parser = argparse.ArgumentParser(description="Process a jsonl file.")
    parser.add_argument('--data_set', default='all', help="Dataset to use for evaluation.")
    parser.add_argument('--sample_type', help="sampling type")
    parser.add_argument('--ftype', help="specific sampling type")
    parser.add_argument('--n_instances', help="no. of instances")
    args = parser.parse_args()

    data_set = args.data_set
    sample_type = args.sample_type
    n_instances = args.n_instances
    preds_dir = Path(__file__).parent.parent.joinpath('datasets', 'test', data_set)
    if args.ftype:
        preds_rs1 = preds_dir.joinpath(f'{sample_type}_{args.ftype}_{n_instances}_formatted_det_rs_2021.json')
        preds_rs2 = preds_dir.joinpath(f'{sample_type}_{args.ftype}_{n_instances}_formatted_det_rs_2022.json')
        preds_rs3 = preds_dir.joinpath(f'{sample_type}_{args.ftype}_{n_instances}_formatted_det_rs_2023.json')
    else:
        preds_rs1 = preds_dir.joinpath(f'{sample_type}_{n_instances}_formatted_det_rs_2021.json')
        preds_rs2 = preds_dir.joinpath(f'{sample_type}_{n_instances}_formatted_det_rs_2022.json')
        preds_rs3 = preds_dir.joinpath(f'{sample_type}_{n_instances}_formatted_det_rs_2023.json')

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
    sepr = '#'
    for i in range(210):
        sepr = sepr + '#'
    print(sepr)
    print("")

    # camel_rouge_scores, camel_rougel_f_scores, camel_rouge_l_mean = pre_process.rouge(preds, gts)
    # camel_cossims_scores, camel_cossims_mean = pre_process.cossims(gts, preds)
    # camel_perp = pre_process.perplexity(preds)
    # print(camel_rouge_l_mean)
    # print(f'For {filename}: rouge mean:{camel_rouge_l_mean} cossims mean:{camel_cossims_mean}\n\n perp:{camel_perp} \n\n')
    
    _, _, camel_rouge_l_mean1 = pre_process.rouge(preds1, gts)
    _, camel_cossims_mean1 = pre_process.cossims(gts, preds1)
    camel_perp1 = pre_process.perplexity(preds1)
    print(f'For {filename}_{data_set}:\n\n rouge1:{camel_rouge_l_mean1} cos1:{camel_cossims_mean1} perp1:{camel_perp1} \n')

    _, _, camel_rouge_l_mean2 = pre_process.rouge(preds2, gts)
    _, camel_cossims_mean2 = pre_process.cossims(gts, preds2)
    camel_perp2 = pre_process.perplexity(preds2)

    _, _, camel_rouge_l_mean3 = pre_process.rouge(preds3, gts)
    _, camel_cossims_mean3 = pre_process.cossims(gts, preds3)
    camel_perp3 = pre_process.perplexity(preds3)

    camel_rouge_l_mean = (camel_rouge_l_mean1 + camel_rouge_l_mean2 + camel_rouge_l_mean3)/3.0
    camel_cossims_mean = (camel_cossims_mean1 + camel_cossims_mean2 + camel_cossims_mean3)/3.0
    camel_perp         = (camel_perp1 + camel_perp2 + camel_perp3)/3.0
    print(f'For {filename}_{data_set}:\n\n rouge1:{camel_rouge_l_mean1} cos1:{camel_cossims_mean1} perp1:{camel_perp1}\n rouge3:{camel_rouge_l_mean2} cos3:{camel_cossims_mean2} perp3:{camel_perp2}\n\n rouge mean:{camel_rouge_l_mean} cossims mean:{camel_cossims_mean} perp:{camel_perp} \n\n')
