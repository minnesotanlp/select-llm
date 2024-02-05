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

class MetricLength():
    def __init__(self):
        # self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.model_path = "/corpora/models/LLaMA/converted/7B" 
        # self.model_path = '/corpora/llama_2_hf/llama2/7bweights/'

    def get_length(self, data):

        file_path = '/corpora/InstructTune/cloned_ait_repo/active-instruction-tuning/datasets/scores/trained_inference_30%_length.json'

        if not os.path.exists(file_path):
            input=data['input'].tolist()
            output=data['output'].tolist()
            instruction=data['instruction'].tolist()

            concat_input = [x + y for x,y in zip(input, instruction)]
            
            metric_length = np.array([len(inp) for inp in concat_input], dtype=np.float64)
            metric_length /= np.max(metric_length)

            metric_length = metric_length.tolist()
            with open(file_path, 'w') as f:
                json.dump(metric_length, f)
        else:
            with open(file_path, 'r') as f:
                metric_length = json.load(f)

        return metric_length

    def get_rouge(self, pred_sentences, gt_sentences):

        file_path = '/corpora/InstructTune/cloned_ait_repo/active-instruction-tuning/datasets/scores/trained_inference_30%_rouge.json'

        if not os.path.exists(file_path):
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
            with open(file_path, 'w') as f:
                json.dump(rougel_scores, f)
        else:
            with open(file_path, 'r') as f:
                rougel_scores = json.load(f)

        return rougel_scores

    # def get_perp(self, llama_sentences):
    #     file_path = './trained_inference_30%_perps.json'
    #     if not os.path.exists(file_path):
    #         tokenizer = AutoTokenizer.from_pretrained(self.model_path)
    #         model = AutoModelForCausalLM.from_pretrained(self.model_path)
    #         perplexities = []
    #         i = 0
    #         #Comment this out later:
    #         llama_sentences = llama_sentences[12000:]
    #         ###
    #         for sentence in llama_sentences:
    #             tok_embed = tokenizer(sentence, return_tensors='pt') 
    #             # del tok_embed['token_type_ids']
    #             with torch.no_grad():
    #                 outputs = model(**tok_embed, labels=tok_embed['input_ids'])
    #             perplexity = torch.exp(outputs.loss).item()
    #             perplexities.append(perplexity)
    #             # print(perplexity)
    #             # i += 1
    #             # if i == 10:
    #             #     break
    #         with open(file_path, 'w') as f:
    #             json.dump(perplexities, f)
    #     else:
    #         with open(file_path, 'r') as f:
    #             perplexities = json.load(f)

    #     return perplexities

    #Added Batch processing to perp calculation
    def get_perp(self, llama_sentences):
        file_path = '/corpora/InstructTune/cloned_ait_repo/active-instruction-tuning/datasets/scores/trained_inference_30%_perps.json'
        print(len(llama_sentences))
        if not os.path.exists(file_path):
            #Loading Llama 2 model (converted huggingface weights)
            # tokenizer = LlamaTokenizer.from_pretrained(self.model_path)
            # if tokenizer.pad_token is None:
            #     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            # model = LlamaForCausalLM.from_pretrained(self.model_path)
            # model.resize_token_embeddings(len(tokenizer))
            # model.config.vocab_size = len(tokenizer)
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model = AutoModelForCausalLM.from_pretrained(self.model_path)
            new_vocab_size = (len(tokenizer) + 7) // 8 * 8
            model.resize_token_embeddings(new_vocab_size)
            model.config.vocab_size = new_vocab_size

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if torch.cuda.is_available():
                model = model.to('cuda')

            BATCH_SIZE = 16
            data_loader = DataLoader(llama_sentences, batch_size=BATCH_SIZE, shuffle=False)
            perplexities = []
            for batch in data_loader:
                tok_embed = tokenizer(batch, return_tensors='pt', padding=True, truncation=True) 
                if torch.cuda.is_available():
                    tok_embed = {key: val.to('cuda') for key, val in tok_embed.items()}

                max_token_id = max([max(seq) for seq in tok_embed['input_ids']])
                if max_token_id >= model.config.vocab_size:
                    raise ValueError(f"Token ID {max_token_id} is out of range (0, {model.config.vocab_size})")

                with torch.no_grad():
                    outputs = model(**tok_embed, labels=tok_embed['input_ids'])

                # Get the logits and labels
                logits = outputs.logits
                labels = tok_embed['input_ids']

                # Calculate the loss for each sentence in the batch
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                active_loss = labels.view(-1) != -100
                active_logits = logits.view(-1, model.config.vocab_size)
                active_labels = labels.view(-1)
                losses = loss_fct(active_logits, active_labels)
                losses = losses[active_loss]

                # Reshape the losses to have the same shape as labels
                losses = losses.view(labels.size())

                # Calculate the perplexity for each sentence
                sentence_losses = losses.sum(dim=1)
                sentence_lengths = (labels != -100).sum(dim=1)
                sentence_perplexities = torch.exp(sentence_losses / sentence_lengths)

                # Convert to numpy and extend the list of perplexities
                batch_perplexities = sentence_perplexities.cpu().numpy()
                # print(batch_perplexities)
                perplexities.extend(batch_perplexities)
            perplexities = [float(x) for x in perplexities]
            with open(file_path, 'w') as f:
                json.dump(perplexities, f)
        else:
            with open(file_path, 'r') as f:
                perplexities = json.load(f)

        return perplexities
     
    def meta_vector(self, perp_scores, length, rouge):
        all_measurements = []
        all_measurements.append(torch.tensor(perp_scores).unsqueeze(0))
        all_measurements.append(torch.tensor(length).unsqueeze(0))
        all_measurements.append(torch.tensor(rouge).unsqueeze(0))
        return torch.cat(all_measurements, dim=0).t()

if __name__=='__main__':
    metrics = MetricLength()
    length = metrics.get_length()
    # print(lamini_length)