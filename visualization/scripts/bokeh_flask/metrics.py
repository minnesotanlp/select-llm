import argparse
import json
import random
from sentence_transformers import SentenceTransformer, util
import torch
from rouge import Rouge
from transformers import AutoTokenizer, AutoModelForCausalLM 
from transformers import GenerationConfig 

#To Do:
#Structure it such that this file gets called in the bokeh flask file (?)
# Put the alpaca and lamini dataset sampling  files in the same folder as well. 

class Preprocess():
    def __init__(self):
        self.meta_data_file = args.meta_data_file
        self.prompt_file = args.prompt_file
        self.llama_output_file = args.llama_output_file
        self.llama_cleaned_op_file = args.llama_cleaned_op_file 
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.model_path = "/corpora/models/LLaMA/converted/7B" 
    
    def alpaca_process(self):
        '''
        Preprocess and get grount truth answers, as well as predicted answrs from the alpaca file format
        '''
        gt_sentences    = [] 
        llama_sentences = []
        with open(self.meta_data_file, 'r') as meta_file, open(self.prompt_file, 'r') as prompt_file,open(self.llama_output_file, 'r') as llama_output_file, open(self.llama_cleaned_op_file, 'w') as cleaned_file:
            for gt_line, prompt_line, llama_line in zip(meta_file, prompt_file, llama_output_file):
                
                #Uncomment below lines for the usual use case
                # prompt_data = json.loads(prompt_line)
                # llama_data = json.loads(llama_line)
                # gt_sentences.append(json.loads(gt_line['example']['output']).strip())
                # prompt_output = prompt_data['prompt']
                # llama_output = llama_data["generation"]
                # op_sentence = llama_output.replace(prompt_output, "").strip()
                # llama_sentences.append(op_sentence)
                # # json.dump([op_sentence], cleaned_file)
                # # cleaned_file.write("\n")

                #Getting cossims b/w prompts and predictions
                prompt_data = json.loads(prompt_line)
                llama_data = json.loads(llama_line)
                prompt_output = prompt_data['prompt']
                llama_output = llama_data[0]
                op_sentence = llama_output.replace(prompt_output, "").strip()
                llama_sentences.append(op_sentence)
                gt_sentences.append(prompt_output)

        return gt_sentences, llama_sentences

    def lamini_process(self):
        '''
        Preprocess and get grount truth answers, as well as predicted answrs from the lamini file format
        '''
        gt_sentences    = [] 
        llama_sentences = []
        with open(self.meta_data_file, 'r') as meta_file, open(self.prompt_file, 'r') as prompt_file,open(self.llama_output_file, 'r') as llama_output_file, open(self.llama_cleaned_op_file, 'w') as cleaned_file:
            for gt_line, prompt_line, lamini_line in zip(meta_file, prompt_file, llama_output_file):

                # gt_data = json.loads(gt_line['example']['output'])
                # prompt_data = json.loads(prompt_line)
                # lamini_data = json.loads(lamini_line)
                # gt_sentences.append(gt_data['response'].strip())
                # prompt_output = prompt_data['prompt']
                # lamini_output = lamini_data["generation"]
                # op_sentence = lamini_output.replace(prompt_output, "").strip()
                # llama_sentences.append(op_sentence)
                # # json.dump([op_sentence], cleaned_file) 
                # # cleaned_file.write("\n")

                #Getting cossims b/w prompts and predictions
                prompt_data = json.loads(prompt_line)
                lamini_data = json.loads(lamini_line)
                prompt_output = prompt_data['prompt']
                lamini_output = lamini_data[0]
                op_sentence = lamini_output.replace(prompt_output, "").strip()
                llama_sentences.append(op_sentence)
                gt_sentences.append(prompt_output)

        return gt_sentences, llama_sentences

    def Perplexity(self, llama_sentences):
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModelForCausalLM.from_pretrained(self.model_path)
        perplexities = []
        for sentence in llama_sentences:
            tok_embed = tokenizer(sentence, return_tensors='pt') 
            del tok_embed['token_type_ids']
            with torch.no_grad():
                outputs = model(**tok_embed, labels=tok_embed['input_ids'])
            perplexity = torch.exp(outputs.loss).item()
            perplexities.append(perplexity)
        # with open('./alpaca_perplex_scores.json', 'w') as f:
        #     json.dump(perplexities, f)
        return perplexities

    def cossims(self, gt_sentences, llama_sentences):
        embeddings_gt = self.model.encode(gt_sentences)
        embeddings_llama = self.model.encode(llama_sentences)

        cos_sim = util.cos_sim(embeddings_gt, embeddings_llama)

        # Add all pairs to a list with their cosine similarity score
        all_sentence_combinations = []

        for i in range(len(cos_sim)):
            all_sentence_combinations.append(round(cos_sim[i][i].item(), 4))

        # output_file_path = './alpaca_cossim_scores.json'
        # with open(output_file_path, 'w') as f:
        #     json.dump(all_sentence_combinations, f)
        return all_sentence_combinations

    def rouge(self, llama_sentences, gt_sentences):
        rouge = Rouge()
        rouge_scores = []
        rougel_scores = []
        for p, g in zip(llama_sentences, gt_sentences):
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

        # with open('./alpaca_rouge_scores.json', 'w') as f:
        #     json.dump(rouge_scores, f)

        # with open('./alpaca_rougel_scores.json', 'w') as f:
        #     json.dump(rougel_scores, f)

        return rouge_scores, rougel_scores



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process a jsonl file.')
    parser.add_argument('meta_data_file', type=str, help='The path to the ground truth jsonl file.')
    parser.add_argument('prompt_file', type=str, help='The path to the prompts file.')
    parser.add_argument('llama_output_file', type=str, help='The path to the Llama model output jsonl file.')
    parser.add_argument('llama_cleaned_op_file', type=str, help='The path to the new cleaned file that will be created')
    args = parser.parse_args()

    pre_process = Preprocess()

    # alpaca_gt_sentences, alpaca_llama_sentences = pre_process.alpaca_process()
    lamini_gt_sentences, lamini_llama_sentences = pre_process.lamini_process()

    # alpaca_perplexities = pre_process.Perplexity(alpaca_llama_sentences)
    #lamini_perplexities = pre_process.Perplexity(lamini_gt_sentences, lamini_llama_sentences)

    # #Customized code here
    # alprompts = args.prompt_file
    # alpaca_prompts = []
    # for line in alprompts:
    #     for key,value in line.items():
    #         alpaca_prompts.append(value)

    lamini_prompt_op_cossims = pre_process.cossims(lamini_gt_sentences, lamini_llama_sentences)
    with open('../../datasets/lamini/lamini_prompt_op_cossims.jsonl', 'w') as f:
        json.dump(lamini_prompt_op_cossims, f)





