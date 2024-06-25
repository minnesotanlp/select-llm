import ast
import time
import math
import random
import argparse
import torch
import openai
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from dotenv import dotenv_values
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sklearn.metrics import pairwise_distances, pairwise
from sklearn.cluster import KMeans
# from llama_cpp import Llama
# import ollama
import re

openai_config = dotenv_values(Path(__file__).parent.parent.joinpath(".env"))
CLIENT = openai.OpenAI(api_key=openai_config["OPENAI_API_KEY"])

class Coreset_Greedy:
    def __init__(self, all_pts, random_state):
        self.all_pts = np.array(all_pts)
        self.dset_size = len(all_pts)
        self.min_distances = None
        self.already_selected = []
        self.random_state = random_state

        # reshape
        feature_len = self.all_pts.shape[1]

        self._set_seed(self.random_state)
        
        # self.first_time = True

    def _set_seed(self, random_state):
        deterministic = True
        random.seed(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        torch.cuda.manual_seed_all(random_state)

    def update_dist(self, centers, only_new=True, reset_dist=False):
        if reset_dist:
            self.min_distances = None
        if only_new:
            centers = [p for p in centers if p not in self.already_selected]
        
        if centers is not None:
            x = self.all_pts[centers] # pick only centers
            dist = pairwise_distances(self.all_pts, x, metric='euclidean')
            #dist = 1 -1 * pairwise.cosine_similarity(self.all_pts, x)

            if self.min_distances is None:
                self.min_distances = np.min(dist, axis=1).reshape(-1,1)
            else:
                self.min_distances = np.minimum(self.min_distances, dist)
    
    def sample(self, already_selected, sample_size):

        # initially updating the distances
        self.update_dist(already_selected, only_new=False, reset_dist=True)
        self.already_selected = already_selected

        new_batch = []
        for _ in range(sample_size):
            if self.already_selected == []:
                ind = np.random.choice(np.arange(self.dset_size))
            else:
                ind = np.argmax(self.min_distances)
            
            assert ind not in already_selected
            self.update_dist([ind],only_new=True, reset_dist=False)
            new_batch.append(ind)
        
        max_distance = max(self.min_distances)

        return new_batch, max_distance

class SelectSampler:
    def __init__(self, n_instances, random_state, data_set, ftype, local_selection_model):
        self.n_instances = n_instances
        self.random_state = random_state
        self.data_set = data_set
        self.ftype = ftype
        self.local_selection_model = local_selection_model
        self._set_seed(self.random_state)

        if self.data_set == 'dolly':
            self.n_size = 14
        else:
            self.n_size = 51
        
        self._set_embed()
        if self.local_selection_model == 'mixtral':
            self._load_mixtral_model()
        # elif self.local_selection_model == 'llama':
        #     self._load_llama_gguf_model()
    
    def _set_seed(self, random_state):
        deterministic = True
        random.seed(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        torch.cuda.manual_seed_all(random_state)

    def _set_embed(self):
        data_path = Path(__file__).parent.parent.joinpath(f'datasets/data/{self.data_set}/data.json')
        self.data_train = pd.read_csv(data_path)
        self.train_sents = []

        for i in range(len(self.data_train)):
            sent = f"### Instruction: {self.data_train['instruction'][i]}"
            if type(self.data_train['input'][i]) is str:
                sent += f"\n\n### Input: {self.data_train['input'][i]}"
            sent += "\n\n### Response:"
            self.train_sents.append(sent)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        self.train_embeddings = np.zeros((len(self.train_sents), 768))
        self.train_embeddings = model.encode(self.train_sents)
    
    def _load_mixtral_model(self):
        print(f'Loading {self.local_selection_model} model')
        self.mixtral_model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        self.mixtral_tokenizer = AutoTokenizer.from_pretrained(self.mixtral_model_id)
        self.mixtral_model = AutoModelForCausalLM.from_pretrained(self.mixtral_model_id, load_in_8bit=True, device_map="auto")
        print('model loaded\n')
    
    def _load_llama_model(self):
        print(f'Loading {self.local_selection_model} model')
        self.llama_model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
        self.llama_pipeline = pipeline(
            "text-generation",
            model=self.llama_model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        self.llama_tokenizer = self.llama_pipeline.tokenizer
        print('model loaded\n')
    
    def _load_llama_gguf_model(self):
        print(f'Loading {self.local_selection_model} model')
        self.llama_model = Llama(
                model_path="/corpora/InstructTune/cloned_ait/new_repo/models/llama.cpp/models/Meta-Llama-3-70B-Instruct.Q2_K.gguf",
                n_gpu_layers=-1, # Uncomment to use GPU acceleration
                seed=self.random_state, # Uncomment to set a specific seed
                n_ctx=4096, # Uncomment to increase the context window
                    )
        print('model loaded\n')

    def prompt_local_select(self, dataset_train, indices, num):
        text = ""
        n_samples = len(indices)

        text += f"The following are {n_samples} unannotated examples of instructions that describe a task, each indicated by a number identifier [].\n\n"    
        
        for i in range(len(indices)):
            text += f"[{i+1}]\n\n"
            text += f"### Instruction:\n {dataset_train['instruction'][indices[i]]}\n\n"
            if type(dataset_train['input'][indices[i]]) == str:
                text += f"### Input:\n {dataset_train['input'][indices[i]]}\n\n"
        
        example1 = str(list(1 + np.arange(num))) 
        example2 = str(list(1 + num + np.arange(num)))
        
        text += f"Examine the provided list of {n_samples} instructions, each uniquely identified by a number in brackets []. " 
        text += f"Your task is to select {num} instructions that will be annotated by human annotators for model fine-tuning. "
        text += f"Look for instructions that are clear and relevant, exhibit a high level of complexity and detail, represent a diverse range of scenarios and contexts, offer significant instructional value and potential learning gain, and present unique challenges and specificity. "
        text += f"These selected instructions should ideally be the most beneficial for model fine-tuning after being annotated by human annotators. Present your selections using the format []. e.g., {example1} or {example2}.\n\n"
        text += f"The most impactful {num} instructions (only identifiers) are:"
        
        return text
    
    def call_api_sllm(self, query):
        # model = "gpt-3.5-turbo-1106"
        model = "gpt-3.5-turbo-0125"
        waiting_time = 0.5
        
        response = None
        while response is None:
            try:
                messages = [
                        {"role": "system", "content": query},
                ]
                
                    # ChatGPT API 호출하기
                response = CLIENT.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=256
                )
            except:
                time.sleep(waiting_time)
                if waiting_time < 5:
                    waiting_time += 0.5
                else:
                    break
        if response is not None:
            try:
                answer = response.choices[0].message.content
            except:
                answer = 'N/A'
            usage_data = dict(response.usage) if response.usage else {}
            n_input_tokens = usage_data.get('prompt_tokens', 0)
            n_output_tokens = usage_data.get('completion_tokens', 0)
        else:
            answer = 'N/A'
            n_input_tokens = 0
            n_output_tokens = 0
            
        return answer, n_input_tokens, n_output_tokens
    
    def call_mx_sllm(self, query):
        answer = None
        messages = [
                {"role": "user", "content": query},
        ]
        
        input_ids = self.mixtral_tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
        input_text = self.mixtral_tokenizer.decode(input_ids[0], skip_special_tokens=True)
        outputs = self.mixtral_model.generate(input_ids, max_new_tokens=40)
        answer = self.mixtral_tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = answer[len(input_text):].strip()
        answer = re.search(r'\[\d+\]', answer).group()
        n_input_tokens = input_ids.shape[1]
        n_output_tokens = outputs.shape[1]
        return answer, n_input_tokens, n_output_tokens
    
    def call_llama_sllm(self, query):
        messages = [
            {"role": "system", "content": "You are an intelligent chatbot that will carefully answer the user query"},
            {"role": "user", "content": query},
        ]

        terminators = [
            self.llama_tokenizer.eos_token_id,
            self.llama_tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.llama_pipeline(
            messages,
            max_new_tokens=10,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

        answer = outputs[0]["generated_text"][-1]['content']

        n_input_tokens = len(self.llama_tokenizer(messages[1]['content'], return_tensors='pt').input_ids[0])
        n_output_tokens = len(self.llama_tokenizer(answer, return_tensors='pt').input_ids[0])

        return answer, n_input_tokens, n_output_tokens
    
    def call_llama_gguf_sllm(self, query):
        output = self.llama_model(
                    query, # Prompt
                    max_tokens=10, # Generate up to 32 tokens, set to None to generate up to the end of the context window
                    stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
                    echo=False # Echo the prompt back in the output
                ) # Generate a completion, can also call create_completion
        answer = output['choices'][0]['text']
        n_input_tokens = 0
        n_output_tokens = 0

        return answer, n_input_tokens, n_output_tokens

    def call_ollama_sllm(self, query, num):
        while True:
            try:
                response = ollama.chat(model='llama3:70b-instruct-q8_0', messages=[
                    {
                        'role': 'user',
                        'content': query,
                    }]
                    , options={
                        'num_predict': 20,
                        'seed': self.random_state
                    })

                waiting_time = 0.5
                answer = None
                while answer is None:
                    try:
                        answer = response['message']['content']
                    except:
                        time.sleep(waiting_time)
                        if waiting_time < 5:
                            waiting_time += 0.5
                        else:
                            break

                if answer is not None:
                    if num == 1:
                        answer = re.search(r'\[\d+\]', answer).group()
                    else:
                        answer = [int(num) for num in re.search(r'\[(\d+(?:\s*,\s*\d+)*)\]', answer).group(1).split(',')]
                    break

            except Exception as e:
                print(f'Error occurred: {e}. Retrying...')

        n_input_tokens = 0
        n_output_tokens = 0

        return answer, n_input_tokens, n_output_tokens
    
    def get_diverse_kmeans(self, train_embeddings, region_size, n_regions):
        n_samples = len(train_embeddings)
        gap = region_size * n_regions - n_samples

        expand_indices = list(np.arange(n_samples)) + list(np.arange(n_samples)[:gap])
        train_embeddings_expand = train_embeddings[np.array(expand_indices).astype(np.int64)]

        kmeans = KMeans(n_clusters=region_size, random_state=0, n_init="auto").fit(train_embeddings)
        dist_to_centers = kmeans.transform(train_embeddings_expand) # N x K 

        labels = -1 * np.ones(region_size * n_regions).astype(np.int64)
        for j in range(n_regions):
            for i in range(region_size):
                selected_indices = np.argsort(dist_to_centers[:, i])[0]
                labels[selected_indices] = i
                dist_to_centers[selected_indices] = 20000 
        if (labels == -1).sum() > 0:
            raise ValueError

        res = []
        for k in range(region_size):
            selected_labels = (labels == k).nonzero()[0]
            for p in range(n_regions):
                if k == 0:
                    res.append([selected_labels[p] % n_samples])
                else:
                    res[p].append(selected_labels[p] % n_samples)
        return res
    
    def selectllm_final(self, data_train, train_embeddings, local_output, n_outputs, g_pre=False):
        n_samples = len(train_embeddings)
        region_size = math.ceil(n_samples / n_outputs)
        input_tokens, output_tokens = 0, 0 
        
        if g_pre:
            global_regions = np.array(g_pre).astype(np.int64)
        else:
            global_regions = self.get_diverse_kmeans(train_embeddings, region_size, n_outputs)
            
        res = []
        res_bef = []
        new_res = []
        
        for i in tqdm(range(len(global_regions))):
            query = self.prompt_local_select(data_train, global_regions[i], local_output)
            
            if self.local_selection_model == 'gpt3.5':
                answer, input_token, output_token = self.call_api_sllm(query)
            elif self.local_selection_model == 'mixtral':
                answer, input_token, output_token = self.call_mx_sllm(query)
            elif self.local_selection_model == 'llama':
                answer, input_token, output_token = self.call_ollama_sllm(query, local_output)

            try:
                answer_aft = list(np.array(ast.literal_eval(answer)) - 1)
                if len(answer_aft) > local_output:
                    answer_aft = list(np.array(answer_aft)[:local_output])
            except:
                answer_aft = np.arange(local_output) + 1  
            res_bef.append(list(global_regions[i]))
            res.append(list(global_regions[i][answer_aft]))
            new_res.append(( list(global_regions[i][answer_aft]), list(global_regions[i]) ))

            input_tokens += input_token
            output_tokens += output_token

        print("Total input token: {}".format(input_tokens))
        print("Total output token: {}".format(output_tokens))
        
        return res, res_bef
    
    #Chain of Thought Reasoning Generation for SelectLLM Sampling 

    def prompt_rerank_subset_complex_fewshot_cot(self, dataset_train, indices, num, preds):
        text = ""
        n_samples = len(indices)

        text += f"The following are {n_samples} unannotated examples of instructions that describe a task, each indicated by a number identifier [].\n\n"    
        
        for i in range(len(indices)):
            text += f"[{i+1}]\n\n"
            text += f"### Instruction:\n {dataset_train['instruction'][indices[i]]}\n\n"
            if type(dataset_train['input'][indices[i]]) == str:
                text += f"### Input:\n {dataset_train['input'][indices[i]]}\n\n"
        
        example1 = str(list(1 + np.arange(num))) 
        example2 = str(list(1 + num + np.arange(num)))
        
        text += f"Examine the provided list of {n_samples} instructions, each uniquely identified by a number in brackets []. " 
        text += f"Your task is to select {num} instructions that will be annotated by human annotators for model fine-tuning. "
        text += f"Look for instructions that are clear and relevant, exhibit a high level of complexity and detail, represent a diverse range of scenarios and contexts, offer significant instructional value and potential learning gain, and present unique challenges and specificity. "
        text += f"These selected instructions should ideally be the most beneficial for model fine-tuning after being annotated by human annotators. Present your selections using the format []. e.g., {example1} or {example2}.\n\n"
        
        text += f"The most impactful {num} instructions (only identifiers) are: [{np.where(indices == preds)[0][0] + 1}]\n"
        text += f"Explain why it was chosen, focusing on how it meets the above criteria and its potential contribution to model fine-tuning."
        text += f"Rationale for selection:"
        
        return text
    
    def selectllm_cot(self, data_train, train_embeddings, local_output, n_outputs, preds, g_pre=False):
        n_samples = len(train_embeddings)
        region_size = math.ceil(n_samples / n_outputs)
        input_tokens, output_tokens = 0, 0 
        
        if g_pre:
            global_regions = np.array(g_pre).astype(np.int64)
        else:
            global_regions = self.get_diverse_kmeans(train_embeddings, region_size, n_outputs)
        
        res = []
        for i in tqdm(range(len(global_regions))):
            global_regions[i] = np.array(global_regions[i]).astype(np.int64)
            query = self.prompt_rerank_subset_complex_fewshot_cot(data_train, global_regions[i], local_output, preds[i])
            answer, input_token, output_token = self.call_api(query)
            
            try:
                res.append(answer)
            except:
                res.append('N/A')
            input_tokens += input_token
            output_tokens += output_token

        print("Total input token: {}".format(input_tokens))
        print("Total output token: {}".format(output_tokens))
        
        return res

    #List-Based Sorting for Proof of Concept Analysis

    def prompt_rerank_list(self, dataset_train, indices, k):
        #Proof of Concept Prompting to test capabilities of LLMs for Instruction Selection
        text = ""
        num = k 
        
        text = "This is RankGPT, an intelligent assistant that can rank instructions based on their impactfulness and informativeness for model fine-tuning, when labeled by humans, like active learning.\n\n"
        text += f"The following are {num} examples of instructions that describe a task, each indicated by a number identifier [].\n\n"
        
        for i in range(len(indices)):
            text += f"[{i+1}]"
            text += f"\n\n### Instruction:\n {dataset_train['instruction'][indices[i]]}\n\n"
            if type(dataset_train['input'][indices[i]]) == str:
                text += f"### Input:\n {dataset_train['input'][indices[i]]}\n\n"
            else:
                text += f"### Input:\n N\A\n\n"
        
        text += f"I will rank the {num} instructions above based on their impactfulness and informativeness for model fine-tuning when labeled by humans, like active learning. The examples will be listed in descending order using identifiers, and the most impactful examples should be listed first, and the output format should be [] > [] > etc, e.g., [1] > [2] > etc.\n\n"
        text += f"The ranking results of the {num} examples (only identifiers) is:"
        
        return text
    
    def selectllm_list(self, data_train, train_embeddings, local_output, n_outputs, g_pre=False):
        n_samples = len(train_embeddings)
        region_size = math.ceil(n_samples / n_outputs)
        input_tokens, output_tokens = 0, 0 
        
        if g_pre:
            global_regions = np.array(g_pre).astype(np.int64)
        else:
            global_regions = self.get_diverse_kmeans(train_embeddings, region_size, n_outputs)

        res = []
        res_bef = []
        for i in tqdm(range(len(global_regions))):
            global_regions[i] = np.array(global_regions[i]).astype(np.int64)
            query = self.prompt_rerank_list(data_train, global_regions[i], len(global_regions[i]))
            answer, input_token, output_token = self.call_api(query)
            try:
                answer_aft = np.array(self._get_list(answer)).astype(np.int64) - 1
            except:
                answer_aft = np.arange(len(global_regions[i]))
            
            res_bef.append(global_regions[i])
            try:
                res.append(global_regions[i][answer_aft])
            except:
                res.append(global_regions[i])
            input_tokens += input_token
            output_tokens += output_token

        print("Total input token: {}".format(input_tokens))
        print("Total output token: {}".format(output_tokens))
        
        return res, res_bef

    def _cleaning(self, txts):
        while txts[0] in [' ', '[']:
            txts = txts[1:]
        while txts[-1] in [' ', ']']:
            txts = txts[:-1]
        return int(txts)
    
    def _get_list(self, outputs):
        splitted = outputs.split('>')
        res = []
        
        for split in splitted:
            res.append(self._cleaning(split))
        return res

    def __call__(self):
        n_outs = self.n_instances//1000
        
        if self.ftype == 'random':
            regions_1k = self.get_random_divide(self.train_embeddings, self.n_size, 1000)
        elif self.ftype == 'diverse':
            regions_1k = self.get_diverse_divide(self.train_embeddings, self.n_size, 1000)
        else:
            regions_1k = self.get_similar_divide(self.train_embeddings, self.n_size, 1000)
        
        complex_1k = self.ours_v1(self.data_train, self.train_embeddings, local_output=n_outs, n_outputs=1000, p_method='complex', g_method=regions_1k, g_pre=True)
        complex_1k_fin = np.vstack(complex_1k[0])
        
def get_args():
    parser = argparse.ArgumentParser(description="SelectLLM Sampling")
    parser.add_argument('--n_instances', type=int, default=1000, help='Number of instances to sample.')
    parser.add_argument('--data_set', default='all', help='Dataset to use for sampling.')
    parser.add_argument('--random_state', type=int, default=2023, help='Random state for reproducibility.')
    parser.add_argument('--ftype', type=str, help='Type of Prompt Complexity: Simple or complex')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    sampler = SelectSampler(args)
    sampler()


