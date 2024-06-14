import json, os, pickle
import numpy as np
import pandas as pd
from pathlib import Path
from jsonargparse import ArgumentParser
import numpy as np
import hdbscan
import torch
import random

from tqdm import tqdm
from rouge import Rouge
from InstructorEmbedding import INSTRUCTOR
from sentence_transformers import SentenceTransformer
from cardinal.clustering import KCenterGreedy
from instruct_features import InfoMetrics

import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

pd.options.mode.chained_assignment = None

SAMPLED_PATH = Path(__file__).parent.parent.joinpath('datasets/sampled')
DATA_PATH = Path(__file__).parent.parent.joinpath('datasets/data')

if not SAMPLED_PATH.exists():
    SAMPLED_PATH.mkdir(parents=True)
if not DATA_PATH.exists():
    DATA_PATH.mkdir(parents=True)

MODEL_NAME = 'all-mpnet-base-v2'
ROUGE_N_SAMPLES = 1000

class Sampler:
    def __init__(self, args):
        self.sample_type = args.sample_type
        self.n_instances = args.n_instances
        self.random_state = args.random_state
        self.data_set = args.data_set
        self.ftype = args.ftype
        self.local_selection_model = args.local_model
        self._set_seed(self.random_state)

    def load_dataset(self):
        saved_path = DATA_PATH.joinpath(self.data_set)
        saved_file = saved_path.joinpath('data.json')
        
        sampled_results = pd.read_csv(saved_file)

        if all(column in sampled_results.columns for column in ['instruction', 'input', 'output', 'source']):
            print("DataFrame loaded successfully with the desired columns.")
        else:
            raise NameError("Some columns are missing from the loaded DataFrame.")
        print(f"DataFrame shape: {len(sampled_results)}")
        
        return sampled_results

    def save_sampled(self, sampled: pd.DataFrame, sampling_type: str, n_instances: int):
        if self.sample_type in ['length', 'perplexity']:
            path = SAMPLED_PATH.joinpath(f'{self.data_set}/{self.sample_type}/{self.ftype}/{self.n_instances}/{self.random_state}')
        elif self.sample_type == 'selectllm':
            path = SAMPLED_PATH.joinpath(f'{self.data_set}/{self.sample_type}/{self.ftype}/{self.local_selection_model}/{self.n_instances}/{self.random_state}')
        else:
            path = SAMPLED_PATH.joinpath(f'{self.data_set}/{self.sample_type}/{self.n_instances}/{self.random_state}')
        if not path.exists():
            path.mkdir(parents=True)
                
        sampled.to_parquet(path.joinpath(f'sampled_{sampling_type}_{n_instances}.parquet.gzip'), compression='gzip')
        print(f"Save sampled dataset in {path}.")

    def random_sampling(self, n_instances: int, random_state: int):
        total = self.load_dataset()
        sampled = total.sample(
            n=n_instances,
            random_state=random_state,
        )
        assert self.n_instances == len(sampled), 'Random Sampling Length Mismatch'
        return sampled

    def CBS_Sbert(self, n_instances: int):
        '''
        Coreset Based Sampling with SBert Embeddings
        '''
        save_embed_path = DATA_PATH.joinpath(self.data_set, self.sample_type)
        save_embed_file = save_embed_path.joinpath('embeds.pkl')
        save_csv_file = save_embed_path.joinpath('samples.csv')

        if not save_embed_file.exists():
            total = self.load_dataset()
            model = SentenceTransformer(MODEL_NAME, 'cuda')
            sentences_df = total[['instruction', 'input', 'output']]
            sentences_df.reset_index(drop=True, inplace=True)

            batch_size = 500
            chunk_size = 1000  
            num_chunks = len(sentences_df) // chunk_size + 1
            sentences_df[['instruction', 'input', 'output']] = sentences_df[['instruction', 'input', 'output']].fillna('')
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size
                
                sentences_df.loc[start_idx:end_idx, 'sentences'] = (
                    sentences_df.loc[start_idx:end_idx, 'instruction']
                    .str.cat(sentences_df.loc[start_idx:end_idx, 'input'], sep=' ')
                    .str.strip()
                    .str.replace(r'\s+', ' ', regex=True)
                )
            
            sentences_df['sentences'] = self._process_in_batches(sentences_df['sentences'], batch_size, sent_tokenize)
            sentences = sentences_df['sentences'].to_list()

            embeddings = [model.encode(s) for s in sentences]

            #Save the embeddings for future use:
            save_embed_path.mkdir(parents=True, exist_ok=True)
            with open(save_embed_file, 'wb') as f:
                pickle.dump(embeddings, f)
            sentences_df.to_csv(save_csv_file, index=False)
        else:
            sentences_df = pd.read_csv(save_csv_file)
            sentences = sentences_df['sentences'].to_list()
            with open(save_embed_file, 'rb') as f:
                embeddings = pickle.load(f)

        # Perform mean pooling
        pooled_embeddings = list(map(lambda x: np.mean(x, axis=0), embeddings))

        normalized_embeddings = list(map(self._l2_normalization, pooled_embeddings))
        normalized_embeddings = pd.Series(normalized_embeddings)

        min_cluster_size = 2
        min_samples = 1
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
        clusterer.fit(normalized_embeddings.to_list())

        embeddings_label = normalized_embeddings.to_frame('embeddings')
        embeddings_label['label'] = clusterer.labels_
        embeddings_label['probability'] = clusterer.probabilities_

        centroids = embeddings_label.groupby('label')['embeddings'].apply(lambda x: np.mean(x, axis=0))

        embedding_fun = lambda x: x
        sample_size = n_instances

        raw_arrays = np.stack(embeddings_label.query(f"label != -1").embeddings.values)
        centroids_arrays = np.stack(centroids[1:].values)

        sampler = KCenterGreedy(
            embedding_fun=embedding_fun,
            batch_size=sample_size,
        )

        sampler.fit(centroids_arrays, None)
        indices = sampler.select_samples(raw_arrays)
        indices = embeddings_label.query(f"label != -1").iloc[indices].index

        sampled = sentences_df.loc[indices, ['instruction', 'input', 'output']]

        assert self.n_instances == len(sampled), 'CBS_Sbert Sampling Length Mismatch'
        return sampled

    def CBS_Instr(self, n_instances: int):
        '''
        Coreset Based Sampling with Instructor Embeddings
        '''
        save_embed_path = DATA_PATH.joinpath(self.data_set, self.sample_type)
        save_embed_file = save_embed_path.joinpath('embeds.pkl')
        save_csv_file = save_embed_path.joinpath('samples.csv')

        if not save_embed_file.exists():
            total = self.load_dataset()
            model = INSTRUCTOR('hkunlp/instructor-xl', device='cuda')
            task_description = "Answer the following question: " # temporal task description
            
            sentences_df = total[['instruction', 'input', 'output']]
            sentences_df.reset_index(drop=True, inplace=True)

            batch_size = 500
            chunk_size = 1000  
            num_chunks = len(sentences_df) // chunk_size + 1
            sentences_df[['instruction', 'input', 'output']] = sentences_df[['instruction', 'input', 'output']].fillna('')
            sentences_df['sentences'] = sentences_df['instruction'].str.strip() + ' ' + sentences_df['input'].str.strip()
            sentences_df['sentences'] = sentences_df['sentences'].str.strip()
            sentences = sentences_df['sentences'].to_list()

            print(f"corset sampling instruction+input encoding starts: ")
            embeddings = [model.encode([task_description, s], device='cuda')[1] for s in tqdm(sentences)]

            #Save the embeddings for future use:
            save_embed_path.mkdir(parents=True, exist_ok=True)
            with open(save_embed_file, 'wb') as f:
                pickle.dump(embeddings, f)
            sentences_df.to_csv(save_csv_file, index=False)
        else:
            sentences_df = pd.read_csv(save_csv_file)
            sentences = sentences_df['sentences'].to_list()
            with open(save_embed_file, 'rb') as f:
                embeddings = pickle.load(f)

        normalized_embeddings = list(map(self._l2_normalization, embeddings))
        normalized_embeddings = pd.Series(normalized_embeddings)
        min_cluster_size = 6
        min_samples = 4
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
        clusterer.fit(normalized_embeddings.to_list())

        embeddings_label = normalized_embeddings.to_frame('embeddings')
        embeddings_label['label'] = clusterer.labels_
        embeddings_label['probability'] = clusterer.probabilities_

        centroids = embeddings_label.groupby('label')['embeddings'].apply(lambda x: np.mean(x, axis=0))

        embedding_fun = lambda x: x
        sample_size = n_instances

        raw_arrays = np.stack(embeddings_label.query(f"label != -1").embeddings.values)
        centroids_arrays = np.stack(centroids[1:].values)

        sampler = KCenterGreedy(
            embedding_fun=embedding_fun,
            batch_size=sample_size,
        )

        sampler.fit(centroids_arrays, None)
        indices = sampler.select_samples(raw_arrays)
        indices = embeddings_label.query(f"label != -1").iloc[indices].index

        sampled = sentences_df.loc[indices, ['instruction', 'input', 'output']]

        assert self.n_instances == len(sampled), 'CBS_Instr Sampling Length Mismatch'
        return sampled
    
    #To do: Testing
    def length_sampling(self, n_instances: int, data_set: str):
        prompts_path = DATA_PATH.joinpath(self.data_set, 'data.json')
        inferences_path = DATA_PATH.joinpath(self.data_set, 'preds.json')
        gt_path = DATA_PATH.joinpath(self.data_set, 'gts.json')
        total = pd.read_csv(prompts_path)
        total[['instruction', 'input', 'output']] = total[['instruction', 'input', 'output']].fillna('')

        with open(inferences_path, 'r') as f:
            total_infr = json.load(f)
        with open(gt_path, 'r') as f:
            total_orig = json.load(f)

        metrics = InfoMetrics(self.data_set)
        data_length = metrics.get_length(total)

        total['lengths'] = np.array(data_length)

        if self.ftype =='short':
            total['normalized_lengths'] = (total['lengths'] - total['lengths'].min()) / (total['lengths'].max() - total['lengths'].min())
            total_filtered = total.sort_values('normalized_lengths', ascending=True)
        elif self.ftype =='long':
            total['normalized_lengths'] = (total['lengths'] - total['lengths'].min()) / (total['lengths'].max() - total['lengths'].min())
            total_filtered = total.sort_values('normalized_lengths', ascending=False)[['instruction', 'input', 'output', 'normalized_lengths']]

        total_filtered = total_filtered.reset_index(drop=True)
        sampled = total_filtered.loc[((total_filtered.index>=0) & (total_filtered.index<n_instances)), ['instruction', 'input', 'output']]

        assert self.n_instances == len(sampled), 'Length Sampling Length Mismatch'
        return sampled

    #To do: Testing
    def perplexity_sampling(self, n_instances: int, data_set: str):
        prompts_path = DATA_PATH.joinpath(self.data_set, 'data.json')
        inferences_path = DATA_PATH.joinpath(self.data_set, 'preds.json')
        gt_path = DATA_PATH.joinpath(self.data_set, 'gts.json')
        total = pd.read_csv(prompts_path)
        total[['instruction', 'input', 'output']] = total[['instruction', 'input', 'output']].fillna('')

        with open(inferences_path, 'r') as f:
            total_infr = json.load(f)
        with open(gt_path, 'r') as f:
            total_orig = json.load(f)

        metrics = InfoMetrics(self.data_set)
        data_perp = metrics.get_perp(total_infr)
        total['perp'] = np.array(data_perp)

        if self.ftype == 'low':
            total_filtered = total.sort_values('perp', ascending=True)[['instruction', 'input', 'output']]
        elif self.ftype == 'high':
            total_filtered = total.sort_values('perp', ascending=False)[['instruction', 'input', 'output']]
        elif self.ftype == 'medium':
            total_sorted = total.sort_values('perp')
            middle_index = len(total_sorted) // 2
            half_n = self.n_instances // 2 #Assuming n is even
            start_index = middle_index - half_n
            end_index = middle_index + half_n
            total_filtered = total_sorted.iloc[start_index:end_index][['instruction', 'input', 'output', 'perp']]

        total_filtered = total_filtered.reset_index(drop=True)
        sampled = total_filtered.loc[((total_filtered.index>=0) & (total_filtered.index<n_instances)), ['instruction', 'input', 'output']]

        assert self.n_instances == len(sampled), 'Perplexity Sampling Length Mismatch'
        return sampled

    def openended_sampling(self, n_instances):
        data = self.load_dataset()
        infer_path = DATA_PATH.joinpath(self.data_set, 'preds_3.json')
        with open(infer_path, 'r') as f:
            inferences = json.load(f)
        bigram_dict = {}
        sample_dict = {}
        answers_dict = {}
        for i,infers_dict in enumerate(inferences):
            prompt = list(infers_dict.keys())[0]
            # if prompt == "what are the different types of music genres ":
                # print(infers_dict)
            sent1 = infers_dict[prompt][0]
            sent2 = infers_dict[prompt][1]
            sent3 = infers_dict[prompt][2]
            all_answers = sent1 + sent2 + sent3
            #Dropping prompts which had no inferences generated
            #Diversity is not captured well all the time because of inferences like :{'what are the different types of music genres ': ['2020', '2020', '2020']}
            if len(sent1) != 0 and len(sent2) !=0 and len(sent3) != 0:
                sent1_bigrams = self._extract_bigrams(sent1)
                sent2_bigrams = self._extract_bigrams(sent2)
                sent3_bigrams = self._extract_bigrams(sent3)
                all_bigrams = sent1_bigrams + sent2_bigrams + sent3_bigrams
                unique_bigrams = set(all_bigrams)
                num_unique_bigrams = len(unique_bigrams)
                bigram_dict[prompt] = num_unique_bigrams
                answers_dict[all_answers] = num_unique_bigrams
                sample_dict[i] = num_unique_bigrams
        
        with open(Path(__file__).parent.joinpath(f'{self.data_set}_open_scores'), 'w') as f:
            json.dump(sample_dict, f)
        bigram_dict = dict(sorted(bigram_dict.items(), key=lambda x:x[1], reverse = True))
        sample_dict = dict(sorted(sample_dict.items(), key=lambda x:x[1], reverse = True))
        answers_dict = dict(sorted(answers_dict.items(), key=lambda x:x[1], reverse = True))
        sample_indices = list(sample_dict.keys())
        final_indices = sample_indices[:n_instances]
        final_indices = sorted(final_indices)
        prompts = list(bigram_dict.keys())
        scores = list(bigram_dict.values())
        # print("length of scores:", len(scores))
        answers = list(answers_dict.keys())
        final_data = data.iloc[final_indices]

        assert self.n_instances == len(final_data), 'openEnded Sampling Length Mismatch'
        return final_data

    def select_sampling(self):
        from selectllm import SelectSampler
        select_sampling = SelectSampler(self.n_instances, self.random_state, self.data_set, self.ftype, self.local_selection_model)
        train_sents = select_sampling.train_sents
        n_size = select_sampling.n_size
        train_embeddings = select_sampling.train_embeddings
        data_train = select_sampling.data_train
        n_outs = (self.n_instances//1000)

        if self.ftype == 'random':
            regions_1k = select_sampling.get_random_divide(train_embeddings, n_size, 1000)
            complex_1k = select_sampling.ours_v1(data_train, train_embeddings, local_output=n_outs, n_outputs=1000, p_method='complex', g_method=regions_1k, g_pre=True)
        elif self.ftype == 'diverse':
            regions_1k = select_sampling.get_diverse_kmeans(train_embeddings, region_size=n_size, n_regions=1000)
            complex_1k = select_sampling.selectllm_final(data_train, train_embeddings, local_output=n_outs, n_outputs=1000, g_pre=regions_1k)
        else:
            regions_1k = select_sampling.get_similar_divide(train_embeddings, n_size, 1000)
            complex_1k = select_sampling.ours_v1(data_train, train_embeddings, local_output=n_outs, n_outputs=1000, p_method='complex', g_method=regions_1k, g_pre=True)
            
        complex_1k_fin = np.vstack(complex_1k[0])

        indices = complex_1k_fin.flatten()
        data_file = DATA_PATH.joinpath(self.data_set, 'data.json')
        data = pd.read_csv(data_file)
        selected_data = data.iloc[indices]

        assert self.n_instances == len(selected_data), 'SelectLLM Sampling Length Mismatch'
        return selected_data
    
    def coreset_sampling(self):
        from selectllm import Coreset_Greedy, SelectSampler
        select_sampling = SelectSampler(self.n_instances, self.random_state, self.data_set, self.ftype)
        train_sents = select_sampling.train_sents
        train_embeddings = select_sampling.train_embeddings
        data_train = select_sampling.data_train

        # init_indices = data_train.sample(n=100).index
        init_indices = np.random.choice(len(train_embeddings), size=100, replace=False).tolist()

        coreset = Coreset_Greedy(train_embeddings, self.random_state)
        coreset_indices = coreset.sample(already_selected=init_indices, sample_size=self.n_instances)
        # complex_1k_fin = np.vstack(complex_1k[0])

        # indices = complex_1k_fin.flatten()
        data_file = DATA_PATH.joinpath(self.data_set, 'data.json')
        data = pd.read_csv(data_file)
        selected_data = data.iloc[coreset_indices[0]]

        assert self.n_instances == len(selected_data), 'Coreset Sampling Length Mismatch'
        return selected_data
    
    def diversity_sampling(self, n_instances: int):
        data_path = DATA_PATH.joinpath(self.data_set, 'data.json')
        rouge_folder_path = DATA_PATH.joinpath(self.data_set, 'rouge')
        rouge_file_path = DATA_PATH.joinpath(self.data_set, 'rouge', 'rouge.json')
        
        with open(data_path, 'r') as f:
            data = pd.read_csv(data_path)
        data['prompt'] = data['instruction'] + ' ' + data['input']
        data['prompt'] = data['prompt'].str.strip()
        
        def calculate_rouge(sentences1, sentences2):
            rouge = Rouge()
            rouge_scores = []
            rougel_scores = []
            for p, g in zip(sentences1, sentences2):
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

            return rougel_scores
        
        if not os.path.exists(rouge_file_path):
            result = []
            for idx, row in tqdm(data.iterrows(), total=data.shape[0]):
                data_rm_itself = data.drop(labels=[idx])
                data_sampled = data_rm_itself.sample(n=ROUGE_N_SAMPLES)
                prompt_repeat = [row['prompt'] for _ in range(ROUGE_N_SAMPLES)]
                sample_rouge_max = max(calculate_rouge(prompt_repeat, data_sampled['prompt']))
                result.append({'instruction': row['instruction'], 'input': row['input'], 'output': row['output'], 'rouge_max': sample_rouge_max})
            
            sorted_result = sorted(result, key=lambda x: x['rouge_max'])
            
            if not os.path.exists(rouge_folder_path):
                os.mkdir(rouge_folder_path)
            with open(rouge_file_path, 'w') as f:
                json.dump(sorted_result, f)
        else:
            with open(rouge_file_path, 'r') as f:
                sorted_result = json.load(f)
        
        sampled = pd.DataFrame(sorted_result)
            
        return sampled.loc[:(n_instances-1), ['instruction', 'input', 'output']]
    
    def _extract_bigrams(self, sentence):
        words = sentence.split()
        return [(words[i], words[i+1]) for i in range(len(words) - 1)]

    def _process_in_batches(self, df, batch_size, func):
        results = []
        
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            result = batch.apply(func)
            results.extend(result)
        
        return results

    def _l2_normalization(self, array):
        norm = np.sqrt(np.sum(array**2))
        return array / norm
    
    def _set_seed(self, random_state):
        deterministic = True
        random.seed(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        torch.cuda.manual_seed_all(random_state)

    def __call__(self):
        assert self.sample_type in ['random', 'coreset', 'length', 'perplexity', 'rouge', 'oe', 'selectllm', 'cbs_sbert', 'cbs_instr']
        
        if self.sample_type == 'random':
            sampled = self.random_sampling(self.n_instances, random_state=self.random_state)
        elif self.sample_type == 'cbs_sbert':
            sampled = self.CBS_Sbert(self.n_instances)
        elif self.sample_type == 'cbs_instr':
            sampled = self.CBS_Instr(self.n_instances)
        elif self.sample_type == 'rouge':
            sampled = self.diversity_sampling(self.n_instances)
        elif self.sample_type == 'oe':
            sampled = self.openended_sampling(self.n_instances)
        elif self.sample_type == 'selectllm':
            sampled = self.select_sampling()
        elif self.sample_type == 'coreset':
            sampled = self.coreset_sampling()
        elif self.sample_type == 'length':
            sampled = self.length_sampling(self.n_instances, self.data_set)
        elif self.sample_type == 'perplexity':
            sampled = self.perplexity_sampling(self.n_instances, self.data_set)
        
        self.save_sampled(sampled, self.sample_type, self.n_instances)

def get_args():
    parser = ArgumentParser(description="Sampler for active learning.")
    parser.add_argument('-s','--sample_type', default='coreset', choices=['random', 'coreset', 'length', 'perplexity', 'rouge','oe','selectllm', 'cbs_sbert', 'cbs_instr'], help="Type of sampling algorithm to use.")
    parser.add_argument('-n','--n_instances', type=int, default=1000, help="Number of instances to sample.")
    parser.add_argument('-d','--data_set', default='all', help="Dataset to use for sampling. Options:[cleaned_alpaca, dolly]")
    parser.add_argument('-r','--random_state', type=int, default=2023, help="Random state for reproducibility.")
    parser.add_argument('-f','--ftype', type=str, help="Length:[long, short], Perplexity:[low, high, medium] SelectLLM:[similar, diverse, random]")
    parser.add_argument('-l','--local_model', type=str, help="Model for SelectLLM. Options: [gpt3.5, mixtral, haiku, gpt4o]")
    
    return parser.parse_args()

if __name__ == '__main__':

    args = get_args()
    sampler = Sampler(args)
    sampler()

'''To do:
1) Add Simple Testing functions for each sampling technique
'''