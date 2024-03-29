# SelectLLM

## Overview

Enable instruction-tuning in Large Language Models with the SelectLLM method. ([Paper](https://arxiv.org/abs/2401.16553))

## Setup

Clone the repository.

```bash
git clone https://github.com/minnesotanlp/select-llm.git
cd select-llm
```

Create a virtual environment. We recommend using Python version 3.10.6.

```bash
python -m venv select-llm-venv
source select-llm-venv/bin/activate
```

Install required packages. (`requirements.txt` includes all submodule packages except flash-atten.)

```bash
pip install -r requirements.txt
```

Additionally, install flash-atten package after installing `requirements.txt`.

```bash
pip install flash-attn==2.3.0
```

Because SelectLLM is based on OPENAI's API, need to create an .env file to save API key.

```bash
vim .env
```

```env
# .env file
OPENAI_API_KEY="Your OPENAI API Key"
```

You are all set.

## Datasets

- Cleaned Alpaca Dataset
- Dolly

For now, we use ready-made datasets to do sampling. Datasets are available under datasets/data/

### Sampling

```bash
python scripts/sampling.py \
--sample_type selectllm \
--n_instances 1000 \
--data_set dolly \
--random_state 2021 \
--ftype diverse
```

```
--sample_type: Sampling algorithm type - (selectllm, random, coreset, rouge, etc)

--data_set: A specific dataset or combined dataset of all to run sampling on - (all, dolly, cleaned_alpaca, etc)`

--n_instances: No. of examples to sample from the dataset (eg: 1000/2000/3000)

--random_state: Random Seed Number for reproducibility (eg: 2021, 2022, 2023)

--ftype (Optional): diverse/random/similar for selectllm and perp/length for perplexity/Length based sampling
```

Note: Choose ftype as diverse to utilize the SelectLLM algorithm based on our paper

Sampled dataset will be saved under `datasets/sampled/{data_set}/{sampling_type}/{n_instances}/{random_state}`

## Fine-tuning
For llama 2 7B model, sign up at and request for weights from: https://llama.meta.com/llama-downloads

User can store model by creating the below defined llama path and model path given in the bash script (Optional). 

```bash
CUDA_VISIBLE_DEVICES=0 python llama2/scripts/finetune.py \
--data_path datasets/sampled/dolly/selectllm/diverse/1000/2021/sampled_selectllm_1000.parquet.gzip \
--sample_type selectllm \
--data_set dolly \
--n_instances 1000 \
--random_state 2021 \
--ftype diverse \
--llama_path llama_2_hf/Instructllama-2-7b/ \
--model_path llama_2_hf/llama-2-7b/7bweights
```

```
--data_path: Path to the saved sampled data given by:

datasets/sampled/{data_set}/{sampling_type}/{ftype}/{n_instances}/{random_state}/sampled_{sampling_type}_{n_instances}.parquet.gzip
ftype is optional, so it can be omitted.

--model_path: Path to where the llama 2 HuggingFace converted weights are stored

--llama_path: Directory path where you want the different llama 2 finetuned model weights to be saved
```

## Inferences

Generate inferences from the fine-tuned model with the test dataset.

```bash
CUDA_VISIBLE_DEVICES=0 python llama2/scripts/generated_inferences_finetuned.py \
--sample_type selectllm \
--data_set dolly \
--n_instances 1000 \
--det True \
--random_state 2021 \
--ftype diverse \
--llama_path llama_2_hf/Instructllama-2-7b
```

```bash
--det: Deterministic Sampling or not
```

## Evaluation
Compare inferences to the ground-truth by cosine similarities, rouge, and perplexity. 3 sets of inferences for each random state (2021, 2022, 2023) need to be generated before running the evaluation.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/metrics.py \
--data_set dolly \
--sample_type selectllm \
--n_instances 1000 \
--ftype diverse
```
