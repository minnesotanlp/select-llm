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

CURRENT REQUIREMENTS.TXT IS OUTDATED AND WILL BE UPDATED SOON:

Install required packages. (`requirements.txt` includes all submodule packages except flash-atten.)

```bash
pip install -r requirements.txt
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
--sample_type random \
--n_instances 1000 \
--data_set dolly \
--random_state 2021 \
--local_model gpt3.5
```

```
--sample_type: Sampling algorithm type - (selectllm, random, coreset, rouge, etc)

--data_set: A specific dataset or combined dataset of all to run sampling on - (all, dolly, cleaned_alpaca, etc)`

--n_instances: No. of examples to sample from the dataset (eg: 1000/2000/3000)

--random_state: Random Seed Number for reproducibility (eg: 2021, 2022, 2023)

--ftype (Optional): diverse/random/similar for selectllm and perp/length for perplexity/Length based sampling

--local_model: Model to use for Selectllm (Currently: llama/gpt3.5) [llama is llama 3 70B Q8, requires ollama set up]
```

Note: Choose ftype as diverse to utilize the SelectLLM algorithm based on our paper

Sampled dataset will be saved under `datasets/sampled/{data_set}/{sampling_type}/{n_instances}/{random_state}`

## Fine-tuning

Finetuning is based on the unsloth ai library. 

For llama 2 7B model, can sign up at and request for weights from: https://llama.meta.com/llama-downloads

```bash
CUDA_VISIBLE_DEVICES=0 python training/finetune.py \
--data_path datasets/sampled/dolly/random/1000/2021/sampled_random_1000.parquet.gzip \
--sample_type random \
--data_set dolly \
--n_instances 1000 \
--random_state 2021 \
--mistral_path mistral_directory \
--finetune_model mistral
```

```
--data_path: Path to the saved sampled data given by:

datasets/sampled/{data_set}/{sampling_type}/{n_instances}/{random_state}/sampled_{sampling_type}_{n_instances}.parquet.gzip

or for selectllm sampling by:

datasets/sampled/{data_set}/{sampling_type}/{f_type}/{local_model}/{n_instances}/{random_state}/sampled_{sampling_type}_{n_instances}.parquet.gzip 

--mistral_path: Directory where you want the different finetuned model weights to be saved

--finetune_model: Finetuning mistral 7B v0.3 or Llama 2 7B [options: mistral, llama]
```

## Inferences

Generate inferences from the fine-tuned model with the test Dolly or Cleaned Alpaca datasets.

```bash
CUDA_VISIBLE_DEVICES=0 python training/generate_inferences_testsets.py \
--sample_type random \
--data_set dolly \
--n_instances 1000 \
--det True \
--random_state 2021 \
--mistral_path mistral_directory \
--finetune_model mistral
```

```bash
--det: Deterministic Sampling or not
```

## Evaluation
Compare inferences to the ground-truth by rouge and cosine similarity (Uncomment code to enable perplexity too). 3 sets of inferences for each random state (2021, 2022, 2023) need to be generated before running the evaluation.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval.py \
--data_set dolly \
--sample_type random \
--n_instances 1000 \
--finetune_model mistral
```

Evaluation steps on MT-Bench and Alpaca-Eval to be added soon.
