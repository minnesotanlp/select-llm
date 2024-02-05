#!/bin/bash 

# Sampling commands

# CUDA_VISIBLE_DEVICES=1 python ../scripts/sampling.py --sample_type llm_search --n_instances 1000 --data_set dolly --ftype diverse_kmeans_rand --random_state 2021
# CUDA_VISIBLE_DEVICES=1 python ../scripts/sampling.py --sample_type llm_search --n_instances 1000 --data_set dolly --ftype diverse_kmeans_rand --random_state 2022
# CUDA_VISIBLE_DEVICES=1 python ../scripts/sampling.py --sample_type llm_search --n_instances 1000 --data_set dolly --ftype diverse_kmeans_rand --random_state 2023
# CUDA_VISIBLE_DEVICES=1 python ../scripts/sampling.py --sample_type llm_search --n_instances 1000 --data_set dolly --ftype diverse_kmeans_open --random_state 2021
# CUDA_VISIBLE_DEVICES=1 python ../scripts/sampling.py --sample_type llm_search --n_instances 1000 --data_set dolly --ftype diverse_kmeans_open --random_state 2022
# CUDA_VISIBLE_DEVICES=1 python ../scripts/sampling.py --sample_type llm_search --n_instances 1000 --data_set dolly --ftype diverse_kmeans_open --random_state 2023
# CUDA_VISIBLE_DEVICES=1 python ../scripts/sampling.py --sample_type llm_search --n_instances 1000 --data_set dolly --ftype list_first --random_state 2021
# CUDA_VISIBLE_DEVICES=1 python ../scripts/sampling.py --sample_type llm_search --n_instances 1000 --data_set dolly --ftype list_last --random_state 2021
# CUDA_VISIBLE_DEVICES=1 python ../scripts/sampling.py --sample_type llm_search --n_instances 1000 --data_set dolly --ftype list_center --random_state 2021

 ## Finetuning commands

# CUDA_VISIBLE_DEVICES=1 python ../llama2/scripts/finetune.py --data_path /corpora/InstructTune/ait_single/active-instruction-tuning/datasets/sampled/dolly/llm_search/diverse_kmeans_rand/1000/2021/sampled_llm_search_1000.parquet.gzip --sample_type llm_search --n_instances 1000 --data_set dolly --ftype diverse_kmeans_rand --random_state 2021
# CUDA_VISIBLE_DEVICES=1 python ../llama2/scripts/finetune.py --data_path /corpora/InstructTune/ait_single/active-instruction-tuning/datasets/sampled/dolly/llm_search/diverse_kmeans_rand/1000/2022/sampled_llm_search_1000.parquet.gzip --sample_type llm_search --n_instances 1000 --data_set dolly --ftype diverse_kmeans_rand --random_state 2022
# CUDA_VISIBLE_DEVICES=1 python ../llama2/scripts/finetune.py --data_path /corpora/InstructTune/ait_single/active-instruction-tuning/datasets/sampled/dolly/llm_search/diverse_kmeans_rand/1000/2023/sampled_llm_search_1000.parquet.gzip --sample_type llm_search --n_instances 1000 --data_set dolly --ftype diverse_kmeans_rand --random_state 2023

# CUDA_VISIBLE_DEVICES=1 python ../llama2/scripts/finetune.py --data_path /corpora/InstructTune/ait_single/active-instruction-tuning/datasets/sampled/dolly/llm_search/diverse_kmeans_open/1000/2021/sampled_llm_search_1000.parquet.gzip --sample_type llm_search --n_instances 1000 --data_set dolly --ftype diverse_kmeans_open --random_state 2021
# CUDA_VISIBLE_DEVICES=1 python ../llama2/scripts/finetune.py --data_path /corpora/InstructTune/ait_single/active-instruction-tuning/datasets/sampled/dolly/llm_search/diverse_kmeans_open/1000/2022/sampled_llm_search_1000.parquet.gzip --sample_type llm_search --n_instances 1000 --data_set dolly --ftype diverse_kmeans_open --random_state 2022
# CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/finetune.py --data_path /corpora/InstructTune/ait_single/active-instruction-tuning/datasets/sampled/dolly/llm_search/diverse_kmeans_open/1000/2023/sampled_llm_search_1000.parquet.gzip --sample_type llm_search --n_instances 1000 --data_set dolly --ftype diverse_kmeans_open --random_state 2023

# CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/finetune.py --data_path /corpora/InstructTune/ait_single/active-instruction-tuning/datasets/sampled/dolly/llm_search/list_first/1000/2021/sampled_llm_search_1000.parquet.gzip --sample_type llm_search --n_instances 1000 --data_set dolly --ftype list_first --random_state 2023
# CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/finetune.py --data_path /corpora/InstructTune/ait_single/active-instruction-tuning/datasets/sampled/dolly/llm_search/list_last/1000/2021/sampled_llm_search_1000.parquet.gzip --sample_type llm_search --n_instances 1000 --data_set dolly --ftype list_last --random_state 2023
# CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/finetune.py --data_path /corpora/InstructTune/ait_single/active-instruction-tuning/datasets/sampled/dolly/llm_search/list_center/1000/2021/sampled_llm_search_1000.parquet.gzip --sample_type llm_search --n_instances 1000 --data_set dolly --ftype list_center --random_state 2023


# # Inference Commands 
# CUDA_VISIBLE_DEVICES=1 python ../llama2/scripts/infer_finetuned.py --sample_type llm_search --data_set dolly --n_instances 1000 --format True --det True --ftype diverse_kmeans_rand --random_state 2021 &  
# CUDA_VISIBLE_DEVICES=1 python ../llama2/scripts/infer_finetuned.py --sample_type llm_search --data_set dolly --n_instances 1000 --format True --det True --ftype diverse_kmeans_rand --random_state 2022 & 
# CUDA_VISIBLE_DEVICES=1 python ../llama2/scripts/infer_finetuned.py --sample_type llm_search --data_set dolly --n_instances 1000 --format True --det True --ftype diverse_kmeans_rand --random_state 2023 &  
# wait

# CUDA_VISIBLE_DEVICES=1 python ../llama2/scripts/infer_finetuned.py --sample_type llm_search --data_set dolly --n_instances 1000 --format True --det True --ftype diverse_kmeans_open --random_state 2021 &  
# CUDA_VISIBLE_DEVICES=1 python ../llama2/scripts/infer_finetuned.py --sample_type llm_search --data_set dolly --n_instances 1000 --format True --det True --ftype diverse_kmeans_open --random_state 2022 & 
# CUDA_VISIBLE_DEVICES=1 python ../llama2/scripts/infer_finetuned.py --sample_type llm_search --data_set dolly --n_instances 1000 --format True --det True --ftype diverse_kmeans_open --random_state 2023 &  
# wait

# CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/infer_finetuned.py --sample_type llm_search --data_set dolly --n_instances 1000 --format True --det True --ftype list_first --random_state 2023 &  
# CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/infer_finetuned.py --sample_type llm_search --data_set dolly --n_instances 1000 --format True --det True --ftype list_center --random_state 2023 & 
# CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/infer_finetuned.py --sample_type llm_search --data_set dolly --n_instances 1000 --format True --det True --ftype list_last --random_state 2023 &  
# wait


# Run Metrics


# CUDA_VISIBLE_DEVICES=1 python ../scripts/metrics2.py --data_set dolly --sample_type llm_search --ftype diverse_kmeans_rand --n_instances 1000
# CUDA_VISIBLE_DEVICES=1 python ../scripts/metrics2.py --data_set dolly --sample_type llm_search --ftype diverse_kmeans_open --n_instances 1000
CUDA_VISIBLE_DEVICES=1 python ../scripts/metrics.py --data_set dolly --sample_type llm_search --ftype list_first --n_instances 1000
CUDA_VISIBLE_DEVICES=1 python ../scripts/metrics.py --data_set dolly --sample_type llm_search --ftype list_last --n_instances 1000
CUDA_VISIBLE_DEVICES=1 python ../scripts/metrics.py --data_set dolly --sample_type llm_search --ftype list_center --n_instances 1000
