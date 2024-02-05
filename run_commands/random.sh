#!/bin/bash

# dataset="dolly"
# sample_types=("random" "oe")
# random_states=(2021 2022 2023)
# n_instances_values=(1000 2000 3000)

# for sample_type in "${sample_types[@]}"; do
#     for random_state in "${random_states[@]}"; do
#         for n_instances in "${n_instances_values[@]}"; do

#             # Run sampling.py
#             CUDA_VISIBLE_DEVICES=0 python ../scripts/sampling.py --sample_type "$sample_type" --n_instances "$n_instances" --data_set "$dataset" --random_state "$random_state"

#             # Run finetune.py
#             CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/finetune.py --data_path "../datasets/sampled/${dataset}/${sample_type}/${n_instances}/${random_state}/sampled_${sample_type}_${n_instances}.parquet.gzip" --sample_type "$sample_type" --data_set "$dataset" --n_instances "$n_instances" --random_state "$random_state"

#         done
#     done
# done
# # Run infer_finetuned.py


#CHANGE TEST AND METRIC SETS FOR CA LATER
# CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/infer_cross.py --sample_type random --data_set cleaned_alpaca --n_instances 3000 --format True --det True --random_state 2021 --cross_dataset dolly &
# CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/infer_cross.py --sample_type oe --data_set cleaned_alpaca --n_instances 3000 --format True --det True --random_state 2021 --cross_dataset dolly &
# CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/infer_cross.py --sample_type llm_search --data_set cleaned_alpaca --n_instances 3000 --format True --det True --random_state 2021 --ftype coreset --cross_dataset dolly &
# wait

# CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/infer_cross.py --sample_type llm_search --data_set cleaned_alpaca --n_instances 3000 --format True --det True --random_state 2021 --ftype diverse_kmeans_complex_v2_fine --cross_dataset dolly &
# CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/infer_cross.py --sample_type llm_search --data_set cleaned_alpaca --n_instances 3000 --format True --det True --random_state 2021 --ftype diverse_kmeans_complex_more2 --cross_dataset dolly &
# wait

# CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/infer_cross.py --sample_type llm_search --data_set dolly --n_instances 1000 --format True --det True --random_state 2021 --ftype diverse_kmeans --cross_dataset cleaned_alpaca &
CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/infer_cross.py --sample_type llm_search --data_set cleaned_alpaca --n_instances 1000 --format True --det True --random_state 2021 --ftype diverse_kmeans --cross_dataset dolly &
CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/infer_cross.py --sample_type llm_search --data_set cleaned_alpaca --n_instances 3000 --format True --det True --random_state 2021 --ftype diverse_kmeans --cross_dataset dolly &
wait

# # #Eval

# CUDA_VISIBLE_DEVICES=0 python ../scripts/metrics_cross.py --data_set cleaned_alpaca --sample_type random --n_instances 3000 --cross_dataset dolly
# CUDA_VISIBLE_DEVICES=0 python ../scripts/metrics_cross.py --data_set cleaned_alpaca --sample_type oe --n_instances 3000 --cross_dataset dolly
# CUDA_VISIBLE_DEVICES=0 python ../scripts/metrics_cross.py --data_set cleaned_alpaca --sample_type llm_search --ftype coreset --n_instances 3000 --cross_dataset dolly
# CUDA_VISIBLE_DEVICES=0 python ../scripts/metrics_cross.py --data_set cleaned_alpaca --sample_type llm_search --ftype diverse_kmeans_complex_v2_fine --n_instances 3000 --cross_dataset dolly
# CUDA_VISIBLE_DEVICES=0 python ../scripts/metrics_cross.py --data_set cleaned_alpaca --sample_type llm_search --ftype diverse_kmeans_complex_more2 --n_instances 3000 --cross_dataset dolly
# CUDA_VISIBLE_DEVICES=0 python ../scripts/metrics_cross.py --data_set dolly --sample_type llm_search --ftype diverse_kmeans --n_instances 1000 --cross_dataset cleaned_alpaca
CUDA_VISIBLE_DEVICES=0 python ../scripts/metrics_cross.py --data_set cleaned_alpaca --sample_type llm_search --ftype diverse_kmeans --n_instances 1000 --cross_dataset dolly
CUDA_VISIBLE_DEVICES=0 python ../scripts/metrics_cross.py --data_set cleaned_alpaca --sample_type llm_search --ftype diverse_kmeans --n_instances 3000 --cross_dataset dolly
