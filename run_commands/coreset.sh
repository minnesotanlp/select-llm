# dataset="cleaned_alpaca"
# sample_types=("instructor")
# random_states=(2021)
# n_instances_values=(1000 2000 3000)

# for sample_type in "${sample_types[@]}"; do
#     for random_state in "${random_states[@]}"; do
#         for n_instances in "${n_instances_values[@]}"; do

#             rm -rf /corpora/InstructTune/elkdir/active-instruction-tuning/datasets/data/${dataset}/instructor
#             # Run sampling.py
#             CUDA_VISIBLE_DEVICES=0 python ../scripts/sampling.py --sample_type "$sample_type" --n_instances "$n_instances" --data_set "$dataset" --random_state "$random_state"

#             # Run finetune.py
#             CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/finetune.py --data_path "../datasets/sampled/${dataset}/${sample_type}/${n_instances}/${random_state}/sampled_${sample_type}_${n_instances}.parquet.gzip" --sample_type "$sample_type" --data_set "$dataset" --n_instances "$n_instances" --random_state "$random_state"

#         done
#     done
# done
# # Run infer_finetuned.py



# CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/infer_finetuned.py --sample_type instructor --data_set cleaned_alpaca --n_instances 1000 --format True --det True --random_state 2021 & 
# CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/infer_finetuned.py --sample_type instructor --data_set cleaned_alpaca --n_instances 2000 --format True --det True --random_state 2021 &
# CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/infer_finetuned.py --sample_type instructor --data_set cleaned_alpaca --n_instances 3000 --format True --det True --random_state 2021 &
# wait

#Eval

CUDA_VISIBLE_DEVICES=1 python ../scripts/metrics.py --data_set cleaned_alpaca --sample_type instructor --n_instances 1000
CUDA_VISIBLE_DEVICES=1 python ../scripts/metrics.py --data_set cleaned_alpaca --sample_type instructor --n_instances 2000
CUDA_VISIBLE_DEVICES=1 python ../scripts/metrics.py --data_set cleaned_alpaca --sample_type instructor --n_instances 3000