CUDA_VISIBLE_DEVICES=0 python ../scripts/sampling.py --sample_type selectllm --ftype diverse --n_instances 1000 --data_set dolly --random_state 2021 -l llama

#!/bin/bash
# python ../scripts/sampling.py --sample_type coreset --n_instances 3000 --data_set dolly --random_state 2021
# python ../scripts/sampling.py --sample_type coreset --n_instances 3000 --data_set dolly --random_state 2022
# python ../scripts/sampling.py --sample_type coreset --n_instances 3000 --data_set dolly --random_state 2023




# CUDA_VISIBLE_DEVICES=1 python ../mistral7b/scripts/finetune.py --data_path ../datasets/sampled/dolly/coreset/3000/2021/sampled_coreset_3000.parquet.gzip --sample_type coreset  --data_set dolly  --n_instances 3000  --random_state 2021  --mistral_path /corpora/mistral_models/ 
# CUDA_VISIBLE_DEVICES=1 python ../mistral7b/scripts/finetune.py --data_path ../datasets/sampled/dolly/coreset/3000/2022/sampled_coreset_3000.parquet.gzip --sample_type coreset  --data_set dolly  --n_instances 3000  --random_state 2022  --mistral_path /corpora/mistral_models/ 
# CUDA_VISIBLE_DEVICES=1 python ../mistral7b/scripts/finetune.py --data_path ../datasets/sampled/dolly/coreset/3000/2023/sampled_coreset_3000.parquet.gzip --sample_type coreset  --data_set dolly  --n_instances 3000  --random_state 2023  --mistral_path /corpora/mistral_models/ 

# CUDA_VISIBLE_DEVICES=1 python ../mistral7b/scripts/generated_inferences_finetuned.py --sample_type coreset --data_set dolly --n_instances 3000 --det True --random_state 2021 --mistral_path /corpora/mistral_models/ &
# CUDA_VISIBLE_DEVICES=1 python ../mistral7b/scripts/generated_inferences_finetuned.py --sample_type coreset --data_set dolly --n_instances 3000 --det True --random_state 2022 --mistral_path /corpora/mistral_models/ &
# CUDA_VISIBLE_DEVICES=1 python ../mistral7b/scripts/generated_inferences_finetuned.py --sample_type coreset --data_set dolly --n_instances 3000 --det True --random_state 2023 --mistral_path /corpora/mistral_models/ &
# wait

# CUDA_VISIBLE_DEVICES=1 python ../mistral7b/scripts/finetune.py --data_path ../datasets/sampled/dolly/random/3000/2021/sampled_random_3000.parquet.gzip --sample_type random  --data_set dolly  --n_instances 3000  --random_state 2021  --mistral_path /corpora/mistral_models/ 
# CUDA_VISIBLE_DEVICES=1 python ../mistral7b/scripts/finetune.py --data_path ../datasets/sampled/dolly/random/3000/2022/sampled_random_3000.parquet.gzip --sample_type random  --data_set dolly  --n_instances 3000  --random_state 2022  --mistral_path /corpora/mistral_models/ 
# CUDA_VISIBLE_DEVICES=1 python ../mistral7b/scripts/finetune.py --data_path ../datasets/sampled/dolly/random/3000/2023/sampled_random_3000.parquet.gzip --sample_type random  --data_set dolly  --n_instances 3000  --random_state 2023  --mistral_path /corpora/mistral_models/ 


# CUDA_VISIBLE_DEVICES=1 python ../mistral7b/scripts/generated_inferences_finetuned.py --sample_type random --data_set dolly --n_instances 3000 --det True --random_state 2021 --mistral_path /corpora/mistral_models/ &
# CUDA_VISIBLE_DEVICES=1 python ../mistral7b/scripts/generated_inferences_finetuned.py --sample_type random --data_set dolly --n_instances 3000 --det True --random_state 2022 --mistral_path /corpora/mistral_models/ &
# CUDA_VISIBLE_DEVICES=1 python ../mistral7b/scripts/generated_inferences_finetuned.py --sample_type random --data_set dolly --n_instances 3000 --det True --random_state 2023 --mistral_path /corpora/mistral_models/ &
# wait

# CUDA_VISIBLE_DEVICES=0 python ../mistral7b/scripts/finetune.py --data_path ../datasets/sampled/dolly/selectllm/diverse/3000/2021/sampled_selectllm_3000.parquet.gzip --sample_type selectllm  --data_set dolly  --n_instances 3000  --random_state 2021  --mistral_path /corpora/mistral_models/ --ftype diverse
# CUDA_VISIBLE_DEVICES=0 python ../mistral7b/scripts/finetune.py --data_path ../datasets/sampled/dolly/selectllm/diverse/3000/2022/sampled_selectllm_3000.parquet.gzip --sample_type selectllm  --data_set dolly  --n_instances 3000  --random_state 2022  --mistral_path /corpora/mistral_models/ --ftype diverse
# CUDA_VISIBLE_DEVICES=0 python ../mistral7b/scripts/finetune.py --data_path ../datasets/sampled/dolly/selectllm/diverse/3000/2023/sampled_selectllm_3000.parquet.gzip --sample_type selectllm  --data_set dolly  --n_instances 3000  --random_state 2023  --mistral_path /corpora/mistral_models/ --ftype diverse

# CUDA_VISIBLE_DEVICES=0 python ../mistral7b/scripts/generated_inferences_finetuned.py --sample_type selectllm --data_set dolly --n_instances 3000 --det True --random_state 2021 --mistral_path /corpora/mistral_models/ --ftype diverse &
# CUDA_VISIBLE_DEVICES=0 python ../mistral7b/scripts/generated_inferences_finetuned.py --sample_type selectllm --data_set dolly --n_instances 3000 --det True --random_state 2022 --mistral_path /corpora/mistral_models/ --ftype diverse &
# CUDA_VISIBLE_DEVICES=0 python ../mistral7b/scripts/generated_inferences_finetuned.py --sample_type selectllm --data_set dolly --n_instances 3000 --det True --random_state 2023 --mistral_path /corpora/mistral_models/ --ftype diverse &
# wait

# CUDA_VISIBLE_DEVICES=0 python ../scripts/metrics.py \
# --data_set dolly \
# --sample_type random \
# --n_instances 3000

# CUDA_VISIBLE_DEVICES=0 python ../scripts/metrics.py \
# --data_set dolly \
# --sample_type coreset \
# --n_instances 3000

# CUDA_VISIBLE_DEVICES=0 python ../scripts/metrics.py \
# --data_set dolly \
# --sample_type selectllm \
# --n_instances 3000 \
# --ftype diverse