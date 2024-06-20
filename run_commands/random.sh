# CUDA_VISIBLE_DEVICES=0 python ../scripts/sampling.py --sample_type selectllm --ftype diverse --n_instances 2000 --data_set dolly --random_state 2021 -l llama
# CUDA_VISIBLE_DEVICES=0 python ../scripts/sampling.py --sample_type selectllm --ftype diverse --n_instances 2000 --data_set dolly --random_state 2022 -l llama
# CUDA_VISIBLE_DEVICES=0 python ../scripts/sampling.py --sample_type selectllm --ftype diverse --n_instances 2000 --data_set dolly --random_state 2023 -l llama

# CUDA_VISIBLE_DEVICES=1 python ../scripts/sampling.py --sample_type selectllm --ftype diverse --n_instances 3000 --data_set dolly --random_state 2021 -l llama
# CUDA_VISIBLE_DEVICES=1 python ../scripts/sampling.py --sample_type selectllm --ftype diverse --n_instances 3000 --data_set dolly --random_state 2022 -l llama
# CUDA_VISIBLE_DEVICES=1 python ../scripts/sampling.py --sample_type selectllm --ftype diverse --n_instances 3000 --data_set dolly --random_state 2023 -l llama


# source /corpora/InstructTune/cloned_ait/new_repo/envs/mistralnew/bin/activate


# source /corpora/InstructTune/new_env_repo/test2/bin/activate

# CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/finetune.py \
# --data_path ../datasets/sampled/dolly/selectllm/diverse/llama/3000/2021/sampled_selectllm_3000.parquet.gzip \
# --sample_type selectllm \
# --ftype diverse \
# --local_model llama \
# --data_set dolly \
# --n_instances 3000 \
# --random_state 2021 \
# --llama_path /corpora/llama/llama_2_hf/Instructllama-2-7b/ \
# --model_path /corpora/llama/llama_2_hf/llama-2-7b/7bweights/

# CUDA_VISIBLE_DEVICES=1 python ../llama2/scripts/generated_inferences_finetuned.py -s selectllm -f diverse -d dolly -n 3000 --det True -r 2021 -lp /corpora/llama/llama_2_hf/Instructllama-2-7b/ -l llama &
# CUDA_VISIBLE_DEVICES=1 python ../llama2/scripts/generated_inferences_finetuned.py -s selectllm -f diverse -d dolly -n 3000 --det True -r 2022 -lp /corpora/llama/llama_2_hf/Instructllama-2-7b/ -l llama &
# CUDA_VISIBLE_DEVICES=1 python ../llama2/scripts/generated_inferences_finetuned.py -s selectllm -f diverse -d dolly -n 3000 --det True -r 2023 -lp /corpora/llama/llama_2_hf/Instructllama-2-7b/ -l llama &
# wait

# CUDA_VISIBLE_DEVICES=0 python ../mistral7b/scripts/finetune.py -dp ../datasets/sampled/dolly/selectllm/diverse/3000/2021/sampled_selectllm_3000.parquet.gzip -s selectllm -f diverse -d dolly  -n 3000  -r 2021 -l gpt3.5 -mp /corpora/mistral_models/ 
CUDA_VISIBLE_DEVICES=0 python ../mistral7b/scripts/finetune.py -dp ../datasets/sampled/dolly/selectllm/diverse/3000/2022/sampled_selectllm_3000.parquet.gzip -s selectllm -f diverse -d dolly  -n 3000  -r 2022 -l gpt3.5 -mp /corpora/mistral_models/ 
CUDA_VISIBLE_DEVICES=0 python ../mistral7b/scripts/finetune.py -dp ../datasets/sampled/dolly/selectllm/diverse/3000/2023/sampled_selectllm_3000.parquet.gzip -s selectllm -f diverse -d dolly  -n 3000  -r 2023 -l gpt3.5 -mp /corpora/mistral_models/ 


CUDA_VISIBLE_DEVICES=0 python ../mistral7b/scripts/generated_inferences_finetuned.py --sample_type selectllm --ftype diverse -l gpt3.5 --data_set dolly --n_instances 3000 --det True --random_state 2021 --mistral_path /corpora/mistral_models/ &
CUDA_VISIBLE_DEVICES=0 python ../mistral7b/scripts/generated_inferences_finetuned.py --sample_type selectllm --ftype diverse -l gpt3.5 --data_set dolly --n_instances 3000 --det True --random_state 2022 --mistral_path /corpora/mistral_models/ &
CUDA_VISIBLE_DEVICES=0 python ../mistral7b/scripts/generated_inferences_finetuned.py --sample_type selectllm --ftype diverse -l gpt3.5 --data_set dolly --n_instances 3000 --det True --random_state 2023 --mistral_path /corpora/mistral_models/ &
wait

source /corpora/InstructTune/new_env_repo/test2/bin/activate
CUDA_VISIBLE_DEVICES=0 python ../scripts/eval.py -d dolly -s selectllm -f diverse -l gpt3.5 -n 3000 