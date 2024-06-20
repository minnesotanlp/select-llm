# CUDA_VISIBLE_DEVICES=0 python ../scripts/sampling.py --sample_type selectllm --ftype diverse --n_instances 2000 --data_set dolly --random_state 2021 -l llama
# CUDA_VISIBLE_DEVICES=0 python ../scripts/sampling.py --sample_type selectllm --ftype diverse --n_instances 2000 --data_set dolly --random_state 2022 -l llama
# CUDA_VISIBLE_DEVICES=0 python ../scripts/sampling.py --sample_type selectllm --ftype diverse --n_instances 2000 --data_set dolly --random_state 2023 -l llama

# CUDA_VISIBLE_DEVICES=1 python ../scripts/sampling.py --sample_type selectllm --ftype diverse --n_instances 3000 --data_set dolly --random_state 2021 -l llama
# CUDA_VISIBLE_DEVICES=1 python ../scripts/sampling.py --sample_type selectllm --ftype diverse --n_instances 3000 --data_set dolly --random_state 2022 -l llama
# CUDA_VISIBLE_DEVICES=1 python ../scripts/sampling.py --sample_type selectllm --ftype diverse --n_instances 3000 --data_set dolly --random_state 2023 -l llama


source /corpora/InstructTune/cloned_ait/new_repo/envs/mistralunsloth/bin/activate

CUDA_VISIBLE_DEVICES=1 python ../mistral7b/scripts/finetune.py -dp ../datasets/sampled/dolly/rouge/3000/2021/sampled_rouge_3000.parquet.gzip -s rouge -d dolly  -n 3000  -r 2021 -mp /corpora/mistral_models/ -fm llama
CUDA_VISIBLE_DEVICES=1 python ../mistral7b/scripts/finetune.py -dp ../datasets/sampled/dolly/rouge/3000/2022/sampled_rouge_3000.parquet.gzip -s rouge -d dolly  -n 3000  -r 2022 -mp /corpora/mistral_models/ -fm llama
CUDA_VISIBLE_DEVICES=1 python ../mistral7b/scripts/finetune.py -dp ../datasets/sampled/dolly/rouge/3000/2023/sampled_rouge_3000.parquet.gzip -s rouge -d dolly  -n 3000  -r 2023 -mp /corpora/mistral_models/ -fm llama


CUDA_VISIBLE_DEVICES=1 python ../mistral7b/scripts/generated_inferences_finetuned.py -s rouge -d dolly -n 3000 --det True -r 2021 -mp /corpora/mistral_models/ -fm llama &
CUDA_VISIBLE_DEVICES=1 python ../mistral7b/scripts/generated_inferences_finetuned.py -s rouge -d dolly -n 3000 --det True -r 2022 -mp /corpora/mistral_models/ -fm llama &
CUDA_VISIBLE_DEVICES=1 python ../mistral7b/scripts/generated_inferences_finetuned.py -s rouge -d dolly -n 3000 --det True -r 2023 -mp /corpora/mistral_models/ -fm llama &
wait

source /corpora/InstructTune/new_env_repo/test2/bin/activate
CUDA_VISIBLE_DEVICES=1 python ../scripts/eval.py -d dolly -s rouge -n 3000 -fm llama


