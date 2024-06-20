# source /corpora/InstructTune/new_env_repo/test2/bin/activate

# CUDA_VISIBLE_DEVICES=1 python ../scripts/sampling.py --sample_type random --n_instances 2000 --data_set dolly --random_state 2021
# CUDA_VISIBLE_DEVICES=1 python ../scripts/sampling.py --sample_type random --n_instances 2000 --data_set dolly --random_state 2022
# CUDA_VISIBLE_DEVICES=1 python ../scripts/sampling.py --sample_type random --n_instances 2000 --data_set dolly --random_state 2023


source /corpora/InstructTune/cloned_ait/new_repo/envs/mistralnew/bin/activate


CUDA_VISIBLE_DEVICES=0 python ../mistral7b/scripts/finetune.py -dp ../datasets/sampled/dolly/random/2000/2021/sampled_random_2000.parquet.gzip -s random -d dolly  -n 2000  -r 2021 -mp /corpora/mistral_models/ -fm llama
CUDA_VISIBLE_DEVICES=0 python ../mistral7b/scripts/finetune.py -dp ../datasets/sampled/dolly/random/2000/2022/sampled_random_2000.parquet.gzip -s random -d dolly  -n 2000  -r 2022 -mp /corpora/mistral_models/ -fm llama
CUDA_VISIBLE_DEVICES=0 python ../mistral7b/scripts/finetune.py -dp ../datasets/sampled/dolly/random/2000/2023/sampled_random_2000.parquet.gzip -s random -d dolly  -n 2000  -r 2023 -mp /corpora/mistral_models/ -fm llama


CUDA_VISIBLE_DEVICES=0 python ../mistral7b/scripts/generated_inferences_finetuned.py -s random -d dolly -n 2000 --det True -r 2021 -mp /corpora/mistral_models/ -fm llama &
CUDA_VISIBLE_DEVICES=0 python ../mistral7b/scripts/generated_inferences_finetuned.py -s random -d dolly -n 2000 --det True -r 2022 -mp /corpora/mistral_models/ -fm llama &
CUDA_VISIBLE_DEVICES=0 python ../mistral7b/scripts/generated_inferences_finetuned.py -s random -d dolly -n 2000 --det True -r 2023 -mp /corpora/mistral_models/ -fm llama &
wait

source /corpora/InstructTune/new_env_repo/test2/bin/activate
CUDA_VISIBLE_DEVICES=0 python ../scripts/eval.py -d dolly -s random -n 2000 