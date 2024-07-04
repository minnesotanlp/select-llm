# source /corpora/InstructTune/cloned_ait/new_repo/envs/mistralunsloth/bin/activate

# CUDA_VISIBLE_DEVICES=0 python ../training/finetune_noval.py -dp ../datasets/sampled/dolly/alpagasus/3000/2021/sampled_alpagasus_3000.parquet.gzip -s alpagasus -d dolly  -n 3000  -r 2021 -mp /corpora/mistral_models_NoVal/ -fm mistral
# CUDA_VISIBLE_DEVICES=0 python ../training/finetune_noval.py -dp ../datasets/sampled/dolly/alpagasus/3000/2022/sampled_alpagasus_3000.parquet.gzip -s alpagasus -d dolly  -n 3000  -r 2022 -mp /corpora/mistral_models_NoVal/ -fm mistral
# CUDA_VISIBLE_DEVICES=0 python ../training/finetune_noval.py -dp ../datasets/sampled/dolly/alpagasus/3000/2023/sampled_alpagasus_3000.parquet.gzip -s alpagasus -d dolly  -n 3000  -r 2023 -mp /corpora/mistral_models_NoVal/ -fm mistral


# CUDA_VISIBLE_DEVICES=0 python ../training/generate_inferences_noval.py -s alpagasus -d dolly -n 3000 --det True -r 2021 -mp /corpora/mistral_models_NoVal/ -fm mistral &
# CUDA_VISIBLE_DEVICES=0 python ../training/generate_inferences_noval.py -s alpagasus -d dolly -n 3000 --det True -r 2022 -mp /corpora/mistral_models_NoVal/ -fm mistral &
# CUDA_VISIBLE_DEVICES=0 python ../training/generate_inferences_noval.py -s alpagasus -d dolly -n 3000 --det True -r 2023 -mp /corpora/mistral_models_NoVal/ -fm mistral &
# wait

# source /corpora/InstructTune/new_env_repo/test2/bin/activate
# CUDA_VISIBLE_DEVICES=0 python ../scripts/eval_noval.py -d dolly -s alpagasus -n 3000 -fm mistral


source /corpora/InstructTune/new_env_repo/testmar27/bin/activate

CUDA_VISIBLE_DEVICES=0 python ../scripts/sampling.py -d dolly -s selectllm -f diverse -l llama -n 3000 -r 2021
# CUDA_VISIBLE_DEVICES=0 python ../scripts/sampling.py -d dolly -s selectllm -f diverse -l llama -n 3000 -r 2022
# CUDA_VISIBLE_DEVICES=0 python ../scripts/sampling.py -d dolly -s selectllm -f diverse -l llama -n 3000 -r 2023

# source /corpora/InstructTune/cloned_ait/new_repo/envs/mistralunsloth/bin/activate

# CUDA_VISIBLE_DEVICES=0 python ../training/finetune_noval.py -dp ../datasets/sampled/dolly/selectllm/diverse/llama/3000/2021/sampled_selectllm_3000.parquet.gzip -s selectllm -f diverse -l llama -d dolly  -n 3000  -r 2021 -mp /corpora/mistral_models_NoVal/ -fm mistral
# CUDA_VISIBLE_DEVICES=0 python ../training/finetune_noval.py -dp ../datasets/sampled/dolly/selectllm/diverse/llama/3000/2022/sampled_selectllm_3000.parquet.gzip -s selectllm -f diverse -l llama -d dolly  -n 3000  -r 2022 -mp /corpora/mistral_models_NoVal/ -fm mistral
# CUDA_VISIBLE_DEVICES=0 python ../training/finetune_noval.py -dp ../datasets/sampled/dolly/selectllm/diverse/llama/3000/2023/sampled_selectllm_3000.parquet.gzip -s selectllm -f diverse -l llama -d dolly  -n 3000  -r 2023 -mp /corpora/mistral_models_NoVal/ -fm mistral

# CUDA_VISIBLE_DEVICES=0 python ../training/generate_inferences_noval.py -s selectllm -f diverse -l llama -d dolly -n 3000 --det True -r 2021 -mp /corpora/mistral_models_NoVal/ -fm mistral &
# CUDA_VISIBLE_DEVICES=0 python ../training/generate_inferences_noval.py -s selectllm -f diverse -l llama -d dolly -n 3000 --det True -r 2022 -mp /corpora/mistral_models_NoVal/ -fm mistral &
# CUDA_VISIBLE_DEVICES=0 python ../training/generate_inferences_noval.py -s selectllm -f diverse -l llama -d dolly -n 3000 --det True -r 2023 -mp /corpora/mistral_models_NoVal/ -fm mistral &
# wait

# source /corpora/InstructTune/new_env_repo/test2/bin/activate
# CUDA_VISIBLE_DEVICES=0 python ../scripts/eval_noval.py -d dolly -s selectllm -f diverse -l llama -n 3000 -fm mistral
