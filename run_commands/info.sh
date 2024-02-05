#Sampling
# CUDA_VISIBLE_DEVICES=0 python ../scripts/sampling.py --sample_type infoverse --n_instances 1000 --data_set cleaned_alpaca --ftype length_small --random_state 2021 
# CUDA_VISIBLE_DEVICES=0 python ../scripts/sampling.py --sample_type infoverse --n_instances 2000 --data_set cleaned_alpaca --ftype length_small --random_state 2021
# CUDA_VISIBLE_DEVICES=0 python ../scripts/sampling.py --sample_type infoverse --n_instances 3000 --data_set cleaned_alpaca --ftype length_small --random_state 2021
# CUDA_VISIBLE_DEVICES=0 python ../scripts/sampling.py --sample_type infoverse --n_instances 1000 --data_set cleaned_alpaca --ftype length_big --random_state 2021 
# CUDA_VISIBLE_DEVICES=0 python ../scripts/sampling.py --sample_type infoverse --n_instances 2000 --data_set cleaned_alpaca --ftype length_big --random_state 2021
# CUDA_VISIBLE_DEVICES=0 python ../scripts/sampling.py --sample_type infoverse --n_instances 3000 --data_set cleaned_alpaca --ftype length_big --random_state 2021
# CUDA_VISIBLE_DEVICES=0 python ../scripts/sampling.py --sample_type infoverse --n_instances 1000 --data_set cleaned_alpaca --ftype length_big --random_state 2022 
# CUDA_VISIBLE_DEVICES=0 python ../scripts/sampling.py --sample_type infoverse --n_instances 2000 --data_set cleaned_alpaca --ftype length_big --random_state 2022
# CUDA_VISIBLE_DEVICES=0 python ../scripts/sampling.py --sample_type infoverse --n_instances 3000 --data_set cleaned_alpaca --ftype length_big --random_state 2022
# CUDA_VISIBLE_DEVICES=0 python ../scripts/sampling.py --sample_type infoverse --n_instances 1000 --data_set cleaned_alpaca --ftype length_big --random_state 2023 
# CUDA_VISIBLE_DEVICES=0 python ../scripts/sampling.py --sample_type infoverse --n_instances 2000 --data_set cleaned_alpaca --ftype length_big --random_state 2023
# CUDA_VISIBLE_DEVICES=0 python ../scripts/sampling.py --sample_type infoverse --n_instances 3000 --data_set cleaned_alpaca --ftype length_big --random_state 2023
# CUDA_VISIBLE_DEVICES=0 python ../scripts/sampling.py --sample_type infoverse --n_instances 1000 --data_set cleaned_alpaca --ftype perp --random_state 2021
# CUDA_VISIBLE_DEVICES=0 python ../scripts/sampling.py --sample_type infoverse --n_instances 2000 --data_set cleaned_alpaca --ftype perp --random_state 2021
# CUDA_VISIBLE_DEVICES=0 python ../scripts/sampling.py --sample_type infoverse --n_instances 3000 --data_set cleaned_alpaca --ftype perp --random_state 2021

# CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/finetune.py --data_path ../datasets/sampled/cleaned_alpaca/infoverse/perp/1000/2021/sampled_infoverse_1000.parquet.gzip --sample_type infoverse --data_set cleaned_alpaca --n_instances 1000 --random_state 2021 --ftype perp
# CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/finetune.py --data_path ../datasets/sampled/cleaned_alpaca/infoverse/perp/2000/2021/sampled_infoverse_2000.parquet.gzip --sample_type infoverse --data_set cleaned_alpaca --n_instances 2000 --random_state 2021 --ftype perp
# CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/finetune.py --data_path ../datasets/sampled/cleaned_alpaca/infoverse/perp/3000/2021/sampled_infoverse_3000.parquet.gzip --sample_type infoverse --data_set cleaned_alpaca --n_instances 3000 --random_state 2021 --ftype perp

# CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/finetune.py --data_path ../datasets/sampled/cleaned_alpaca/infoverse/length_small/1000/2021/sampled_infoverse_1000.parquet.gzip --sample_type infoverse --data_set cleaned_alpaca --n_instances 1000 --random_state 2021 --ftype length_small
# CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/finetune.py --data_path ../datasets/sampled/cleaned_alpaca/infoverse/length_small/2000/2021/sampled_infoverse_2000.parquet.gzip --sample_type infoverse --data_set cleaned_alpaca --n_instances 2000 --random_state 2021 --ftype length_small
# CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/finetune.py --data_path ../datasets/sampled/cleaned_alpaca/infoverse/length_small/3000/2021/sampled_infoverse_3000.parquet.gzip --sample_type infoverse --data_set cleaned_alpaca --n_instances 3000 --random_state 2021 --ftype length_small

# CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/finetune.py --data_path ../datasets/sampled/cleaned_alpaca/infoverse/length_big/1000/2021/sampled_infoverse_1000.parquet.gzip --sample_type infoverse --data_set cleaned_alpaca --n_instances 1000 --random_state 2021 --ftype length_big
# CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/finetune.py --data_path ../datasets/sampled/cleaned_alpaca/infoverse/length_big/2000/2021/sampled_infoverse_2000.parquet.gzip --sample_type infoverse --data_set cleaned_alpaca --n_instances 2000 --random_state 2021 --ftype length_big
# CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/finetune.py --data_path ../datasets/sampled/cleaned_alpaca/infoverse/length_big/3000/2021/sampled_infoverse_3000.parquet.gzip --sample_type infoverse --data_set cleaned_alpaca --n_instances 3000 --random_state 2021 --ftype length_big
# CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/finetune.py --data_path ../datasets/sampled/cleaned_alpaca/infoverse/length_big/1000/2022/sampled_infoverse_1000.parquet.gzip --sample_type infoverse --data_set cleaned_alpaca --n_instances 1000 --random_state 2022 --ftype length_big
# CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/finetune.py --data_path ../datasets/sampled/cleaned_alpaca/infoverse/length_big/2000/2022/sampled_infoverse_2000.parquet.gzip --sample_type infoverse --data_set cleaned_alpaca --n_instances 2000 --random_state 2022 --ftype length_big
# CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/finetune.py --data_path ../datasets/sampled/cleaned_alpaca/infoverse/length_big/3000/2022/sampled_infoverse_3000.parquet.gzip --sample_type infoverse --data_set cleaned_alpaca --n_instances 3000 --random_state 2022 --ftype length_big
# CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/finetune.py --data_path ../datasets/sampled/cleaned_alpaca/infoverse/length_big/1000/2023/sampled_infoverse_1000.parquet.gzip --sample_type infoverse --data_set cleaned_alpaca --n_instances 1000 --random_state 2023 --ftype length_big
# CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/finetune.py --data_path ../datasets/sampled/cleaned_alpaca/infoverse/length_big/2000/2023/sampled_infoverse_2000.parquet.gzip --sample_type infoverse --data_set cleaned_alpaca --n_instances 2000 --random_state 2023 --ftype length_big
# CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/finetune.py --data_path ../datasets/sampled/cleaned_alpaca/infoverse/length_big/3000/2023/sampled_infoverse_3000.parquet.gzip --sample_type infoverse --data_set cleaned_alpaca --n_instances 3000 --random_state 2023 --ftype length_big
# CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/finetune.py --data_path ../datasets/sampled/cleaned_alpaca/infoverse/rouge/10000/sampled_infoverse_10000.parquet.gzip --sample_type infoverse --data_set cleaned_alpaca --n_instances 10000 --ftype rouge
# CUDA_VISIBLE_DEVICES=1 python ../llama2/scripts/finetune.py --data_path ../datasets/sampled/cleaned_alpaca/infoverse/rouge/20000/sampled_infoverse_20000.parquet.gzip --sample_type infoverse --data_set cleaned_alpaca --n_instances 20000 --ftype rouge

# CUDA_VISIBLE_DEVICES=1 python ../llama2/scripts/finetune.py --data_path ../datasets/sampled/cleaned_alpaca/infoverse/perp/1000/sampled_infoverse_1000.parquet.gzip --sample_type infoverse --data_set cleaned_alpaca --n_instances 1000 --ftype perp
# CUDA_VISIBLE_DEVICES=1 python ../llama2/scripts/finetune.py --data_path ../datasets/sampled/cleaned_alpaca/infoverse/perp/10000/sampled_infoverse_10000.parquet.gzip --sample_type infoverse --data_set cleaned_alpaca --n_instances 10000 --ftype perp
# CUDA_VISIBLE_DEVICES=1 python ../llama2/scripts/finetune.py --data_path ../datasets/sampled/cleaned_alpaca/infoverse/perp/20000/sampled_infoverse_20000.parquet.gzip --sample_type infoverse --data_set cleaned_alpaca --n_instances 20000 --ftype perp

# CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/infer_finetuned.py --sample_type infoverse --data_set cleaned_alpaca --n_instances 1000 --format True --det True --ftype length_big --random_state 2021 &
# CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/infer_finetuned.py --sample_type infoverse --data_set cleaned_alpaca --n_instances 2000 --format True --det True --ftype length_big --random_state 2021 &
# CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/infer_finetuned.py --sample_type infoverse --data_set cleaned_alpaca --n_instances 3000 --format True --det True --ftype length_big --random_state 2021 &
# wait

# CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/infer_finetuned.py --sample_type infoverse --data_set cleaned_alpaca --n_instances 1000 --format True --det True --ftype length_big --random_state 2022 &
# CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/infer_finetuned.py --sample_type infoverse --data_set cleaned_alpaca --n_instances 2000 --format True --det True --ftype length_big --random_state 2022 &
# CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/infer_finetuned.py --sample_type infoverse --data_set cleaned_alpaca --n_instances 3000 --format True --det True --ftype length_big --random_state 2022 &
# wait

# CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/infer_finetuned.py --sample_type infoverse --data_set cleaned_alpaca --n_instances 1000 --format True --det True --ftype length_big --random_state 2023 &
# CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/infer_finetuned.py --sample_type infoverse --data_set cleaned_alpaca --n_instances 2000 --format True --det True --ftype length_big --random_state 2023 &
# CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/infer_finetuned.py --sample_type infoverse --data_set cleaned_alpaca --n_instances 3000 --format True --det True --ftype length_big --random_state 2023 &
# wait

# CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/infer_finetuned.py --sample_type infoverse --data_set cleaned_alpaca --n_instances 1000 --format True --det True --ftype length_small --random_state 2021 &
# CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/infer_finetuned.py --sample_type infoverse --data_set cleaned_alpaca --n_instances 2000 --format True --det True --ftype length_small --random_state 2021 &
# CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/infer_finetuned.py --sample_type infoverse --data_set cleaned_alpaca --n_instances 3000 --format True --det True --ftype length_small --random_state 2021 &
# wait

# CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/infer_finetuned.py --sample_type infoverse --data_set cleaned_alpaca --n_instances 1000 --format True --det True --ftype perp --random_state 2021 &
# CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/infer_finetuned.py --sample_type infoverse --data_set cleaned_alpaca --n_instances 2000 --format True --det True --ftype perp --random_state 2021 &
# CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/infer_finetuned.py --sample_type infoverse --data_set cleaned_alpaca --n_instances 3000 --format True --det True --ftype perp --random_state 2021 &
# wait


# CUDA_VISIBLE_DEVICES=1 python ../llama2/scripts/infer_finetuned.py --sample_type infoverse --data_set cleaned_alpaca --n_instances 10000 --format True --det True --ftype length_big
# CUDA_VISIBLE_DEVICES=1 python ../llama2/scripts/infer_finetuned.py --sample_type infoverse --data_set cleaned_alpaca --n_instances 20000 --format True --det True --ftype length_big
# CUDA_VISIBLE_DEVICES=1 python ../llama2/scripts/infer_finetuned.py --sample_type infoverse --data_set cleaned_alpaca --n_instances 20000 --format True --det True --ftype length_small

# CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/infer_finetuned.py --sample_type infoverse --data_set cleaned_alpaca --n_instances 1000 --format True --det True --ftype rouge
# CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/infer_finetuned.py --sample_type infoverse --data_set cleaned_alpaca --n_instances 10000 --format True --det True --ftype rouge
# CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/infer_finetuned.py --sample_type infoverse --data_set cleaned_alpaca --n_instances 20000 --format True --det True --ftype rouge
# CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/infer_finetuned.py --sample_type infoverse --data_set cleaned_alpaca --n_instances 1000 --format True --det True --ftype perp
# CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/infer_finetuned.py --sample_type infoverse --data_set cleaned_alpaca --n_instances 10000 --format True --det True --ftype perp
# CUDA_VISIBLE_DEVICES=0 python ../llama2/scripts/infer_finetuned.py --sample_type infoverse --data_set cleaned_alpaca --n_instances 20000 --format True --det True --ftype perp


CUDA_VISIBLE_DEVICES=0 python ../scripts/metrics.py --data_set cleaned_alpaca --sample_type infoverse --ftype perp --n_instances 1000
CUDA_VISIBLE_DEVICES=0 python ../scripts/metrics.py --data_set cleaned_alpaca --sample_type infoverse --ftype perp --n_instances 2000
CUDA_VISIBLE_DEVICES=0 python ../scripts/metrics.py --data_set cleaned_alpaca --sample_type infoverse --ftype perp --n_instances 3000

CUDA_VISIBLE_DEVICES=0 python ../scripts/metrics2.py --data_set cleaned_alpaca --sample_type infoverse --ftype length_big --n_instances 1000
CUDA_VISIBLE_DEVICES=0 python ../scripts/metrics2.py --data_set cleaned_alpaca --sample_type infoverse --ftype length_big --n_instances 2000
CUDA_VISIBLE_DEVICES=0 python ../scripts/metrics2.py --data_set cleaned_alpaca --sample_type infoverse --ftype length_big --n_instances 3000

CUDA_VISIBLE_DEVICES=0 python ../scripts/metrics.py --data_set cleaned_alpaca --sample_type infoverse --ftype length_small --n_instances 1000
CUDA_VISIBLE_DEVICES=0 python ../scripts/metrics.py --data_set cleaned_alpaca --sample_type infoverse --ftype length_small --n_instances 2000
CUDA_VISIBLE_DEVICES=0 python ../scripts/metrics.py --data_set cleaned_alpaca --sample_type infoverse --ftype length_small --n_instances 3000

# CUDA_VISIBLE_DEVICES=0 python ../scripts/metrics.py --data_set dolly --sample_type instructor --n_instances 1000
# CUDA_VISIBLE_DEVICES=0 python ../scripts/metrics.py --data_set dolly --sample_type instructor --n_instances 2000
# CUDA_VISIBLE_DEVICES=0 python ../scripts/metrics.py --data_set dolly --sample_type instructor --n_instances 3000