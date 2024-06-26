import os
import argparse
import time
import random
import torch
import numpy as np
import wandb
import pandas as pd
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, EarlyStoppingCallback
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import TrainingArguments
from trl import SFTTrainer


def set_seed(random_state):
    deterministic = True
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)

parser = argparse.ArgumentParser(description="Data path for finetuning")
parser.add_argument('-dp','--data_path', type=str, help="Sampled Dataset Path")
parser.add_argument('-s','--sample_type', type=str, help="Sampling algo used")
parser.add_argument('-d','--data_set', type=str, help="Dataset Name")
parser.add_argument('-n','--n_instances', type=str, help="Sampled instances")
parser.add_argument('-f','--ftype', type=str, help="Type of infoverse single feature (perp/rouge/length)")
parser.add_argument('-r','--random_state', type=int, default=2023, choices=[2023,2022,2021] ,help="Random state for reproducibility.")
parser.add_argument('-lp','--llama_path', type=str, required=True, help="Directory where finetuned HF llama weights are stored")
parser.add_argument('-mp','--model_path', type=str, required=True, help="Directory where original HF llama weights are stored")
parser.add_argument('-l','--local_model', type=str, help="Model for SelectLLM. Options: [gpt3.5, mixtral]")

args = parser.parse_args()

start_time = time.time()

dataset_path = args.data_path
sample_type = args.sample_type
n_instances = args.n_instances
data_set = args.data_set
ftype = args.ftype
random_state = args.random_state
local_selection_model = args.local_model
OUTPUT_DIR = Path(args.llama_path)
MODEL_ID = Path(args.model_path)
set_seed(random_state)

if sample_type == 'infoverse':
    run_name = f'{data_set}_{sample_type}_{ftype}_{n_instances}_rs{random_state}'
    new_output_dir = OUTPUT_DIR.joinpath(data_set, sample_type, ftype, n_instances, str(random_state))
elif sample_type == 'selectllm':
    run_name = f'{local_selection_model}_{data_set}_{sample_type}_{ftype}_{n_instances}_rs{random_state}'
    new_output_dir = OUTPUT_DIR.joinpath(data_set, sample_type, ftype, local_selection_model, n_instances, str(random_state)) 
else:
    run_name = f'{data_set}_{sample_type}_{n_instances}_rs{random_state}'
    new_output_dir = OUTPUT_DIR.joinpath(data_set, sample_type, n_instances, str(random_state))

if not new_output_dir.exists():
    new_output_dir.mkdir(parents=True)

wandb.init(project="SelectLLM_Finetuning", entity="ritikparkar789", name=run_name)
print(dataset_path)
data = pd.read_parquet(dataset_path)
data = data[["instruction", "input", "output"]]

dataset = Dataset.from_pandas(data)
try:
    dataset = dataset.remove_columns(['__index_level_0__'])
except:
    pass

def format_instruction(sample): 
	return f"""### Instruction:
{sample['instruction']}

### Input:
{sample['input']}

### Response:
{sample['output']}
"""


splits = dataset.train_test_split(test_size=0.2)
train_data = splits['train']
val_data = splits['test']
train_data = train_data.map(lambda x: {'text': format_instruction(x)})
val_data = val_data.map(lambda x: {'text': format_instruction(x)})

use_flash_attention = False
# COMMENT IN TO USE FLASH ATTENTION
# replace attention with flash attention
if torch.cuda.get_device_capability()[0] >= 8:
    from utils.llama_patch import replace_attn_with_flash_attn
    print("Using flash attention")
    replace_attn_with_flash_attn()
    use_flash_attention = True

# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(str(MODEL_ID.resolve()), quantization_config=bnb_config, use_cache=False, device_map="auto")
model.config.pretraining_tp = 1

# Validate that the model is using flash attention, by comparing doc strings
if use_flash_attention:
    from utils.llama_patch import forward
    assert model.model.layers[0].self_attn.forward.__doc__ == forward.__doc__, "Model is not using flash attention"


tokenizer = AutoTokenizer.from_pretrained(str(MODEL_ID.resolve()))
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# LoRA config based on QLoRA paper
peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
)

# prepare model for training
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)
from utils.llama_patch import upcast_layer_for_flash_attention
model = upcast_layer_for_flash_attention(model, torch.bfloat16)

wandb.watch(model)

args = TrainingArguments(
    output_dir=new_output_dir,
    num_train_epochs=20,
    max_steps=-1,
    per_device_train_batch_size=6 if use_flash_attention else 4,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    optim='paged_adamw_32bit',
    logging_steps=5,
    save_strategy='epoch',
    report_to = 'wandb',
    learning_rate=2e-4,
    bf16=True,
    tf32=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type='constant',
    evaluation_strategy='epoch',  # Evaluate the model every 'eval_steps'
    load_best_model_at_end=True, 
    disable_tqdm=False # disable tqdm since with packing values are in correct
)

max_seq_length = 4096 # max sequence length for model and packing of the dataset

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset = val_data,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    formatting_func=format_instruction,
    args=args,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.01)]
)

# train
trainer.train() # there will not be a progress bar since tqdm is disabled

# save model
trainer.save_model()

wandb.finish()
end_Time = time.time()

print(f"Total time taking for finetuning:{end_Time-start_time} with flash attn enabled:{use_flash_attention} \n")