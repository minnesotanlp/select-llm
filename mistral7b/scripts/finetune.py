import os
import argparse
import time
import random
import torch
import numpy as np
import wandb
import pandas as pd
from pathlib import Path
from datasets import Dataset, load_dataset
from transformers import EarlyStoppingCallback
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, PeftConfig, PeftModel
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments



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
parser.add_argument('-mp','--mistral_path', type=str, required=True, help="Directory where finetuned HF mistral weights are stored")
parser.add_argument('-l','--local_model', type=str, help="Model for SelectLLM. Options: [gpt3.5, mixtral]")
parser.add_argument('-fm','--finetune_model', type=str, help="Model to finetune: [llama, mistral]")

args = parser.parse_args()

start_time = time.time()

dataset_path = args.data_path
sample_type = args.sample_type
n_instances = args.n_instances
data_set = args.data_set
ftype = args.ftype
random_state = args.random_state
local_selection_model = args.local_model
finetune_model = args.finetune_model
OUTPUT_DIR = Path(args.mistral_path)
MODEL_ID = "unsloth/mistral-7b-bnb-4bit" if finetune_model=='mistral' else "unsloth/llama-2-7b-bnb-4bit"
set_seed(random_state)

if sample_type == 'infoverse':
    run_name = f'{finetune_model}_{data_set}_{sample_type}_{ftype}_{n_instances}_rs{random_state}_ES=0'
    new_output_dir = OUTPUT_DIR.joinpath(data_set, finetune_model, sample_type, ftype, n_instances, str(random_state))
elif sample_type == 'selectllm':
    run_name = f'{finetune_model}_{local_selection_model}_{data_set}_{sample_type}_{ftype}_{n_instances}_rs{random_state}'
    new_output_dir = OUTPUT_DIR.joinpath(data_set, finetune_model, sample_type, ftype, local_selection_model, n_instances, str(random_state)) 
else:
    run_name = f'{finetune_model}_{data_set}_{sample_type}_{n_instances}_rs{random_state}_ES=0'
    new_output_dir = OUTPUT_DIR.joinpath(data_set, finetune_model, sample_type, n_instances, str(random_state))

if not new_output_dir.exists():
    new_output_dir.mkdir(parents=True)

wandb.init(project="SelectLLM_Finetuning", entity="ritikparkar789", name=run_name)

data = pd.read_parquet(dataset_path)
data = data[["instruction", "input", "output"]]

dataset = Dataset.from_pandas(data)
try:
    dataset = dataset.remove_columns(['__index_level_0__'])
except:
    pass

#Only processes one example at a time. 
def format_instruction(sample, eostoken):
	return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    
### Instruction:
{sample['instruction']}

### Input:
{sample['input']}

### Response:
{sample['output']}
""" + eostoken

def format_instructions_batch(examples, eostoken):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:

{instruction}

### Input:

{input}

### Response:

{output}

""" + eostoken
        texts.append(text)
    
    return {"text": texts}

dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
max_seq_length = 2048

# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(MODEL_ID, dtype = dtype, load_in_4bit = True,max_seq_length = max_seq_length)

# LoRA config based on QLoRA paper
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = True,
    random_state = random_state,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

EOS_TOKEN = tokenizer.eos_token

splits = dataset.train_test_split(test_size=0.2)
train_data = splits['train']
val_data = splits['test']

#Formatting each instruction as a batch
train_data = train_data.map(format_instructions_batch, batched = True, fn_kwargs={'eostoken': EOS_TOKEN})
val_data = val_data.map(format_instructions_batch, batched = True, fn_kwargs={'eostoken': EOS_TOKEN})
#Formatting each instruction one by one
# train_data = train_data.map(lambda x: {'text': format_instruction(x, EOS_TOKEN)})
# val_data = val_data.map(lambda x: {'text': format_instruction(x, EOS_TOKEN)})

wandb.watch(model)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset=train_data,
    eval_dataset = val_data,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.01)],
    args = TrainingArguments(
        per_device_train_batch_size = 10,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 600,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 20, #Every 20 steps earlystopping threshold will be evaluated
        optim = "adamw_8bit",
        report_to = 'wandb',
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = random_state,
        output_dir = new_output_dir,
        evaluation_strategy = 'steps',
        load_best_model_at_end = True
    ),
)


# train
trainer.train() # there will not be a progress bar since tqdm is disabled

# save model
trainer.save_model()

wandb.finish()
end_Time = time.time()

print(f"Total time taking for finetuning:{end_Time-start_time}\n")