import json
import torch
import time
import os
import re
import pandas as pd
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import argparse
from pathlib import Path
import random
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

def set_seed(random_state):
    deterministic = True
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description="Run Inference using the finetuned model")
parser.add_argument('-s','--sample_type', type=str, help="Sampling algo used")
parser.add_argument('-d','--data_set', type=str, help="Dataset Name")
parser.add_argument('-n','--n_instances', type=str, help="Sampled instances")
parser.add_argument('-f','--ftype', type=str, help="Type of infoverse single feature (perp/rouge/length)")
parser.add_argument('-r','--random_state', type=int, default=2023, choices=[2023,2022,2021] ,help="Random state for reproducibility.")
parser.add_argument('--det', type=str2bool, required=True, help='Deterministic Inferences or not')
parser.add_argument('-lp','--llama_path', type=str, required=True, help="Directory where finetuned HF llama weights are stored")
parser.add_argument('-l','--local_model', type=str, help="Model for SelectLLM. Options: [gpt3.5, mixtral]")

args = parser.parse_args()
sample_type = args.sample_type
n_instances = args.n_instances
det = args.det
data_set = args.data_set
ftype = args.ftype
random_state = args.random_state
local_selection_model = args.local_model
set_seed(random_state)

init_time = time.time()

TEST_DIR = os.path.join('../datasets/test/', data_set)
DATA_PATH = os.path.join('../datasets/data/', data_set) 
TEST_DIR = Path(TEST_DIR)
LLAMA_DIR = args.llama_path
if sample_type=='infoverse':
    MODEL_DIR = os.path.join(LLAMA_DIR, data_set, sample_type, ftype, n_instances, str(random_state))
elif sample_type == 'selectllm':
    MODEL_DIR = os.path.join(LLAMA_DIR, data_set, sample_type, ftype, local_selection_model, n_instances, str(random_state)) 
else:
    MODEL_DIR = os.path.join(LLAMA_DIR, data_set, sample_type, n_instances, str(random_state))

if data_set == 'all':
    NEWTOKS = 250
elif data_set=='dolly':
    NEWTOKS = 150#300
elif data_set == 'cleaned_alpaca':
    NEWTOKS = 250#650 
else:
    NEWTOKS = 200


formattype = "_formatted" 
det_str = "_det" if det else ""

if sample_type=='infoverse':
    SAVE_TEST_FILE = f'{sample_type}_{ftype}_{n_instances}{formattype}{det_str}_rs_{random_state}.json' 
elif sample_type == 'selectllm':
    SAVE_TEST_FILE = f'{sample_type}_{ftype}_{local_selection_model}_{n_instances}{formattype}{det_str}_rs_{random_state}.json' 
else:
    SAVE_TEST_FILE = f'{sample_type}_{n_instances}{formattype}{det_str}_rs_{random_state}.json'
TEST_PATH = os.path.join(TEST_DIR, SAVE_TEST_FILE)

def extract_response(text):
    # Regular expression to match '### Response:' and capture everything after it
    match = re.search(r'### Response:\s*(.*)', text, re.DOTALL)
    
    # If match is found, return the captured group; otherwise, return the entire text
    return match.group(1).strip() if match else text.strip()

def format_instruction(sample):
	return f"""### Instruction:
            {sample['instruction']}

            ### Input:
            {sample['input']}

            ### Response:
            """

# Load the JSON file containing the prompts
prompts_path = os.path.join(DATA_PATH, 'prompts.json')
prompts_format_path = os.path.join(DATA_PATH, 'prompt_format.json')

prompts = []

if os.path.exists(prompts_format_path):
    with open(prompts_format_path, 'r') as f:
        prompts = json.load(f)
else:
    sampled_path = os.path.join(DATA_PATH, 'test_data.json')
    with open(sampled_path, "r") as f:
        data_df = pd.read_csv(f)
        data_df[['instruction', 'input', 'output']] = data_df[['instruction', 'input', 'output']].fillna('')
        try:
            for index, row in data_df.iterrows():
                data_dict = {}
                data_dict['instruction'] = row['instruction']
                data_dict['input'] = row['input']
                data_dict['output'] = row['output']
                prompts.append(format_instruction(data_dict))
        except:
            for line in f:
                exam = json.loads(line)
                for x in exam:
                    prompts.append(format_instruction(x))
    with open(prompts_format_path, 'w') as  f:
        json.dump(prompts, f)

print(f"Loaded Model from:{MODEL_DIR} \nPrompts from:{prompts_format_path} \nSaving to :{TEST_PATH} \n \
        Length of prompts:{len(prompts)}")

if torch.cuda.get_device_capability()[0] >= 8:
    from utils.llama_patch import replace_attn_with_flash_attn
    print("Using flash attention")
    replace_attn_with_flash_attn()
    use_flash_attention = True

if use_flash_attention:
    # unpatch flash attention
    from utils.llama_patch import unplace_flash_attn_with_attn
    unplace_flash_attn_with_attn()

# load base LLM model and tokenizer\
model = AutoPeftModelForCausalLM.from_pretrained(
    MODEL_DIR,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    load_in_4bit=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = model.to('cuda')

max_model_token_limit = model.config.max_position_embeddings
# max_model_token_limit = 512

batch_size = 8
outputs = []

for param in model.parameters():
    param.requires_grad = False

model_infers = []
zero_time = time.time()

for i in range(len(prompts)):
    st = time.time()
    input_tokens = tokenizer(prompts[i], return_tensors="pt",  truncation=True, return_attention_mask=True)
    if torch.cuda.is_available():
        input_tokens = {k: v.to('cuda') for k, v in input_tokens.items()}

    if det==True:
        with torch.no_grad():
            output_tokens = model.generate(
                input_ids=input_tokens["input_ids"],
                attention_mask=input_tokens["attention_mask"],
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
                num_return_sequences=1,
                max_new_tokens=NEWTOKS, #Camel dataset ground truth has an average of 530 tokens/dolly: 339/cleanedalpaca: 657
                min_new_tokens = 5
            )
    else:
        with torch.no_grad():
            output_tokens = model.generate(
                # input_ids=input_tokens,
                input_ids=input_tokens["input_ids"],
                attention_mask=input_tokens["attention_mask"],
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
                # temperature=0.9, 
                top_k=10,       
                # top_p=0.9,      
                num_return_sequences=1,
                # eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=NEWTOKS #Camel dataset ground truth has an average of 530 tokens and dolly has 339
            )

    decoded_outputs = tokenizer.batch_decode(output_tokens.detach().cpu().numpy(), skip_special_tokens=False)[0][len(prompts[i]):]
    model_infers.append(decoded_outputs)
    
    del input_tokens, output_tokens
    
    et = time.time()
    elapsed_time = et - st 


TEST_DIR.mkdir(parents=True, exist_ok=True)
with open(TEST_PATH, "w") as f:
    json.dump(model_infers, f)

fin_time = time.time()
final_time = fin_time - init_time
print(f'program execution time is:{final_time}')
