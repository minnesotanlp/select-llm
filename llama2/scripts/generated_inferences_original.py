import json
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline, LlamaForCausalLM, LlamaTokenizer
import torch
import time
import os
import argparse

init_time = time.time()

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Inferences')
parser.add_argument('--data_set', type=str, help='type of data')
parser.add_argument('--sample',default=False, type=str2bool, required=True, help='Sample more than 1 inference')
parser.add_argument('--sample_iter', type=int, help='No. of inferences to sample')
args = parser.parse_args()

data_set = args.data_set
sample_iter = args.sample_iter
sample = args.sample
# Load the JSON file containing the prompts
PROMPT_PATH = os.path.join("/corpora/InstructTune/ait_single/active-instruction-tuning/datasets/data/",data_set)
PROMPT_FILE = os.path.join(PROMPT_PATH, 'prompts.json')
if sample == True:
    SAVE_FILE = os.path.join(PROMPT_PATH, f'preds_{sample_iter}.json')
else:
    SAVE_FILE = os.path.join(PROMPT_PATH, 'preds.json')


prompts = []

with open(PROMPT_FILE, 'r') as f:
    prompts = json.load(f)
        
# Initialize the model and tokenizer
model_path = "/corpora/llama/llama_2_hf/llama-2-7b/7bweights"
tokenizer = LlamaTokenizer.from_pretrained(model_path, pad_token_id=2, padding_side='left')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = LlamaForCausalLM.from_pretrained(model_path)

model = model.to('cuda')

# max_model_token_limit = model.config.max_position_embeddings
max_model_token_limit = 512

# Batch processing
batch_size = 4
outputs = []
# # max_token_length = max([len(tokenizer.encode(prompt)) for prompt in prompts])
# total_tokens = sum([len(tokenizer.encode(prompt)) for prompt in prompts])
# # print(f"Maximum token length in 140k prompts: {max_token_length}") #11225
# avg_token_length = total_tokens / len(prompts)
# print(f"Average token length in 140k prompts: {avg_token_length}") #
for param in model.parameters():
    param.requires_grad = False

model_infers = []
zero_time = time.time()
start_ind = 0

if os.path.exists('./sampled_inference_140k.json'):
    with open("sampled_inference_140k.json", "r") as f:
        model_infers = json.load(f)
        start_ind = len(model_infers)
        print("Loaded samples upto:", start_ind)
        print(len(prompts))
for i in range(start_ind, len(prompts), batch_size):
    st = time.time()
    # batch = [prompt["instruction"] for prompt in prompts[i:i+batch_size]]
    batch = prompts[i:i+batch_size]
    # Tokenize the batch
    input_tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length = max_model_token_limit)
    if torch.cuda.is_available():
        input_tokens = {k: v.to('cuda') for k, v in input_tokens.items()}

    if sample == True:
        # Generate output tokens
        sample_dict = {}
        with torch.no_grad():
            output_tokens = model.generate(
                input_ids=input_tokens["input_ids"],
                attention_mask=input_tokens["attention_mask"],
                do_sample=True, 
                top_k=10,
                top_p=0.9,  
                temperature=1.0,
                num_return_sequences=sample_iter,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens = 100)  

        start_indices = input_tokens["input_ids"].shape[1]
        decoded_outputs = [tokenizer.decode(output[start_indices:], skip_special_tokens=True) for output in output_tokens]

        # Update the prompts
        for k in range(0, len(decoded_outputs), sample_iter):
            sample_dict = {}
            sample_dict[batch[k//3]] = [output for output in decoded_outputs[k:k+sample_iter]]
            model_infers.append(sample_dict)
            
        del input_tokens
        del output_tokens
        torch.cuda.empty_cache()
        
        et = time.time()

        elapsed_time = et - st 
        if i < 64:
            print(f'for iter:{i} elapsed time is:{elapsed_time}')
        if i % 10 == 0:
            print(f'iter no.:{i} time taken:{et - zero_time}')
            with open(SAVE_FILE, "w") as f:
                json.dump(model_infers, f)

    else:
        # Generate output tokens
        with torch.no_grad():
            output_tokens = model.generate(
                input_ids=input_tokens["input_ids"],
                attention_mask=input_tokens["attention_mask"],
                do_sample=False, 
                # top_k = 10,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens = 120)  # Change according to dataset ground truth avg size

        start_indices = input_tokens["input_ids"].shape[1]
        decoded_outputs = [tokenizer.decode(output[start_indices:], skip_special_tokens=True) for output in output_tokens]

        # Update the prompts
        for k, generated_output in enumerate(decoded_outputs):
            model_infers.append(generated_output)
        
        del input_tokens
        del output_tokens
        torch.cuda.empty_cache()
        
        et = time.time()

        elapsed_time = et - st 
        if i < 64:
            print(f'for iter:{i} elapsed time is:{elapsed_time}')
        if i % 7000 == 0:
            print(f'iter no.:{i} time taken:{et - zero_time}')
            with open(SAVE_FILE, "w") as f:
                json.dump(model_infers, f)

# Save the updated JSON object back to a file
with open(SAVE_FILE, "w") as f:
    json.dump(model_infers, f)

fin_time = time.time()
final_time = fin_time - init_time
print(f'program execution time is:{final_time}')
