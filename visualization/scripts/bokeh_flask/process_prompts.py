'''Load the Lamini dataset and sample 1000 points and store them in jsonl files'''

from datasets import load_dataset
import json
import random
import argparse
import os

class ProcessPrompts():
    def __init__(self):
        pass
    
    def lamini(self):

        dataset = load_dataset("MBZUAI/LaMini-instruction")

        data = [x for x in dataset['train']]
        # sampled_dataset = random.sample(data, 1000)
        sampled_indices = random.sample(list(enumerate(data)))

        # dir_name = os.path.dirname(args.output_file)

        curr_dir = os.getcwd()
        meta_data_file = '../../datasets/'+args.model+'/metadata.jsonl'
        output_prompts = '../../datasets/lamini/lamini_prompts.jsonl'

        with open(output_prompts, 'w') as f, open(meta_data_file, 'w') as m:
            for idx in sampled_indices:
                json.dump({'prompt': data[idx]['instruction']}, f)
                f.write("\n")
                json.dump({'idx':idx, 'example':data[idx]}, m)
                m.write("\n")


    def alpaca(self):
        with open(args.input_file, 'r') as input_file:
            # Load all data as a list of dictionaries
            all_data = json.load(input_file)

        # random_samples = random.sample(all_data, 1000)
        sampled_indices = random.sample(list(enumerate(data)))
        # dir_name = os.path.dirname(args.output_file)
        # meta_data_file = os.path.join(dir_name, 'metadata'+args.model+'.jsonl')
        meta_data_file = '../../datasets/'+args.model+'/metadata.jsonl'
        output_prompts = '../../datasets/alpaca/alpaca_prompts.jsonl'
        #Continue here
        
        with open(args.output_prompt_file, 'w') as output_prompt_file, open(meta_data_file, 'w') as m:
            for idx in sampled_indices:
                new_data = {
                    "prompt": all_data[idx]["instruction"] + (" " + all_data[idx]["input"] if all_data[idx]["input"] else "")
                }

                output_prompt_file.write(json.dumps(new_data) + '\n')
                json.dump({'idx':idx, 'example':all_data[idx]}, m)
                m.write("\n")


if __name_=='__main__':
    parser = argparse.ArgumentParser(description='Process a json file.')
    parser.add_argument('-m','--model', type=str, required=True,help='Choose the dataset to process: lamini/alpaca')
    parser.add_argument('-i','--input_file', type=str, required=True, help='The path to the input json file.')
    parser.add_argument('-o','--output_file', type=str, required=True, help='The path to the output prompts jsonl file.')
    args = parser.parse_args()

    processor = ProcessPrompts()
    if args.model='lamini':
        processor.lamini()
    elif args.model="alpaca":
        processor.alpaca()
