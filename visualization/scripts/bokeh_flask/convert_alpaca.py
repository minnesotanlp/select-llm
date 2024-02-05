import argparse
import json
import random

# Create an argument parser
parser = argparse.ArgumentParser(description='Process a json file.')
parser.add_argument('input_file', type=str, help='The path to the input json file.')
parser.add_argument('output_file', type=str, help='The path to the output jsonl file.')
parser.add_argument('output_data_file', type=str, help='The path to the output data jsonl file.')
parser.add_argument('output_gt_file', type=str, help='The path to the gt' )

# Parse the arguments
args = parser.parse_args()

# Open the input file
with open(args.input_file, 'r') as input_file:
    # Load all data as a list of dictionaries
    all_data = json.load(input_file)

# Choose 1000 random samples from all data
random_samples = random.sample(all_data, 1000)

# Open the output files
with open(args.output_file, 'w') as output_file, open(args.output_data_file, 'w') as output_data_file, open(args.output_gt_file, 'w') as output_gt_file:
    # Iterate through each random sample
    for sample in random_samples:
        # Create a new JSON object with "prompt" as the combination of "instruction" and "input"
        new_data = {
            "prompt": sample["instruction"] + (" " + sample["input"] if sample["input"] else "")
        }

        # Write the new JSON object to the output file
        output_file.write(json.dumps(new_data) + '\n')

        # Write the output to the output data file
        output_gt_file.write(json.dumps(sample["output"]) + '\n')

        # Write the original sample to the output data file
        output_data_file.write(json.dumps(sample) + '\n')

