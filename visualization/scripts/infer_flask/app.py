from flask import Flask, render_template
import os
import json
import re 
import argparse

app = Flask(__name__)

DATA_ROOT = '/corpora/InstructTune/ait_single/active-instruction-tuning/datasets/test/'
PROMPTS_PATH = '/corpora/InstructTune/ait_single/active-instruction-tuning/datasets/data/'
GROUND_TRUTH_PATH = '/corpora/InstructTune/ait_single/active-instruction-tuning/datasets/data/'

def order_models(model):
    order = {
        "random": 1,
        "coreset": 2,
        "infoverse": 3,
        "oe": 4,
        "rouge":5
    }
    return order.get(model.split("_")[0], 999), model

@app.route('/')
def index():
    return render_template('compare.html')

@app.route('/<data_name>')
def display_data_name(data_name):
    return get_sizes(data_name)

@app.route('/<data_name>/sizes')
def get_sizes(data_name):
    path = os.path.join(DATA_ROOT, data_name)
    
    sizes = set()
    for file in os.listdir(path):
        match = re.search(r"_([0-9]+)_", file)
        if match:
            sizes.add(int(match.group(1)))
    
    print(f"Identified sizes for {data_name}: {sizes}")  # Debug print statement
    return render_template('dataset.html', data_name=data_name, sizes=sorted(sizes))

@app.route('/<data_name>/<int:size>')
def display_data_size(data_name, size):
    # Load prompts and ground truth
    prompt_path = os.path.join(PROMPTS_PATH, data_name, 'test_prompts.json')
    with open(prompt_path, 'r') as f:
        prompts = json.load(f)
        prompts = prompts[:50]
    gts_path = os.path.join(GROUND_TRUTH_PATH, data_name, 'test_gts.json')
    with open(gts_path, 'r') as f:
        ground_truth = json.load(f)
        ground_truth = ground_truth[:50]
    
    # Load model responses based on dataset and size
    path = os.path.join(DATA_ROOT, data_name)
    model_responses = {}
    for file in os.listdir(path):
        # Check for exact size match
        if f"_{size}_" in file or file.endswith(f"_{size}.json"):
            with open(os.path.join(path, file), 'r') as f:
                model_responses[file.split('.')[0]] = json.load(f)[:50]

    ordered_models = sorted(model_responses.keys(), key=order_models)
    model_responses = {model: model_responses[model] for model in ordered_models}

    return render_template('compare.html', prompts=prompts, ground_truth=ground_truth, model_responses=model_responses)


@app.route('/<data_name>/all')
def display_data_all(data_name):
    # Load prompts and ground truth
    prompt_path = os.path.join(PROMPTS_PATH, data_name, 'test_prompts.json')
    with open(prompt_path, 'r') as f:
        prompts = json.load(f)
        prompts = prompts[:50]
    gts_path = os.path.join(GROUND_TRUTH_PATH, data_name, 'test_gts.json')
    with open(gts_path, 'r') as f:
        ground_truth = json.load(f)
        ground_truth = ground_truth[:50]
    
    # Load all model responses based on dataset
    path = os.path.join(DATA_ROOT, data_name)
    model_responses = {}
    for file in os.listdir(path):
        if file.endswith(".json"):
            with open(os.path.join(path, file), 'r') as f:
                model_responses[file.split('.')[0]] = json.load(f)[:50]

    ordered_models = sorted(model_responses.keys(), key=order_models)
    model_responses = {model: model_responses[model] for model in ordered_models}

    return render_template('compare.html', prompts=prompts, ground_truth=ground_truth, model_responses=model_responses)

if __name__ == '__main__':
    app.run(debug=True)
