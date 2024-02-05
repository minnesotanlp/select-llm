
import json
from pathlib import Path
from datasets import load_dataset
from jsonargparse import CLI

DATASETS_BASE_PATH = Path('datasets/raw')

DATASETS = {
    'cleaned_alpaca': {
        'path': 'yahma/alpaca-cleaned',
    },
    'lamini': {
        'path': 'MBZUAI/LaMini-instruction',
    },
    'hc3': {
        'path': 'Hello-SimpleAI/HC3',
        'name': 'all',
    },
    'prosocial_dialog': {
        'path': 'allenai/prosocial-dialog',
        'name': 'en',
    },
    'xp3': {
        'path': 'bigscience/xP3',
        'name': 'en',
    },
    'gpt4all': {
        'path': 'nomic-ai/gpt4all_prompt_generations',
    },
    # 'ultra_chat': {
    #     'path': 'stingning/ultrachat',
    # }
    'dolly': {
        'path': 'databricks/databricks-dolly-15k',
    },
    'open_assistant': {
        'path': 'OpenAssistant/oasst1',
    },
    'flan2021': {
        'path': 'conceptofmind/flan2021_submix_original',
    }
    # 'stack_exchange_preferences': {
    #     'path': 'HuggingFaceH4/stack-exchange-preferences',
    # },
    # 'hh_rlhf': {
    #     'path': 'Anthropic/hh-rlhf',
    # },
    # 'stanford_human_preferences': {
    #     'path': 'stanfordnlp/SHP',
    # },
}


def main(
    
):
    """
    """

    for dataset in DATASETS:
        
        if dataset+'.parquet.gzip' in [p.name for p in DATASETS_BASE_PATH.iterdir()]:
            print(f'{dataset}.parquet.gzip already exists.')
            continue
        
        tmp_df = load_dataset(**DATASETS[dataset], split='train').to_pandas()
        
        tmp_df.to_parquet(DATASETS_BASE_PATH.joinpath(f'{dataset}.parquet.gzip'), compression='gzip')
        print(f'save {dataset}.parquet.gzip')
        
    

if __name__ == '__main__':
    
    CLI(main)