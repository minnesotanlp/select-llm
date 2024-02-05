
import json, gc
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from jsonargparse import CLI

RAW_PATH = Path('datasets/raw')
DEDUP_PATH = Path('datasets/dedup')

DATASETS = {
    'alpaca': {
        'raw_path': RAW_PATH.joinpath('alpaca_data.json'),
        'dedup_path': DEDUP_PATH.joinpath('alpaca_data_dedup.parquet.gzip'),
    },
    'cleaned_alpaca': {
        'raw_path': RAW_PATH.joinpath('cleaned_alpaca.parquet.gzip'),
        'dedup_path': DEDUP_PATH.joinpath('cleaned_alpaca_dedup.parquet.gzip'),    
    },
    'chat_alpaca': {
        'raw_path': RAW_PATH.joinpath('chat_alpaca.json'),
        'dedup_path': DEDUP_PATH.joinpath('chat_alpaca_dedup.parquet.gzip'),
    },
    # 'instruction_wild': {
    #     'dedup_path': '',
    # },
    'flan2021': {
        'raw_path': RAW_PATH.joinpath('flan2021.parquet.gzip'),
        'dedup_path': DEDUP_PATH.joinpath('flan2021_dedup.parquet.gzip'),
        'column_match': {
            'inputs': 'instruction',
            'targets': 'output',
        },
    },
    'super_natural_instructions': {
        'raw_path': RAW_PATH.joinpath('super_natural_instructions.json'),
        'dedup_path': DEDUP_PATH.joinpath('super_natural_instructions_dedup.parquet.gzip'),
    },
    'lamini': {
        'raw_path': RAW_PATH.joinpath('lamini.parquet.gzip'),
        'dedup_path': DEDUP_PATH.joinpath('lamini_dedup.parquet.gzip'),
        'column_match': {
            'response': 'output',
        },
    },
    'hc3': {
        'raw_path': RAW_PATH.joinpath('hc3.parquet.gzip'),
        'dedup_path': DEDUP_PATH.joinpath('hc3_dedup.parquet.gzip'),
        'column_match': {
            'question': 'instruction',
            'chatgpt_answers': 'output', # another option: human_asnwers
        },
    },
    'prosocial_dialog': {
        'raw_path': RAW_PATH.joinpath('prosocial_dialog.parquet.gzip'),
        'dedup_path': DEDUP_PATH.joinpath('prosocial_dialog_dedup.parquet.gzip'),
        'column_match': {
            'context': 'instruction',
            'response': 'output',
        },
    },
    'xp3': {
        'raw_path': RAW_PATH.joinpath('xp3.parquet.gzip'),
        'dedup_path': DEDUP_PATH.joinpath('xp3_dedup.pickle'),
        'column_match': {
            'inputs': 'instruction',
            'targets': 'output',
        },
    },
    'gpt4all': {
        'raw_path': RAW_PATH.joinpath('gpt4all.parquet.gzip'),
        'dedup_path': DEDUP_PATH.joinpath('gpt4all_dedup.parquet.gzip'),
        'column_match': {
            'prompt': 'instruction',
            'response': 'output',
        },
    },
    'gpt_4_llm': {
        'raw_path': RAW_PATH.joinpath('alpaca_gpt4_data.json'),
        'dedup_path': DEDUP_PATH.joinpath('alpaca_gpt4_data_dedup.parquet.gzip'),
    },
    'dolly': {
        'raw_path': RAW_PATH.joinpath('dolly.parquet.gzip'),
        'dedup_path': DEDUP_PATH.joinpath('dolly_dedup.parquet.gzip'),
        'column_match': {
            'context': 'input',
            'response': 'output',
        },
    },
    'open_assistant': {
        'raw_path': RAW_PATH.joinpath('open_assistant.json'),
        'dedup_path': DEDUP_PATH.joinpath('open_assistant_dedup.parquet.gzip'),
    },
    'unnatural_instructions': {
        'raw_path': RAW_PATH.joinpath('unnatural_instructions.json'),
        'dedup_path': DEDUP_PATH.joinpath('unnatural_instructions_dedup.parquet.gzip'),
    },
}


def load_data(df_type: str, drop_duplicates: bool, save: bool):

    def load_by_file_type(file_path: Path):
        if '.json' in file_path.name:
            tmp = pd.read_json(file_path)
        elif '.parquet.gzip' in file_path.name:
            tmp = pd.read_parquet(file_path)
        elif '.pickle' in file_path.name:
            tmp = pd.read_pickle(file_path)    
        return tmp
    
    concat_list = []
    for dataset_name, dataset_info in DATASETS.items():
        print(f'load {dataset_name}')
        
        if df_type == 'raw':
            file_path = dataset_info['raw_path']
        elif df_type == 'dedup':
            file_path = dataset_info['dedup_path']
            
        tmp = load_by_file_type(file_path)
    
        if dataset_info.get('column_match', None):
            tmp.rename(columns=dataset_info['column_match'], inplace=True)
        
        tmp['source'] = dataset_name
        
        add_column_list = [c for c in ['instruction', 'input', 'output'] if c not in tmp.columns]
        for column in add_column_list:
            tmp[column] = ''
            
        tmp = tmp[['instruction', 'input', 'output', 'source']]
        tmp.fillna(value='', inplace=True)
        tmp = tmp.applymap(str)
        print(f'{dataset_name} shape: {tmp.shape}')
        
        if drop_duplicates:
            tmp.drop_duplicates(inplace=True, ignore_index=True)
            print(f'after dropping duplicates, {dataset_name} shape: {tmp.shape}')

            # save dataframes
            if save:
                if dataset_name == 'xp3':
                    save_file_name = file_path.name.replace('.parquet.gzip', '_dedup.pickle')
                    tmp.to_pickle(DEDUP_PATH.joinpath(save_file_name))
                elif '.json' in file_path.name:
                    save_file_name = file_path.name.replace('.json', '_dedup.parquet.gzip')
                elif '.parquet.gzip' in file_path.name:
                    save_file_name = file_path.name.replace('.parquet.gzip', '_dedup.parquet.gzip')
                tmp.to_parquet(DEDUP_PATH.joinpath(save_file_name), compression='gzip')
    
        concat_list.append(tmp.copy())
        
    total_df = pd.concat(concat_list, ignore_index=True)
    
    return total_df
    

def drop_duplicates():
    for dataset_name, dataset_info in DATASETS.items():
        print(f'load {dataset_name}')
        
        file_name = dataset_info['path'].name
        if '.json' in file_name:
            tmp = pd.read_json(dataset_info['path'])
        elif '.parquet.gzip' in file_name:
            tmp = pd.read_parquet(dataset_info['path'])
    
        if dataset_info.get('column_match', None):
            tmp.rename(columns=dataset_info['column_match'], inplace=True)
        
        tmp['source'] = dataset_name
        
        add_column_list = [c for c in ['instruction', 'input', 'output'] if c not in tmp.columns]
        for column in add_column_list:
            tmp[column] = ''
            
        tmp = tmp[['instruction', 'input', 'output', 'source']]
        tmp.fillna(value='', inplace=True)
        tmp = tmp.applymap(str)
        print(f'{dataset_name} shape: {tmp.shape}')
        tmp.drop_duplicates(inplace=True, ignore_index=True)
        print(f'{dataset_name} shape: {tmp.shape}')

        # save the dataframe
        if '.json' in file_name:
            save_file_name = file_name.replace('.json', '_dedup.parquet.gzip')
        elif '.parquet.gzip' in file_name:
            save_file_name = file_name.replace('.parquet.gzip', '_dedup.parquet.gzip')
            # save_file_name = file_name.replace('.parquet.gzip', '.pickle')
            
        tmp.to_parquet(DEDUP_PATH.joinpath(save_file_name), compression='gzip')
        # tmp.to_pickle(DEDUP_PATH.joinpath(save_file_name))
        del tmp
        
def groupby_sentences():
    
    concat_list = []
    for dataset_name, dataset_info in DATASETS.items():
        if dataset_name == 'xp3':
            continue
        print(f'load {dataset_name}')
        
        file_name = dataset_info['path'].name
        if '.parquet.gzip' in file_name:
            tmp = pd.read_parquet(dataset_info['path'])
        elif '.pickle' in file_name:
            tmp = pd.read_pickle(dataset_info['path'])
    
        if dataset_info.get('column_match', None):
            tmp.rename(columns=dataset_info['column_match'], inplace=True)
        
        tmp['source'] = dataset_name
        
        add_column_list = [c for c in ['instruction', 'input', 'output'] if c not in tmp.columns]
        for column in add_column_list:
            tmp[column] = ''
            
        tmp = tmp[['instruction', 'input', 'output', 'source']]
        tmp.fillna(value='', inplace=True)
        print(f'{dataset_name} shape: {tmp.shape}')
    
        concat_list.append(tmp.copy())
        del tmp
        gc.collect()
        
    total_df = pd.concat(concat_list, ignore_index=True)
    del concat_list
    gc.collect()
    
    total_df['sentences'] = total_df['instruction'] + total_df['input'] + total_df['output']
    grouped = total_df.groupby(by='sentences')['source'].apply(list)
    
    grouped.to_pickle('datasets/grouped/grouped_by_sentences.pickle')
    print('Saved the grouped data.')


def test():
    
    # concat_list = []
    for dataset_name, dataset_info in DATASETS.items():
        if dataset_name not in ['dolly', 'gpt4all', 'gpt_4_llm', 'open_assistant', 'unnatural_instructions']:
            continue
        
        print(f'load {dataset_name}')
        
        file_name = dataset_info['path'].name
        if '.json' in file_name:
            tmp = pd.read_json(dataset_info['path'])
        elif '.parquet.gzip' in file_name:
            tmp = pd.read_parquet(dataset_info['path'])
    
        if dataset_info.get('column_match', None):
            tmp.rename(columns=dataset_info['column_match'], inplace=True)
        
        tmp['source'] = dataset_name
        
        add_column_list = [c for c in ['instruction', 'input', 'output'] if c not in tmp.columns]
        for column in add_column_list:
            tmp[column] = ''
            
        tmp = tmp[['instruction', 'input', 'output', 'source']]
        tmp.fillna(value='', inplace=True)
        tmp = tmp.applymap(str)
        print(f'{dataset_name} shape: {tmp.shape}')
        tmp.drop_duplicates(inplace=True, ignore_index=True)
        print(f'{dataset_name} shape: {tmp.shape}')
    
        # concat_list.append(tmp.copy())
        
    # total_df = pd.concat(concat_list, ignore_index=True)
    
    # return total_df

def main(
    df_type: str = 'dedup',
    drop_duplicates: bool = False,
    save: bool = False,
):
    assert df_type in ['raw', 'dedup']
    
    total_df = load_data(df_type: str)
    
    print(f'before dropping duplicates')
    print(f'total shape: {total_df.shape}')
    grouped = total_df.groupby('source').count()
    print(grouped)
    
    total_df = total_df.applymap(str)
    total_df.drop_duplicates(inplace=True)
    print(f'after dropping duplicates')
    print(f'total shape: {total_df.shape}')
    grouped = total_df.groupby('source').count()
    print(grouped)

    for i in range(0, len(total_df), 10000000):
        total_df.iloc[i:(i+10000000)].to_parquet(f'datasets/total/total_{i}_{i+10000000}.parquet.gzip', compression='gzip')
        print(f'total_df from {i} to {i+10000000} has been saved')


if __name__ == '__main__':
    # groupby_sentences()
    # drop_duplicates()
    CLI(main)