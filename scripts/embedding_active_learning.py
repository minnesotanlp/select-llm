import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL_NAME = 'all-mpnet-base-v2'

model = SentenceTransformer(MODEL_NAME, 'cuda')
sentences_df = pd.read_parquet('sampled_1000_sentences_df.parquet.gzip')
sentences = sentences_df['sentences'].to_list()
embeddings = [model.encode(s) for s in sentences]

with open('sampled_1000_embeddings.pickle', 'wb') as f:
    pickle.dump(embeddings, f)