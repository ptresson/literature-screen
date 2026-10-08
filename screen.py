import pandas as pd
from transformers import *
import torch
from torch.nn import CosineSimilarity
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv('./raw.csv',sep=';')
print(df.head())
df = df[:10]

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

model.eval()

embeddings = []
for idx, row in df.iterrows():
    text = 'TITLE:' + str(row['Article Title']) + ' ABSTRACT:' + str(row['Abstract'])
    texts = [text]
    batch = tokenizer(
            texts, 
            truncation=True, 
            return_tensors="pt",
            padding='max_length', 
            max_length=512
            )

    with torch.no_grad():
        out = model(**batch)
        emb = out.last_hidden_state
    embeddings.append(emb.squeeze().flatten())


template = embeddings[2]
embeddings = torch.stack(embeddings, dim=0)
cos = CosineSimilarity(dim=1)
sim = cos(template, embeddings)
print(sim.shape)
print(sim)
