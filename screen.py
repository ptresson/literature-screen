import pandas as pd
from transformers import *
import torch
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv('./raw.csv',sep=';')
print(df.head())

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

model.eval()

texts = ["first sentence", "second sentence"]

# Tokenize (padding for batches)
batch = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    out = model(**batch)

# batch dict contains input_ids, attention_mask (and sometimes token_type_ids).
# Pass the attention_mask so padded tokens are ignored!
emb = out.last_hidden_state
print(emb)
