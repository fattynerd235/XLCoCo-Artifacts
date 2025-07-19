import os
import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import login
from tqdm import tqdm
from pathlib import Path

def load_graphModel():
    model_path = r"GraphCodeBert\graphcodebert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
    return tokenizer, model

def compute_similarity_batch(source_batch, target_batch, tokenizer, model):
    tokens_src = tokenizer(source_batch, return_tensors='pt', padding=True, truncation=True, max_length=512)
    tokens_des = tokenizer(target_batch, return_tensors='pt', padding=True, truncation=True, max_length=512)

    if torch.cuda.is_available():
        tokens_src = {k: v.to("cuda") for k, v in tokens_src.items()}
        tokens_des = {k: v.to("cuda") for k, v in tokens_des.items()}

    with torch.no_grad():
        outputs_src = model(**tokens_src)
        outputs_des = model(**tokens_des)
        emb_src = outputs_src.last_hidden_state[:, 0, :]
        emb_des = outputs_des.last_hidden_state[:, 0, :]
        sim_scores = F.cosine_similarity(emb_src, emb_des).cpu().numpy()

    return sim_scores

def dataload(filename, batch_size=32):
    tokenizer, model = load_graphModel()
    df = pd.read_csv(filename)
    similarities = []

    for i in tqdm(range(0, len(df), batch_size), desc=f"Processing {filename}"):
        batch_df = df.iloc[i:i+batch_size]
        source_batch = batch_df["src_source_code"].tolist()
        target_batch = batch_df["des_source_code"].tolist()

        batch_similarities = compute_similarity_batch(source_batch, target_batch, tokenizer, model)
        similarities.extend(batch_similarities)

    df["GraphCodeBert_Similarity"] = similarities
    name_without_extension = Path(filename).stem
    df.to_csv(name_without_extension + "_bert.csv", index=False)
    df.to_csv(filename, index=False)

def main():
    cloneContent = '../Data/CSVFiles'
    # filenames = ['JavaCSharpFeatures.csv', 'JavaPythonFeatures.csv', 'CSharpPythonFeatures.csv']
    filenames = ['JavaCSharpNonCloneFeatures.csv', 'JavaPythonNonCloneFeatures.csv', 'CSharpPythonNonCloneFeatures.csv']

    for filename in filenames:
        full_path = os.path.join(cloneContent, filename)
        print(f"Working on {filename}")
        dataload(full_path)
        print(f"Done with {filename}")

if __name__ == '__main__':
    main()
