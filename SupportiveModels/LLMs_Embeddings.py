import os
import openai
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# Set your OpenAI API key
openai.api_key = ""  # Replace with your actual key

# Function to compute cosine similarity
def cosine_similarity_matrix(embeddings1, embeddings2):
    embeddings1 = np.array(embeddings1)
    embeddings2 = np.array(embeddings2)
    dot_products = np.sum(embeddings1 * embeddings2, axis=1)
    norms1 = np.linalg.norm(embeddings1, axis=1)
    norms2 = np.linalg.norm(embeddings2, axis=1)
    return dot_products / (norms1 * norms2)

# Function to embed a batch of strings
def get_embeddings_batch(text_list, model="text-embedding-3-large"):
    response = openai.embeddings.create(
        input=text_list,
        model=model
    )
    return [item.embedding for item in response.data]

# Function to process one CSV file
def load_data(filename, batch_size=100):
    df = pd.read_csv(filename)
    similarities = []

    for i in tqdm(range(0, len(df), batch_size), desc=f"Processing {filename}"):
        batch_df = df.iloc[i:i+batch_size]
        source_codes = batch_df["src_source_code"].tolist()
        target_codes = batch_df["des_source_code"].tolist()

        try:
            src_embeddings = get_embeddings_batch(source_codes)
            des_embeddings = get_embeddings_batch(target_codes)
        except Exception as e:
            print(f"Error at batch {i}: {e}")
            similarities.extend([None] * len(source_codes))
            continue

        batch_similarities = cosine_similarity_matrix(src_embeddings, des_embeddings)
        similarities.extend(batch_similarities)

    df["text_3_L_similarity"] = similarities
    name_without_extension = Path(filename).stem
    df.to_csv(name_without_extension + "_embeddings.csv", index=False)
    print(f"Saved: {filename}")

# Main processing loop
if __name__ == "__main__":
    cloneContent = '../Data/CSVFiles'
    # filenames = ['CSharpPythonFeatures.csv', 'JavaCSharpFeatures.csv', 'JavaPythonFeatures.csv']
    #filenames = ['JavaCSharpNonCloneFeatures.csv', 'JavaPythonNonCloneFeatures.csv', 'CSharpPythonNonCloneFeatures.csv']
    filenames = ['CSharpPythonNonCloneFeatures.csv']

    for filename in filenames:
        full_path = os.path.join(cloneContent, filename)
        print(f" Working on: {filename}")
        load_data(full_path, batch_size=10)
        print(f" Done: {filename}")
