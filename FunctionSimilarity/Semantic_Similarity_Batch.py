import pandas as pd
import asyncio
from openai import AsyncOpenAI
from tqdm import tqdm
import re

# OpenAI client setup
client = AsyncOpenAI(api_key="")# Please provide API key


# Prompt Template
def generate_prompt(src_code, des_code):
    return f"""# Code Semantic and Syntax Similarity Scoring

You are a code analysis expert. Given two code blocks, your job is to compare their **similarity** based on:
- Code Semantics
- Code Functionality

## Task
Analyze the two code blocks and return a single **numeric similarity score** in the range `0–100`, representing how 
similar they are.

## Output Format (strict)
Respond with **only one line** in the following format:

score: <**numeric similarity score**>
- No explanation, no rationale, and no additional output.

## Code A:
{src_code}

## Code B:
{des_code}
"""


# Extract score from GPT response
def parse_score(response_content):
    match = re.search(r"score:\s*(\d+)", response_content)
    return float(match.group(1)) / 100 if match else None


# Asynchronous GPT call for one pair
async def get_score(src_code, des_code):
    prompt = generate_prompt(src_code, des_code)
    try:
        response = await client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system",
                 "content": "You are a code analysis expert trained to compute semantic and syntax similarity between code blocks."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=10
        )
        return parse_score(response.choices[0].message.content)
    except Exception as e:
        print("Error:", e)
        return None


# Batch processor with concurrency
async def process_file_async(input_file, output_file, batch_size=10):
    df = pd.read_csv(input_file)
    scores = []

    # Create batches
    for i in tqdm(range(0, len(df), batch_size)):
        tasks = []
        batch = df.iloc[i:i + batch_size]
        for _, row in batch.iterrows():
            tasks.append(get_score(row['src_source_code'], row['des_source_code']))

        batch_scores = await asyncio.gather(*tasks)
        scores.extend(batch_scores)

    df['gpt_semantic_similarity'] = scores
    df.to_csv(output_file, index=False)
    print(f"Saved results to {output_file}")


# Entrypoint
if __name__ == "__main__":
    input_csv = "./../Data/CSVFiles/JavaPythonFeatures.csv"
    output_csv = "./../Data/CSVFiles/JavaPythonFeatures.csv"

    asyncio.run(process_file_async(input_csv, output_csv, batch_size=10))
