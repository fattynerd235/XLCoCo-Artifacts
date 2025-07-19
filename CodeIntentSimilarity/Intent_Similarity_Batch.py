import pandas as pd
import asyncio
from openai import AsyncOpenAI
from tqdm import tqdm
import re

# -----------------------------
# 1. OpenAI Client Setup
# -----------------------------
client = AsyncOpenAI(api_key="") # Please provide API key

# -----------------------------
# 2. Intent Similarity Prompt
# -----------------------------
def generate_intent_prompt(src_code, des_code):
    return f"""
# Code Intent Similarity Scoring

You are a code analysis expert. Given two code blocks, your job is to compare their **intent** based on:
- Functionality
- Data flow
- In-code pre-conditions and post-conditions

## Task
Analyze the two code blocks and return a single **numeric similarity score** in the range `0–100`, representing how similar their intent is.

## Output Format (strict)
Respond with **only one line** in the following format:

score: <**numeric similarity score**>
- No explanation, no rationale, and no additional output.

## Code A:
{src_code}

## Code B:
{des_code}
"""

# -----------------------------
# 3. GPT Call for One Pair
# -----------------------------
async def get_intent_score(src_code, des_code):
    prompt = generate_intent_prompt(src_code, des_code)
    try:
        response = await client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are a code analysis expert trained to compute intent similarity between code blocks."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=10
        )
        content = response.choices[0].message.content.strip()
        match = re.search(r"score:\s*(\d+)", content)
        return float(match.group(1)) / 100 if match else None
    except Exception as e:
        print("Error during GPT call:", e)
        return None

# -----------------------------
# 4. Process CSV in Batches
# -----------------------------
async def process_file_async(input_csv, output_csv, batch_size=10):
    df = pd.read_csv(input_csv)
    score_list = []

    for i in tqdm(range(0, len(df), batch_size)):
        tasks = []
        batch = df.iloc[i:i + batch_size]
        for _, row in batch.iterrows():
            tasks.append(get_intent_score(row['src_source_code'], row['des_source_code']))

        results = await asyncio.gather(*tasks)
        score_list.extend(results)

    df['gpt_intent_similarity'] = score_list
    df.to_csv(output_csv, index=False)
    print(f"Output saved to: {output_csv}")

# -----------------------------
# 5. Entry Point
# -----------------------------
if __name__ == "__main__":
    input_path = "./../Data/CSVFiles/JavaCSharpFeatures.csv"
    output_path = "./../Data/CSVFiles/JavaCSharpFeatures.csv"
    asyncio.run(process_file_async(input_path, output_path, batch_size=10))
