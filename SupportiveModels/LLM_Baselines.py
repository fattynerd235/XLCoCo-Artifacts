# Simple Prompt, Reasoning and Separate Explanation

import pandas as pd
import asyncio
from openai import AsyncOpenAI
from tqdm import tqdm
import re
import os
from pathlib import Path


client = AsyncOpenAI(api_key="")  # Please use API Key

# ----------------------
# Merged Prompt Function
# ----------------------
def getPrompt_for_similarity(src_code, des_code):
    return f"""
# Dual Code Similarity Evaluation

You are a code analysis expert. Given two code blocks, your job is to evaluate their similarity **thrice**, using three different scoring standards:

---

### Prompt A – Simple Prompt
Consider the overall structure and logic of the following two codes and determine similarity between the two codes

**Output Format (strict)**:
Respond with exactly one line for this evaluation in the following format:

score_A: <numeric similarity score in range 0–100>

---

### Prompt B – Reasoning Similarity
Perform a detailed reasoning process for detecting code clones in the following two code snippets, regardless of the programming language. 
Based on your analysis, respond with a similarity score between the two codes

**Output Format (strict)**:
Respond with exactly one line for this evaluation in the following format:

score_B: <numeric similarity score in range 0–100>

---

### Prompt C – Separate Explanation Similarity
Analyze the following two code snippets and determine the code similarity based on the Similarity/Reasoning/Difference 
Integrated information. Based on your analysis, respond with a similarity score between the two codes 

**Output Format (strict)**:
Respond with exactly one line for this evaluation in the following format:

score_C: <numeric similarity score in range 0–100>

---

### Code A:
{src_code}

### Code B:
{des_code}

Return only:
- The `score_A` line
- Then the `score_B` line
- Then the `score_C` line
- No explanation, no additional output.
"""

# -----------------------------
# Async function to get scores
# -----------------------------
async def get_dual_score(src_code, des_code):
    prompt = getPrompt_for_similarity(src_code, des_code)
    try:
        response = await client.chat.completions.create(
            model="gpt-4.1-mini",
            #model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are a code analysis expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=15  # just enough for both lines
        )
        content = response.choices[0].message.content.strip()

        # Extract scores using regex
        match_A = re.search(r"score_A:\s*(\d+)", content)
        match_B = re.search(r"score_B:\s*(\d+)", content)
        match_C = re.search(r"score_C:\s*(\d+)", content)

        score_A = float(match_A.group(1)) / 100 if match_A else None
        score_B = float(match_B.group(1)) / 100 if match_B else None
        score_C = float(match_C.group(1)) / 100 if match_C else None

        return score_A, score_B, score_C

    except Exception as e:
        print("Error during GPT call:", e)
        return None, None

# ---------------------------------------------
# Async batch processor to run the whole CSV
# ---------------------------------------------
async def process_file_async(input_csv, output_csv, batch_size=10):
    df = pd.read_csv(input_csv)
    score_A_list = []
    score_B_list = []
    score_C_list = []

    for i in tqdm(range(0, len(df), batch_size)):
        tasks = []
        batch = df.iloc[i:i + batch_size]
        for _, row in batch.iterrows():
            tasks.append(get_dual_score(row['src_source_code'], row['des_source_code']))
            #tasks.append(get_dual_score(row['src_code'], row['des_code']))

        results = await asyncio.gather(*tasks)

        for score_A, score_B, score_C in results:
            score_A_list.append(score_A)
            score_B_list.append(score_B)
            score_C_list.append(score_C)

    # Save results
    df['gpt_simple_similarity'] = score_A_list
    df['gpt_reasoning_similarity'] = score_B_list
    df['gpt_seperate_explanation_similarity'] = score_B_list
    df.to_csv(output_csv, index=False)
    print(f"Saved scored output to: {output_csv}")

if __name__ == "__main__":
    cloneContent = '../Data/CSVFiles'
    #filenames = ['JavaCSharpFeatures.csv', 'JavaPythonFeatures.csv', 'CSharpPythonFeatures.csv']
    filenames = ['JavaPythonNonCloneFeatures.csv']

    for filename in filenames:
        print(f'Working with file {filename}')
        input_path = os.path.join(cloneContent, filename)
        name_without_ext = Path(input_path).stem
        output_path = name_without_ext + "_baselines.csv"
        asyncio.run(process_file_async(input_path, output_path, batch_size=10))
        #list_available_models()