import pandas as pd
import tiktoken
from tqdm import tqdm

enc = tiktoken.encoding_for_model("gpt-4o-mini")  # o200k_base

data = pd.read_parquet("cleaned_dataset.parquet")

prompt_tokens = 0
answer_tokens = 0

for index, row in tqdm(data.iterrows(), total=data.shape[0], desc="Processing rows"):
    prompt = f"""Code solution for:
{row['Body']}
Provide only the code solution in Python, no explanations."""

    prompt_tokens += len(enc.encode(prompt, disallowed_special=()))
    answer_tokens += len(enc.encode(row['AiAnswer'], disallowed_special=()))

print("prompt token", prompt_tokens)
print("answer tokens", answer_tokens)
