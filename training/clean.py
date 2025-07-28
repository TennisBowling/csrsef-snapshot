import pandas as pd
import json

input_file = "merged_results.parquet"
output_file = "cleaned_dataset.parquet"

df = pd.read_parquet(input_file)

df = df[df["success"] != False]

def extract_from_row(row):
    try:
        data = json.loads(row["data"]) if row["data"] else {}
    except (TypeError, json.JSONDecodeError):
        data = {}

    def get_value(key_upper, key_lower):
        return (
            data.get(key_upper) or
            data.get(key_lower) or
            row.get(key_upper) or
            row.get(key_lower)
        )

    return {
        "Body": get_value("Body", "body"),
        "AiAnswer": get_value("AiAnswer", "aianswer"),
        "Score": get_value("Score", "score"),
        "Title": get_value("Title", "title"),
        "model_used": data.get("model_used", row.get("model_used", "unknown"))
    }

cleaned_data = df.apply(extract_from_row, axis=1)

cleaned_df = pd.DataFrame(cleaned_data.tolist())

key_columns = ["Body", "AiAnswer", "Score", "Title", "model_used"]
cleaned_df.dropna(subset=key_columns, how="all", inplace=True)

model_rename_map = {
    "Meta-Llama-3.1-8B-Instruct-Q6_K": "Llama-3.1-8B-Instruct-Q6_K",
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "gemini-1.5-flash": "gemini-1.5-flash-002",
    "Llama-3-3-70B-Instruct": "Llama-3.3-70B-Instruct",
    "gpt-4o": "gpt-4o-2024-11-20",
    "Llama-3-3-70B-Instruct-qnycp": "Llama-3.3-70B-Instruct",
    "Meta-Llama-3.1-8B-Instruct-Q8_0": "Llama-3.1-8B-Instruct-Q8_0"
}

cleaned_df["model_used"] = cleaned_df["model_used"].replace(model_rename_map)

cleaned_df.rename(columns={"model_used": "ModelUsed"}, inplace=True)

filtered_df = cleaned_df.dropna(subset=['Body', 'AiAnswer', 'Title', 'ModelUsed'])

filtered_df.to_parquet(output_file, index=False)
print(f"Cleaned dataset written to {output_file}")
