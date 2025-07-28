import pandas as pd
import pyarrow.parquet as pq
import json
import re
from typing import Optional
import argparse

def extract_code(text: str) -> Optional[str]:
    pattern = r"```python\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    pattern = r"```\n?(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    return None

model_rename_map = {
    "Meta-Llama-3.1-8B-Instruct-Q6_K": "Llama-3.1-8B-Instruct-Q6_K",
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "gemini-1.5-flash": "gemini-1.5-flash-002",
    "Llama-3-3-70B-Instruct": "Llama-3.3-70B-Instruct",
    "gpt-4o-batch": "gpt-4o-2024-11-20",
    "gpt-4o": "gpt-4o-2024-11-20",
    "Llama-3-3-70B-Instruct-qnycp": "Llama-3.3-70B-Instruct",
    "Meta-Llama-3.1-8B-Instruct-Q8_0": "Llama-3.1-8B-Instruct-Q8_0"
}

def process_parquet(input_file: str, output_file: str):
    df = pq.read_table(input_file).to_pandas()
    print(f"Loaded {len(df)} rows from parquet file")
    
    originals = []
    rewrittens = []
    model_useds = []
    full_paths = []
    
    failed_success_rows = 0
    no_code_rows = 0
    error_rows = 0
    
    for idx, row in df.iterrows():
        try:
            data_value = row['data']
            if isinstance(data_value, str):
                data = json.loads(data_value)
            else:
                data = data_value
            
            # Skip if success is False
            if not data.get('success', True):
                failed_success_rows += 1
                continue
            
            analysis = data.get('analysis', '')
            original_code = data.get('code', '')
            
            rewritten_code = extract_code(analysis)
            
            # Skip if no code could be extracted
            if rewritten_code is None:
                no_code_rows += 1
                continue
            
            # Renae the model
            model_used = data.get('model_used', 'unknown')
            model_used = model_rename_map.get(model_used, model_used)
            
            full_path = data.get('full_path', '')
            
            originals.append(original_code)
            rewrittens.append(rewritten_code)
            model_useds.append(model_used)
            full_paths.append(full_path)
            
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            error_rows += 1
            continue
    
    new_df = pd.DataFrame({
        'original': originals,
        'rewritten': rewrittens,
        'model_used': model_useds,
        'full_path': full_paths
    })
    
    new_df.to_parquet(output_file, index=False)
    
    print(f"Processed {len(new_df)} valid rows")
    print(f"Skipped {failed_success_rows} rows with success=False")
    print(f"Skipped {no_code_rows} rows with no extractable code")
    print(f"Skipped {error_rows} rows due to errors")
    print(f"Saved processed data to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a parquet file to reorganize the data column')
    parser.add_argument('input_file', help='Path to the input parquet file')
    parser.add_argument('output_file', help='Path to save the processed parquet file')
    
    args = parser.parse_args()
    
    process_parquet(args.input_file, args.output_file)