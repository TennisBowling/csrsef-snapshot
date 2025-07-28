import pandas as pd
import json
from typing import Dict, Optional
import re
import ast

def extract_from_json(json_str: str) -> Dict[str, Optional[str]]:
    """Extract required fields from JSON string."""
    try:
        data = json.loads(json_str)
        return {
            'AiAnswer': data.get('AiAnswer'),
            'model_used': data.get('model_used'),
            'Score': data.get('Score'),
            'Title': data.get('Title'),
            'Body': data.get('Body')
        }
    except (json.JSONDecodeError, TypeError):
        return {
            'AiAnswer': None,
            'model_used': None,
            'Score': None,
            'Title': None,
            'Body': None
        }

def extract_code(text: str) -> Optional[str]:
    """Extract code from between triple backticks."""
    # Try Python-specific code blocks first
    pattern = r"```python\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    # Try generic code blocks
    pattern = r"```\n?(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    return None

def validate_code(code: str) -> bool:
    """Validate if the code is parseable Python."""
    try:
        ast.parse(code)
        return True
    except (SyntaxError, Exception):
        return False

def process_row(row):
    """Process a single row, trying different sources for the required fields."""
    # Initialize with None values
    result = {
        'AiAnswer': None,
        'model_used': None,
        'Score': None,
        'Title': None,
        'Body': None
    }
    
    # Try to extract from data column first
    if pd.notna(row['data']):
        json_data = extract_from_json(row['data'])
        result.update({k: v for k, v in json_data.items() if pd.notna(v)})
    
    # Fall back to direct columns for any missing values
    if pd.isna(result['AiAnswer']) and pd.notna(row['AiAnswer']):
        result['AiAnswer'] = row['AiAnswer']
    
    if pd.isna(result['model_used']) and pd.notna(row['model_used']):
        result['model_used'] = row['model_used']
    
    if pd.isna(result['Score']) and pd.notna(row['Score']):
        result['Score'] = row['Score']
    
    if pd.isna(result['Title']) and pd.notna(row['Title']):
        result['Title'] = row['Title']
        
    if pd.isna(result['Body']) and pd.notna(row['Body']):
        result['Body'] = row['Body']
    
    return pd.Series(result)

def is_success(value) -> bool:
    """Check if the success value indicates success."""
    if pd.isna(value):
        return True
    
    if isinstance(value, bool):
        return value
    
    if isinstance(value, (int, float)):
        return bool(value)
    
    if isinstance(value, str):
        value = value.lower().strip()
        failure_indicators = {'no', 'false', 'fail', 'failed', 'failure', '0', 'none'}
        return value not in failure_indicators
    
    return True

def process_code_in_df(df):
    """Process and validate code in the dataframe."""
    valid_indices = []
    processed_codes = []
    
    for idx, row in df.iterrows():
        if pd.isna(row['code']):
            continue
            
        code_text = row['code']
        # Extract code if possible, otherwise use the original text
        extracted_code = extract_code(code_text)
        final_code = extracted_code if extracted_code is not None else code_text.strip()
        
        # Remove any remaining triple backticks if present
        final_code = re.sub(r'^```.*\n?|```$', '', final_code).strip()
        
        # Validate the code
        if validate_code(final_code):
            processed_codes.append(final_code)
            valid_indices.append(idx)
    
    return df.loc[valid_indices].copy().assign(code=processed_codes)

# Main processing pipeline
def main():
    # Read the parquet file
    df = pd.read_parquet('merged_results.parquet')

    # Filter out unsuccessful rows
    df = df[df['success'].apply(is_success)]

    # Process each row
    processed_df = df.apply(process_row, axis=1)

    # Drop rows where all required fields are None
    processed_df = processed_df.dropna(how='all')

    # Convert Score to numeric, handling any conversion errors
    processed_df['Score'] = pd.to_numeric(processed_df['Score'], errors='coerce')

    # Define model name mapping
    model_mapping = {
        'Llama-3-3-70B-Instruct': 'Llama-3.3-70B-Instruct',
        'gpt-4o-mini': 'gpt-4o-mini-2024-07-18',
        'gemini-1.5-flash': 'gemini-1.5-flash-002',
        'Meta-Llama-3.1-8B-Instruct-Q6_K': 'Llama-3.1-8B-Instruct-Q6_K',
        'gpt-4o': 'gpt-4o-2024-11-20',
        'Llama-3-3-70B-Instruct-qnycp': 'Llama-3.3-70B-Instruct',
        'Meta-Llama-3.1-8B-Instruct-Q8_0': 'Llama-3.1-8B-Instruct-Q8_0',
        "gpt-4o-batch": "gpt-4o-2024-11-20",
        "gpt-4o-mini-batch": "gpt-4o-mini-2024-07-18"
    }

    # Apply the mapping to model_used column
    processed_df['model_used'] = processed_df['model_used'].replace(model_mapping)
    processed_df = processed_df.rename(columns={'AiAnswer': 'code', 'Score': 'score', 'Title': 'title', 'Body': 'body'})

    # Process and validate code
    final_df = process_code_in_df(processed_df)

    # Save the processed data
    final_df.to_parquet('cleaned_ai_dataset.parquet')

    # Print statistics
    print(f"Original rows: {len(df)}")
    print(f"Processed rows: {len(processed_df)}")
    print(f"Final rows after code validation: {len(final_df)}")
    print("Null value counts in processed data:")
    print(final_df.isnull().sum())
    print("Unique model types after mapping:")
    print(final_df['model_used'].value_counts())

if __name__ == "__main__":
    main()