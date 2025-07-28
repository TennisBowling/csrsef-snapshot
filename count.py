import pandas as pd
import argparse
import os
from datetime import datetime
import numpy as np

def deduplicate_parquet(input_path, output_path=None, subset=None, keep='first'):
    if output_path is None:
        dir_name, file_name = os.path.split(input_path)
        base_name, _ = os.path.splitext(file_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(dir_name, f"{base_name}_deduplicated_{timestamp}.parquet")
    
    print(f"Reading parquet file: {input_path}")
    df = pd.read_parquet(input_path)
    
    original_count = len(df)
    print(f"Original row count: {original_count:,}")
    
    if subset is None:
        print("Creating string representation of all rows for deduplication...")
        
        temp_df = pd.DataFrame()
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # For object columns that might contain arrays or other unhashable types
                temp_df[col] = df[col].astype(str)
            else:
                temp_df[col] = df[col]
        
        print("Finding duplicates...")
        is_duplicate = temp_df.duplicated(keep=keep)
        
    else:
        # If subset is specified, only check those columns
        print(f"Creating string representation of specified columns: {subset}")
        
        temp_df = pd.DataFrame()
        
        for col in subset:
            if col in df.columns:
                if df[col].dtype == 'object':
                    temp_df[col] = df[col].astype(str)
                else:
                    temp_df[col] = df[col]
            else:
                print(f"Warning: Column '{col}' not found in the dataframe")
        
        print("Finding duplicates based on specified columns...")
        is_duplicate = temp_df.duplicated(subset=subset, keep=keep)
    
    # Keep only non-duplicate rows from the original dataframe
    print("Removing duplicates...")
    df_deduplicated = df[~is_duplicate]
    
    new_count = len(df_deduplicated)
    print(f"New row count: {new_count:,}")
    print(f"Removed {original_count - new_count:,} duplicate rows")
    
    print(f"Writing deduplicated data to: {output_path}")
    df_deduplicated.to_parquet(output_path, index=False)
    print("Done!")
    
    return output_path, original_count, new_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove duplicates from parquet file")
    parser.add_argument("input_file", help="Path to input parquet file")
    parser.add_argument("--output_file", help="Path to output parquet file (optional)")
    parser.add_argument("--columns", help="Comma-separated list of columns to check for duplicates (optional)")
    parser.add_argument("--keep", choices=['first', 'last', 'none'], default='first',
                        help="Which duplicates to keep: 'first', 'last', or 'none' (removes all duplicates)")
    
    args = parser.parse_args()
    
    subset_cols = None
    if args.columns:
        subset_cols = [col.strip() for col in args.columns.split(',')]
    
    keep_option = False if args.keep == 'none' else args.keep
    
    deduplicate_parquet(args.input_file, args.output_file, subset_cols, keep_option)