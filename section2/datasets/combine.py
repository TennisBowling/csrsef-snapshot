import os
import glob
from multiprocessing import Pool, cpu_count, Manager
import pandas as pd
from functools import partial
import numpy as np
from tqdm import tqdm

def combine_chunk(file_list, progress_queue):
    dfs = []
    for file in file_list:
        try:
            df = pd.read_parquet(file)
            dfs.append(df)
            progress_queue.put(1)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            progress_queue.put(1)
    
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return None

def divide_and_conquer_combine(input_path, output_file, num_cores=None):
    parquet_files = glob.glob(os.path.join(input_path, "*.parquet"))
    
    if not parquet_files:
        raise ValueError(f"No parquet files found in {input_path}")
    
    if num_cores is None:
        num_cores = cpu_count()
    
    total_files = len(parquet_files)
    print(f"Found {total_files} parquet files")
    print(f"Using {num_cores} CPU cores")
    
    chunk_size = max(1, total_files // num_cores)
    file_chunks = np.array_split(parquet_files, num_cores)
    
    manager = Manager()
    progress_queue = manager.Queue()
    
    with Pool(num_cores) as pool:
        async_results = [pool.apply_async(combine_chunk, 
                                        args=(chunk, progress_queue)) 
                        for chunk in file_chunks]
        
        with tqdm(total=total_files, 
                 desc="Processing files", 
                 unit="file") as pbar:
            completed = 0
            while completed < total_files:
                # Update progress bar whenever a file is processed
                if not progress_queue.empty():
                    progress_queue.get()
                    completed += 1
                    pbar.update(1)
        
        chunk_results = [res.get() for res in async_results]
    
    print("Combining chunks...")
    dfs_to_combine = [df for df in chunk_results if df is not None]
    final_df = pd.concat(dfs_to_combine, ignore_index=True)
    
    print("Saving combined file...")
    final_df.to_parquet(output_file, index=False)
    print(f"Combined parquet file saved to {output_file}")
    print(f"Final DataFrame shape: {final_df.shape}")

if __name__ == "__main__":
    input_path = "."
    output_file = "github.parquet"
    num_cores = 119
    
    divide_and_conquer_combine(input_path, output_file, num_cores)