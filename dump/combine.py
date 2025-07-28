import os
import glob
import time
import multiprocessing
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

def _merge_chunk(file_chunk):
    # Read the first file to get the schema, and handle empty file.
    if not file_chunk:
        return pd.DataFrame()

    first_file_path = file_chunk[0]
    first_table = pq.read_table(first_file_path)

    # Check if the first file is empty
    if first_table.num_rows == 0:
        # Create an empty DataFrame with the correct schema
        df = first_table.to_pandas()
    else:
        df = first_table.to_pandas()

    for file_path in file_chunk[1:]:  # start from the second file
        try:
            table = pq.read_table(file_path)
            if table.num_rows > 0: # only append if not empty
                df = pd.concat([df, table.to_pandas()], ignore_index=True)
        except Exception as e:
            print(f"Error reading or merging file {file_path}: {e}")
            continue
    return df


def merge_parquet_files(input_dir, output_file="combined.parquet", num_processes=None):
    if num_processes is None:
        num_processes = 50
    print(f"Using {num_processes} processes.")

    # Get a list of all parquet files in the input directory
    parquet_files = glob.glob(os.path.join(input_dir, "*.parquet"))
    if not parquet_files:
        print(f"No parquet files found in the directory: {input_dir}")
        return

    # Divide the list of files into chunks for parallel processing
    chunk_size = max(1, len(parquet_files) // num_processes)  # Ensure chunk_size is at least 1 so int division
    file_chunks = [parquet_files[i:i + chunk_size] for i in range(0, len(parquet_files), chunk_size)]

    pool = multiprocessing.Pool(processes=num_processes)

    results = pool.map(_merge_chunk, file_chunks)

    # Close the pool and wait for all processes to complete.
    pool.close()
    pool.join()

    # Combine the results from each process.
    combined_df = pd.DataFrame() # Start with empty
    for df in results:
        if df is not None: # Only concat if not None
          if len(df) > 0:
            combined_df = pd.concat([combined_df, df], ignore_index=True)

    
    # Write the combined DataFrame to a new parquet file.
    try:
        combined_table = pa.Table.from_pandas(combined_df)
        pq.write_table(combined_table, output_file)
        print(f"Successfully merged parquet files to {output_file}")
    except Exception as e:
        # Sometimes this doesnt write
        print(f"Error writing combined parquet file: {e}")

if __name__ == "__main__":
    import sys

    input_directory = sys.argv[1]
    start_time = time.time()
    merge_parquet_files(input_directory)
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")