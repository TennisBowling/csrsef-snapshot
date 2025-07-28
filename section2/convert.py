import pyarrow.parquet as pq
import sqlite3
from tqdm import tqdm
import pandas as pd

def convert_parquet_to_sqlite():
    try:
        print("Opening parquet file...")
        parquet_file = pq.ParquetFile('github.parquet')
        total_rows = parquet_file.metadata.num_rows
        
        chunk_size = min(100000, total_rows // 100)
        print(f"Processing approximately {total_rows:,} rows in chunks of {chunk_size:,}...")
        
        with sqlite3.connect('github.sqlite', isolation_level='IMMEDIATE') as conn:
            conn.execute('PRAGMA journal_mode=WAL')
            conn.execute('PRAGMA synchronous=NORMAL')
            conn.execute('PRAGMA wal_autocheckpoint=1000')
            conn.execute('PRAGMA mmap_size=30000000000')
            conn.execute('PRAGMA page_size=4096')
            conn.execute('PRAGMA cache_size=-2000000')
            conn.execute('PRAGMA temp_store=MEMORY')
            conn.execute('PRAGMA journal_mode=OFF')
            
            conn.execute('BEGIN TRANSACTION')
            
            conn.execute('DROP TABLE IF EXISTS github_data')
            
            conn.execute('''
                CREATE TABLE github_data (
                    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    code TEXT,
                    full_path TEXT
                )
            ''')
            
            pbar = tqdm(total=total_rows, unit='rows')
            
            for batch in parquet_file.iter_batches(batch_size=chunk_size):
                df = batch.to_pandas()
                print(df.columns)
                
                df.to_sql('github_data', conn, if_exists='append', index=False)
                
                pbar.update(len(df))
                
                del df
            
            conn.execute('COMMIT')
            pbar.close()
            
            print("Creating indexes (this might take a while)...")
            with tqdm(total=2, desc="Creating indexes") as idx_pbar:
                print("Creating full_path index...")
                conn.execute('CREATE INDEX idx_full_path ON github_data(full_path)')
                idx_pbar.update(1)
                
                print("Creating code index...")
                conn.execute('CREATE INDEX idx_code ON github_data(code)')
                idx_pbar.update(1)
            
            print("Optimizing database...")
            conn.execute('PRAGMA journal_mode=WAL')
            conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
            conn.execute('PRAGMA journal_mode=DELETE')
            conn.execute('VACUUM')
            conn.execute('PRAGMA journal_mode=WAL')
            
        print("Conversion completed successfully!")
        
    except FileNotFoundError:
        print("Error: github.parquet file not found!")
    except sqlite3.Error as e:
        print(f"SQLite error occurred: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    convert_parquet_to_sqlite()