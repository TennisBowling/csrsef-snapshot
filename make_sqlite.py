#!/usr/bin/env python3

import sys
import os
import sqlite3
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

def convert_parquet_to_sqlite(parquet_file, sqlite_file, table_name='data'):
    print(f"Reading parquet file: {parquet_file}")
    
    try:
        parquet_schema = pq.read_schema(parquet_file)
        if 'code' not in parquet_schema.names:
            print(f"Error: The parquet file does not contain a 'code' column.")
            print(f"Available columns: {parquet_schema.names}")
            return False
        
        df = pd.read_parquet(parquet_file)
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        return False
    
    print(f"parquet file read successfully. Found {len(df)} rows.")
    
    if os.path.exists(sqlite_file):
        print(f"Warning: SQLite database file {sqlite_file} already exists. It will be overwritten.")
        os.remove(sqlite_file)
    
    try:
        print(f"Creating SQLite database: {sqlite_file}")
        conn = sqlite3.connect(sqlite_file)
        
        df.to_sql(table_name, conn, index=False)
        
        print(f"Creating index on 'code' column")
        conn.execute(f"CREATE INDEX idx_{table_name}_code ON {table_name} (code)")
        
        cursor = conn.cursor()

        print(f"SQLite database created successfully with rows in table '{table_name}'")
        
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        print(f"Table schema:")
        for col in columns:
            print(f"  {col[1]} ({col[2]})")
        
        conn.close()
        return True
    except Exception as e:
        print(f"Error creating SQLite database: {e}")
        if os.path.exists(sqlite_file):
            os.remove(sqlite_file)
        return False

def main():
    if len(sys.argv) < 3:
        print("Usage: python parquet_to_sqlite.py input_file.parquet output_file.sqlite [table_name]")
        sys.exit(1)
    
    parquet_file = sys.argv[1]
    sqlite_file = sys.argv[2]
    table_name = sys.argv[3] if len(sys.argv) > 3 else 'data'
    
    if not os.path.exists(parquet_file):
        print(f"Error: parquet file {parquet_file} does not exist.")
        sys.exit(1)
    
    success = convert_parquet_to_sqlite(parquet_file, sqlite_file, table_name)
    if success:
        print("Conversion completed successfully")
    else:
        print("Conversion failed")
        sys.exit(1)

if __name__ == "__main__":
    main()