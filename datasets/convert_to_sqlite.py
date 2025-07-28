import pandas as pd
import sqlite3
from pathlib import Path

def convert_parquet_to_sqlite(parquet_path: str, sqlite_path: str):
    df = pd.read_parquet(parquet_path, columns=['Body', 'Title', 'Score'])
    
    # Add row_id column starting from 1
    df.insert(0, 'row_id', range(1, len(df) + 1))
    
    print("First 3 rows from parquet:")
    print(df.head(3))

    with sqlite3.connect(sqlite_path) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS posts (
                row_id BIGINT PRIMARY KEY,
                Body TEXT,
                Title TEXT,
                Score INTEGER
            )
        ''')
        
        df.to_sql('posts', conn, if_exists='replace', index=False)
        
        conn.execute('CREATE INDEX idx_score ON posts(Score)')
        conn.execute('CREATE INDEX idx_title ON posts(Title)')
        conn.execute("CREATE INDEX idx_row ON posts(row_id)")
        
        # Verify the row count
        row_count = conn.execute('SELECT COUNT(*) FROM posts').fetchone()[0]
        print(f"\nSuccessfully converted {row_count} rows to SQLite")
    
        print("First 3 rows from SQLite:")
        for row in conn.execute('SELECT * FROM posts LIMIT 3'):
            print("Row from SQLite:")
            print(f"Row ID: {row[0]}")
            print(f"Title: {row[2]}")
            print(f"Score: {row[3]}")
            print(f"Body: {row[1][:200]}..." if len(row[1]) > 200 else f"Body: {row[1]}")

if __name__ == "__main__":
    convert_parquet_to_sqlite('so_dataset.parquet', 'so_dataset.sqlite')