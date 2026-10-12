import sqlite3
import os
from download_data import download_stock_data

from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / 'data' / 'portfolio.db'

def create_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)  # Ensure the data directory exists
    conn = sqlite3.connect(DB_PATH)
    return conn

def load_data(conn, df):
    df.to_sql(
        'stock_prices',
        conn,
        if_exists='replace',
        index=False
    )
    print('Data loaded into the database.')
    cursor = conn.execute('SELECT COUNT(*) FROM stock_prices')
    count = cursor.fetchone()[0]
    print(f'Total rows in the database: {count}')

if __name__ == "__main__":
    df = download_stock_data(['MS', 'JPM', 'BAC', '^GSPC'])
    conn = create_connection()
    load_data(conn, df)
    conn.close()
