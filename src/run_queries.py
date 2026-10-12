import sqlite3
import pandas as pd

from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / 'data' / 'portfolio.db'

def run_query(sql):
    if not DB_PATH.exists():
        raise FileNotFoundError("Run the data loader first: python3 src/load_to_db")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(sql, conn)
    conn.close()
    return df

def get_summary_stats():
    sql = """
    SELECT 
        ticker,
        COUNT(*) AS trading_days,
        ROUND(AVG((close-open)/open*100), 2) AS avg_return_pct,
        ROUND(MIN(close), 2) AS lowest_close,
        ROUND(MAX(close), 2) AS highest_close
    FROM stock_prices
    GROUP BY ticker
    ORDER BY ticker
    """
    return run_query(sql)

def get_prices_for_models():
    sql = """
    SELECT 
        date,
        ticker,
        close
    FROM stock_prices
    ORDER BY ticker, date
    """
    df = run_query(sql)
    pivot = df.pivot(index='date', columns='ticker', values='close')
    return pivot

if __name__ == "__main__":
    print(get_summary_stats())
    print(get_prices_for_models())
