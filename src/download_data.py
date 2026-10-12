import yfinance as yf
import pandas as pd
from pathlib import Path

TICKERS = ['MS', 'JPM', 'BAC']
START_DATE = '2024-01-01' 
END_DATE = '2025-01-01'

def download_stock_data(tickers=None, start=START_DATE, end=END_DATE):
    yf.set_tz_cache_location(str(Path(__file__).resolve().parents[1] / "data" / ".yfinance_cache"))
    all_data = []

    for ticker in (TICKERS if tickers is None else tickers):
        print(f"Downloading data for {ticker}...")
        stock = yf.Ticker(ticker)
        df = stock.history(start=start, end=end, auto_adjust=True, raise_errors=True)
        if df.empty:
            raise ValueError(f"No prices returned for {ticker}")
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]  # Lowercase column names
        df['ticker'] = ticker
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')  # Convert Date to string for easier handling
        df = df[['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']]
        all_data.append(df)

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined[['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']]
    return combined

if __name__ == "__main__":
    stock_data = download_stock_data()
    print(stock_data.head())
    print(f'Total rows downloaded: {len(stock_data)}')
