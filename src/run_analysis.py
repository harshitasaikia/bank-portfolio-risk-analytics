"""Run with: python3 src/run_analysis.py --refresh"""
import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from analytics import prepare_returns, performance, calculate_risk
from download_data import download_stock_data, TICKERS, START_DATE, END_DATE
from load_to_db import create_connection, load_data, DB_PATH
from run_queries import get_prices_for_models, get_summary_stats

ROOT = Path(__file__).resolve().parents[1]


def save_charts(prices, series, risk, losses, portfolio_value, directory):
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'bank-portfolio-matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True, exist_ok=True)
    def finish(name, title, ylabel):
        plt.title(title)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(directory / name, dpi=150)
        plt.close()
    prices.plot(figsize=(10, 5))
    finish('stock_prices.png', 'Adjusted stock closing prices', 'Adjusted price (USD)')
    (prices / prices.iloc[0] * 100).plot(figsize=(10, 5))
    finish('normalized_prices.png', 'Stock growth indexed to 100', 'Index')
    initial_date = prices.index[prices.index < series.index[0]][-1]
    wealth = pd.concat([pd.Series([portfolio_value], index=[initial_date]), series.wealth * portfolio_value])
    wealth.plot(figsize=(10, 5))
    finish('portfolio_value.png', 'Daily rebalanced portfolio', 'Portfolio value (USD)')
    (series.drawdown * 100).plot.area(figsize=(10, 5), color='firebrick')
    finish('drawdown.png', 'Portfolio drawdown', 'Drawdown (%)')
    plt.figure(figsize=(10, 5))
    plt.hist(losses, bins=100, alpha=0.7)
    mc = risk[risk.method == 'monte_carlo']
    for row in mc.itertuples():
        plt.axvline(row.var_dollars, linestyle='--', label=f'{row.confidence:.0%} VaR')
    plt.xlabel('One-day loss (USD; negative = gain)')
    plt.legend()
    finish('monte_carlo_losses.png', 'Simulated loss distribution', 'Scenarios')
    risk.pivot(index='confidence', columns='method', values='var_dollars').plot.bar(figsize=(10, 5), rot=0)
    finish('var_comparison.png', 'One-day VaR comparison', 'VaR (USD)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh', action='store_true', help='Download prices and replace the stock_prices table')
    parser.add_argument('--start', default=START_DATE, help='Download start (inclusive), used with --refresh')
    parser.add_argument('--end', default=END_DATE, help='Download end (exclusive), used with --refresh')
    parser.add_argument('--weights', nargs=3, type=float, default=[1/3]*3, metavar=('MS', 'JPM', 'BAC'))
    parser.add_argument('--portfolio-value', type=float, default=1_000_000)
    parser.add_argument('--risk-free-rate', type=float, default=0.07, help='Annual effective rate; 0.07 is a notebook example assumption')
    parser.add_argument('--simulations', type=int, default=100_000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--confidence', nargs='+', type=float, default=[0.90, 0.95, 0.99])
    args = parser.parse_args()
    if any(not np.isfinite(x) or x < 0 for x in args.weights) or not np.isclose(sum(args.weights), 1):
        parser.error('Weights must be nonnegative, finite and sum to one (MS JPM BAC).')
    if args.portfolio_value <= 0 or not np.isfinite(args.portfolio_value):
        parser.error('Portfolio value must be positive and finite.')
    if args.risk_free_rate <= -1 or not np.isfinite(args.risk_free_rate):
        parser.error('Risk-free rate must be finite and greater than -1.')
    if args.simulations < 1000 or any(not 0 < c < 1 for c in args.confidence):
        parser.error('Use at least 1000 simulations and confidence levels between 0 and 1.')
    if args.refresh:
        if pd.Timestamp(args.start) >= pd.Timestamp(args.end):
            parser.error('Start date must be before end date.')
        data = download_stock_data(TICKERS + ['^GSPC'], args.start, args.end)
        conn = create_connection()
        try:
            load_data(conn, data)
        finally:
            conn.close()
    if not DB_PATH.exists():
        parser.error('Database not found. Run with --refresh to download data first.')
    prices = get_prices_for_models()
    prices.index = pd.to_datetime(prices.index)
    if not set(TICKERS + ['^GSPC']).issubset(prices.columns):
        parser.error('Database needs all three stocks and ^GSPC. Run with --refresh.')
    returns, weights = prepare_returns(prices, dict(zip(TICKERS, args.weights)))
    benchmark = prices['^GSPC'].pct_change(fill_method=None)
    stats, series = performance(returns, weights, benchmark, args.risk_free_rate)
    risk, losses = calculate_risk(returns, weights, args.portfolio_value, args.confidence, args.simulations, args.seed)
    output = ROOT / 'data' / 'results'
    output.mkdir(parents=True, exist_ok=True)
    stats.to_csv(output / 'performance.csv', index_label='metric')
    risk.to_csv(output / 'risk.csv', index=False)
    series.to_csv(output / 'portfolio_daily.csv', index_label='date')
    returns.to_csv(output / 'asset_simple_returns.csv', index_label='date')
    np.log1p(returns).to_csv(output / 'asset_log_returns.csv', index_label='date')
    returns.cov().to_csv(output / 'covariance.csv', index_label='ticker')
    returns.corr().to_csv(output / 'correlation.csv', index_label='ticker')
    get_summary_stats().to_csv(output / 'stock_summary.csv', index=False)
    metadata = dict(vars(args), weights=weights.to_dict(), benchmark='^GSPC (price index)',
                    price_start=str(prices.index.min().date()), price_end=str(prices.index.max().date()),
                    return_observations=len(returns), missing_return_rows_excluded=len(prices)-1-len(returns),
                    horizon_trading_days=1, rebalancing='daily constant weights',
                    price_adjustment='yfinance auto_adjust=True for pipeline downloads',
                    loss_sign='positive=loss; negative=gain', normal_model='multivariate normal simple returns')
    (output / 'assumptions.json').write_text(json.dumps(metadata, indent=2) + '\n')
    save_charts(prices[TICKERS], series, risk, losses, args.portfolio_value, ROOT / 'charts')
    print('\nPerformance (returns in decimal units):\n', stats.to_string())
    print('\nOne-day risk (USD):\n', risk.to_string(index=False))
    print(f'\nResults: {output}\nCharts: {ROOT / "charts"}')


if __name__ == '__main__':
    main()
