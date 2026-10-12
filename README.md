# Bank portfolio analytics

Downloads adjusted daily prices for MS, JPM and BAC, stores them in SQLite,
and analyzes a constant-weight portfolio with daily rebalancing. The benchmark
is the S&P 500 price index (`^GSPC`), not SPY or a total-return index.

## Run

```bash
python3 -m pip install -r requirements.txt
python3 src/run_analysis.py --refresh
```

`--refresh` downloads prices and **replaces** the `stock_prices` table only after
all four downloads succeed. Omit it to reuse the local database:

```bash
python3 src/run_analysis.py
python3 src/run_analysis.py --weights 0.5 0.3 0.2 --portfolio-value 250000 --risk-free-rate 0.04 --simulations 200000
```

Weights are in **MS JPM BAC** order. Defaults: equal weights, $1,000,000,
2024-01-01 through 2024-12-31 (exclusive download end 2025-01-01), 252 trading
days per year, 100,000 simulations, seed 42, and 90%, 95%, 99% confidence.
The 7% annual effective risk-free rate is retained from the reference notebook
as an example assumption; it is not a fetched market rate. Set it explicitly
for your analysis. `--start` and `--end` apply to downloads with `--refresh`.
All paths resolve relative to the project, regardless of working directory.

## Files and data flow

- `src/download_data.py`: downloads OHLCV with explicit `auto_adjust=True`,
  standardizes dates, and rejects empty downloads.
- `src/load_to_db.py`: creates `data/portfolio.db` and stores `stock_prices`.
- `src/run_queries.py`: per-ticker SQL summaries and pivoted closing prices.
  Its `avg_return_pct` is the open-to-close percentage, not close-to-close return.
- `src/analytics.py`: returns, performance metrics, and risk calculations.
- `src/run_analysis.py`: configuration, orchestration, CSV reports, and charts.
- `src/load_to_db` and `src/run_queries`: original commands retained as wrappers.
- `tests/test_analytics.py`: deterministic numerical and edge-case checks.
- `sql/`: reserved for future standalone SQL files.

## Outputs

`data/results/` contains performance, risk, daily portfolio series, asset simple
and log returns, covariance and correlation matrices, and stock summaries as CSV.
`assumptions.json` records parameters, sample dates, and excluded return rows.
Returns and volatility are decimals (0.10 = 10%). Undefined ratios are blank/NaN.

`charts/` contains adjusted and normalized stock prices, portfolio value,
drawdown, the simulated loss distribution, and a comparison of VaR methods.
Outputs are overwritten on subsequent runs.

## Calculation conventions

Asset simple return: `price[t] / price[t-1] - 1`. Portfolio simple return is the
weighted sum of asset returns, assuming frictionless daily rebalancing. Portfolio
log return is `log1p(portfolio_simple_return)`, not a weighted sum of asset logs.
Missing daily quotes are not forward-filled or bridged before calculating returns.
Complete return rows are used; benchmark returns are aligned by date for beta/alpha.
With missing observations, reported growth compounds only available return rows;
inspect `missing_return_rows_excluded` before interpreting a full-period result.

- Geometric annualized return: `exp(mean(portfolio_log_return)*252)-1`.
- Arithmetic annualized return: `mean(portfolio_simple_return)*252`.
- Volatility: sample standard deviation (`ddof=1`) times `sqrt(252)`.
- Beta: covariance with the benchmark divided by benchmark variance.
- Alpha: annualized mean daily excess return less beta times benchmark excess
  return. Annual effective risk-free rate is converted to a daily effective rate.
- Sharpe: mean daily excess return / sample daily standard deviation, times
  `sqrt(252)`.
- Sortino: mean daily excess return / root mean squared shortfall below the daily
  risk-free target, times `sqrt(252)`. Denominator includes all observations.
- Maximum drawdown includes initial capital in the running peak.
- Calmar: geometric annualized return / maximum drawdown magnitude.
- Treynor: arithmetic annualized daily excess return / beta.

These conventions correct several approximations in `Investment.html`, so its
printed results will not necessarily match these reports. The index benchmark
also lacks reinvested dividends, unlike adjusted stock prices.

## One-day VaR and expected shortfall

All methods use the same weights, value, daily observations and confidence levels.
Loss is `-portfolio_value * return`: a positive result denotes loss; a negative
VaR denotes a gain at the specified quantile. No values are clipped to zero.
VaR is a loss threshold, not the worst possible loss. At 99% confidence the model
assigns 1% probability to losses exceeding that threshold.

1. **Historical**: empirical confidence quantile of daily portfolio losses.
   Expected shortfall averages the worst `(1-confidence)*n` losses, weighting a
   fractional boundary observation when necessary.
2. **Variance-covariance**: estimate daily mean vector `mu` and sample covariance
   matrix `Sigma`; `mu_p = w @ mu`, `sigma_p = sqrt(w @ Sigma @ w)`.
   Under normal simple returns, `VaR = value * (z_c*sigma_p - mu_p)`.
   Expected shortfall is `value * (sigma_p*phi(z_c)/(1-c) - mu_p)`.
3. **Monte Carlo**: sample correlated multivariate normal daily asset returns
   with the same mean and covariance, convert to portfolio losses, and calculate
   empirical VaR and expected shortfall. A fixed random seed makes runs repeatable.

Monte Carlo should converge to the analytical result under this shared model;
it does not automatically capture fat tails. Normal simple returns are an
approximation and can theoretically fall below -100%. Multi-day compounded
simulation, transaction costs, and out-of-sample backtesting are not implemented.
One year supplies only a few observations in the 99% historical tail.

Formula reference: [MathWorks portfolio normal VaR](https://www.mathworks.com/help/risk/estimate-var-using-parametric-methods.html).

## Tests

```bash
python3 -m unittest discover -s tests -v
```

Tests cover label-aligned weights, missing prices, initial-capital drawdown,
portfolio log returns, downside deviation, analytical covariance, Monte Carlo
convergence/reproducibility, empirical tail averaging and degenerate returns.
