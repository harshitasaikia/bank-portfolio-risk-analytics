"""Daily rebalanced, long-only equity portfolio analytics and one-day risk."""
from statistics import NormalDist
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def prepare_returns(prices, weights):
    weights = pd.Series(weights, dtype=float)
    if weights.empty or not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1):
        raise ValueError('Weights must be finite, nonnegative and sum to one.')
    prices = prices.loc[:, weights.index].sort_index()
    if prices.index.has_duplicates:
        raise ValueError('Price dates must be unique.')
    if ((prices <= 0) | (~np.isfinite(prices) & prices.notna())).any().any():
        raise ValueError('Prices must be positive and finite where present.')
    # Calculate before dropping missing rows: never bridge a missing daily quote.
    returns = prices.pct_change(fill_method=None).dropna()
    if len(returns) < 2:
        raise ValueError('At least two complete daily return observations are required.')
    return returns, weights


def performance(returns, weights, benchmark, risk_free_rate=0.07):
    if not np.isfinite(risk_free_rate) or risk_free_rate <= -1:
        raise ValueError('Annual risk-free rate must be finite and greater than -1.')
    portfolio = returns.dot(weights)
    logs = np.log1p(portfolio)
    daily_rf = np.expm1(np.log1p(risk_free_rate) / TRADING_DAYS)
    excess = portfolio - daily_rf
    volatility = portfolio.std(ddof=1)
    downside = np.sqrt(np.mean(np.minimum(excess, 0) ** 2))
    wealth = (1 + portfolio).cumprod()
    # Include initial capital in the high-water mark, even if day one loses money.
    peak = wealth.cummax().clip(lower=1)
    drawdown = wealth / peak - 1
    cagr = np.expm1(logs.mean() * TRADING_DAYS)
    aligned = pd.concat([portfolio.rename('portfolio'), benchmark.rename('benchmark')], axis=1).dropna()
    beta = alpha = np.nan
    if len(aligned) >= 2 and aligned.benchmark.var(ddof=1) > 0:
        beta = aligned.cov().loc['portfolio', 'benchmark'] / aligned.benchmark.var(ddof=1)
        alpha = ((aligned.portfolio.mean() - daily_rf) - beta * (aligned.benchmark.mean() - daily_rf)) * TRADING_DAYS
    def ratio(a, b):
        return a / b if np.isfinite(b) and abs(b) > 1e-15 else np.nan
    stats = pd.Series({
        'observations': len(portfolio), 'benchmark_observations': len(aligned),
        'total_return': wealth.iloc[-1] - 1,
        'annualized_geometric_return': cagr,
        'annualized_arithmetic_return': portfolio.mean() * TRADING_DAYS,
        'annualized_log_return': logs.mean() * TRADING_DAYS,
        'annualized_volatility': volatility * np.sqrt(TRADING_DAYS),
        'annualized_log_volatility': logs.std(ddof=1) * np.sqrt(TRADING_DAYS),
        'beta': beta, 'annualized_alpha': alpha,
        'sharpe_ratio': ratio(excess.mean(), volatility) * np.sqrt(TRADING_DAYS),
        'sortino_ratio': ratio(excess.mean(), downside) * np.sqrt(TRADING_DAYS),
        'max_drawdown': -drawdown.min(), 'calmar_ratio': ratio(cagr, -drawdown.min()),
        'treynor_ratio': ratio(excess.mean() * TRADING_DAYS, beta),
    }, name='value')
    series = pd.DataFrame({'simple_return': portfolio, 'log_return': logs, 'wealth': wealth, 'drawdown': drawdown})
    return stats, series


def calculate_risk(returns, weights, portfolio_value=1_000_000, confidences=(0.90, 0.95, 0.99), simulations=100_000, seed=42):
    """Loss quantiles: positive means loss, negative means gain; no clipping.

    Monte Carlo and analytical methods share a multivariate normal simple-return
    model. This is a one-day linear exposure model, not a multi-day price model.
    """
    if not np.isfinite(portfolio_value) or portfolio_value <= 0:
        raise ValueError('Portfolio value must be positive and finite.')
    if not isinstance(simulations, (int, np.integer)) or simulations < 1000:
        raise ValueError('Use at least 1000 Monte Carlo simulations.')
    confidences = tuple(confidences)
    if not confidences or any(not 0 < c < 1 for c in confidences):
        raise ValueError('Confidence levels must be between zero and one.')
    weights = pd.Series(weights, dtype=float).reindex(returns.columns)
    if weights.isna().any() or not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1):
        raise ValueError('Weights must match return columns and sum to one.')
    if len(returns) < 2 or not np.isfinite(returns.to_numpy()).all():
        raise ValueError('At least two finite return observations are required.')
    mean = returns.mean().to_numpy()
    cov = returns.cov().to_numpy()
    w = weights.to_numpy()
    mu = float(w @ mean)
    sigma = float(np.sqrt(max(w @ cov @ w, 0)))
    rng = np.random.default_rng(seed)
    scenarios = rng.multivariate_normal(mean, cov, size=simulations, check_valid='raise')
    simulated_losses = -portfolio_value * (scenarios @ w)
    historical_losses = -portfolio_value * returns.dot(weights).to_numpy()
    normal = NormalDist()
    rows = []
    for confidence in confidences:
        z = normal.inv_cdf(confidence)
        var = portfolio_value * (z * sigma - mu)
        es = portfolio_value * (sigma * normal.pdf(z) / (1 - confidence) - mu)
        rows.append(('variance_covariance', confidence, var, es))
        for name, losses in [('historical', historical_losses), ('monte_carlo', simulated_losses)]:
            threshold = float(np.quantile(losses, confidence))
            # Empirical ES integrates the worst (1-c)*n observations, including
            # a fractional boundary observation when the tail size is noninteger.
            tail_size = (1 - confidence) * len(losses)
            ordered = np.sort(losses)[::-1]
            whole = int(np.floor(tail_size))
            fraction = tail_size - whole
            tail_sum = ordered[:whole].sum()
            if fraction > 0:
                tail_sum += fraction * ordered[whole]
            rows.append((name, confidence, threshold, tail_sum / tail_size))
    result = pd.DataFrame(rows, columns=['method', 'confidence', 'var_dollars', 'expected_shortfall_dollars'])
    result['var_fraction'] = result.var_dollars / portfolio_value
    result['horizon_trading_days'] = 1
    return result, simulated_losses
