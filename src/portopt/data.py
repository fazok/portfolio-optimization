"""Market data: Yahoo Finance prices (cached) and a simulated universe for offline use.

The universe matches the companion `market-risk-var` project: 10 liquid ETFs
across equities, real estate, Treasuries, credit and gold, all trading since 2004.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_TICKERS = {
    "SPY": "US large cap equity",
    "QQQ": "US tech equity",
    "IWM": "US small cap equity",
    "EFA": "Developed ex-US equity",
    "EEM": "Emerging markets equity",
    "VNQ": "US real estate (REITs)",
    "TLT": "US Treasuries 20y+",
    "IEF": "US Treasuries 7-10y",
    "LQD": "US investment-grade credit",
    "GLD": "Gold",
}
DEFAULT_START = "2005-01-01"
DEFAULT_END = "2025-12-31"
TRADING_DAYS = 252


def load_prices(
    tickers: list[str] | None = None,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    cache_dir: str | Path | None = "data/raw",
) -> pd.DataFrame:
    """Adjusted close prices (dividends reinvested), cached as Parquet."""
    tickers = list(tickers or DEFAULT_TICKERS)
    cache = None
    if cache_dir is not None:
        cache = Path(cache_dir) / f"prices_{'-'.join(sorted(tickers))}_{start}_{end}.parquet"
        if cache.exists():
            return pd.read_parquet(cache)[tickers]

    import yfinance as yf  # optional dependency: pip install -e ".[data]"

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"][tickers].dropna(how="any")
    if prices.empty:
        raise RuntimeError("No data returned by Yahoo Finance (network or ticker issue).")
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        prices.to_parquet(cache)
    return prices


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns."""
    return prices.pct_change().dropna(how="any")


def make_synthetic(n_days: int = 4000, seed: int = 0) -> pd.DataFrame:
    """Simulated daily returns: a two-factor model (equity, rates) with fat tails.

    Expected returns are modest and close to each other, as in real markets, so
    the estimation-error problem of mean-variance optimisation shows up here too.
    """
    rng = np.random.default_rng(seed)
    equity = rng.standard_t(5, n_days) * np.sqrt(3 / 5) * 0.011
    rates = rng.standard_t(6, n_days) * np.sqrt(4 / 6) * 0.004
    spec = {  # name: (equity beta, rates beta, idiosyncratic vol, annual drift)
        "SPY": (1.0, 0.0, 0.002, 0.09), "QQQ": (1.15, 0.0, 0.006, 0.11),
        "IWM": (1.15, 0.0, 0.007, 0.08), "EFA": (0.9, 0.0, 0.005, 0.06),
        "EEM": (1.1, 0.0, 0.008, 0.06), "VNQ": (1.1, 0.3, 0.009, 0.07),
        "TLT": (-0.2, 2.2, 0.002, 0.04), "IEF": (-0.08, 1.0, 0.001, 0.03),
        "LQD": (0.1, 0.9, 0.002, 0.04), "GLD": (0.05, 0.4, 0.010, 0.06),
    }
    out = {}
    for k, (b_eq, b_r, iv, drift) in spec.items():
        eps = rng.standard_t(5, n_days) * np.sqrt(3 / 5) * iv
        out[k] = drift / TRADING_DAYS + b_eq * equity + b_r * rates + eps
    return pd.DataFrame(out, index=pd.bdate_range("2008-01-01", periods=n_days))
