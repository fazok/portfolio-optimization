"""Performance and risk metrics, and a bootstrap test of Sharpe ratio differences."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import TRADING_DAYS


def annual_return(r: pd.Series) -> float:
    return float((1 + r).prod() ** (TRADING_DAYS / len(r)) - 1)


def annual_vol(r: pd.Series) -> float:
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe(r, rf: float = 0.0) -> float:
    r = np.asarray(r, dtype=float) - rf / TRADING_DAYS
    return float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sortino(r: pd.Series) -> float:
    downside = np.sqrt(np.mean(np.minimum(r, 0) ** 2)) * np.sqrt(TRADING_DAYS)
    return float(r.mean() * TRADING_DAYS / downside)


def drawdown(r: pd.Series) -> pd.Series:
    wealth = (1 + r).cumprod()
    return wealth / wealth.cummax() - 1


def max_drawdown(r: pd.Series) -> float:
    return float(drawdown(r).min())


def hist_var_es(r: pd.Series, a: float = 0.99) -> tuple[float, float]:
    """1-day historical VaR and ES, as positive losses."""
    losses = -np.asarray(r)
    var = float(np.quantile(losses, a))
    return var, float(losses[losses >= var].mean())


def performance_table(sim, rf: float = 0.0) -> pd.DataFrame:
    rows = []
    years = len(sim.returns) / TRADING_DAYS
    for name in sim.returns.columns:
        r = sim.returns[name]
        var, es = hist_var_es(r)
        rows.append({
            "strategy": name,
            "return_%": 100 * annual_return(r),
            "vol_%": 100 * annual_vol(r),
            "sharpe": sharpe(r, rf),
            "sortino": sortino(r),
            "max_dd_%": 100 * max_drawdown(r),
            "calmar": annual_return(r) / abs(max_drawdown(r)),
            "turnover_pa_%": 100 * sim.turnover[name].sum() / years,
            "var99_%": 100 * var,
            "es99_%": 100 * es,
        })
    return pd.DataFrame(rows).set_index("strategy")


def _circular_block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, n, size=int(np.ceil(n / block)))
    idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
    return idx % n


def sharpe_diff_test(r1, r2, block: int = 21, n_boot: int = 2000,
                     seed: int = 0) -> dict[str, float]:
    """Circular block bootstrap test of H0: SR1 = SR2.

    Blocks of ``block`` days keep the serial dependence and volatility clustering,
    and the two series are resampled jointly so their correlation is preserved
    (in the spirit of Ledoit & Wolf 2008).
    """
    a, b = np.asarray(r1, dtype=float), np.asarray(r2, dtype=float)
    diff = sharpe(a) - sharpe(b)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = _circular_block_indices(len(a), block, rng)
        boot[i] = sharpe(a[idx]) - sharpe(b[idx])
    p = float(np.mean(np.abs(boot - diff) >= abs(diff)))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"sharpe_diff": diff, "ci_low": float(lo), "ci_high": float(hi), "p_value": p}


def tests_vs_benchmark(sim, benchmark: str, **kw) -> pd.DataFrame:
    rows = []
    for name in sim.returns.columns:
        if name == benchmark:
            continue
        t = sharpe_diff_test(sim.returns[name], sim.returns[benchmark], **kw)
        rows.append({"strategy": name, **t})
    return pd.DataFrame(rows).set_index("strategy")


def period_table(sim, periods: dict[str, tuple[str, str]]) -> pd.DataFrame:
    """Cumulative return (%) of each strategy over stress periods."""
    out = {}
    for label, (s, e) in periods.items():
        part = sim.returns.loc[s:e]
        if len(part):
            out[label] = 100 * ((1 + part).prod() - 1)
    return pd.DataFrame(out).rename_axis("strategy")


STRESS_PERIODS = {
    "GFC Sep08-Feb09": ("2008-09-01", "2009-02-28"),
    "Euro crisis H2 2011": ("2011-07-01", "2011-12-31"),
    "COVID Feb-Mar 2020": ("2020-02-19", "2020-03-23"),
    "Rates 2022": ("2022-01-01", "2022-12-31"),
}
