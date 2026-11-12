"""Estimators and portfolio construction rules.

Inputs are annualised: ``mu`` (expected returns) and ``cov`` (covariance).
All long-only optimisers take ``bounds = (lower, upper)`` per asset and enforce
full investment (weights sum to 1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

from .data import TRADING_DAYS

# --- Estimators --------------------------------------------------------------------


def sample_moments(returns: pd.DataFrame | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    R = np.asarray(returns, dtype=float)
    return R.mean(axis=0) * TRADING_DAYS, np.cov(R, rowvar=False) * TRADING_DAYS


def ledoit_wolf_cov(returns) -> np.ndarray:
    """Ledoit & Wolf (2004) shrinkage towards a scaled identity (annualised)."""
    from sklearn.covariance import LedoitWolf

    return LedoitWolf().fit(np.asarray(returns, dtype=float)).covariance_ * TRADING_DAYS


def bayes_stein_mean(returns, cov: np.ndarray | None = None) -> np.ndarray:
    """Jorion (1986) Bayes-Stein mean: shrink sample means towards the min-variance mean.

    The shrinkage intensity grows when sample means are close together relative to
    their estimation noise, which is the usual situation for asset returns.
    """
    R = np.asarray(returns, dtype=float)
    T, n = R.shape
    mu = R.mean(axis=0)
    S = np.cov(R, rowvar=False) if cov is None else cov / TRADING_DAYS
    S_inv = np.linalg.pinv(S)
    ones = np.ones(n)
    mu_0 = ones @ S_inv @ mu / (ones @ S_inv @ ones)
    d = mu - mu_0
    phi = (n + 2) / ((n + 2) + T * d @ S_inv @ d)
    return ((1 - phi) * mu + phi * mu_0) * TRADING_DAYS


# --- Helpers -----------------------------------------------------------------------


def portfolio_vol(w: np.ndarray, cov: np.ndarray) -> float:
    return float(np.sqrt(w @ cov @ w))


def risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Share of portfolio variance coming from each asset (sums to 1)."""
    return w * (cov @ w) / (w @ cov @ w)


def _solve(objective, n: int, bounds, constraints=(), x0=None) -> np.ndarray:
    lo, hi = bounds
    if n * hi < 1 - 1e-12:
        raise ValueError(f"Upper bound {hi} too tight for {n} assets to sum to 1.")
    x0 = np.full(n, 1.0 / n) if x0 is None else x0
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}, *constraints]
    res = minimize(objective, x0, method="SLSQP", bounds=[(lo, hi)] * n, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-12})
    if not res.success:
        raise RuntimeError(f"Optimisation failed: {res.message}")
    w = np.clip(res.x, lo, hi)
    return w / w.sum()


# --- Allocation rules --------------------------------------------------------------


def equal_weight(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def min_variance_closed_form(cov: np.ndarray) -> np.ndarray:
    """Unconstrained global minimum variance: inv(S) 1 / (1' inv(S) 1)."""
    x = np.linalg.solve(cov, np.ones(len(cov)))
    return x / x.sum()


def min_variance(cov: np.ndarray, bounds=(0.0, 1.0)) -> np.ndarray:
    return _solve(lambda w: w @ cov @ w, len(cov), bounds)


def max_sharpe(mu: np.ndarray, cov: np.ndarray, rf: float = 0.0, bounds=(0.0, 1.0)) -> np.ndarray:
    """Tangency portfolio: maximise (w'mu - rf) / sqrt(w' S w)."""
    def neg_sharpe(w):
        return -(w @ mu - rf) / np.sqrt(w @ cov @ w)

    # Start from the best single-asset-weighted guess for robustness.
    x0 = np.clip(np.maximum(mu - rf, 0) + 1e-6, bounds[0], None)
    x0 = x0 / x0.sum()
    return _solve(neg_sharpe, len(mu), bounds, x0=np.clip(x0, bounds[0], bounds[1]))


def risk_parity(cov: np.ndarray) -> np.ndarray:
    """Equal risk contribution (Maillard, Roncalli & Teiletche 2010).

    Uses the convex formulation of Spinu (2013): minimise 0.5 y'Sy - (1/n) sum log y
    over y > 0, then normalise. Every asset then contributes 1/n of the variance.
    """
    n = len(cov)
    scale = np.sqrt(np.diag(cov))

    def f(y):
        return 0.5 * y @ cov @ y - np.sum(np.log(y)) / n

    def grad(y):
        return cov @ y - 1.0 / (n * y)

    res = minimize(f, 1.0 / scale, jac=grad, method="L-BFGS-B", bounds=[(1e-10, None)] * n,
                   options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 2000})
    return res.x / res.x.sum()


def hrp(cov: np.ndarray) -> np.ndarray:
    """Hierarchical Risk Parity (Lopez de Prado 2016).

    1. Cluster assets on the correlation distance sqrt((1 - rho) / 2).
    2. Reorder the covariance so similar assets sit together (quasi-diagonalisation).
    3. Recursively split the ordered list in two and allocate between the halves
       in inverse proportion to their (inverse-variance) cluster variance.
    """
    std = np.sqrt(np.diag(cov))
    corr = np.clip(cov / np.outer(std, std), -1, 1)
    dist = np.sqrt(np.clip((1 - corr) / 2, 0, None))
    np.fill_diagonal(dist, 0.0)
    tree = to_tree(linkage(squareform(dist, checks=False), method="single"))

    def cluster_var(idx: list[int]) -> float:
        sub = cov[np.ix_(idx, idx)]
        ivp = 1 / np.diag(sub)
        ivp /= ivp.sum()
        return float(ivp @ sub @ ivp)

    # Canonical leaf order. In the original algorithm, which child of a merge comes
    # first depends on how the assets happen to be listed, so the allocation does
    # too. Ordering children by (cluster size, cluster variance) removes that.
    def leaves(node) -> list[int]:
        if node.is_leaf():
            return [node.id]
        kids = [leaves(node.get_left()), leaves(node.get_right())]
        kids.sort(key=lambda c: (len(c), cluster_var(c)))
        return kids[0] + kids[1]

    order = leaves(tree)

    w = np.ones(len(cov))
    clusters = [order]
    while clusters:
        nxt = []
        for c in clusters:
            if len(c) < 2:
                continue
            left, right = c[: len(c) // 2], c[len(c) // 2:]
            v_l, v_r = cluster_var(left), cluster_var(right)
            alpha = 1 - v_l / (v_l + v_r)
            w[left] *= alpha
            w[right] *= 1 - alpha
            nxt += [left, right]
        clusters = nxt
    return w / w.sum()


def efficient_frontier(mu: np.ndarray, cov: np.ndarray, n_points: int = 40,
                       bounds=(0.0, 1.0)) -> pd.DataFrame:
    """Minimum-variance portfolios for a grid of target returns (long-only)."""
    w_mv = min_variance(cov, bounds)
    lo_ret = float(w_mv @ mu)
    # Highest return reachable under the bounds: fill the best assets up to the cap.
    hi_ret = float(_solve(lambda w: -(w @ mu), len(mu), bounds) @ mu)
    rows = []
    for target in np.linspace(lo_ret, hi_ret, n_points):
        cons = [{"type": "eq", "fun": lambda w, t=target: w @ mu - t}]
        try:
            w = _solve(lambda w: w @ cov @ w, len(mu), bounds, constraints=cons, x0=w_mv)
        except RuntimeError:
            continue
        rows.append({"ret": float(w @ mu), "vol": portfolio_vol(w, cov), "weights": w})
    return pd.DataFrame(rows)
