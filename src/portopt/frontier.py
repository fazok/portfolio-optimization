"""Markowitz frontier analysis and the cost of estimation error.

``split_frontier`` estimates the frontier on the first half of the sample and then
measures how those same portfolios actually performed in the second half. The gap
between the promised (in-sample) and realised (out-of-sample) frontiers is the
estimation error that motivates shrinkage, risk-based allocation and 1/N.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import optimize as opt


def frontier_summary(returns: pd.DataFrame, bounds=(0.0, 1.0), n_points: int = 40) -> dict:
    mu, cov = opt.sample_moments(returns)
    front = opt.efficient_frontier(mu, cov, n_points, bounds)
    w_tan = opt.max_sharpe(mu, cov, bounds=bounds)
    w_mv = opt.min_variance(cov, bounds)
    w_ew = opt.equal_weight(len(mu))
    points = {
        "Tangency": w_tan, "Min variance": w_mv, "Equal weight": w_ew,
    }
    return {
        "frontier": front,
        "assets": pd.DataFrame({"ret": mu, "vol": np.sqrt(np.diag(cov))},
                               index=returns.columns),
        "points": {k: (float(w @ mu), opt.portfolio_vol(w, cov), w) for k, w in points.items()},
    }


def split_frontier(returns: pd.DataFrame, split: str | None = None, bounds=(0.0, 1.0),
                   n_points: int = 30) -> dict:
    """In-sample frontier (first half) vs its out-of-sample realisation (second half)."""
    split = split or returns.index[len(returns) // 2]
    first, second = returns.loc[:split], returns.loc[split:].iloc[1:]
    mu1, cov1 = opt.sample_moments(first)
    mu2, cov2 = opt.sample_moments(second)
    front_is = opt.efficient_frontier(mu1, cov1, n_points, bounds)
    realised = pd.DataFrame({
        "ret": [w @ mu2 for w in front_is["weights"]],
        "vol": [opt.portfolio_vol(w, cov2) for w in front_is["weights"]],
    })
    front_true = opt.efficient_frontier(mu2, cov2, n_points, bounds)
    w_tan = opt.max_sharpe(mu1, cov1, bounds=bounds)
    w_ew = opt.equal_weight(len(mu1))
    return {
        "split": str(pd.Timestamp(split).date()),
        "in_sample": front_is[["ret", "vol"]],
        "realised": realised,
        "ex_post": front_true[["ret", "vol"]],
        "tangency": {"promised": (w_tan @ mu1, opt.portfolio_vol(w_tan, cov1)),
                     "realised": (w_tan @ mu2, opt.portfolio_vol(w_tan, cov2))},
        "equal_weight": {"promised": (w_ew @ mu1, opt.portfolio_vol(w_ew, cov1)),
                         "realised": (w_ew @ mu2, opt.portfolio_vol(w_ew, cov2))},
    }
