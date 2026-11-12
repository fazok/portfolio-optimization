"""Walk-forward back-test: strategies, rebalancing, weight drift and transaction costs.

At each month-end rebalance date *t*, a strategy sees only the returns of the
``lookback`` days up to and including *t* and sets target weights, which apply
from day *t*+1. Between rebalances the weights drift with prices. Trading from the
drifted to the target weights costs ``cost_bps`` per unit of turnover (one way).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import optimize as opt

# --- Strategies --------------------------------------------------------------------


@dataclass
class Strategy:
    name: str
    short: str

    def weights(self, window: pd.DataFrame, bounds) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class EqualWeight(Strategy):
    def __init__(self):
        super().__init__("Equal weight (1/N)", "1/N")

    def weights(self, window, bounds):
        return opt.equal_weight(window.shape[1])


class MinVariance(Strategy):
    def __init__(self, shrink: bool):
        self.shrink = shrink
        super().__init__("Min variance (Ledoit-Wolf)" if shrink else "Min variance (sample)",
                         "MinVar-LW" if shrink else "MinVar")

    def weights(self, window, bounds):
        cov = opt.ledoit_wolf_cov(window) if self.shrink else opt.sample_moments(window)[1]
        return opt.min_variance(cov, bounds)


class MaxSharpe(Strategy):
    def __init__(self, shrink: bool):
        self.shrink = shrink
        super().__init__("Max Sharpe (Bayes-Stein + LW)" if shrink else "Max Sharpe (sample)",
                         "MaxSR-shrunk" if shrink else "MaxSR")

    def weights(self, window, bounds):
        if self.shrink:
            cov = opt.ledoit_wolf_cov(window)
            mu = opt.bayes_stein_mean(window, cov)
        else:
            mu, cov = opt.sample_moments(window)
        return opt.max_sharpe(mu, cov, bounds=bounds)


class RiskParity(Strategy):
    def __init__(self):
        super().__init__("Risk parity (ERC)", "ERC")

    def weights(self, window, bounds):
        return opt.risk_parity(opt.ledoit_wolf_cov(window))


class HRP(Strategy):
    def __init__(self):
        super().__init__("Hierarchical risk parity", "HRP")

    def weights(self, window, bounds):
        return opt.hrp(opt.sample_moments(window)[1])


def default_strategies() -> list[Strategy]:
    return [EqualWeight(), MinVariance(False), MinVariance(True), MaxSharpe(False),
            MaxSharpe(True), RiskParity(), HRP()]


# --- Engine ------------------------------------------------------------------------


def rebalance_dates(index: pd.DatetimeIndex, lookback: int) -> pd.DatetimeIndex:
    """Last trading day of each month, once ``lookback`` days of history exist."""
    s = pd.Series(index, index=index)
    month_ends = s.groupby([index.year, index.month]).max()
    return pd.DatetimeIndex(month_ends[month_ends >= index[lookback - 1]].values)


def target_weights(returns: pd.DataFrame, strategies: list[Strategy], lookback: int = 756,
                   bounds=(0.0, 0.4)) -> dict[str, pd.DataFrame]:
    """Target weights of each strategy at each rebalance date (the expensive step)."""
    dates = rebalance_dates(returns.index, lookback)
    pos = returns.index.get_indexer(dates)
    out = {}
    for s in strategies:
        rows = [s.weights(returns.iloc[p - lookback + 1:p + 1], bounds) for p in pos]
        out[s.name] = pd.DataFrame(rows, index=dates, columns=returns.columns)
    return out


@dataclass
class SimResult:
    returns: pd.DataFrame  # daily net returns, one column per strategy
    turnover: pd.DataFrame  # one-way turnover at each rebalance date
    weights: dict[str, pd.DataFrame]  # target weights at rebalance dates


def simulate(returns: pd.DataFrame, targets: dict[str, pd.DataFrame],
             cost_bps: float = 10.0) -> SimResult:
    """Apply target weights with drift between rebalances and proportional costs.

    The first allocation is treated as the starting portfolio (no cost), so every
    strategy starts invested on the day after the first rebalance date.
    """
    R = returns.to_numpy()
    idx = returns.index
    net, turn = {}, {}
    for name, W in targets.items():
        pos = {p: i for i, p in enumerate(idx.get_indexer(W.index))}
        start = min(pos)
        w = W.iloc[0].to_numpy().copy()
        daily = np.zeros(len(idx) - start - 1)
        tvr = np.zeros(len(W))
        for k, t in enumerate(range(start + 1, len(idx))):
            r = R[t] @ w
            w = w * (1 + R[t]) / (1 + r)  # drift to end of day t
            cost = 0.0
            if t in pos:  # rebalance at the close of day t
                target = W.iloc[pos[t]].to_numpy()
                tvr[pos[t]] = np.abs(target - w).sum()
                cost = tvr[pos[t]] * cost_bps / 1e4
                w = target.copy()
            daily[k] = (1 + r) * (1 - cost) - 1
        net[name] = pd.Series(daily, index=idx[start + 1:])
        turn[name] = pd.Series(tvr, index=W.index)
    return SimResult(pd.DataFrame(net), pd.DataFrame(turn), targets)
