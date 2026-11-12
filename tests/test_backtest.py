import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portopt.backtest import (
    EqualWeight,
    default_strategies,
    rebalance_dates,
    simulate,
    target_weights,
)
from portopt.data import make_synthetic
from portopt.metrics import max_drawdown, sharpe, sharpe_diff_test

ROOT = Path(__file__).resolve().parents[1]


def test_drift_turnover_and_costs_by_hand():
    idx = pd.bdate_range("2024-01-29", periods=4)  # Mon 29 Jan .. Thu 1 Feb
    R = pd.DataFrame({"A": [0.0, 0.10, 0.0, 0.0], "B": [0.0, 0.0, 0.0, 0.0]}, index=idx)
    W = pd.DataFrame([[0.5, 0.5], [0.5, 0.5]], index=[idx[0], idx[2]], columns=["A", "B"])
    sim = simulate(R, {"x": W}, cost_bps=100)
    # Day 2: A +10% on half the book -> +5%; weights drift to (0.55/1.05, 0.5/1.05).
    # Day 3 close: rebalance back to 50/50.
    drifted = np.array([0.55, 0.5]) / 1.05
    turnover = np.abs(np.array([0.5, 0.5]) - drifted).sum()
    assert sim.returns["x"].iloc[0] == pytest.approx(0.05)
    assert sim.turnover["x"].iloc[1] == pytest.approx(turnover)
    assert sim.returns["x"].iloc[1] == pytest.approx(-turnover * 0.01)


def test_zero_costs_equal_weight_matches_manual_rebalanced_portfolio():
    R = make_synthetic(900)
    T = target_weights(R, [EqualWeight()], lookback=252)
    net = simulate(R, T, cost_bps=0).returns.iloc[:, 0]
    assert net.notna().all() and len(net) == len(R) - R.index.get_loc(T["Equal weight (1/N)"]
                                                                      .index[0]) - 1


def test_rebalance_dates_are_month_ends_after_lookback():
    idx = pd.bdate_range("2020-01-01", "2020-12-31")
    d = rebalance_dates(idx, 60)
    assert all(d.to_series().dt.is_month_end | (d.to_series().dt.day >= 26))
    assert d[0] >= idx[59]


def test_no_lookahead_weights_ignore_future_returns():
    R = make_synthetic(1200)
    T1 = target_weights(R, default_strategies()[1:3], lookback=500)
    R2 = R.copy()
    R2.iloc[-50:] *= 5  # change only the last 50 days
    T2 = target_weights(R2, default_strategies()[1:3], lookback=500)
    cutoff = R.index[-51]
    for k in T1:
        assert np.allclose(T1[k].loc[:cutoff], T2[k].loc[:cutoff])


def test_sharpe_and_drawdown_basics():
    r = pd.Series([0.01, -0.02, 0.01, 0.0])
    assert max_drawdown(r) == pytest.approx(-0.02)
    assert sharpe(np.r_[np.full(10, 0.01), np.full(10, 0.02)]) > 0


def test_sharpe_test_does_not_reject_identical_strategies():
    r = pd.Series(np.random.default_rng(0).normal(0.0003, 0.01, 2000))
    out = sharpe_diff_test(r, r, n_boot=200)
    assert out["sharpe_diff"] == 0 and out["p_value"] == 1.0


def test_sharpe_test_detects_a_large_difference():
    g = np.random.default_rng(1)
    common = g.normal(0, 0.01, 3000)
    good = common + 0.002 + g.normal(0, 0.002, 3000)
    out = sharpe_diff_test(good, common, n_boot=500)
    assert out["sharpe_diff"] > 0 and out["p_value"] < 0.01


def test_full_pipeline_on_synthetic_data(tmp_path):
    spec = importlib.util.spec_from_file_location("run_backtest", ROOT / "scripts/run_backtest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sim, perf, tests = mod.run("synthetic", tmp_path, lookback=252, n_boot=100,
                               n_synthetic=900)
    assert len(perf) == 7 and len(tests) == 6
    assert (tmp_path / "report.md").exists() and (tmp_path / "figures/frontier_oos.png").exists()
    assert json.loads((tmp_path / "metrics.json").read_text())["source"] == "synthetic"
