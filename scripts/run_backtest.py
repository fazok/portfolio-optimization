"""Walk-forward back-test of all strategies, with report, figures and metrics.

Usage:
    python scripts/run_backtest.py --source yahoo       # 10 ETFs, 2005-2025
    python scripts/run_backtest.py --source synthetic   # simulated, no network
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from portopt import data as D
from portopt import plots
from portopt.backtest import default_strategies, simulate, target_weights
from portopt.frontier import frontier_summary, split_frontier
from portopt.metrics import (
    STRESS_PERIODS,
    performance_table,
    period_table,
    sharpe,
    tests_vs_benchmark,
)
from portopt.report import write_report

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = "Equal weight (1/N)"


def run(source: str, out_dir: Path, lookback: int = 756, cap: float = 0.4,
        cost_bps: float = 10.0, n_boot: int = 2000, n_synthetic: int = 4000):
    if source == "yahoo":
        R = D.to_returns(D.load_prices(cache_dir=ROOT / "data/raw"))
        assets = f"{R.shape[1]} ETFs ({', '.join(R.columns)})"
    else:
        R = D.make_synthetic(n_synthetic)
        assets = f"{R.shape[1]} simulated assets"
    print(f"[data] {source}: {len(R):,} days x {R.shape[1]} assets")

    t0 = time.time()
    targets = target_weights(R, default_strategies(), lookback=lookback, bounds=(0.0, cap))
    sim = simulate(R, targets, cost_bps=cost_bps)
    print(f"[backtest] {len(next(iter(targets.values())))} rebalances x {len(targets)} "
          f"strategies in {time.time() - t0:.0f}s")

    perf = performance_table(sim)
    tests = tests_vs_benchmark(sim, BENCHMARK, n_boot=n_boot)
    periods = period_table(sim, STRESS_PERIODS)
    costs = pd.DataFrame({
        bps: {k: sharpe(v) for k, v in simulate(R, targets, cost_bps=bps).returns.items()}
        for bps in (0, 10, 25, 50)
    }).T.rename_axis("cost_bps")
    fs = frontier_summary(R, bounds=(0.0, 1.0))
    sf = split_frontier(R, bounds=(0.0, 1.0))

    fig = out_dir / "figures"
    plots.frontier_plot(fs, fig / "frontier.png",
                        f"Markowitz frontier, {R.index[0].year}-{R.index[-1].year} (in-sample)")
    plots.split_frontier_plot(sf, fig / "frontier_oos.png")
    plots.wealth_plot(sim, fig / "wealth.png")
    plots.sharpe_ci_plot(tests, fig / "sharpe_vs_1n.png")
    plots.cost_plot(costs, fig / "costs.png")
    plots.weights_plot(sim, ["Max Sharpe (sample)", "Min variance (Ledoit-Wolf)",
                             "Hierarchical risk parity"], fig / "weights.png")

    ctx = {
        "source": source, "assets": assets, "lookback": lookback, "cap": cap,
        "cost_bps": cost_bps, "n_boot": n_boot,
        "start": str(sim.returns.index[0].date()), "end": str(sim.returns.index[-1].date()),
        "performance": perf, "tests": tests, "periods": periods, "costs": costs, "split": sf,
    }
    write_report(out_dir / "report.md", ctx)
    metrics = {
        "source": source, "period": [ctx["start"], ctx["end"]],
        "performance": perf.reset_index().to_dict(orient="records"),
        "tests_vs_1n": tests.reset_index().to_dict(orient="records"),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(perf[["return_%", "vol_%", "sharpe", "max_dd_%", "turnover_pa_%"]].round(2))
    print(tests.round(3))
    print(f"[report] {out_dir / 'report.md'}")
    return sim, perf, tests


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["yahoo", "synthetic"], default="yahoo")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--lookback", type=int, default=756)
    ap.add_argument("--cap", type=float, default=0.4)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    out = args.out or ROOT / ("reports" if args.source == "yahoo" else "reports/synthetic")
    run(args.source, out, lookback=args.lookback, cap=args.cap, cost_bps=args.cost_bps,
        n_boot=args.n_boot)


if __name__ == "__main__":
    main()
