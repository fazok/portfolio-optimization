"""Build and execute the walkthrough notebook (logic lives in the package).

    python scripts/build_notebook.py            # Yahoo data (cached after first run)
    python scripts/build_notebook.py synthetic  # offline
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
SOURCE = sys.argv[1] if len(sys.argv) > 1 else "yahoo"

CELLS = [
    ("md", "# Markowitz vs 1/N: walkthrough\n"
           "Frontier, estimation error, then a walk-forward back-test of seven allocation "
           "rules. All logic lives in the `portopt` package."),
    ("code", f'SOURCE = "{SOURCE}"\n'
             "import warnings; warnings.filterwarnings('ignore')\n"
             "import numpy as np, pandas as pd\n"
             "from portopt import data as D, optimize as opt\n"
             "pd.set_option('display.precision', 3)\n"
             "R = (D.to_returns(D.load_prices(cache_dir='../data/raw')) if SOURCE == 'yahoo'\n"
             "     else D.make_synthetic(4000))\n"
             "mu, cov = opt.sample_moments(R)\n"
             "pd.DataFrame({'mu_%': 100 * mu, 'vol_%': 100 * np.sqrt(np.diag(cov))}, "
             "index=R.columns)"),
    ("md", "## One covariance, five allocations"),
    ("code", "lw = opt.ledoit_wolf_cov(R)\n"
             "pd.DataFrame({\n"
             "    '1/N': opt.equal_weight(len(mu)),\n"
             "    'MinVar': opt.min_variance(lw, (0, 0.4)),\n"
             "    'MaxSharpe': opt.max_sharpe(mu, cov, bounds=(0, 0.4)),\n"
             "    'ERC': opt.risk_parity(lw),\n"
             "    'HRP': opt.hrp(cov),\n"
             "}, index=R.columns).mul(100).round(1)"),
    ("md", "Risk parity check: every asset contributes 10% of the variance."),
    ("code", "(100 * opt.risk_contributions(opt.risk_parity(lw), lw)).round(2)"),
    ("md", "## Estimation error: promised vs realised"),
    ("code", "from portopt.frontier import split_frontier\n"
             "sf = split_frontier(R)\n"
             "{k: {s: round(v[0] / v[1], 2) for s, v in sf[k].items()} "
             "for k in ('tangency', 'equal_weight')}"),
    ("md", "## Walk-forward back-test (monthly, 3-year window, 10 bp costs)"),
    ("code", "from portopt.backtest import default_strategies, target_weights, simulate\n"
             "from portopt.metrics import performance_table, tests_vs_benchmark\n"
             "sim = simulate(R, target_weights(R, default_strategies()), cost_bps=10)\n"
             "performance_table(sim)"),
    ("code", "tests_vs_benchmark(sim, 'Equal weight (1/N)', n_boot=1000)"),
]


def main():
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(s) if k == "md" else nbf.v4.new_code_cell(s)
                for k, s in CELLS]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                 "language": "python"}
    out = ROOT / "notebooks"
    out.mkdir(exist_ok=True)
    NotebookClient(nb, timeout=1200, resources={"metadata": {"path": str(out)}}).execute()
    nbf.write(nb, out / "walkthrough.ipynb")
    print("built notebooks/walkthrough.ipynb")


if __name__ == "__main__":
    main()
