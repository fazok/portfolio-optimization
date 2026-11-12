"""Figures for the README and the report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .metrics import drawdown  # noqa: E402

NAVY, TEAL, AMBER, RED, GREY = "#1f3a5f", "#2a9d8f", "#e9a23b", "#c0392b", "#8a8f98"
PALETTE = ["#111111", NAVY, "#3d8bd9", RED, "#e07a5f", TEAL, AMBER, GREY, "#6c5b7b",
           "#7f8c3a"]
plt.rcParams.update({
    "figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "font.size": 9,
})


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def frontier_plot(fs: dict, path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    f = fs["frontier"]
    ax.plot(100 * f["vol"], 100 * f["ret"], color=NAVY, lw=2, label="Efficient frontier")
    a = fs["assets"]
    ax.scatter(100 * a["vol"], 100 * a["ret"], color=GREY, s=18)
    for name, row in a.iterrows():
        ax.annotate(name, (100 * row["vol"], 100 * row["ret"]), fontsize=7,
                    xytext=(3, 2), textcoords="offset points")
    for (label, (r, v, _)), c, m in zip(fs["points"].items(), [RED, TEAL, AMBER], "*os"):
        ax.scatter(100 * v, 100 * r, color=c, s=120 if m == "*" else 50, marker=m,
                   zorder=3, label=label)
    ax.set(xlabel="Annualised volatility (%)", ylabel="Annualised return (%)", title=title)
    ax.legend(fontsize=8)
    return _save(fig, path)


def split_frontier_plot(sf: dict, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    ax.plot(100 * sf["in_sample"]["vol"], 100 * sf["in_sample"]["ret"], color=NAVY, lw=2,
            label=f"Promised: frontier estimated up to {sf['split']}")
    ax.plot(100 * sf["realised"]["vol"], 100 * sf["realised"]["ret"], color=RED, lw=2,
            ls="--", marker="o", ms=3, label="Realised: same portfolios afterwards")
    ax.plot(100 * sf["ex_post"]["vol"], 100 * sf["ex_post"]["ret"], color=GREY, lw=1.2,
            label="Ex-post optimal frontier (unknowable)")
    for key, color, label in [("tangency", RED, "Tangency"), ("equal_weight", TEAL, "1/N")]:
        (r0, v0), (r1, v1) = sf[key]["promised"], sf[key]["realised"]
        ax.annotate("", xy=(100 * v1, 100 * r1), xytext=(100 * v0, 100 * r0),
                    arrowprops={"arrowstyle": "->", "color": color, "lw": 1.5})
        ax.scatter([100 * v0, 100 * v1], [100 * r0, 100 * r1], color=color, s=40, zorder=3,
                   label=f"{label}: promised -> realised")
    ax.set(xlabel="Annualised volatility (%)", ylabel="Annualised return (%)",
           title="Estimation error: in-sample promise vs out-of-sample reality")
    ax.legend(fontsize=7)
    return _save(fig, path)


def wealth_plot(sim, path: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    for name, c in zip(sim.returns.columns, PALETTE):
        r = sim.returns[name]
        lw = 2.2 if "1/N" in name else 1.2
        ax1.plot((1 + r).cumprod(), color=c, lw=lw, label=name)
        ax2.plot(100 * drawdown(r), color=c, lw=lw * 0.7)
    ax1.set(yscale="log", title="Growth of 1, net of 10 bp costs (log scale)")
    ax1.legend(fontsize=7, ncol=2)
    ax2.set(ylabel="Drawdown (%)")
    return _save(fig, path)


def weights_plot(sim, names: list[str], path: Path) -> Path:
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 3.4), sharey=True)
    for ax, name in zip(np.atleast_1d(axes), names):
        W = sim.weights[name]
        ax.stackplot(W.index, (100 * W).T.to_numpy(), labels=W.columns,
                     colors=PALETTE[: W.shape[1]], alpha=0.9)
        ax.set(title=name, ylim=(0, 100))
    np.atleast_1d(axes)[0].set_ylabel("Weight (%)")
    np.atleast_1d(axes)[-1].legend(fontsize=6, loc="center left", bbox_to_anchor=(1, 0.5))
    return _save(fig, path)


def cost_plot(cost_table: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, c in zip(cost_table.columns, PALETTE):
        ax.plot(cost_table.index, cost_table[name], marker="o", color=c, label=name,
                lw=2.2 if "1/N" in name else 1.2)
    ax.set(xlabel="Transaction cost (bp per unit of turnover)", ylabel="Sharpe ratio",
           title="Sensitivity to trading costs")
    ax.legend(fontsize=7)
    return _save(fig, path)


def sharpe_ci_plot(tests: pd.DataFrame, path: Path) -> Path:
    t = tests.sort_values("sharpe_diff")
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    colors = [TEAL if p < 0.05 and d > 0 else RED if p < 0.05 else GREY
              for d, p in zip(t["sharpe_diff"], t["p_value"])]
    ax.errorbar(t["sharpe_diff"], range(len(t)), fmt="none", ecolor=GREY,
                xerr=[t["sharpe_diff"] - t["ci_low"], t["ci_high"] - t["sharpe_diff"]])
    ax.scatter(t["sharpe_diff"], range(len(t)), color=colors, zorder=3, s=40)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels(t.index)
    ax.set(xlabel="Sharpe ratio minus 1/N (95% block-bootstrap interval)",
           title="Does optimisation beat 1/N?")
    return _save(fig, path)
