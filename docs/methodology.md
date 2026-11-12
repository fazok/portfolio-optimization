# Methodology

## 1. Data

The same universe as the companion `market-risk-var` project: 10 ETFs (SPY,
QQQ, IWM, EFA, EEM, VNQ, TLT, IEF, LQD, GLD), daily adjusted closes from Yahoo
Finance, 2005 to 2025. Returns are simple daily returns in USD. Annualisation
uses 252 days, and Sharpe ratios use a zero risk-free rate. All strategies are
compared on the same basis, so a cash rate would lower every Sharpe ratio
without changing much in the comparison.

## 2. Estimators

* **Sample moments**: mean and covariance of the last 756 days (3 years).
* **Ledoit-Wolf (2004)**: optimal shrinkage of the sample covariance towards a
  scaled identity (scikit-learn implementation).
* **Bayes-Stein (Jorion 1986)**: `mu_BS = (1 - phi) mu + phi mu_0 1`, where
  `mu_0` is the mean of the minimum-variance portfolio and
  `phi = (n+2) / ((n+2) + T (mu - mu_0 1)' S^-1 (mu - mu_0 1))`. Shrinkage is
  strong when the sample means are close together relative to their noise.

## 3. Allocation rules

All rules are long-only and fully invested. The optimisers (min variance, max
Sharpe) cap weights at 40%. Risk parity and HRP have closed-form-like
allocation rules and are left uncapped, as in their original papers. ERC never
went above 34% in one asset, but HRP reached 56% in IEF (7-10y Treasuries): its
inverse-variance logic concentrates in the lowest-volatility asset. This is a
known HRP trait and is kept visible rather than hidden by a cap.

* **Min variance**: `min w'Sw`.
* **Max Sharpe**: `max w'mu / sqrt(w'Sw)`, solved with SLSQP from a feasible
  start.
* **Risk parity (ERC)**: each asset contributes `1/n` of the variance. Solved
  with the convex reformulation of Spinu (2013),
  `min 0.5 y'Sy - (1/n) sum ln y_i`, then `w = y / sum y`.
* **HRP (López de Prado 2016)**:
  1. Single-linkage clustering on the distance `d = sqrt((1 - rho) / 2)`.
  2. Quasi-diagonal ordering of the leaves.
  3. Recursive bisection, splitting weight between the two halves in inverse
     proportion to their inverse-variance cluster variance.

**Implementation note on HRP.** In the original algorithm, which child of each
merge is placed first depends on the order in which assets are listed, and the
bisection then splits that list in half. Listing the same assets in a different
order can therefore give different weights. Here the children are ordered by
(cluster size, cluster variance), which makes the allocation depend only on the
covariance matrix. A unit test checks the invariance.

## 4. Walk-forward back-test

* **Rebalance dates:** the last trading day of each month, once 756 days of
  history exist.
* **No look-ahead:** weights set at the close of day *t* use returns up to *t*
  and earn returns from *t*+1. A test changes future returns and checks that
  past weights don't move.
* **Drift:** between rebalances, `w_{t+1} = w_t (1 + r_t) / (1 + w_t' r_t)`.
* **Costs:** `c x sum |w_target - w_drifted|` (one-way turnover) is deducted at
  each rebalance. The initial allocation is free for every strategy.

## 5. Inference: is a Sharpe ratio difference real?

Sharpe ratios of correlated strategies are compared with a **circular block
bootstrap**: 21-day blocks, 2,000 joint resamples of both return series. This
keeps volatility clustering, autocorrelation and the correlation between the
two strategies, in the spirit of Ledoit & Wolf (2008). The two-sided p-value is
`P(|SR*_diff - SR_diff| >= |SR_diff|)`, and the confidence interval is the
2.5–97.5% percentile range.

## 6. Estimation-error experiment

The frontier is estimated on the first half of the sample (2005 to mid-2015).
The same portfolios are then held through the second half, and their realised
return and volatility are plotted against the promised frontier and the
(unknowable) ex-post frontier. The comparison is unconstrained by the 40% cap,
to match the textbook frontier.

## 7. Discussion

* Every optimised rule beats 1/N on Sharpe ratio and drawdown, but none
  significantly. Kan & Zhou (2007) and DeMiguel et al. (2009) show that with
  realistic estimation windows the expected gain from optimisation is small
  relative to sampling noise.
* The benchmark matters. 1/N over this universe is 60% equities, so comparing
  Sharpe ratios mostly compares stock-heavy and bond-heavy portfolios over a
  period of falling rates. 2022 is the stress test of that bond allocation.
* With 10 assets and 756 observations, the covariance matrix is well estimated
  (Ledoit-Wolf shrinkage intensity of only 1–4% across windows). The noise is in the means, which
  is why Bayes-Stein helps and covariance shrinkage doesn't.

## 8. Extensions (research directions)

1. **Black-Litterman** with views from simple momentum or value signals.
2. **Nonlinear shrinkage** (Ledoit & Wolf 2017) and larger universes (e.g. 100
   stocks) where covariance estimation error dominates.
3. **Volatility targeting** and risk-based overlays using the VaR/ES engine from
   `market-risk-var`.
4. **Robust optimisation** (uncertainty sets on mu) vs resampled efficiency (Michaud).
5. **Multiple-testing control** across strategies (e.g. a Model Confidence Set
   on negative Sharpe, or a Bonferroni-Holm correction).

## References

* DeMiguel, V., Garlappi, L., Uppal, R. (2009). Optimal versus naive diversification: how inefficient is the 1/N portfolio strategy? *Review of Financial Studies*, 22(5).
* Jorion, P. (1986). Bayes-Stein estimation for portfolio analysis. *Journal of Financial and Quantitative Analysis*, 21(3).
* Kan, R., Zhou, G. (2007). Optimal portfolio choice with parameter uncertainty. *JFQA*, 42(3).
* Ledoit, O., Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2).
* Ledoit, O., Wolf, M. (2008). Robust performance hypothesis testing with the Sharpe ratio. *Journal of Empirical Finance*, 15(5).
* López de Prado, M. (2016). Building diversified portfolios that outperform out of sample. *Journal of Portfolio Management*, 42(4).
* Maillard, S., Roncalli, T., Teiletche, J. (2010). The properties of equally weighted risk contribution portfolios. *Journal of Portfolio Management*, 36(4).
* Markowitz, H. (1952). Portfolio selection. *Journal of Finance*, 7(1).
* Spinu, F. (2013). An algorithm for computing risk parity weights. SSRN 2297383.
