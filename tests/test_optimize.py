import numpy as np
import pytest

from portopt import optimize as opt
from portopt.data import make_synthetic

rng = np.random.default_rng(0)


def random_cov(n: int, seed: int = 0) -> np.ndarray:
    g = np.random.default_rng(seed)
    A = g.normal(size=(n, n))
    vols = g.uniform(0.05, 0.3, n)
    C = A @ A.T + n * np.eye(n)
    d = np.sqrt(np.diag(C))
    corr = C / np.outer(d, d)
    return corr * np.outer(vols, vols)


def test_min_variance_matches_closed_form_when_unconstrained():
    cov = random_cov(6)
    w_cf = opt.min_variance_closed_form(cov)
    w_num = opt.min_variance(cov, bounds=(-5.0, 5.0))
    assert np.allclose(w_num, w_cf, atol=1e-5)


def test_weights_sum_to_one_and_respect_bounds():
    R = make_synthetic(1500)
    mu, cov = opt.sample_moments(R)
    for w in [opt.min_variance(cov, (0, 0.4)), opt.max_sharpe(mu, cov, bounds=(0, 0.4))]:
        assert w.sum() == pytest.approx(1.0)
        assert w.min() >= -1e-9 and w.max() <= 0.4 + 1e-9


def test_upper_bound_too_tight_is_rejected():
    with pytest.raises(ValueError):
        opt.min_variance(random_cov(4), bounds=(0, 0.2))


def test_risk_parity_equalises_risk_contributions():
    cov = random_cov(8, seed=3)
    w = opt.risk_parity(cov)
    assert np.allclose(opt.risk_contributions(w, cov), 1 / 8, atol=1e-6)


def test_risk_parity_is_inverse_vol_for_uncorrelated_assets():
    vols = np.array([0.1, 0.2, 0.4])
    w = opt.risk_parity(np.diag(vols**2))
    inv = (1 / vols) / (1 / vols).sum()
    assert np.allclose(w, inv, atol=1e-6)


def test_tangency_has_the_highest_sharpe_on_the_frontier():
    mu = np.array([0.04, 0.06, 0.08, 0.10, 0.05])
    cov = random_cov(5, seed=1)
    w_t = opt.max_sharpe(mu, cov)
    sr_t = (w_t @ mu) / opt.portfolio_vol(w_t, cov)
    front = opt.efficient_frontier(mu, cov, n_points=25)
    assert (front["ret"] / front["vol"]).max() <= sr_t + 1e-6
    # The frontier is increasing in volatility and return.
    assert np.all(np.diff(front["ret"]) > -1e-9)


def test_hrp_is_invariant_to_asset_order():
    cov = random_cov(7, seed=4)
    w = opt.hrp(cov)
    perm = rng.permutation(7)
    w_perm = opt.hrp(cov[np.ix_(perm, perm)])
    assert np.allclose(w_perm, w[perm], atol=1e-10)
    assert w.sum() == pytest.approx(1.0) and (w > 0).all()


def test_hrp_on_two_uncorrelated_assets_is_inverse_variance():
    var = np.array([0.01, 0.04])
    w = opt.hrp(np.diag(var))
    assert np.allclose(w, (1 / var) / (1 / var).sum())


def test_bayes_stein_shrinks_means_towards_each_other():
    R = make_synthetic(800).to_numpy()
    mu_s, _ = opt.sample_moments(R)
    mu_bs = opt.bayes_stein_mean(R)
    assert mu_bs.std() < mu_s.std()
