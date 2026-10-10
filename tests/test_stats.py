"""Statistics helpers of the P6 analysis (analysis/stats.py; docs/DECISIONS.md K-057 ... K-063). No result files."""

import math

import numpy as np
import pytest
from scipy import stats as st
from sklearn.metrics import roc_auc_score

from analysis import stats as X
from src.evaluate import binary_metrics


def test_cliffs_delta_known_values():
    assert X.cliffs_delta([1, 2, 3], [1, 1]) == pytest.approx(4 / 6)
    assert X.cliffs_delta([5, 6], [1, 2]) == 1.0
    assert X.cliffs_delta([1, 2], [5, 6]) == -1.0
    assert X.cliffs_delta([0, 0], [0, 0]) == 0.0


def test_holm_adjustment_and_nan_outside_family():
    adj = X.holm([0.01, 0.04, 0.03, np.nan])
    assert adj[:3] == pytest.approx([0.03, 0.06, 0.06])
    assert np.isnan(adj[3])
    assert X.holm([0.6, 0.9]) == pytest.approx([1.0, 1.0])


def test_exact_mwu_matches_scipy_without_ties():
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=4), rng.normal(size=5) + 0.5
    u, p = X.exact_mwu(x, y)
    ref = st.mannwhitneyu(x, y, alternative="two-sided", method="exact")
    assert u == ref.statistic and p == pytest.approx(ref.pvalue)


def test_minimum_attainable_p_values():
    assert X.min_p_mwu(4, 5) == pytest.approx(2 / 126)        # K-059: ROAD 4 stealthy vs 5 coarse
    assert X.exact_mwu([10, 11, 12, 13], [1, 2, 3, 4, 5])[1] == pytest.approx(2 / 126)
    assert X.min_p_wilcoxon(3) == 0.25                          # K-057: 3 primary AS-1 units
    # Holm over a family of 4 cannot bring the smallest attainable p below 0.05 (K-059)
    assert 4 * X.min_p_mwu(4, 5) > 0.05


def test_exact_mwu_with_ties_is_a_valid_probability():
    u, p = X.exact_mwu([0, 0, 0, 0.01], [0, 0, 0, 0, 0])
    assert 0 < p <= 1 and u == pytest.approx(10 + 2.5)          # 15 ties (1/2 each) + 5 wins


def test_block_length_rule():
    assert X.block_length(1000, "cube_root") == 10
    assert X.block_length(1001, "cube_root") == 11
    assert X.block_length(68, "cube_root") == 5
    assert X.block_length(7, 1) == 1


@pytest.mark.parametrize("n,L", [(1, 1), (10, 3), (68, 5), (100, 100)])
def test_moving_block_counts_keep_the_stratum_size(n, L):
    c = X.mbb_counts(n, L, 50, np.random.default_rng(0))
    assert c.shape == (50, n) and (c.sum(axis=1) == n).all() and (c >= 0).all()


def test_moving_block_of_full_length_is_the_sample_itself():
    assert (X.mbb_counts(12, 12, 5, np.random.default_rng(0)) == 1).all()


def test_blocks_are_contiguous_runs():
    c = X.mbb_counts(30, 30, 1, np.random.default_rng(0))   # one block covers everything
    assert (c == 1).all()
    c = X.mbb_counts(9, 3, 200, np.random.default_rng(2))    # three blocks of three
    # every drawn block contributes to 3 consecutive items: the count profile changes only at block borders
    assert (c.sum(axis=1) == 9).all()


def test_stratum_resampling_is_deterministic_per_name():
    a = X.mbb_counts(40, 4, 10, X.stratum_rng(7, 4, "road|seg|1"))
    b = X.mbb_counts(40, 4, 10, X.stratum_rng(7, 4, "road|seg|1"))
    c = X.mbb_counts(40, 4, 10, X.stratum_rng(7, 4, "road|seg|0"))
    assert (a == b).all() and not (a == c).all()


def test_seed_draws_differ_between_models_and_repeat():
    a, b = X.seed_draws(1, "cnn1d", 100, 3), X.seed_draws(1, "conv_ae", 100, 3)
    assert a.shape == (100, 3) and a.min() >= 0 and a.max() <= 2
    assert (a == X.seed_draws(1, "cnn1d", 100, 3)).all() and not (a == b).all()


def test_hierarchical_mean_over_drawn_seeds():
    theta = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])     # seed s has value s in every replicate
    draws = np.array([[0, 0, 0], [2, 2, 1]])
    assert X.hier(theta, draws) == pytest.approx([0.0, 5 / 3])


def _sample(n, seed, ties=True):
    rng = np.random.default_rng(seed)
    y = rng.random(n) < 0.4
    s = rng.normal(size=n) + y
    return y, (np.round(s, 1) if ties else s)


@pytest.mark.parametrize("ties", [True, False])
def test_unit_weights_reproduce_the_evaluation_metrics(ties):
    y, s = _sample(300, 3, ties)
    thr = 0.4
    m = X.Prepared(s, y, thr).metrics(np.ones((1, len(y)), dtype=np.float32))
    ref = binary_metrics(y, s > thr, s)
    for k in ("recall", "f1", "roc_auc"):
        assert m[k][0] == pytest.approx(ref[k], abs=1e-12)


def test_weighted_auc_equals_the_explicit_pair_sum():
    y, s = _sample(120, 4)
    C = np.random.default_rng(5).integers(0, 4, size=(25, len(y))).astype(np.float32)
    got = X.Prepared(s, y, 0.0).metrics(C)["roc_auc"]
    a, b = s[y][:, None], s[~y][None, :]
    K = (a > b) + 0.5 * (a == b)
    for i in range(len(C)):
        w = C[i].astype(float)
        exp = (w[y][:, None] * w[~y][None, :] * K).sum() / (y.sum() * (~y).sum())
        assert got[i] == pytest.approx(exp, abs=1e-12)


def test_weighted_auc_matches_sklearn_on_the_expanded_sample():
    y, s = _sample(80, 6)
    rng = np.random.default_rng(7)
    w = np.zeros(len(y), dtype=int)                 # stratified weights: class totals preserved
    for cls in (True, False):
        idx = np.flatnonzero(y == cls)
        w[idx] = np.bincount(rng.integers(0, len(idx), len(idx)), minlength=len(idx))
    got = X.Prepared(s, y, 0.0).metrics(w[None, :].astype(np.float32))
    ys, ss = np.repeat(y, w), np.repeat(s, w)
    assert got["roc_auc"][0] == pytest.approx(roc_auc_score(ys, ss), abs=1e-12)
    ref = binary_metrics(ys, ss > 0.0, ss)
    assert got["recall"][0] == pytest.approx(ref["recall"]) and got["f1"][0] == pytest.approx(ref["f1"])


def test_pareto_mask_minimises_and_maximises():
    pts = np.array([[1, 1, 0.9],      # best latency and size
                    [2, 2, 0.95],     # better recall -> kept
                    [2, 2, 0.9],      # dominated by row 0
                    [3, 1, 0.9]])     # dominated by row 0
    assert X.pareto_mask(pts, ["min", "min", "max"]).tolist() == [True, True, False, False]


def test_identical_points_do_not_dominate_each_other():
    assert X.pareto_mask(np.array([[1, 1], [1, 1]]), ["min", "max"]).all()


def test_robust_front_requires_domination_in_both_sessions():
    # candidate B is dominated by A in session 1 only (latency swaps) -> stays on the robust front (K-060)
    s1 = np.array([[1.0, 1.0, 0.5], [1.2, 1.0, 0.5]])
    s2 = np.array([[1.2, 1.0, 0.5], [1.0, 1.0, 0.5]])
    on = X.pareto_mask(s1, ["min", "min", "max"]) | X.pareto_mask(s2, ["min", "min", "max"])
    assert on.tolist() == [True, True]


def test_verdict_labels():
    assert X.verdict([True, True]) == "doğrulandı"
    assert X.verdict([True, False]) == "kısmen doğrulandı"
    assert X.verdict([False, False]) == "doğrulanmadı"
    assert X.verdict([True, True], partial_max=True) == "kısmen doğrulandı"


def test_absolute_difference_is_biased_upward_for_the_noisier_model():
    """K-063: with no true effect in either model, E|d_noisy| - E|d_quiet| > 0 (why the signed check exists)."""
    rng = np.random.default_rng(0)
    noisy, quiet = rng.normal(0, 0.02, 20000), rng.normal(0, 0.002, 20000)
    assert (np.abs(noisy) - np.abs(quiet)).mean() == pytest.approx(0.018 * math.sqrt(2 / math.pi), rel=0.05)
    assert abs((noisy - quiet).mean()) < 0.001
