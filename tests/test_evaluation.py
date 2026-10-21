import numpy as np

from rtrec.evaluation import bootstrap_ci, coverage, genre_diversity, paired_diff_ci


def test_bootstrap_ci_contains_mean_and_is_ordered():
    rng = np.random.default_rng(0)
    values = rng.binomial(1, 0.3, size=500).astype(float)
    mean, lo, hi = bootstrap_ci(values, n_resamples=500)
    assert lo <= mean <= hi
    assert abs(mean - 0.3) < 0.1


def test_paired_difference_detects_a_clear_gap():
    a = np.ones(300)
    b = np.zeros(300)
    mean, lo, hi = paired_diff_ci(a, b, n_resamples=200)
    assert mean == 1.0 and lo == 1.0 and hi == 1.0


def test_genre_diversity_and_coverage():
    genre = np.array([0, 0, 1, 2, 2])
    assert genre_diversity([0, 1, 3], genre) == 2 / 3
    assert genre_diversity([], genre) == 0.0
    assert coverage([[0, 1], [1, 2]], n_items=10) == 0.3
