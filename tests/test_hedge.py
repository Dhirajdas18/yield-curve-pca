import numpy as np

from yc import hedge, pca

M = list(range(1, 11))


def _past(seed=0):
    rng = np.random.default_rng(seed)
    mat = np.arange(1, 11)
    return (rng.normal(0, 30, (60, 1)) + rng.normal(0, 10, (60, 1)) * (mat - 5.5) / 4.5
            + rng.normal(0, 2, (60, 10)))


def test_duration_hedge_cancels_a_parallel_move():
    book = np.ones(10)
    H = hedge.hedge(book, _past(), M, [10], "duration")
    assert np.isclose(H.sum(), 10) and H[9] == 10
    assert np.isclose((book - H) @ np.full(10, 25.0), 0)


def test_pca_hedge_has_no_exposure_to_its_components_and_minvar_beats_it_in_sample():
    book, past = np.ones(10), _past(1)
    H = hedge.hedge(book, past, M, [2, 10], "pca")
    L = pca.pca(past)["loadings"][:, :2]
    assert np.allclose(L.T @ (book - H), 0, atol=1e-9)
    Hm = hedge.hedge(book, past, M, [2, 10], "minvar")
    var = lambda h: np.var(past @ (book - h))  # noqa: E731
    assert var(Hm) <= var(H) + 1e-9


def test_backtest_sizes_each_month_on_the_previous_window_only():
    X = _past(2)
    X = np.vstack([X, X[:5] * 3])
    r = hedge.backtest(X, M, np.ones(10), [10], "duration", window=60)
    assert len(r) == 5
    assert np.isclose(r[0], -(np.ones(10) - np.eye(10)[9] * 10) @ X[60])
