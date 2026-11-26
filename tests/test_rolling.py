import numpy as np

from yc import pca


def test_captured_is_one_for_the_full_basis_and_matches_explained_share():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (120, 5)) @ rng.normal(0, 1, (5, 5))
    p = pca.pca(X)
    assert np.isclose(pca.captured(X, p["loadings"]), 1.0)
    assert np.isclose(pca.captured(X, p["loadings"][:, :2]), p["share"][:2].sum())


def test_rolling_windows_score_the_next_months_with_past_shapes_only():
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, (100, 4))
    r = pca.rolling(X, list(range(100)), window=60, ahead=12)
    assert len(r) == 41 and r[0]["end"] == 59
    first = r[0]
    expect = pca.captured(X[60:72], pca.pca(X[:60])["loadings"][:, :3])
    assert np.isclose(first["next_captured"], expect)
    assert first["next_captured"] <= first["next_best"] + 1e-12
    assert "next_captured" not in r[-1]  # no 12 months left after the last window
