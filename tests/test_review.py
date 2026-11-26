import numpy as np
import pandas as pd
import pytest

from yc import data
from yc import predict as P
from yc.reports import changes


def _part(dates, value):
    return pd.DataFrame({"date": pd.to_datetime(dates), "maturity": 1, "yield": value})


def test_merge_keeps_the_newest_edition_and_refuses_gaps():
    old = _part(["2020-01-31", "2020-02-29"], 6.0)
    new = _part(["2020-02-29", "2020-03-31"], 7.0)
    m = data.merge([old, new]).set_index("date")["yield"]
    assert m[pd.Timestamp("2020-02-29")] == 7.0 and m[pd.Timestamp("2020-01-31")] == 6.0
    with pytest.raises(ValueError, match="Mar 2020"):
        data.merge([_part(["2020-02-29"], 6.0), _part(["2020-04-30"], 6.0)])


def test_changes_uses_the_longest_consecutive_complete_run():
    idx = pd.date_range("2020-01-31", periods=8, freq="ME")
    w = pd.DataFrame({1: np.arange(8.0), 2: [1, 1, np.nan, 1, 1, 1, 1, 1]}, index=idx)
    d = changes(w, [1, 2])
    assert list(d.index) == list(idx[4:]) and np.allclose(d[1], 100)


def test_nw_t_with_zero_lags_is_whites_t():
    rng = np.random.default_rng(3)
    x = rng.normal(size=200)
    y = 0.5 * x + rng.normal(size=200) * (1 + np.abs(x))
    b, t, _ = P.nw_t(y, x[:, None], 0)
    Z = np.column_stack([np.ones(200), x])
    e = y - Z @ b
    inv = np.linalg.inv(Z.T @ Z)
    white = np.sqrt(np.diag(inv @ (Z.T * e**2) @ Z @ inv))
    assert np.allclose(t, b / white)
