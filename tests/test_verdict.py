import numpy as np
import pandas as pd

from yc import verdict as V


def test_curve_moves_level_and_slope_in_bp():
    idx = pd.date_range("2020-01-31", periods=3, freq="ME")
    w = pd.DataFrame({m: [6.0, 6.0 + 0.1 * m / 10, 6.0] for m in range(1, 11)}, index=idx)
    cm = V.curve_moves(w)
    assert np.isclose(cm["slope"].iloc[0], 9.0)  # 10y +10 bp, 1y +1 bp
    assert np.isclose(cm["level"].iloc[0], 5.5)


def test_stock_rate_betas_recovers_the_level_coefficient():
    idx = pd.date_range("2015-01-31", periods=60, freq="ME")
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"nifty": rng.normal(0, 0.05, 60), "level": rng.normal(0, 20, 60)},
                     index=idx)
    r = pd.DataFrame({"A": 1.0 * X["nifty"] - 0.0005 * X["level"] + rng.normal(0, 1e-4, 60)})
    b = V.stock_rate_betas(r, X)
    assert np.isclose(b["A"], -5.0, atol=0.1)  # -5% per 100 bp
