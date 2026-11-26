import numpy as np
import pandas as pd

from yc import predict as P


def test_oos_uses_only_outcomes_known_by_the_forecast_date():
    idx = pd.date_range("2000-01-31", periods=200, freq="ME")
    x = pd.Series(np.sin(np.arange(200) / 7), index=idx)
    y = 2 * x + 0.5  # perfectly predictable
    y.iloc[150:] = 2 * x.iloc[150:] + 5.0  # the relation shifts at month 150
    f = P.oos(y, x.rename("s").to_frame(), h=12, start="2005-01-31", min_obs=24)
    t = idx[155]
    # at month 155 the fit has seen pairs up to month 143 only, so it still predicts the old line
    assert np.isclose(f.at[t, "model"], 2 * x[t] + 0.5)
    assert P.oos_r2(f.loc[:idx[149]]) > 0.99


def test_excess_return_of_a_par_bond_when_yields_stay_flat_is_the_term_spread():
    idx = pd.date_range("2000-01-31", periods=24, freq="ME")
    w = pd.DataFrame({1: 6.0, 9: 7.0, 10: 7.0}, index=idx)
    r = P.excess_return(w).dropna()
    assert np.allclose(r, 1.0)  # 7% coupon, no price change, minus the 6% 1-year yield
