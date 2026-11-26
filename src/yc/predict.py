"""Does the curve predict? 12-month-ahead outcomes from today's slope and curvature, in-sample
(Newey-West t) and out of sample (expanding window, scored against the historical mean)."""

import numpy as np
import pandas as pd


def par_duration(y: pd.Series, n: int) -> pd.Series:
    """Modified duration of an n-year par bond at yield y (%), annual coupons (an approximation:
    Indian government bonds pay half-yearly)."""
    r = y / 100
    return (1 - (1 + r) ** -n) / r


def excess_return(w: pd.DataFrame, n: int = 10, h: int = 12) -> pd.Series:
    """Return over the next `h` months of an n-year par bond bought today, above the 1-year yield,
    in % (coupon plus the price change as its yield moves to the (n-1)-year yield in a year)."""
    yn, y1 = w[n], w[1]
    later = w[n - 1].shift(-h)
    return yn - par_duration(yn, n - 1) * (later - yn) - y1


def signals(w: pd.DataFrame) -> pd.DataFrame:
    """Slope (10y minus 1y) and curvature (2 x 5y minus 1y minus 10y), in percentage points."""
    return pd.DataFrame({"slope": w[10] - w[1], "curvature": 2 * w[5] - w[1] - w[10]})


def nw_t(y: np.ndarray, X: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray, float]:
    Z = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(Z, y, rcond=None)
    e = y - Z @ b
    ze = Z * e[:, None]
    S = ze.T @ ze
    for k in range(1, lags + 1):
        g = ze[k:].T @ ze[:-k]
        S += (1 - k / (lags + 1)) * (g + g.T)
    inv = np.linalg.inv(Z.T @ Z)
    se = np.sqrt(np.diag(inv @ S @ inv))
    r2 = 1 - (e @ e) / np.sum((y - y.mean()) ** 2)
    return b, b / se, float(r2)


def oos(y: pd.Series, X: pd.DataFrame, h: int = 12, start: str = "2006-01-31",
        min_obs: int = 96) -> pd.DataFrame:
    """At each month t from `start`: fit y on X using only pairs (s, s + h) with s + h <= t,
    forecast y at t from X at t; the benchmark forecast is the mean of those same past outcomes."""
    df = pd.concat([y.rename("y"), X], axis=1)
    rows = {}
    dates = df.index
    for i, t in enumerate(dates):
        if t < pd.Timestamp(start) or pd.isna(df.at[t, "y"]) or df.loc[t, X.columns].isna().any():
            continue
        past = df.iloc[: max(0, i - h + 1)].dropna()
        if len(past) < min_obs:
            continue
        Z = np.column_stack([np.ones(len(past)), past[X.columns].to_numpy()])
        b, *_ = np.linalg.lstsq(Z, past["y"].to_numpy(), rcond=None)
        rows[t] = {"y": df.at[t, "y"], "model": float(np.r_[1, df.loc[t, X.columns]] @ b),
                   "mean": float(past["y"].mean())}
    return pd.DataFrame(rows).T


def oos_r2(f: pd.DataFrame) -> float:
    return float(1 - ((f["y"] - f["model"]) ** 2).sum() / ((f["y"] - f["mean"]) ** 2).sum())
