"""Curves and equities: NIFTY's and single stocks' monthly sensitivity to yield-curve moves, and
the curve as a regime input for NIFTY returns. Stock data from the local NSE archive
(YC_LAKE_DIR, default /path/to/lake), which starts in 2013."""

import os
from pathlib import Path

import numpy as np
import pandas as pd


def lake_dir() -> Path:
    return Path(os.environ.get("YC_LAKE_DIR", Path.home() / "data" / "lake"))


def nifty_monthly() -> pd.Series:
    """NIFTY's month-end return (price index, no dividends)."""
    n = pd.read_parquet(lake_dir() / "market.parquet").set_index("date")["nifty"].dropna()
    m = n.resample("ME").last()
    return (m / m.shift(1) - 1).iloc[1:].rename("nifty")


def curve_moves(w: pd.DataFrame) -> pd.DataFrame:
    """Each month's level move (the average change of the 1-10 year yields) and slope move (the
    change in 10y minus 1y), in basis points."""
    d = w[list(range(1, 11))].diff() * 100
    return pd.DataFrame({"level": d.mean(axis=1), "slope": d[10] - d[1]}).dropna()


def stock_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Month-end split-adjusted closes and point-in-time universe membership (in_uni at the
    month's last trading day) for archive stocks."""
    p = pd.read_parquet(lake_dir() / "panel.parquet",
                        columns=["date", "symbol", "adj_close", "in_uni"])
    px = p.pivot_table(index="date", columns="symbol", values="adj_close").resample("ME").last()
    uni = p.pivot_table(index="date", columns="symbol", values="in_uni",
                        aggfunc="last").resample("ME").last()
    return px, uni.fillna(False).astype(bool)


def stock_rate_betas(r: pd.DataFrame, X: pd.DataFrame, a=None, b=None,
                     min_months: int = 36) -> pd.Series:
    """Each stock's monthly return regressed on NIFTY's return and the level move over [a, b]:
    the level coefficient, as % return per 100 bp rise in yields."""
    out = {}
    for s_ in r:
        df = pd.concat([r[s_].loc[a:b].rename("y"), X], axis=1, sort=True).dropna()
        if len(df) < min_months:
            continue
        Z = np.column_stack([np.ones(len(df)), df[["nifty", "level"]].to_numpy()])
        out[s_] = np.linalg.lstsq(Z, df["y"].to_numpy(), rcond=None)[0][2] * 1e4
    return pd.Series(out)
