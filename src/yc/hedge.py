"""Hedging a bond book's curve risk, sized from past data only.

A book is its exposure in rupees per basis point at each maturity (its key-rate DV01s, sign
chosen so a rise in yields loses money): its P&L in a month is -exposure . (yield changes in bp).
Hedges are positions in a few maturities (the hedge bonds), with exposures that offset the book's:
- duration: one bond, sized to the book's total exposure (right if the curve moves in parallel);
- pca-k: k bonds, sized so the hedged book has no exposure to the first k components;
- minimum variance: the same bonds as pca-3, sized to minimise the hedged book's variance under
  the past covariance (a regression hedge).
Linear in yields: convexity, carry and roll-down are left out.
"""

import numpy as np

from yc import pca as _pca


def hedge(book: np.ndarray, past: np.ndarray, mats: list[int], bonds: list[int],
          method: str) -> np.ndarray:
    """Hedge exposures (rupees per bp at each maturity in `mats`) for `book`, sized on `past`
    monthly changes (rows) of the same maturities."""
    idx = [mats.index(b) for b in bonds]
    A = np.zeros((len(mats), len(bonds)))
    A[idx, range(len(bonds))] = 1.0
    if method == "duration":
        h = np.array([book.sum()])
    elif method.startswith("pca"):
        k = len(bonds)
        L = _pca.pca(past)["loadings"][:, :k]
        h = np.linalg.solve(L.T @ A, L.T @ book)
    elif method == "minvar":
        S = np.cov(past, rowvar=False)
        h = np.linalg.solve(A.T @ S @ A, A.T @ S @ book)
    else:
        raise ValueError(method)
    return A @ h


def backtest(X: np.ndarray, mats: list[int], book: np.ndarray, bonds: list[int], method: str,
             window: int = 60) -> np.ndarray:
    """Each month from `window` on: size the hedge on the previous `window` months, then the
    hedged book's P&L that month (rupees)."""
    out = []
    for t in range(window, len(X)):
        H = hedge(book, X[t - window:t], mats, bonds, method)
        out.append(-(book - H) @ X[t])
    return np.array(out)
