"""Principal components of yield-curve moves, mirrored in yc.js.

Input: a T x N array (months x maturities). PCA here is the eigen-decomposition of the sample
covariance of the columns (each centred on its own mean, not standardised: all are yields in the
same units). Signs are fixed so results are comparable: the first component (level) has a positive
sum of loadings; the second (slope) rises from the shortest to the longest maturity; the third
(curvature) is negative in the middle (its middle loading is below the average of its ends).
"""

import numpy as np


def pca(X: np.ndarray) -> dict:
    X = np.asarray(X, float)
    C = np.cov(X, rowvar=False)
    vals, vecs = np.linalg.eigh(C)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    n = vecs.shape[0]
    if vecs[:, 0].sum() < 0:
        vecs[:, 0] *= -1
    if n > 1 and vecs[-1, 1] < vecs[0, 1]:
        vecs[:, 1] *= -1
    if n > 2:
        mid = vecs[n // 2, 2]
        if mid > (vecs[0, 2] + vecs[-1, 2]) / 2:
            vecs[:, 2] *= -1
    scores = (X - X.mean(axis=0)) @ vecs
    return {"values": vals, "share": vals / vals.sum(), "loadings": vecs, "scores": scores,
            "mean": X.mean(axis=0)}


def captured(X: np.ndarray, basis: np.ndarray) -> float:
    """Share of the variance of X (T x N, centred on its own mean) that lies in the span of the
    orthonormal columns of `basis` (N x k)."""
    Xc = np.asarray(X, float) - np.mean(X, axis=0)
    total = float((Xc ** 2).sum())
    return float(((Xc @ basis) ** 2).sum() / total)


def rolling(X, dates, window: int = 60, ahead: int = 12, k: int = 3) -> list[dict]:
    """For each window of `window` consecutive rows: the shares of the first k components, the
    cosine of each component with `ref` loadings (filled in by the caller), and how much of the
    next `ahead` rows' variance the window's first k components capture, against the most any k
    components could capture there (the next rows' own first k)."""
    X = np.asarray(X, float)
    out = []
    for e in range(window, len(X) + 1):
        p = pca(X[e - window:e])
        row = {"end": dates[e - 1], "share": p["share"][:k], "loadings": p["loadings"][:, :k]}
        if e + ahead <= len(X):
            nxt = X[e:e + ahead]
            row["next_captured"] = captured(nxt, p["loadings"][:, :k])
            row["next_best"] = float(pca(nxt)["share"][:k].sum())
        out.append(row)
    return out
