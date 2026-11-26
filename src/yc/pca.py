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
