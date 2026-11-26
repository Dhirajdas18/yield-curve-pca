"""Nelson-Siegel curves with a fixed decay (Diebold and Li, 2006): each month,
y(tau) = b0 + b1 * f1(tau) + b2 * f2(tau), with f1 = (1 - e^(-tau/lam)) / (tau/lam) and
f2 = f1 - e^(-tau/lam), fitted by least squares; lam (years) is chosen on training months only."""

import numpy as np


def basis(mats, lam: float) -> np.ndarray:
    t = np.asarray(mats, float) / lam
    f1 = (1 - np.exp(-t)) / t
    return np.column_stack([np.ones_like(t), f1, f1 - np.exp(-t)])


def fit(Y: np.ndarray, mats, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Betas (T x 3) and fitted yields (T x N) for each row of Y."""
    B = basis(mats, lam)
    betas = np.linalg.lstsq(B, np.asarray(Y, float).T, rcond=None)[0].T
    return betas, betas @ B.T


GRID = np.arange(0.25, 10.01, 0.25)


def choose_lam(Y: np.ndarray, mats, grid=GRID) -> float:
    """Decay with the smallest root-mean-square fitting error on Y."""
    err = [np.sqrt(np.mean((Y - fit(Y, mats, g)[1]) ** 2)) for g in grid]
    return float(grid[int(np.argmin(err))])
