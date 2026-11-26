import numpy as np

from yc import ns

M = list(range(1, 11))


def test_fit_recovers_betas_and_choose_lam_recovers_the_decay():
    rng = np.random.default_rng(0)
    betas = np.column_stack([rng.normal(7, 1, 50), rng.normal(-1, 0.5, 50),
                             rng.normal(0.5, 0.5, 50)])
    Y = betas @ ns.basis(M, 2.0).T
    b, fitted = ns.fit(Y, M, 2.0)
    assert np.allclose(b, betas) and np.allclose(fitted, Y)
    assert ns.choose_lam(Y, M) == 2.0


def test_basis_limits():
    B = ns.basis([0.001, 1000.0], 1.5)
    assert np.allclose(B[:, 0], 1) and np.isclose(B[0, 1], 1, atol=1e-3)
    assert B[1, 1] < 0.01 and B[0, 2] < 0.01 and B[1, 2] < 0.01  # curvature vanishes at both ends
