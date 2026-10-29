"""Exact benchmark: the paper's open-loop Nash equilibrium.

Solves the non-symmetric matrix Riccati equation (4.1) and the linear BSDE (4.2)
for OU signal alpha and OU uninformed flow xi, written as ell_t = L(t) s_t with
s = (alpha, xi). The equilibrium speeds are the first two rows of

    (v, eta, Z)^T = L(t) s_t + P(t) X_t,     X = (Q^B, Q^I, Y).

The paper only proves existence for restricted parameters (see the audit); here we
just integrate and report whether the solution stays finite.
"""
import numpy as np
from scipy.integrate import solve_ivp

from .config import GameConfig


def matrices(cfg: GameConfig):
    a, b, Ups, Xi = cfg.a, cfg.b, cfg.Ups, cfg.Xi
    vphi = cfg.phi - Ups / 2
    A = np.array([[0, 0, 0], [0, 0, 0], [0, 0, -Xi]], float)
    B = np.array([[1, -1, 0], [0, 1, 0], [Ups, 0, 0]], float)
    Ah = np.array([[(2 * cfg.rB + Xi * Ups) / (2 * a), 0, Xi / (2 * a)],
                   [0, cfg.rI / b, Xi / (2 * b)],
                   [-1, 0, 0]], float)
    Bh = np.array([[0, -Ups / (2 * a), -Xi ** 2 * Ups / (2 * a)],
                   [-Ups / (2 * b), 0, 0],
                   [0, 0, Xi]], float)
    G = np.diag([-vphi / a, -cfg.psi / b, 0.0])
    # forward drift b_t = Bs s_t, backward drift bhat_t = Bhs s_t, with s = (alpha, xi)
    Bs = np.array([[0, -1], [0, 0], [0, 0]], float)
    Bhs = np.array([[-1 / (2 * a), -Ups / (2 * a)], [-1 / (2 * b), 0], [0, 0]], float)
    K = np.diag([-cfg.kappa_a, -cfg.kappa_x])
    return A, B, Ah, Bh, G, Bs, Bhs, K


def solve(cfg: GameConfig, rtol=1e-9, atol=1e-11):
    A, B, Ah, Bh, G, Bs, Bhs, K = matrices(cfg)

    def rhs(t, y):
        P = y[:9].reshape(3, 3)
        L = y[9:].reshape(3, 2)
        dP = Ah + Bh @ P - P @ A - P @ B @ P
        dL = (Bh - P @ B) @ L + Bhs - P @ Bs - L @ K
        return np.concatenate([dP.ravel(), dL.ravel()])

    y0 = np.concatenate([G.ravel(), np.zeros(6)])
    sol = solve_ivp(rhs, [cfg.T, 0.0], y0, method="Radau", rtol=rtol, atol=atol,
                    dense_output=True)
    ok = sol.success and np.all(np.isfinite(sol.y))
    return sol, ok


class RiccatiSolution:
    """P(t), L(t) sampled on the left end points of the simulation grid."""

    def __init__(self, cfg: GameConfig):
        self.cfg = cfg
        sol, ok = solve(cfg)
        self.ok = ok
        if not ok:
            raise RuntimeError("Riccati solution blew up / integration failed")
        tk = np.arange(cfg.n) * cfg.dt
        Y = sol.sol(tk)                          # (15, n)
        self.P = Y[:9].T.reshape(cfg.n, 3, 3)
        self.L = Y[9:].T.reshape(cfg.n, 3, 2)
        self.max_abs = float(np.abs(sol.y).max())

    def gains(self, row):
        """Coefficients of row (0 = broker v, 1 = informed eta) on [QB, QI, Y, alpha, xi]."""
        return np.concatenate([self.P[:, row, :], self.L[:, row, :]], axis=1)  # (n, 5)
