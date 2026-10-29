"""Counterfactual backtest of a broker on real client flow and real prices.

At step k (bin start t_k) the broker sees only information available in real time:
  Q^B_k, Q^I_k (inventories from flows up to k-1), her own impact state Y_k,
  alpha_k (signal from informed flow up to k-1), xi_{k-1} (last bin's uninformed flow).
She chooses her lit speed v_k. Then the real client flows of bin k arrive, inventories move, and her
own transient impact Y is added to the real price path (she is hypothetical, so her impact is modelled):
  client trades at S_k + Y_k (+ b*eta_k, c*xi_k premia), lit trades at S_k + Y_k + a*v_k.
"""
import numpy as np
import torch

V_CLIP_MULT = 50.0


def run(strategy, eta, xi, alpha, S, p, dt):
    """eta, xi, alpha: (E, n); S: (E, n+1). Returns per-episode metrics."""
    E, n = eta.shape
    QB = np.zeros(E); QI = np.zeros(E); Y = np.zeros(E); X = np.zeros(E)
    decay = np.exp(-p["Xi"] * dt)
    imp = p["Ups"] * ((1 - decay) / p["Xi"] if p["Xi"] > 0 else dt)
    xi_prev = np.zeros(E)
    vmax = V_CLIP_MULT * (np.abs(eta).mean() + np.abs(xi).mean())
    traded = np.zeros(E); run_pen = np.zeros(E); client_rev = np.zeros(E)
    if hasattr(strategy, "reset"):
        strategy.reset(E)
    for k in range(n):
        obs = np.stack([QB, QI, Y, alpha[:, k], xi_prev], 1)
        v = np.clip(strategy(obs, k, eta, xi), -vmax, vmax)
        Se = S[:, k] + Y
        X += (-(Se + p["a"] * v) * v + (Se + p["b"] * eta[:, k]) * eta[:, k]
              + (Se + p["c"] * xi[:, k]) * xi[:, k]) * dt
        client_rev += (p["b"] * eta[:, k] ** 2 + p["c"] * xi[:, k] ** 2) * dt
        traded += np.abs(v) * dt
        run_pen += p["rB"] * QB ** 2 * dt
        QB = QB + (v - eta[:, k] - xi[:, k]) * dt
        QI = QI + eta[:, k] * dt
        Y = Y * decay + imp * v
        xi_prev = xi[:, k]
    mtm = X + QB * (S[:, n] + Y)
    obj = mtm - p["phi"] * QB ** 2 - run_pen
    return {"mtm": mtm, "objective": obj, "terminal_inventory": QB, "lit_volume": traded,
            "client_revenue": client_rev}


# ------------------------------------------------------------------ strategies
class TorchPolicy:
    """Wraps a brokergame policy (Riccati, time-gain or MLP) acting on [QB, QI, Y, alpha, xi]."""

    def __init__(self, pol):
        self.pol = pol

    def reset(self, E):
        self.pol.reset(E)

    def __call__(self, obs, k, eta, xi):
        with torch.no_grad():
            return self.pol.act(torch.tensor(obs, dtype=torch.float32), k)[:, 0].numpy()


class NoSignal(TorchPolicy):
    """Same policy with the signal switched off (ablation: value of alpha)."""

    def __call__(self, obs, k, eta, xi):
        o = obs.copy(); o[:, 3] = 0.0
        return super().__call__(o, k, eta, xi)


class NoInformedInventory(TorchPolicy):
    """Same policy with Q^I set to zero (ablation: the model's assumption that the informed client
    unwinds his position by the close)."""

    def __call__(self, obs, k, eta, xi):
        o = obs.copy(); o[:, 1] = 0.0
        return super().__call__(o, k, eta, xi)


def internalise(T_frac_liq=0.05, n=None):
    """Hold all client flow; liquidate linearly over the last 5% of the hour."""
    def f(obs, k, eta, xi):
        start = int((1 - T_frac_liq) * n)
        if k < start:
            return np.zeros(len(obs))
        steps_left = n - k
        return -obs[:, 0] / (steps_left * (1.0 / n))
    return f


def externalise(n=None):
    """Hedge last bin's client flow immediately and work off any residual inventory."""
    def f(obs, k, eta, xi):
        dt = 1.0 / n
        last = (eta[:, k - 1] + xi[:, k - 1]) if k > 0 else np.zeros(len(obs))
        return last - obs[:, 0] / max((n - k) * dt, dt)
    return f


def inventory_decay(n=None, tau0=0.05):
    """Signal-free inventory control: v = -Q^B / (T - t + tau0)."""
    def f(obs, k, eta, xi):
        t = k / n
        return -obs[:, 0] / (1.0 - t + tau0)
    return f
