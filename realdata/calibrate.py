"""Calibrate the game's parameters on training episodes.

Units: time in hours (an episode is T = 1 hour), prices in quote currency (USDT),
quantities in base currency (ETH). Every estimator below uses training days only.
"""
import numpy as np


def ar1_to_ou(x, dt):
    """Fit x_{k+1} = phi x_k + e by least squares on a (n_episodes, n) array; map to OU (kappa, sigma)."""
    a, b = x[:, :-1].ravel(), x[:, 1:].ravel()
    phi = float(np.dot(a, b) / np.dot(a, a))
    phi = min(max(phi, 1e-4), 0.9999)
    resid = b - phi * a
    kappa = -np.log(phi) / dt
    sigma = float(resid.std() * np.sqrt(2 * kappa / (1 - phi ** 2)))
    return kappa, sigma, phi


def ewma(x, half_life_steps):
    """Causal EWMA along axis 1 (value at k uses data up to k)."""
    lam = 0.5 ** (1.0 / half_life_steps)
    out = np.zeros_like(x, dtype=float)
    acc = np.zeros(x.shape[0])
    for k in range(x.shape[1]):
        acc = lam * acc + (1 - lam) * x[:, k]
        out[:, k] = acc
    return out


def signal(eta, S, dt, half_life_steps, horizon_steps, coef=None):
    """Ex-ante signal alpha_k = coef * EWMA(informed flow)_{k-1}, in price units per hour.
    coef is fitted by regressing the forward price drift over `horizon_steps` on the lagged EWMA.
    Returns alpha (same shape as eta), coef, and the in-sample R^2 of the drift regression."""
    z = ewma(eta, half_life_steps)
    z = np.concatenate([np.zeros((z.shape[0], 1)), z[:, :-1]], 1)       # lag one step: no look-ahead
    n = S.shape[1] - 1
    h = horizon_steps
    fwd = (S[:, h:n + 1] - S[:, :n + 1 - h]) / (h * dt)                   # drift over the next h steps
    zz = z[:, :n + 1 - h]
    if coef is None:
        coef = float(np.dot(zz.ravel(), fwd.ravel()) / np.dot(zz.ravel(), zz.ravel()))
    pred = coef * zz
    r2 = 1 - np.var(fwd.ravel() - pred.ravel()) / np.var(fwd.ravel())
    return coef * z, coef, float(r2)


def impact(F, S, dt, J=20):
    """Obizhaeva-Wang transient impact from the price response to total signed flow F.
    Regress dS_k on F_k, ..., F_{k-J}; the cumulative response R_l = sum_{j<=l} g_j should follow
    Ups * dt * exp(-Xi * l * dt). Returns Ups, Xi and the fitted kernel."""
    dS = np.diff(S, axis=1)[:, J:]
    X = np.stack([F[:, J - j:F.shape[1] - j][:, :dS.shape[1]] for j in range(J + 1)], -1)
    X = X.reshape(-1, J + 1) * dt
    y = dS.ravel()
    g, *_ = np.linalg.lstsq(X, y, rcond=None)
    R = np.cumsum(g)
    Ups = float(g[0])
    pos = R > 0.05 * R[0]
    if Ups <= 0 or pos.sum() < 3:
        return max(Ups, 0.0), 0.0, g
    l = np.arange(J + 1)[pos] * dt
    slope = np.polyfit(l, np.log(R[pos] / R[0]), 1)[0]
    Xi = float(max(-slope, 0.0))
    return Ups, Xi, g


def roll_half_spread(prices):
    """Roll (1984) estimator of the half-spread from tick-by-tick price changes."""
    d = np.diff(prices)
    c = np.cov(d[1:], d[:-1])[0, 1]
    return float(np.sqrt(-c)) if c < 0 else 0.0


def calibrate(train, dt, fee_rate=0.001, half_life_steps=4, horizon_steps=10,
              liquidation_frac=0.05, tick_prices=None):
    """train: dict with arrays eta, xi, S of shape (episodes, n[+1]). Returns a parameter dict and
    diagnostics. Cost conventions:
      per-unit lit cost   = fee * mean price + half-spread  =: kappa_tc
      a = b = c = kappa_tc / v_ref,  v_ref = mean |total client flow|  (cost per unit at a typical speed)
      phi = psi = a / (liquidation_frac * T): terminal penalty equal to the cost of liquidating the
      inventory over the last `liquidation_frac` of the hour."""
    eta, xi, S = train["eta"], train["xi"], train["S"]
    T = 1.0
    sig_S = float(np.diff(S, axis=1).std() / np.sqrt(dt))
    k_x, s_x, phi_x = ar1_to_ou(xi, dt)
    alpha, coef, r2 = signal(eta, S, dt, half_life_steps, horizon_steps)
    k_a, s_a, phi_a = ar1_to_ou(alpha, dt)
    F = eta + xi
    Ups, Xi, g = impact(F, S, dt)
    hs = roll_half_spread(tick_prices) if tick_prices is not None else 0.0
    kappa_tc = fee_rate * float(S.mean()) + hs
    v_ref = float(np.abs(F).mean())
    a = kappa_tc / v_ref
    pen = a / (liquidation_frac * T)
    params = dict(T=T, a=a, b=a, c=a, phi=pen, psi=pen, rB=0.0, rI=0.0, Ups=Ups, Xi=Xi,
                  kappa_a=k_a, sig_a=s_a, kappa_x=k_x, sig_x=s_x, sig_S=sig_S)
    diag = dict(signal_coef=coef, signal_R2_train=r2, signal_half_life_steps=half_life_steps,
                signal_horizon_steps=horizon_steps, half_spread=hs, kappa_tc=kappa_tc,
                v_ref=v_ref, phi_xi=phi_x, phi_alpha=phi_a, impact_kernel=g.tolist(),
                mean_price=float(S.mean()))
    return params, diag
