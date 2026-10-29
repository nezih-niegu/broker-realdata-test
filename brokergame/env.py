"""Batched, differentiable simulator of the broker / informed / uninformed game.

Timing inside a step k (left point t_k):
  1. both agents observe their information set at t_k
  2. broker chooses lit speed v_k (and, with pricing, the quote m_k for the informed client)
  3. informed chooses eta_k (seeing m_k if pricing); uninformed sends xi_k (exogenous)
  4. inventories, impact Y and the price move to t_{k+1}

Rewards use the mark-to-market decomposition
    X_T + Q_T S_T = q_0 S_0 + sum_k Q_{k+1} dS_k - costs,
and drop the zero-mean martingale part Q_{k+1} sig_S dW_k (Q_{k+1} is decided before dW_k),
which is an unbiased variance reduction. Set exact=True to keep it.
"""
from collections import defaultdict
import math
import torch

from .config import GameConfig

OBS_NAMES = {
    ("full", "B"): ["QB", "QI", "Y", "alpha", "xi"],
    ("full", "I"): ["QB", "QI", "Y", "alpha", "xi"],
    ("anon", "B"): ["QB", "Y", "flow_prev", "dS_prev"],
    ("anon", "I"): ["QI", "Y", "alpha"],
    # decomposition of E4: only the broker loses information
    ("anonB", "B"): ["QB", "Y", "flow_prev", "dS_prev"],
    ("anonB", "I"): ["QB", "QI", "Y", "alpha", "xi"],
}


V_MAX = 2000.0  # speed bound (equilibrium speeds are O(10-100)); stops unstable searches


def _soft_clip(x):
    return V_MAX * torch.tanh(x / V_MAX)


def obs_names(cfg: GameConfig, who: str):
    names = list(OBS_NAMES[(cfg.info, who)])
    if who == "I" and cfg.pricing:
        names.append("quote")
    return names


def _ou_path(x0, kappa, sig, eps, dt):
    """Exact OU discretisation. eps: (B, n)."""
    Bsz, n = eps.shape
    e = math.exp(-kappa * dt)
    sd = sig * math.sqrt((1 - e * e) / (2 * kappa))
    out = torch.empty(Bsz, n + 1, dtype=eps.dtype)
    out[:, 0] = x0
    for k in range(n):
        out[:, k + 1] = out[:, k] * e + sd * eps[:, k]
    return out


def sample_exogenous(cfg: GameConfig, batch: int, gen: torch.Generator, dtype=torch.float32):
    n, dt = cfg.n, cfg.dt
    eps = torch.randn(3, batch, n, generator=gen, dtype=dtype)
    return {
        "alpha": _ou_path(0.0, cfg.kappa_a, cfg.sig_a, eps[0], dt),
        "xi": _ou_path(0.0, cfg.kappa_x, cfg.sig_x, eps[1], dt),
        "dW": eps[2] * math.sqrt(dt),
    }


def rollout(cfg: GameConfig, pols, ex, detach=(), fixed=None, extra=None,
            exact=False, record=False):
    """Simulate one batch.

    pols   : {"B": policy, "I": policy}; policy.reset(batch); policy.act(obs, k) -> (batch, d)
    detach : agents whose actions are treated as given processes (no gradient flows through
             them). Detaching the opponent gives the *open-loop* best-response gradient used in
             the paper's Nash concept; detaching nobody gives the *closed-loop* (feedback) one.
    fixed  : {"B"/"I": tensor (batch, n)} replaces that agent's actions by a recorded process.
    extra  : {"B"/"I": tensor (batch, n, d)} additional exogenous observations for that agent.
    """
    n, dt = cfg.n, cfg.dt
    al, xi, dW = ex["alpha"], ex["xi"], ex["dW"]
    Bsz = al.shape[0]
    zeros = torch.zeros(Bsz, dtype=al.dtype)
    QB = zeros + cfg.qB0
    QI = zeros + cfg.qI0
    Y = zeros + cfg.Y0
    RB = zeros.clone()
    RI = zeros.clone()
    flow_prev = zeros.clone()
    dS_prev = zeros.clone()
    decay = math.exp(-cfg.Xi * dt)
    imp = cfg.Ups * ((1 - decay) / cfg.Xi if cfg.Xi > 0 else dt)
    rec = defaultdict(list)
    for p in pols.values():
        p.reset(Bsz)

    for k in range(n):
        a_k, x_k = al[:, k], xi[:, k]
        if cfg.info == "full":
            full = torch.stack([QB, QI, Y, a_k, x_k], 1)
            oB, oI = full, full
        elif cfg.info == "anonB":
            oB = torch.stack([QB, Y, flow_prev, dS_prev], 1)
            oI = torch.stack([QB, QI, Y, a_k, x_k], 1)
        else:
            oB = torch.stack([QB, Y, flow_prev, dS_prev], 1)
            oI = torch.stack([QI, Y, a_k], 1)
        if extra is not None and "B" in extra:
            oB = torch.cat([oB, extra["B"][:, k]], 1)

        outB = pols["B"].act(oB, k)
        v = _soft_clip(outB[:, 0])
        if cfg.pricing:
            m = cfg.b_lo + (cfg.b_hi - cfg.b_lo) * torch.sigmoid(outB[:, 1])
            oI = torch.cat([oI, m[:, None]], 1)
        else:
            m = zeros + cfg.b
        if fixed is not None and "B" in fixed:
            v = fixed["B"][:, k]
        if "B" in detach:
            v, m = v.detach(), m.detach()
            if cfg.pricing:
                oI = torch.cat([oI[:, :-1], m[:, None]], 1)

        if extra is not None and "I" in extra:
            oI = torch.cat([oI, extra["I"][:, k]], 1)
        outI = pols["I"].act(oI, k)
        eta = _soft_clip(outI[:, 0])
        # outside option: informed may also trade directly in the lit market (second output)
        etaL = _soft_clip(outI[:, 1]) if cfg.outside_cost is not None else zeros
        if fixed is not None and "I" in fixed:
            eta = fixed["I"][:, k]
        if "I" in detach:
            eta, etaL = eta.detach(), etaL.detach()

        # state update
        QB_n = QB + (v - eta - x_k) * dt
        QI_n = QI + (eta + etaL) * dt
        Y_n = Y * decay + imp * (v + etaL)       # lit trades of either agent move the price
        dS = a_k * dt + (Y_n - Y)
        if exact:
            dS = dS + cfg.sig_S * dW[:, k]
        RB = RB + QB_n * dS + (-cfg.a * v ** 2 + m * eta ** 2 + cfg.c * x_k ** 2) * dt \
            - cfg.rB * QB ** 2 * dt
        RI = RI + QI_n * dS - m * eta ** 2 * dt - cfg.rI * QI ** 2 * dt
        if cfg.outside_cost is not None:
            RI = RI - cfg.outside_cost * etaL ** 2 * dt

        if record:
            for name, val in (("v", v), ("eta", eta), ("etaL", etaL), ("m", m), ("QB", QB), ("QI", QI),
                              ("Y", Y), ("alpha", a_k), ("xi", x_k)):
                rec[name].append(val.detach())
        flow_prev = (eta + x_k).detach() if "I" in detach else (eta + x_k)
        dS_prev = dS + (0 if exact else cfg.sig_S * dW[:, k])  # broker observes the real price
        QB, QI, Y = QB_n, QI_n, Y_n

    RB = RB - cfg.phi * QB ** 2
    RI = RI - cfg.psi * QI ** 2
    out = {"RB": RB, "RI": RI, "QB_T": QB, "QI_T": QI}
    if record:
        out["rec"] = {k: torch.stack(v, 1) for k, v in rec.items()}
    return out
