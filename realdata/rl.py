"""Reinforcement-learning brokers trained on replayed real data.

* `features`              : the observation an RL broker sees at step k. The core set is the paper's
                            information set (Q^B, Q^I, Y, alpha, lagged xi) plus time and lagged eta.
                            `extended=True` adds the last 30-s price change and the last 5-min
                            return, which are available in real time but have no place in the model.
* `ReplayBrokerEnv`       : gymnasium env; one episode = one real hour drawn from the training set;
                            accounting identical to backtest.run. Reward = change in mark-to-market
                            wealth net of the (action-independent) client spread revenue, minus the
                            running inventory penalty, minus the terminal penalty at the last step,
                            divided by REWARD_SCALE. Summed over an episode it equals
                            (objective - client revenue) / REWARD_SCALE.
* `SB3Strategy`           : wraps a trained Stable-Baselines3 model as a backtest strategy.
* `DPSPolicy` / `train_dps`: differentiable policy search (deep-hedging style): a policy network
                            trained by backpropagating the episode objective through the replayed
                            backtest, with early stopping on validation hours.
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces

from . import backtest as BT

V_SCALE = 10000.0         # action a in [-1, 1] -> lit speed v = a * V_SCALE (BTC per hour); 2000 binds in bursts
REWARD_SCALE = 100.0


class Scales:
    def __init__(self, D, p, dt):
        self.q = 50.0
        self.qi = float(np.abs(np.cumsum(D["eta"] * dt, 1)).std()) + 1e-9
        self.y = max(p["Ups"] * 500.0 / max(p["Xi"], 1.0), 1e-6)
        self.a = float(D["alpha"].std()) + 1e-9
        self.x = float(D["xi"].std()) + 1e-9
        self.e = float(D["eta"].std()) + 1e-9
        dS = np.diff(D["S"], axis=1)
        self.ds = float(dS.std()) + 1e-9


def features(QB, QI, Y, alpha_k, xi_prev, eta_prev, k, n, sc, extended=False, dS_prev=None, r5=None):
    f = [np.full_like(QB, k / n), QB / sc.q, QI / sc.qi, Y / sc.y, alpha_k / sc.a,
         xi_prev / sc.x, eta_prev / sc.e]
    if extended:
        f += [dS_prev / sc.ds, r5 / (sc.ds * np.sqrt(10))]
    return np.clip(np.stack(f, -1), -10, 10).astype(np.float32)


def lagged_price_feats(S, k):
    dS_prev = S[:, k] - S[:, k - 1] if k >= 1 else np.zeros(S.shape[0])
    r5 = S[:, k] - S[:, max(k - 10, 0)]
    return dS_prev, r5


class ReplayBrokerEnv(gym.Env):
    metadata = {}

    def __init__(self, D, p, dt, sc, extended=False, seed=0, reward_scale=REWARD_SCALE):
        super().__init__()
        self.D, self.p, self.dt, self.sc, self.ext = D, p, dt, sc, extended
        self.rs = reward_scale
        self.E, self.n = D["eta"].shape
        nf = 9 if extended else 7
        self.observation_space = spaces.Box(-10, 10, (nf,), np.float32)
        self.action_space = spaces.Box(-1, 1, (1,), np.float32)
        self.rng = np.random.default_rng(seed)
        self.decay = np.exp(-p["Xi"] * dt)
        self.imp = p["Ups"] * ((1 - self.decay) / p["Xi"] if p["Xi"] > 0 else dt)

    def _obs(self):
        e, k = self.e, self.k
        eta_prev = self.D["eta"][e, k - 1] if k > 0 else 0.0
        xi_prev = self.D["xi"][e, k - 1] if k > 0 else 0.0
        kw = {}
        if self.ext:
            S = self.D["S"][e:e + 1]
            d1, r5 = lagged_price_feats(S, k)
            kw = {"dS_prev": d1, "r5": r5}
        return features(np.array([self.QB]), np.array([self.QI]), np.array([self.Y]),
                        np.array([self.D["alpha"][e, k]]), np.array([xi_prev]), np.array([eta_prev]),
                        k, self.n, self.sc, self.ext, **kw)[0]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.e = int(self.rng.integers(self.E))
        self.k = 0
        self.QB = self.QI = self.Y = self.X = 0.0
        return self._obs(), {}

    def step(self, action):
        p, dt, e, k = self.p, self.dt, self.e, self.k
        v = float(np.clip(action[0], -1, 1)) * V_SCALE
        eta, xi, S = self.D["eta"][e, k], self.D["xi"][e, k], self.D["S"][e]
        w0 = self.X + self.QB * (S[k] + self.Y)
        Se = S[k] + self.Y
        client = (p["b"] * eta ** 2 + p["c"] * xi ** 2) * dt
        self.X += (-(Se + p["a"] * v) * v + (Se + p["b"] * eta) * eta + (Se + p["c"] * xi) * xi) * dt
        pen = p["rB"] * self.QB ** 2 * dt
        self.QB += (v - eta - xi) * dt
        self.QI += eta * dt
        self.Y = self.Y * self.decay + self.imp * v
        self.k += 1
        w1 = self.X + self.QB * (S[self.k] + self.Y)
        r = w1 - w0 - client - pen
        done = self.k >= self.n
        if done:
            r -= p["phi"] * self.QB ** 2
        obs = self._obs() if not done else np.zeros(self.observation_space.shape, np.float32)
        return obs, float(r / self.rs), done, False, {}


class FeatureStrategy:
    """Base class: builds features inside backtest.run from (obs, k, eta, xi) and the price path."""

    def __init__(self, sc, n, extended=False, S=None):
        self.sc, self.n, self.ext, self.S = sc, n, extended, S

    def feats(self, obs, k, eta, xi):
        eta_prev = eta[:, k - 1] if k > 0 else np.zeros(len(obs))
        kw = {}
        if self.ext:
            d1, r5 = lagged_price_feats(self.S, k)
            kw = {"dS_prev": d1, "r5": r5}
        return features(obs[:, 0], obs[:, 1], obs[:, 2], obs[:, 3], obs[:, 4], eta_prev, k, self.n,
                        self.sc, self.ext, **kw)


class SB3Strategy(FeatureStrategy):
    """Works for feed-forward and recurrent (sb3-contrib RecurrentPPO) models."""

    def __init__(self, model, sc, n, extended=False, S=None):
        super().__init__(sc, n, extended, S)
        self.model = model

    def reset(self, E):
        self.state, self.start = None, np.ones(E, dtype=bool)

    def __call__(self, obs, k, eta, xi):
        f = self.feats(obs, k, eta, xi)
        if hasattr(self, "start"):
            a, self.state = self.model.predict(f, state=self.state, episode_start=self.start, deterministic=True)
            self.start = np.zeros(len(f), dtype=bool)
        else:
            a, _ = self.model.predict(f, deterministic=True)
        return np.clip(a[:, 0], -1, 1) * V_SCALE


# ------------------------------------------------------------------ differentiable policy search
class DPSPolicy(nn.Module):
    def __init__(self, nf, hidden=64, linear=False):
        super().__init__()
        self.net = nn.Linear(nf, 1) if linear else nn.Sequential(
            nn.Linear(nf, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        last = self.net if linear else self.net[-1]
        nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)

    def forward(self, f):
        return torch.tanh(self.net(f))[..., 0] * V_SCALE


def dps_objective(pol, D, p, dt, sc, extended=False, idx=None):
    """Differentiable replay of backtest.run; returns per-episode (objective - client revenue)."""
    eta = torch.tensor(D["eta"][idx] if idx is not None else D["eta"], dtype=torch.float32)
    xi = torch.tensor(D["xi"][idx] if idx is not None else D["xi"], dtype=torch.float32)
    al = torch.tensor(D["alpha"][idx] if idx is not None else D["alpha"], dtype=torch.float32)
    Snp = D["S"][idx] if idx is not None else D["S"]
    S = torch.tensor(Snp, dtype=torch.float32)
    E, n = eta.shape
    decay = float(np.exp(-p["Xi"] * dt))
    imp = p["Ups"] * ((1 - decay) / p["Xi"] if p["Xi"] > 0 else dt)
    QB = torch.zeros(E); QI = torch.zeros(E); Y = torch.zeros(E); J = torch.zeros(E)
    for k in range(n):
        eta_prev = eta[:, k - 1] if k > 0 else torch.zeros(E)
        xi_prev = xi[:, k - 1] if k > 0 else torch.zeros(E)
        f = [torch.full((E,), k / n), QB / sc.q, QI / sc.qi, Y / sc.y, al[:, k] / sc.a,
             xi_prev / sc.x, eta_prev / sc.e]
        if extended:
            d1 = S[:, k] - S[:, k - 1] if k >= 1 else torch.zeros(E)
            r5 = S[:, k] - S[:, max(k - 10, 0)]
            f += [d1 / sc.ds, r5 / (sc.ds * np.sqrt(10))]
        v = pol(torch.clamp(torch.stack(f, -1), -10, 10))
        QB_n = QB + (v - eta[:, k] - xi[:, k]) * dt
        Y_n = Y * decay + imp * v
        # wealth change net of client revenue: Q_{k+1}(S_{k+1}+Y_{k+1}) - Q_k(S_k+Y_k) - (S_k+Y_k)(Q_{k+1}-Q_k) - a v^2 dt
        J = J + QB_n * ((S[:, k + 1] + Y_n) - (S[:, k] + Y)) - p["a"] * v ** 2 * dt - p["rB"] * QB ** 2 * dt
        QB, QI, Y = QB_n, QI + eta[:, k] * dt, Y_n
    return J - p["phi"] * QB ** 2


class DPSStrategy(FeatureStrategy):
    def __init__(self, pol, sc, n, extended=False, S=None):
        super().__init__(sc, n, extended, S)
        self.pol = pol

    def __call__(self, obs, k, eta, xi):
        with torch.no_grad():
            return self.pol(torch.tensor(self.feats(obs, k, eta, xi))).numpy()


def train_dps(fit, val, p, dt, sc, extended=False, linear=False, seed=0, epochs=300, batch=64,
              lr=3e-3, patience=40, log=print, tag="DPS"):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    nf = 9 if extended else 7
    pol = DPSPolicy(nf, linear=linear)
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    best, best_state, bad = -np.inf, None, 0
    E = fit["eta"].shape[0]
    for ep in range(epochs):
        perm = rng.permutation(E)
        for i in range(0, E, batch):
            idx = perm[i:i + batch]
            loss = -dps_objective(pol, fit, p, dt, sc, extended, idx).mean() / REWARD_SCALE
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(pol.parameters(), 5.0)
            opt.step()
        with torch.no_grad():
            v = dps_objective(pol, val, p, dt, sc, extended).mean().item()
        if v > best:
            best, best_state, bad = v, copy.deepcopy(pol.state_dict()), 0
        else:
            bad += 1
        if ep % 20 == 0:
            log(f"[{tag}] epoch {ep:3d} val objective-net-of-client {v:+.1f} (best {best:+.1f})")
        if bad >= patience:
            break
    pol.load_state_dict(best_state)
    return pol, best
