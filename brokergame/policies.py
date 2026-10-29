"""Agent policies.

* RiccatiPolicy    : the paper's closed-form equilibrium (benchmark, not learned).
* TimeGainPolicy   : action = sum_j g_j(t) * obs_j + g_0(t); the gains g(t) come from a small
                     network of time. This is exactly the class that contains the LQ
                     equilibrium, so learning in it is a sharp test.
* MLPPolicy        : general nonlinear feedback policy of (time, obs).
* RecurrentPolicy  : time-gain part + GRU memory over the observation history. Used by the
                     broker when she cannot see alpha and must infer it from flows and prices.
"""
import math
import numpy as np
import torch
import torch.nn as nn


def time_features(cfg, k):
    tau = (cfg.T - k * cfg.dt) / cfg.T                       # time left, in (0, 1]
    return [tau, math.sqrt(tau), cfg.dt / (tau * cfg.T + cfg.dt), 1.0]


N_TIME = 4


class Policy(nn.Module):
    out_dim = 1

    def reset(self, batch):
        pass

    def act(self, obs, k):
        raise NotImplementedError


class RiccatiPolicy(Policy):
    """Uses raw obs [QB, QI, Y, alpha, xi] (full info only)."""

    def __init__(self, gains):
        super().__init__()
        self.register_buffer("g", torch.tensor(np.asarray(gains), dtype=torch.float32))

    def act(self, obs, k):
        return (obs[:, :5] * self.g[k]).sum(1, keepdim=True)


class _Normaliser(nn.Module):
    def __init__(self, scales, act_scale):
        super().__init__()
        self.register_buffer("s", torch.tensor(scales, dtype=torch.float32))
        self.act_scale = act_scale

    def __call__(self, obs):
        return obs / self.s


class TimeGainPolicy(Policy):
    def __init__(self, cfg, scales, act_scale, out_dim=1, hidden=64):
        super().__init__()
        self.cfg, self.out_dim, self.d = cfg, out_dim, len(scales)
        self.norm = _Normaliser(scales, act_scale)
        self.net = nn.Sequential(nn.Linear(N_TIME, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, out_dim * (self.d + 1)))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        tf = torch.tensor([time_features(cfg, k) for k in range(cfg.n)], dtype=torch.float32)
        self.register_buffer("tf", tf)
        self._G = None

    def gains_table(self):
        return self.net(self.tf).view(self.cfg.n, self.out_dim, self.d + 1)

    def reset(self, batch):
        self._G = self.gains_table()

    def act(self, obs, k):
        x = torch.cat([self.norm(obs), torch.ones(obs.shape[0], 1)], 1)
        return self.norm.act_scale * torch.einsum("bd,od->bo", x, self._G[k])

    def init_copy_feature(self, j, out=0):
        """Start as action = obs_j (used to warm-start open-loop best responses)."""
        with torch.no_grad():
            idx = out * (self.d + 1) + j
            self.net[-1].bias[idx] = float(self.norm.s[j]) / self.norm.act_scale
        return self

    def raw_gains(self, out=0):
        """Gains in raw units on the observation vector (for comparison with Riccati)."""
        with torch.no_grad():
            G = self.gains_table()[:, out, :self.d] * self.norm.act_scale / self.norm.s
        return G.numpy()


class MLPPolicy(Policy):
    def __init__(self, cfg, scales, act_scale, out_dim=1, hidden=64):
        super().__init__()
        self.cfg, self.out_dim = cfg, out_dim
        self.norm = _Normaliser(scales, act_scale)
        d = len(scales)
        self.lin = TimeGainPolicy(cfg, scales, act_scale, out_dim, hidden)
        self.net = nn.Sequential(nn.Linear(d + N_TIME, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, out_dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def reset(self, batch):
        self.lin.reset(batch)

    def act(self, obs, k):
        tf = torch.tensor(time_features(self.cfg, k)).expand(obs.shape[0], N_TIME)
        nl = self.net(torch.cat([self.norm(obs), tf], 1))
        return self.lin.act(obs, k) + self.norm.act_scale * nl


class RecurrentPolicy(Policy):
    def __init__(self, cfg, scales, act_scale, out_dim=1, hidden=32):
        super().__init__()
        self.cfg, self.out_dim, self.hidden = cfg, out_dim, hidden
        self.norm = _Normaliser(scales, act_scale)
        d = len(scales)
        self.lin = TimeGainPolicy(cfg, scales, act_scale, out_dim)
        self.cell = nn.GRUCell(d + N_TIME, hidden)
        self.head = nn.Sequential(nn.Linear(hidden + d + N_TIME, 64), nn.Tanh(),
                                  nn.Linear(64, out_dim))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.h = None

    def reset(self, batch):
        self.lin.reset(batch)
        self.h = torch.zeros(batch, self.hidden)

    def act(self, obs, k):
        tf = torch.tensor(time_features(self.cfg, k)).expand(obs.shape[0], N_TIME)
        z = torch.cat([self.norm(obs), tf], 1)
        self.h = self.cell(z, self.h)       # memory includes the current observation
        nl = self.head(torch.cat([self.h, z], 1))
        return self.lin.act(obs, k) + self.norm.act_scale * nl
