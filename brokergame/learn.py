"""Multi-agent learning dynamics and equilibrium diagnostics.

Learning = simultaneous gradient play: every iteration both agents take one Adam step on
their OWN expected wealth, with pathwise gradients through the differentiable simulator.
A rest point of these dynamics is a Nash equilibrium within the policy class:

  mode="open"   : the opponent's actions are treated as a fixed process when an agent
                  differentiates (the paper's open-loop Nash, Definition 2.2).
  mode="closed" : gradients flow through the opponent's feedback rule, so each agent
                  anticipates how the other reacts to the state (closed-loop / feedback Nash,
                  which is the subgame-consistent notion).

Exploitability = how much an agent gains by best-responding (re-training alone) while the
other is frozen. Near zero for both agents is the certificate that we are at a Nash point.
"""
import copy
import time
import math
import torch

from .env import rollout, sample_exogenous


def _params(p):
    return [q for q in p.parameters() if q.requires_grad]


def train(cfg, pols, iters=1500, batch=1024, lr=3e-3, mode="closed", seed=0,
          learners=("B", "I"), log_every=100, log=print, tag=""):
    gen = torch.Generator().manual_seed(seed)
    opts = {i: torch.optim.Adam(_params(pols[i]), lr=lr, foreach=False) for i in learners}
    scheds = {i: torch.optim.lr_scheduler.CosineAnnealingLR(opts[i], iters, eta_min=lr * 0.03)
              for i in learners}
    other = {"B": "I", "I": "B"}
    hist = []
    t0 = time.time()
    for it in range(1, iters + 1):
        ex = sample_exogenous(cfg, batch, gen)
        grads, vals = {}, {}
        if mode == "closed":
            out = rollout(cfg, pols, ex)
            for i in learners:
                loss = -out["R" + i].mean()
                grads[i] = torch.autograd.grad(loss, _params(pols[i]), retain_graph=True)
                vals[i] = -loss.item()
        else:
            for i in learners:
                out = rollout(cfg, pols, ex, detach=(other[i],))
                loss = -out["R" + i].mean()
                grads[i] = torch.autograd.grad(loss, _params(pols[i]))
                vals[i] = -loss.item()
        for i in learners:
            for p, g in zip(_params(pols[i]), grads[i]):
                p.grad = g
            torch.nn.utils.clip_grad_norm_(_params(pols[i]), 10.0)
            opts[i].step()
            scheds[i].step()
        hist.append(vals)
        if log_every and (it % log_every == 0 or it == 1):
            s = " ".join(f"J{i}={vals[i]:+.4f}" for i in learners)
            log(f"[{tag}] it {it:5d} {s}  ({time.time() - t0:.0f}s)")
    return hist


@torch.no_grad()
def evaluate(cfg, pols, batch=20000, seed=12345, exact=False, record=False, chunks=4):
    """Mean and standard error of both agents' objectives on a fixed evaluation set."""
    gen = torch.Generator().manual_seed(seed)
    RB, RI, recs = [], [], []
    for _ in range(chunks):
        ex = sample_exogenous(cfg, batch // chunks, gen)
        out = rollout(cfg, pols, ex, exact=exact, record=record)
        RB.append(out["RB"]); RI.append(out["RI"])
        if record:
            recs.append(out["rec"])
    RB, RI = torch.cat(RB), torch.cat(RI)
    res = {"JB": RB.mean().item(), "JB_se": (RB.std() / math.sqrt(len(RB))).item(),
           "JI": RI.mean().item(), "JI_se": (RI.std() / math.sqrt(len(RI))).item()}
    if record:
        res["rec"] = {k: torch.cat([r[k] for r in recs]) for k in recs[0]}
    return res


@torch.no_grad()
def paired_diff(cfg, pols_a, pols_b, who, batch=20000, seed=777):
    """E[R_who(a) - R_who(b)] with common random numbers, and its standard error."""
    gen = torch.Generator().manual_seed(seed)
    ex = sample_exogenous(cfg, batch, gen)
    d = rollout(cfg, pols_a, ex)["R" + who] - rollout(cfg, pols_b, ex)["R" + who]
    return d.mean().item(), (d.std() / math.sqrt(len(d))).item()


def exploitability(cfg, pols, who, make_br=None, mode="closed", iters=600, batch=1024,
                   lr=2e-3, seed=99, log=print, tag=""):
    """Gain of `who` from best-responding while the opponent is frozen.

    closed : opponent keeps its feedback rule (reacts to the deviator's state changes).
    open   : opponent's realised equilibrium actions are frozen as a process; the deviator
             additionally observes the equilibrium-path states so that its policy class can
             represent the open-loop best response.
    """
    other = "I" if who == "B" else "B"
    if mode == "closed":
        br = copy.deepcopy(pols[who]) if make_br is None else make_br()
        test = {who: br, other: pols[other]}
        for p in _params(pols[other]):
            p.requires_grad_(False)
        train(cfg, test, iters=iters, batch=batch, lr=lr, mode="closed", seed=seed,
              learners=(who,), log_every=0)
        for p in pols[other].parameters():
            p.requires_grad_(True)
        gain, se = paired_diff(cfg, test, pols, who)
        return {"gain": gain, "se": se}

    # ---- open-loop ----
    assert make_br is not None, "open-loop BR needs a policy taking the extra observations"
    br = make_br()
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(_params(br), lr=lr, foreach=False)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=lr * 0.03)

    def frozen_paths(ex):
        with torch.no_grad():
            rec = rollout(cfg, pols, ex, record=True)["rec"]
        act = rec["eta"] if other == "I" else rec["v"]
        own = rec["v"] if who == "B" else rec["eta"]
        # extra obs: equilibrium-path states, opponent's and own equilibrium action.
        # make_br() is expected to start at "copy own equilibrium action" (gain 1 on the last
        # feature), so the best response is searched starting from the equilibrium.
        extra = torch.stack([rec["QB"], rec["QI"], rec["Y"], act, own], 2)
        return act, extra

    for it in range(iters):
        ex = sample_exogenous(cfg, batch, gen)
        act, extra = frozen_paths(ex)
        test = {who: br, other: pols[other]}
        out = rollout(cfg, test, ex, fixed={other: act}, extra={who: extra})
        loss = -out["R" + who].mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(_params(br), 10.0)
        opt.step(); sched.step()

    with torch.no_grad():
        gen = torch.Generator().manual_seed(777)
        ex = sample_exogenous(cfg, 20000, gen)
        act, extra = frozen_paths(ex)
        dev = rollout(cfg, {who: br, other: pols[other]}, ex, fixed={other: act},
                      extra={who: extra})["R" + who]
        base = rollout(cfg, pols, ex)["R" + who]
        d = dev - base
    return {"gain": d.mean().item(), "se": (d.std() / math.sqrt(len(d))).item()}


def train_coop(cfg, pols, lam, iters=800, batch=1024, lr=3e-3, seed=0, log_every=200,
               log=print, tag=""):
    """Cooperative (planner) problem: both policies jointly maximise lam*J^B + (1-lam)*J^I.
    Sweeping lam traces the Pareto frontier of the strategy profiles (no side payments)."""
    gen = torch.Generator().manual_seed(seed)
    params = _params(pols["B"]) + _params(pols["I"])
    opt = torch.optim.Adam(params, lr=lr, foreach=False)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=lr * 0.03)
    for it in range(1, iters + 1):
        ex = sample_exogenous(cfg, batch, gen)
        out = rollout(cfg, pols, ex)
        W = lam * out["RB"] + (1 - lam) * out["RI"]
        loss = -W.mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        opt.step(); sched.step()
        if log_every and (it % log_every == 0 or it == 1):
            log(f"[{tag} lam={lam:.2f}] it {it:5d} W={-loss.item():+.4f} "
                f"JB={out['RB'].mean().item():+.4f} JI={out['RI'].mean().item():+.4f}")


def train_frontier_point(cfg, pols, target_I, iters=800, batch=1024, lr=3e-3, mu=200.0,
                         seed=0, log_every=200, log=print, tag=""):
    """One point of the Pareto frontier by the epsilon-constraint method:
    maximise J^B + J^I subject to J^I = target_I (augmented Lagrangian on the batch means).
    Well posed even where weighted sums diverge, because total surplus is bounded."""
    gen = torch.Generator().manual_seed(seed)
    params = _params(pols["B"]) + _params(pols["I"])
    opt = torch.optim.Adam(params, lr=lr, foreach=False)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=lr * 0.03)
    nu, ema = 0.0, None
    for it in range(1, iters + 1):
        ex = sample_exogenous(cfg, batch, gen)
        out = rollout(cfg, pols, ex)
        JB, JI = out["RB"].mean(), out["RI"].mean()
        gap = JI - target_I
        loss = -(JB + JI) - nu * gap + 0.5 * mu * gap ** 2
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        opt.step(); sched.step()
        ema = gap.item() if ema is None else 0.9 * ema + 0.1 * gap.item()
        if it % 10 == 0:
            nu -= mu * ema * 0.1
        if log_every and (it % log_every == 0 or it == 1):
            log(f"[{tag} target={target_I:+.3f}] it {it:5d} JB={JB.item():+.4f} JI={JI.item():+.4f} nu={nu:+.3f}")
