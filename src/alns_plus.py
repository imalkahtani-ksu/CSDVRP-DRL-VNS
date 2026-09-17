"""
alns_plus.py — ALNS+ and DRL-ALNS for the C-SDVRP.

Both methods apply one of 18 destroy/repair actions per iteration
(3 removal operators x 2 insertion operators x 3 removal sizes of 10%, 20%
and 35% of n), improve the result with the shared RVND local search, and
accept it with simulated annealing:
    accept if Z_cand <= Z_curr, otherwise with probability
    exp(-(Z_cand - Z_curr) / T),  T_0 = 0.05 * Z_0,  T <- 0.9975 * T each step.

ALNS+    : roulette-wheel action selection with adaptive weights
           (scores 9 / 5 / 1 for new best / improving / accepted-worse,
           segments of 100 iterations, reaction factor 0.15, floor 0.01).
DRL-ALNS : the action is chosen by a PPO policy from the 17-feature state
           of _state(); reward = 100 * (Z_best before - Z_best after) / Z_0.
"""
import math
from collections import deque
from typing import Dict, Tuple

import numpy as np
import torch

from .solution      import Solution
from .clarke_wright import clarke_wright
from .search_common import CpuBudget, rvnd, destroy_repair, removal_count

REMOVALS = ["random", "worst", "related"]
REPAIRS = ["best", "regret"]
SIZE_TIERS = [0.10, 0.20, 0.35]
N_DESTROY, N_REPAIR, N_SIZE = 3, 2, 3
N_ACTIONS_PLUS = N_DESTROY * N_REPAIR * N_SIZE      # 18
STATE_DIM_PLUS = 17

T_INIT_FRAC = 0.05
COOLING_RATE = 0.9975


def decode(action: int) -> Tuple[int, int, int]:
    """action -> (destroy_idx, repair_idx, size_idx)."""
    size = action % N_SIZE
    rep = (action // N_SIZE) % N_REPAIR
    des = action // (N_SIZE * N_REPAIR)
    return des, rep, size


def _sa_accept(dz: float, T: float, rng) -> bool:
    if dz <= 0:
        return True
    if T < 1e-10:
        return False
    return bool(rng.random() < math.exp(-dz / T))


def _cv(xs):
    if not xs:
        return 0.0
    m = float(np.mean(xs))
    return 0.0 if m < 1e-9 else float(np.std(xs) / m)


def _state(cur: Solution, best: Solution, Z0, frac, T, T0, stag,
           recent_impr, recent_actions) -> np.ndarray:
    """17 features:
    0 Z_cur/Z0   1 routing/Z0   2 emission term/Z0   3 split term/Z0
    4 mean route load/Q   5 CPU progress   6 (Z_cur-Z_best)/Z_best (<=2)
    7 T/T0   8 split customers/n   9 log-scaled stagnation
    10 improvement rate (last 20)   11-13 removal-operator shares (last 30)
    14 share of greedy insertion (last 30)   15 CV of route lengths
    16 CV of route costs"""
    inst = cur.instance
    n = inst.n
    c = cur.components()
    Zc = c["total"]
    Zb = best.objective()
    sc = cur.split_counts()
    loads, lens, costs = [], [], []
    dm = inst._dist
    for r in cur.routes:
        if not r.stops:
            continue
        loads.append(r.load / inst.vtypes[r.vtype_idx]["capacity"])
        lens.append(len(r.stops))
        prev, rc = 0, 0.0
        for cid, _ in r.stops:
            rc += dm[prev, cid]
            prev = cid
        costs.append(rc + dm[prev, 0])
    s = np.zeros(STATE_DIM_PLUS, dtype=np.float32)
    s[0] = min(Zc / Z0, 3.0)
    s[1] = min(c["routing"] / Z0, 3.0)
    s[2] = min(c["emission"] / Z0, 3.0)
    s[3] = min(c["split"] / Z0, 3.0)
    s[4] = float(np.mean(loads)) if loads else 0.0
    s[5] = frac
    s[6] = min(2.0, (Zc - Zb) / max(Zb, 1e-9))
    s[7] = min(1.0, T / max(T0, 1e-9))
    s[8] = sum(1 for v in sc.values() if v > 1) / max(n, 1)
    s[9] = min(1.0, math.log1p(stag) / math.log1p(200))
    s[10] = (sum(recent_impr) / len(recent_impr)) if recent_impr else 0.0
    if recent_actions:
        for a in recent_actions:
            d, r, _ = decode(a)
            s[11 + d] += 1.0
            if r == 0:
                s[14] += 1.0
        s[11:15] /= len(recent_actions)
    s[15] = min(2.0, _cv(lens))
    s[16] = min(2.0, _cv(costs))
    return s


class _ALNSBase:
    name = "ALNS"

    def _select(self, st, rng, w):
        raise NotImplementedError

    def solve(self, instance, seed=None, time_limit=None, I_max=None,
              train=False) -> Dict:
        rng = np.random.default_rng(seed)
        budget = CpuBudget(time_limit)
        I_max = I_max or 10**9
        n = instance.n
        sol0 = clarke_wright(instance, seed=seed)
        Z0 = sol0.objective()
        cur = Solution(instance, rvnd(sol0.routes, instance, rng))
        best = cur
        Zc = cur.objective()
        T = T_INIT_FRAC * Zc
        T0 = T
        qs = [removal_count(n, f) for f in SIZE_TIERS]

        w = np.ones(N_ACTIONS_PLUS)
        score = np.zeros(N_ACTIONS_PLUS)
        use = np.zeros(N_ACTIONS_PLUS)
        stag = 0
        recent_impr = deque(maxlen=20)
        recent_actions = deque(maxlen=30)
        history = [(0.0, best.objective())]
        stored = 0
        it = 0
        self._prepare(seed)

        while it < I_max and not budget.expired():
            st = None
            if self.uses_state:
                st = _state(cur, best, Z0, budget.fraction(), T, T0, stag,
                            recent_impr, recent_actions)
            a, logp, val = self._select(st, rng, w)
            d, r, s = decode(a)
            Zb_before = best.objective()
            cand = destroy_repair(cur, REMOVALS[d], qs[s], REPAIRS[r], rng)
            if cand is not None:
                cand = Solution(instance, rvnd(cand.routes, instance, rng))
                Zn = cand.objective()
                accepted = _sa_accept(Zn - Zc, T, rng)
                if Zn < Zb_before - 1e-9:
                    best = cand
                    score[a] += 9
                    history.append((budget.elapsed(), Zn))
                elif Zn < Zc - 1e-9:
                    score[a] += 5
                elif accepted:
                    score[a] += 1
                if accepted:
                    cur, Zc = cand, Zn
            improved = best.objective() < Zb_before - 1e-9
            stag = 0 if improved else stag + 1
            use[a] += 1
            if (it + 1) % 100 == 0:
                for j in range(N_ACTIONS_PLUS):
                    if use[j] > 0:
                        w[j] = max(0.01, 0.85 * w[j] + 0.15 * score[j] / use[j])
                score[:] = 0
                use[:] = 0
            if train and self.uses_state:
                reward = 100.0 * (Zb_before - best.objective()) / max(Z0, 1e-9)
                self.agent.store(st, a, reward, val, logp, False)
                stored += 1
            recent_impr.append(1 if improved else 0)
            recent_actions.append(a)
            T *= COOLING_RATE
            it += 1

        cpu = budget.elapsed()
        train_stats = {}
        if train and self.uses_state and stored > 1:
            self.agent._dones[-1] = True
            train_stats = self.agent.update(last_value=0.0)
            self.agent.net.eval()
        return {"best": best, "Z_best": best.objective(), "Z_init": Z0,
                "history": history, "cpu_s": cpu, "wall_s": budget.wall(),
                "iters": it, "method": self.name, "train_stats": train_stats}

    def _prepare(self, seed):
        pass


class ClassicalALNSPlus(_ALNSBase):
    """ALNS+: adaptive roulette over the 18 actions."""
    name = "ALNS+"
    uses_state = False

    def _select(self, st, rng, w):
        return int(rng.choice(N_ACTIONS_PLUS, p=w / w.sum())), 0.0, 0.0


class DRLALNSPlus(_ALNSBase):
    """DRL-ALNS: PPO policy over the 18 actions."""
    name = "DRL-ALNS"
    uses_state = True

    def __init__(self, agent):
        self.agent = agent

    def _prepare(self, seed):
        torch.manual_seed(0 if seed is None else int(seed))
        self.agent.net.eval()

    def _select(self, st, rng, w):
        return self.agent.select_action(st)
