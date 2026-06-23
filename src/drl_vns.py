"""
DRL-guided Variable Neighborhood Search for the C-SDVRP.

Standard VNS cycles its shake neighborhood with a fixed rule (k = 1,2,...,
k_max, reset on improvement). Here a PPO agent chooses, at each step from the
current search state, which shake neighborhood and strength to apply. The
RVND local search and the acceptance rule are the same as the plain VNS
baseline, so the comparison isolates the effect of the learned schedule.

Action space (9): {random, related, worst} removal x {small, medium, large}.
State (12 features): normalized current and best objective, gap to best,
search progress, stagnation counter, recent-improvement rate, split
fraction, and recent removal-strength usage.
"""
import time
import math
from typing import Dict, List

import numpy as np

from .instance      import CSDVRPInstance
from .solution      import Route, Solution
from . import operators as _ops
from .operators     import (random_removal, related_removal, worst_removal,
                            best_insertion, regret_insertion)
from .alns_plus     import _rvnd
from .clarke_wright import clarke_wright, _pick_vehicle_tight

REMOVE_OPS = [random_removal, related_removal, worst_removal]
SIZE_TIERS = [0.08, 0.18, 0.32]
N_ACTIONS_VNS = len(REMOVE_OPS) * len(SIZE_TIERS)     # 9
STATE_DIM_VNS = 12

def decode_vns(a):
    return a // len(SIZE_TIERS), a % len(SIZE_TIERS)   # (op_idx, size_idx)


def _shake(sol: Solution, op_idx: int, size_frac: float, rng):
    n = sol.instance.n
    q = max(2, min(int(math.ceil(size_frac * n)), max(2, n - 1)))
    op = REMOVE_OPS[op_idx]
    saved = _ops._removal_size
    _ops._removal_size = lambda _n, _q=q: _q
    try:
        routes, removed = op(sol, rng)
    finally:
        _ops._removal_size = saved
    if not removed:
        return sol
    repair = regret_insertion if size_idx_global[0] % 2 else best_insertion
    routes = repair(sol.instance, routes, removed)
    return Solution(sol.instance, [r for r in routes if r.stops])

# small module-global to alternate repair without widening the signature
size_idx_global = [0]


def _state(sol, sol_best, Z0, it, I_max, stagn, recent_impr, strength_hist):
    Z = sol.objective(); Zb = sol_best.objective()
    sc = sol.split_counts(); n = sol.instance.n
    nsplit = sum(1 for v in sc.values() if v > 1)
    s = np.zeros(STATE_DIM_VNS, dtype=np.float32)
    s[0] = min(Z / max(Z0, 1e-6), 3.0)
    s[1] = min(Zb / max(Z0, 1e-6), 3.0)
    s[2] = min((Z - Zb) / max(Zb, 1e-6), 1.0)
    s[3] = it / max(I_max, 1)
    s[4] = min(stagn / 100.0, 1.0)
    s[5] = recent_impr
    s[6] = nsplit / max(n, 1)
    s[7] = 1.0 - it / max(I_max, 1)
    s[8:11] = strength_hist          # 3 size-tier usage fractions (recent)
    s[11] = 1.0
    return s


class DRLVNS:
    """PPO-guided VNS: the agent selects the shake neighborhood each step."""

    def solve(self, instance, agent, I_max=None, seed=None,
              train=False, time_limit=None) -> Dict:
        rng = np.random.default_rng(seed)
        I_max = I_max or 3000
        sol0 = clarke_wright(instance, seed=seed)
        Z0 = sol0.objective()
        cur = Solution(instance, _rvnd([r.copy() for r in sol0.routes], instance, rng))
        best = cur.copy()
        stagn = 0
        recent = []                      # 1 if improved best in last steps
        strength_use = np.zeros(3)
        hist = []
        t0 = time.perf_counter()
        agent.net.eval()
        for it in range(I_max):
            if time_limit and time.perf_counter() - t0 > time_limit:
                break
            sh = strength_use / max(strength_use.sum(), 1.0)
            ri = float(np.mean(recent[-50:])) if recent else 0.0
            st = _state(cur, best, Z0, it, I_max, stagn, ri, sh)
            action, logp, val = agent.select_action(st)
            op_idx, size_idx = decode_vns(action)
            size_idx_global[0] = size_idx
            cand = _shake(cur, op_idx, SIZE_TIERS[size_idx], rng)
            cand = Solution(instance, _rvnd([r.copy() for r in cand.routes], instance, rng))
            improved_best = cand.objective() < best.objective() - 1e-9
            # VNS acceptance: descend on the incumbent
            reward = (cur.objective() - cand.objective()) / max(Z0, 1e-6)
            if cand.objective() < cur.objective() - 1e-9:
                cur = cand; stagn = 0
            else:
                stagn += 1
            if improved_best:
                best = cand.copy()
            recent.append(1 if improved_best else 0)
            strength_use[size_idx] += 1
            if train:
                agent.store(st, action, reward, val, logp, it == I_max - 1)
            hist.append(action)
        if train:
            agent.update(last_value=0.0); agent.net.eval()
        for r in best.routes:
            if r.stops: r.vtype_idx = _pick_vehicle_tight(instance, r.load)
        best.invalidate()
        return {"best": best, "Z_best": best.objective(), "Z_init": Z0,
                "time_s": time.perf_counter() - t0, "iters": it + 1,
                "method": "DRL-VNS", "op_history": hist}
