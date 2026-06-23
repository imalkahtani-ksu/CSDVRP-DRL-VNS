"""
vns.py — Variable Neighborhood Search baseline for the C-SDVRP.

A split-delivery-aware VNS with Random Variable Neighborhood Descent (RVND)
local search, adapted from the multi-start VNS framework of Ferreira et al.
for split-delivery routing. Shares the Clarke--Wright initialization and the
exact objective of solution.py, providing a strong non-learning metaheuristic
baseline distinct from ALNS (no adaptive operator weighting, no SA acceptance;
deterministic descent with shaking).
"""

import time
import math
from typing import Dict, List

import numpy as np

from .instance      import CSDVRPInstance
from .solution      import Route, Solution
from .operators     import (random_removal, related_removal,
                            best_insertion, regret_insertion,
                            or_opt, two_opt_star)
from .clarke_wright import clarke_wright, _pick_vehicle_tight


def _rvnd(sol: Solution, rng: np.random.Generator) -> Solution:
    """Random Variable Neighborhood Descent: shuffle local-search operators,
    apply with first-improvement, restart on any improvement."""
    ls_ops = [
        lambda routes: or_opt(routes, sol.instance, max_chain=1),
        lambda routes: or_opt(routes, sol.instance, max_chain=2),
        lambda routes: two_opt_star(routes, sol.instance),
    ]
    cur = sol
    improved = True
    while improved:
        improved = False
        order = list(range(len(ls_ops)))
        rng.shuffle(order)
        for idx in order:
            routes = [r.copy() for r in cur.routes if r.stops]
            new_routes = ls_ops[idx](routes)
            cand = Solution(cur.instance, [r for r in new_routes if r.stops])
            if cand.objective() < cur.objective() - 1e-9:
                cur = cand
                improved = True
                break
    return cur


def _shake(sol: Solution, strength: int, rng: np.random.Generator) -> Solution:
    """Perturbation: remove a growing number of (related) customers and
    reinsert them greedily, escaping the current local optimum."""
    n = sol.instance.n
    q = min(max(2, strength * max(2, n // 20)), max(3, n // 2))
    # alternate removal operator by shake strength parity for diversity
    if strength % 2 == 0:
        routes, removed = random_removal(sol, rng)
    else:
        routes, removed = related_removal(sol, rng)
    if not removed:
        return sol
    # reinsert (regret on odd, best on even) for varied recombination
    repair = regret_insertion if strength % 2 else best_insertion
    routes = repair(sol.instance, routes, removed)
    routes = [r for r in routes if r.stops]
    return Solution(sol.instance, routes)


class VNS:
    """Variable Neighborhood Search with RVND local search."""

    def solve(self, instance: CSDVRPInstance, I_max: int = None,
              seed: int = None, time_limit: float = None,
              k_max: int = 6) -> Dict:
        rng   = np.random.default_rng(seed)
        I_max = I_max or 3000

        sol_curr = clarke_wright(instance, seed=seed)
        Z_init   = sol_curr.objective()
        sol_curr = _rvnd(sol_curr, rng)
        sol_best = sol_curr.copy()

        history = [sol_best.objective()]
        t0 = time.perf_counter()
        it = 0
        while it < I_max:
            if time_limit and (time.perf_counter() - t0) > time_limit:
                break
            k = 1
            while k <= k_max and it < I_max:
                cand = _shake(sol_best, k, rng)
                cand = _rvnd(cand, rng)
                it += 1
                if cand.objective() < sol_best.objective() - 1e-9:
                    sol_best = cand.copy()
                    k = 1                       # reset to first neighborhood
                else:
                    k += 1                      # widen the shake
                history.append(sol_best.objective())
            if time_limit and (time.perf_counter() - t0) > time_limit:
                break

        elapsed = time.perf_counter() - t0

        # Right-size vehicles to minimize emission (mirrors ALNS post-processing)
        for r in sol_best.routes:
            if r.stops:
                r.vtype_idx = _pick_vehicle_tight(sol_best.instance, r.load)
        sol_best.invalidate()

        return {
            "best":    sol_best,
            "Z_best":  sol_best.objective(),
            "Z_init":  Z_init,
            "history": history,
            "time_s":  elapsed,
            "iters":   it,
            "method":  "VNS",
        }
