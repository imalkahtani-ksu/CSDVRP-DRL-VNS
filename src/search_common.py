"""
Pieces shared by every metaheuristic in the comparison.

CpuBudget   : time budget measured in process CPU seconds, so results do not
              depend on other programs running on the same machine.
rvnd        : the single RVND local search used by VNS, ALNS+ and the
              nine-action VNS family (Or-opt with chains of 1 and 2, plus
              2-opt* when n <= RVND_2OPT_MAX_N).
destroy_repair : remove about size_frac * n customers with one of the three
              removal operators, reinsert them, and return the new solution,
              or None when the reinsertion cannot respect N1/N2.
"""
import math
import time
from typing import List, Optional

import numpy as np

from . import operators as _ops
from .operators import (random_removal, worst_removal, related_removal,
                        best_insertion, regret_insertion,
                        or_opt, two_opt_star)
from .solution import Route, Solution

RVND_2OPT_MAX_N = 120

REMOVAL_OPS = {"random": random_removal, "worst": worst_removal,
               "related": related_removal}
REPAIR_OPS = {"best": best_insertion, "regret": regret_insertion}


class CpuBudget:
    """CPU-time budget for one run (seconds of process time)."""

    def __init__(self, seconds: Optional[float]):
        self.seconds = seconds
        self.t0 = time.process_time()
        self.w0 = time.perf_counter()

    def elapsed(self) -> float:
        return time.process_time() - self.t0

    def wall(self) -> float:
        return time.perf_counter() - self.w0

    def fraction(self) -> float:
        if not self.seconds:
            return 0.0
        return min(1.0, self.elapsed() / self.seconds)

    def expired(self) -> bool:
        return bool(self.seconds) and self.elapsed() >= self.seconds


def rvnd(routes: List[Route], inst, rng: np.random.Generator) -> List[Route]:
    """Random variable neighborhood descent, first improvement, restart on
    every improvement."""
    ops = [lambda r: or_opt(r, inst, max_chain=1),
           lambda r: or_opt(r, inst, max_chain=2)]
    if inst.n <= RVND_2OPT_MAX_N:
        ops.append(lambda r: two_opt_star(r, inst))
    cur = [x.copy() for x in routes if x.stops]
    cur_obj = Solution(inst, cur).objective()
    improved = True
    while improved:
        improved = False
        order = list(range(len(ops)))
        rng.shuffle(order)
        for idx in order:
            cand = ops[idx]([x.copy() for x in cur if x.stops])
            cand = [x for x in cand if x.stops]
            obj = Solution(inst, cand).objective()
            if obj < cur_obj - 1e-9:
                cur, cur_obj = cand, obj
                improved = True
                break
    return cur


def removal_count(n: int, size_frac: float) -> int:
    return max(2, min(int(math.ceil(size_frac * n)), max(2, n - 1)))


def destroy_repair(sol: Solution, removal: str, q: int, repair: str,
                   rng: np.random.Generator) -> Optional[Solution]:
    """Remove q customers with the named removal operator and reinsert them.
    Returns None if nothing was removed or the repair was infeasible."""
    op = REMOVAL_OPS[removal]
    saved = _ops._removal_size
    _ops._removal_size = lambda _n, _q=q: _q
    try:
        base = Solution(sol.instance, [r.copy() for r in sol.routes])
        routes, removed = op(base, rng)
    finally:
        _ops._removal_size = saved
    if not removed:
        return None
    routes = REPAIR_OPS[repair](sol.instance, routes, removed, rng=rng)
    if routes is None:
        return None
    return Solution(sol.instance, [r for r in routes if r.stops])
