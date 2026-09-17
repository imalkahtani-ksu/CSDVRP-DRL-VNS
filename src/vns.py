"""
vns.py — standard Variable Neighborhood Search baseline for the C-SDVRP.

Shaking follows the usual VNS schedule: neighborhood k = 1, ..., k_max with a
removal count that grows with k; k is reset to 1 after an improvement and
increased otherwise. Even k uses random removal and greedy best insertion,
odd k uses related removal and regret-2 insertion. Every shaken solution is
improved by the shared RVND and accepted only if it improves the incumbent.
"""
from typing import Dict

import numpy as np

from .instance      import CSDVRPInstance
from .solution      import Solution
from .clarke_wright import clarke_wright
from .search_common import CpuBudget, rvnd, destroy_repair


def shake_size(n: int, k: int) -> int:
    return min(max(2, k * max(2, n // 20)), max(3, n // 2))


class VNS:
    """Variable Neighborhood Search with RVND local search."""

    name = "VNS"

    def solve(self, instance: CSDVRPInstance, seed: int = None,
              time_limit: float = None, I_max: int = None,
              k_max: int = 6) -> Dict:
        rng = np.random.default_rng(seed)
        budget = CpuBudget(time_limit)
        I_max = I_max or 10**9

        sol0 = clarke_wright(instance, seed=seed)
        Z_init = sol0.objective()
        best = Solution(instance, rvnd(sol0.routes, instance, rng))

        history = [(0.0, best.objective())]
        it = 0
        k = 1
        while it < I_max and not budget.expired():
            removal = "random" if k % 2 == 0 else "related"
            repair = "best" if k % 2 == 0 else "regret"
            cand = destroy_repair(best, removal, shake_size(instance.n, k),
                                  repair, rng)
            it += 1
            if cand is not None:
                cand = Solution(instance, rvnd(cand.routes, instance, rng))
            if cand is not None and cand.objective() < best.objective() - 1e-9:
                best = cand
                k = 1
                history.append((budget.elapsed(), best.objective()))
            else:
                k = k + 1 if k < k_max else 1

        return {"best": best, "Z_best": best.objective(), "Z_init": Z_init,
                "history": history, "cpu_s": budget.elapsed(),
                "wall_s": budget.wall(), "iters": it, "method": self.name}
