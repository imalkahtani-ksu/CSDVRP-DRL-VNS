"""
drl_vns.py — nine-action VNS family and the proposed DRL-VNS.

All methods in this module share the same search engine and differ only in
how the shake action is chosen:

    shake     : remove q customers with one of three removal operators
                (random, related, worst) and reinsert them with greedy best
                insertion; q is 8%, 18% or 32% of n (small, medium, large).
                Action a = 3 * removal_index + strength_index (9 actions).
    descent   : shared RVND local search.
    acceptance: the candidate replaces the incumbent only if it is strictly
                better (the incumbent is therefore also the best solution).

Action selectors
    policy    : PPO policy on the 12-feature state below   -> DRL-VNS
    cyclic    : standard VNS schedule over the 9 actions,
                ordered by strength then operator; reset
                after an improvement, advance otherwise    -> VNS-9
    small     : small strength, removal operator uniform   -> Fixed-small
    random    : uniform over the 9 actions                 -> Random-9
    roulette  : adaptive weights (score 9 on improvement,
                segments of 100 steps, reaction 0.15)      -> Roulette-9
    marginal  : i.i.d. draws from a fixed action
                distribution, ignoring the state           -> Marginal-9

State (all features in [0, 1] or close to it)
    0  Z_best / Z_CW                     quality relative to the start
    1  CPU time used / budget            search progress
    2  log(1+stag) / log(201)            steps since the last improvement
    3  improvements in the last 20 steps / 20
    4  10 * min(0.1, (Z_cand - Z_best) / Z_best) of the last rejected shake
    5  min(1, n / 200)                   instance size
    6  split customers / n               in the incumbent
    7  split-eligible customers / n      instance feature
    8-10 share of small / medium / large shakes in the last 20 steps
    11 mean route load / Q               in the incumbent

Reward: r_t = 100 * (Z_best before step - Z_best after step) / Z_CW >= 0.
"""
import math
from collections import deque
from typing import Dict, Optional

import numpy as np
import torch

from .instance      import CSDVRPInstance
from .solution      import Solution
from .clarke_wright import clarke_wright
from .search_common import CpuBudget, rvnd, destroy_repair, removal_count

REMOVALS = ["random", "related", "worst"]
STRENGTHS = [0.08, 0.18, 0.32]
STRENGTH_NAMES = ["small", "medium", "large"]
REPAIR = "best"
N_ACTIONS_VNS = 9
STATE_DIM_VNS = 12
# neighborhood order used by the cyclic schedule: by strength, then operator
CYCLIC_ORDER = [3 * o + s for s in range(3) for o in range(3)]


def decode_vns(a: int):
    return a // 3, a % 3          # (removal_index, strength_index)


def _state(best: Solution, Z0, frac, stag, recent_impr, last_gap, n,
           elig_frac, recent_str) -> np.ndarray:
    Zb = best.objective()
    sc = best.split_counts()
    s = np.zeros(STATE_DIM_VNS, dtype=np.float32)
    s[0] = min(Zb / max(Z0, 1e-9), 1.5)
    s[1] = frac
    s[2] = min(1.0, math.log1p(stag) / math.log1p(200))
    s[3] = (sum(recent_impr) / len(recent_impr)) if recent_impr else 0.0
    s[4] = last_gap
    s[5] = min(1.0, n / 200.0)
    s[6] = sum(1 for v in sc.values() if v > 1) / max(n, 1)
    s[7] = elig_frac
    if recent_str:
        for k in recent_str:
            s[8 + k] += 1.0
        s[8:11] /= len(recent_str)
    loads = [r.load / best.instance.vtypes[r.vtype_idx]["capacity"]
             for r in best.routes if r.stops]
    s[11] = float(np.mean(loads)) if loads else 0.0
    return s


class VNS9:
    """Nine-action VNS with a pluggable action selector."""

    def __init__(self, selector: str = "cyclic", agent=None,
                 marginal: Optional[np.ndarray] = None, name: str = None):
        assert selector in ("policy", "cyclic", "small", "random",
                            "roulette", "marginal")
        self.selector = selector
        self.agent = agent
        self.marginal = None if marginal is None else np.asarray(marginal, float)
        self.name = name or {"policy": "DRL-VNS", "cyclic": "VNS-9",
                             "small": "Fixed-small", "random": "Random-9",
                             "roulette": "Roulette-9",
                             "marginal": "Marginal-9"}[selector]

    def solve(self, instance: CSDVRPInstance, seed: int = None,
              time_limit: float = None, I_max: int = None,
              train: bool = False) -> Dict:
        sel = self.selector
        rng = np.random.default_rng(seed)
        if sel == "policy":
            torch.manual_seed(0 if seed is None else int(seed))
            self.agent.net.eval()
        budget = CpuBudget(time_limit)
        I_max = I_max or 10**9
        n = instance.n

        sol0 = clarke_wright(instance, seed=seed)
        Z0 = sol0.objective()
        best = Solution(instance, rvnd(sol0.routes, instance, rng))
        elig_frac = sum(1 for i in range(1, n + 1)
                        if int(instance.U_max[i]) > 1) / max(n, 1)
        qs = [removal_count(n, f) for f in STRENGTHS]

        w = np.ones(N_ACTIONS_VNS)
        seg_score = np.zeros(N_ACTIONS_VNS)
        seg_use = np.zeros(N_ACTIONS_VNS)
        k = 0
        stag = 0
        last_gap = 0.0
        recent_impr = deque(maxlen=20)
        recent_str = deque(maxlen=20)

        usage = np.zeros((3, 3, N_ACTIONS_VNS))     # phase x stagnation x action
        uses = np.zeros(N_ACTIONS_VNS)
        wins = np.zeros(N_ACTIONS_VNS)
        history = [(0.0, best.objective())]
        stored = 0
        it = 0

        while it < I_max and not budget.expired():
            frac = budget.fraction()
            st = None
            if sel == "policy":
                st = _state(best, Z0, frac, stag, recent_impr, last_gap, n,
                            elig_frac, recent_str)
                a, logp, val = self.agent.select_action(st)
            elif sel == "cyclic":
                a = CYCLIC_ORDER[k]
            elif sel == "small":
                a = 3 * int(rng.integers(3)) + 0
            elif sel == "random":
                a = int(rng.integers(N_ACTIONS_VNS))
            elif sel == "roulette":
                a = int(rng.choice(N_ACTIONS_VNS, p=w / w.sum()))
            else:
                a = int(rng.choice(N_ACTIONS_VNS, p=self.marginal))
            o_idx, s_idx = decode_vns(a)

            Zb = best.objective()
            cand = destroy_repair(best, REMOVALS[o_idx], qs[s_idx], REPAIR, rng)
            if cand is not None:
                cand = Solution(instance, rvnd(cand.routes, instance, rng))
            improved = cand is not None and cand.objective() < Zb - 1e-9

            phase = min(2, int(frac * 3))
            sbin = 0 if stag < 5 else (1 if stag < 20 else 2)
            usage[phase, sbin, a] += 1
            uses[a] += 1
            if improved:
                best = cand
                wins[a] += 1
                stag = 0
                last_gap = 0.0
                history.append((budget.elapsed(), best.objective()))
            else:
                stag += 1
                last_gap = 0.0 if cand is None else \
                    10.0 * min(0.1, max(0.0, (cand.objective() - Zb) / Zb))

            if sel == "policy" and train:
                reward = 100.0 * (Zb - best.objective()) / max(Z0, 1e-9)
                self.agent.store(st, a, reward, val, logp, False)
                stored += 1
            if sel == "cyclic":
                k = 0 if improved else (k + 1) % N_ACTIONS_VNS
            elif sel == "roulette":
                seg_score[a] += 9.0 if improved else 0.0
                seg_use[a] += 1
                if (it + 1) % 100 == 0:
                    for j in range(N_ACTIONS_VNS):
                        if seg_use[j] > 0:
                            w[j] = max(0.01, 0.85 * w[j]
                                       + 0.15 * seg_score[j] / seg_use[j])
                    seg_score[:] = 0
                    seg_use[:] = 0
            recent_impr.append(1 if improved else 0)
            recent_str.append(s_idx)
            it += 1

        cpu = budget.elapsed()
        train_stats = {}
        if sel == "policy" and train and stored > 1:
            self.agent._dones[-1] = True
            train_stats = self.agent.update(last_value=0.0)
            self.agent.net.eval()

        return {"best": best, "Z_best": best.objective(), "Z_init": Z0,
                "history": history, "cpu_s": cpu, "wall_s": budget.wall(),
                "iters": it, "method": self.name,
                "usage": usage, "uses": uses, "wins": wins,
                "train_stats": train_stats}


class DRLVNS(VNS9):
    """The proposed method: VNS9 with the PPO selector."""

    def __init__(self, agent):
        super().__init__(selector="policy", agent=agent, name="DRL-VNS")
