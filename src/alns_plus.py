"""
ALNS solvers for the C-SDVRP with intensified local search.

Every accepted candidate is refined by Random Variable Neighborhood Descent
(Or-opt of chain length 1-2 and 2-opt*). The action space has 18 entries:
three destroy operators x two repair operators x three removal-size tiers
(small/medium/large), so the removal strength is chosen alongside the
operator pair.

ClassicalALNSPlus picks actions with an adaptive roulette wheel; DRLALNSPlus
picks them with a PPO policy.
"""
import time
import math
from typing import Dict, List, Tuple

import numpy as np

from .instance      import CSDVRPInstance
from .solution      import Route, Solution
from . import operators as _ops
from .operators     import (random_removal, worst_removal, related_removal,
                            best_insertion, regret_insertion,
                            or_opt, two_opt_star, _remove_customers)
from .clarke_wright import clarke_wright, _pick_vehicle_tight
from .alns          import _build_state, _sa_accept, _imax_for_n, \
                           T_INIT_FRAC, COOLING_RATE

DESTROY_OPS = [random_removal, worst_removal, related_removal]
REPAIR_OPS  = [best_insertion, regret_insertion]
SIZE_TIERS  = [0.10, 0.20, 0.35]          # removal fraction tiers
N_DESTROY, N_REPAIR, N_SIZE = 3, 2, 3
N_ACTIONS_PLUS = N_DESTROY * N_REPAIR * N_SIZE      # = 18

def decode(action: int) -> Tuple[int, int, int]:
    """action -> (destroy_idx, repair_idx, size_tier_idx)."""
    size = action % N_SIZE
    rep  = (action // N_SIZE) % N_REPAIR
    des  = action // (N_SIZE * N_REPAIR)
    return des, rep, size


def _rvnd(routes: List[Route], inst: CSDVRPInstance,
          rng: np.random.Generator) -> List[Route]:
    """Random Variable Neighborhood Descent (shared with the VNS baseline)."""
    ops = [lambda r: or_opt(r, inst, max_chain=1),
           lambda r: or_opt(r, inst, max_chain=2)]
    if inst.n <= 120:
        ops.append(lambda r: two_opt_star(r, inst))
    cur = [x.copy() for x in routes if x.stops]
    cur_obj = Solution(inst, cur).objective()
    improved = True
    while improved:
        improved = False
        order = list(range(len(ops))); rng.shuffle(order)
        for idx in order:
            cand = ops[idx]([x.copy() for x in cur if x.stops])
            cand = [x for x in cand if x.stops]
            obj = Solution(inst, cand).objective()
            if obj < cur_obj - 1e-9:
                cur, cur_obj = cand, obj
                improved = True
                break
    return cur


def _removal_by_size(sol: Solution, d_idx: int, size_frac: float,
                     rng: np.random.Generator):
    """Apply destroy operator d_idx, removing about size_frac*n customers.
    The removal count is passed to all three operators by temporarily
    overriding operators._removal_size, so the size tier applies to random,
    worst and related removal alike."""
    n = sol.instance.n
    q = max(2, min(int(math.ceil(size_frac * n)), max(2, n - 1)))
    op = [random_removal, worst_removal, related_removal][d_idx]
    saved = _ops._removal_size
    _ops._removal_size = lambda _n, _q=q: _q
    try:
        routes, removed = op(sol, rng)
    finally:
        _ops._removal_size = saved
    return routes, removed


def _apply(sol: Solution, d_idx: int, r_idx: int, size_frac: float,
           rng: np.random.Generator, strong_ls: bool = True) -> Solution:
    repair = REPAIR_OPS[r_idx]
    base   = Solution(sol.instance, [r.copy() for r in sol.routes])
    new_routes, removed = _removal_by_size(base, d_idx, size_frac, rng)
    if not removed:
        return sol
    new_routes = repair(sol.instance, new_routes, removed)
    new_routes = [r for r in new_routes if r.stops]
    if strong_ls and sol.instance.n <= 200:
        new_routes = _rvnd(new_routes, sol.instance, rng)
    s = Solution(sol.instance, [r for r in new_routes if r.stops])
    return s


class ClassicalALNSPlus:
    """Enhanced ALNS: roulette over 18 (operator x size) actions + strong LS."""

    def solve(self, instance, I_max=None, seed=None, time_limit=None) -> Dict:
        rng   = np.random.default_rng(seed)
        I_max = I_max or _imax_for_n(instance.n)
        sol_curr = clarke_wright(instance, seed=seed)
        sol_curr = Solution(instance, _rvnd([r.copy() for r in sol_curr.routes],
                                            instance, rng))
        sol_best = sol_curr.copy()
        Z_init   = clarke_wright(instance, seed=seed).objective()
        Z_curr   = sol_curr.objective()
        T = T_INIT_FRAC * Z_curr;
        w = np.ones(N_ACTIONS_PLUS); score = np.zeros(N_ACTIONS_PLUS); use = np.zeros(N_ACTIONS_PLUS)
        t0 = time.perf_counter(); history=[sol_best.objective()]
        for it in range(I_max):
            if time_limit and time.perf_counter()-t0 > time_limit: break
            a = int(rng.choice(N_ACTIONS_PLUS, p=w/w.sum()))
            d_idx, r_idx, s_idx = decode(a)
            cand = _apply(sol_curr, d_idx, r_idx, SIZE_TIERS[s_idx], rng)
            Zc = cand.objective(); dz = Zc - Z_curr
            if Zc < sol_best.objective()-1e-9:
                score[a]+=9; sol_best = cand.copy()
            elif Zc < Z_curr-1e-9: score[a]+=5
            elif _sa_accept(dz,T,rng): score[a]+=1
            if Zc <= Z_curr+1e-9 or _sa_accept(dz,T,rng):
                sol_curr, Z_curr = cand, Zc
            use[a]+=1
            if (it+1)%100==0:
                for k in range(N_ACTIONS_PLUS):
                    if use[k]>0:
                        w[k]=0.85*w[k]+0.15*(score[k]/use[k]); w[k]=max(w[k],0.01)
                score[:]=0; use[:]=0
            T*=COOLING_RATE; history.append(sol_best.objective())
        for r in sol_best.routes:
            if r.stops: r.vtype_idx=_pick_vehicle_tight(instance,r.load)
        sol_best.invalidate()
        return {"best":sol_best,"Z_best":sol_best.objective(),"Z_init":Z_init,
                "history":history,"time_s":time.perf_counter()-t0,"iters":it+1,
                "method":"ClassicalALNSPlus"}


class DRLALNSPlus:
    """PPO-guided ALNS over 18 (operator x size) actions + strong LS."""

    def solve(self, instance, agent, I_max=None, seed=None,
              train=False, time_limit=None) -> Dict:
        rng   = np.random.default_rng(seed)
        I_max = I_max or _imax_for_n(instance.n)
        sol0  = clarke_wright(instance, seed=seed)
        Z_init = sol0.objective()
        sol_curr = Solution(instance, _rvnd([r.copy() for r in sol0.routes], instance, rng))
        sol_best = sol_curr.copy()
        Z_curr = sol_curr.objective()
        T = T_INIT_FRAC * Z_curr; T0 = T
        op_hist=[]; history=[sol_best.objective()]; t0=time.perf_counter()
        agent.net.eval()
        for it in range(I_max):
            if time_limit and time.perf_counter()-t0 > time_limit: break
            state = _build_state(sol_curr, sol_curr, sol_best, it, I_max, T, T0, op_hist)
            action, logp, val = agent.select_action(state)
            d_idx, r_idx, s_idx = decode(action)
            cand = _apply(sol_curr, d_idx, r_idx, SIZE_TIERS[s_idx], rng)
            Zc = cand.objective(); dz = Zc - Z_curr
            reward = (Z_curr - Zc)/max(Z_init,1e-6)
            if Zc < sol_best.objective()-1e-9: sol_best = cand.copy()
            if Zc <= Z_curr+1e-9 or _sa_accept(dz,T,rng):
                sol_curr, Z_curr = cand, Zc
            op_hist.append(d_idx*N_REPAIR + r_idx)   # 0..5 for state history
            if train:
                agent.store(state, action, reward, val, logp, it==I_max-1)
            T*=COOLING_RATE; history.append(sol_best.objective())
        if train:
            agent.update(last_value=0.0); agent.net.eval()
        for r in sol_best.routes:
            if r.stops: r.vtype_idx=_pick_vehicle_tight(instance,r.load)
        sol_best.invalidate()
        return {"best":sol_best,"Z_best":sol_best.objective(),"Z_init":Z_init,
                "history":history,"time_s":time.perf_counter()-t0,"iters":I_max,
                "method":"DRL-ALNS+","op_history":op_hist}
