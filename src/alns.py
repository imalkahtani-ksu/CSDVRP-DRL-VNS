"""
alns.py — Classical ALNS and DRL-ALNS for C-SDVRP.

ClassicalALNS : weight-based roulette operator selection (baseline)
DRLALNS       : PPO-guided operator selection (proposed method)

Both share: Clarke-Wright init, same operators, same SA acceptance.

SA schedule: T_{t+1} = rho * T_t,  T_0 = T_INIT_FRAC * Z_0
"""

import copy
import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .instance       import CSDVRPInstance
from .solution       import Route, Solution
from .operators      import (random_removal, worst_removal, related_removal,
                             best_insertion, regret_insertion,
                             or_opt, two_opt_star)
from .clarke_wright  import clarke_wright, _pick_vehicle_tight
from .ppo            import ACTION_MAP

T_INIT_FRAC  = 0.05   # T_0 = T_INIT_FRAC * Z_init (accept up to 5% worse)
COOLING_RATE = 0.9975 # rho

SIGMA_BEST    = 9     # reward for new best solution
SIGMA_IMPROVE = 5     # reward for improving current
SIGMA_ACCEPT  = 1     # reward for accepted (SA) but not improving
WEIGHT_DECAY  = 0.85  # w_{t+1} = decay * w_t + (1-decay) * score

# (DRL phase limit removed — agent is now trained with per-instance I_max
#  matching _imax_for_n(n), so it operates in-distribution for all iterations.)

DESTROY_OPS = [random_removal, worst_removal, related_removal]
REPAIR_OPS  = [best_insertion, regret_insertion]
OP_NAMES    = ["D1-Rand", "D2-Worst", "D3-Related", "R1-Best", "R2-Regret"]

def _apply_operator_pair(sol: Solution, d_idx: int, r_idx: int,
                         rng: np.random.Generator,
                         use_local_search: bool = True) -> Solution:
    """Apply destroy operator d_idx then repair operator r_idx,
    followed by a lightweight Or-opt / 2-opt* local search pass."""
    destroy = DESTROY_OPS[d_idx]
    repair  = REPAIR_OPS[r_idx]

    routes_copy = [r.copy() for r in sol.routes]
    new_routes, removed = destroy(Solution(sol.instance, routes_copy), rng)

    if not removed:
        return sol   # nothing removed; return unchanged

    new_routes = repair(sol.instance, new_routes, removed, rng=rng)
    if new_routes is None:
        return sol
    new_routes = [r for r in new_routes if r.stops]

    # Post-repair local search (Or-opt only; 2-opt* skipped for speed on large n)
    if use_local_search and sol.instance.n <= 100:
        new_routes = or_opt(new_routes, sol.instance, max_chain=1)

    new_sol        = Solution(sol.instance, new_routes)
    new_sol.routes = [r for r in new_sol.routes if r.stops]
    return new_sol

def _sa_accept(delta_z: float, temp: float, rng: np.random.Generator) -> bool:
    """Simulated annealing acceptance."""
    if delta_z < 0:
        return True
    if temp < 1e-10:
        return False
    return bool(rng.random() < math.exp(-delta_z / temp))

def _imax_for_n(n: int) -> int:
    """Iteration budget scaled by instance size."""
    if n <= 20:  return 1_500
    if n <= 50:  return 3_000
    if n <= 100: return 5_000
    if n <= 150: return 5_000   # XL-150: same budget as L-100 (fleet redesign fixes convergence)
    return 6_000                # XL-200: modest extra budget for 200-customer instances

#  17-DIM STATE VECTOR

def _build_state(sol: Solution, sol_init: Solution, sol_best: Solution,
                 iteration: int, I_max: int, T: float, T_init: float,
                 op_history: List[int]) -> np.ndarray:
    """
    Construct the 17-dimensional state vector for the PPO agent.
    All values are normalised to [0,1] or small positive range.
    """
    inst    = sol.instance
    n       = inst.n
    Z       = sol.objective()
    Z_init  = sol_init.objective()
    Z_best  = sol_best.objective()
    Z_norm  = max(Z_init, 1e-6)

    comps   = sol.components()
    rc      = comps["routing"]
    ep      = comps["emission"]
    sp      = comps["split"]

    # Fleet utilisation: mean route load / vehicle capacity
    utils = []
    for route in sol.routes:
        if route.stops:
            cap = inst.vtypes[route.vtype_idx]["capacity"]
            utils.append(min(1.0, route.load / max(cap, 1e-6)))
    fleet_util = float(np.mean(utils)) if utils else 0.0

    # Split fraction
    sc  = sol.split_counts()
    n_split = sum(1 for v in sc.values() if v > 1)
    split_frac = n_split / max(n, 1)

    # N1 violation (soft): sum of excess splits normalised by n
    n1_viol = sum(max(0, sc.get(i, 0) - (inst.alpha[i] + 1))
                  for i in range(1, n + 1)) / max(n, 1)

    # N2 violation: number of customers where total visits > U_max
    n2_viol = sum(1 for i in range(1, n + 1)
                  if inst.U_max[i] > 0 and sc.get(i, 0) > inst.U_max[i]
                  ) / max(n, 1)

    # Operator usage history (last 30 steps)
    hist = op_history[-30:] if len(op_history) >= 30 else op_history
    hist_arr = np.zeros(6, dtype=np.float32)
    if hist:
        for a in hist:
            hist_arr[a] += 1
        hist_arr /= len(hist)
    d_hist = hist_arr[:3]   # D1,D2,D3 fractions
    r_hist_r1 = hist_arr[0] + hist_arr[2] + hist_arr[4]  # fraction using R1
    r_hist_r2 = hist_arr[1] + hist_arr[3] + hist_arr[5]  # fraction using R2
    total_r    = r_hist_r1 + r_hist_r2
    if total_r > 0:
        r_hist_r1 /= total_r
        r_hist_r2 /= total_r

    # Route length and cost diversity (normalised std dev)
    route_lengths = [len(r.stops) for r in sol.routes if r.stops]
    route_costs   = []
    dm = inst._dist; c = inst.dist_cost
    for route in sol.routes:
        if not route.stops:
            continue
        rc_r = 0.0; prev = 0
        for cid, _ in route.stops:
            rc_r += c * dm[prev, cid]; prev = cid
        rc_r += c * dm[prev, 0]
        route_costs.append(rc_r)

    def _norm_std(arr):
        if not arr or np.mean(arr) < 1e-9:
            return 0.0
        return float(np.std(arr) / (np.mean(arr) + 1e-9))

    state = np.array([
        np.clip(Z / Z_norm, 0, 5),                              # 1
        np.clip(rc / Z_norm, 0, 5),                             # 2
        np.clip(ep / Z_norm, 0, 5),                             # 3
        np.clip(sp / Z_norm, 0, 5),                             # 4
        fleet_util,                                              # 5
        iteration / max(I_max, 1),                              # 6
        np.clip((Z - Z_best) / max(Z_best, 1e-6), 0, 2),       # 7
        np.clip(T / max(T_init, 1e-6), 0, 1),                  # 8
        split_frac,                                              # 9
        np.clip(n1_viol, 0, 1),                                 # 10
        np.clip(n2_viol, 0, 1),                                 # 11
        d_hist[0], d_hist[1], d_hist[2],                        # 12-14
        r_hist_r1,                                              # 15
        np.clip(_norm_std(route_lengths), 0, 2),                # 16
        np.clip(_norm_std(route_costs),   0, 2),                # 17
    ], dtype=np.float32)

    return state

#  CLASSICAL ALNS  (baseline)

class ClassicalALNS:
    """
    Classical ALNS with weight-based roulette operator selection.
    Follows Ropke & Pisinger (2006) scoring scheme.
    """

    def solve(self, instance: CSDVRPInstance,
              I_max: int    = None,
              seed:  int    = None,
              time_limit: float = None) -> Dict:
        rng    = np.random.default_rng(seed)
        I_max  = I_max or _imax_for_n(instance.n)

        # Initialise
        sol_curr = clarke_wright(instance, seed=seed)
        sol_best = sol_curr.copy()
        Z_init   = sol_curr.objective()
        Z_curr   = Z_init
        T        = T_INIT_FRAC * Z_init
        T_init   = T

        # Weights for 6 operator pairs
        weights = np.ones(6, dtype=np.float64)
        scores  = np.zeros(6, dtype=np.float64)
        usage   = np.zeros(6, dtype=np.int64)

        history = []
        t_start = time.perf_counter()

        for it in range(I_max):
            if time_limit and (time.perf_counter() - t_start) > time_limit:
                break

            # Roulette selection
            w_sum  = weights.sum()
            probs  = weights / w_sum
            action = int(rng.choice(6, p=probs))
            d_idx  = action // 2
            r_idx  = action  % 2

            # Apply operators
            sol_new = _apply_operator_pair(sol_curr, d_idx, r_idx, rng)
            Z_new   = sol_new.objective()
            delta_z = Z_new - Z_curr

            # Scoring
            if Z_new < sol_best.objective() - 1e-9:
                scores[action] += SIGMA_BEST
                sol_best = sol_new.copy()
            elif Z_new < Z_curr - 1e-9:
                scores[action] += SIGMA_IMPROVE
            elif _sa_accept(delta_z, T, rng):
                scores[action] += SIGMA_ACCEPT

            if Z_new <= Z_curr + 1e-9 * abs(Z_curr) or _sa_accept(delta_z, T, rng):
                sol_curr = sol_new
                Z_curr   = Z_new

            usage[action] += 1

            # Update weights every 100 iterations
            if (it + 1) % 100 == 0:
                for a in range(6):
                    if usage[a] > 0:
                        avg_score    = scores[a] / usage[a]
                        weights[a]   = WEIGHT_DECAY * weights[a] + (1 - WEIGHT_DECAY) * avg_score
                        weights[a]   = max(weights[a], 0.01)
                scores[:] = 0
                usage[:]  = 0

            T *= COOLING_RATE
            history.append(sol_best.objective())

        elapsed = time.perf_counter() - t_start

        # Post-process: reassign vehicles to the tightest (smallest) feasible type.
        # During search we used slack vehicles (≈55 % utilisation) to allow
        # cross-route insertions; now we minimise emission by right-sizing.
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
            "iters":   it + 1,
            "method":  "ClassicalALNS",
        }

#  DRL-ALNS  (proposed method)

class DRLALNS:
    """
    DRL-enhanced ALNS: PPO agent selects operator pairs adaptively.
    Can be used in inference mode (agent.net eval) or training mode (agent.store).
    """

    def solve(self, instance: CSDVRPInstance, agent,
              I_max: int   = None,
              seed:  int   = None,
              train: bool  = False,
              time_limit: float = None) -> Dict:

        rng    = np.random.default_rng(seed)
        I_max  = I_max or _imax_for_n(instance.n)

        # Initialise
        sol_curr = clarke_wright(instance, seed=seed)
        sol_best = sol_curr.copy()
        Z_init   = sol_curr.objective()
        Z_curr   = Z_init
        T        = T_INIT_FRAC * Z_init
        T_init   = T

        op_history: List[int] = []
        history:   List[float] = []
        t_start = time.perf_counter()

        agent.net.eval()

        for it in range(I_max):
            if time_limit and (time.perf_counter() - t_start) > time_limit:
                break

            # Operator selection: pure DRL for all iterations
            # Agent trained with I_max = _imax_for_n(n), so it/I_max is always
            # in [0,1] and the policy is in-distribution throughout.
            state = _build_state(sol_curr, sol_curr, sol_best, it,
                                 I_max, T, T_init, op_history)
            action, log_prob, value = agent.select_action(state)

            d_idx = action // 2
            r_idx = action  % 2

            sol_new = _apply_operator_pair(sol_curr, d_idx, r_idx, rng)
            Z_new   = sol_new.objective()
            delta_z = Z_new - Z_curr

            # Reward: normalised improvement
            reward = (Z_curr - Z_new) / max(Z_init, 1e-6)

            improved = Z_new < sol_best.objective() - 1e-9
            if improved:
                sol_best = sol_new.copy()

            accepted = (Z_new <= Z_curr + 1e-9) or _sa_accept(delta_z, T, rng)
            if accepted:
                sol_curr = sol_new
                Z_curr   = Z_new

            op_history.append(action)

            if train:
                done = (it == I_max - 1)
                agent.store(state, action, reward, value, log_prob, done)

            T *= COOLING_RATE
            history.append(sol_best.objective())

        if train:
            agent.update(last_value=0.0)
            agent.net.eval()

        elapsed = time.perf_counter() - t_start

        # Post-process: right-size vehicles (minimise emission) after route
        # loads are finalised — mirrors the same step in ClassicalALNS.
        for r in sol_best.routes:
            if r.stops:
                r.vtype_idx = _pick_vehicle_tight(sol_best.instance, r.load)
        sol_best.invalidate()

        return {
            "best":       sol_best,
            "Z_best":     sol_best.objective(),
            "Z_init":     Z_init,
            "history":    history,
            "time_s":     elapsed,
            "iters":      I_max,
            "method":     "DRL-ALNS",
            "op_history": op_history,
        }
