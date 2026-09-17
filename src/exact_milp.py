"""
exact_milp.py — Exact MILP solver for the C-SDVRP (small instances).

Single-commodity-flow formulation, linearized to match solution.py's
objective exactly:

    Z = c * sum_{i,j,k} d_ij x_ijk                         (routing cost)
      + mu * sum_i Gamma_i                                  (emission penalty, N3)
      + delta * sum_i max(0, s_i - 1)                       (split penalty, N1/N3)

Arc emission arriving at customer i (matches solution.py exactly):
    E_i = sum_{j,k} [ e_k d_ji x_jik + (e_k beta_k / Q_k) d_ji f_jik ]
    Gamma_i >= E_i - eta_i d_i ,   Gamma_i >= 0

Connectivity / subtour elimination via single-commodity flow f_ijk
(load carried on arc) with the depot as the unique source.

Designed for homogeneous-fleet small instances (n <= ~15) so CBC can
prove optimality or return a valid lower bound + gap within a time limit.
"""

import time
from typing import Dict

import pulp

from .instance import CSDVRPInstance


EPS_DELIVERY = 0.01   # minimum quantity of any visit (y_ik >= eps * v_ik)


def solve_exact(inst: CSDVRPInstance, time_limit: int = 600,
                split_eligible=None, msg: bool = False,
                K: int = None) -> Dict:
    """
    Solve the C-SDVRP exactly (or to a time-limited bound) with CBC.

    Parameters
    ----------
    inst : CSDVRPInstance      homogeneous fleet expected (single vtype, count K)
    time_limit : int           CBC wall-clock limit in seconds
    split_eligible : array-like[bool] or None
        split_eligible[i] True => customer i may be served by >1 vehicle (N1).
        If None, derived from inst (alpha[i] > 0).

    Returns dict with Z (best feasible), lower_bound, gap, status,
    routing/emission/split components, and solve time.
    """
    n   = inst.n
    C   = list(range(1, n + 1))           # customers
    V   = list(range(0, n + 1))           # nodes incl depot
    dm  = inst._dist
    dem = inst.demand
    eta = inst.eta
    c   = inst.dist_cost
    mu  = inst.mu
    delta = inst.delta

    # Homogeneous fleet: take the (single) vehicle type and its count
    vt   = inst.vtypes[0]
    Q    = float(vt["capacity"])
    e_k  = float(vt["emit_base"])
    beta = float(vt["emit_beta"])
    # Number of vehicle copies in the model. The problem has an unlimited
    # homogeneous fleet; the caller passes K (by default the number of routes
    # of the Clarke-Wright solution plus one, which is always feasible) and
    # the result reports how many vehicles the optimum actually used.
    if K is None:
        from .clarke_wright import clarke_wright
        K = clarke_wright(inst, seed=0).n_routes_active() + 1
    Kset = list(range(K))

    if split_eligible is None:
        split_eligible = [bool(inst.alpha[i] > 0) for i in range(n + 1)]
    Umax = [int(inst.U_max[i]) for i in range(n + 1)]

    arcs = [(i, j) for i in V for j in V if i != j]

    prob = pulp.LpProblem("C_SDVRP_exact", pulp.LpMinimize)

    # Variables
    x = {(i, j, k): pulp.LpVariable(f"x_{i}_{j}_{k}", cat="Binary")
         for (i, j) in arcs for k in Kset}
    f = {(i, j, k): pulp.LpVariable(f"f_{i}_{j}_{k}", lowBound=0, upBound=Q)
         for (i, j) in arcs for k in Kset}
    y = {(i, k): pulp.LpVariable(f"y_{i}_{k}", lowBound=0, upBound=Q)
         for i in C for k in Kset}
    vis = {(i, k): pulp.LpVariable(f"vis_{i}_{k}", cat="Binary")
           for i in C for k in Kset}
    Gam = {i: pulp.LpVariable(f"G_{i}", lowBound=0) for i in C}
    sx  = {i: pulp.LpVariable(f"sx_{i}", lowBound=0) for i in C}  # split excess

    # Objective
    routing = pulp.lpSum(c * dm[i, j] * x[i, j, k] for (i, j) in arcs for k in Kset)
    emission = mu * pulp.lpSum(Gam[i] for i in C)
    split    = delta * pulp.lpSum(sx[i] for i in C)
    prob += routing + emission + split

    # Demand satisfaction
    for i in C:
        prob += pulp.lpSum(y[i, k] for k in Kset) == dem[i]

    # Link delivery to visit; visit requires entering the node
    for i in C:
        for k in Kset:
            prob += y[i, k] <= Q * vis[i, k]
            prob += y[i, k] >= EPS_DELIVERY * vis[i, k]
            prob += vis[i, k] == pulp.lpSum(x[j, i, k] for j in V if j != i)  # enter
            prob += pulp.lpSum(x[i, j, k] for j in V if j != i) == vis[i, k]  # leave

    # Each vehicle leaves depot at most once (one route)
    for k in Kset:
        prob += pulp.lpSum(x[0, j, k] for j in C) <= 1
        prob += pulp.lpSum(x[0, j, k] for j in C) == pulp.lpSum(x[j, 0, k] for j in C)

    # Symmetry breaking (identical vehicles): use lower indices first
    # Removes the K! permutation symmetry that cripples branch-and-bound.
    # (Usage ordering only; a load tie-break was found to hurt CBC's primal
    #  heuristics and is therefore omitted.)
    for k in Kset[:-1]:
        prob += (pulp.lpSum(x[0, j, k] for j in C)
                 >= pulp.lpSum(x[0, j, k + 1] for j in C))

    # Capacity
    for k in Kset:
        prob += pulp.lpSum(y[i, k] for i in C) <= Q

    # Single-commodity flow (load) + subtour elimination
    for k in Kset:
        # load leaving depot = total delivered by vehicle k
        prob += pulp.lpSum(f[0, j, k] for j in C) == pulp.lpSum(y[i, k] for i in C)
        prob += pulp.lpSum(f[j, 0, k] for j in C) == 0   # returns empty
    for i in C:
        for k in Kset:
            # inflow - outflow = delivered at i
            prob += (pulp.lpSum(f[j, i, k] for j in V if j != i)
                     - pulp.lpSum(f[i, j, k] for j in V if j != i)) == y[i, k]
    for (i, j) in arcs:
        for k in Kset:
            prob += f[i, j, k] <= Q * x[i, j, k]

    # N1 (split activation) + N2 (visit cap): split count s_i
    for i in C:
        s_i = pulp.lpSum(vis[i, k] for k in Kset)
        cap = max(1, Umax[i]) if split_eligible[i] else 1   # N1 and N2
        prob += s_i <= cap
        prob += s_i >= 1                                # must be served
        prob += sx[i] >= s_i - 1                        # split excess >= s_i - 1

    # Emission linearization (matches solution.py arrival emissions)
    for i in C:
        E_i = pulp.lpSum(
            e_k * dm[j, i] * x[j, i, k] + (e_k * beta / Q) * dm[j, i] * f[j, i, k]
            for j in V if j != i for k in Kset)
        prob += Gam[i] >= E_i - eta[i] * dem[i]

    # Solve (capture CBC log to extract the true dual bound)
    import tempfile, os, re
    log_path = os.path.join(tempfile.gettempdir(), f"cbc_{inst.name}_{int(time.time()*1000)}.log")
    t0 = time.perf_counter()
    solver = pulp.PULP_CBC_CMD(msg=1 if msg else 0, timeLimit=time_limit,
                               gapRel=0.0, logPath=log_path)
    prob.solve(solver)
    elapsed = time.perf_counter() - t0

    pulp_status = pulp.LpStatus[prob.status]
    Z = pulp.value(prob.objective)
    rc = pulp.value(routing) if Z is not None else None
    em = pulp.value(emission) if Z is not None else None
    sp = pulp.value(split) if Z is not None else None

    # Parse CBC log for proven optimality and the best dual bound
    proven = False
    lb = None
    cbc_version = None
    try:
        with open(log_path, "r", errors="replace") as fh:
            log = fh.read()
        if ("Optimal solution found" in log) or ("Result - Optimal" in log):
            proven = True
        mv = re.search(r"Version:\s*([0-9.]+)", log)
        if mv:
            cbc_version = mv.group(1)
        # "Lower bound:  X" appears for time-limited runs; capture last value
        m = re.findall(r"Lower bound:\s*([0-9.eE+\-]+)", log)
        if m:
            lb = float(m[-1])
        # CBC also prints "gap" lines; the "best possible" objective is the bound
        m2 = re.findall(r"best possible\s+([0-9.eE+\-]+)", log)
        if m2:
            try:
                lb = float(m2[-1]) if lb is None else lb
            except ValueError:
                pass
    except FileNotFoundError:
        pass
    finally:
        try: os.remove(log_path)
        except OSError: pass

    if proven:
        lb = Z
        gap = 0.0
        status = "Optimal"
    else:
        status = "TimeLimit" if elapsed >= time_limit - 1 else pulp_status
        gap = ((Z - lb) / lb * 100.0) if (Z is not None and lb not in (None, 0)) else None

    used = None
    if Z is not None:
        used = sum(1 for k in Kset
                   if sum((pulp.value(x[0, j, k]) or 0) for j in C) > 0.5)
    return {
        "status": status, "proven_optimal": proven, "Z": Z,
        "vehicles_used": used, "cbc_version": cbc_version,
        "lower_bound": lb, "gap_pct": gap,
        "routing": rc, "emission": em, "split": sp,
        "time_s": elapsed, "n": n, "K": K,
    }
