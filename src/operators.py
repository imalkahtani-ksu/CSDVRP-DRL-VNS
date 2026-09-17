"""
operators.py — ALNS destroy and repair operators for C-SDVRP.

Destroy operators:
  D1  random_removal   : remove q random customers
  D2  worst_removal    : remove q highest-cost customers
  D3  related_removal  : Shaw removal (proximity-based)

Repair operators:
  R1  best_insertion   : greedy best-position insertion
  R2  regret_insertion : regret-2 insertion (minimise future regret)

Constraint handling (hard): vehicle capacity; N1, a customer that is not
split-eligible (U_max = 1) is served by exactly one vehicle; N2, an eligible
customer is served by at most U_max vehicles; and a vehicle visits a customer
at most once. The split penalty delta (N3) is charged in the objective.
"""

import math
import random
from typing import Dict, List, Tuple

import numpy as np

from .instance import CSDVRPInstance
from .solution  import Route, Solution

def _removal_size(n: int) -> int:
    """Number of customers to remove: 15-20% of n, clamped to [2, 20]."""
    return max(2, min(int(math.ceil(0.18 * n)), 20))

def _remove_customers(routes: List[Route], to_remove: set
                      ) -> Tuple[List[Route], Dict[int, float]]:
    """
    Remove all visits to customers in `to_remove` from routes.
    Returns (new_routes, undelivered_demand_dict).
    Undelivered demand = original demand - still in other routes.
    """
    new_routes  = []
    still_in    = {}   # cid -> remaining qty in surviving stops
    removed_qty = {}   # cid -> qty removed in this call

    for route in routes:
        new_stops = []
        for cid, qty in route.stops:
            if cid in to_remove:
                removed_qty[cid] = removed_qty.get(cid, 0.0) + qty
            else:
                new_stops.append((cid, qty))
                still_in[cid]   = still_in.get(cid, 0.0) + qty
        new_route         = route.copy()
        new_route.stops   = new_stops
        new_routes.append(new_route)

    return new_routes, removed_qty

def random_removal(sol: Solution, rng: np.random.Generator
                   ) -> Tuple[List[Route], Dict[int, float]]:
    """D1: Remove q randomly selected customers."""
    q       = _removal_size(sol.instance.n)
    all_cids = list({cid for r in sol.routes for cid, _ in r.stops})
    if not all_cids:
        return sol.routes, {}
    chosen  = set(rng.choice(all_cids, size=min(q, len(all_cids)), replace=False))
    return _remove_customers(sol.routes, chosen)

def worst_removal(sol: Solution, rng: np.random.Generator
                  ) -> Tuple[List[Route], Dict[int, float]]:
    """
    D2: Remove q customers that contribute most to the objective.
    Cost of customer i = routing cost of incoming/outgoing arcs + split penalty share.
    Uses 'noise' factor for diversification.
    """
    inst  = sol.instance
    q     = _removal_size(inst.n)
    dm    = inst._dist
    c_km  = inst.dist_cost

    # Compute per-customer routing cost contribution
    cust_cost: Dict[int, float] = {}
    for route in sol.routes:
        if not route.stops:
            continue
        stops  = route.stops
        prev   = 0
        for idx, (cid, qty) in enumerate(stops):
            nxt = stops[idx + 1][0] if idx + 1 < len(stops) else 0
            # Cost if this stop is removed (saving = arc before + arc after - direct arc)
            saving = c_km * (dm[prev, cid] + dm[cid, nxt] - dm[prev, nxt])
            cust_cost[cid] = cust_cost.get(cid, 0.0) + saving
            prev = cid

    if not cust_cost:
        return sol.routes, {}

    # Add small noise for diversification (1 ± 0.01 random noise)
    noisy = {c: v * (1 + rng.uniform(-0.01, 0.01)) for c, v in cust_cost.items()}
    sorted_custs = sorted(noisy, key=lambda c: noisy[c], reverse=True)
    chosen       = set(sorted_custs[:q])
    return _remove_customers(sol.routes, chosen)

def related_removal(sol: Solution, rng: np.random.Generator
                    ) -> Tuple[List[Route], Dict[int, float]]:
    """
    D3: Shaw (1998) related removal. Start with one random customer and
    iteratively add the most related (closest) unselected customer.
    Relatedness: r(i,j) = phi * dist(i,j)/d_max + (1-phi) * |dem_i - dem_j|/D_range
    """
    inst  = sol.instance
    q     = _removal_size(inst.n)
    dm    = inst._dist
    demand = inst.demand
    n      = inst.n

    # Customers present in solution
    present = list({cid for r in sol.routes for cid, _ in r.stops})
    if not present:
        return sol.routes, {}

    d_max   = dm[1:n+1, 1:n+1].max() + 1e-9
    D_range = demand[1:n+1].max() - demand[1:n+1].min() + 1e-9
    phi     = 0.7    # weight of distance vs. demand similarity

    selected = [int(rng.choice(present))]
    remaining = set(present) - set(selected)

    while len(selected) < q and remaining:
        # Relatedness: average similarity to already-selected customers
        relatedness = {}
        for cand in remaining:
            r_score = 0.0
            for sel in selected:
                geo  = dm[sel, cand] / d_max
                dem  = abs(demand[sel] - demand[cand]) / D_range
                r_score += phi * geo + (1 - phi) * dem
            relatedness[cand] = r_score / len(selected)
        # Select the most related (smallest r_score = closest/similar)
        best = min(relatedness, key=lambda c: relatedness[c])
        selected.append(best)
        remaining.discard(best)

    return _remove_customers(sol.routes, set(selected))

def _insertion_delta(route: Route, pos: int, cid: int, qty: float,
                     dm: np.ndarray, dist_cost: float) -> float:
    """
    Cost increase when inserting (cid, qty) at position `pos` in route.stops.
    pos=0 → insert before first stop; pos=len(stops) → append.
    """
    stops = route.stops
    prev  = 0     if pos == 0            else stops[pos - 1][0]
    nxt   = 0     if pos == len(stops)   else stops[pos][0]
    return dist_cost * (dm[prev, cid] + dm[cid, nxt] - dm[prev, nxt])

def _best_position(route: Route, cid: int, qty: float,
                   inst: CSDVRPInstance) -> Tuple[int, float]:
    """Return (best_pos, delta_cost) for inserting (cid, qty) into route."""
    vt  = inst.vtypes[route.vtype_idx]
    cap = vt["capacity"]
    if route.load + qty > cap + 1e-9:
        return -1, float("inf")   # infeasible capacity
    best_pos   = 0
    best_delta = float("inf")
    for pos in range(len(route.stops) + 1):
        delta = _insertion_delta(route, pos, cid, qty, inst._dist, inst.dist_cost)
        if delta < best_delta:
            best_delta = delta
            best_pos   = pos
    return best_pos, best_delta

def _insertion_options(inst: CSDVRPInstance, routes: List[Route],
                       route_loads: List[float], cid: int, remaining: float,
                       visited: set) -> List[Tuple[float, int, int, float]]:
    """
    Feasible placements for the next piece of customer `cid`.

    `visited` holds the indices of routes that already serve `cid` in the
    current repair. A route may serve a customer at most once, and the total
    number of vehicles serving `cid` may not exceed U_max[cid] (U_max = 1 for
    customers that are not split-eligible, which is how N1 is stored). A
    partial piece is only allowed if the rest can still be delivered with the
    visits that remain.

    Returns a list of (cost, route_idx, position, quantity); route_idx = -1
    stands for a new vehicle. The cost is the distance increase plus the split
    penalty delta for every extra visit this placement creates or forces.
    """
    dm, c_km = inst._dist, inst.dist_cost
    Q = inst.max_cap
    u = max(1, int(inst.U_max[cid]))
    v = len(visited)
    left = u - v - 1
    if left < 0:
        return []
    pen = inst.delta

    def feasible(piece):
        rest = remaining - piece
        if rest <= 1e-9:
            return True
        return left > 0 and rest <= left * Q + 1e-9

    def extra(piece):
        e = pen if v >= 1 else 0.0
        if piece < remaining - 1e-9:
            e += pen
        return e

    opts = []
    for ri, route in enumerate(routes):
        if ri in visited:
            continue
        cap = inst.vtypes[route.vtype_idx]["capacity"]
        avail = cap - route_loads[ri]
        if avail <= 1e-9:
            continue
        piece = min(remaining, avail)
        if not feasible(piece):
            continue
        stops = route.stops
        best_d, best_p = float("inf"), 0
        for pos in range(len(stops) + 1):
            prev = 0 if pos == 0 else stops[pos - 1][0]
            nxt = 0 if pos == len(stops) else stops[pos][0]
            d = c_km * (dm[prev, cid] + dm[cid, nxt] - dm[prev, nxt])
            if d < best_d:
                best_d, best_p = d, pos
        opts.append((best_d + extra(piece), ri, best_p, piece))

    piece = min(remaining, Q)
    if feasible(piece):
        d0 = c_km * (dm[0, cid] + dm[cid, 0])
        opts.append((d0 + extra(piece), -1, 0, piece))
    return opts


def _place(inst: CSDVRPInstance, routes: List[Route], route_loads: List[float],
           cid: int, ri: int, pos: int, qty: float) -> int:
    """Apply one placement; returns the index of the route that received it."""
    if ri >= 0:
        routes[ri].stops.insert(pos, (cid, qty))
        route_loads[ri] += qty
        return ri
    vt_idx = 0
    for k, vt in enumerate(inst.vtypes):
        if vt["capacity"] >= qty - 1e-9:
            vt_idx = k
            break
    routes.append(Route(vtype_idx=vt_idx, stops=[(cid, qty)]))
    route_loads.append(qty)
    return len(routes) - 1


def best_insertion(inst: CSDVRPInstance, routes: List[Route],
                   removed: Dict[int, float], rng=None) -> List[Route]:
    """
    R1: greedy best insertion. Customers are processed in random order; each
    one is placed at its cheapest feasible position, splitting only when the
    customer is split-eligible and within its visit cap.
    Returns None if some customer cannot be placed (the move is then rejected).
    """
    order = list(removed.keys())
    if rng is not None:
        order = [order[i] for i in rng.permutation(len(order))]
    else:
        random.shuffle(order)
    route_loads = [sum(q for _, q in r.stops) for r in routes]

    for cid in order:
        remaining = float(inst.demand[cid])
        visited = set()
        while remaining > 1e-6:
            opts = _insertion_options(inst, routes, route_loads, cid,
                                      remaining, visited)
            if not opts:
                return None
            _, ri, pos, qty = min(opts, key=lambda o: o[0])
            visited.add(_place(inst, routes, route_loads, cid, ri, pos, qty))
            remaining -= qty
    return routes


def regret_insertion(inst: CSDVRPInstance, routes: List[Route],
                     removed: Dict[int, float], rng=None) -> List[Route]:
    """
    R2: regret-2 insertion. At each step the customer with the largest gap
    between its best and second-best feasible placement is inserted first.
    Same N1/N2 and one-visit-per-route rules as best_insertion.
    Returns None if some customer cannot be placed.
    """
    route_loads = [sum(q for _, q in r.stops) for r in routes]
    remaining = {cid: float(inst.demand[cid]) for cid in removed}
    visited = {cid: set() for cid in removed}

    while remaining:
        best_cid, best_regret, best_opt = None, -1.0, None
        for cid in sorted(remaining):
            opts = _insertion_options(inst, routes, route_loads, cid,
                                      remaining[cid], visited[cid])
            if not opts:
                return None
            opts.sort(key=lambda o: o[0])
            second = opts[1][0] if len(opts) > 1 else opts[0][0] + 1e9
            regret = second - opts[0][0]
            if regret > best_regret:
                best_cid, best_regret, best_opt = cid, regret, opts[0]
        _, ri, pos, qty = best_opt
        visited[best_cid].add(_place(inst, routes, route_loads, best_cid,
                                     ri, pos, qty))
        remaining[best_cid] -= qty
        if remaining[best_cid] < 1e-6:
            del remaining[best_cid]
    return routes


#  POST-IMPROVEMENT OPERATORS (applied after repair, before acceptance)

def or_opt(routes: List[Route], inst: CSDVRPInstance, max_chain: int = 1
           ) -> List[Route]:
    """
    Or-opt: true single-pass chain-relocation (O(n²·K) per call).
    Scans every source route once; for each route applies the single best
    improving relocation to any other route.  Does NOT restart — safe to
    call on every ALNS iteration without dominating run-time.
    """
    dm   = inst._dist
    c_km = inst.dist_cost

    # Pre-compute route loads once to avoid O(n_stops) recomputation in inner loop
    route_loads = [sum(q for _, q in r.stops) for r in routes]
    # a chain may not move into a route that already serves one of its customers
    route_custs = [{c for c, _ in r.stops} for r in routes]

    for ri in range(len(routes)):
        r = routes[ri]
        n_stops = len(r.stops)
        if n_stops < 2:
            continue
        # Find the single best improving move for this source route
        global_best_gain = 1e-6
        global_best = None   # (start, chain_len, rj, pos)
        for chain_len in range(1, min(max_chain + 1, n_stops)):
            for start in range(n_stops - chain_len + 1):
                chain      = r.stops[start: start + chain_len]
                chain_load = sum(q for _, q in chain)
                prev_r = r.stops[start - 1][0] if start > 0 else 0
                nxt_r  = (r.stops[start + chain_len][0]
                          if start + chain_len < n_stops else 0)
                removal_save = c_km * (
                    dm[prev_r, chain[0][0]] + dm[chain[-1][0], nxt_r]
                    - dm[prev_r, nxt_r])

                chain_ids = {c for c, _ in chain}
                for rj in range(len(routes)):
                    if rj == ri:
                        continue
                    if chain_ids & route_custs[rj]:
                        continue
                    vt = inst.vtypes[routes[rj].vtype_idx]
                    if route_loads[rj] + chain_load > vt["capacity"] + 1e-9:
                        continue
                    sj = routes[rj].stops
                    for pos in range(len(sj) + 1):
                        pj  = sj[pos - 1][0] if pos > 0 else 0
                        nj  = sj[pos][0]     if pos < len(sj) else 0
                        ins = c_km * (dm[pj, chain[0][0]] +
                                      dm[chain[-1][0], nj] - dm[pj, nj])
                        gain = removal_save - ins
                        if gain > global_best_gain:
                            global_best_gain = gain
                            global_best = (start, chain_len, rj, pos)

        if global_best is not None:
            start, chain_len, rj, pos = global_best
            chain = r.stops[start: start + chain_len]
            sj    = routes[rj].stops
            routes[ri].stops = r.stops[:start] + r.stops[start + chain_len:]
            routes[rj].stops = sj[:pos] + chain + sj[pos:]
            # Update cached loads
            chain_load = sum(q for _, q in chain)
            route_loads[ri] -= chain_load
            route_loads[rj] += chain_load
            route_custs[ri] = {c for c, _ in routes[ri].stops}
            route_custs[rj] = {c for c, _ in routes[rj].stops}

    return [rt for rt in routes if rt.stops]

def two_opt_star(routes: List[Route], inst: CSDVRPInstance) -> List[Route]:
    """
    2-opt*: exchange the tails of two routes at the cheapest crossing point.
    Single pass, first improvement.
    """
    dm   = inst._dist
    c_km = inst.dist_cost
    improved = True
    while improved:
        improved = False
        for ri in range(len(routes) - 1):
            for rj in range(ri + 1, len(routes)):
                si = routes[ri].stops
                sj = routes[rj].stops
                if not si or not sj:
                    continue
                cap_i = inst.vtypes[routes[ri].vtype_idx]["capacity"]
                cap_j = inst.vtypes[routes[rj].vtype_idx]["capacity"]

                for pi in range(len(si)):
                    # Split ri after pi: prefix si[:pi+1], tail si[pi+1:]
                    load_i_prefix = sum(q for _, q in si[:pi + 1])
                    load_i_tail   = sum(q for _, q in si[pi + 1:])
                    tail_i        = si[pi + 1:]

                    for pj in range(len(sj)):
                        load_j_prefix = sum(q for _, q in sj[:pj + 1])
                        load_j_tail   = sum(q for _, q in sj[pj + 1:])
                        tail_j        = sj[pj + 1:]

                        # Check capacity of swapped routes
                        if load_i_prefix + load_j_tail > cap_i + 1e-9:
                            continue
                        if load_j_prefix + load_i_tail > cap_j + 1e-9:
                            continue

                        # Current cost of crossing arcs
                        end_i   = si[pi][0]
                        end_j   = sj[pj][0]
                        start_i = si[pi + 1][0] if pi + 1 < len(si) else 0
                        start_j = sj[pj + 1][0] if pj + 1 < len(sj) else 0
                        next_i  = si[pi + 1][0] if pi + 1 < len(si) else 0
                        next_j  = sj[pj + 1][0] if pj + 1 < len(sj) else 0

                        old_cost = c_km * (dm[end_i, next_i] + dm[end_j, next_j])
                        new_cost = c_km * (dm[end_i, next_j] + dm[end_j, next_i])

                        if new_cost - old_cost < -1e-6:
                            new_i = si[:pi + 1] + tail_j
                            new_j = sj[:pj + 1] + tail_i
                            # a vehicle may serve each customer at most once
                            if (len({c for c, _ in new_i}) < len(new_i) or
                                    len({c for c, _ in new_j}) < len(new_j)):
                                continue
                            routes[ri].stops = new_i
                            routes[rj].stops = new_j
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

    return [r for r in routes if r.stops]
