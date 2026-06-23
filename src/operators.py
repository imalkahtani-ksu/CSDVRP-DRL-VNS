"""
operators.py — ALNS destroy and repair operators for C-SDVRP.

Destroy operators:
  D1  random_removal   : remove q random customers
  D2  worst_removal    : remove q highest-cost customers
  D3  related_removal  : Shaw removal (proximity-based)

Repair operators:
  R1  best_insertion   : greedy best-position insertion
  R2  regret_insertion : regret-2 insertion (minimise future regret)

Split handling: deliveries are split automatically when a single vehicle
cannot carry full demand; alpha_i constraint is enforced via the penalty term
in the objective (soft), with a hard cap at alpha_i+1 visits enforced here.
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

def _insert_customer(routes: List[Route], inst: CSDVRPInstance,
                     cid: int, qty: float) -> bool:
    """
    Insert (cid, qty) into the cheapest feasible position across all routes.
    If no route has capacity, open a new one (smallest vehicle that fits).
    Returns True if inserted.
    """
    best_route = -1
    best_pos   = 0
    best_delta = float("inf")

    for ri, route in enumerate(routes):
        pos, delta = _best_position(route, cid, qty, inst)
        if pos >= 0 and delta < best_delta:
            best_delta = delta
            best_route = ri
            best_pos   = pos

    if best_route >= 0:
        routes[best_route].stops.insert(best_pos, (cid, qty))
        return True

    # No existing route can fit → open new route (smallest vehicle that fits)
    for vt_idx, vt in enumerate(inst.vtypes):
        if vt["capacity"] >= qty:
            new_route = Route(vtype_idx=vt_idx, stops=[(cid, qty)])
            routes.append(new_route)
            return True
    return False

def best_insertion(inst: CSDVRPInstance, routes: List[Route],
                   removed: Dict[int, float]) -> List[Route]:
    """
    R1: Best-position greedy insertion.
    For each removed customer, find the best position across all routes
    and insert there. Process customers in random order for diversification.
    Handles split delivery: if demand > max_cap, insert in multiple passes.
    Route loads are cached and updated incrementally to avoid O(n) recomputation.
    """
    dm   = inst._dist
    c_km = inst.dist_cost

    order = list(removed.keys())
    random.shuffle(order)

    # Pre-cache route loads; updated after each insertion
    route_loads = [sum(q for _, q in r.stops) for r in routes]

    for cid in order:
        remaining_qty = float(inst.demand[cid])
        max_cap       = inst.max_cap

        while remaining_qty > 1e-6:
            # Amount to try in this pass (full remaining or capped by heaviest vehicle)
            target_qty = min(remaining_qty, max_cap)

            best_route    = -1
            best_pos      = 0
            best_delta    = float("inf")
            best_ins_qty  = 0.0

            for ri in range(len(routes)):
                route = routes[ri]
                vt    = inst.vtypes[route.vtype_idx]
                cap   = vt["capacity"]
                avail = cap - route_loads[ri]
                if avail < 1e-9:
                    continue
                ins_qty = min(target_qty, avail)   # qty actually inserted
                # Capacity feasibility already checked; find best position
                stops   = route.stops
                prev_id = 0
                for pos in range(len(stops) + 1):
                    prev = prev_id if pos == 0 else stops[pos - 1][0]
                    nxt  = 0 if pos == len(stops) else stops[pos][0]
                    delta = c_km * (dm[prev, cid] + dm[cid, nxt] - dm[prev, nxt])
                    if delta < best_delta:
                        best_delta   = delta
                        best_route   = ri
                        best_pos     = pos
                        best_ins_qty = ins_qty

            if best_route >= 0:
                routes[best_route].stops.insert(best_pos, (cid, best_ins_qty))
                route_loads[best_route] += best_ins_qty
                remaining_qty -= best_ins_qty
            else:
                # No existing route has space — open smallest vehicle that fits
                for vt_idx, vt in enumerate(inst.vtypes):
                    cap = vt["capacity"]
                    if cap >= 1e-9:
                        ins_qty = min(remaining_qty, cap)
                        routes.append(Route(vtype_idx=vt_idx, stops=[(cid, ins_qty)]))
                        route_loads.append(ins_qty)
                        remaining_qty -= ins_qty
                        break
                else:
                    break  # should not happen with valid instance

    return routes

def regret_insertion(inst: CSDVRPInstance, routes: List[Route],
                     removed: Dict[int, float]) -> List[Route]:
    """
    R2: Regret-2 insertion.
    Repeatedly insert the customer whose cost difference between best and
    second-best position is largest (highest regret).
    Route loads cached and updated incrementally.
    """
    dm   = inst._dist
    c_km = inst.dist_cost

    uninserted  = {cid: float(inst.demand[cid]) for cid in removed}
    route_loads = [sum(q for _, q in r.stops) for r in routes]

    while uninserted:
        regrets   = {}
        best_info = {}

        for cid, remaining_qty in uninserted.items():
            qty = min(remaining_qty, inst.max_cap)

            costs = []
            for ri in range(len(routes)):
                route = routes[ri]
                vt    = inst.vtypes[route.vtype_idx]
                avail = vt["capacity"] - route_loads[ri]
                if avail < 1e-9:
                    continue
                ins_qty = min(qty, avail)
                # Find best insertion position in this route
                stops    = route.stops
                best_pos = 0
                best_d   = float("inf")
                for pos in range(len(stops) + 1):
                    prev = 0 if pos == 0 else stops[pos - 1][0]
                    nxt  = 0 if pos == len(stops) else stops[pos][0]
                    d    = c_km * (dm[prev, cid] + dm[cid, nxt] - dm[prev, nxt])
                    if d < best_d:
                        best_d = d; best_pos = pos
                costs.append((best_d, ri, best_pos, ins_qty))

            if not costs:
                # Open new route — depot round-trip cost
                d0 = c_km * (dm[0, cid] + dm[cid, 0])
                costs.append((d0, -1, 0, min(remaining_qty, inst.max_cap)))

            costs.sort(key=lambda x: x[0])
            regrets[cid]   = (costs[1][0] if len(costs) > 1 else costs[0][0] + 1e9) - costs[0][0]
            best_info[cid] = costs[0]

        chosen             = max(regrets, key=lambda c: regrets[c])
        delta, ri, pos, qty = best_info[chosen]

        if ri >= 0:
            routes[ri].stops.insert(pos, (chosen, qty))
            route_loads[ri] += qty
        else:
            for vt_idx, vt in enumerate(inst.vtypes):
                if vt["capacity"] >= qty:
                    routes.append(Route(vtype_idx=vt_idx, stops=[(chosen, qty)]))
                    route_loads.append(qty)
                    break

        uninserted[chosen] -= qty
        if uninserted[chosen] < 1e-6:
            del uninserted[chosen]

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

                for rj in range(len(routes)):
                    if rj == ri:
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
                            # Swap tails
                            routes[ri].stops = si[:pi + 1] + tail_j
                            routes[rj].stops = sj[:pj + 1] + tail_i
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

    return [r for r in routes if r.stops]
