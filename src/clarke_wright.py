"""
clarke_wright.py — Clarke-Wright savings initialisation adapted for C-SDVRP.

Steps:
  1. For each customer, create one route per required vehicle (handles splits
     when demand > max vehicle capacity).
  2. Compute savings s(i,j) = d(0,i) + d(0,j) - d(i,j) for all pairs.
  3. Merge routes greedily (descending savings) when capacity permits.
  4. Assign vehicle types: smallest feasible vehicle to each route (minimises
     empty load, hence reduces emission).
"""

import math
from typing import List

import numpy as np

from .instance import CSDVRPInstance
from .solution  import Route, Solution

def clarke_wright(instance: CSDVRPInstance, seed: int = None) -> Solution:
    """
    Clarke-Wright savings initialisation.
    Returns a Solution with all demands satisfied.
    """
    rng     = np.random.default_rng(seed)
    inst    = instance
    n       = inst.n
    dm      = inst._dist
    demand  = inst.demand
    max_cap = inst.max_cap

    # Step 1: seed routes (one per required delivery chunk per customer)
    routes: List[Route] = []

    for i in range(1, n + 1):
        d_i       = float(demand[i])
        remaining = d_i
        # Assign vehicle type: pick smallest that fits the chunk
        while remaining > 1e-9:
            chunk = min(remaining, max_cap)
            vt_idx = _pick_vehicle(inst, chunk)
            routes.append(Route(vtype_idx=vt_idx, stops=[(i, chunk)]))
            remaining -= chunk

    # Step 2: compute savings
    savings = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            s = float(dm[0, i] + dm[0, j] - dm[i, j])
            savings.append((s, i, j))
    savings.sort(key=lambda x: -x[0])

    # Step 3: greedy merge
    # Route ownership: customer → set of route indices ending/starting at that customer
    # For standard CW: only merge route ending at i with route starting at j
    # (interior stops are locked)
    # We track which route each customer is "at the tail" and "at the head" of.

    tail_of: dict = {}   # cid → route_idx where cid is the LAST stop
    head_of: dict = {}   # cid → route_idx where cid is the FIRST stop
    route_open: List[bool] = [True] * len(routes)

    # Initialise: each seed route has one stop → it's both head and tail
    for ri, r in enumerate(routes):
        if r.stops:
            cid = r.stops[-1][0]
            qty = r.stops[-1][1]
            # Only track if it's the single stop (initial single-customer routes)
            if len(r.stops) == 1:
                tail_of[ri] = cid   # using ri as key for clarity
                head_of[ri] = cid

    # Rebuild as: cid → list of (route_idx, position) for endpoint tracking
    # Simpler approach: use end-customer index
    cid_tail: dict = {}   # cid → route_idx where cid is tail stop
    cid_head: dict = {}   # cid → route_idx where cid is head stop

    for ri, r in enumerate(routes):
        if r.stops:
            cid_tail[r.stops[-1][0]] = ri
            cid_head[r.stops[0][0]]  = ri

    # Perform merges
    for s, i, j in savings:
        if s <= 0:
            break
        # Route ending at i merges with route starting at j
        ri = cid_tail.get(i, -1)
        rj = cid_head.get(j, -1)
        if ri < 0 or rj < 0 or ri == rj:
            continue
        if not route_open[ri] or not route_open[rj]:
            continue
        # Check capacity
        rt_i = routes[ri]
        rt_j = routes[rj]
        if rt_i.load + rt_j.load > inst.vtypes[rt_i.vtype_idx]["capacity"] + 1e-9:
            continue
        # Merge rt_j into rt_i
        rt_i.stops.extend(rt_j.stops)
        rt_i.invalidate_if_applicable()
        # Reassign vehicle if needed (might need larger type)
        rt_i.vtype_idx = _pick_vehicle(inst, rt_i.load)
        # Update head/tail pointers
        del cid_tail[i]   # i no longer a tail of ri
        new_tail = rt_i.stops[-1][0]
        cid_tail[new_tail] = ri
        del cid_head[j]
        # Close rj
        route_open[rj] = False

    # Step 4: collect active routes
    active = [r for ri, r in enumerate(routes) if route_open[ri] and r.stops]
    # Re-assign vehicle types to minimise waste
    for r in active:
        r.vtype_idx = _pick_vehicle(inst, r.load)

    return Solution(inst, active)

def _pick_vehicle(inst: CSDVRPInstance, load: float,
                   target_util: float = 0.55) -> int:
    """Return index of vehicle type whose utilisation is closest to target_util.

    Targeting ~55 % utilisation leaves capacity slack so that ALNS can
    consolidate routes by inserting additional customers.  The smallest
    feasible vehicle is always a valid fallback (when all types fall above
    the target, their penalties are still compared and the least-bad one wins).
    """
    best_idx    = 0
    best_penalty = float("inf")
    for idx, vt in enumerate(inst.vtypes):
        cap = vt["capacity"]
        if cap < load - 1e-9:
            continue          # vehicle too small – skip
        penalty = abs(load / cap - target_util)
        if penalty < best_penalty:
            best_penalty = penalty
            best_idx     = idx
    return best_idx

def _pick_vehicle_tight(inst: CSDVRPInstance, load: float) -> int:
    """Return index of the *smallest* vehicle type that can carry `load`.

    Used as a post-processing step after ALNS to minimise vehicle size
    (and hence fuel/emission cost) once route loads are finalised.
    """
    best_idx = 0
    best_cap = float("inf")
    for idx, vt in enumerate(inst.vtypes):
        cap = vt["capacity"]
        if cap >= load - 1e-9 and cap < best_cap:
            best_cap = cap
            best_idx = idx
    return best_idx

# Monkey-patch Route to have an invalidate method used above
def _route_invalidate(self):
    pass  # Route doesn't cache; placeholder for compatibility

Route.invalidate_if_applicable = _route_invalidate
