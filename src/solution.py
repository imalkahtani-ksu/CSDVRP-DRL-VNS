"""
solution.py — Solution representation and objective computation for C-SDVRP.

A Solution is a list of routes. Each route is assigned one vehicle type and
contains an ordered list of (customer_id, quantity_delivered) pairs.

Objective (Eq. 1):
  Z = dist_cost * sum(d_ij * x_ijk)   [routing cost]
    + mu * sum(Gamma_i)               [excess emission penalty, N3]
    + delta * sum(max(0, s_i - 1))    [split activation penalty, N1]

Arc emission (Eq. 9):
  E_ijk = e_k * dist(i,j) * (1 + beta_k * w_ijk / Q_k)

Excess emission (Eq. 10-11):
  E_i^A = sum over incoming arcs of E_jik
  Gamma_i = max(0, E_i^A - eta_i * D_i)
"""

import copy
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

import numpy as np

from .instance import CSDVRPInstance

@dataclass
class Route:
    """One vehicle route: vehicle type index + ordered delivery stops."""
    vtype_idx: int                            # index into instance.vtypes
    stops: List[Tuple[int, float]] = field(default_factory=list)
    # stops[k] = (customer_id, quantity_delivered)

    @property
    def load(self) -> float:
        return sum(q for _, q in self.stops)

    @property
    def n_stops(self) -> int:
        return len(self.stops)

    def copy(self) -> "Route":
        return Route(self.vtype_idx, list(self.stops))

class Solution:
    """
    Full solution to a C-SDVRP instance.
    Maintains routes and caches the objective value.
    """

    def __init__(self, instance: CSDVRPInstance, routes: List[Route]):
        self.instance = instance
        self.routes   = routes
        self._obj_cache: float = None   # invalidated on mutation

    # objective decomposition
    def objective(self) -> float:
        if self._obj_cache is not None:
            return self._obj_cache
        rc  = self._routing_cost()
        ep  = self._emission_penalty()
        sp  = self._split_penalty()
        self._obj_cache = rc + ep + sp
        return self._obj_cache

    def invalidate(self):
        self._obj_cache = None

    def components(self) -> Dict[str, float]:
        """Return all three cost components as a dict."""
        return {
            "routing":  self._routing_cost(),
            "emission": self._emission_penalty(),
            "split":    self._split_penalty(),
            "total":    self.objective(),
        }

    # routing cost
    def _routing_cost(self) -> float:
        cost = 0.0
        inst = self.instance
        c    = inst.dist_cost
        dm   = inst._dist
        for route in self.routes:
            if not route.stops:
                continue
            prev = 0
            for cid, _ in route.stops:
                cost += c * dm[prev, cid]
                prev  = cid
            cost += c * dm[prev, 0]   # return to depot
        return cost

    # emission penalty (N3)
    def _emission_penalty(self) -> float:
        inst          = self.instance
        n             = inst.n
        cust_emit     = np.zeros(n + 1)   # arrival emissions per node (index 0..n)

        for route in self.routes:
            if not route.stops:
                continue
            vt      = inst.vtypes[route.vtype_idx]
            e_k     = vt["emit_base"]
            beta_k  = vt["emit_beta"]
            Q_k     = vt["capacity"]
            dm      = inst._dist

            load = route.load          # vehicle load when leaving depot
            prev = 0
            for cid, qty in route.stops:
                dist = dm[prev, cid]
                # w_ijk = load carried on arc (prev → cid)
                arc_e = e_k * dist * (1.0 + beta_k * load / Q_k)
                cust_emit[cid] += arc_e
                load  -= qty            # unload qty at this customer
                prev   = cid

        # Excess emission per customer
        total_penalty = 0.0
        eta    = inst.eta
        demand = inst.demand
        mu     = inst.mu
        for i in range(1, n + 1):
            threshold = eta[i] * demand[i]
            gamma_i   = max(0.0, cust_emit[i] - threshold)
            total_penalty += mu * gamma_i

        return total_penalty

    # split penalty (N1)
    def _split_penalty(self) -> float:
        sc = self.split_counts()
        return self.instance.delta * sum(max(0, v - 1) for v in sc.values())

    # split counts: cid → number of vehicle visits
    def split_counts(self) -> Dict[int, int]:
        sc: Dict[int, int] = {}
        for route in self.routes:
            for cid, _ in route.stops:
                sc[cid] = sc.get(cid, 0) + 1
        return sc

    # delivered quantities: cid → total quantity
    def delivered(self) -> Dict[int, float]:
        dq: Dict[int, float] = {}
        for route in self.routes:
            for cid, qty in route.stops:
                dq[cid] = dq.get(cid, 0.0) + qty
        return dq

    # feasibility checks
    def is_feasible(self, tol: float = 1e-6) -> Tuple[bool, str]:
        inst = self.instance
        # Check demand satisfaction
        dq   = self.delivered()
        for i in range(1, inst.n + 1):
            if abs(dq.get(i, 0.0) - inst.demand[i]) > tol:
                return False, f"Demand not met at node {i}"
        # Check vehicle capacity
        for ri, route in enumerate(self.routes):
            Q = inst.vtypes[route.vtype_idx]["capacity"]
            if route.load > Q + tol:
                return False, f"Capacity exceeded on route {ri}"
        # Check N1: split bound
        sc = self.split_counts()
        for i in range(1, inst.n + 1):
            alpha_i = inst.alpha[i]
            if alpha_i > 0 and sc.get(i, 1) > alpha_i + 1:
                # alpha_i=1 means at most 2 visits (1 base + 1 split)
                # We penalise but don't hard-violate in heuristic (soft constraint via penalty)
                pass
        return True, "OK"

    # utility
    def n_routes_active(self) -> int:
        return sum(1 for r in self.routes if r.stops)

    def copy(self) -> "Solution":
        new_routes = [r.copy() for r in self.routes]
        s          = Solution(self.instance, new_routes)
        s._obj_cache = self._obj_cache
        return s

    def __repr__(self):
        return (f"Solution(Z={self.objective():.3f}, "
                f"routes={self.n_routes_active()}, "
                f"splits={sum(max(0,v-1) for v in self.split_counts().values())})")
