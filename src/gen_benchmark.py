"""
C-SDVRP instance generator.

A homogeneous fleet of K identical vehicles (capacity Q) is used so the exact
MILP, the metaheuristics and the case study all solve the same problem class.
A customer is split-eligible when its demand meets a capacity-based threshold
d_i >= beta_Q * Q (a fixed value, not a fraction of the customer's own
demand), so the split-activation constraint restricts a well-defined subset
of customers. Demands mix many small "regular" customers with a minority of
large "bulk" customers, so splitting is worthwhile for some customers and the
constraints affect the optimal routes. Each instance is reproducible from
(n, size_class, seed).
"""
import math
import numpy as np
import pandas as pd

from .instance import CSDVRPInstance

# Vehicle capacity by size class (single homogeneous type)
CLASS_Q = {"S": 15.0, "M": 20.0, "L": 25.0, "XL": 30.0}

# Fixed model parameters (shared across all instances)
BETA_Q   = 0.5      # N1 threshold = BETA_Q * Q
U_MAX    = 3        # N2 visit cap for split-eligible customers
DELTA    = 15.0     # N3 per-excess-visit emission penalty
MU       = 0.05     # emission-penalty weight
EMIT_E   = 0.30     # base emission coefficient
EMIT_B   = 0.25     # load sensitivity of emissions


FRAC_LARGE = 0.25      # fraction of "bulk" (split-eligible) customers


def make_instance(n: int, size_class: str = "M", seed: int = 0,
                  beta_Q: float = BETA_Q, u_max: int = U_MAX,
                  frac_large: float = FRAC_LARGE) -> CSDVRPInstance:
    rng = np.random.default_rng(seed)
    Q   = CLASS_Q.get(size_class, 20.0)

    coords = rng.uniform(0, 100, (n + 1, 2)).round(2)
    coords[0] = [50.0, 50.0]

    # Demand model: most customers are "regular" (small, fit several per
    # vehicle so routing is non-trivial); a minority are "bulk" customers
    # whose demand is a large fraction of Q and therefore split-eligible.
    tau = beta_Q * Q                       # N1 threshold (exogenous)
    demands = rng.uniform(1.0, 0.25 * Q, n)            # regular customers
    n_large = max(1, round(frac_large * n))
    large_idx = rng.choice(n, size=n_large, replace=False)
    demands[large_idx] = rng.uniform(0.5 * Q, 0.9 * Q, n_large)  # bulk (>= tau)
    demands = demands.round(2)

    rows = [{"node_id": 0, "x": 50.0, "y": 50.0, "demand": 0.0,
             "alpha": 0, "U_max": 0, "eta": 0.0, "node_type": "depot"}]
    for i in range(n):
        d = float(demands[i])
        eligible = d >= tau
        rows.append({
            "node_id": i + 1,
            "x": float(coords[i + 1, 0]), "y": float(coords[i + 1, 1]),
            "demand": d,
            "alpha": (u_max - 1) if eligible else 0,   # >0 marks split-eligible
            "U_max": u_max if eligible else 1,          # N1: ineligible => 1 visit
            "eta": round(float(rng.uniform(0.6, 1.4)), 3),
            "node_type": "customer",
        })
    nodes = pd.DataFrame(rows)

    K = math.ceil(float(demands.sum()) / Q) + 1
    vehicles = pd.DataFrame([{
        "vtype_id": 1, "label": "Uniform", "capacity": Q,
        "emit_base": EMIT_E, "emit_beta": EMIT_B, "count": K,
        "total_cap": round(Q * K, 1),
    }])
    n_elig = int(sum(1 for r in rows[1:] if r["U_max"] > 1))
    params = {
        "instance": f"CSDVRP-{size_class}-n{n}-s{seed}", "n": n,
        "size_class": size_class, "K_total": K, "total_fleet_cap": float(Q * K),
        "Q": Q, "beta_Q": beta_Q, "tau": round(tau, 2), "U_max": u_max,
        "delta": DELTA, "mu": MU, "dist_cost": 1.0,
        "n_split_eligible": n_elig,
        "fleet_util_pct": round(float(demands.sum()) / (Q * K) * 100, 1),
    }
    return CSDVRPInstance(nodes, vehicles, params)
