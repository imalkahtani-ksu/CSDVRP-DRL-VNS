"""
instance.py — C-SDVRP instance loader and data container.

Novel constraints embedded in node/vehicle data:
  N1  alpha_i  : max split visits per customer
  N2  U_max    : receiving dock capacity per time-slot
  N3  eta_i    : emission threshold coefficient
"""

import json
import math
import numpy as np
import pandas as pd
from pathlib import Path

class CSDVRPInstance:
    """Holds all data for one C-SDVRP instance plus a precomputed distance matrix."""

    def __init__(self, nodes: pd.DataFrame, vehicles: pd.DataFrame, params: dict):
        self.nodes    = nodes.copy().reset_index(drop=True)
        self.vehicles = vehicles.copy().reset_index(drop=True)
        self.params   = params
        self.n        = int(params["n"])          # number of customers (excl. depot)
        self.name     = params.get("instance", "unknown")

        # precomputed Euclidean distance matrix (n+1) × (n+1)
        coords = self.nodes[["x", "y"]].values   # shape (n+1, 2)
        diff   = coords[:, None, :] - coords[None, :, :]  # broadcasting
        self._dist = np.sqrt((diff ** 2).sum(-1))          # (n+1, n+1)

        # convenience arrays (indexed 0..n, 0=depot)
        self.demand  = self.nodes["demand"].values.copy()   # float[n+1]
        self.alpha   = self.nodes["alpha"].values.copy()    # int[n+1]  (0 for depot)
        self.U_max   = self.nodes["U_max"].values.copy()    # int[n+1]
        self.eta     = self.nodes["eta"].values.copy()      # float[n+1]

        # vehicle type list (list of dicts, indexed 0..n_types-1)
        self.vtypes = self.vehicles.to_dict("records")
        self.n_types = len(self.vtypes)

        # Global params
        self.delta     = float(params.get("delta",     15.0))
        self.mu        = float(params.get("mu",         0.05))
        self.dist_cost = float(params.get("dist_cost",  1.0))

    # distance query
    def dist(self, i: int, j: int) -> float:
        return float(self._dist[i, j])

    # total fleet capacity
    @property
    def total_fleet_cap(self) -> float:
        return sum(v["capacity"] * v["count"] for v in self.vtypes)

    # max vehicle capacity across all types
    @property
    def max_cap(self) -> float:
        return max(v["capacity"] for v in self.vtypes)

    # total demand
    @property
    def total_demand(self) -> float:
        return float(self.demand[1:].sum())

    # factory: load from benchmark folder
    @classmethod
    def from_folder(cls, folder: Path) -> "CSDVRPInstance":
        folder = Path(folder)
        nodes    = pd.read_csv(folder / "nodes.csv")
        vehicles = pd.read_csv(folder / "vehicles.csv")
        with open(folder / "params.json") as f:
            params = json.load(f)
        return cls(nodes, vehicles, params)

    # factory: generate random instance (for training)
    @classmethod
    def random(cls, n: int, size_class: str = "M", seed: int = None) -> "CSDVRPInstance":
        """Generate a random C-SDVRP training instance."""
        rng = np.random.default_rng(seed)

        # Coordinates
        coords = rng.uniform(0, 100, (n + 1, 2)).round(2)
        depot  = rng.uniform(35, 65, 2).round(2)
        coords[0] = depot

        # Demand range by class
        d_range = {"S": (1.0, 5.0), "M": (1.5, 7.0), "L": (2.0, 9.0), "XL": (3.0, 12.0)}
        lo, hi  = d_range.get(size_class, (1.5, 7.0))
        demands = rng.uniform(lo, hi, n).round(2)
        median_d = float(np.median(demands))

        # Build nodes DataFrame
        rows = [{"node_id": 0, "x": float(depot[0]), "y": float(depot[1]),
                 "demand": 0.0, "alpha": 0, "U_max": 0, "eta": 0.0, "node_type": "depot"}]
        for i in range(n):
            d = float(demands[i])
            rows.append({
                "node_id": i + 1,
                "x": float(coords[i + 1, 0]),
                "y": float(coords[i + 1, 1]),
                "demand": d,
                "alpha": 1 if d > median_d else 0,
                "U_max": int(np.clip(math.ceil(d / hi * 3), 1, 3)),
                "eta": round(float(rng.uniform(0.60, 1.40)), 3),
                "node_type": "customer"
            })
        nodes = pd.DataFrame(rows)

        # Fleet templates
        fleets = {
            "S":  [dict(vtype_id=1, label="Light",  capacity=5.0,  emit_base=0.20, emit_beta=0.30),
                   dict(vtype_id=2, label="Medium", capacity=10.0, emit_base=0.28, emit_beta=0.28)],
            "M":  [dict(vtype_id=1, label="Light",  capacity=5.0,  emit_base=0.20, emit_beta=0.30),
                   dict(vtype_id=2, label="Medium", capacity=10.0, emit_base=0.28, emit_beta=0.28),
                   dict(vtype_id=3, label="Heavy",  capacity=18.0, emit_base=0.38, emit_beta=0.26)],
            "L":  [dict(vtype_id=1, label="Light",   capacity=5.0,  emit_base=0.20, emit_beta=0.30),
                   dict(vtype_id=2, label="Medium",  capacity=10.0, emit_base=0.28, emit_beta=0.28),
                   dict(vtype_id=3, label="Heavy",   capacity=18.0, emit_base=0.38, emit_beta=0.26),
                   dict(vtype_id=4, label="XHeavy",  capacity=26.0, emit_base=0.50, emit_beta=0.24)],
            "XL": [dict(vtype_id=1, label="Medium",   capacity=20.0,  emit_base=0.28, emit_beta=0.28),
                   dict(vtype_id=2, label="Heavy",    capacity=35.0,  emit_base=0.38, emit_beta=0.26),
                   dict(vtype_id=3, label="XHeavy",   capacity=55.0,  emit_base=0.50, emit_beta=0.24),
                   dict(vtype_id=4, label="XXHeavy",  capacity=80.0,  emit_base=0.65, emit_beta=0.22),
                   dict(vtype_id=5, label="Mega",     capacity=120.0, emit_base=0.75, emit_beta=0.20)],
        }
        fleet_tmpl = fleets.get(size_class, fleets["M"])

        # Dynamic fleet sizing at ~65% utilisation
        total_demand  = float(demands.sum())
        avg_cap       = sum(v["capacity"] for v in fleet_tmpl) / len(fleet_tmpl)
        K             = max(len(fleet_tmpl), math.ceil(total_demand / (0.65 * avg_cap)))
        n_types       = len(fleet_tmpl)
        base, rem     = divmod(K, n_types)
        veh_rows = []
        for ti, vt in enumerate(fleet_tmpl):
            row = dict(vt)
            row["count"]     = base + (1 if ti < rem else 0)
            row["total_cap"] = round(vt["capacity"] * row["count"], 1)
            veh_rows.append(row)
        vehicles = pd.DataFrame(veh_rows)

        params = {
            "instance": f"train-{n}-{seed}", "n": n,
            "K_total": K, "total_fleet_cap": float(vehicles["total_cap"].sum()),
            "delta": 15.0, "mu": 0.05, "dist_cost": 1.0,
            "time_window_h": 7.0, "service_min": 20,
            "coord_range": [0, 100], "dist_metric": "Euclidean"
        }
        return cls(nodes, vehicles, params)

    def __repr__(self):
        return (f"CSDVRPInstance(name={self.name!r}, n={self.n}, "
                f"K={sum(v['count'] for v in self.vtypes)}, "
                f"demand={self.total_demand:.1f}T)")
