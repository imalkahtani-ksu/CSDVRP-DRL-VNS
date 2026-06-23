"""
run_price_of_control.py — Structural analysis of the C-SDVRP constraints.

Quantifies the "price of control": how each novel constraint (N1, N2, N3)
changes the optimal/near-optimal solution relative to the unconstrained SDVRP.
For each instance we solve five configurations with the SAME solver budget:

    FULL     : C-SDVRP with N1 + N2 + N3 active
    no-N1    : every customer split-eligible (split activation removed)
    no-N2    : visit cap relaxed to a large value
    no-N3    : emission penalty delta = 0
    SDVRP    : none of N1/N2/N3 (classical split-delivery VRP)

Reports, per configuration: total cost, routing cost, emission penalty,
number of split customers, and total excess visits. The deltas vs SDVRP
reveal the structural effect (and managerial trade-offs) of each control.

Outputs results/price_of_control.csv
"""
import sys, os, csv, copy
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.gen_benchmark import make_instance
from src.alns          import _imax_for_n
from src.alns_plus     import ClassicalALNSPlus
from src.vns           import VNS

BASE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(BASE, "results")
os.makedirs(OUT, exist_ok=True)

SIZES  = [("S", 15), ("M", 30), ("L", 50)]
SEEDS  = [1, 2, 3, 4, 5]
N_RUNS = 5


def variant(inst, mode):
    """Return a copy of inst with one constraint relaxed."""
    import pandas as pd
    nodes = inst.nodes.copy()
    params = dict(inst.params)
    if mode == "no-N1" or mode == "SDVRP":
        # all customers split-eligible
        nodes.loc[nodes["node_type"] == "customer", "alpha"] = inst.params["U_max"] - 1
        nodes.loc[nodes["node_type"] == "customer", "U_max"] = inst.params["U_max"]
    if mode == "no-N2" or mode == "SDVRP":
        nodes.loc[nodes["node_type"] == "customer", "U_max"] = 999
    if mode == "no-N3" or mode == "SDVRP":
        params["delta"] = 0.0
    from src.instance import CSDVRPInstance
    return CSDVRPInstance(nodes, inst.vehicles.copy(), params)


def best_solution(inst, n):
    best = None
    for r in range(N_RUNS):
        res = ClassicalALNSPlus().solve(inst, I_max=_imax_for_n(n), seed=r)
        if best is None or res["Z_best"] < best.objective():
            best = res["best"]
    for r in range(N_RUNS):
        res = VNS().solve(inst, I_max=2000, seed=r)
        if res["Z_best"] < best.objective():
            best = res["best"]
    return best


def metrics(sol):
    comps = sol.components()
    sc = sol.split_counts()
    n_split = sum(1 for v in sc.values() if v > 1)
    excess  = sum(max(0, v - 1) for v in sc.values())
    return comps["total"], comps["routing"], comps["emission"], comps["split"], n_split, excess


def main():
    rows = []
    for sc, n in SIZES:
        for seed in SEEDS:
            base = make_instance(n, size_class=sc, seed=seed)
            elig = base.params["n_split_eligible"]
            for mode in ["FULL", "no-N1", "no-N2", "no-N3", "SDVRP"]:
                inst = base if mode == "FULL" else variant(base, mode)
                sol  = best_solution(inst, n)
                # always evaluate cost under the FULL objective for fair comparison?
                # No: each config optimizes its own objective; we report its own metrics
                tot, rt, em, sp, nsp, exc = metrics(sol)
                rows.append({
                    "size_class": sc, "n": n, "seed": seed,
                    "n_split_eligible": elig, "config": mode,
                    "total": round(tot, 2), "routing": round(rt, 2),
                    "emission": round(em, 3), "split_pen": round(sp, 2),
                    "n_split_cust": nsp, "excess_visits": exc,
                })
            r = rows[-5:]
            full = r[0]; sdvrp = r[4]
            print(f"{sc} n={n} s={seed} elig={elig}: "
                  f"FULL cost={full['total']} splits={full['n_split_cust']} | "
                  f"SDVRP cost={sdvrp['total']} splits={sdvrp['n_split_cust']}", flush=True)

    keys = list(rows[0].keys())
    with open(os.path.join(OUT, "price_of_control.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"\nSaved {len(rows)} rows to results/price_of_control.csv")


if __name__ == "__main__":
    main()
