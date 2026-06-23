"""
run_exact_gaps.py — Optimality-gap experiment for the C-SDVRP.

For small homogeneous, N1-active instances:
  1. Solve the MILP exactly (CBC) to obtain a proven optimum or a valid
     lower bound + MILP gap.
  2. Run CW, VNS, Classical ALNS, and DRL-ALNS on the same instances.
  3. Report each method's optimality gap relative to the MILP optimum
     (proven instances) or best known bound.

Outputs results/exact_gaps.csv
"""
import sys, os, time, csv
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.gen_exact_instance import make_exact_instance
from src.exact_milp        import solve_exact
from src.alns              import _imax_for_n
from src.alns_plus         import ClassicalALNSPlus, DRLALNSPlus, N_ACTIONS_PLUS
from src.vns               import VNS
from src.ppo               import PPOAgent

BASE       = os.path.dirname(os.path.abspath(__file__))
OUT        = os.path.join(BASE, "results")
os.makedirs(OUT, exist_ok=True)
MODEL_PATH = os.path.join(BASE, "models", "ppo_drl_alns.pt")

SIZES      = [6, 8, 10, 12, 14]
SEEDS      = [1, 2, 3, 4, 5]
MILP_TL    = 300       # seconds per MILP
N_RUNS     = 3         # metaheuristic runs per instance

def main():
    agent = PPOAgent(n_actions=N_ACTIONS_PLUS)
    if os.path.exists(MODEL_PATH):
        agent.load(MODEL_PATH, for_inference=True)

    rows = []
    for n in SIZES:
        for seed in SEEDS:
            inst = make_exact_instance(n, seed=seed, size_class="S")
            elig = inst.params["n_split_eligible"]
            ex   = solve_exact(inst, time_limit=MILP_TL)
            bound = ex["lower_bound"]
            ref   = ex["Z"] if ex["proven_optimal"] else bound  # gap denominator

            def gap(z):
                return (z - ref) / ref * 100.0 if ref else float("nan")

            # metaheuristics: best of N_RUNS (enhanced methods)
            Imax = _imax_for_n(n)
            alns_best = min(ClassicalALNSPlus().solve(inst, I_max=Imax, seed=r)["Z_best"]
                            for r in range(N_RUNS))
            vns_best  = min(VNS().solve(inst, I_max=2000, seed=r)["Z_best"]
                            for r in range(N_RUNS))
            drl_best  = min(DRLALNSPlus().solve(inst, agent, I_max=Imax, seed=r,
                            train=False)["Z_best"] for r in range(N_RUNS))
            from src.clarke_wright import clarke_wright
            cw_z = clarke_wright(inst, seed=0).objective()

            row = {
                "n": n, "seed": seed, "split_eligible": elig,
                "MILP_status": ex["status"], "proven": ex["proven_optimal"],
                "MILP_Z": round(ex["Z"], 3) if ex["Z"] else None,
                "MILP_LB": round(bound, 3) if bound else None,
                "MILP_gap_pct": round(ex["gap_pct"], 2) if ex["gap_pct"] is not None else None,
                "MILP_time_s": round(ex["time_s"], 1),
                "CW_Z": round(cw_z, 3), "CW_gap": round(gap(cw_z), 2),
                "VNS_Z": round(vns_best, 3), "VNS_gap": round(gap(vns_best), 2),
                "ALNS_Z": round(alns_best, 3), "ALNS_gap": round(gap(alns_best), 2),
                "DRL_Z": round(drl_best, 3), "DRL_gap": round(gap(drl_best), 2),
            }
            rows.append(row)
            print(f"n={n:2d} s={seed} elig={elig:2d} | MILP {ex['status']:9s} "
                  f"Z={row['MILP_Z']} LB={row['MILP_LB']} "
                  f"| CW {row['CW_gap']:+.1f}% VNS {row['VNS_gap']:+.1f}% "
                  f"ALNS {row['ALNS_gap']:+.1f}% DRL {row['DRL_gap']:+.1f}%", flush=True)

    keys = list(rows[0].keys())
    with open(os.path.join(OUT, "exact_gaps.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"\nSaved {len(rows)} rows to results/exact_gaps.csv")

    # summary by n (proven instances only)
    print("\nMean optimality gap by n (proven-optimal instances):")
    for n in SIZES:
        sub = [r for r in rows if r["n"] == n and r["proven"]]
        if sub:
            import statistics as st
            print(f"  n={n:2d} ({len(sub)} proven): "
                  f"CW {st.mean(r['CW_gap'] for r in sub):.2f}% | "
                  f"VNS {st.mean(r['VNS_gap'] for r in sub):.2f}% | "
                  f"ALNS {st.mean(r['ALNS_gap'] for r in sub):.2f}% | "
                  f"DRL {st.mean(r['DRL_gap'] for r in sub):.2f}%")

if __name__ == "__main__":
    main()
