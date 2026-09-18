"""
run_exact_milp.py — exact MILP solutions for the small instances.

    python run_exact_milp.py --shard 0 --nshards 2

For n = 6, 8, 10, 12 and instance seeds 1-5 the MILP is solved with CBC
(time limit TL seconds, relative gap 0) using K = (Clarke-Wright routes + 1)
vehicle copies. For every instance solved to proven optimality the model is
solved again with K + 1 vehicles to check that the extra vehicle does not
change the optimum. Output: results_v2/exact_milp_shard<k>.csv
"""
import argparse, csv, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from src.gen_exact_instance import make_exact_instance
from src.exact_milp import solve_exact
from src.clarke_wright import clarke_wright

SIZES = [6, 8, 10, 12]
SEEDS = [1, 2, 3, 4, 5]
TL = 600

FIELDS = ["instance", "n", "seed", "split_eligible", "total_demand", "Q",
          "cw_routes", "K", "status", "proven", "Z", "LB", "gap_pct",
          "time_s", "vehicles_used", "cbc_version",
          "K1_status", "K1_Z", "K1_time_s", "K1_vehicles_used"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--sizes", default=",".join(str(s) for s in SIZES))
    ap.add_argument("--tl", type=int, default=TL)
    ap.add_argument("--tag", default="")
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    args = ap.parse_args()
    sizes = [int(x) for x in args.sizes.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    jobs = [(n, s) for n in sizes for s in seeds][args.shard::args.nshards]
    out = ROOT / "results_v2" / f"exact_milp{args.tag}_shard{args.shard}.csv"
    out.parent.mkdir(exist_ok=True)
    done = set()
    if out.exists():
        done = {r["instance"] for r in csv.DictReader(open(out, newline=""))}
    new = not out.exists()
    fh = open(out, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new:
        w.writeheader()
    for n, s in jobs:
        inst = make_exact_instance(n, seed=s)
        if inst.name in done:
            continue
        cw = clarke_wright(inst, seed=0).n_routes_active()
        K = cw + 1
        r = solve_exact(inst, time_limit=args.tl, K=K)
        row = {"instance": inst.name, "n": n, "seed": s,
               "split_eligible": inst.params["n_split_eligible"],
               "total_demand": round(inst.total_demand, 2),
               "Q": inst.vtypes[0]["capacity"], "cw_routes": cw, "K": K,
               "status": r["status"], "proven": r["proven_optimal"],
               "Z": r["Z"], "LB": r["lower_bound"], "gap_pct": r["gap_pct"],
               "time_s": round(r["time_s"], 2),
               "vehicles_used": r["vehicles_used"],
               "cbc_version": r["cbc_version"]}
        if r["proven_optimal"]:
            r1 = solve_exact(inst, time_limit=args.tl, K=K + 1)
            row.update({"K1_status": r1["status"], "K1_Z": r1["Z"],
                        "K1_time_s": round(r1["time_s"], 2),
                        "K1_vehicles_used": r1["vehicles_used"]})
        w.writerow(row)
        fh.flush()
        print(f"{inst.name}: {r['status']} Z={r['Z']} LB={r['lower_bound']} "
              f"K={K} used={r['vehicles_used']} {r['time_s']:.0f}s "
              f"| K+1: {row.get('K1_status')} {row.get('K1_Z')}", flush=True)


if __name__ == "__main__":
    main()
