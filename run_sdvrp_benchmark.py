"""
run_sdvrp_benchmark.py — the solvers of this paper on the classical SDVRP.

    python tools/fetch_sdvrp_set1.py          # once
    python run_sdvrp_benchmark.py --shard 0 --nshards 4

Switching the three controls off (every customer split-eligible, no visit cap,
delta = 0 and mu = 0) turns the C-SDVRP into the classical SDVRP, so the same
code can be run on the public SET-1 instances and compared with the reference
values of the DIMACS 2022 challenge. Distances are Euclidean rounded to the
nearest integer, as in the challenge.

Note that the DRL-VNS policy is trained on C-SDVRP instances with delta = 15
and on at most 75 customers; running it here is therefore an out-of-
distribution transfer test, not a like-for-like comparison.

Output: results_v2/sdvrp_set1_shard<k>.csv
"""
import argparse, csv, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
torch.set_num_threads(1)

from src.instance import CSDVRPInstance
from src.clarke_wright import clarke_wright
from src.vns import VNS
from src.alns_plus import ClassicalALNSPlus
from src.drl_vns import VNS9, DRLVNS, N_ACTIONS_VNS, STATE_DIM_VNS
from src.ppo import PPOAgent

DATA = ROOT / "data" / "sdvrp_set1"
RES = ROOT / "results_v2"
MODELS = ROOT / "models_v2"
N_RUNS = 3
METHODS = ["VNS", "ALNS+", "VNS-9", "DRL-VNS"]

FIELDS = ["instance", "n", "Q", "method", "run", "budget_s", "Z_best",
          "Z_init", "best_known", "gap_pct", "cpu_s", "iters", "routes",
          "split_customers", "demand_viol", "cap_viol", "dup_visit_viol"]


def load_instance(path: Path) -> CSDVRPInstance:
    nums = path.read_text().split()
    k = 0
    n = int(nums[k]); k += 1
    Q = float(nums[k]); k += 1
    demands = [float(nums[k + i]) for i in range(n)]
    k += n
    coords = []
    for _ in range(n + 1):
        coords.append((float(nums[k]), float(nums[k + 1])))
        k += 2
    rows = [{"node_id": 0, "x": coords[0][0], "y": coords[0][1], "demand": 0.0,
             "alpha": 0, "U_max": 0, "eta": 0.0, "node_type": "depot"}]
    for i in range(1, n + 1):
        rows.append({"node_id": i, "x": coords[i][0], "y": coords[i][1],
                     "demand": demands[i - 1], "alpha": 1, "U_max": 999,
                     "eta": 0.0, "node_type": "customer"})
    nodes = pd.DataFrame(rows)
    veh = pd.DataFrame([{"vtype_id": 1, "label": "SDVRP", "capacity": Q,
                         "emit_base": 0.0, "emit_beta": 0.0,
                         "count": n, "total_cap": Q * n}])
    params = {"instance": path.stem, "n": n, "size_class": "SDVRP", "Q": Q,
              "delta": 0.0, "mu": 0.0, "dist_cost": 1.0, "U_max": 999,
              "n_split_eligible": n}
    inst = CSDVRPInstance(nodes, veh, params)
    # the challenge rounds Euclidean distances to the nearest integer
    xy = nodes[["x", "y"]].values
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    inst._dist = np.rint(d)
    return inst


def budget_for(n: int) -> int:
    if n <= 50:
        return 60
    if n <= 100:
        return 120
    if n <= 200:
        return 240
    return 300


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    bks = {}
    bf = DATA / "best_known.csv"
    if bf.exists():
        for r in csv.DictReader(open(bf)):
            bks[r["instance"]] = int(r["best_known"])

    agents = {}

    def policy(k):
        if k not in agents:
            a = PPOAgent(n_actions=N_ACTIONS_VNS, state_dim=STATE_DIM_VNS)
            a.load(str(MODELS / f"vns_seed{k}_best.pt"), for_inference=True)
            agents[k] = a
        return agents[k]

    jobs = []
    for i in range(1, 22):
        for m in METHODS:
            for r in range(N_RUNS):
                jobs.append((f"SD{i}", m, r))
    jobs = [jobs[i] for i in np.random.default_rng(7).permutation(len(jobs))]
    jobs = jobs[args.shard::args.nshards]

    RES.mkdir(exist_ok=True)
    out = RES / f"sdvrp_set1_shard{args.shard}.csv"
    done = set()
    if out.exists():
        for r in csv.DictReader(open(out, newline="")):
            done.add((r["instance"], r["method"], int(r["run"])))
    new = not out.exists()
    fh = open(out, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new:
        w.writeheader()

    cache = {}
    for name, method, run in jobs:
        if (name, method, run) in done:
            continue
        if name not in cache:
            cache.clear()
            cache[name] = load_instance(DATA / f"{name}.txt")
        inst = cache[name]
        tb = budget_for(inst.n)
        if method == "VNS":
            solver = VNS()
        elif method == "ALNS+":
            solver = ClassicalALNSPlus()
        elif method == "VNS-9":
            solver = VNS9("cyclic")
        else:
            solver = DRLVNS(policy(run % 5 + 1))
        res = solver.solve(inst, seed=run, time_limit=tb)
        a = res["best"].audit()
        ref = bks.get(name)
        row = {"instance": name, "n": inst.n, "Q": inst.vtypes[0]["capacity"],
               "method": method, "run": run, "budget_s": tb,
               "Z_best": round(res["Z_best"], 2),
               "Z_init": round(res["Z_init"], 2), "best_known": ref,
               "gap_pct": round((res["Z_best"] - ref) / ref * 100, 3) if ref else None,
               "cpu_s": round(res["cpu_s"], 2), "iters": res["iters"],
               "routes": a["routes"], "split_customers": a["split_customers"],
               "demand_viol": a["demand_viol"], "cap_viol": a["cap_viol"],
               "dup_visit_viol": a["dup_visit_viol"]}
        w.writerow(row)
        fh.flush()
        print(f"{name} n={inst.n} {method} run{run}: Z={row['Z_best']} "
              f"bks={ref} gap={row['gap_pct']}%", flush=True)
    print("shard", args.shard, "done", flush=True)


if __name__ == "__main__":
    main()
