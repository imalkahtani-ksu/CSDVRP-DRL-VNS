"""
tune_vns.py — tune the VNS baseline on validation instances.

    python tune_vns.py

The learned methods are selected on held-out validation instances, so the
baseline is given the same courtesy: its one free parameter, the number of
shake neighborhoods k_max, is swept on the same validation instances (never
on the 42 benchmark instances) under the standard CPU budgets. The setting
with the best mean improvement over Clarke-Wright is the one used in the
paper. Output: results_v2/tune_vns.csv
"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
torch.set_num_threads(1)

from src.gen_benchmark import make_instance
from src.vns import VNS
from src.budgets import BUDGET

# the held-out instances used for model selection, never the test set
VALID = [("S", 20, 9000), ("M", 50, 9002), ("L", 75, 9004),
         ("L", 100, 9006), ("XL", 150, 9007)]
KMAX = [2, 4, 6, 8, 12]
RUNS = 3


def main():
    out = ROOT / "results_v2" / "tune_vns.csv"
    out.parent.mkdir(exist_ok=True)
    rows = []
    for k in KMAX:
        gains = []
        for sc, n, seed in VALID:
            inst = make_instance(n, sc, seed=seed)
            for r in range(RUNS):
                res = VNS().solve(inst, seed=r, time_limit=BUDGET[n], k_max=k)
                g = (res["Z_init"] - res["Z_best"]) / res["Z_init"] * 100
                gains.append(g)
                rows.append({"k_max": k, "n": n, "run": r,
                             "Z_best": round(res["Z_best"], 4),
                             "Z_init": round(res["Z_init"], 4),
                             "impr_pct": round(g, 4),
                             "iters": res["iters"]})
        print(f"k_max={k:2d}: mean improvement {np.mean(gains):.3f}% "
              f"over {len(gains)} runs", flush=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    best = max(KMAX, key=lambda k: np.mean([r["impr_pct"] for r in rows
                                            if r["k_max"] == k]))
    print(f"\nbest k_max on the validation instances: {best}")
    print("wrote", out)


if __name__ == "__main__":
    main()
