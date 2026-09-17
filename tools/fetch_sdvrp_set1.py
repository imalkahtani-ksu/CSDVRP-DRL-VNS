"""
fetch_sdvrp_set1.py — download the classical SDVRP benchmark SET-1 (SD1-SD21).

    python tools/fetch_sdvrp_set1.py

The instances and the reference results come from the Alkaid-SDVRP repository
(MIT licence), which is the software and data snapshot of

    Lin, He, Jiang, Ma, Su and Lu, "AlkaidSD: An Efficient Open-Source Solver
    for the Split Delivery Vehicle Routing Problem", INFORMS Journal on
    Computing, 2024, DOI 10.1287/ijoc.2024.0606.

The reference value of each instance is the best solution value reported for
that instance by the seven solvers of the 12th DIMACS Implementation
Challenge (SDVRP track), as tabulated in that repository's README. The files
are not redistributed here; this script fetches them into data/sdvrp_set1/.

Instance format: first line "n Q", then n demands, then n+1 coordinate pairs
starting with the depot. Distances are Euclidean rounded to the nearest
integer, which is what the challenge solvers use.
"""
import csv
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sdvrp_set1"
RAW = "https://raw.githubusercontent.com/HUST-Smart/Alkaid-SDVRP/main"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for i in range(1, 22):
        name = f"SD{i}.txt"
        dest = OUT / name
        if dest.exists():
            continue
        with urllib.request.urlopen(f"{RAW}/data/SET-1/{name}") as r:
            dest.write_bytes(r.read())
        print("fetched", name)

    with urllib.request.urlopen(f"{RAW}/README.md") as r:
        readme = r.read().decode("utf-8", "replace")
    rows = []
    for line in readme.splitlines():
        m = re.match(r"\|\s*(SD\d+)\s*\|(.+)\|\s*$", line.strip())
        if not m:
            continue
        vals = []
        for c in [c.strip() for c in m.group(2).split("|")][0::3]:
            try:
                vals.append(int(c))
            except ValueError:
                pass
        if vals:
            rows.append({"instance": m.group(1), "best_known": min(vals),
                         "n_solvers": len(vals)})
    rows.sort(key=lambda r: int(r["instance"][2:]))
    with open(OUT / "best_known.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["instance", "best_known", "n_solvers"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} reference values to {OUT/'best_known.csv'}")


if __name__ == "__main__":
    main()
